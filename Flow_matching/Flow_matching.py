import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==========================
# 1) Utilitaires / Seeds
# ==========================

def set_seed(seed: int = 42):
    """Fixe la graine aléatoire pour rendre la démo reproductible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# 2) Données : Deux lunes (Two Moons)
# ==========================================

def sample_two_moons(n: int, noise: float = 0.05) -> np.ndarray:
    """
    Génère n points (n/2 par lune) formant deux croissants de lune imbriqués.
    - Lune 1 : arc supérieur d’un cercle (angles 0..pi)
    - Lune 2 : arc inférieur décalé
    Un bruit gaussien est ajouté.
    """
    n1 = n // 2
    n2 = n - n1

    theta1 = np.random.rand(n1) * np.pi
    theta2 = np.random.rand(n2) * np.pi

    # lune 1 (arc supérieur)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    # lune 2 (arc inférieur décalé)
    x2 = np.stack([1.0 - np.cos(theta2), -np.sin(theta2) - 0.5], axis=1)

    X = np.concatenate([x1, x2], axis=0)
    X += np.random.normal(scale=noise, size=X.shape)
    return X.astype(np.float32)


def sample_gaussian(n: int) -> np.ndarray:
    """Échantillonne n points i.i.d depuis une gaussienne standard N(0, I) en 2D."""
    return np.random.randn(n, 2).astype(np.float32)


# ========================================================
# 3) Réseau : Champ de vitesse u_θ(x, t)
# ========================================================

class TimeFourier(nn.Module):
    """
    Encodage temporel sin/cos (positional encoding) pour stabiliser l’apprentissage
    du champ de vitesse en fonction du temps t ∈ [0,1].
    """
    def __init__(self, n_freqs: int = 6):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] dans [0,1]  →  concat[sin(2π f_i t), cos(2π f_i t)]  (dimension 2*n_freqs)
        """
        t = t.view(-1, 1)                       
        angles = 2.0 * math.pi * t * self.freqs 
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class VelocityNet(nn.Module):
    """
    Petit MLP qui approxime le champ de vitesse u_θ(x,t) en 2D.
    Entrée : (x ∈ R^2, encodage(t) ∈ R^{2*n_freqs})
    Sortie : u ∈ R^2
    """
    def __init__(self, hidden: int = 64, n_freqs_time: int = 6):
        super().__init__()
        self.t_embed = TimeFourier(n_freqs=n_freqs_time)
        in_dim = 2 + 2 * n_freqs_time
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tfeat = self.t_embed(t)                 
        h = torch.cat([x, tfeat], dim=1)        
        return self.net(h)                      # u_θ(x,t)


# ========================================================
# 4) Entraînement Flow Matching
# ========================================================

def fm_train(model: nn.Module,
             steps: int = 1200,
             batch_size: int = 1024,
             lr: float = 2e-3,
             moon_noise: float = 0.05,
             log_every: int = 200,
             device: torch.device = torch.device("cpu")) -> list:
    """
    Entraîne u_θ via l’objectif Flow Matching avec chemin linéaire :
        x_t = (1 - t) * x0 + t * x1,    cible u_cond = x1 - x0
    où x0 ~ data (two moons), x1 ~ N(0, I), t ~ U[0,1].
    Retourne la liste des pertes pour tracer la courbe d’entraînement.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    losses = []
    for it in range(1, steps + 1):
        # échantillonnage du batch
        x0 = torch.from_numpy(sample_two_moons(batch_size, noise=moon_noise)).to(device)  
        x1 = torch.from_numpy(sample_gaussian(batch_size)).to(device)                     
        t = torch.rand(batch_size, device=device)                                        # U[0,1]

        # chemin + cible
        xt = (1.0 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
        u_target = x1 - x0  

        # prédiction u_theta(xt, t)
        u_pred = model(xt, t)

        # MSE
        loss = F.mse_loss(u_pred, u_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if it % log_every == 0:
            print(f"[FM] step {it:4d}/{steps} | loss={loss.item():.6f}")

    return losses


# ========================================================
# 5) Intégration ODE (t=1 -> t=0) — Heun (RK2)
# ========================================================

@torch.no_grad()
def integrate_flow(model: nn.Module,
                   n_samples: int = 3000,
                   n_steps: int = 200,
                   save_times=(1.0, 0.75, 0.5, 0.25, 0.0),
                   device: torch.device = torch.device("cpu")) -> dict:
    """
    Intègre l’ODE dx/dt = u_θ(x,t) depuis t=1 jusqu’à t=0 avec Heun (RK2).
    - n_samples : nombre de particules à transporter (échantillons initiaux ~ N(0,I))
    - n_steps   : nombre de pas d’intégration (fixe)
    - save_times: temps auxquels on sauvegarde l’état (pour visualiser les étapes)
    Retour: dict {time: np.array [n,2]} (copies CPU pour plotting)
    """
    model.eval()
    # état initial : échantillons de la Gaussienne (base) au temps 1
    x = torch.from_numpy(sample_gaussian(n_samples)).to(device)
    dt = -1.0 / n_steps 
    saved = {1.0: x.detach().cpu().numpy().copy()}

    t = torch.ones(n_samples, device=device)
    for _ in range(n_steps):
        # Heun (RK2)
        k1 = model(x, t)               # u(x,t)
        x_euler = x + dt * k1
        t_next = t + dt
        k2 = model(x_euler, t_next)    # u(x + dt*k1, t+dt)
        x = x + dt * 0.5 * (k1 + k2)
        t = t_next.clamp(min=0.0)      

        for ts in save_times:
            if ts not in saved and float(t.mean().item()) <= ts + 1e-6:
                saved[ts] = x.detach().cpu().numpy().copy()

 
    if 0.0 not in saved:
        saved[0.0] = x.detach().cpu().numpy().copy()

    return saved


# ========================================================
# 6) Visualisation
# ========================================================

def plot_points(points: np.ndarray, title: str):
    """Affiche un nuage de points 2D (scatter) avec un titre."""
    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.8)
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()


def plot_training_curve(losses: list):
    """Trace la courbe de perte d’entraînement (MSE) au fil des itérations."""
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Itération")
    plt.ylabel("Perte MSE (u_theta vs cible)")
    plt.title("Flow Matching — Courbe d’entraînement")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Flow Matching 2D : Gaussienne -> Deux lunes")
    parser.add_argument("--steps", type=int, default=1200, help="Itérations d'entraînement")
    parser.add_argument("--batch", type=int, default=1024, help="Taille de batch d'entraînement")
    parser.add_argument("--lr", type=float, default=2e-3, help="Taux d'apprentissage")
    parser.add_argument("--noise", type=float, default=0.05, help="Bruit sur les deux lunes")
    parser.add_argument("--n_samples", type=int, default=3000, help="Échantillons pour l'intégration")
    parser.add_argument("--n_steps", type=int, default=200, help="Pas d'intégration ODE (Heun)")
    parser.add_argument("--cpu", action="store_true", help="Forcer CPU (par défaut : CUDA si dispo)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    print("Device :", device)

    model = VelocityNet(hidden=64, n_freqs_time=6).to(device)

    # Entraînement Flow Matching
    losses = fm_train(model,
                      steps=args.steps,
                      batch_size=args.batch,
                      lr=args.lr,
                      moon_noise=args.noise,
                      log_every=200,
                      device=device)


    plot_training_curve(losses)


    snapshots = integrate_flow(model,
                               n_samples=args.n_samples,
                               n_steps=args.n_steps,
                               save_times=(1.0, 0.75, 0.5, 0.25, 0.0),
                               device=device)

 
    for ts in [1.0, 0.75, 0.5, 0.25, 0.0]:
        titre = f"Échantillons au temps t = {ts:.2f}"
        if ts == 1.0:
            titre += " (distribution de départ ~ N(0,I))"
        if ts == 0.0:
            titre += " (distribution cible ~ Deux lunes)"
        plot_points(snapshots[ts], titre)


if __name__ == "__main__":
    main()
