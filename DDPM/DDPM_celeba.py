from __future__ import annotations
import math
import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000                    # nombre total de pas de diffusion
NUM_SAMPLES = 10            # nombre d'images à générer
SNAPSHOT_STEPS = [900, 700, 500, 300, 100, 0]  # pour visualiser la trajectoire
PRINT_EVERY = 50            # fréquence des logs
SEED = 40
torch.manual_seed(SEED)


# =========================
# Modèle
# =========================
def load_pretrained_unet(model_id: str = "google/ddpm-celebahq-256",
                         use_safetensors: bool = False,
                         device: str = DEVICE) -> tuple[UNet2DModel, int, int, int]:
    """
    Charge un UNet2DModel pré-entraîné depuis diffusers et renvoie (model, C, H, W).
    Déduit automatiquement la taille (32x32 / 256x256, etc.) depuis la config.
    """
    model = UNet2DModel.from_pretrained(model_id, use_safetensors=use_safetensors).to(device)
    model.eval()
    C = model.config.in_channels
    H = W = model.config.sample_size
    return model, C, H, W


# =========================
# Scheduler DDPM
# =========================
class DDPM:
    """
    Scheduler DDPM minimal (linéaire en beta) pour la phase de sampling.
    Gère les quantités : betas, alphas, alphas_cumprod, alphas_cumprod_prev.
    Fournit une étape 'step' qui calcule x_{t-1} à partir de x_t et de la prédiction de bruit.
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: str = "cpu"):
        """
        Initialise les coefficients du scheduler (planning linéaire sur beta).
        """
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

    def step(self, model_output: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        """
        Une étape de sampling DDPM :
        - estime x0 = (x_t - sqrt(1 - ᾱ_t) * ε_θ) / sqrt(ᾱ_t)
        - calcule la moyenne du postérieur q(x_{t-1} | x_t, x0)
        - échantillonne x_{t-1}
        """
        beta_t = self.betas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]

        # x0 prédit (borné dans [-1,1] pour stabilité d'affichage)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-1, 1)

        # moyenne du postérieur (forme "ε-pred" équivalente)
        coeff_x0 = torch.sqrt(alpha_bar_prev)
        coeff_eps = torch.sqrt(torch.clamp(1 - alpha_bar_prev - beta_t, min=1e-8))
        mean = coeff_x0 * x0_pred + coeff_eps * model_output

        # ajoute le bruit de transition si t>0
        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev


# =========================
# Sampling
# =========================
def sample_ddpm(model: UNet2DModel,
                scheduler: DDPM,
                *,
                T: int,
                num_samples: int,
                C: int, H: int, W: int,
                device: str = DEVICE,
                log_every: int = PRINT_EVERY,
                snapshot_steps: list[int] | None = None) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Génère des échantillons par reverse diffusion DDPM.
    Retourne (x_T=images finales, snapshots={t: batch_à_t} pour certains pas).
    """
    x = torch.randn((num_samples, C, H, W), device=device)  # bruit initial
    snapshots: dict[int, torch.Tensor] = {}
    snapshot_set = set(snapshot_steps or [])

    with torch.no_grad():
        for t in reversed(range(T)):
            if (t % log_every) == 0:
                print(f"[DDPM] step {t:4d} / {T}")

            # t pour tout le batch
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # prédiction de bruit ε_θ(x_t, t)
            eps = model(x, t_tensor).sample

            # étape DDPM vers x_{t-1}
            x = scheduler.step(eps, t, x)  # <- commentaires légers : voir docstring de DDPM.step

            # snapshots optionnels
            if t in snapshot_set:
                snapshots[t] = x.clone()

    return x, snapshots


# =========================
# Visualisation
# =========================
def show_snapshots(snapshots: dict[int, torch.Tensor]):
    """
    Affiche l'évolution de la 1ère image du batch pour quelques pas t (graphiques matplotlib).
    """
    if not snapshots:
        return

    ts = sorted(snapshots.keys(), reverse=True)
    plt.figure(figsize=(3 * len(ts), 3))
    for i, t in enumerate(ts):
        plt.subplot(1, len(ts), i + 1)
        img = snapshots[t][0].detach().cpu().clamp(-1, 1)
        img = (img + 1) / 2  # [0,1]
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"t={t}", fontsize=10)
    plt.tight_layout()
    plt.show()


def show_grid(batch: torch.Tensor, rows: int = 2, cols: int = 5):
    """
    Affiche une grille (rows x cols) d'images depuis un batch en [0,1].
    Par défaut : 2x5 pour 10 images (comme dans le script original).
    """
    imgs = (batch.detach().cpu().float().clamp(-1, 1) + 1) / 2  # [0,1]
    n = min(batch.size(0), rows * cols)

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def main():
    """
    Point d'entrée :
    - charge le modèle pré-entraîné
    - instancie le scheduler DDPM
    - lance l’échantillonnage
    - affiche snapshots et grille finale
    """
    model, C, H, W = load_pretrained_unet(
        model_id="google/ddpm-celebahq-256",
        use_safetensors=False,
        device=DEVICE,
    )

    ddpm = DDPM(num_train_timesteps=T, device=DEVICE)

    x_final, snapshots = sample_ddpm(
        model, ddpm,
        T=T,
        num_samples=NUM_SAMPLES,
        C=C, H=H, W=W,
        device=DEVICE,
        log_every=PRINT_EVERY,
        snapshot_steps=SNAPSHOT_STEPS,
    )

    show_snapshots(snapshots)                  
    show_grid(x_final, rows=2, cols=5)       


if __name__ == "__main__":
    main()
