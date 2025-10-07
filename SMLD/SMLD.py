import os
from dataclasses import dataclass
from typing import Tuple

import torch
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, ScoreSdeVeScheduler



@dataclass
class Config:
    """Regroupe tous les hyperparamètres et options d’exécution."""
    model_id: str = "google/ncsnpp-celebahq-256"   # dépôt HF du modèle NCSN++
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Taille d’échantillonnage et affichage
    num_samples: int = 6           # nombre d’images à générer
    out_path: str = "smld_grid.png"

    # Discrétisation VE (nombre de niveaux de σ)
    num_sigmas: int = 1000

    # Corrector: nombre de pas de Langevin 
    steps_per_sigma: int = 3

    # Échelle de pas de Langevin
    eta_ld: float = 0.12

    # Utiliser le pas prédicteur reverse VE-SDE entre niveaux
    use_predictor: bool = True

    # Clamp doux pour stabilité
    clamp_after_each_step: bool = True

    # Grille d’affichage
    grid_rows: int = 2
    grid_cols: int = 3


# =========================
# Utilitaires généraux
# =========================

def set_seed(cfg: Config) -> None:
    """Fixe les graines pour reproductibilité."""
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)


def to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    """Déplace un tenseur vers le device demandé (cpu/cuda)."""
    return x.to(device, non_blocking=True)


# =========================
# Chargement modèle / scheduler
# =========================

def load_model_and_scheduler(cfg: Config) -> Tuple[UNet2DModel, ScoreSdeVeScheduler]:
    """
    Charge le réseau de score (NCSN++) et le scheduler VE-SDE.
    Retourne (unet, scheduler) en mode eval, sur le bon device.
    """
    # Modèle (fp32)
    unet = UNet2DModel.from_pretrained(cfg.model_id, use_safetensors=False)
    unet = unet.to(cfg.device).eval()

    # Scheduler VE-SDE lié au même dépôt (contient σ_max / σ_min)
    sched = ScoreSdeVeScheduler.from_pretrained(cfg.model_id)


    if hasattr(sched, "set_sigmas"):
        sched.set_sigmas(cfg.num_sigmas)
    else:
        sched.set_timesteps(cfg.num_sigmas)

    return unet, sched


# =========================
# Sampler SMLD (VE-SDE)
# =========================

@torch.no_grad()
def smld_generate(cfg: Config,
                  unet: UNet2DModel,
                  sched: ScoreSdeVeScheduler) -> torch.Tensor:
    """
    Génère des échantillons via SMLD (annealed Langevin dynamics) avec schéma
    Predictor–Corrector (M pas de Langevin par σ, puis un pas prédicteur reverse VE).
    Renvoie un tenseur d’images mises à l’échelle dans [0,1]: [B, C, H, W].
    """

    # Dimensions à partir de la config du modèle
    C = unet.config.in_channels
    H = W = unet.config.sample_size

    # Suite {σ_i} du scheduler: [σ_1 ... σ_L] avec σ_1 = σ_max, σ_L = σ_min
    sigmas = to_device(sched.sigmas, cfg.device).float()
    sigma_max = float(sigmas[0].item())
    sigma_min = float(sigmas[-1].item())

    # Initialisation en loi N(0, σ_max^2 I)
    x = torch.randn((cfg.num_samples, C, H, W), device=cfg.device) * sigma_max

    for i, sigma_i in enumerate(sigmas):
        # scalaire (float) et batch de "timesteps"
        sigma_i = float(sigma_i.item())
        t = torch.full((cfg.num_samples,), sigma_i, device=cfg.device, dtype=torch.float32)

        # Pas de Langevin au niveau i
        eps_i = cfg.eta_ld * (sigma_i * sigma_i) / (sigma_min * sigma_min)


        for _ in range(cfg.steps_per_sigma):
            x_in = sched.scale_model_input(x, t) if hasattr(sched, "scale_model_input") else x

            # s_\theta(x, σ_i) – réseau de score
            score = unet(x_in, t).sample

            # Bruit standard
            noise = torch.randn_like(x)

            # Pas de Langevin (SMLD): x <- x + ε s + √(2ε) ξ
            x = x + eps_i * score + torch.sqrt(torch.tensor(2.0 * eps_i, device=x.device)) * noise

            if cfg.clamp_after_each_step:
                x = x.clamp(-1, 1)

        # -------- Predictor: 1 pas reverse du VE-SDE (Euler–Maruyama) --------
        if cfg.use_predictor:
            x_in = sched.scale_model_input(x, t) if hasattr(sched, "scale_model_input") else x
            score = unet(x_in, t).sample

            if hasattr(sched, "step_pred"):
                try:
                    x = sched.step_pred(model_output=score, timestep=t, sample=x).prev_sample
                except TypeError:
                    x = sched.step_pred(model_output=score, sample=x, timestep=t).prev_sample
            else:
                drift_eps = 0.25 * eps_i
                x = x + drift_eps * score

            if cfg.clamp_after_each_step:
                x = x.clamp(-1, 1)

    # -------- Dénoyautage final (Tweedie) --------
    # x0 ≈ x + σ_min^2 * s(x, σ_min)
    t_last = torch.full((cfg.num_samples,), float(sigma_min), device=cfg.device, dtype=torch.float32)
    x_in = sched.scale_model_input(x, t_last) if hasattr(sched, "scale_model_input") else x
    score_last = unet(x_in, t_last).sample
    x = x + (sigma_min * sigma_min) * score_last

    # Mise à l’échelle pour affichage/sauvegarde: [-1,1] -> [0,1]
    imgs = (x.float().clamp(-1, 1) + 1) / 2.0
    return imgs


# =========================
# Visualisation
# =========================

def plot_grid(imgs: torch.Tensor, rows: int, cols: int, save_path: str = None) -> None:
    """
    Affiche une grille (rows x cols) d’images [B,C,H,W] normalisées dans [0,1],
    et sauvegarde si save_path est fourni.
    """
    B = imgs.size(0)
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(min(B, rows * cols)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(imgs[i].permute(1, 2, 0).detach().cpu().numpy())
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def main() -> None:
    """Point d’entrée: charge le modèle, génère des échantillons, affiche et sauvegarde une grille."""
    cfg = Config()

 
    total_cells = cfg.grid_rows * cfg.grid_cols
    if cfg.num_samples != total_cells:
        cfg.num_samples = total_cells  

    set_seed(cfg)
    unet, sched = load_model_and_scheduler(cfg)
    imgs = smld_generate(cfg, unet, sched)
    plot_grid(imgs, cfg.grid_rows, cfg.grid_cols, save_path=cfg.out_path)

    print(f"Génération terminée. Grille sauvegardée dans: {os.path.abspath(cfg.out_path)}")


if __name__ == "__main__":
    main()
