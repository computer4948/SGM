import argparse
from pathlib import Path

import torch
from tqdm import trange
from PIL import Image
import torchvision.transforms as T
import pandas as pd

import deepinv
from deepinv.models import DiffUNet


def load_image(path: str, size: int, device: str) -> torch.Tensor:
    """Charge une image RGB -> tensor [-1,1] de shape [3,H,W]."""
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])  # [0,1]
    x = tfm(img).to(device)
    return x * 2 - 1  # [-1,1]


def make_center_hole_mask(h: int, w: int, hole: int, device: str) -> torch.Tensor:
    """Mask binaire [1=observé, 0=manquant] avec un trou carré centré."""
    mask = torch.ones(1, h, w, device=device)
    r = max(1, hole // 2)
    hs, ws = h // 2, w // 2
    mask[:, hs - r : hs + r, ws - r : ws + r] = 0
    return mask


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    """Chaîne 'm ± s' avec arrondi."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def save_metrics_table(means: dict, stds: dict, exp_name: str, out_dir: Path):
    """
    Sauvegarde un tableau de métriques (CSV + LaTeX).
    `means` et `stds` sont des dicts {metric_name: valeur_tensor_ou_float}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise en float
    to_float = lambda x: float(x.detach().item() if isinstance(x, torch.Tensor) else x)
    metrics = sorted(means.keys())
    data = {"DPS_DDIM": [format_mean_std(to_float(means[m]), to_float(stds[m])) for m in metrics]}
    df = pd.DataFrame(data, index=[m.upper() for m in metrics])

    csv_path = out_dir / f"metrics_{exp_name}.csv"
    tex_path = out_dir / f"metrics_{exp_name}.tex"

    df.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(escape=False, header=True))

    print("\n=== METRICS TABLE ===")
    print(df.to_markdown())
    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {tex_path}")


# =========================
# DPS-DDIM
# =========================
@torch.no_grad(False)
def run_dps_ddim(
    y: torch.Tensor,
    physics,
    alphas_cumprod: torch.Tensor,
    model: DiffUNet,
    *,
    zeta: float = 1.0,
    rho: float = 1.0,
    starting_noise: torch.Tensor | None = None,
    num_steps: int = 1000,
    batch_size: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Implémentation DPS avec schéma DDIM.
    Hypothèse : x, y dans [-1,1], shape [3,H,W].
    Retourne la reconstruction x_hat dans [-1,1].
    """
    xt = torch.randn_like(y, device=device) if starting_noise is None else starting_noise.clone()

    for t in trange(num_steps - 1, -1, -1, desc="[DPS-DDIM]"):
        xt.requires_grad_()
        times = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Prédiction de bruit et estimateur x0
        noise_pred = model.forward_diffusion(xt, times)[:, :3]
        x0_hat = (xt - torch.sqrt(1 - alphas_cumprod[t]) * noise_pred) / torch.sqrt(alphas_cumprod[t])

        # Guidage DPS : gradient de la fidélité aux mesures
        loss = ((physics(x0_hat) - y) ** 2).sum()
        grad = torch.autograd.grad(loss, xt)[0]

        if t == 0:
            return x0_hat.detach()

        with torch.no_grad():
            # Mise à jour DDIM
            sigma_t = torch.sqrt(
                (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) *
                (1 - alphas_cumprod[t] / alphas_cumprod[t - 1])
            )
            sigma_t = rho * sigma_t

            coef_xt = torch.sqrt(alphas_cumprod[t - 1] / alphas_cumprod[t])
            coef_noise = (
                torch.sqrt(1 - alphas_cumprod[t - 1] - sigma_t ** 2)
                - torch.sqrt(alphas_cumprod[t - 1] * (1 - alphas_cumprod[t]) / alphas_cumprod[t])
            )

            xt_prime = coef_xt * xt + coef_noise * noise_pred + sigma_t * torch.randn_like(xt, device=device)
            xt = (xt_prime - (zeta / torch.sqrt(loss)) * grad).detach()


# =========================
# Évaluation & sauvegarde métriques
# =========================
def compute_metrics(x_ref: torch.Tensor, x_hat_list: list[torch.Tensor], device: str) -> tuple[dict, dict]:
    """
    Calcule PSNR / SSIM / LPIPS pour une ou plusieurs reconstructions.
    Retourne (means, stds) où chaque entrée est {metric_name: tensor(float)}.
    """
    metrics = {
        "psnr": deepinv.metric.PSNR(min_pixel=-1, max_pixel=1),
        "ssim": deepinv.metric.SSIM(min_pixel=-1, max_pixel=1),
        "lpips": deepinv.metric.LPIPS(device=device),
    }
    # Stockage des valeurs par métrique
    values = {k: [] for k in metrics.keys()}

    x_ref_b = x_ref.unsqueeze(0)  # (1,3,H,W) attendu par les métriques
    for x_hat in x_hat_list:
        x_hat_b = x_hat.unsqueeze(0)
        for name, fn in metrics.items():
            v = fn(x_ref_b, x_hat_b)
            values[name].append(v.detach().cpu())

    means = {name: torch.stack(vals).mean() for name, vals in values.items()}
    stds  = {name: torch.stack(vals).std(unbiased=False) for name, vals in values.items()}
    return means, stds


# =========================
# Main (test sur image fournie)
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="DPS-DDIM Inpainting (DeepInv) avec sauvegarde de métriques – à partir d'une image fournie."
    )
    parser.add_argument("--image", type=str, required=True, help="Chemin de l'image d'entrée (RGB).")
    parser.add_argument("--img_size", type=int, default=256, help="Taille de redimensionnement carré.")
    parser.add_argument("--hole", type=int, default=120, help="Taille du trou carré (pixels) pour l'inpainting.")
    parser.add_argument("--noise_level", type=float, default=0.2, help="Amplitude du bruit additif sur la mesure.")
    parser.add_argument("--steps", type=int, default=1000, help="Nombre d'itérations de diffusion.")
    parser.add_argument("--zeta", type=float, default=1.0, help="Poids du guidage DPS.")
    parser.add_argument("--rho", type=float, default=1.0, help="Échelle du bruit DDIM.")
    parser.add_argument("--name", type=str, default="run", help="Nom d'expérience (suffixe des fichiers).")
    parser.add_argument("--out_dir", type=str, default="./results", help="Dossier de sortie pour les métriques.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)

    # 1) Chargement image & physique d'inpainting
    x_true = load_image(args.image, size=args.img_size, device=device)       # [-1,1], [3,H,W]
    C, H, W = x_true.shape
    mask = make_center_hole_mask(H, W, args.hole, device=device)
    physics = deepinv.physics.Inpainting(img_size=(3, H, W), mask=mask, device=device)

    # 2) Modèle & planning alpha-bar(t)
    denoiser = DiffUNet(large_model=False).to(device).eval()
   
    alphas_cumprod = (denoiser.get_alpha_prod()[-1] ** 2).to(device)

    # 3) Génération de la mesure y (dans [-1,1])
    y = physics(x_true) + args.noise_level * torch.randn_like(x_true)
    y = y * 2 - 1

    # 4) Reconstruction via DPS-DDIM
    x_hat = run_dps_ddim(
        y=y,
        physics=physics,
        alphas_cumprod=alphas_cumprod,
        model=denoiser,
        zeta=args.zeta,
        rho=args.rho,
        num_steps=args.steps,
        batch_size=1,
        device=device,
    )

    # 5) Évaluation & sauvegarde métriques (sur une seule reco ; extensible à plusieurs)
    means, stds = compute_metrics(x_true, [x_hat], device=device)
    save_metrics_table(means, stds, exp_name=args.name, out_dir=out_dir)

    print("[DONE] Reconstruction terminée et métriques sauvegardées.")


if __name__ == "__main__":
    main()
