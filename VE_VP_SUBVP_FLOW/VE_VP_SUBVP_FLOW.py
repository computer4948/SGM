import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from torchdiff.sde import HyperParamsSDE, ReverseSDE, SampleSDE
from torchdiff.utils import Metrics


# =========================================
# 1) Config + utilitaires
# =========================================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Données (CelebA)
    celeba_root: str = "./celeba"
    img_size: int = 256                
    batch_size_train: int = 32
    batch_size_val: int = 8
    num_workers: int = min(8, os.cpu_count() or 2)

    # Entraînement
    epochs: int = 1                    
    lr: float = 2e-4

    # SDE / discrétisation
    num_steps: int = 1000
    beta_start: float = 1e-4           # VP / sub-VP
    beta_end: float = 2e-2
    sigma_start: float = 1e-3          # VE
    sigma_end: float = 1.0

    # Génération
    image_shape: Tuple[int, int, int] = (3, 256, 256)  # (C,H,W)
    batch_size_gen: int = 1

    # Sorties
    out_grid_path: str = "grid_methods.png"
    out_metrics_path: str = "metrics_bar.png"


def set_seed(seed: int) -> None:
    """Fixe les graines pour assurer une certaine reproductibilité."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_01(img: torch.Tensor) -> torch.Tensor:
    """
    Convertit une image dans [-1,1] vers [0,1] et permute pour imshow.
    Entrée: [3,H,W] ; Sortie: [H,W,3]
    """
    img01 = (img.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    return img01.permute(1, 2, 0)


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calcule le PSNR entre x et y dans [-1,1], forme [1,3,H,W].
    """
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    mse = torch.mean((x01 - y01) ** 2).item()
    return 10.0 * math.log10(1.0 / (mse + eps))


# =========================================
# 2) Data: CelebA (train/valid)
# =========================================

def build_dataloaders(cfg: Config):
    """
    Construit les DataLoaders CelebA: train (pour apprentissage), valid (pour métriques).
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # vers [-1,1]
    ])

    celeba_root = cfg.celeba_root
    train_dataset = datasets.CelebA(
        root=celeba_root, split="train", transform=transform, download=True
    )
    val_dataset = datasets.CelebA(
        root=celeba_root, split="valid", transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size_train, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size_val, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False
    )
    return train_loader, val_loader


# =========================================
# 3) Réseau Noise Predictor (U-Net simple)
# =========================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodage temporel sinusoidal (type Transformer) pour le pas de diffusion t (scalaire).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] scalaire (on suppose t dans [0,1] ou index -> on normalise)
        Retour: [B,dim]
        """
        t = t.float().view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -(math.log(10000.0) / (half - 1))
        )
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb


class ResBlock(nn.Module):
    """
    Bloc résiduel avec injection du temps via une MLP.
    """
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t_add = self.time_mlp(t_emb).view(t_emb.size(0), -1, 1, 1)
        h = h + t_add

        h = self.conv2(h)
        h = self.norm2(h)
        out = self.act(h) + self.skip(x)
        return out


class UNetNoisePredictor(nn.Module):
    """
    U-Net pour prédire le bruit ε_θ(x_t, t).
    """
    def __init__(self, in_channels=3, base=64, time_dim=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        # Down
        self.rb1 = ResBlock(in_channels, base, time_dim)
        self.down1 = nn.Conv2d(base, base, 4, stride=2, padding=1)  # /2
        self.rb2 = ResBlock(base, base*2, time_dim)
        self.down2 = nn.Conv2d(base*2, base*2, 4, stride=2, padding=1)  # /4

        # Bottleneck
        self.rb3 = ResBlock(base*2, base*2, time_dim)

        # Up
        self.up1 = nn.ConvTranspose2d(base*2, base*2, 4, stride=2, padding=1)  # x2
        self.rb4 = ResBlock(base*4, base, time_dim)
        self.up2 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)  # x2
        self.rb5 = ResBlock(base*2, base, time_dim)

        self.out = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """
        x: [B,3,H,W], t: [B] (réel ou index). Sortie: bruit prédit ε_θ de même forme que x.
        """
        t_emb = self.time_emb(t)

        # Down
        h1 = self.rb1(x, t_emb)
        d1 = self.down1(h1)
        h2 = self.rb2(d1, t_emb)
        d2 = self.down2(h2)

        # Bottleneck
        mid = self.rb3(d2, t_emb)

        # Up (skip connections)
        u1 = self.up1(mid)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.rb4(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.rb5(u2, t_emb)

        return self.out(u2)


# =========================================
# 4) Entraînement rapide (objectif VP : L_simple)
# =========================================

def vp_training_step(model, x0, alphas_bar, optimizer):
    """
    Une itération style DDPM (VP) :
      - on échantillonne un pas k
      - on forme x_k = sqrt(alpha_bar_k) x0 + sqrt(1-alpha_bar_k) eps
      - le modèle prédit eps ; on minimise MSE(eps_pred, eps)
    """
    b = x0.size(0)
    k = torch.randint(0, len(alphas_bar), (b,), device=x0.device)
    a_bar = alphas_bar[k].view(b, 1, 1, 1)

    eps = torch.randn_like(x0)
    xk = torch.sqrt(a_bar) * x0 + torch.sqrt(1. - a_bar) * eps

    eps_pred = model(xk, k.float())   # t = index k
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


def precompute_alphas_bar(num_steps, beta_start, beta_end, device):
    """
    Pré-calcule alpha_bar_k = prod_{i<=k} (1 - beta_i) pour un schedule linéaire.
    """
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas_bar


def train_noise_predictor(cfg: Config, train_loader: DataLoader) -> nn.Module:
    """
    Entraîne le petit U-Net pour prédire ε (objectif VP).
    Retourne le modèle entraîné.
    """
    model = UNetNoisePredictor(in_channels=3, base=64, time_dim=256).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    alphas_bar = precompute_alphas_bar(cfg.num_steps, cfg.beta_start, cfg.beta_end, cfg.device)

    model.train()
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for x, _ in train_loader:
            x = x.to(cfg.device)  
            loss = vp_training_step(model, x, alphas_bar, optimizer)
            global_step += 1
            if global_step % 200 == 0:
                print(f"[train] epoch {epoch+1} step {global_step} | loss={loss:.4f}")
    t1 = time.time()
    print(f"[train] terminé en {t1 - t0:.1f}s (epochs={cfg.epochs})")
    model.eval()
    return model


# =========================================
# 5) ReverseSDE (VE / VP / sub-VP / Flow) + échantillonnage
# =========================================

def make_reverse_sde(cfg: Config, method: str) -> ReverseSDE:
    """
    Construit le processus inverse pour une des méthodes: 've', 'vp', 'subvp', 'flow'.
    """
    hp = HyperParamsSDE(
        num_steps=cfg.num_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        sigma_start=cfg.sigma_start,
        sigma_end=cfg.sigma_end
    )
    try:
        return ReverseSDE(hp, method=method)
    except Exception:
        if method.lower() == "flow":
            
            return ReverseSDE(hp, method="pf")
        raise


@torch.no_grad()
def sample_one(cfg: Config, reverse_proc: ReverseSDE, noise_pred: nn.Module) -> torch.Tensor:
    """
    Échantillonne 1 image avec TorchDiff SampleSDE :
      - reverse_proc : ReverseSDE (VE/VP/subVP/flow)
      - noise_pred   : notre U-Net qui prédit ε
    Retour: [1,3,H,W] dans [-1,1]
    """
    sampler = SampleSDE(
        reverse_diffusion=reverse_proc,
        noise_predictor=noise_pred,
        image_shape=cfg.image_shape,
        conditional_model=None,
        batch_size=cfg.batch_size_gen,
        in_channels=cfg.image_shape[0],
        device=cfg.device
    )
    return sampler()


# =========================================
# 6) Métriques (SSIM TorchDiff + PSNR)
# =========================================

def build_metrics(cfg: Config) -> Metrics:
    """Construit l’objet métriques TorchDiff (ici SSIM)."""
    return Metrics(device=cfg.device, fid=False, ssim=True, lpips=False)


@torch.no_grad()
def evaluate_pair(metrics_obj: Metrics, x_hat: torch.Tensor, x_ref: torch.Tensor) -> Dict[str, float]:
    """
    Calcule SSIM (TorchDiff) + PSNR entre un sample x_hat et une référence x_ref ([-1,1]).
    """
    try:
        m = metrics_obj(x_hat, x_ref)
    except TypeError:
        m = metrics_obj.evaluate(x_hat, x_ref)

    out = {}
    if isinstance(m, dict):
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[k.lower()] = float(v)
    out["psnr"] = psnr(x_hat, x_ref)
    return out



def main():
    cfg = Config()
    set_seed(cfg.seed)

    C = cfg.image_shape[0]
    cfg.image_shape = (C, cfg.img_size, cfg.img_size)

    # a) Data
    train_loader, val_loader = build_dataloaders(cfg)
    val_batch, _ = next(iter(val_loader))
    ref_img = val_batch[:1].to(cfg.device)  

    # b) Entraîne le réseau (objectif VP)
    noise_predictor = train_noise_predictor(cfg, train_loader)

    # c) Construit les 4 processus inverses
    methods = ["ve", "vp", "subvp", "flow"]
    reverses = {m: make_reverse_sde(cfg, m) for m in methods}

    # d) Échantillonne 1 image par méthode
    generated = {}
    for m in methods:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        generated[m] = sample_one(cfg, reverses[m], noise_predictor)  # [1,3,H,W]

    # e) Métriques
    metrics_obj = build_metrics(cfg)
    scores: Dict[str, Dict[str, float]] = {}
    for m in methods:
        scores[m] = evaluate_pair(metrics_obj, generated[m], ref_img)

    # f) Grille (VE/VP/sub-VP/Flow)
    plt.figure(figsize=(10, 10))
    titles = {
        "ve": "VE (variance-exploding)",
        "vp": "VP (variance-preserving)",
        "subvp": "sub-VP",
        "flow": "Flow (prob. flow ODE)"
    }
    for idx, m in enumerate(methods, 1):
        plt.subplot(2, 2, idx)
        plt.imshow(to_01(generated[m][0]))
        plt.axis("off")
        sc = scores[m]
        ssim_txt = f"SSIM: {sc.get('ssim', float('nan')):.4f}" if "ssim" in sc else ""
        psnr_txt = f"PSNR: {sc['psnr']:.2f} dB"
        plt.title(f"{titles[m]}\n{ssim_txt}  {psnr_txt}")
    plt.tight_layout()
    plt.savefig(cfg.out_grid_path, dpi=150)
    plt.show()

    # g) Barplot des métriques
    ssim_vals, psnr_vals = [], []
    for m in methods:
        ssim_vals.append(scores[m].get("ssim", float("nan")))
        psnr_vals.append(scores[m]["psnr"])

    xs = range(len(methods))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(xs, ssim_vals)
    plt.xticks(xs, [m.upper() for m in methods])
    plt.title("SSIM")
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(xs, psnr_vals)
    plt.xticks(xs, [m.upper() for m in methods])
    plt.title("PSNR (dB)")

    plt.tight_layout()
    plt.savefig(cfg.out_metrics_path, dpi=150)
    plt.show()

    print("Grille d’images :", os.path.abspath(cfg.out_grid_path))
    print("Graphique métriques :", os.path.abspath(cfg.out_metrics_path))
    print("Scores par méthode :", scores)


if __name__ == "__main__":
    main()
