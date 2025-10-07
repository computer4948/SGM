import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from tqdm import tqdm
from torchinfo import summary


DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS      = 1000
BATCH_SIZE     = 64
IMAGE_SIZE     = 32
IN_CHANNELS    = 1
NUM_WORKERS    = 2
LR             = 1e-4
N_EPOCHS       = 20
SEED           = 1111
PRINT_EVERY    = 100
SNAPSHOT_STEPS = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 1]

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# =========================
# Données
# =========================
def build_dataloaders(batch_size: int = BATCH_SIZE, image_size: int = IMAGE_SIZE):
    """Crée les DataLoaders train/test pour MNIST + normalisation dans [-1,1]."""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # [0,1] -> [-1,1]
    ])
    train_ds = datasets.MNIST(root="./data_src", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data_src", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True, num_workers=NUM_WORKERS)
    return train_ds, test_ds, train_dl, test_dl


def show_random_samples(dataset, k: int = 10, title: str = "Échantillons aléatoires"):
    """Affiche k échantillons aléatoires d’un dataset MNIST (images brutes)."""
    fig, axes = plt.subplots(1, k, figsize=(1.2 * k, 1.2))
    fig.suptitle(title, fontsize=14, y=1.15)
    for ax in axes:
        idx = np.random.randint(len(dataset))
        ax.imshow(dataset.data[idx], cmap="gray")
        ax.set_title(int(dataset.targets[idx]), fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# =========================
# Planificateur (scheduler) DDPM
# =========================
class DDPMScheduler:
    """
    Planificateur de variance DDPM avec β linéaire.
    - Pré-calcul des α_t, ᾱ_t et racines utiles
    - Ajout de bruit (diffusion) pour l'entraînement
    - Boucle d’échantillonnage (reverse diffusion)
    """

    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: str = DEVICE):
        """Initialise les coefficients β_t et les produits cumulés ᾱ_t."""
        self.timesteps = timesteps
        self.device = device

        self.betas  = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)  # ᾱ_t
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Ajoute du bruit à x selon q(x_t | x_0) = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
        s1 = self.sqrt_alpha_bar[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return s1 * x + s2 * noise

    def sample(
        self,
        model: nn.Module,
        num_samples: int,
        channels: int,
        img_size: int,
        log_interval: int = 100,
    ):
        """
        Échantillonne x_0 depuis x_T ~ N(0,I) via la reverse diffusion.
        Retourne (x_0, snapshots) où snapshots contient des états intermédiaires.
        """
        model.eval()
        with torch.inference_mode():
            x = torch.randn((num_samples, channels, img_size, img_size), device=self.device)
            snapshots = []

            for i in tqdm(reversed(range(self.timesteps)), desc="DDPM sampling"):
                
                t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)

                
                pred_noise = model(x, t)

                
                alphas    = self.alphas[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                betas     = self.betas[t][:, None, None, None]

                
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

                
                x = (1.0 / torch.sqrt(alphas)) * (
                        x - ((1 - alphas) / torch.sqrt(1 - alpha_bar)) * pred_noise
                    ) + torch.sqrt(betas) * noise

                if ((i + 1) % log_interval == 0) or (i == 0):
                    snapshots.append(x.clone())

            return x, snapshots


# =========================
# Blocs du modèle (U-Net)
# =========================
class ResidualBlock(nn.Module):
    """Bloc résiduel simple : Conv → GN → SiLU → Conv → GN (+ skip optionnel)."""

    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_ch is None:
            mid_ch = out_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_ch),
        )

    def forward(self, x):
        return x + self.net(x) if self.residual else self.net(x)


class SelfAttention2D(nn.Module):
    """Bloc d’attention multi-têtes sur cartes 2D (GroupNorm + MHA)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha  = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.view(b, c, h * w)
        y = self.norm(y).transpose(1, 2)          # (b, hw, c)
        y, _ = self.mha(y, y, y)
        y = y.transpose(1, 2).view(b, c, h, w)
        return x + y


class DownsampleBlock(nn.Module):
    """Bloc de descente : MaxPool → ResBlock(res) → ResBlock ; ajoute l’embedding temporel."""

    def __init__(self, in_ch, out_ch, t_emb_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_ch, in_ch, residual=True),
            ResidualBlock(in_ch, out_ch),
        )
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))

    def forward(self, x, t_emb):
        x = self.down(x)
        t = self.t_proj(t_emb)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return x + t


class UpsampleBlock(nn.Module):
    """Bloc de montée : Upsample → concat skip → ResBlock(res) → ResBlock ; + embedding temporel."""

    def __init__(self, in_ch, out_ch, t_emb_dim=256):
        super().__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up  = nn.Sequential(
            ResidualBlock(in_ch, in_ch, residual=True),
            ResidualBlock(in_ch, out_ch, mid_ch=in_ch // 2),
        )
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))

    def forward(self, x, skip, t_emb):
        x = self.up2(x)
        x = torch.cat([skip, x], dim=1)
        x = self.up(x)
        t = self.t_proj(t_emb)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return x + t


class UNetDenoiser(nn.Module):
    """U-Net 2D pour prédire le bruit ε_θ(x_t, t) (1 canal MNIST)."""

    def __init__(self, t_emb_dim=256, device: str = DEVICE):
        super().__init__()
        self.device = device
        self.t_emb_dim = t_emb_dim

        # Encoder
        self.enc_in = ResidualBlock(1, 64)
        self.enc_d1 = DownsampleBlock(64, 128, t_emb_dim)
        self.attn_1 = SelfAttention2D(128)
        self.enc_d2 = DownsampleBlock(128, 256, t_emb_dim)
        self.attn_2 = SelfAttention2D(256)
        self.enc_d3 = DownsampleBlock(256, 256, t_emb_dim)
        self.attn_3 = SelfAttention2D(256)

        # Bottleneck
        self.mid_1 = ResidualBlock(256, 512)
        self.mid_2 = ResidualBlock(512, 512)
        self.mid_3 = ResidualBlock(512, 256)

        # Decoder
        self.dec_u1 = UpsampleBlock(512, 128, t_emb_dim)
        self.attn_4 = SelfAttention2D(128)
        self.dec_u2 = UpsampleBlock(256, 64, t_emb_dim)
        self.attn_5 = SelfAttention2D(64)
        self.dec_u3 = UpsampleBlock(128, 64, t_emb_dim)
        self.attn_6 = SelfAttention2D(64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def _positional_embedding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        """Embeddings sinusoïdaux pour les pas de temps t."""
        half = channels // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=self.device) / half)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(self, x, t):
        """Prédit ε_θ(x_t, t)."""
        t = t.unsqueeze(1).float()
        t_emb = self._positional_embedding(t, self.t_emb_dim)

        x1 = self.enc_in(x)
        x2 = self.attn_1(self.enc_d1(x1, t_emb))
        x3 = self.attn_2(self.enc_d2(x2, t_emb))
        x4 = self.attn_3(self.enc_d3(x3, t_emb))

        x4 = self.mid_3(self.mid_2(self.mid_1(x4)))

        x = self.attn_4(self.dec_u1(x4, x3, t_emb))
        x = self.attn_5(self.dec_u2(x,  x2, t_emb))
        x = self.attn_6(self.dec_u3(x,  x1, t_emb))
        return self.out(x)


# =========================
# Entraînement / Évaluation
# =========================
def train_one_epoch(model, scheduler, dataloader, optimizer, criterion, epoch_idx: int):
    """Boucle d'entraînement pour une époque (DDPM : prédiction du bruit)."""
    model.train()
    running = []

    for x, _ in tqdm(dataloader, desc=f"Train {epoch_idx:02d}"):
        x = x.to(DEVICE)
        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
        noise = torch.randn_like(x)
        x_noisy = scheduler.add_noise(x, noise, t)      
        pred = model(x_noisy, t)                        
        loss = criterion(pred, noise)                   

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running.append(loss.item())

    return float(np.mean(running))


@torch.inference_mode()
def evaluate(model, scheduler, dataloader, criterion, epoch_idx: int):
    """Boucle d’évaluation (MSE bruit)."""
    model.eval()
    running = []

    for x, _ in tqdm(dataloader, desc=f"Eval  {epoch_idx:02d}"):
        x = x.to(DEVICE)
        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
        noise = torch.randn_like(x)
        x_noisy = scheduler.add_noise(x, noise, t)
        pred = model(x_noisy, t)
        loss = criterion(pred, noise)
        running.append(loss.item())

    return float(np.mean(running))


def plot_loss_curves(train_losses, val_losses):
    """Trace la courbe de pertes d’entraînement/validation."""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 3.5))
    plt.plot(epochs, train_losses, "-o", label="Entraînement")
    plt.plot(epochs, val_losses,   "-o", label="Validation")
    plt.xlabel("Époque"); plt.ylabel("MSE"); plt.title("Courbes de perte")
    plt.xticks(epochs); plt.legend(); plt.tight_layout(); plt.show()


# =========================
# Visualisation échantillonnage
# =========================
def show_sampling_snapshots(snapshots, steps=SNAPSHOT_STEPS):
    """Affiche la trajectoire d’une image durant l’échantillonnage (snapshots)."""
    if not snapshots:
        return
    k = min(len(snapshots), len(steps))
    plt.figure(figsize=(2 * k, 2))
    for i in range(k):
        img = snapshots[i][0].detach().cpu().clamp(-1, 1)
        img = (img + 1) / 2
        ax = plt.subplot(1, k, i + 1)
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"t={steps[i]}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_sample_grid(batch, nrow=16, title="Échantillons aléatoires depuis le bruit"):
    """Affiche une grille d’images normalisées en [0,1]."""
    grid = make_grid(batch.detach().cpu(), nrow=nrow, normalize=True)
    plt.figure(figsize=(12, 2.5))
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.show()


def main():
    """Point d’entrée : data → modèle → entraînement → échantillonnage → visualisations."""
    
    train_ds, test_ds, train_dl, test_dl = build_dataloaders(BATCH_SIZE, IMAGE_SIZE)
    show_random_samples(train_ds, 10, "Échantillons aléatoires (train)")
    show_random_samples(test_ds,  10, "Échantillons aléatoires (test)")

    
    ddpm = DDPMScheduler(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=2e-2, device=DEVICE)
    model = UNetDenoiser(t_emb_dim=256).to(DEVICE)
    print("\nRésumé du modèle :")
    summary(model, input_size=(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtypes=[torch.float32,], verbose=0)

    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    
    train_losses, val_losses = [], []
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss = train_one_epoch(model, ddpm, train_dl, optimizer, criterion, epoch)
        va_loss = evaluate(model, ddpm, test_dl, criterion, epoch)
        train_losses.append(tr_loss); val_losses.append(va_loss)
        print(f"Époque {epoch:02d}/{N_EPOCHS} | Train: {tr_loss:.4f} | Val: {va_loss:.4f}")

    plot_loss_curves(train_losses, val_losses)

    
    _, snapshots = ddpm.sample(model=model, num_samples=1, channels=IN_CHANNELS, img_size=IMAGE_SIZE)
    show_sampling_snapshots(snapshots, steps=SNAPSHOT_STEPS)

    
    sampled_imgs, _ = ddpm.sample(model=model, num_samples=64, channels=IN_CHANNELS, img_size=IMAGE_SIZE)
    show_sample_grid(sampled_imgs, nrow=16, title="Sampling aléatoire depuis le bruit")


if __name__ == "__main__":
    main()
