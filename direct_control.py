"""Direct mean-field control for adult-to-child transport in face latent space."""

import os
import math
import random
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# ============================================================
# Default MFC training config (no data paths; used with project samplers)
# ============================================================
DEFAULT_MFC_TRAIN_CFG = {
    "epochs": 1000,
    "lr": 2e-4,
    "grad_clip": 1.0,
    "T": 1.0,
    "num_steps": 50,
    "eps": 0.05,
    "lambda_running": 0.05,
    "lambda_terminal": 5.0,
    "lambda_moments": 10.0,
    "lambda_sw2": 5.0,
    "sw2_proj": 1024,
    "moment_eps": 1e-6,
    "log_every": 25,
    "save_every": 50,
}


# ============================================================
# Utilities (used by both core API and standalone)
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sliced_w2(x: torch.Tensor, y: torch.Tensor, n_proj: int) -> torch.Tensor:
    """Sliced Wasserstein-2^2 with random 1D projections."""
    B, D = x.shape
    dirs = torch.randn(n_proj, D, device=x.device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    x_p = x @ dirs.T
    y_p = y @ dirs.T
    x_s, _ = torch.sort(x_p, dim=0)
    y_s, _ = torch.sort(y_p, dim=0)
    return ((x_s - y_s) ** 2).mean()


def denorm01(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] -> [0,1]. Used by standalone visualization."""
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_grid(imgs01: torch.Tensor, out_path: str, nrow: int = 4):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(imgs01, out_path, nrow=nrow)


# ============================================================
# Drift policy and solver in latent space
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding; t is [B]
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device)
        )
        args = t[:, None] / freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DriftPolicy(nn.Module):
    """alpha(t, z) for z in R^{z_dim}."""

    def __init__(self, z_dim: int, hidden: int = 1024, time_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: [B,z_dim], t: [B]
        te = self.time_emb(t)
        return self.net(torch.cat([z, te], dim=1))


class MFCSolver(nn.Module):
    def __init__(self, policy: nn.Module, train_cfg: dict):
        super().__init__()
        self.policy = policy
        self.T = float(train_cfg["T"])
        self.num_steps = int(train_cfg["num_steps"])
        self.eps = float(train_cfg["eps"])

        self.lambda_running = float(train_cfg["lambda_running"])
        self.lambda_terminal = float(train_cfg["lambda_terminal"])

        self.lambda_moments = float(train_cfg["lambda_moments"])
        self.lambda_sw2 = float(train_cfg["lambda_sw2"])
        self.sw2_proj = int(train_cfg["sw2_proj"])
        self.moment_eps = float(train_cfg["moment_eps"])

    def terminal_loss(self, zT: torch.Tensor, z_tar: torch.Tensor) -> torch.Tensor:
        # moments
        mean_loss = (zT.mean(0) - z_tar.mean(0)).pow(2).mean()
        std_loss = ((zT.std(0, unbiased=False) + self.moment_eps) -
                    (z_tar.std(0, unbiased=False) + self.moment_eps)).pow(2).mean()

        # sliced W2
        sw2 = sliced_w2(zT, z_tar, self.sw2_proj)

        return self.lambda_moments * (mean_loss + std_loss) + self.lambda_sw2 * sw2

    def rollout(self, z0: torch.Tensor, z_tar: torch.Tensor):
        z = z0
        running_cost = torch.tensor(0.0, device=z.device)

        dt = self.T / self.num_steps
        sqrt_dt = math.sqrt(dt)

        for k in range(self.num_steps):
            # Use *continuous* time in [0,1]
            t = torch.full((z.size(0),), float(k) / float(self.num_steps), device=z.device)
            u = self.policy(z, t)

            running_cost = running_cost + 0.5 * u.pow(2).mean() * dt

            if self.eps > 0:
                z = z + u * dt + self.eps * sqrt_dt * torch.randn_like(z)
            else:
                z = z + u * dt

        term = self.terminal_loss(z, z_tar)
        total = self.lambda_running * running_cost + self.lambda_terminal * term
        return z, total, running_cost.detach(), term.detach()

    @torch.no_grad()
    def sample_path(self, z0: torch.Tensor):
        z = z0
        dt = self.T / self.num_steps
        sqrt_dt = math.sqrt(dt)
        for k in range(self.num_steps):
            t = torch.full((z.size(0),), float(k) / float(self.num_steps), device=z.device)
            u = self.policy(z, t)
            if self.eps > 0:
                z = z + u * dt + self.eps * sqrt_dt * torch.randn_like(z)
            else:
                z = z + u * dt
        return z

    def get_drift(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Drift at (t, z). t: [B] or scalar; z: [B, z_dim]. For compatibility with project inference."""
        if t.dim() == 0:
            t = t.expand(z.size(0))
        return self.policy(z, t)


# ============================================================
# Training: sampler-based API
# ============================================================

def training_mfc(
    sampler_x,
    sampler_y,
    solver: MFCSolver,
    train_cfg: Optional[dict] = None,
    batch_size: int = 64,
    device: str = "cpu",
    seed: Optional[int] = 42,
    test_every: Optional[int] = None,
    num_test_images: int = 16,
    alae_model=None,
    x_inds_test=None,
    test_latents=None,
    custom_img_path: Optional[str] = None,
    custom_img_size: int = 1024,
    metrics_hook=None, metrics_n_gen=512, **kwargs
) -> MFCSolver:
    """
    Train the MFC solver using latent samplers only (no images, no latent model).

    sampler_x, sampler_y: objects with .sample(batch_size) returning tensors of shape (B, z_dim).
    train_cfg: dict with keys from DEFAULT_MFC_TRAIN_CFG. If None, uses DEFAULT_MFC_TRAIN_CFG.
    test_every: if set, every test_every epochs generate num_test_images young-from-adult images
        and log to wandb (requires alae_model, x_inds_test, test_latents).
    """
    if train_cfg is None:
        train_cfg = dict(DEFAULT_MFC_TRAIN_CFG)
    else:
        cfg = dict(DEFAULT_MFC_TRAIN_CFG)
        cfg.update(train_cfg)
        train_cfg = cfg

    if seed is not None:
        seed_everything(seed)

    solver = solver.to(device)
    solver.train()
    opt = torch.optim.AdamW(solver.policy.parameters(), lr=train_cfg["lr"])
    log_every = train_cfg.get("log_every", 25)
    save_every = train_cfg.get("save_every", 100)

    use_wandb = test_every is not None and alae_model is not None and x_inds_test is not None and test_latents is not None

    for ep in range(1, train_cfg["epochs"] + 1):
        z0 = sampler_x.sample(batch_size).to(device)
        z_tar = sampler_y.sample(batch_size).to(device)

        opt.zero_grad(set_to_none=True)
        zT, loss, run_cost, term_cost = solver.rollout(z0, z_tar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.policy.parameters(), train_cfg["grad_clip"])
        opt.step()

        if use_wandb:
            try:
                import wandb
                wandb.log({"train/loss": loss.item(), "train/run_cost": run_cost.item(), "train/term_cost": term_cost.item()}, step=ep)
            except Exception:
                pass
            if ep % test_every == 0 or ep == 1:
                try:
                    import wandb
                    from face_utils import generate_young_images_grid_mfc
                    grid = generate_young_images_grid_mfc(
                        solver, alae_model, x_inds_test, test_latents,
                        num_images=num_test_images, device=device, 
                    )
                    wandb.log({"young_from_adult": wandb.Image(grid)}, step=ep)

                    if metrics_hook is not None:
                        try:
                            metrics = metrics_hook(ep, solver)
                            if isinstance(metrics, dict) and len(metrics) > 0:
                                wandb.log(metrics, step=ep)
                        except Exception as e:
                            print(f"[MFC] metrics_hook failed: {e}")
                    
                    if custom_img_path is not None and os.path.isfile(custom_img_path):
                        from face_utils import translate_custom_image_mfc_to_numpy
                        inp_np, recon_np, out_np = translate_custom_image_mfc_to_numpy(
                            solver=solver,
                            alae_model=alae_model,      # CPU model
                            img_path=custom_img_path,
                            device_train=device,        # GPU
                            img_size=custom_img_size,
                        )
                        wandb.log(
                            {
                                "custom/input": wandb.Image(inp_np),
                                "custom/recon": wandb.Image(recon_np),     # encode->decode sanity check
                                "custom/output": wandb.Image(out_np),
                            },
                            step=ep,
                        )
                except Exception as e:
                    print(f"[MFC] wandb image log skipped: {e}")

        if ep % log_every == 0 or ep == 1:
            print(
                f"[MFC ep {ep:05d}] loss={loss.item():.4f}  run={run_cost.item():.4f}  term={term_cost.item():.4f}"
            )

    solver.eval()
    return solver


# ============================================================
# Standalone: data loaders + latent model (only for __main__ with image folders)
# ============================================================

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build ImageFolder loaders for adult/child image dirs. Used only in standalone mode."""
    img_size = cfg["img_size"]
    tform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    adult_ds = tv_datasets.ImageFolder(root=cfg["adult_root"], transform=tform)
    child_ds = tv_datasets.ImageFolder(root=cfg["child_root"], transform=tform)
    adult_loader = DataLoader(
        adult_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], pin_memory=(cfg["device"] == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
    )
    child_loader = DataLoader(
        child_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], pin_memory=(cfg["device"] == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
    )
    return adult_loader, child_loader


class _LatentSamplerFromLoader:
    """Wraps an image DataLoader + LatentModel to expose .sample(batch_size) -> latents [B, z_dim]."""

    def __init__(self, loader: DataLoader, latent_model, device: str):
        self.loader = loader
        self.latent_model = latent_model
        self.device = device
        self._iter = None

    def _next_batch(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            imgs, _ = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            imgs, _ = next(self._iter)
        return imgs.to(self.device)

    def sample(self, batch_size: int):
        imgs = self._next_batch()
        if imgs.size(0) != batch_size:
            imgs = self._next_batch()  # allow different batch_size if loader has fixed size
        with torch.no_grad():
            z = self.latent_model.encode(imgs[:batch_size])
        return z


class LatentModel(nn.Module):
    """Frozen FFHQ encoder and decoder for the image-folder experiment."""

    def __init__(self, device):
        super().__init__()
        from face_data import load_alae_model
        self.model = load_alae_model(device)

    @torch.no_grad()
    def encode(self, images):
        from alae.alae_ffhq_inference import encode
        return encode(self.model, images)

    @torch.no_grad()
    def decode(self, latents):
        from alae.alae_ffhq_inference import decode
        return decode(self.model, latents)


def _make_translation_grid(latent: LatentModel, z0: torch.Tensor, zT: torch.Tensor, nrow: int):
    x0 = latent.decode(z0)
    xT = latent.decode(zT)
    grid0 = make_grid(denorm01(x0).cpu(), nrow=nrow)
    gridT = make_grid(denorm01(xT).cpu(), nrow=nrow)
    return grid0, gridT


def train_standalone(cfg: dict):
    """Run MFC training with image folders and an ALAE latent model (standalone mode)."""
    device = cfg["device"]
    seed_everything(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    adult_loader, child_loader = build_dataloaders(cfg)
    latent = LatentModel(device)
    sampler_x = _LatentSamplerFromLoader(adult_loader, latent, device)
    sampler_y = _LatentSamplerFromLoader(child_loader, latent, device)

    z_dim = int(cfg["latent"]["z_dim"])
    policy = DriftPolicy(z_dim=z_dim)
    solver = MFCSolver(policy, cfg["train"])
    solver = training_mfc(
        sampler_x, sampler_y, solver,
        train_cfg=cfg["train"],
        batch_size=cfg["batch_size"],
        device=device,
        seed=cfg["seed"],
        custom_img_path=cfg.get("custom_image_path"),
        custom_img_size=cfg.get("custom_image_size", 1024),
    )

    out_dir = cfg["out_dir"]
    torch.save({"policy": solver.policy.state_dict(), "cfg": cfg}, os.path.join(out_dir, "policy_final.pt"))

    # One visualization batch
    n = min(cfg.get("num_sample", 16), cfg["batch_size"])
    adult_imgs, _ = next(iter(adult_loader))
    adult_imgs = adult_imgs.to(device)[:n]
    with torch.no_grad():
        z0s = latent.encode(adult_imgs)
        zTs = solver.sample_path(z0s)
    grid0, gridT = _make_translation_grid(latent, z0s, zTs, nrow=int(math.sqrt(n)))
    save_image(gridT, os.path.join(out_dir, "adult2child_final.png"))
    save_image(torch.cat([grid0.unsqueeze(0), gridT.unsqueeze(0)], dim=0), os.path.join(out_dir, "adult2child_sidebyside_final.png"), nrow=1)


# Full config for standalone run (image folders + ALAE); not used by project pipeline.
CONFIG_STANDALONE = {
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "seed": 42,
    "adult_root": "./data/adult",
    "child_root": "./data/child",
    "img_size": 1024,
    "batch_size": 16,
    "num_workers": 4,
    "latent": {
        "z_dim": 512,
    },
    "train": dict(DEFAULT_MFC_TRAIN_CFG),
    "out_dir": "logs_adult2child_mfc",
    "num_sample": 16,
    "custom_image_path": os.path.join("data", "custom", "new.png"),
    "custom_image_size": 1024,
}


if __name__ == "__main__":
    train_standalone(CONFIG_STANDALONE)
