"""McKean–Vlasov FBSDE solver for soft-constrained distribution steering."""
import os
from dataclasses import dataclass
from dataclasses import fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------ basic MLP ------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3, act=nn.SiLU):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(act())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------ Sliced Wasserstein^2 (empirical) ------------------
def sliced_wasserstein2_empirical(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256) -> torch.Tensor:
    """
    x: [B, d], y: [B2, d]
    Returns scalar SW2^2(x,y) computed with random 1D projections.
    """
    device = x.device
    d = x.shape[1]
    # random unit vectors
    proj = torch.randn(n_proj, d, device=device)
    proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-12)

    x1 = x @ proj.t()  # [B, n_proj]
    y1 = y @ proj.t()  # [B2, n_proj]

    x1_sorted, _ = torch.sort(x1, dim=0)
    y1_sorted, _ = torch.sort(y1, dim=0)

    m = min(x1_sorted.shape[0], y1_sorted.shape[0])
    diff = x1_sorted[:m] - y1_sorted[:m]
    sw2 = (diff * diff).mean()  # mean over samples and projections
    return sw2


@torch.no_grad()
def _time_grid(T: float, num_steps: int, device):
    dt = T / num_steps
    t = torch.linspace(0.0, T, num_steps + 1, device=device)
    return t, dt


@dataclass
class FBSDETrainCfg:
    epochs: int = 800
    lr: float = 2e-4
    grad_clip: float = 1.0
    T: float = 1.0
    num_steps: int = 50
    eps: float = 0.05

    k_terminal: float = 10.0

    fw_nfreq: int = 4098
    fw_lam: float = 1.0
    fw_xi_scale: float = 0.3

    log_every: int = 50
    save_every: int = 50
@torch.no_grad()
def _sample_xi(n_freq: int, d: int, device, xi_scale: float = 1.0):
    """
    Frequencies for Monte Carlo integration.
    We use xi ~ N(0, xi_scale^2 I). (You can swap to other samplers.)
    """
    return xi_scale * torch.randn(n_freq, d, device=device)


def fourier_wasserstein_sq_partial_mu_empirical(
    x: torch.Tensor,           # [B,d] query points (usually xT)
    x_particles: torch.Tensor, # [B,d] particles for μ (usually xT)
    y_tar: torch.Tensor,       # [B2,d] target samples
    n_freq: int = 1024,
    lam: float = 2.0,
    xi_scale: float = 0.044,   # IMPORTANT in 512-d
    chunk_size: int = 256,     # memory control
) -> torch.Tensor:
    """
    ∂_μ G(μ)(x) for Fourier–Wasserstein^2 with Monte Carlo frequencies.
    Memory-safe: uses (B×m) complex matmul instead of (B×m×d).
    Returns real tensor [B,d].
    """
    device = x.device
    B, d = x.shape
    #print("xT norm mean", x_particles.norm(dim=1).mean().item())
    grad_acc = torch.zeros(B, d, device=device)

    # sample all xi once (or sample per chunk; either is fine)
    xi_all = xi_scale * torch.randn(n_freq, d, device=device)  # [M,d]
    n_chunks = (n_freq + chunk_size - 1) // chunk_size

    for s in range(0, n_freq, chunk_size):
        xi = xi_all[s:s+chunk_size]          # [m,d]
        m = xi.shape[0]

        # weights 1/(1+|xi|^2)^lam
        w = 1.0 / (1.0 + (xi * xi).sum(dim=1)).pow(lam)   # [m]

        # φ_μ(ξ) and φ_tar(ξ)
        dot_mu  = x_particles @ xi.t()      # [B,m]
        dot_tar = y_tar       @ xi.t()      # [B2,m]
        phi_mu  = torch.exp(1j * dot_mu).mean(dim=0)      # [m] complex
        phi_tar = torch.exp(1j * dot_tar).mean(dim=0)     # [m] complex
        diff    = (phi_mu - phi_tar) * w.to(torch.complex64)  # [m] complex

        # e^{-i<xi,x>}
        dot_x = x @ xi.t()                   # [B,m]
        exp_minus = torch.exp(-1j * dot_x)   # [B,m] complex

        # coeff_{b,m} = (-i) e^{-i<xi,x_b>} * diff_m
        coeff = (-1j) * exp_minus * diff.view(1, m)       # [B,m] complex

        # grad_chunk = 2 * Re( (1/m) * coeff @ xi )
        grad_chunk = (coeff @ xi.to(torch.complex64)).real / float(m)  # [B,d]
        grad_acc += grad_chunk

    grad = 2.0 * grad_acc / float(n_chunks)
    return grad


class MKVFBSDEModel(nn.Module):
    """
    Learns y0(x) and z(t,x), then simulates:
      X_{i+1} = X_i - eps^2 Y_i dt + eps sqrt(dt) dW_i
      Y_{i+1} = Y_i + Z_i dW_i
    with b=0 and Dx b = 0.

    Terminal loss: E|Y_T - k * approx_partial_mu_G|^2
    """
    def __init__(self, z_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.y0_net = MLP(z_dim, z_dim, hidden=hidden, depth=depth)
        # z_net takes (t, x) -> z_dim
        self.z_net = MLP(1 + z_dim, z_dim, hidden=hidden, depth=depth)

    def forward_simulate(
        self,
        x0: torch.Tensor,          # [B,d]
        eps: float,
        T: float,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          xT: [B,d]
          yT: [B,d]
        """
        device = x0.device
        B, d = x0.shape
        _, dt = _time_grid(T, num_steps, device)
        sqrtdt = dt**0.5

        x = x0
        y = self.y0_net(x0)

        for i in range(num_steps):
            t_i = torch.full((B, 1), float(i) * dt, device=device)
            z = self.z_net(torch.cat([t_i, x], dim=1))  # [B,d]
            dW = torch.randn(B, d, device=device) * sqrtdt

            # X step (b=0)
            x = x - (eps**2) * y * dt + eps * dW
            # Y step (Dx b = 0)
            y = y + z * dW

        return x, y

    def terminal_target_partial_mu_G(
        self,
        xT: torch.Tensor,          # [B,d] requires grad
        y_tar: torch.Tensor,       # [B2,d]
        sw2_proj: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Approximate ∂_μ G(μ_T)(x) using gradient of empirical penalty wrt particles.

        We set G(μ)=SW2^2(μ, μ_tar). Empirical: G_emp(xT, y_tar).
        Then use autograd: grad_xT G_emp -> [B,d].
        The scaling between grad wrt samples and Lions derivative depends on conventions;
        we include a tunable `scale` to absorb constants (often 1.0 works in practice).
        """
        xT_req = xT.detach().requires_grad_(True)
        lossG = sliced_wasserstein2_empirical(xT_req, y_tar, n_proj=sw2_proj)
        (grad_xT,) = torch.autograd.grad(lossG, xT_req, create_graph=False, retain_graph=False)
        return scale * grad_xT.detach()  # [B,d]


def training_fbsde(
    sampler_x,
    sampler_y,
    model: MKVFBSDEModel,
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
    fixed_test_idx=None,
) -> MKVFBSDEModel:

    if train_cfg is None:
        cfg = FBSDETrainCfg()
    else:
        # start from defaults
        base = FBSDETrainCfg().__dict__.copy()
        base.update(train_cfg)

    # keep only fields that exist in the dataclass
        allowed = {f.name for f in fields(FBSDETrainCfg)}
        base = {k: v for k, v in base.items() if k in allowed}

        cfg = FBSDETrainCfg(**base)
        #print("[debug] k_terminal =", cfg.k_terminal)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    use_wandb = (
        test_every is not None
        and alae_model is not None
        and x_inds_test is not None
        and test_latents is not None
    )

    for ep in range(1, cfg.epochs + 1):
        x0 = sampler_x.sample(batch_size).to(device)  # adults
        ytar = sampler_y.sample(batch_size).to(device)  # children target batch

        opt.zero_grad(set_to_none=True)

        xT, yT = model.forward_simulate(
            x0=x0, eps=cfg.eps, T=cfg.T, num_steps=cfg.num_steps
        )
        #print("mean |sigma^2 * yT|:", (cfg.eps**2) * yT.norm(dim=1).mean().item())
        #print("mean |sigma * dW| scale:", cfg.eps * math.sqrt(cfg.T/cfg.num_steps))

        # terminal condition target: k * ∂_μ G(μ_T)(X_T)
        target = cfg.k_terminal * fourier_wasserstein_sq_partial_mu_empirical(
            x=xT,
            x_particles=xT,
            y_tar=ytar,
            n_freq=cfg.fw_nfreq,
            lam=cfg.fw_lam,
            xi_scale=cfg.fw_xi_scale,
        )
        #print(target)
        loss = F.mse_loss(yT, target)
        #print("target mean abs", target.abs().mean().item(),
        #"target max abs", target.abs().max().item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if use_wandb:
            try:
                import wandb
                wandb.log({"train/fbsde_loss": float(loss.item())}, step=ep)
            except Exception:
                pass

            if ep % test_every == 0 or ep == 1:
                try:
                    import wandb
                    from face_utils import generate_young_images_grid_fbsde
                    grid = generate_young_images_grid_fbsde(
                        model=model,
                        alae_model=alae_model,
                        fixed_test_idx=fixed_test_idx,   # <--- use fixed
                        test_latents=test_latents,
                        num_images=num_test_images,
                        device=device,
                        eps=cfg.eps,
                        T=cfg.T,
                        num_steps=cfg.num_steps,
                    )
                    wandb.log({"young_from_adult_fbsde": wandb.Image(grid)}, step=ep)

                    if custom_img_path is not None and os.path.isfile(custom_img_path):
                        from face_utils import translate_custom_image_fbsde_to_numpy
                        inp_np, recon_np, out_np = translate_custom_image_fbsde_to_numpy(
                            model=model,
                            alae_model=alae_model,
                            img_path=custom_img_path,
                            device_train=device,
                            img_size=custom_img_size,
                            eps=cfg.eps,
                            T_horizon=cfg.T,
                            num_steps=cfg.num_steps,
                        )
                        wandb.log(
                            {
                                "custom/input": wandb.Image(inp_np),
                                "custom/recon": wandb.Image(recon_np),
                                "custom/output_fbsde": wandb.Image(out_np),
                            },
                            step=ep,
                        )
                except Exception as e:
                    print(f"[FBSDE] wandb image log skipped: {e}")

        if ep % cfg.log_every == 0 or ep == 1:
            print(f"[FBSDE ep {ep:05d}] loss={loss.item():.6f}")

    model.eval()
    return model