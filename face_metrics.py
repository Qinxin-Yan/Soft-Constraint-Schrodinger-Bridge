"""Optional FID and KID metrics for face translation."""

from typing import Tuple
import numpy as np
import torch
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception as e:
    FrechetInceptionDistance = None
    KernelInceptionDistance = None

def _require_torchmetrics():
    if FrechetInceptionDistance is None or KernelInceptionDistance is None:
        raise ImportError(
            "torchmetrics is required for FID/KID.\n"
            "Install:\n"
            "  python -m pip install -U torchmetrics torchvision\n"
        )


def _to_uint8_nchw(img: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor to uint8 NCHW in [0,255].
    Accepts:
      - float in [0,1] or [-1,1]
      - uint8 in [0,255]
    Shapes:
      - NCHW or NHWC (will convert to NCHW)
    """
    if img.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {img.shape}")

    # NHWC -> NCHW
    if img.shape[-1] == 3 and img.shape[1] != 3:
        img = img.permute(0, 3, 1, 2).contiguous()

    if img.dtype == torch.uint8:
        return img

    img = img.float()
    if img.min() < -0.1:  # likely [-1,1]
        img = (img * 0.5 + 0.5).clamp(0, 1)
    else:
        img = img.clamp(0, 1)

    img = (img * 255.0).round().to(torch.uint8)
    return img


def _iter_batches(n: int, bs: int):
    for s in range(0, n, bs):
        yield s, min(s + bs, n)

class FIDKIDEvaluator:
    """
    Precompute real-children stats once.
    Then at each eval: load real-state, update with generated fake images, compute FID/KID.
    """
    def __init__(
        self,
        *,
        child_real_images_u8: torch.Tensor,   # uint8 NCHW CPU
        metric_device: str = "cuda",
        real_bs: int = 64,
        kid_subsets: int = 50,
        kid_subset_size: int = 1000,
    ):
        _require_torchmetrics()
        self.metric_device = metric_device

        # Build "real-state" once
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
        kid = KernelInceptionDistance(
            subset_size=min(kid_subset_size, child_real_images_u8.shape[0]),
            subsets=kid_subsets,
            normalize=False,
        ).to(metric_device)

        # Update metrics with real images
        real = child_real_images_u8
        n_real = real.shape[0]
        for s in range(0, n_real, real_bs):
            batch = real[s:s+real_bs].to(metric_device)
            fid.update(batch, real=True)
            kid.update(batch, real=True)

        # Save state dicts (these contain real distribution stats)
        fid.persistent(True)
        kid.persistent(True)
        self._fid_real_state = fid.state_dict()
        self._kid_real_state = kid.state_dict()

        # Keep configs to re-instantiate fresh objects each time
        self._kid_subsets = kid_subsets
        self._kid_subset_size = kid_subset_size

    @torch.no_grad()
    def compute_from_fake_uint8(self, fake_u8_nchw: torch.Tensor, fake_bs: int = 64):
        """fake_u8_nchw: uint8 NCHW CPU or GPU. Returns (fid, kid_mean)."""
        _require_torchmetrics()
        md = self.metric_device

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(md)
        kid = KernelInceptionDistance(
            subset_size=min(self._kid_subset_size, fake_u8_nchw.shape[0]),
            subsets=self._kid_subsets,
            normalize=False,
        ).to(md)

        # Load real state
        fid.persistent(True)
        kid.persistent(True)
        fid.load_state_dict(self._fid_real_state)
        kid.load_state_dict(self._kid_real_state)

        # Update with fake
        n_fake = fake_u8_nchw.shape[0]
        for s in range(0, n_fake, fake_bs):
            batch = fake_u8_nchw[s:s+fake_bs].to(md)
            fid.update(batch, real=False)
            kid.update(batch, real=False)

        fid_val = float(fid.compute().detach().cpu().item())
        kid_mean, _kid_std = kid.compute()
        kid_val = float(kid_mean.detach().cpu().item())
        return fid_val, kid_val


@torch.no_grad()
def _decode_latents_to_images_uint8(
    alae_model,
    latents: torch.Tensor,   # [N, 512], CPU or GPU
    decode_bs: int = 32,
    device_decode: str = "cpu",
) -> torch.Tensor:
    """
    Decode latents -> uint8 images using ALAE decode() in mini-batches.
    Returns uint8 NCHW [N,3,H,W] in [0,255] on CPU.
    """
    from alae.alae_ffhq_inference import decode

    alae_model = alae_model.to(device_decode)
    alae_model.eval()
    alae_model.requires_grad_(False)

    latents = latents.detach()
    outs = []
    for s, e in _iter_batches(latents.shape[0], decode_bs):
        z = latents[s:e].to(device_decode)
        x = decode(alae_model, z)  # expected float in [-1,1], NCHW
        x_u8 = _to_uint8_nchw(x).cpu()
        outs.append(x_u8)
    return torch.cat(outs, dim=0)


def _load_images_from_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr


def _infer_image_array_layout(arr: np.ndarray) -> str:
    """
    Return one of: 'NCHW_uint8', 'NHWC_uint8', 'NCHW_float', 'NHWC_float'
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected images array with 4 dims, got {arr.shape}")

    # common cases
    if arr.shape[1] == 3:
        ch_first = True
    elif arr.shape[-1] == 3:
        ch_first = False
    else:
        # fall back: assume NHWC
        ch_first = False

    is_uint8 = (arr.dtype == np.uint8)
    if ch_first and is_uint8:
        return "NCHW_uint8"
    if (not ch_first) and is_uint8:
        return "NHWC_uint8"
    if ch_first and (arr.dtype in (np.float16, np.float32, np.float64)):
        return "NCHW_float"
    return "NHWC_float"


def _select_child_images(
    images_npy: np.ndarray,
    ages_full: np.ndarray,
    train_size: int,
    child_mask_test: np.ndarray,
) -> torch.Tensor:
    """
    Select real children images for the *test portion* using `child_mask_test`,
    returning a torch tensor of shape [N,3,H,W] uint8 on CPU.

    Assumption:
      - images_npy aligns with full dataset ordering (same as latents.npy),
        OR it only contains test images (length == total_train_test - train_size).
    """
    layout = _infer_image_array_layout(images_npy)

    total_full = ages_full.shape[0]
    test_len_expected = total_full - train_size

    if images_npy.shape[0] == total_full:
        # full dataset images
        test_images = images_npy[train_size:]
    elif images_npy.shape[0] == test_len_expected:
        # only test images
        test_images = images_npy
    else:
        raise ValueError(
            f"test_images.npy length {images_npy.shape[0]} does not match full {total_full} "
            f"or test-only {test_len_expected}. Please check dataset alignment."
        )

    child_imgs = test_images[child_mask_test]  # subset of test images

    t = torch.from_numpy(child_imgs)
    # Convert to NCHW uint8
    if layout.startswith("NHWC"):
        t = t.permute(0, 3, 1, 2).contiguous()
    if t.dtype != torch.uint8:
        t = _to_uint8_nchw(t)
    return t.cpu()


@torch.no_grad()
def compute_fid_kid_gen_vs_child(
    *,
    solver,
    alae_model,
    adult_test_latents: np.ndarray,      # [N_test, 512], numpy
    child_real_images_u8: torch.Tensor,  # [N_child,3,H,W], uint8
    device_train: torch.device,
    num_gen: int = 5000,
    num_real: int = 5000,
    rollout_bs: int = 64,
    decode_bs: int = 32,
    metric_device: str = "cuda",
    kid_subsets: int = 50,
    kid_subset_size: int = 1000,
) -> Tuple[float, float]:
    """
    Generate children images from adult latents via solver.sample_path, decode with ALAE,
    and compute FID/KID against real children images.
    """
    _require_torchmetrics()

    # --- pick subsets (avoid huge memory) ---
    adult_test_latents = np.asarray(adult_test_latents)
    n_adult = adult_test_latents.shape[0]
    n_gen = min(num_gen, n_adult)
    gen_idx = np.random.choice(n_adult, size=n_gen, replace=False)
    z0_all = torch.tensor(adult_test_latents[gen_idx], dtype=torch.float32)

    n_real = min(num_real, child_real_images_u8.shape[0])
    real_idx = torch.randperm(child_real_images_u8.shape[0])[:n_real]
    real_imgs = child_real_images_u8[real_idx]  # uint8 on CPU

    # --- init metrics ---
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    kid = KernelInceptionDistance(
        subset_size=min(kid_subset_size, n_gen, n_real),
        subsets=kid_subsets,
        normalize=False,
    ).to(metric_device)

    # --- update real first (children) ---
    for s, e in _iter_batches(n_real, bs=rollout_bs):
        batch = real_imgs[s:e].to(metric_device)
        fid.update(batch, real=True)
        kid.update(batch, real=True)

    # --- generate + decode in chunks, update fake ---
    for s, e in _iter_batches(n_gen, bs=rollout_bs):
        z0 = z0_all[s:e].to(device_train)
        zT = solver.sample_path(z0).detach().cpu()  # [bs,512] on CPU

        fake_u8 = _decode_latents_to_images_uint8(
            alae_model=alae_model,
            latents=zT,
            decode_bs=decode_bs,
            device_decode="cpu",
        )  # uint8 CPU

        fake_u8 = fake_u8.to(metric_device)
        fid.update(fake_u8, real=False)
        kid.update(fake_u8, real=False)

    fid_val = float(fid.compute().detach().cpu().item())
    kid_mean, kid_std = kid.compute()
    kid_val = float(kid_mean.detach().cpu().item())

    return fid_val, kid_val

@torch.no_grad()
def build_real_child_images_from_latents(
    *,
    alae_model,
    latents_full: np.ndarray,   # latents.npy, shape [N,512]
    ages_full: np.ndarray,      # age.npy, shape [N]
    num_real: int = 5000,
    decode_bs: int = 32,
    device_decode: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """
    Build the REAL reference images for FID/KID by:
      1) selecting child indices from ages_full (<18, !=-1)
      2) decoding their latents with ALAE
    Returns uint8 NCHW [num_real,3,H,W] on CPU.
    """
    rng = np.random.default_rng(seed)
    ages_full = np.asarray(ages_full).reshape(-1)
    child_idx = np.where((ages_full < 18) & (ages_full != -1))[0]
    if child_idx.size == 0:
        raise RuntimeError("No child samples found in ages_full (<18).")

    n = min(num_real, child_idx.size)
    sel = rng.choice(child_idx, size=n, replace=False)
    z = torch.tensor(latents_full[sel], dtype=torch.float32)

    real_u8 = _decode_latents_to_images_uint8(
        alae_model=alae_model,
        latents=z,
        decode_bs=decode_bs,
        device_decode=device_decode,
    )
    return real_u8  # uint8 CPU
