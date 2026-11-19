"""FFHQ latent sampling and face translation previews for both solvers."""

import math
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image

from alae.alae_ffhq_inference import encode as alae_encode, decode as alae_decode


class TensorSampler:
    def __init__(self, tensor, device='cuda'):
        self.device = device
        self.tensor = torch.clone(tensor).to(device)

    def sample(self, size=5):
        assert size <= self.tensor.shape[0]

        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False),
                           device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)


def generate_young_images_grid_mfc(
    solver,
    alae_model,
    x_inds_test,
    test_latents,
    num_images=16,
    device="cpu",
    nrow=None,
    SEED=21,
):
    assert make_grid is not None, "torchvision is required for make_grid"

    if SEED is not None:
        torch.manual_seed(int(SEED))
        np.random.seed(int(SEED))

    n = min(num_images, len(x_inds_test))
    inds = np.random.choice(len(x_inds_test), size=n, replace=False)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(n)))

    z0 = torch.tensor(test_latents[x_inds_test[inds]], dtype=torch.float32, device=device)
    with torch.no_grad():
        zT = solver.sample_path(z0)

    # ALAE decode on CPU (stable)
    decoded = decode_deterministic(alae_model, zT.detach().cpu())
    decoded_01 = (decoded * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(decoded_01, nrow=nrow, padding=2).detach().cpu()
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return grid_np

@torch.no_grad()
def decode_deterministic(alae_model, z):
    z = z[:, None, :].repeat(1, alae_model.mapping_f.num_layers, 1)
    layer_count = 9
    decoded = []
    for i in range(z.shape[0]):
        r = alae_model.decoder(z[i][None, ...], layer_count - 1, 1, noise=False)
        decoded.append(r)
    return torch.cat(decoded, dim=0)


@torch.no_grad()
def translate_custom_image_mfc_to_numpy(
    solver,
    alae_model,          # keep on CPU
    img_path: str,
    device_train="cuda:0",
    img_size=1024,
    recon_deterministic=True,
):
    tfm = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0)  # CPU, [-1,1]

    def to_uint8(img_bchw_01: torch.Tensor) -> np.ndarray:
        img = img_bchw_01[0].permute(1, 2, 0).contiguous()
        return (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    input_np = to_uint8((x * 0.5 + 0.5).clamp(0, 1))

    # encode using their wrapper (this calls model.encode with correct args)
    z0 = alae_encode(alae_model, x)  # [1,512] CPU

    # recon
    if recon_deterministic:
        recon = decode_deterministic(alae_model, z0)
    else:
        recon = alae_decode(alae_model, z0)  # stochastic

    recon_np = to_uint8((recon * 0.5 + 0.5).clamp(0, 1))

    # MFC on GPU
    zT = solver.sample_path(z0.to(device_train)).cpu()

    # output
    out = decode_deterministic(alae_model, zT) if recon_deterministic else alae_decode(alae_model, zT)
    out_np = to_uint8((out * 0.5 + 0.5).clamp(0, 1))

    return input_np, recon_np, out_np

@torch.no_grad()
def generate_young_images_grid_fbsde(
    model,
    alae_model,
    fixed_test_idx,   # <--- add
    test_latents,
    num_images: int,
    device,
    eps: float,
    T: float,
    num_steps: int,
):
    import numpy as np
    import torch
    from torchvision.utils import make_grid
    from alae.alae_ffhq_inference import decode

    if fixed_test_idx is None:
        raise ValueError("fixed_test_idx is None; pass a fixed index array for reproducible logging.")
    idx = np.asarray(fixed_test_idx)[:num_images]

    z0 = torch.tensor(test_latents[idx], dtype=torch.float32, device=device)

    xT, _ = model.forward_simulate(x0=z0, eps=eps, T=T, num_steps=num_steps)
    xT = xT.detach().cpu()

    x_img = decode(alae_model, xT)
    x01 = (x_img * 0.5 + 0.5).clamp(0, 1)

    grid = make_grid(x01, nrow=int(np.sqrt(num_images)))
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return grid_np

@torch.no_grad()
def translate_custom_image_fbsde_to_numpy(
    model,
    alae_model,
    img_path: str,
    device_train,
    img_size: int,
    eps: float,
    T_horizon: float,
    num_steps: int,
):
    from alae.alae_ffhq_inference import decode

    tfm = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0)  # CPU

    z0 = alae_encode(alae_model, x)

    # recon sanity check
    recon = decode(alae_model, z0)
    recon01 = (recon * 0.5 + 0.5).clamp(0, 1)
    recon_np = (recon01[0].permute(1,2,0).numpy() * 255).astype("uint8")

    # forward simulate on GPU
    xT, _yT = model.forward_simulate(x0=z0.to(device_train), eps=eps, T=T_horizon, num_steps=num_steps)
    xT = xT.detach().cpu()

    out = decode(alae_model, xT)
    out01 = (out * 0.5 + 0.5).clamp(0, 1)
    out_np = (out01[0].permute(1,2,0).numpy() * 255).astype("uint8")

    inp01 = (x * 0.5 + 0.5).clamp(0, 1)
    inp_np = (inp01[0].permute(1,2,0).numpy() * 255).astype("uint8")

    return inp_np, recon_np, out_np
