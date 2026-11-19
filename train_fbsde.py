"""Train the McKean–Vlasov FBSDE solver on FFHQ adult and child latents."""

import os
import numpy as np
import torch

from FBSDE import MKVFBSDEModel, training_fbsde
from face_data import load_face_split, load_alae_model, start_logging


CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dim": 512,
    "batch_size": 215,
    "train_size": 60000,
    "test_size": 10000,
    "Testevery": 50,
    "num_test_images": 36,
    "wandb_project": "Soft-Constraint-Schrodinger-Bridge",
    "wandb_run_name": "FBSDE_adult2child",
    "model_dir": "model",
    "data_dir": "data",
    "train": {
        "epochs": 1800,
        "lr": 2e-4,
        "grad_clip": 1.0,
        "T": 1.0,
        "num_steps": 60,
        "eps": 0.5,

        "k_terminal": 100.0,   # your k in Y_T = k ∂_μ G

        # Fourier–Wasserstein^2 derivative hyperparameters
        "fw_nfreq": 2048,
        "fw_lam": 2.0,
        "fw_xi_scale": 0.01,

        "log_every": 50,
        "save_every": 50,
    },
}


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    X_sampler, Y_sampler, test_latents, x_inds_test = load_face_split(cfg, device)
    train_cfg = dict(cfg["train"])
    model = MKVFBSDEModel(z_dim=cfg["dim"], hidden=256, depth=3)
    run = start_logging(cfg)
    test_every = cfg["Testevery"] if run is not None else 0
    num_test_images = min(cfg["num_test_images"], len(x_inds_test))
    alae_model = load_alae_model("cpu") if test_every > 0 else None
    rng = np.random.default_rng(12345)
    fixed_test_idx = rng.choice(x_inds_test, size=num_test_images, replace=False)

    model = training_fbsde(
        sampler_x=X_sampler,
        sampler_y=Y_sampler,
        model=model,
        train_cfg=train_cfg,
        batch_size=cfg["batch_size"],
        device=device,
        seed=42,
        test_every=test_every,
        num_test_images=num_test_images,
        alae_model=alae_model,
        x_inds_test=x_inds_test,
        test_latents=test_latents,
        custom_img_path=os.path.join(cfg["data_dir"], "custom", "new.png"),
        custom_img_size=1024,
        fixed_test_idx=fixed_test_idx,
    )

    os.makedirs(cfg["model_dir"], exist_ok=True)
    save_path = os.path.join(cfg["model_dir"], "fbsde_policy.pt")
    torch.save({"model": model.state_dict(), "train_cfg": train_cfg}, save_path)
    print("Saved", save_path)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
