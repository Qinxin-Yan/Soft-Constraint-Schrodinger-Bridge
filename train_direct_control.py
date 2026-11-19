"""Train the direct-control solver on FFHQ adult and child latents."""

import os
import torch

from direct_control import DEFAULT_MFC_TRAIN_CFG, DriftPolicy, MFCSolver, training_mfc
from face_data import load_face_split, load_alae_model, start_logging


CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dim": 512,
    "batch_size": 512,
    "train_size": 60000,
    "test_size": 10000,
    "Testevery": 50,
    "num_test_images": 36,
    "wandb_project": "Soft-Constraint-Schrodinger-Bridge",
    "wandb_run_name": None,
    "model_dir": "model",
    "data_dir": "data",
    # MFC training (override DEFAULT_MFC_TRAIN_CFG as needed)
    "train": {
        "epochs": 800,
        "lr": 2e-4,
        "grad_clip": 1.0,
        "T": 1.0,
        "num_steps": 50,
        "eps": 0.5,
        "lambda_running": 0.05,
        "lambda_terminal": 5.0,
        "lambda_moments": 10.0,
        "lambda_sw2": 5.0,
        "sw2_proj": 512,
        "moment_eps": 1e-6,
        "log_every": 50,
        "save_every": 50,
    },
}


def main():
    cfg = CONFIG
    device = torch.device(cfg["device"])
    X_sampler, Y_sampler, test_latents, x_inds_test = load_face_split(cfg, device)
    train_cfg = {**DEFAULT_MFC_TRAIN_CFG, **cfg["train"]}
    solver = MFCSolver(DriftPolicy(cfg["dim"]), train_cfg)
    run = start_logging(cfg)
    test_every = cfg["Testevery"] if run is not None else 0
    num_test_images = cfg["num_test_images"]
    alae_model = load_alae_model("cpu") if test_every > 0 else None

    solver = training_mfc(
        X_sampler,
        Y_sampler,
        solver,
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
    )

    os.makedirs(cfg["model_dir"], exist_ok=True)
    save_path = os.path.join(cfg["model_dir"], "mfc_policy.pt")
    torch.save({"policy": solver.policy.state_dict(), "train_cfg": train_cfg}, save_path)
    print("Saved", save_path)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
