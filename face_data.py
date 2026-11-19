"""Shared FFHQ face latents and pretrained ALAE model for the two experiments."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import numpy as np
import torch

from alae.alae_ffhq_inference import load_model
from face_utils import TensorSampler


FFHQ_FILES = {
    "latents.npy": "1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
    "age.npy": "1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
}
ALAE_WEIGHTS = "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth"


def load_face_split(cfg, device):
    directory = Path(cfg["data_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, file_id in FFHQ_FILES.items():
        path = directory / name
        if not path.is_file():
            import gdown

            with TemporaryDirectory(dir=directory) as tmp:
                target = Path(tmp) / name
                if not gdown.download(id=file_id, output=str(target), quiet=False):
                    raise RuntimeError(f"Could not download {name}. Place it in {directory}.")
                target.replace(path)

    latents = np.load(directory / "latents.npy", mmap_mode="r")
    ages = np.load(directory / "age.npy").reshape(-1)
    split = cfg["train_size"]
    if latents.shape != (len(ages), cfg["dim"]):
        raise ValueError("Expected one latent vector of the configured dimension per age label.")
    if not 0 < split < len(ages):
        raise ValueError("train_size must leave both training and held-out examples.")
    train_ages = ages[:split]
    adults = latents[:split][train_ages >= 18]
    children = latents[:split][(train_ages >= 0) & (train_ages < 18)]
    if min(len(adults), len(children)) < cfg["batch_size"]:
        raise ValueError("Each training age group must contain at least batch_size examples.")
    test = slice(split, split + cfg["test_size"])
    adult_test_indices = np.flatnonzero(ages[test] >= 18)
    if not len(adult_test_indices):
        raise ValueError("The held-out split contains no adult faces.")
    return (
        TensorSampler(torch.as_tensor(adults, dtype=torch.float32), device),
        TensorSampler(torch.as_tensor(children, dtype=torch.float32), device),
        latents[test],
        adult_test_indices,
    )


def load_alae_model(device="cpu"):
    directory = Path(__file__).resolve().parent / "alae"
    artifacts = directory / "training_artifacts" / "ffhq"
    weights = artifacts / "model_157.pth"
    if not weights.is_file():
        artifacts.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=artifacts) as tmp:
            target = Path(tmp) / weights.name
            urlretrieve(ALAE_WEIGHTS, target)
            target.replace(weights)
    model = load_model(str(directory / "ffhq.yaml"), str(artifacts))
    return model.eval().requires_grad_(False).to(device)


def start_logging(cfg):
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb

    return wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)
