"""Translate an aligned face with a trained FBSDE or direct-control checkpoint."""

import argparse

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from alae.alae_ffhq_inference import encode, decode
from direct_control import DriftPolicy, MFCSolver
from face_data import load_alae_model
from FBSDE import MKVFBSDEModel


@torch.no_grad()
def translate(checkpoint, input_path, output_path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    alae = load_alae_model("cpu")
    transform = transforms.Compose([
        transforms.Resize(1024), transforms.CenterCrop(1024),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    with Image.open(input_path) as image:
        initial = encode(alae, transform(image.convert("RGB")).unsqueeze(0))
    dim = initial.shape[1]
    if "policy" in state:
        cfg = state["train_cfg"] if "train_cfg" in state else state["cfg"]["train"]
        policy = DriftPolicy(dim)
        policy.load_state_dict(state["policy"])
        terminal = MFCSolver(policy, cfg).eval().sample_path(initial)
    else:
        cfg = state["train_cfg"]
        model = MKVFBSDEModel(dim).eval()
        model.load_state_dict(state["model"])
        terminal, _ = model.forward_simulate(initial, cfg["eps"], cfg["T"], cfg["num_steps"])
    save_image((decode(alae, terminal) * 0.5 + 0.5).clamp(0, 1), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to fbsde_policy.pt or mfc_policy.pt")
    parser.add_argument("input", help="Aligned adult face image")
    parser.add_argument("output", help="Path for the translated image")
    args = parser.parse_args()
    translate(args.checkpoint, args.input, args.output)
