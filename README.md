# Soft-Constraint Schrodinger Bridge

This repository implements the two numerical methods from Section 6 of the manuscript and the adult-to-child face translation experiment in Section 7.2. Both methods start from adult FFHQ faces (age 18 and above) and learn to steer their distribution toward the distribution of faces under 18. Training uses unpaired samples in the 512-dimensional latent space of a fixed ALAE encoder and decoder.

| Method | Implementation | Training entry point |
| --- | --- | --- |
| McKean–Vlasov FBSDE (Algorithm 6.1) | [FBSDE.py](FBSDE.py) learns the initial adjoint and its martingale coefficient using the terminal mismatch for a sampled Fourier penalty. | [train_fbsde.py](train_fbsde.py) |
| Direct mean-field control (Algorithm 6.2) | [direct_control.py](direct_control.py) learns the drift by differentiating the control cost and a terminal penalty combining sliced Wasserstein distance and moment matching. | [train_direct_control.py](train_direct_control.py) |

## Train on FFHQ

From the repository root, install the Python dependencies and run either trainer.

```bash
python -m pip install -r requirements.txt
python train_fbsde.py
python train_direct_control.py
```

Both trainers read aligned rows from `data/latents.npy` and `data/age.npy` and download missing arrays on first use. By default, they split the first 60,000 rows into adult and child training groups, skip unknown ages, and take held-out adult inputs from the next 10,000 rows. Each script's `CONFIG` specifies its experiment settings. These saved settings differ from the illustrative parameters in Section 7.2.

With W&B enabled, the trainers log image grids and use the pretrained FFHQ ALAE weights. Set `WANDB_MODE=offline` for local logging or `WANDB_MODE=disabled` to train on latents without image previews. The trainers write `model/fbsde_policy.pt` and `model/mfc_policy.pt`, respectively.

## Translate a face

After training, apply either checkpoint to an aligned adult portrait. The example below uses the image included in the repository.

```bash
python translate_face.py model/fbsde_policy.pt data/custom/new.png fbsde_face.png
python translate_face.py model/mfc_policy.pt data/custom/new.png direct_control_face.png
```

The translator encodes the input, simulates the learned stochastic dynamics, and decodes a 1024 × 1024 image. The pretrained ALAE checkpoint downloads on first use and is reused from `alae/training_artifacts/ffhq/model_157.pth`.

The face experiment uses FFHQ and the pretrained [Adversarial Latent Autoencoder](https://github.com/podgorskiy/ALAE). The precomputed latent and age arrays follow the setup cited in Section 7.2 of the manuscript.
