import torch
import numpy as np
from diffusers import AutoencoderKL
from tqdm import tqdm
import math
import argparse
import yaml
import os

from src.ema import calculate_posthoc_ema
from utils import get_model
from diffusion import create_diffusion


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.result_dir, "config.yaml"), "r") as f:
        train_args = yaml.safe_load(f)

    # Load model
    model = get_model(train_args).to(device)
    model = torch.compile(model)

    if args.ckpt is not None:
        # For debugging purposes, load a specific checkpoint instead of EMA
        state_dict = torch.load(os.path.join(args.result_dir, "checkpoints", f"{args.ckpt}.pt"), map_location=device, weights_only=True)["model"]
    else:
        # Load EMA state_dict
        state_dict = calculate_posthoc_ema(args.ema_std, os.path.join(args.result_dir, "ema"))

    model.load_state_dict(state_dict)
    model.eval()

    # Setup VAE
    if args.use_vae:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # Setup denormalization
    mean = torch.tensor(train_args["stats_mean"]).reshape(1, -1, 1, 1).to(device)
    std = torch.tensor(train_args["stats_std"]).reshape(1, -1, 1, 1).to(device)
    
    # Setup diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))

    samples_gather = []
    n = args.batch_size
    for _ in tqdm(range(math.ceil(args.num_samples / n))):
        z = torch.randn(n, train_args["in_channels"], train_args["input_size"], train_args["input_size"], device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        if args.cfg_scale > 1.0:
            # Use CFG
            z = torch.cat([z, z], dim=0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            # No CFG
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )

        if args.cfg_scale > 1.0:
            # Remove null class samples
            samples, _ = samples.chunk(2, dim=0)

        samples = samples * std + mean
        if args.use_vae:
            samples = vae.decode(samples).sample

        # Convert to numpy
        samples = samples.clamp(-1, 1)
        samples = (255 * (samples + 1) / 2).byte()
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()

        samples_gather.append(samples)

    samples = np.concatenate(samples_gather, axis=0)
    samples = samples[:args.num_samples]

    os.makedirs(os.path.join(args.result_dir, "fid_samples"), exist_ok=True)
    np.savez(os.path.join(args.result_dir, "fid_samples", args.output_file), arr_0=samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--use-vae", type=bool, default=True)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-classes", type=int, default=1_000)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default="samples.npz", help="Filename in which to store samples.")

    parser.add_argument("--ema-std", type=float, default=0.05)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load instead of EMA (should not include .pt extension).")

    args = parser.parse_args()
    main(args)
