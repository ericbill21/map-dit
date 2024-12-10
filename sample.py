import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL
import argparse
import yaml
import os

from src.ema import calculate_posthoc_ema
from utils import get_model, CLS_LOC_MAPPING
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
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True, disable=train_args["disable_compile"])

    if args.ckpt is not None:
        # For debugging purposes, load a specific checkpoint instead of EMA
        state_dict = torch.load(os.path.join(args.result_dir, "checkpoints", f"{args.ckpt}.pt"), map_location=device, weights_only=True)
        model.load_state_dict(state_dict["model"])
    else:
        # Load EMA state_dict
        ema_state_dict = calculate_posthoc_ema(args.ema_std, os.path.join(args.result_dir, "ema"))
        model.load_state_dict(ema_state_dict)

    model.eval()

    # Labels to condition the model on
    class_labels = [args.class_label] * 64

    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, 4, train_args["input_size"], train_args["input_size"], device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup CFG
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images
    diffusion = create_diffusion(str(args.num_sampling_steps))
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    # Remove null class samples
    samples, _ = samples.chunk(2, dim=0)

    # Denormalize samples
    stats = torch.load(os.path.join(train_args["data_path"], "stats.pt"), weights_only=True)
    mean = stats["mean"][None, :, None, None].to(device)
    std = stats["std"][None, :, None, None].to(device)
    samples = samples * std + mean

    # Load VAE
    if args.use_vae:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        samples = vae.decode(samples).sample.cpu()

    samples = samples.clamp(-1, 1)

    # Save and display images
    save_image(samples, args.output_file, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"output class: {CLS_LOC_MAPPING[args.class_label]} ({args.class_label})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--use-vae", type=bool, default=True)
    parser.add_argument("--output-file", type=str, default="sample.png")
    parser.add_argument("--class-label", type=int, default=88)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ema-std", type=float, default=0.05)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to load instead of EMA (should not include .pt extension).")

    args = parser.parse_args()
    main(args)
