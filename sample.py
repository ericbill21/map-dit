import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from models import DiT_models
import argparse


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
    ).to(device)

    state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict["model"])
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model on
    class_labels = [416] * 64

    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, 3, args.image_size, args.image_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup CFG
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images
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

    # Save and display images
    save_image(samples, "sample.png", nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XS/2")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128, 256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
