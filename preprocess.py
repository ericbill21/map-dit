import argparse
import torch

from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset
from diffusers.models import AutoencoderKL

import os

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}...")

    print(f"Loading data...")
    ds = load_dataset("benjamin-paine/imagenet-1k-128x128", cache_dir=args.hf_cache)["train"]

    print(f"Loading model...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=args.hf_cache).to(device)

    torch.set_grad_enabled(False)

    transform = transforms.Compose([
                    transforms.functional.pil_to_tensor,
                    transforms.ConvertImageDtype(torch.float32),
                ])


    total_len = len(ds)
    latents, labels = [], []
    for idx in tqdm(range(0, total_len, args.batch_size)):
        tail = min(idx+args.batch_size, total_len)

        imgs = []
        for img in ds[idx:tail]["image"]:
            imgs.append(transform(img))
        
        imgs = torch.stack(imgs, dim=0).to(device)
        latent = vae.encode(imgs).latent_dist.sample()
        latents.append(latent.cpu())
        
        labels += ds[idx:tail]["label"]

    print("Concatenating data...")
    latents = torch.cat(latents, dim=0)
    labels = torch.tensor(labels)
    stats = { "mean": latents.mean(dim=[0, 2, 3]), "std": latents.std(dim=[0, 2, 3]) }

    # Save data and labels
    print(f"Saving data to \"{args.output_dir}\"...")
    torch.save(latents, os.path.join(args.output_dir, "features.pt"))
    torch.save(labels,  os.path.join(args.output_dir, "labels.pt"))
    torch.save(stats,   os.path.join(args.output_dir, "stats.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", type=str, required=True, help="Path to directory to save features.pt and labels.pt")
    parser.add_argument("--hf-cache", type=str, default=None, help="Path to directory to save HuggingFace datasets")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to use for encoding images")

    args = parser.parse_args()
    main(args)
