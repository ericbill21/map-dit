import torch
from torchvision import transforms
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import argparse
import os

from utils import create_logger

def main(args):
    torch.set_grad_enabled(False)

    logger = create_logger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device {device}")

    logger.info("loading data...")
    ds = load_dataset("benjamin-paine/imagenet-1k-128x128")["train"]

    logger.info("loading vae...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # This dataset is already center cropped, so we don't need to do it again
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    total_len = len(ds)

    means = torch.zeros(total_len, 4, 16, 16)
    stds = torch.zeros(total_len, 4, 16, 16)
    labels = []

    for idx in tqdm(range(0, total_len, args.batch_size), desc="encoding images"):
        tail = min(idx+args.batch_size, total_len)

        imgs = []
        for img in ds[idx:tail]["image"]:
            imgs.append(transform(img))

        imgs = torch.stack(imgs, dim=0).to(device)
        dist = vae.encode(imgs).latent_dist
        means[idx:tail] = dist.mean.cpu()
        stds[idx:tail] = dist.std.cpu()

        labels += ds[idx:tail]["label"]

    labels = torch.tensor(labels)

    # Compute mean and std of this MoG
    logger.info("computing mean and std of MoG...")
    mean = means.mean(dim=[0, 2, 3])
    var = torch.square(stds).mean(dim=[0, 2, 3]) + torch.square(means - mean[None, :, None, None]).mean(dim=[0, 2, 3])
    std = var.sqrt()

    # Save data and labels
    logger.info(f"saving data to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(means, os.path.join(args.output_dir, "posterior_means.pt"))
    torch.save(stds, os.path.join(args.output_dir, "posterior_stds.pt"))
    torch.save(labels, os.path.join(args.output_dir, "labels.pt"))
    torch.save({ "mean": mean, "std": std }, os.path.join(args.output_dir, "stats.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="Path to directory to save dataset (posterior_means.pt, posterior_stds.pt, labels.pt)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to use for encoding images")
    args = parser.parse_args()
    main(args)