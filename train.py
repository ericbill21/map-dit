import argparse
import math
import os
from glob import glob
from time import time

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusion import create_diffusion
from src.models import DIT_MODELS
from utils import create_logger, get_model

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup experiment directory
    exp_dir = setup_experiment(args.model, args.results_dir)
    logger = create_logger(exp_dir, verbose=args.verbose)
    logger.info(f"using device {device}")
    logger.info(f"experiment directory created at {exp_dir}")

    # Setup data
    dataset = CustomDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    logger.info(f"dataset contains {len(dataset):,} data points ({args.data_path}, {dataset.channels}x{dataset.data_size}x{dataset.data_size})")

    # Save arguments
    args.in_channels = dataset.channels
    args.input_size = dataset.data_size
    args.stats_std = [float(x) for x in dataset.stats["std"]]
    args.stats_mean = [float(x) for x in dataset.stats["mean"]]
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Setup diffusion process
    diffusion = create_diffusion(timestep_respacing="")

    model = get_model(args).to(device)
    model = torch.compile(model)
    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # Setup learning rate scheduler 
    if args.num_lin_warmup is None:
        args.num_lin_warmup = args.num_steps // 150

    if args.start_decay is None:
        args.start_decay = args.num_steps // 10

    scheduler = LambdaLR(opt, create_lr_lambda(args.num_lin_warmup, args.start_decay))

    # Important! (This enables embedding dropout for CFG)
    model.train()

    # Variables for monitoring/logging purposes
    train_steps = 0
    epochs = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    logger.info(f"training for {args.num_steps} steps...")

    while train_steps < args.num_steps:
        logger.info(f"beginning epoch {epochs}...")

        for x, y in loader:
            # Push data to GPU
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # Compute loss
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            # Update weights
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            opt.step()

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Update EMA
            scheduler.step()

            if train_steps % args.log_every == 0:
                # Measure training speed
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Compute average loss
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(f"(step={train_steps:07d}) train loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                logger.debug(f"(memory) current={bytes_to_gb(torch.cuda.memory_allocated()):.2f}GB, max={bytes_to_gb(torch.cuda.max_memory_allocated()):.2f}GB")

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                }

                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{train_steps:07d}.pt")
                logger.info(f"saving checkpoint to {checkpoint_path} at step {train_steps}...")
                torch.save(checkpoint, checkpoint_path)

        epochs += 1
    
    logger.info("done!")


class CustomDataset(Dataset):
    def __init__(self, data_path: str):
        self.posterior_means = torch.load(os.path.join(data_path, "posterior_means.pt"), weights_only=True)
        self.posterior_stds = torch.load(os.path.join(data_path, "posterior_stds.pt"), weights_only=True)
        self.labels = torch.load(os.path.join(data_path, "labels.pt"), weights_only=True)
        self.stats = torch.load(os.path.join(data_path, "stats.pt"), weights_only=True)

        mean = self.stats["mean"]
        std = self.stats["std"]
        self.transform = transforms.Normalize(mean, std)

        assert self.posterior_means.shape[0] == self.labels.shape[0] == self.posterior_stds.shape[0]

    @property
    def data_size(self):
        return self.posterior_means.shape[2]

    @property
    def channels(self):
        return self.posterior_means.shape[1]

    def __len__(self):
        return self.posterior_means.shape[0]

    def __getitem__(self, idx):
        mean = self.posterior_means[idx]
        std = self.posterior_stds[idx]

        # Sample from latent distribution
        eps = torch.randn_like(mean)
        feature = mean + eps * std

        return self.transform(feature), self.labels[idx]


def create_lr_lambda(num_lin_warmup: int, start_decay: int):
    """Create a function that returns the learning rate at a given step for the scheduler.

    Args:
        num_lin_warmup: number of steps for linear warmup
        start_decay: step to start decaying the learning rate

    """

    def lr_lambda(step):
        if step + 1 < num_lin_warmup:
            return (step + 1) / num_lin_warmup

        if step >= start_decay:
            return 1.0 / math.sqrt(max(step / start_decay, 1))

        return 1.0

    return lr_lambda


def setup_experiment(model_name: str, results_dir: os.PathLike):
    """Create an experiment directory for the current run."""

    # Make results directory
    os.makedirs(results_dir, exist_ok=True)

    experiment_index = len(glob(os.path.join(results_dir, "*")))
    model_string_name = model_name.replace("/", "-")
    experiment_dir = os.path.join(results_dir, f"{experiment_index:03d}-{model_string_name}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Make experiment directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir


def bytes_to_gb(n):
    return n * 1e-9


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()

    # Training loop
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DIT_MODELS.keys()), default="DiT-S/4")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, help="0: warning, 1: info, 2: debug", choices=[0, 1, 2], default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    # Learning rate scheduler
    parser.add_argument("--num-lin-warmup", type=int, default=None, help="Number of steps for linear warmup of the learning rate")
    parser.add_argument("--start-decay", type=int, default=None, help="Step to start decaying the learning rate")
    parser.add_argument("--attn-scale", type=float, default=0.3, help="Attention residual")

    args = parser.parse_args()
    main(args)