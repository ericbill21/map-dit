import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusion import create_diffusion
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import yaml
import argparse
import logging
import os
import math
from ema import EMA

from src.models import DIT_MODELS
from utils import get_model

from torch.optim.lr_scheduler import LambdaLR


torch.set_float32_matmul_precision("high")


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
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Setup diffusion process
    diffusion = create_diffusion(timestep_respacing="")

    model = get_model(args).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True, disable=args.disable_compile)
    ema = EMA(model, results_dir=exp_dir, stds=[0.05, 0.1])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = LambdaLR(opt, create_lr_lambda(args.num_lin_warmup, args.start_decay))

    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Important! (This enables embedding dropout for CFG)
    model.train()

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    logger.info(f"training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        logger.info(f"beginning epoch {epoch}...")

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            scheduler.step()
            ema.update(t=train_steps, t_delta=1)

            # Log loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
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
                torch.save(checkpoint, checkpoint_path)
      
                logger.info(f"saved checkpoint to {checkpoint_path} at step {train_steps:07d}")
            
            # Save EMA snapshot
            if train_steps % args.ema_snapshot_every == 0 and train_steps > 0:
                ema.save_snapshot(train_steps)
                logger.info(f"saved EMA snapshot to {ema.results_dir} at step {train_steps:07d}")
    
    logger.info("done!")


class CustomDataset(Dataset):
    def __init__(self, data_path: str):
        self.features = torch.load(os.path.join(data_path, "features.pt"), weights_only=True)
        self.labels = torch.load(os.path.join(data_path, "labels.pt"), weights_only=True)
        self.stats = torch.load(os.path.join(data_path, "stats.pt"), weights_only=True)

        if self.features.dtype == torch.uint8:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(self.stats["mean"], self.stats["std"], inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize(self.stats["mean"], self.stats["std"], inplace=True)
            ])

        assert self.features.shape[0] == self.labels.shape[0]
        assert self.features.shape[2] == self.features.shape[3]

    @property
    def data_size(self):
        return self.features.shape[2]

    @property
    def channels(self):
        return self.features.shape[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.features[idx]), self.labels[idx]


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


def create_logger(logging_dir, verbose: int=1):
    verbose_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    logging.basicConfig(
        level=verbose_map.get(verbose, logging.INFO),
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    logger = logging.getLogger(__name__)
    return logger


def bytes_to_gb(n):
    return n * 1e-9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Standard
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DIT_MODELS.keys()), default="DiT-XS/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, help="0: warning, 1: info, 2: debug", choices=[0, 1, 2], default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--disable-compile", action="store_true")

    # Scheduler
    parser.add_argument("--num-lin-warmup", type=int, default=700, help="Number of steps for linear warmup of the learning rate")
    parser.add_argument("--start-decay", type=int, default=10_000, help="Step to start decaying the learning rate")

    # EMA
    parser.add_argument("--ema-snapshot-every", type=int, default=1_600, help="Number of steps to save EMA snapshots")

    # Flags
    parser.add_argument("--use-cosine-attention", action="store_true")
    parser.add_argument("--use-weight-normalization", action="store_true")
    parser.add_argument("--use-forced-weight-normalization", action="store_true")
    parser.add_argument("--use-mp-residual", action="store_true")
    parser.add_argument("--use-mp-silu", action="store_true")
    parser.add_argument("--use-no-layernorm", action="store_true")
    parser.add_argument("--use-mp-pos-enc", action="store_true")
    parser.add_argument("--use-fourier", action="store_true")
    parser.add_argument("--use-mp-fourier", action="store_true")

    args = parser.parse_args()
    main(args)
