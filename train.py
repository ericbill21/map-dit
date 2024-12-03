import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import yaml
import argparse
import logging
import os
import math

from src.models import DIT_MODELS
from diffusion import create_diffusion

from torch.optim.lr_scheduler import LRScheduler

class CustomScheduler(LRScheduler):
    """ Custom learning rate scheduler that combines a linear warmup with a square root decay

    Args:
        base_lr: a float representing the initial learning rate
        batch_size: an integer representing the batch size
        warmup_images: an integer representing the ending number of images to use for linear warmup
        decay_batches: an integer representing the starting number of batches to use for square root decay

    """
    def __init__(self, 
        optimizer: torch.optim.Optimizer, 
        base_lr: float,
        batch_size: int,
        warmup_images: int=1e6, 
        decay_images: int=10e6):

        self.base_lr = base_lr
        self.batch_size = batch_size
        self.warmup_images = warmup_images
        self.decay_images = decay_images

        super(CustomScheduler, self).__init__(optimizer, last_epoch=-1, verbose="deprecated")
        self._step_count = 1

    def get_lr(self):
        res = self.base_lr

        # Results in linear rampup from 0 to 1 over the first 'rampup_Mimg' million images
        if self.warmup_images > 0:
            res *= min(self._step_count / self.warmup_images, 1)

        # Result in a delayed decay of the learning rate
        if self.decay_images > 0:
            res /= math.sqrt(max(self._step_count / self.decay_images, 1))

        # Update _step_count
        self._step_count += self.batch_size

        self.last_lr = [res for _ in self.optimizer.param_groups]
        return self.last_lr


torch.set_float32_matmul_precision("high")


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup experiment directory
    exp_dir = setup_experiment(args.model, args.results_dir)
    logger = create_logger(exp_dir, verbose=args.verbose)
    logger.info(f"using device {device}")

    logger.info(f"experiment directory created at {exp_dir}")

    # Save arguments
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Setup data
    dataset = CustomDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    logger.info(f"dataset contains {len(dataset):,} data points ({args.data_path}, {dataset.channels}x{dataset.data_size}x{dataset.data_size})")

    # Setup diffusion process
    diffusion = create_diffusion(timestep_respacing="")

    model = DIT_MODELS[args.model](
        input_size=dataset.data_size,
        num_classes=args.num_classes,
        use_cosine_attention=args.use_cosine_attention,
        use_mp_attention=args.use_mp_attention,
        use_weight_normalization=args.use_weight_normalization,
        use_forced_weight_normalization=args.use_forced_weight_normalization,
        use_mp_residual=args.use_mp_residual,
        use_mp_silu=args.use_mp_silu,
        use_no_layernorm=args.use_no_layernorm,
        use_mp_pos_enc=args.use_mp_pos_enc,
    ).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True, disable=args.disable_compile)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = CustomScheduler(opt, base_lr=1e-4, batch_size=args.batch_size, warmup_images=args.warmup_images, decay_images=args.decay_images)

    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Exponential moving average
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    # Important! (This enables embedding dropout for CFG)
    model.train()
    ema.eval()

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
            update_ema(ema, model)

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
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                }
                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"{train_steps:07d}.pt")
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"saved checkpoint to {checkpoint_path}")
    
    logger.info("done!")


class CustomDataset(Dataset):
    def __init__(self, data_path: str):
        self.features = torch.load(os.path.join(data_path, "features.pt"), weights_only=True)
        self.labels = torch.load(os.path.join(data_path, "labels.pt"), weights_only=True)

        if self.features.dtype == torch.uint8:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            self.transform = None

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
        feature = self.features[idx]

        if self.transform is not None:
            feature = self.transform(feature)

        return feature, self.labels[idx]


def setup_experiment(model_name, results_dir):
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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


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
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, help="0: warning, 1: info, 2: debug", choices=[0, 1, 2], default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    # Scheduler
    parser.add_argument("--warmup-images", type=int, default=1e6)
    parser.add_argument("--decay-images", type=int, default=5e6)
    parser.add_argument("--disable-compile", action="store_true")

    # Flags
    parser.add_argument("--use-cosine-attention", action="store_true")
    parser.add_argument("--use-mp-attention", action="store_true")
    parser.add_argument("--use-weight-normalization", action="store_true")
    parser.add_argument("--use-forced-weight-normalization", action="store_true")
    parser.add_argument("--use-mp-residual", action="store_true")
    parser.add_argument("--use-mp-silu", action="store_true")
    parser.add_argument("--use-no-layernorm", action="store_true")
    parser.add_argument("--use-mp-pos-enc", action="store_true")

    args = parser.parse_args()
    main(args)
