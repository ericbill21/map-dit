import torch
import einops
import math


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float=0.5) -> torch.Tensor:
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def normalize(x: torch.Tensor, eps=1e-4) -> torch.Tensor:
    # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
    # multiply by sqrt(in_dim) to compensate
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    x_norm = torch.add(eps, x_norm, alpha=1.0 / math.sqrt(x.shape[-1]))
    return x.div(x_norm)


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """Computes the mean magnitude."""
    return x.square().mean(-1).sqrt().mean()


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, C, H, W)
        patch_size: int (P)
    
    Returns: (B, (H / P) * (W / P), C * P * P)
    """

    return einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)


def unpatchify(x: torch.Tensor, input_size: int, patch_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, (H / P) * (W / P), C * P * P)
        input_size: int (H or W)
        patch_size: int (P)
    
    Returns: (B, C, H, W)
    """

    return einops.rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=input_size // patch_size, w=input_size // patch_size, p1=patch_size, p2=patch_size)
