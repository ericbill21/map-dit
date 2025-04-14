import torch
import einops
import math


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """Computes the mean magnitude."""
    return x.square().mean(-1).sqrt().mean()


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return mp_sum(x * (1 + scale.unsqueeze(1)), shift.unsqueeze(1), t=0.5)


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float=0.5) -> torch.Tensor:
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def normalize(x: torch.Tensor, eps=1e-4) -> torch.Tensor:
    # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
    # multiply by sqrt(in_dim) to compensate
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * math.sqrt(x.shape[-1]) / (norm + eps)

def chunk_normalize(w: torch.Tensor, n: int, eps=1e-4) -> torch.Tensor:
    # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
    # multiply by sqrt(in_dim) to compensate
    # Additionally, we normalize to each chunk, hence the view
    out_dim, in_dim = w.shape
    w_view = w.view(n, out_dim//n, in_dim)
    norm = torch.linalg.vector_norm(w_view, dim=-1, keepdim=True)
    w = w_view * math.sqrt(in_dim) / (norm + eps)
    return w.reshape(out_dim, in_dim)


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
