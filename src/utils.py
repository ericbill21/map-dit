import math

import einops
import numpy as np
import torch


def magnitude(x: torch.Tensor) -> torch.Tensor:
    """Computes the mean magnitude."""
    return x.square().mean(-1).sqrt().mean()


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float=0.5) -> torch.Tensor:
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def normalize(x: torch.Tensor, eps=1e-4) -> torch.Tensor:
    # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
    # multiply by sqrt(in_dim) to compensate
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * math.sqrt(x.shape[-1]) / (norm + eps)


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


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        grid_size: int of the grid height and width

    Returns: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim) (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py"""

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim: output dimension for each position (D)
        pos: (M,)

    Returns: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
