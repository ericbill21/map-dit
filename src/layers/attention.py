import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.basic.mp_linear import MPLinear
from src.utils import normalize


def qkv_normalize(x: torch.Tensor, eps=1e-4) -> torch.Tensor:
    x_view = x.view(3, -1, x.shape[-1])
    x_view = normalize(x_view, eps=eps)
    return x_view.view(-1, x.shape[-1])

class Attention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int):
        super().__init__()

        assert in_dim % num_heads == 0

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_weight = nn.Parameter(torch.randn(3 * in_dim, in_dim))
        self.out_proj = MPLinear(in_dim, in_dim)

    def forward(self, x):
        """
        Args:
            x: (...B, T, D)

        Returns: (...B, T, D)
        """

        T = x.shape[-2]

        if self.training:
            with torch.no_grad():
                self.qkv_weight.copy_(qkv_normalize(self.qkv_weight))

        w = qkv_normalize(self.qkv_weight) / math.sqrt(self.in_dim)
        q, k, v = F.linear(x, w).chunk(3, dim=-1)                           # 3 * (...B, T, D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)     # (...B, H, T, D')
        out = out.transpose(-3, -2)                                         # (...B, T, H, D')
        out = out.reshape(*x.shape)                                         # (...B, T, D)
        
        # Since attention can only decrease the magnitude, we renormalize the output to unit magnitude
        return normalize(self.out_proj(out))
