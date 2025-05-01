import torch.nn as nn
import torch.nn.functional as F
import math

from src.basic.mp_linear import MPLinear, MPLinearChunk
from src.utils import normalize


class Attention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int):
        super().__init__()

        assert in_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.qkv_proj = MPLinearChunk(in_dim, in_dim, 3)
        self.out_proj = MPLinear(in_dim, in_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (...B, T, D)

        Returns: (...B, T, D)
        """

        T = x.shape[-2]

        q, k, v = self.qkv_proj(x)                                          # (...B, T, 3 * D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        # cosine attention
        q = normalize(q)
        k = normalize(k)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)     # (...B, H, T, D')
        out = out.transpose(-3, -2)                                         # (...B, T, H, D')
        out = out.reshape(*x.shape)                                         # (...B, T, D)

        return self.out_proj(out)
