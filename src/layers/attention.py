import torch.nn as nn
import torch.nn.functional as F
import math

from src.basic.mp_linear import MPLinear
from src.utils import normalize


class Attention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        use_cosine_attention: bool,
        use_mp_attention: bool,
        use_wn: bool,
        use_forced_wn: bool,
    ):
        super().__init__()

        assert in_dim % num_heads == 0

        self.use_cosine = use_cosine_attention
        self.use_mp_attention = use_mp_attention
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.qkv = MPLinear(in_dim, 3 * in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.out_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (...B, T, D)

        Returns: (...B, T, D)
        """

        T = x.shape[-2]

        q, k, v = self.qkv(x).chunk(3, dim=-1)                              # (...B, T, D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        if self.use_cosine:
            q = normalize(q)
            k = normalize(k)

        attn = q @ k.transpose(-1, -2) * self.scale          # (...B, H, T, T)
        attn = F.softmax(attn, dim=-1)                       # (...B, H, T, T)
        out = attn @ v                                       # (...B, H, T, D')

        out = out.transpose(-3, -2)                                         # (...B, T, H, D')
        out = out.reshape(*x.shape)                                         # (...B, T, D)

        return self.out_proj(out)
