import torch
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
        use_wn: bool,
        use_forced_wn: bool,
    ):
        super().__init__()

        assert in_dim % num_heads == 0

        self.use_cosine = use_cosine_attention
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.q_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.k_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.v_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.out_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def qkv_normalize(self, w: torch.Tensor, eps=1e-4) -> torch.Tensor:
        # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
        # multiply by sqrt(in_dim) to compensate
        # Additionally, we normalize according to query, key, or value, hence the view
        w_view = w.view(3, self.in_dim, self.in_dim)
        norm = torch.linalg.vector_norm(w_view, dim=-1, keepdim=True)
        w = w_view * math.sqrt(self.in_dim) / (norm + eps)
        return w.reshape(3 * self.in_dim, self.in_dim)

    def forward(self, x):
        """
        Args:
            x: (...B, T, D)

        Returns: (...B, T, D)
        """

        T = x.shape[-2]

        q = self.q_proj(x)                                                  # (...B, T, D)
        k = self.k_proj(x)                                                  # (...B, T, D)
        v = self.v_proj(x)                                                  # (...B, T, D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        if self.use_cosine:
            q = normalize(q)
            k = normalize(k)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)     # (...B, H, T, D')
        out = out.transpose(-3, -2)                                         # (...B, T, H, D')
        out = out.reshape(*x.shape)                                         # (...B, T, D)

        return self.out_proj(out)
