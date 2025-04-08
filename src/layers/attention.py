import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.basic.mp_linear import MPLinear
from src.utils import normalize, magnitude


class Attention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        use_cosine_attention: bool,
        use_wn: bool,
        use_forced_wn: bool,
        use_sigmoid_attn: bool,
        force_magnitude: bool,
    ):
        super().__init__()

        assert in_dim % num_heads == 0
        
        # attention type
        self.use_cosine = use_cosine_attention
        self.use_sigmoid_attn = use_sigmoid_attn
        self.force_magnitude = force_magnitude

        if use_sigmoid_attn and not use_cosine_attention:
            raise ValueError("Sigmoid attention requires cosine attention.")

        # attention parameters
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # attention projections
        self.in_dim = in_dim
        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.qkv_weigth = nn.Parameter(torch.empty(3 * in_dim, in_dim))
        self.out_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)

        if use_wn:
            nn.init.normal_(self.qkv_weigth)
        else:
            nn.init.kaiming_uniform_(self.qkv_weigth)

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

        T, D = x.shape[-2:]

        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.qkv_weigth.copy_(self.qkv_normalize(self.qkv_weigth))

        qkv_w = self.qkv_weigth
        if self.use_wn:
            qkv_w = self.qkv_normalize(qkv_w) / math.sqrt(self.in_dim)

        q, k, v = F.linear(x, qkv_w).chunk(3, dim=-1)                       # 3 * (...B, T, D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        if self.use_cosine:
            q = normalize(q)
            k = normalize(k)

        # (...B, H, T, D')
        if self.use_sigmoid_attn:
            out = 1.8402 / math.sqrt(T) * F.sigmoid(q @ k.transpose(-2, -1) * self.scale) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        if self.force_magnitude:
            out = magnitude(x) * normalize(out)
        
        out = out.transpose(-3, -2)                                         # (...B, T, H, D')
        out = out.reshape(*x.shape)                                         # (...B, T, D)

        return self.out_proj(out)