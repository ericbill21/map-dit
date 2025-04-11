from typing import Optional

import torch
import torch.nn as nn

from src.layers.mlp import MLP
from src.layers.rotation_modulation import RotationModulation
from src.layers.sigmoid_attention import SigmoidAttention
from src.utils import mp_sum


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotation_dim: int,
        cond_dim: Optional[int]=None,
        mlp_ratio: float=4.0,
    ):
        super().__init__()

        if cond_dim is None:
            cond_dim = hidden_size

        self.attn = SigmoidAttention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)
        self.mod1 = RotationModulation(hidden_size, rotation_dim, cond_dim)
        self.mod2 = RotationModulation(hidden_size, rotation_dim, cond_dim)

        self.mp_attn = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mp_mlp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x, c):
        x = mp_sum(x, self.attn(self.mod1(x, c)), t=self.mp_attn.sigmoid())
        x = mp_sum(x, self.mlp(self.mod2(x, c)), t=self.mp_mlp.sigmoid())
        return x
