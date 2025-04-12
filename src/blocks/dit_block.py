from typing import Optional

import torch
import torch.nn as nn

from src.layers.mlp import MLP
from src.layers.rotation_modulation import RotationModulation
from src.layers.attention import Attention
from src.utils import mp_sum, rotate_2d


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float=4.0,
    ):
        super().__init__()

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)
        self.mod_msa = RotationModulation(hidden_size, mlp_ratio=mlp_ratio)
        self.mod_mlp = RotationModulation(hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, x, c):
        scale_msa, shift_msa, gate_msa = self.mod_msa(c)
        scale_mlp, shift_mlp, gate_mlp = self.mod_mlp(c)

        mod_msa = mp_sum(rotate_2d(x, scale_msa), shift_msa.unsqueeze(1), t=0.3)
        x = mp_sum(x, rotate_2d(self.attn(mod_msa), gate_msa), t=0.5)

        mod_mlp = mp_sum(rotate_2d(x, scale_mlp), shift_mlp.unsqueeze(1), t=0.3)
        x = mp_sum(x, rotate_2d(self.mlp(mod_mlp), gate_mlp), t=0.5)

        return x