import torch
import torch.nn as nn

from src.utils import mp_sum, rotate_2d
from src.basic.mp_linear import MPLinear
from src.layers.mlp import MLP
from src.layers.rotation_modulation import RotationModulation


class FinalBlock(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()

        self.linear = MPLinear(hidden_dim, patch_size * patch_size * out_channels)
        self.modulation = RotationModulation(hidden_dim, with_gate=False)

    def forward(self, x, c):
        scale, shift, _ = self.modulation(c)

        x_mod = mp_sum(rotate_2d(x, scale), shift.unsqueeze(1), t=0.3)
        return self.linear(x_mod)