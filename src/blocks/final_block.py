import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.layers.rotation_modulation import RotationModulation


class FinalBlock(nn.Module):
    def __init__(self, hidden_dim: int, rotation_dim: int, patch_size: int, out_channels: int):
        super().__init__()

        self.linear = MPLinear(hidden_dim, patch_size * patch_size * out_channels)
        self.mod = RotationModulation(hidden_dim, rotation_dim, hidden_dim)

    def forward(self, x, c):
        return self.linear(self.mod(x, c))
