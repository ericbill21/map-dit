import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.basic.mp_silu import MPSiLU
from src.basic.mp_linear import MPLinear, MPLinearChunk
from src.utils import modulate, normalize


class MPScale(nn.Module):
    def __init__(self, in_dim: int, angle_dim: int=8, zero_init: bool = True):
        super().__init__()
        self.angle_dim = angle_dim

        self.linear = MPLinear(in_dim, angle_dim)
        self.reference = nn.Parameter(torch.zeros(angle_dim) if zero_init else torch.ones(angle_dim))

    def forward(self, x):
        angle =  torch.matmul(self.linear(x), self.reference) / math.sqrt(self.angle_dim)
        return F.sigmoid(angle)

class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        learn_sigma: bool = True,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.linear = MPLinearChunk(
            hidden_size,
            patch_size * patch_size * out_channels,
            2 if learn_sigma else 1,
        )

        self.modulation = nn.Sequential(
            MPSiLU(),
            MPLinearChunk(hidden_size, hidden_size, 2),
        )
        self.gain_mod = nn.Parameter(torch.tensor(0.0))

        # Allowing the model to learn the scale of the mean and sigma
        self.mean_scale = MPScale(hidden_size, zero_init=False)
        if learn_sigma: self.sigma_scale = MPScale(hidden_size, zero_init=True)

    def forward(self, x, c):
        shift, scale = self.modulation(c)
        x_mod = modulate(x, shift, scale, t=self.gain_mod)

        if self.learn_sigma:
            mean, sigma = self.linear(x_mod)
            return mean * self.mean_scale(c).view(-1, 1, 1), sigma * self.sigma_scale(c).view(-1, 1, 1)
        else:
            return self.linear(x_mod) * self.mean_scale(c)