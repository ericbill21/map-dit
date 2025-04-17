import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.basic.mp_silu import MPSiLU
from src.basic.mp_linear import MPLinear, MPLinearChunk
from src.utils import modulate, normalize


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
            nn.Linear(hidden_size, (1 + learn_sigma) * patch_size * patch_size * out_channels, bias=False),
        )

    def forward(self, x, c):
        if self.learn_sigma:
            gate_mean, gate_sigma = self.modulation(c).chunk(2, dim=-1)
            mean, sigma = self.linear(x)

            return gate_mean.unsqueeze(1) * mean, gate_sigma.unsqueeze(1) * sigma
        else:
            gate_mean = self.modulation(c)
            mean = self.linear(x)

            return gate_mean.unsqueeze(1) * mean