import torch
import torch.nn as nn

from src.basic.mp_silu import MPSiLU
from src.basic.mp_linear import MPLinear, MPLinearChunk
from src.utils import modulate


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
        self.gain_sigma = nn.Parameter(torch.tensor(0.0))

        self.modulation = nn.Sequential(
            MPSiLU(),
            MPLinearChunk(hidden_size, hidden_size, 2),
        )
        self.gain_mod = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, c):
        shift, scale = self.modulation(c)
        x_mod = modulate(x, self.gain_mod * shift, scale)

        if self.learn_sigma:
            mean, sigma = self.linear(x_mod)
            return mean, self.gain_sigma * sigma
        else:
            return self.linear(x_mod)