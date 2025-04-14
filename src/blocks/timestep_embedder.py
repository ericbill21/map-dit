import torch
import torch.nn as nn
import math

from src.layers.mlp import MLP
from src.basic.mp_embedding import MPEmbedding

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        scale = 2 * torch.pi * torch.randn(num_channels) 
        shift = 2 * torch.pi * torch.rand(num_channels)

        self.register_buffer("scale", scale.to(torch.float32))
        self.register_buffer("shift", shift.to(torch.float32))

    def forward(self, x):
        # cos(2 * \pi * (freqs * x + phases))
        res = torch.cos(torch.outer(x, self.scale) + self.shift)
        return math.sqrt(2) * res.to(torch.float32)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int=256,
    ):
        super().__init__()

        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_dim=hidden_size,
        )

        self.embedding = MPFourier(frequency_embedding_size)

    def forward(self, t):
        return self.mlp(self.embedding(t))