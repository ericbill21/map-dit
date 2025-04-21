import torch
import torch.nn as nn

from src.layers.mlp import MLP


class Fourier(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.register_buffer("scale", 2 * torch.pi * torch.randn(num_channels, dtype=torch.float32))
        self.register_buffer("shift", 2 * torch.pi * torch.rand(num_channels, dtype=torch.float32))

    def forward(self, x):
        # cos(2 * \pi * (freqs * x + phases))
        return torch.cos(torch.outer(x.float(), self.scale) + self.shift)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size=256,
    ):
        super().__init__()

        self.embedding = Fourier(frequency_embedding_size)
        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_dim=hidden_size,
        )

    def forward(self, t):
        return self.mlp(self.embedding(t))
