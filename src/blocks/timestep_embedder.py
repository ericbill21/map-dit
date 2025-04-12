import math

import torch
import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU

import torch
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

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            MPLinear(frequency_embedding_size, hidden_size),
            MPSiLU(),
            MPLinear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.embedding = MPFourier(frequency_embedding_size)
        # self.embedding = MPEmbedding(1000, hidden_size)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.embedding(t)
        return self.mlp(t_freq)