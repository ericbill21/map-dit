import torch
import torch.nn as nn
import math

from src.layers.mlp import MLP
from src.basic.mp_fourier import Fourier, MPFourier


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
        use_mp_fourier: bool,
        frequency_embedding_size: int=256,
    ):
        super().__init__()

        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_dim=hidden_size,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu,
        )

        if use_mp_fourier:
            self.emb_fourier = MPFourier(frequency_embedding_size)
        else:
            self.emb_fourier = Fourier(frequency_embedding_size)

    def forward(self, t):
        t_freq = self.emb_fourier(t)
        t_emb = self.mlp(t_freq)        
        return t_emb
