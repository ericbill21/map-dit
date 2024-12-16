import torch
import torch.nn as nn
import math

from src.layers.mlp import MLP
from src.basic.mp_embedding import MPEmbedding


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
        use_mp_embedding: bool,
    ):
        super().__init__()

        self.mlp = MLP(
            hidden_size,
            hidden_size,
            hidden_dim=hidden_size,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu,
        )

        if use_mp_embedding:
            self.embedding = MPEmbedding(1000, hidden_size, use_wn, use_forced_wn)
        else:
            self.embedding = SinusoidalEncoding(hidden_size)

    def forward(self, t):
        return self.mlp(self.embedding(t))


class SinusoidalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_period: float=10000.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be an even number"

        self.register_buffer(
            "div_term",
            torch.exp(-math.log(max_period) * torch.arange(0, hidden_dim, 2, dtype=torch.float) / hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: (...B)
        
        Returns: (...B, hidden_dim)
        """

        pos_div = pos.float().unsqueeze(-1) * self.div_term
        return torch.cat([torch.cos(pos_div), torch.sin(pos_div)], dim=-1)
