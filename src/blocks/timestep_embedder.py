import torch
import torch.nn as nn
import math

from src.layers.mlp import MLP


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
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
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element (may be fractional)
            dim: the dimension of the output (D)
            max_period: controls the minimum frequency of the embeddings

        Returns: positional embeddings (N, D)
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
