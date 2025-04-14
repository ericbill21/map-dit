import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import normalize, chunk_normalize

class MPLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        zero_init: bool=False,
        learn_gain: bool=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.normal_(self.weight)

        if learn_gain:
            self.gain = nn.Parameter(torch.tensor(0. if zero_init else 1.))
        else:
            self.gain = 1.

    def forward(self, x):
        """
        Args:
            x: (...B, in_dim)
        
        Returns: (...B, out_dim)
        """
        # Forced weight normalization
        if self.training:
            with torch.no_grad():
                self.weight.data.copy_(normalize(self.weight))
        
        # Traditional weight normalization (makes sure that the gradient is perpendicular to the
        # weights)
        w = normalize(self.weight) * (self.gain / math.sqrt(self.in_dim))

        return F.linear(x, w)

class MPLinearChunk(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_chunks: int
    ):
        super().__init__()

        self.in_dim = in_dim
        self.n_chunks = n_chunks
        self.weight = nn.Parameter(torch.empty(n_chunks * out_dim, in_dim))
        nn.init.normal_(self.weight)
        
    def forward(self, x):
        # Forced weight normalization
        if self.training:
            with torch.no_grad():
                self.weight.data.copy_(chunk_normalize(self.weight, self.n_chunks))
        
        # Traditional weight normalization (makes sure that the gradient is perpendicular to the
        # weights)
        w = chunk_normalize(self.weight, self.n_chunks) / math.sqrt(self.in_dim)
        return F.linear(x, w).chunk(self.n_chunks, dim=-1)