import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.basic.mp_linear import MPLinear, MPLinearChunk
from src.basic.mp_silu import MPSiLU
from src.utils import chunk_normalize

class AdaLNModulation(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_modulates: int=1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            MPSiLU(),
            MPLinearChunk(hidden_dim, hidden_dim, num_modulates),
        )
        
    def forward(self, x):
        # Forced weight normalization
        if self.training:
            with torch.no_grad():
                self.weight.data.copy_(chunk_normalize(self.weight, self.num_modulates))
        
        # Traditional weight normalization (makes sure that the gradient is perpendicular to the
        # weights)
        w = chunk_normalize(self.weight, self.num_modulates) / math.sqrt(self.in_dim)

        out = F.linear(self.act_fn(x), w)

        if self.num_shifts > 0:
            out[..., :self.num_shifts * self.in_dim] *= self.gain.repeat_interleave(self.in_dim)
        
        return out.chunk(self.num_modulates, dim=-1)
