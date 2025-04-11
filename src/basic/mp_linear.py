import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import normalize


class MPLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.normal_(self.weight)

    def forward(self, x):
        """
        Args:
            x: (...B, in_dim)
        
        Returns: (...B, out_dim)
        """

        # Forced weight normalization (makes sure that the weight magnitude does not grow out of
        # control)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))

        # Traditional weight normalization (makes sure that the gradient is perpendicular to the
        # weights)
        w = normalize(self.weight) / math.sqrt(self.in_dim)

        return F.linear(x, w)
