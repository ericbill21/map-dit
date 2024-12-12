import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import normalize


class MPLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_wn: bool,
        use_forced_wn: bool,
        zero_init: bool=False,
        learn_gain: bool=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))

        if use_wn and learn_gain:
            self.gain = nn.Parameter(torch.tensor(0. if zero_init else 1.))
        else:
            self.gain = 1.

        if use_wn:
            nn.init.normal_(self.weight)
        elif zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        """
        Args:
            x: (...B, in_dim)
        
        Returns: (...B, out_dim)
        """

        # Forced weight normalization
        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))

        # Traditional weight normalization (makes sure that the gradient is perpendicular to the
        # weights)
        if self.use_wn:
            w = normalize(self.weight) * (self.gain / math.sqrt(self.in_dim))
        else:
            w = self.weight

        return F.linear(x, w)
