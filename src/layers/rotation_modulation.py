from typing import Optional

import torch
import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU


class RotationModulation(nn.Module):
    """Applies a condition-dependent rotation in a 2-dimensional subspace."""
    def __init__(self, 
        dim: int,
        mlp_ratio: float=4.0,
        with_gate: bool=True
    ):
        """
        n (int): Dimensionality of conditioning space
        """
        super().__init__()

        self.scale_net = nn.Sequential(
            MPSiLU(),
            MPLinear(dim, dim//2)
        )

        self.shift_net = nn.Sequential(
            MPSiLU(),
            MPLinear(dim, dim)
        )

        if with_gate:
            self.gate_net = nn.Sequential(
                MPSiLU(),
                MPLinear(dim, dim//2)
            )
        else:
            self.gate_net = nn.Identity()
    
    def forward(self, cond):
        """
        Inputs:
            cond (torch.Tensor): Shape (batch_size, cond_dim)
        Output:
            scale (torch.Tensor): Shape (batch_size, cond_dim//2)
            shift (torch.Tensor): Shape (batch_size, cond_dim)
            gate (torch.Tensor): Shape (batch_size, cond_dim//2)
        """

        return self.scale_net(cond), self.shift_net(cond), self.gate_net(cond)
