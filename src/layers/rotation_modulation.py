from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU

    
class theta_net(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            MPSiLU(),
            nn.Linear(in_dim, in_dim//2, bias=False),
        )
        nn.init.zeros_(self.net[1].weight)
    
    def forward(self, x):
        return self.net(x)
    

class shift_net(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            MPSiLU(),
            MPLinear(in_dim, in_dim)
        )

        self.blend = nn.Sequential(
            MPSiLU(),
            nn.Linear(in_dim, 1, bias=False),
        )
        nn.init.zeros_(self.blend[1].weight)

    def forward(self, x):
        return self.net(x), F.sigmoid(self.blend(x) - 2.1972)
    

class RotationModulation(nn.Module):
    """Applies a condition-dependent rotation in a 2-dimensional subspace."""
    def __init__(self, 
        dim: int,
        with_gate: bool=True
    ):
        """
        n (int): Dimensionality of conditioning space
        """
        super().__init__()

        self.scale_net = theta_net(dim)
        self.shift_net = shift_net(dim)
        self.gate_net = theta_net(dim) if with_gate else nn.Identity()
    
    def forward(self, cond):
        """
        Inputs:
            cond (torch.Tensor): Shape (batch_size, cond_dim)
        Output:
            scale (torch.Tensor): Shape (batch_size, cond_dim//2)
            shift (torch.Tensor, torch.Tensor): Shape (batch_size, cond_dim), Shape (batch_size, 1)
            gate (torch.Tensor): Shape (batch_size, cond_dim//2)
        """
        return self.scale_net(cond), self.shift_net(cond), self.gate_net(cond)
