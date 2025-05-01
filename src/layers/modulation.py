import torch
import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU


class ScaleModulation(nn.Module):
    """Applies a condition-dependent scaling.
    
    Args:
        n (int): Dimensionality of the input space.
        d (int): Dimensionality of the condition space.
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            MPSiLU(),
            MPLinear(d, n, zero_init=True, learn_gain=True),
        )

    def forward(self, x, c):
        """
        Inputs:
            x (Tensor): Shape (b, t, n)
            c (Tensor): Shape (b, d)
        """

        scales = self.net(c)  # Shape (b, n)
        return x * (1 + scales.unsqueeze(1))


class ShiftModulation(nn.Module):
    """Applies a condition-dependent shifting.
    
    Args:
        n (int): Dimensionality of the input space.
        d (int): Dimensionality of the condition space.
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            MPSiLU(),
            MPLinear(d, n, zero_init=True, learn_gain=True),
        )

    def forward(self, x, c):
        """
        Inputs:
            x (Tensor): Shape (b, t, n)
            c (Tensor): Shape (b, d)
        """

        shifts = self.net(c)  # Shape (b, n)
        return x + shifts.unsqueeze(1)


class RotationModulation(nn.Module):
    """Applies a condition-dependent rotation.
    
    Args:
        n (int): Dimensionality of the input space.
        d (int): Dimensionality of the condition space.
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            MPSiLU(),
            MPLinear(d, n // 2, zero_init=True, learn_gain=True),
        )

    def forward(self, x, c):
        """
        Inputs:
            x (Tensor): Shape (b, t, n)
            c (Tensor): Shape (b, d)
        """

        d = x.shape[-1]
        rotation_angles = self.net(c)  # Shape (b, n/2)

        cos = torch.cos(rotation_angles).unsqueeze(1)  # Shape (b, 1, n/2)
        sin = torch.sin(rotation_angles).unsqueeze(1)  # Shape (b, 1, n/2)

        x_rot = torch.zeros_like(x)
        x_rot[..., :d//2] = x[..., :d//2] * cos - x[..., d//2:] * sin
        x_rot[..., d//2:] = x[..., :d//2] * sin + x[..., d//2:] * cos

        return x_rot


class ModulationSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x, c):
        for mod in self.mods:
            x = mod(x, c)
        return x
