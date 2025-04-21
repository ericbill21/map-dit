import torch
import torch.nn as nn

from src.layers.mlp import MLP


class RotationModulation(nn.Module):
    """Applies a condition-dependent rotation.
    
    Args:
        n (int): Dimensionality of the input space.
        d (int): Dimensionality of the condition space.
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        self.rotation_mlp = MLP(d, n // 2, hidden_dim=n)
        nn.init.zeros_(self.rotation_mlp.net[-1].weight)
        nn.init.zeros_(self.rotation_mlp.net[-1].bias)

    def forward(self, x, c):
        """
        Inputs:
            x (Tensor): Shape (b, t, n)
            c (Tensor): Shape (b, d)
        """

        d = x.shape[-1]
        rotation_angles = self.rotation_mlp(c)  # Shape (b, n/2)

        cos = torch.cos(rotation_angles).unsqueeze(1)  # Shape (b, n/2, 1)
        sin = torch.sin(rotation_angles).unsqueeze(1)  # Shape (b, n/2, 1)

        x_rot = torch.zeros_like(x)
        x_rot[..., :d//2] = x[..., :d//2] * cos - x[..., d//2:] * sin
        x_rot[..., d//2:] = x[..., :d//2] * sin + x[..., d//2:] * cos

        return x_rot
