from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


def skew_symmetric(v: torch.Tensor, d: int) -> torch.Tensor:
    """
    Inputs:
        v (torch.Tensor): Shape (batch_size, d * (d-1) // 2)
        d (int): Dimensionality of the rotation matrix

    Output (torch.Tensor): Shape (batch_size, d, d)
    """

    A = torch.zeros((v.shape[0], d, d), device=v.device, dtype=v.dtype)
    triu_indices = torch.triu_indices(d, d, offset=1)
    A[:, triu_indices[0], triu_indices[1]] = v
    A[:, triu_indices[1], triu_indices[0]] = -v

    return A


def get_rotation_matrix(v: torch.Tensor, d: int, identity: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Inputs:
        x (torch.Tensor): Shape (batch_size, d * (d-1) // 2)
        d (int): Dimensionality of the rotation matrix
        identity (torch.Tensor): Identity matrix of shape (d, d)

    Output (torch.Tensor): Shape (batch_size, d, d)
    """

    if identity is None:
        identity = torch.eye(d, device=v.device, dtype=v.dtype)

    A = skew_symmetric(v, d)
    return torch.linalg.matrix_exp(A)


class RotationNetwork(nn.Module):
    """Produces a d-dimensional rotation matrix from an n-dimensional input."""

    def __init__(self, n, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n, d * (d-1) // 2),
        )
        self.d = d
        self.register_buffer("I", torch.eye(d, dtype=torch.float32))

    def forward(self, x):
        """
        x: torch.Tensor, shape (b, n)
        """

        v = self.net(x)
        return get_rotation_matrix(v, self.d, self.I)


class RotationModulation(nn.Module):
    """Applies a condition-dependent rotation in a d-dimensional subspace of an n-dimensional space.
    The subspace is learned."""

    def __init__(self, n, d, cond_dim):
        """
        n (int): Dimensionality of full space
        d (int): Dimensionality of subspace
        cond_dim (int): Dimensionality of the conditioning variable
        """

        super().__init__()

        # Learned projection matrix P, mapping from the full space to the subspace
        self.project = orthogonal(nn.Linear(n, d, bias=False), orthogonal_map="householder")
        self.rotation_net = RotationNetwork(cond_dim, d)
        self.d = d
    
    def forward(self, x, cond):
        """
        Inputs:
            x (torch.Tensor): Shape (batch_size, timesteps, n)
            cond (torch.Tensor): Shape (batch_size, cond_dim)

        Output (torch.Tensor): Shape (batch_size, n)
        """

        R = self.rotation_net(cond)

        # Efficient way of implementing x_rot = URU^T x + x_{\perp}, where x_{\perp} = x - UU^T x
        # Here, URU^T is the rotation matrix in the subspace and x_{\perp} is the projection onto the
        # orthogonal complement of the subspace
        z = self.project(x)
        z_rot = torch.bmm(z, R)
        x_rot = x + F.linear(z_rot - z, self.project.weight.T)

        return x_rot
