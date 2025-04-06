import torch
import torch.nn as nn


class RotationNetwork(nn.Module):
    """Produces a d-dimensional rotation matrix from an n-dimensional input."""

    def __init__(self, n, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n, d * (d-1) // 2),
        )
        self.d = d

    def forward(self, x):
        """
        x: torch.Tensor, shape (b, n)
        """

        v = self.net(x)

        # Construct skew-symmetric matrix from vector
        A = torch.zeros((x.shape[0], self.d, self.d), device=v.device, dtype=v.dtype)
        triu_indices = torch.triu_indices(self.d, self.d, offset=1)
        A[:, triu_indices[0], triu_indices[1]] = v
        A[:, triu_indices[1], triu_indices[0]] = -v

        return torch.linalg.matrix_exp(A)


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
        self.P = nn.Parameter(torch.randn(n, d))
        self.rotation_net = RotationNetwork(cond_dim, d)
        self.d = d
    
    def forward(self, x, cond):
        """
        Inputs:
            x (torch.Tensor): Shape (batch_size, timesteps, n)
            cond (torch.Tensor): Shape (batch_size, cond_dim)

        Output (torch.Tensor): Shape (batch_size, n)
        """

        U, _ = torch.linalg.qr(self.P)
        R = self.rotation_net(cond)

        # Efficient way of implementing x_rot = URU^T x + x_{\perp}, where x_{\perp} = x - UU^T x
        # Here, URU^T is the rotation matrix in the subspace and x_{\perp} is the projection onto the
        # orthogonal complement of the subspace
        z = torch.matmul(x, U)
        z_rot = torch.matmul(z, R)
        x_rot = x + torch.matmul(z_rot - z, U.T)

        return x_rot
