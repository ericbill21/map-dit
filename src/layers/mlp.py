import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU
from src.utils import normalize, magnitude


class MPMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_ratio: float=4.0,
        hidden_dim: int=None,
    ):
        super().__init__()

        self.hidden_dim = int(in_dim * mlp_ratio) if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            MPLinear(in_dim, self.hidden_dim),
            MPSiLU(),
            MPLinear(self.hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_ratio: float=4.0,
        hidden_dim: int=None,
    ):
        super().__init__()

        self.hidden_dim = int(in_dim * mlp_ratio) if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
