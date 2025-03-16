import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
        mlp_ratio: float=4.0,
        hidden_dim: int=None,
    ):
        super().__init__()

        self.hidden_dim = int(in_dim * mlp_ratio) if hidden_dim is None else hidden_dim
        
        self.linear_in = MPLinear(in_dim, self.hidden_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.non_linearity = MPSiLU() if use_mp_silu else nn.SiLU() # FIXME: won't take three arguments
        self.linear_out = MPLinear(self.hidden_dim, out_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)

    def forward(self, x, scale, shift):
        x = self.linear_in(x)
        x = self.non_linearity(x, scale, shift)
        return self.linear_out(x)
