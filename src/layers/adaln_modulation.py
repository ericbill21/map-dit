import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.basic.mp_silu import MPSiLU


class AdaLNModulation(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_modulates: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
        rotation: bool = False,
    ):
        super().__init__()

        self.num_modulates = num_modulates
        self.net = nn.Sequential(
            MPSiLU() if use_mp_silu else nn.SiLU(),
            MPLinear(
                hidden_dim,
                2 * num_modulates * hidden_dim // 2 if rotation else 2 * num_modulates * hidden_dim,
                zero_init=True,
                use_wn=use_wn,
                learn_gain=True,
                use_forced_wn=use_forced_wn,
            ),
        )

    def forward(self, x):
        return self.net(x).chunk(2 * self.num_modulates, dim=1)
