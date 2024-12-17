import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.layers.adaln_modulation import AdaLNModulation
from src.utils import modulate


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_silu: bool,
        use_no_layernorm: bool,
    ):
        super().__init__()

        if use_no_layernorm:
            self.norm_final = nn.Identity()
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = MPLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            zero_init=True,
            use_wn=use_wn,
            learn_gain=True,
            use_forced_wn=use_forced_wn,
        )
        self.modulation = AdaLNModulation(
            hidden_size,
            1,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu,
        )

    def forward(self, x, c):
        shift, scale = self.modulation(c)
        return self.linear(modulate(self.norm_final(x), shift, scale))
