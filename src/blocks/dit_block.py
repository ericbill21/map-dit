import torch.nn.functional as F
import torch.nn as nn
import torch

from src.layers.attention import Attention
from src.layers.mlp import MLP
from src.layers.adaln_modulation import AdaLNModulation
from src.utils import mp_sum, modulate

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_cosine_attention: bool,
        use_wn: bool,
        use_forced_wn: bool,
        use_mp_residual: bool,
        use_mp_silu: bool,
        use_no_layernorm: bool,
        use_no_shift: bool,
        learn_blending: bool,
        use_sigmoid_attn: bool,
        use_rotation_modulation: bool,
        force_magnitude: bool,
        mlp_ratio: float=4.0,
    ):
        super().__init__()

        self.use_mp_residual = use_mp_residual

        if use_no_layernorm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            hidden_size,
            num_heads,
            use_cosine_attention=use_cosine_attention,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_sigmoid_attn=use_sigmoid_attn,
            force_magnitude=force_magnitude,
        )

        if use_no_layernorm:
            self.norm2 = nn.Identity()
        else:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp = MLP(
            hidden_size,
            hidden_size,
            mlp_ratio=mlp_ratio,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu,
        )

        self.modulation = AdaLNModulation(
            hidden_size,
            1,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu
        )

    def forward(self, x, c):
        shift_msa, shift_mlp = self.modulation(c)

        x = mp_sum(x, self.attn(mp_sum(x, shift_msa, t=0.5)), t=0.3)
        x = mp_sum(x, self.mlp(mp_sum(x, shift_mlp, t=0.5)), t=0.3)

        return x
