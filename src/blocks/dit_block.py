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
            2 if use_no_shift else 3,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
            use_mp_silu=use_mp_silu
        )
        self.use_no_shift = use_no_shift

        # Learning 
        if learn_blending:
            self.blend_factor_msa = nn.Parameter(torch.tensor(0.0))
            self.blend_factor_mlp = nn.Parameter(torch.tensor(0.0))
        else:
            # TODO: Work arround for now, to get a blending factor of 0.3
            self.blend_factor_msa = torch.tensor(-0.8472977876663208)
            self.blend_factor_mlp = torch.tensor(-0.8472977876663208)

    def forward(self, x, c):

        if self.use_no_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.modulation(c)

            shift_msa = torch.zeros_like(scale_msa)
            shift_mlp = torch.zeros_like(scale_mlp)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)

        if self.use_mp_residual:
            x = mp_sum(x, gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)), t=F.sigmoid(self.blend_factor_msa))
            x = mp_sum(x, gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), t=F.sigmoid(self.blend_factor_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
