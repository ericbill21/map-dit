import torch.nn as nn

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
        self.modulation = AdaLNModulation(hidden_size, 3, use_wn=use_wn, use_forced_wn=use_forced_wn, use_mp_silu=use_mp_silu)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        if self.use_mp_residual:
            x = mp_sum(x, gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)), t=0.3)
            x = mp_sum(x, gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), t=0.3)
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
