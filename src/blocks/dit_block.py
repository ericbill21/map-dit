import torch.nn as nn

from src.layers.adaln_modulation import AdaLNModulation
from src.layers.attention import Attention
from src.layers.mlp import MLP
from src.layers.rotation_modulation import RotationModulation
from src.utils import modulate, mp_sum


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
        use_rotation_modulation: bool,
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
        
        self.use_rot_mod = use_rotation_modulation
        if use_rotation_modulation:
            self.mod1 = RotationModulation(hidden_size, 32, hidden_size)
            self.mod2 = RotationModulation(hidden_size, 32, hidden_size)
        else:
            self.modulation = AdaLNModulation(hidden_size, 3, use_wn=use_wn, use_forced_wn=use_forced_wn, use_mp_silu=use_mp_silu)

    def forward(self, x, c):
        # Rotation modulation
        from src.utils import magnitude

        if self.use_rot_mod:
            if self.use_mp_residual:
                print(1, magnitude(x))
                x = mp_sum(x, self.attn(self.mod1(x, c)), t=0.3)
                print(2, magnitude(x))
                x = mp_sum(x, self.mlp(self.mod2(x, c)), t=0.3)
                print(3, magnitude(x))
            else:
                x = x + self.attn(self.mod1(x, c))
                x = x + self.mlp(self.mod2(x, c))

            return x

        # Scale, shift, gate modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        if self.use_mp_residual:
            x = mp_sum(x, gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)), t=0.3)
            x = mp_sum(x, gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), t=0.3)
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x
