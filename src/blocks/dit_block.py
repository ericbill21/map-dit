import torch
import torch.nn as nn

from src.basic.mp_silu import MPSiLU
from src.basic.mp_linear import MPLinearChunk
from src.layers.attention import Attention
from src.layers.mlp import MLP
from src.utils import mp_sum, modulate

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float=4.0,
        use_no_shift: bool=False,
    ):
        super().__init__()

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)

        self.use_no_shift = use_no_shift

        num_modalities = 4 if use_no_shift else 6
        self.modulation = nn.Sequential(
            MPSiLU(),
            nn.Linear(hidden_size, num_modalities * hidden_size, bias=False),
        )
        nn.init.zeros_(self.modulation[1].weight)

    def forward(self, x, c):
        if self.use_no_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.modulation(c).chunk(4, dim=-1)
            shift_msa = torch.zeros_like(scale_msa)
            shift_mlp = torch.zeros_like(scale_mlp)
        
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c).chunk(6, dim=-1)

        x = mp_sum(x, gate_msa.unsqueeze(1) * self.attn(modulate(x, shift_msa, scale_msa)), t=0.3)
        x = mp_sum(x, gate_mlp.unsqueeze(1) * self.mlp( modulate(x, shift_mlp, scale_mlp)), t=0.3)
        return x