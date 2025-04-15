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
    ):
        super().__init__()

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)
        
        self.modulation = nn.Sequential(
            MPSiLU(),
            MPLinearChunk(hidden_size, hidden_size, 6),
        )
        self.gain_msa = nn.Parameter(torch.tensor(0.0))
        self.gain_mlp = nn.Parameter(torch.tensor(0.0))


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)

        x = mp_sum(x, gate_msa.unsqueeze(1) * self.attn(modulate(x, shift_msa, scale_msa, self.gain_msa)), t=0.3)
        x = mp_sum(x, gate_mlp.unsqueeze(1) * self.mlp(modulate(x, shift_mlp, scale_mlp, self.gain_mlp)), t=0.3)
        return x