import torch.nn as nn

from src.layers.attention import Attention
from src.layers.mlp import MPMLP
from src.utils import mp_sum
from src.layers.rotation_modulation import RotationModulation


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float=4.0,
    ):
        super().__init__()

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MPMLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)

        self.mod1 = RotationModulation(hidden_size, hidden_size)
        self.mod2 = RotationModulation(hidden_size, hidden_size)

    def forward(self, x, c):
        x = mp_sum(x, self.attn(self.mod1(x, c)), t=0.3)
        x = mp_sum(x, self.mlp(self.mod2(x, c)), t=0.3)
        return x
