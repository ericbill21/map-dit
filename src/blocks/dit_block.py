import torch.nn as nn

from src.layers.attention import Attention
from src.layers.mlp import MLP
from src.utils import mp_sum
from src.layers.modulation import ShiftModulation, RotationModulation, ScaleModulation, ModulationSequential


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio=4.0,
        scale_mod=True,
        shift_mod=True,
        gate_mod=True,
        rotation_mod=False,
    ):
        super().__init__()

        self.attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio=mlp_ratio)

        mod1 = []
        mod2 = []

        if scale_mod:
            mod1.append(ScaleModulation(hidden_size, hidden_size))
            mod2.append(ScaleModulation(hidden_size, hidden_size))

        if shift_mod:
            mod1.append(ShiftModulation(hidden_size, hidden_size))
            mod2.append(ShiftModulation(hidden_size, hidden_size))

        if rotation_mod:
            mod1.append(RotationModulation(hidden_size, hidden_size))
            mod2.append(RotationModulation(hidden_size, hidden_size))

        self.mod1 = ModulationSequential(*mod1)
        self.mod2 = ModulationSequential(*mod2)

        if gate_mod:
            self.gate1 = ScaleModulation(hidden_size, hidden_size)
            self.gate2 = ScaleModulation(hidden_size, hidden_size)
        else:
            self.gate1 = nn.Identity()
            self.gate2 = nn.Identity()

    def forward(self, x, c):
        x = mp_sum(x, self.gate1(self.attn(self.mod1(x, c)), c), t=0.3)
        x = mp_sum(x, self.gate2(self.mlp(self.mod2(x, c)), c), t=0.3)
        return x
