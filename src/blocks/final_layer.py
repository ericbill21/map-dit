import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.layers.modulation import RotationModulation, ScaleModulation, ModulationSequential, ShiftModulation


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        scale_mod=True,
        shift_mod=True,
        rotation_mod=False,
    ):
        super().__init__()

        mod = []

        if scale_mod:
            mod.append(ScaleModulation(hidden_size, hidden_size))

        if shift_mod:
            mod.append(ShiftModulation(hidden_size, hidden_size))

        if rotation_mod:
            mod.append(RotationModulation(hidden_size, hidden_size))

        self.mod = ModulationSequential(*mod)
        self.linear = MPLinear(hidden_size, patch_size * patch_size * out_channels, zero_init=True, learn_gain=True)

    def forward(self, x, c):
        return self.linear(self.mod(x, c))
