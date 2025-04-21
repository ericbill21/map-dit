import torch.nn as nn

from src.layers.mlp import MPMLP


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ):
        super().__init__()

        self.mean_mlp = MPMLP(hidden_size, patch_size * patch_size * out_channels)
        self.variance_mlp = MPMLP(hidden_size, patch_size * patch_size * out_channels)
        self.variance_gain = MPMLP(hidden_size, 1, hidden_dim=hidden_size)

    def forward(self, x, c):
        return self.mean_mlp(x), self.variance_gain(c).unsqueeze(-2) * self.variance_mlp(x)
