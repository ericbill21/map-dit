import torch.nn as nn
import torch.nn.functional as F


class MPSiLU(nn.Module):
    """If input in N(0, 1), then output in N(0, 1) in expectation."""

    def forward(self, x):
        return (F.silu(x) - 0.206621) / 0.559538
