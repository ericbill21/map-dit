import torch.nn as nn
import torch.nn.functional as F


class MPSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x) / 0.596
        # return (F.silu(x) - 0.2066) / 0.5595