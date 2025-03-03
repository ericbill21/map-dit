import torch.nn as nn
import torch.nn.functional as F

import math


class MPSiLU(nn.Module):
    def forward(self, x):
        # return F.silu(x) / 0.596
        return F.relu(x) * math.sqrt(2.0)

    
class MPReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) * math.sqrt(2.0)
