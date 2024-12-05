import torch
import numpy as np

class Fourier(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        
        scale = 2 * torch.pi * torch.randn(num_channels) 
        shift = 2 * torch.pi * torch.rand(num_channels)

        self.register_buffer('scale', scale.to(torch.float32))
        self.register_buffer('shift', shift.to(torch.float32))

    def forward(self, x):
        # cos(2 * \pi * (freqs * x + phases))
        res = torch.cos(torch.outer(x, self.scale) + self.shift)
        return res.to(torch.float32)

class MPFourier(Fourier):
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def forward(self, x):
        return np.sqrt(2) * super().forward(x)