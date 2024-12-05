import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import normalize


class MPConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_wn: bool,
        use_forced_wn: bool,
        zero_init: bool=False,
        learn_gain: bool=False,
        stride: int=1,
        padding: int=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        if use_wn:
            self.gain = nn.Parameter(torch.tensor(0. if zero_init else 1.), requires_grad=learn_gain)

        if use_wn:
            nn.init.normal_(self.weight)
        elif zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        """
        Args:
            x: (...B, in_channels, H, W)
        
        Returns: (...B, out_channels, H', W')
        """

        # Forced weight normalization
        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))

        # Traditional weight normalization
        if self.use_wn:
            w = normalize(self.weight) * self.gain / math.sqrt((self.in_channels) * self.kernel_size * self.kernel_size)
        else:
            w = self.weight

        return F.conv2d(x, w, stride=self.stride, padding=self.padding)
