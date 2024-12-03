import torch.nn as nn

from src.basic.mp_conv2d import MPConv2d


class PatchEmbedder(nn.Module):
    """Embeds image patches."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        use_wn: bool,
        use_forced_wn: bool,
    ):
        super().__init__()

        self.proj = MPConv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
        )

    def forward(self, x):
        """
        Args:
            x: (...B, C, H, W)

        Returns: (...B, T, D) where T = H * W / patch_size ** 2
        """

        x = self.proj(x)         # (...B, D, H', W') where H' = H // patch_size, W' = W // patch_size
        x = x.flatten(-2)        # (...B, D, T) where T = H' * W'
        x = x.transpose(-1, -2)  # (...B, T, D)
        return x
