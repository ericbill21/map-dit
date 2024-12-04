import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.blocks.dit_block import DiTBlock
from src.blocks.label_embedder import LabelEmbedder
from src.blocks.timestep_embedder import TimestepEmbedder
from src.blocks.patch_embedder import PatchEmbedder
from src.blocks.final_layer import FinalLayer
from src.pos_embed import get_2d_sincos_pos_embed
from src.utils import mp_sum


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        depth: int,
        hidden_size: int,
        patch_size: int,
        input_size: int=32,
        in_channels: int=3,
        num_heads: int=16,
        mlp_ratio: float=4.0,
        class_dropout_prob: float=0.1,
        num_classes: int=1000,
        learn_sigma: bool=True,
        use_cosine_attention: bool=False,
        use_weight_normalization: bool=False,
        use_forced_weight_normalization: bool=False,
        use_mp_residual: bool=False,
        use_mp_silu: bool=False,
        use_mp_fourier: bool=False,
        use_no_layernorm: bool=False,
        use_mp_pos_enc: bool=False,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_mp_residual = use_mp_residual

        # Add one to the in_channels for input bias
        self.x_embedder = PatchEmbedder(
            patch_size,
            in_channels+1,
            hidden_size,
            use_wn=use_weight_normalization,
            use_forced_wn=use_forced_weight_normalization,
        )
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            use_wn=use_weight_normalization,
            use_forced_wn=use_forced_weight_normalization,
            use_mp_silu=use_mp_silu,
            use_mp_fourier=use_mp_fourier,
        )
        self.y_embedder = LabelEmbedder(
            num_classes,
            hidden_size,
            class_dropout_prob,
        )

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)).float().unsqueeze(0),
            requires_grad=False,
        )
        # Normalize positional embedding to standard variance
        if use_mp_pos_enc:
            self.pos_embed.copy_(self.pos_embed / self.pos_embed.std())

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_cosine_attention=use_cosine_attention,
                use_wn=use_weight_normalization,
                use_forced_wn=use_forced_weight_normalization,
                use_mp_residual=use_mp_residual,
                use_mp_silu=use_mp_silu,
                use_no_layernorm=use_no_layernorm,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
            use_wn=use_weight_normalization,
            use_forced_wn=use_forced_weight_normalization,
            use_mp_silu=use_mp_silu,
            use_no_layernorm=use_no_layernorm,
        )

    def unpatchify(self, x):
        """
        Args:
            x: (N, T, patch_size**2 * C)

        Returns: (N, H, W, C)
        """

        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1]

        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(-1, c, h * p, w * p)
        return imgs
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Args:
            x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels

        Returns: (N, C, H, W)
        """

        # Concatenate 1s to the input for input bias
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)                   # (N, C+1, H, W)

        if self.use_mp_residual:
            x = mp_sum(self.x_embedder(x), self.pos_embed, t=0.5)               # (N, T, D)
        else:
            x = self.x_embedder(x) + self.pos_embed                             # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                                                  # (N, D)
        y = self.y_embedder(y, self.training)                                   # (N, D)

        if self.use_mp_residual:
            c = mp_sum(t, y, t=0.5)                                             # (N, D)
        else:
            c = t + y

        for block in self.blocks:
            if self.training:
                x = checkpoint(self.ckpt_wrapper(block), x, c, use_reentrant=True)  # (N, T, D)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)                                              # (N, T, patch_size ** 2 * out_channels)
        return self.unpatchify(x)                                               # (N, out_channels, H, W)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance."""

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
