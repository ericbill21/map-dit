import torch
import torch.nn as nn

from src.basic.mp_linear import MPLinear
from src.blocks.dit_block import DiTBlock
from src.blocks.label_embedder import LabelEmbedder
from src.blocks.timestep_embedder import TimestepEmbedder
from src.blocks.final_layer import FinalLayer
from src.pos_embed import get_2d_sincos_pos_embed
from src.utils import mp_sum, patchify, unpatchify, normalize

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
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
    
        # Add one to the in_channels for input bias
        self.x_embedder = MPLinear(
            patch_size * patch_size * in_channels + 1,
            hidden_size,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Normalize the positional embedding as its a constant
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)).float().unsqueeze(0)
        pos_embed = normalize(pos_embed - pos_embed.mean(dim=-1, keepdim=True))
        self.register_buffer("pos_embed", pos_embed)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
        )
    
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

        # Extract patches into features and add positional embedding
        x = patchify(x, self.patch_size)                                        # (N, T, (patch_size ** 2) * in_channels)
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)                # (N, T, (patch_size ** 2) * in_channels + 1)
        x = mp_sum(self.x_embedder(x), self.pos_embed, t=0.5)                   # (N, T, D)
   
        t = self.t_embedder(t)                                                  # (N, D)
        y = self.y_embedder(y, self.training)                                   # (N, D)
        c = mp_sum(t, y, t=0.5)                                                 # (N, D)

        for block in self.blocks:
            x = block(x, c)

        mean, variance = self.final_layer(x, c)

        return torch.cat([
            unpatchify(mean, self.input_size, self.patch_size),
            unpatchify(variance, self.input_size, self.patch_size)
        ], dim=1)                                                               # (N, T, patch_size ** 2 * out_channels * 2)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance."""

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[:len(x)//2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
