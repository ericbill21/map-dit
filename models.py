import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def normalize(x, eps=1e-4):
    # Dividing by norm makes the std of the weights equal to 1/sqrt(in_dim), so we
    # multiply by sqrt(in_dim) to compensate (TODO: Add derivation to appendix)
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    x_norm = torch.add(eps, x_norm, alpha=math.sqrt(1.0 / x.shape[-1]))
    return x.div(x_norm)


class MPLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        zero_init: bool=False,
        use_wn: bool=False,
        learn_gain: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim+1))

        if use_wn:
            self.gain = nn.Parameter(torch.tensor(1.), requires_grad=learn_gain)

        if use_forced_wn:
            nn.init.normal_(self.weight)
        elif zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        """
        Args:
            x: (...B, in_dim)
        
        Returns: (...B, out_dim)
        """

        # Forced weight normalization
        w = self.weight.to(torch.float32)
        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Traditional weight normalization
        if self.use_wn:
            w = normalize(w)
            w = w * (self.gain / math.sqrt(self.in_dim + 1))

        w = w.to(x.dtype)

        # Concatenate 1 to input for bias
        x = torch.cat([x, torch.ones(*x.shape[:-1], 1).to(x.device)], dim=-1)
        return F.linear(x, w)


class MPConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int=1,
        padding: int=0,
        zero_init: bool=False,
        use_wn: bool=False,
        learn_gain: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels+1, kernel_size, kernel_size))

        if use_wn:
            self.gain = nn.Parameter(torch.tensor(1.), requires_grad=learn_gain)

        if use_forced_wn:
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
        w = self.weight.to(torch.float32)
        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(w))

        # Traditional weight normalization
        if self.use_wn:
            w = normalize(w)
            w = w * (self.gain / math.sqrt((self.in_channels + 1) * self.kernel_size * self.kernel_size))

        w = w.to(x.dtype)

        # Concatenate 1 to channels for bias
        x = torch.cat([x, torch.ones(*x.shape[:-3], 1, *x.shape[-2:]).to(x.device)], dim=-3)
        return F.conv2d(x, w, stride=self.stride, padding=self.padding)


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int, num_modulates: int, use_wn: bool=False, use_forced_wn: bool=False):
        super().__init__()

        self.num_modulates = num_modulates
        self.net = nn.Sequential(
            nn.SiLU(),
            MPLinear(
                hidden_dim,
                2 * num_modulates * hidden_dim,
                zero_init=True,
                use_wn=use_wn,
                learn_gain=True,
                use_forced_wn=use_forced_wn,
            ),
        )

    def forward(self, x):
        return self.net(x).chunk(2 * self.num_modulates, dim=1)


class Attention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, use_cosine_attention: bool=False, use_wn: bool=False, use_forced_wn: bool=False):
        super().__init__()

        assert in_dim % num_heads == 0

        self.use_cosine = use_cosine_attention
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.qkv = MPLinear(in_dim, 3 * in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.out_proj = MPLinear(in_dim, in_dim, use_wn=use_wn, use_forced_wn=use_forced_wn)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (...B, T, D)
        
        Returns: (...B, T, D)
        """

        T, D = x.shape[-2:]

        q, k, v = self.qkv(x).chunk(3, dim=-1)                              # (...B, T, D)

        q = q.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D') where D' = D / H
        k = k.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')
        v = v.view(-1, T, self.num_heads, self.head_dim).transpose(-3, -2)  # (...B, H, T, D')

        if self.use_cosine:
            q = normalize(q)
            k = normalize(k)
            v = normalize(v)

        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)    # (...B, H, T, D')
        attn = attn.transpose(-3, -2)                                       # (...B, T, H, D')
        attn = attn.reshape(*x.shape)                                       # (...B, T, D)

        return self.out_proj(attn)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_ratio: float=4.0,
        hidden_dim: int=None,
        use_wn: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        self.hidden_dim = int(in_dim * mlp_ratio) if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            MPLinear(in_dim, self.hidden_dim, use_wn=use_wn, use_forced_wn=use_forced_wn),
            nn.SiLU(),
            MPLinear(self.hidden_dim, out_dim, use_wn=use_wn, use_forced_wn=use_forced_wn),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedder(nn.Module):
    """Embeds image patches."""

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int, use_wn: bool=False, use_forced_wn: bool=False):
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


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int=256,
        use_wn: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_dim=hidden_size,
            use_wn=use_wn,
            use_forced_wn=use_forced_wn,
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element (may be fractional)
            dim: the dimension of the output (D)
            max_period: controls the minimum frequency of the embeddings

        Returns: positional embeddings (N, D)
        """

        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        # Init table
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""

        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float=4.0,
        use_cosine_attention: bool=False,
        use_wn: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        # TODO: Flag for disabling layer norm

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, use_cosine_attention=use_cosine_attention, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, hidden_size, mlp_ratio, use_wn=use_wn, use_forced_wn=use_forced_wn)
        self.modulation = AdaLNModulation(hidden_size, 3, use_wn=use_wn, use_forced_wn=use_forced_wn)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn = self.attn(x)
        x = x + gate_msa.unsqueeze(1) * attn
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_wn: bool=False,
        use_forced_wn: bool=False,
    ):
        super().__init__()

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = MPLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            zero_init=True,
            use_wn=use_wn,
            learn_gain=True,
            use_forced_wn=use_forced_wn,
        )
        self.modulation = AdaLNModulation(hidden_size, 1, use_wn=use_wn, use_forced_wn=use_forced_wn)

    def forward(self, x, c):
        shift, scale = self.modulation(c)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


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
        use_no_layernorm: bool=False,
    ):
        super().__init__()

        if use_no_layernorm:
            raise NotImplementedError

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbedder(patch_size, in_channels, hidden_size, use_wn=use_weight_normalization, use_forced_wn=use_forced_weight_normalization)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(hidden_size, input_size // patch_size)).float().unsqueeze(0),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                use_cosine_attention=use_cosine_attention,
                use_wn=use_weight_normalization,
                use_forced_wn=use_forced_weight_normalization,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
            use_wn=use_weight_normalization,
            use_forced_wn=use_forced_weight_normalization,
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

        x = self.x_embedder(x) + self.pos_embed                                 # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                                                  # (N, D)
        y = self.y_embedder(y, self.training)                                   # (N, D)
        c = t + y                                                               # (N, D)

        for block in self.blocks:
            # print(x.std(-1).mean().item())
            x = checkpoint(self.ckpt_wrapper(block), x, c, use_reentrant=True)  # (N, T, D)

        # print(x.std(-1).mean().item())
        # input()

        x = self.final_layer(x, c)                                              # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                                                  # (N, out_channels, H, W)
        return x

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


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        grid_size: int of the grid height and width

    Returns: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim) (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py"""

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Source: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim: output dimension for each position (D)
        pos: (M,)

    Returns: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=4, num_heads=4, **kwargs)

def DiT_XS_8(**kwargs):
    return DiT(depth=6, hidden_size=256, patch_size=8, num_heads=4, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,  "DiT-XL/4": DiT_XL_4,  "DiT-XL/8": DiT_XL_8,
    "DiT-L/2":  DiT_L_2,   "DiT-L/4":  DiT_L_4,   "DiT-L/8":  DiT_L_8,
    "DiT-B/2":  DiT_B_2,   "DiT-B/4":  DiT_B_4,   "DiT-B/8":  DiT_B_8,
    "DiT-S/2":  DiT_S_2,   "DiT-S/4":  DiT_S_4,   "DiT-S/8":  DiT_S_8,
    "DiT-XS/2":  DiT_XS_2,   "DiT-XS/4":  DiT_XS_4,   "DiT-XS/8":  DiT_XS_8,
}
