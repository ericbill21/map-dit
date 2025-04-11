from src.dit import DiT


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, rotation_dim=64, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, rotation_dim=64, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, rotation_dim=64, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, rotation_dim=48, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, rotation_dim=48, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, rotation_dim=48, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, rotation_dim=32, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, rotation_dim=32, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, rotation_dim=32, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, rotation_dim=16, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, rotation_dim=16, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, rotation_dim=16, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=6, hidden_size=256, rotation_dim=16, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=6, hidden_size=256, rotation_dim=16, patch_size=4, num_heads=4, **kwargs)

def DiT_XS_8(**kwargs):
    return DiT(depth=6, hidden_size=256, rotation_dim=16, patch_size=8, num_heads=4, **kwargs)


DIT_MODELS = {
    "DiT-XL/2": DiT_XL_2,  "DiT-XL/4": DiT_XL_4,  "DiT-XL/8": DiT_XL_8,
    "DiT-L/2":  DiT_L_2,   "DiT-L/4":  DiT_L_4,   "DiT-L/8":  DiT_L_8,
    "DiT-B/2":  DiT_B_2,   "DiT-B/4":  DiT_B_4,   "DiT-B/8":  DiT_B_8,
    "DiT-S/2":  DiT_S_2,   "DiT-S/4":  DiT_S_4,   "DiT-S/8":  DiT_S_8,
    "DiT-XS/2": DiT_XS_2,  "DiT-XS/4": DiT_XS_4,  "DiT-XS/8": DiT_XS_8,
}
