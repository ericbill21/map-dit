import torch
import numpy as np

def mp_cat(a, b, dim=1, t=0.5):
    num_a, num_b = a.shape[dim], b.shape[dim]

    C = np.sqrt(
        (num_a + num_b) / (t**2 + (1-t)**2)
    )

    return C * torch.cat([
                    (1 - t) / np.sqrt(num_a) * a,
                    t       / np.sqrt(num_b) * b
                ], dim=dim)