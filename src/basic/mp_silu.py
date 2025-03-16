import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class MPSiLU(nn.Module):
#     def forward(self, x):
#         return F.silu(x) / 0.596


def standard_cdf(x):
    """CDF of N(0,1)."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def standard_pdf(x):
    """PDF of N(0,1)."""
    return torch.exp(-x.square() / 2) / math.sqrt(2 * torch.pi)


def relu_squared_magnitude(mean, std):
    """Computes expected squared magnitude of f(x)=max{0,x}, where x ~ N(mean, std^2)."""
    return (mean.square() + std.square()) * standard_cdf(mean / std) + mean * std * standard_pdf(mean / std)


def identity_squared_magnitude(mean, std):
    """Computes expected squared magnitude of f(x)=x, where x ~ N(mean, std^2)."""
    return mean.square() + std.square()


class MPSiLU(nn.Module):
    def forward(self, x, mean=None, std=None):
        if mean is None or std is None:
            print("mean and std must be provided")
            return F.relu(x) * math.sqrt(2)

        prior_magnitude = identity_squared_magnitude(mean, std)
        post_magnitude = relu_squared_magnitude(mean, std)

        print(f"{x.shape=}")
        print(f"{mean.shape=}")
        print(f"{std.shape=}")
        input()

        return F.relu(x) * torch.sqrt(prior_magnitude / post_magnitude).unsqueeze(-1)