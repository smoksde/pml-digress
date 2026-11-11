import math

import torch


def sinusoidal_timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


def condition_vector(t, y, time_dim, y_dim):
    parts = [sinusoidal_timestep_embedding(t, time_dim)]
    if y_dim:
        if y is None:
            raise ValueError("this model is conditional; pass a label y")
        parts.append(y)
    return torch.cat(parts, -1)


def normalize(y, mean, std):
    return (y - mean.to(y.device)) / std.to(y.device).clamp(min=1e-6)
