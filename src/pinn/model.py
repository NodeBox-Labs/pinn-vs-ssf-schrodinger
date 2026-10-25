"""MLP with separate real/imag outputs, optional Fourier feature embedding."""
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_features: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_features, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, z):
        proj = 2.0 * math.pi * z @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    """Maps (x, t) -> (Re ψ, Im ψ)."""
    def __init__(self, hidden=128, layers=4, activation="tanh",
                 use_fourier=False, fourier_scale=1.0, fourier_dim=128,
                 seed=0):
        super().__init__()
        self.use_fourier = use_fourier
        in_dim = 2
        if use_fourier:
            self.ff = FourierFeatures(2, fourier_dim, fourier_scale, seed=seed)
            in_dim = 2 * fourier_dim
        act = {"tanh": nn.Tanh, "sin": torch.sin, "gelu": nn.GELU}[activation]
        mods = []
        d = in_dim
        for _ in range(layers):
            mods.append(nn.Linear(d, hidden))
            mods.append(act() if activation != "sin" else nn.SiLU())
            d = hidden
        mods.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*mods)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=-1)
        if self.use_fourier:
            z = self.ff(z)
        out = self.net(z)
        return out[..., 0:1], out[..., 1:2]