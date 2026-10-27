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


class HardConstraintPINN(nn.Module):
    def __init__(self, hidden=128, layers=4, L=20.0):
        super().__init__()
        self.L = L
        
        # Base network
        mods = []
        d = 2
        for _ in range(layers):
            mods.append(nn.Linear(d, hidden))
            mods.append(nn.Tanh())
            d = hidden
        mods.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*mods)
        
    def forward(self, x, t, psi0_real, psi0_imag):
        # 1. Hard Periodic Embedding: forces BCs automatically
        # Map x in [-L/2, L/2] to angle [0, 2pi]
        theta = 2.0 * math.pi * x / self.L
        inputs = torch.cat([torch.sin(theta), torch.cos(theta), t], dim=-1)
        
        # 2. Raw network output
        raw_out = self.net(inputs)
        u_raw = raw_out[..., 0:1]
        v_raw = raw_out[..., 1:2]
        
        # 3. Hard Initial Condition: psi(x,0) = psi0(x)
        # Use ansatz: psi(x,t) = psi0(x) + t * N(x,t)
        u = psi0_real + t * u_raw
        v = psi0_imag + t * v_raw
        
        return u, v