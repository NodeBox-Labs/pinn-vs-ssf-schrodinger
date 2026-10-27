"""Hard-Constraint PINN for the 1D Schrödinger / GPE.

Uses:
  1. Hard periodic embedding: maps x -> (sin(2πx/L), cos(2πx/L))
  2. Hard initial condition:   ψ(x,t) = ψ0(x) + t * N(x,t)

This forces the network to satisfy IC and BC exactly, without soft loss penalties.
"""
import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Standard random Fourier features (optional)."""
    def __init__(self, in_dim: int, n_features: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_features, generator=g) * scale
        self.register_buffer("B", B)

    def forward(self, z):
        proj = 2.0 * math.pi * z @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class HardConstraintPINN(nn.Module):
    """PINN with hard periodic BCs and hard initial conditions."""
    def __init__(self, L: float, hidden=128, layers=4, activation="tanh",
                 use_fourier=False, fourier_scale=1.0, fourier_dim=128,
                 seed=0):
        super().__init__()
        self.L = L
        self.use_fourier = use_fourier

        # Input dim is 3: sin(theta), cos(theta), t
        in_dim = 3
        if use_fourier:
            # Fourier features applied to the embedded (sin, cos, t) coordinates
            self.ff = FourierFeatures(3, fourier_dim, fourier_scale, seed=seed)
            in_dim = 2 * fourier_dim

        act_cls = {"tanh": nn.Tanh, "sin": nn.SiLU, "gelu": nn.GELU}[activation]

        mods = []
        d = in_dim
        for _ in range(layers):
            mods.append(nn.Linear(d, hidden))
            mods.append(act_cls())
            d = hidden
        mods.append(nn.Linear(d, 2)) # Output: (u_raw, v_raw)
        self.net = nn.Sequential(*mods)

    def forward(self, x, t, psi0_real, psi0_imag):
        # 1. Hard Periodic Embedding
        # Maps x in [-L/2, L/2] to an angle [0, 2pi]
        theta = 2.0 * math.pi * x / self.L
        z = torch.cat([torch.sin(theta), torch.cos(theta), t], dim=-1)

        if self.use_fourier:
            z = self.ff(z)

        # 2. Raw network output
        raw = self.net(z)
        u_raw, v_raw = raw[..., 0:1], raw[..., 1:2]

        # 3. Hard Initial Condition Ansatz
        # ψ(x,0) = ψ0(x) + 0 * N(x,0) = ψ0(x)
        u = psi0_real + t * u_raw
        v = psi0_imag + t * v_raw

        return u, v
