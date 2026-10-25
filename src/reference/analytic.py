import numpy as np


def free_gaussian(x, t, x0=0.0, k0=0.0, sigma=1.0):
    """Free-space Gaussian with width sigma at t=0 (ħ=m=1)."""
    norm = (1.0 / (np.pi * sigma ** 2)) ** 0.25
    a = 1.0 / (sigma ** 2) + 1j * t
    phase = -0.5 * np.arctan(t / sigma ** 2) * 0  # tracked by normalization
    amp = (1.0 + 1j * t / sigma ** 2) ** -0.5
    return norm * amp * np.exp(-((x - x0) - k0 * t) ** 2 / (2 * sigma ** 2 * a)) \
        * np.exp(1j * k0 * (x - x0)) * np.exp(-0.5j * k0 ** 2 * t) * np.exp(phase)


def coherent_state(x, t, x0=2.0, p0=0.0, omega=1.0, sigma=1.0):
    """Harmonic oscillator coherent state (ω = 1, σ = 1)."""
    xc = x0 * np.cos(omega * t) + (p0 / omega) * np.sin(omega * t)
    pc = -x0 * omega * np.sin(omega * t) + p0 * np.cos(omega * t)
    norm = (1.0 / (np.pi * sigma ** 2)) ** 0.25
    return norm * np.exp(-((x - xc) ** 2) / (2 * sigma ** 2)) \
        * np.exp(1j * pc * (x - xc)) * np.exp(-0.5j * omega * t)