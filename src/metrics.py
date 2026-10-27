import numpy as np


def rel_l2(pred, ref, dx):
    num = np.sum(np.abs(pred - ref) ** 2) * dx
    den = np.sum(np.abs(ref) ** 2) * dx
    return float(np.sqrt(num / max(den, 1e-30)))


def norm_sq(psi, dx):
    return float(np.sum(np.abs(psi) ** 2) * dx)


def energy(psi, x, k, V, g, dx):
    """⟨H⟩ = ∫ (½ |ψ_x|² + V|ψ|² + g/2 |ψ|⁴) dx."""
    dx_ = x[1] - x[0]
    dpsi = np.fft.ifft(1j * k * np.fft.fft(psi))
    kinetic = 0.5 * np.sum(np.abs(dpsi) ** 2) * dx_
    potential = np.sum(V * np.abs(psi) ** 2) * dx_
    interact = 0.5 * g * np.sum(np.abs(psi) ** 4) * dx_
    return float(kinetic + potential + interact)


def spectral_error(err, dx):
    """Return |FFT(err)|^2 and wavenumbers."""
    f = np.fft.fft(err)
    k = 2 * np.pi * np.fft.fftfreq(len(err), d=dx)
    return np.fft.fftshift(k), np.fft.fftshift(np.abs(f) ** 2)
