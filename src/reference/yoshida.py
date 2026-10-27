from __future__ import annotations
import numpy as np


YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_W0 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))


def _strang_step(psi, x, k, dt, V, g):
    """One Strang step of size dt."""
    Veff = V + g * np.abs(psi) ** 2
    psi = psi * np.exp(-0.5j * Veff * dt)
    psi = np.fft.ifft(np.exp(-0.5j * (k ** 2) * dt) * np.fft.fft(psi))
    Veff = V + g * np.abs(psi) ** 2
    psi = psi * np.exp(-0.5j * Veff * dt)
    return psi


def yoshida_step(psi, x, k, dt, V, g):
    """One Yoshida 4th-order step of size dt."""
    psi = _strang_step(psi, x, k, YOSHIDA_W1 * dt, V, g)
    psi = _strang_step(psi, x, k, YOSHIDA_W0 * dt, V, g)
    psi = _strang_step(psi, x, k, YOSHIDA_W1 * dt, V, g)
    return psi


def split_step_yoshida(psi0, x, k, dt, n_steps, V, g,
                       snapshot_every=100):
    """Full evolution with Yoshida 4th-order splitting."""
    psi = psi0.copy()
    times = [0.0]
    snaps = [psi.copy()]
    for n in range(1, n_steps + 1):
        psi = yoshida_step(psi, x, k, dt, V, g)
        if n % snapshot_every == 0:
            times.append(n * dt)
            snaps.append(psi.copy())
    return np.array(times), np.array(snaps)