"""Validate the split-step solver against exact analytic solutions."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.reference.analytic import coherent_state, free_gaussian
from src.reference.split_step import (
    _normalize,
    initial_condition,
    make_grid,
    split_step,
)


def test_free_gaussian():
    L, Nx, T = 40.0, 1024, 1.0
    x, k = make_grid(L, Nx)
    dx = x[1] - x[0]
    psi0 = free_gaussian(x, 0.0, x0=0.0, k0=0.0, sigma=1.0)
    psi0 = _normalize(psi0, dx)     # <-- line 16, CORRECT for Gaussian
    n = 1000
    _, snaps = split_step(psi0, x, k, T / n, n, np.zeros_like(x), 0.0,
                          snapshot_every=n)
    psi_num = snaps[-1]
    psi_exact = free_gaussian(x, T, x0=0.0, k0=0.0, sigma=1.0)
    psi_exact = _normalize(psi_exact, dx)
    err = np.sqrt(np.sum(np.abs(psi_num - psi_exact) ** 2) * dx)
    print(f"Free Gaussian: |psi_num - psi_exact|_2 = {err:.2e}")
    assert err < 1e-6, f"solver-analytic mismatch: {err}"


def test_coherent_state():
    L, Nx, T = 16.0, 512, 2 * np.pi
    x, k = make_grid(L, Nx)
    dx = x[1] - x[0]
    omega = 1.0
    psi0 = coherent_state(x, 0.0, x0=2.0, p0=0.0, omega=omega, sigma=1.0)
    psi0 = _normalize(psi0, dx)     # Gaussian -> correct to normalize
    V = 0.5 * omega ** 2 * x ** 2
    n = 4000
    _, snaps = split_step(psi0, x, k, T / n, n, V, 0.0, snapshot_every=n)
    psi_num = snaps[-1]
    psi_exact = coherent_state(x, T, x0=2.0, p0=0.0, omega=omega, sigma=1.0)
    psi_exact = _normalize(psi_exact, dx)
    overlap = np.abs(np.vdot(psi_exact, psi_num)) / (
        np.linalg.norm(psi_exact) * np.linalg.norm(psi_num)
    )
    print(f"Coherent state: |<psi_exact|psi_num>| = {overlap:.10f}")
    assert overlap > 0.9999, f"coherent state recurrence failed: {overlap}"


def test_soliton_stationarity():
    L, Nx, T = 40.0, 1024, 4.0
    x, k = make_grid(L, Nx)
    dx = x[1] - x[0]

    # Stationary soliton (v=0).
    # DO NOT NORMALIZE: bright soliton stationarity requires A^2 = |g|.
    # For g=-1, A=1 and norm = 2A = 2.
    psi0 = initial_condition(x, {"kind": "soliton", "v": 0.0})

    n = 1000
    _, snaps = split_step(psi0, x, k, T / n, n, np.zeros_like(x), -1.0,snapshot_every=n)
    psi_final = snaps[-1]

    # Stationarity is about DENSITY, not wavefunction. A stationary state
    # only acquires a global phase e^{-i mu t}; the density |psi|^2 is
    # exactly constant. Compare densities, not complex wavefunctions.
    density_err = np.sqrt(np.sum((np.abs(psi_final)**2 - np.abs(psi0)**2)**2) * dx)
    overlap = np.abs(np.vdot(psi0, psi_final)) / (
        np.linalg.norm(psi0) * np.linalg.norm(psi_final)
    )
    print(f"Soliton density drift at t=4: {density_err:.2e}")
    print(f"Soliton phase-invariant overlap: {overlap:.10f}")
    assert density_err < 1e-3, f"Soliton density drifted! err = {density_err:.2e}"
    assert overlap > 0.9999, f"Soliton overlap too small! overlap = {overlap:.6f}"


if __name__ == "__main__":
    test_free_gaussian()
    test_coherent_state()
    test_soliton_stationarity()
    print("All analytic tests passed.")
