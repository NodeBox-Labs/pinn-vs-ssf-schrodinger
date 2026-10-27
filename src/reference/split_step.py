from __future__ import annotations

import numpy as np


def make_grid(L: float, Nx: int):
    x = np.linspace(-L / 2.0, L / 2.0, Nx, endpoint=False)
    dx = L / Nx
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    return x, k


def potential(x: np.ndarray, kind: str = "none", omega: float = 1.0) -> np.ndarray:
    if kind == "none":
        return np.zeros_like(x)
    if kind == "harmonic":
        return 0.5 * omega ** 2 * x ** 2
    raise ValueError(kind)


def initial_condition(x: np.ndarray, cfg: dict) -> np.ndarray:
    kind = cfg["kind"]
    if kind in ("free_gaussian", "gaussian"):
        x0 = cfg.get("x0", 0.0)
        k0 = cfg.get("k0", 0.0)
        w = cfg.get("width", 1.0)
        norm = (1.0 / (np.pi * w ** 2)) ** 0.25
        return norm * np.exp(-((x - x0) ** 2) / (2 * w ** 2)) * np.exp(1j * k0 * x)
    if kind == "coherent":
        x0 = cfg.get("x0", 0.0)
        p0 = cfg.get("p0", 0.0)
        w = cfg.get("width", 1.0)
        norm = (1.0 / (np.pi * w ** 2)) ** 0.25
        return norm * np.exp(-((x - x0) ** 2) / (2 * w ** 2)) * np.exp(1j * p0 * x)
    if kind == "soliton":
        v = cfg["v"]
        x0 = cfg.get("x0", 0.0)
        # Stationary bright soliton for g=-1: A=1, so A^2=|g|.
        A = 1.0
        return (A / np.cosh(A * (x - x0))) * np.exp(1j * v * x)
    if kind == "two_solitons":
        v = cfg["v"]
        x0 = cfg["x0"]
        A = 1.0
        s1 = (A / np.cosh(A * (x - x0))) * np.exp(1j * v * x)
        s2 = (A / np.cosh(A * (x + x0))) * np.exp(-1j * v * x)
        return s1 + s2
    raise ValueError(kind)


    if kind == "two_solitons":
        # Two exact solitons each with A = 0.5 for g = -0.25 each; total g = -1
        # Simplest fix: use A = 0.5 per soliton and let them interact.
        A = 0.5
        v = cfg["v"]
        x0 = cfg["x0"]
        s1 = (A / np.cosh(A * (x - x0))) * np.exp(1j * v * x)
        s2 = (A / np.cosh(A * (x + x0))) * np.exp(-1j * v * x)
        return s1 + s2

    raise ValueError(kind)


def _normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    n = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / n if n > 0 else psi


def split_step(psi0, x, k, dt, n_steps, V, g, snapshot_every=100, save_all=False):
    psi = psi0.copy()
    kinetic_half = np.exp(-0.5j * (k ** 2) * dt)
    times = [0.0]
    snaps = [psi.copy()]
    for n in range(1, n_steps + 1):
        Veff = V + g * np.abs(psi) ** 2
        psi = psi * np.exp(-0.5j * Veff * dt)
        psi = np.fft.ifft(kinetic_half * np.fft.fft(psi))
        Veff = V + g * np.abs(psi) ** 2
        psi = psi * np.exp(-0.5j * Veff * dt)
        if save_all or n % snapshot_every == 0:
            times.append(n * dt)
            snaps.append(psi.copy())
    return np.array(times), np.array(snaps)


def reference_solution(cfg: dict):
    L = cfg["domain"]["L"]
    Nx = cfg["domain"]["Nx"]
    T = cfg["domain"]["T"]
    dt = cfg["domain"]["dt_ref"]
    g = cfg["physics"]["g"]
    omega = cfg["physics"]["omega"]
    pot = cfg["physics"]["potential"]

    x, k = make_grid(L, Nx)
    V = potential(x, pot, omega)
    psi0 = initial_condition(x, cfg["initial"])
    # Skip normalization for solitons: norm is 2A, stationarity requires it.
    if cfg["initial"]["kind"] not in ("soliton", "two_solitons"):
        psi0 = _normalize(psi0, x[1] - x[0])

    n_steps = int(round(T / dt))
    n_snap = 100
    snapshot_every = max(1, n_steps // n_snap)
    times, snaps = split_step(psi0, x, k, dt, n_steps, V, g,
                              snapshot_every=snapshot_every)
    return {"x": x, "t": times, "psi": snaps, "V": V, "g": g}
