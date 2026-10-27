import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.reference.split_step import (make_grid, potential, initial_condition,
                                      _normalize, split_step)
from src.reference.yoshida import split_step_yoshida
from src.reference.analytic import free_gaussian


def run_ssf(dt, Nx, T=2.0, L=20.0, order=2):
    x, k = make_grid(L, Nx)
    dx = x[1] - x[0]
    psi0 = _normalize(free_gaussian(x, 0.0), dx)
    n = int(round(T / dt))
    t0 = time.time()
    if order == 2:
        _, snaps = split_step(psi0, x, k, dt, n, np.zeros_like(x), 0.0,
                              snapshot_every=n)
    else:
        _, snaps = split_step_yoshida(psi0, x, k, dt, n, np.zeros_like(x), 0.0,
                                       snapshot_every=n)
    wall = time.time() - t0
    psi_num = snaps[-1]
    psi_exact = _normalize(free_gaussian(x, T), dx)
    err = np.sqrt(np.sum(np.abs(psi_num - psi_exact) ** 2) * dx)
    return err, wall


def main():
    outdir = "results/figures/convergence"
    os.makedirs(outdir, exist_ok=True)
    Nx = 512

    # --- error vs dt ---
    dts = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3])
    errs_s, errs_y, walls_s, walls_y = [], [], [], []
    for dt in dts:
        e2, w2 = run_ssf(dt, Nx, order=2)
        e4, w4 = run_ssf(dt, Nx, order=4)
        errs_s.append(e2); errs_y.append(e4)
        walls_s.append(w2); walls_y.append(w4)
        print(f"dt={dt:.2e}  Strang={e2:.2e}  Yoshida={e4:.2e}")

    plt.figure()
    plt.loglog(dts, errs_s, "-o", label="Strang (O(dt²))")
    plt.loglog(dts, errs_y, "-s", label="Yoshida (O(dt⁴))")
    plt.xlabel("Δt"); plt.ylabel("Rel-L2 error")
    plt.title("Convergence vs time step (Nx=512)")
    plt.legend(); plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_dt.png"), dpi=140)
    plt.close()

    # --- error vs Nx ---
    Nxs = np.array([128, 256, 512, 1024, 2048])
    errs_s, errs_y = [], []
    for nx in Nxs:
        e2, _ = run_ssf(2.5e-3, nx, order=2)
        e4, _ = run_ssf(2.5e-3, nx, order=4)
        errs_s.append(e2); errs_y.append(e4)
        print(f"Nx={nx}  Strang={e2:.2e}  Yoshida={e4:.2e}")

    plt.figure()
    plt.semilogy(Nxs, errs_s, "-o", label="Strang")
    plt.semilogy(Nxs, errs_y, "-s", label="Yoshida")
    plt.xlabel("Nx"); plt.ylabel("Rel-L2 error")
    plt.title("Spectral convergence vs Nx (Δt=2.5e-3)")
    plt.legend(); plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_Nx.png"), dpi=140)
    plt.close()

    # --- error vs walltime ---
    plt.figure()
    plt.loglog(walls_s, errs_s, "-o", label="Strang")
    plt.loglog(walls_y, errs_y, "-s", label="Yoshida")
    plt.xlabel("Wall-clock time (s)"); plt.ylabel("Rel-L2 error")
    plt.title("Error vs wall time")
    plt.legend(); plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_walltime.png"), dpi=140)
    plt.close()

    print(f"\nWrote convergence plots to {outdir}")


if __name__ == "__main__":
    main()