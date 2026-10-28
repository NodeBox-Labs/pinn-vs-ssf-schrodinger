import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import spectral_error


def load_case(results_dir, name):
    import torch
    import glob
    # Look for the first matching seed file
    paths = glob.glob(os.path.join(results_dir, f"{name}_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoint found for {name}")
    path = sorted(paths)[0]  # Sort to pick seed 42 first if it exists
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck

def plot_case(ck, outdir):
    os.makedirs(outdir, exist_ok=True)
    m = ck["metrics"]; t = np.array(m["t"])
    ref = ck["ref"]; pred = ck["preds"]
    x = ck["config"]["domain"]["L"]
    dx = x / len(ref[0])

    # 1. error vs time
    plt.figure()
    plt.semilogy(t, m["rel_l2_t"], "-o", ms=3)
    plt.xlabel("t"); plt.ylabel("rel-L2"); plt.title(f"{m['case']}: rel-L2 vs t")
    plt.grid(True, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error_vs_time.png"), dpi=140)
    plt.close()

    # 2. loss curves
    h = m["history"]
    plt.figure()
    for key in ["loss", "pde", "ic", "bc"]:
        plt.semilogy(h["iter"], h[key], label=key)
    plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend()
    plt.title(f"{m['case']}: losses"); plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=140)
    plt.close()

    # 3. spectral error at final time
    err = pred[-1] - ref[-1]
    k, P = spectral_error(err, dx)
    plt.figure()
    plt.semilogy(np.fft.fftshift(np.fft.fftfreq(len(err), dx) * 2 * np.pi), P)
    plt.xlabel("k"); plt.ylabel("|FFT(err)|²")
    plt.title(f"{m['case']}: spectral error (t=T)")
    plt.grid(True, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectral_error.png"), dpi=140)
    plt.close()

    # 4. density snapshot at T
    xg = np.linspace(-x/2, x/2, len(ref[0]), endpoint=False)
    plt.figure()
    plt.plot(xg, np.abs(ref[-1])**2, label="ref", lw=2)
    plt.plot(xg, np.abs(pred[-1])**2, "--", label="PINN", lw=2)
    plt.xlabel("x"); plt.ylabel("|ψ|²"); plt.legend()
    plt.title(f"{m['case']}: |ψ|² at t=T")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "density_T.png"), dpi=140)
    plt.close()

    # 5. norm and energy drift
    fig, ax1 = plt.subplots()
    ax1.semilogy(t, m["norm_drift_t"], "C0", label="norm drift")
    ax1.set_xlabel("t"); ax1.set_ylabel("|‖ψ‖²-1|", color="C0")
    ax2 = ax1.twinx()
    ax2.semilogy(t, m["energy_drift_t"], "C1", label="energy drift")
    ax2.set_ylabel("|ΔE/E₀|", color="C1")
    ax1.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "norm_energy_drift.png"), dpi=140)
    plt.close()


def main(results_dir, cases, outroot):
    for c in cases:
        try:
            ck = load_case(results_dir, c)
        except FileNotFoundError:
            print(f"[skip] {c}: no checkpoint")
            continue
        outdir = os.path.join(outroot, c)
        plot_case(ck, outdir)
        print(f"wrote {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/checkpoints")
    p.add_argument("--cases", nargs="+", default=[
        "a_free_gaussian",
        "b_harmonic_coherent",
        "c_gpe_harmonic",
        "d1_bright_soliton",
        "d2_moving_soliton",
        "d3_two_soliton_collision",
        "e_long_time"
    ])
    p.add_argument("--outroot", default="results/figures")
    a = p.parse_args()
    main(a.results_dir, a.cases, a.outroot)
