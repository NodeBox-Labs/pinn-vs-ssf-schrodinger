import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.reference.split_step import (make_grid, potential, initial_condition,
                                     _normalize, split_step)

CFG = {
    "domain": {"L": 20.0, "Nx": 512, "T": 2.0},
    "physics": {"g": 0.0, "omega": 0.0, "potential": "none"},
    "initial": {"kind": "free_gaussian", "x0": 0.0, "k0": 0.0, "width": 1.0},
}

def run(dt):
    L = CFG["domain"]["L"]; Nx = CFG["domain"]["Nx"]; T = CFG["domain"]["T"]
    x, k = make_grid(L, Nx)
    V = potential(x, CFG["physics"]["potential"], CFG["physics"]["omega"])
    psi0 = _normalize(initial_condition(x, CFG["initial"]), x[1]-x[0])
    n = int(round(T/dt))
    _, snaps = split_step(psi0, x, k, dt, n, V, CFG["physics"]["g"],
                          snapshot_every=n)
    return snaps[-1]

def main(case="a"):
    dts = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    sols = [run(dt) for dt in dts]
    final = sols[-1]
    errs = [np.sqrt(np.sum(np.abs(s-final)**2)/np.sum(np.abs(final)**2))
            for s in sols]
    orders = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    print("dt:", dts)
    print("errors:", errs)
    print("orders:", orders)
    print(f"Strang splitting convergence: error(Δt) = {errs[0]:.2e}, "
          f"observed order ≈ {np.mean(orders):.2f}")
    assert np.mean(orders) > 1.5, "solver is not second-order!"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="a")
    main(p.parse_args().case)