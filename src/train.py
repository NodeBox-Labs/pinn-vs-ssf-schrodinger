"""Train a PINN with Hard Periodic BCs and Soft IC loss."""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from metrics import energy, norm_sq, rel_l2
from pinn.losses import pde_residual
from pinn.model import HardBCPINN
from reference.split_step import reference_solution


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "inherit" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("inherit"))
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main(config_path: str, overrides: dict | None = None):
    cfg = load_config(config_path)
    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    set_seed(cfg["seed"])

    L = cfg["domain"]["L"]
    T = cfg["domain"]["T"]
    g = cfg["physics"]["g"]
    omega = cfg["physics"]["omega"]
    pot_kind = cfg["physics"]["potential"]
    device = cfg["device"]

    print(f"[{cfg['case']}] generating reference ...")
    ref = reference_solution(cfg)
    x_ref, t_ref, psi_ref = ref["x"], ref["t"], ref["psi"]
    V_ref, dx = ref["V"], x_ref[1] - x_ref[0]

    # ---------- model ----------
    model = HardBCPINN(
        L=L,
        hidden=cfg["pinn"]["hidden"],
        layers=cfg["pinn"]["layers"],
        activation=cfg["pinn"]["activation"],
        use_fourier=cfg["pinn"]["use_fourier"],
        fourier_scale=cfg["pinn"]["fourier_scale"],
        fourier_dim=cfg["pinn"]["fourier_dim"],
        seed=cfg["seed"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def V_fn(x):
        if pot_kind == "none":
            return torch.zeros_like(x)
        return 0.5 * omega ** 2 * x ** 2

    # Prepare fixed IC data (supervised points at t=0)
    N_ic = 256
    x_ic_np = np.linspace(-L/2, L/2, N_ic)
    u0_np = np.interp(x_ic_np, x_ref, psi_ref[0].real)
    v0_np = np.interp(x_ic_np, x_ref, psi_ref[0].imag)

    x_ic = torch.tensor(x_ic_np, dtype=torch.float32, device=device).unsqueeze(-1)
    t_ic = torch.zeros_like(x_ic)
    u0 = torch.tensor(u0_np, dtype=torch.float32, device=device).unsqueeze(-1)
    v0 = torch.tensor(v0_np, dtype=torch.float32, device=device).unsqueeze(-1)

    print(f"[{cfg['case']}] training PINN ({n_params} params) ...")
    t_start = time.time()

    # ==========================================
    # PHASE 1: Adam
    # ==========================================
    print("  Phase 1: Adam")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000, eta_min=1e-4)

    lambda_ic = 100.0  # Strong penalty for IC

    for it in range(1, 5001):
        batch_size = 1024
        x_c = (torch.rand(batch_size, 1, device=device) - 0.5) * L
        t_c = torch.rand(batch_size, 1, device=device) * T

        # 1. PDE residual
        Ru, Rv = pde_residual(model, x_c, t_c, V_fn, g)
        L_pde = (Ru**2 + Rv**2).mean()

        # 2. IC loss (Soft)
        u_ic, v_ic = model(x_ic, t_ic)
        L_ic = ((u_ic - u0)**2 + (v_ic - v0)**2).mean()

        loss = L_pde + lambda_ic * L_ic

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if it % 1000 == 0:
            print(f"    Adam it {it:5d} | PDE Loss: {L_pde.item():.3e} | IC Loss: {L_ic.item():.3e}")

    # ==========================================
    # PHASE 2: L-BFGS
    # ==========================================
    print("  Phase 2: L-BFGS")
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500,
                              history_size=50, line_search_fn="strong_wolfe")

    # Fixed collocation set for L-BFGS
    N_lbfgs = 5000
    x_l = (torch.rand(N_lbfgs, 1, device=device) - 0.5) * L
    t_l = torch.rand(N_lbfgs, 1, device=device) * T

    def closure():
        lbfgs.zero_grad()
        Ru, Rv = pde_residual(model, x_l, t_l, V_fn, g)
        L_pde = (Ru**2 + Rv**2).mean()

        u_ic_l, v_ic_l = model(x_ic, t_ic)
        L_ic = ((u_ic_l - u0)**2 + (v_ic_l - v0)**2).mean()

        loss = L_pde + lambda_ic * L_ic
        loss.backward()
        return loss

    lbfgs.step(closure)
    final_loss = closure().item()
    print(f"    L-BFGS final Total Loss: {final_loss:.3e}")

    train_time = time.time() - t_start

    # ---------- evaluation ----------
    print(f"[{cfg['case']}] evaluating ...")
    model.eval()
    t_inf0 = time.time()
    x_eval = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(-1)
    preds = []
    for tt in t_ref:
        t_eval = torch.full_like(x_eval, float(tt))
        with torch.no_grad():
            u, v = model(x_eval, t_eval)
        preds.append(u.cpu().numpy().squeeze() + 1j * v.cpu().numpy().squeeze())
    preds = np.array(preds)
    inference_time = time.time() - t_inf0

    errs = [rel_l2(preds[i], psi_ref[i], dx) for i in range(len(t_ref))]

    expected_norm = 2.0 if "soliton" in cfg["initial"]["kind"] else 1.0
    norm_drift = [abs(norm_sq(preds[i], dx) - expected_norm) / expected_norm for i in range(len(t_ref))]

    E_ref0 = energy(psi_ref[0], x_ref, np.fft.fftfreq(len(x_ref), dx) * 2 * np.pi, V_ref, g, dx)
    E_pred = [energy(preds[i], x_ref, np.fft.fftfreq(len(x_ref), dx) * 2 * np.pi, V_ref, g, dx) for i in range(len(t_ref))]
    energy_drift = [abs(e - E_ref0) / max(abs(E_ref0), 1e-12) for e in E_pred]

    out = {
        "case": cfg["case"], "n_params": n_params,
        "train_time_s": train_time, "inference_time_s": inference_time,
        "rel_l2_final": errs[-1], "rel_l2_max": max(errs),
        "norm_drift_max": max(norm_drift), "energy_drift_final": energy_drift[-1],
        "energy_drift_max": max(energy_drift),
        "t": t_ref.tolist(), "rel_l2_t": errs, "norm_drift_t": norm_drift, "energy_drift_t": energy_drift,
    }

    ckpt_dir = cfg["output"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{cfg['case']}_seed{cfg['seed']}.pt")

    torch.save({"model_state": model.state_dict(), "config": cfg, "preds": preds, "ref": psi_ref, "metrics": out}, ckpt_path)
    with open(os.path.join(ckpt_dir, f"{cfg['case']}_seed{cfg['seed']}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[{cfg['case']}] saved {ckpt_path} | Rel-L2: {errs[-1]:.3e}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()

def _parse_overrides(items):
    out = {}
    for s in items:
        k, v = s.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        out[k] = v
    return out

if __name__ == "__main__":
    args = parse_args()
    main(args.config, _parse_overrides(args.override))
