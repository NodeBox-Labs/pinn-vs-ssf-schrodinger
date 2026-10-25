from __future__ import annotations
import argparse, os, time, json, random, yaml
import numpy as np
import torch

from reference.split_step import reference_solution, potential
from pinn.model import PINN
from pinn.losses import pde_residual, ic_loss, bc_loss
from pinn.weighting import GradNorm, SelfAdaptive
from pinn.causal import CausalWeights
from metrics import rel_l2, norm_sq, energy


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
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main(config_path: str, overrides: dict | None = None):
    cfg = load_config(config_path)
    if overrides:
        # shallow merge overrides into nested dicts
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    set_seed(cfg["seed"])

    L = cfg["domain"]["L"]; T = cfg["domain"]["T"]
    g = cfg["physics"]["g"]; omega = cfg["physics"]["omega"]
    pot_kind = cfg["physics"]["potential"]

    device = cfg["device"]
    print(f"[{cfg['case']}] generating reference ...")
    ref = reference_solution(cfg)
    x_ref = ref["x"]; t_ref = ref["t"]; psi_ref = ref["psi"]
    V_ref = ref["V"]; dx = x_ref[1] - x_ref[0]

    # ---------- model ----------
    model = PINN(hidden=cfg["pinn"]["hidden"], layers=cfg["pinn"]["layers"],
                 activation=cfg["pinn"]["activation"],
                 use_fourier=cfg["pinn"]["use_fourier"],
                 fourier_scale=cfg["pinn"]["fourier_scale"],
                 fourier_dim=cfg["pinn"]["fourier_dim"],
                 seed=cfg["seed"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def V_fn(x):
        if pot_kind == "none":
            return torch.zeros_like(x)
        return 0.5 * omega ** 2 * x ** 2

    # ---------- training data ----------
    N_c = cfg["training"]["n_collocation"]
    x_c = (torch.rand(N_c, 1, device=device) - 0.5) * L
    t_c = torch.rand(N_c, 1, device=device) * T

    N_ic = cfg["training"]["n_ic"]
    x_ic = torch.linspace(-L/2, L/2, N_ic, device=device).unsqueeze(-1)
    psi0 = ref["psi"][0]
    # interpolate initial condition onto x_ic grid
    u0 = torch.tensor(np.interp(x_ic.cpu().numpy().squeeze(),
                                x_ref, psi0.real), dtype=torch.float32,
                      device=device).unsqueeze(-1)
    v0 = torch.tensor(np.interp(x_ic.cpu().numpy().squeeze(),
                                x_ref, psi0.imag), dtype=torch.float32,
                      device=device).unsqueeze(-1)

    N_bc = cfg["training"]["n_bc"]
    t_bc = torch.rand(N_bc, 1, device=device) * T

    # ---------- optimizer / weighting ----------
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["iters"],
        eta_min=cfg["training"]["lr_final"])

    weight_mode = cfg["training"]["weighting"]
    gradnorm = GradNorm(3) if weight_mode == "gradnorm" else None
    sa = SelfAdaptive(3) if weight_mode == "self_adaptive" else None
    if sa is not None:
        sa.log_s = torch.nn.Parameter(sa.log_s.to(device))
        opt = torch.optim.Adam(list(model.parameters()) + [sa.log_s],
                               lr=cfg["training"]["lr"])

    causal = CausalWeights(T) if cfg["training"]["causal_weights"] else None

    # ---------- training ----------
    print(f"[{cfg['case']}] training PINN ({n_params} params) ...")
    t_start = time.time()
    hist = {"iter": [], "loss": [], "pde": [], "ic": [], "bc": []}
    for it in range(1, cfg["training"]["iters"] + 1):
        Ru, Rv = pde_residual(model, x_c, t_c, V_fn, g)
        if causal is not None:
            w = causal(t_c)
            L_pde = ((Ru ** 2 + Rv ** 2) * w).mean()
        else:
            L_pde = (Ru ** 2 + Rv ** 2).mean()
        L_ic = ic_loss(model, x_ic, u0, v0)
        L_bc = bc_loss(model, t_bc, L)

        losses = [L_pde, L_ic, L_bc]
        if weight_mode == "fixed":
            total = (cfg["training"]["lambda_pde"] * L_pde
                     + cfg["training"]["lambda_ic"] * L_ic
                     + cfg["training"]["lambda_bc"] * L_bc)
        elif weight_mode == "gradnorm":
            w = gradnorm.update(losses, model).to(device)
            total = (w[0] * L_pde + w[1] * L_ic + w[2] * L_bc)
        else:  # self_adaptive
            total = sa.apply(losses)

        opt.zero_grad(); total.backward(); opt.step(); sched.step()
        if causal is not None and it % 10 == 0:
            causal.step()

        if it % cfg["training"]["log_every"] == 0 or it == 1:
            hist["iter"].append(it)
            hist["loss"].append(float(total))
            hist["pde"].append(float(L_pde))
            hist["ic"].append(float(L_ic))
            hist["bc"].append(float(L_bc))
            print(f"  it {it:6d}  total {float(total):.3e}  "
                  f"pde {float(L_pde):.3e}  ic {float(L_ic):.3e}  "
                  f"bc {float(L_bc):.3e}")
    train_time = time.time() - t_start

    # ---------- evaluation ----------
    print(f"[{cfg['case']}] evaluating ...")
    model.eval()
    t_inf0 = time.time()
    x_eval = torch.tensor(x_ref, dtype=torch.float32, device=device)
    preds = []
    for tt in t_ref:
        t_eval = torch.full_like(x_eval, float(tt))
        with torch.no_grad():
            u, v = model(x_eval.unsqueeze(-1), t_eval.unsqueeze(-1))
        preds.append((u.cpu().numpy().squeeze()
                      + 1j * v.cpu().numpy().squeeze()))
    preds = np.array(preds)
    inference_time = time.time() - t_inf0

    errs = [rel_l2(preds[i], psi_ref[i], dx) for i in range(len(t_ref))]
    norm_drift = [abs(norm_sq(preds[i], dx) - 1.0) for i in range(len(t_ref))]
    E_ref0 = energy(psi_ref[0], x_ref, np.fft.fftfreq(len(x_ref), dx) * 2 * np.pi,
                    V_ref, g, dx)
    E_pred = [energy(preds[i], x_ref,
                     np.fft.fftfreq(len(x_ref), dx) * 2 * np.pi,
                     V_ref, g, dx) for i in range(len(t_ref))]
    energy_drift = [abs(e - E_ref0) / max(abs(E_ref0), 1e-12) for e in E_pred]

    out = {
        "case": cfg["case"],
        "n_params": n_params,
        "train_time_s": train_time,
        "inference_time_s": inference_time,
        "rel_l2_final": errs[-1],
        "rel_l2_max": max(errs),
        "norm_drift_max": max(norm_drift),
        "energy_drift_final": energy_drift[-1],
        "energy_drift_max": max(energy_drift),
        "history": hist,
        "t": t_ref.tolist(),
        "rel_l2_t": errs,
        "norm_drift_t": norm_drift,
        "energy_drift_t": energy_drift,
    }

    ckpt_dir = cfg["output"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{cfg['case']}.pt")
    torch.save({"model_state": model.state_dict(),
                "config": cfg,
                "preds": preds,
                "ref": psi_ref,
                "metrics": out}, ckpt_path)
    with open(os.path.join(ckpt_dir, f"{cfg['case']}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[{cfg['case']}] saved {ckpt_path}")
    print(f"  rel-L2 final: {errs[-1]:.3e}")
    print(f"  norm drift max: {max(norm_drift):.3e}")
    print(f"  train: {train_time:.1f}s  inference: {inference_time:.3f}s")
    return out


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