"""Data-driven MLP baseline: fit ψ(x,t) to reference snapshots, no physics loss."""
import argparse, json, os, sys, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from src.reference.split_step import reference_solution
from src.pinn.model import PINN
from src.metrics import rel_l2

def main(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    if "inherit" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop("inherit"))
        with open(base_path) as f: base = yaml.safe_load(f)
        base.update(cfg); cfg = base

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = cfg["device"]

    ref = reference_solution(cfg)
    x_ref, t_ref, psi_ref = ref["x"], ref["t"], ref["psi"]
    dx = x_ref[1] - x_ref[0]

    # Sample supervised training data from reference
    rng = np.random.default_rng(cfg["seed"])
    n_per_snap = 20
    xs, ts, us, vs = [], [], [], []
    for i, tt in enumerate(t_ref):
        idx = rng.choice(len(x_ref), n_per_snap, replace=False)
        xs.append(x_ref[idx]); ts.append(np.full(n_per_snap, tt))
        us.append(psi_ref[i, idx].real); vs.append(psi_ref[i, idx].imag)

    x_tr = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=device).unsqueeze(-1)
    t_tr = torch.tensor(np.concatenate(ts), dtype=torch.float32, device=device).unsqueeze(-1)
    u_tr = torch.tensor(np.concatenate(us), dtype=torch.float32, device=device).unsqueeze(-1)
    v_tr = torch.tensor(np.concatenate(vs), dtype=torch.float32, device=device).unsqueeze(-1)

    model = PINN(hidden=cfg["pinn"]["hidden"], layers=cfg["pinn"]["layers"],
                 activation=cfg["pinn"]["activation"], use_fourier=cfg["pinn"]["use_fourier"],
                 fourier_scale=cfg["pinn"]["fourier_scale"], fourier_dim=cfg["pinn"]["fourier_dim"],
                 seed=cfg["seed"]).to(device)
                 
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    
    print(f"[baseline] training data-driven MLP on {cfg['case']}...")
    for it in range(1, cfg["training"]["iters"] + 1):
        u, v = model(x_tr, t_tr)
        loss = ((u - u_tr) ** 2 + (v - v_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    x_eval = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(-1)
    preds = []
    for tt in t_ref:
        t_eval = torch.full_like(x_eval, float(tt))
        with torch.no_grad():
            u, v = model(x_eval, t_eval)
        preds.append((u.cpu().numpy().squeeze() + 1j * v.cpu().numpy().squeeze()))
    preds = np.array(preds)

    errs = [rel_l2(preds[i], psi_ref[i], dx) for i in range(len(t_ref))]
    out = {"case": cfg["case"], "rel_l2_final": errs[-1], "rel_l2_t": errs, "t": t_ref.tolist()}
    
    ckpt_dir = cfg["output"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{cfg['case']}_baseline.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    print(f"[baseline] {cfg['case']}: rel-L2 final = {errs[-1]:.3e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)