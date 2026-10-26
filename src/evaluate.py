"""Build master results with seed sweeps."""
import argparse, glob, json, os
import numpy as np
import pandas as pd

<<<<<<< HEAD
def aggregate_seeds(results_dir, case):
    rows = []
    # Look for all seeds for this case
    for f in glob.glob(os.path.join(results_dir, f"{case}_seed*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows: return None
=======

def aggregate_seeds(results_dir, case):
    """Collect all JSONs matching case_seed*.json."""
    rows = []
    for f in glob.glob(os.path.join(results_dir, f"{case}_seed*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        return None
>>>>>>> 3beac0a4ac344e129a93c385751debf2fcbf07dc
    return {
        "case": case,
        "n_seeds": len(rows),
        "rel_l2_final_mean": float(np.mean([r["rel_l2_final"] for r in rows])),
        "rel_l2_final_std": float(np.std([r["rel_l2_final"] for r in rows])),
        "norm_drift_max_mean": float(np.mean([r["norm_drift_max"] for r in rows])),
<<<<<<< HEAD
        "energy_drift_max_mean": float(np.mean([r["energy_drift_max"] for r in rows])),
        "train_time_mean": float(np.mean([r["train_time_s"] for r in rows])),
    }
=======
        "norm_drift_max_std": float(np.std([r["norm_drift_max"] for r in rows])),
        "energy_drift_max_mean": float(np.mean([r["energy_drift_max"] for r in rows])),
        "energy_drift_max_std": float(np.std([r["energy_drift_max"] for r in rows])),
    }


def main(results_dir, output):
    cases = sorted(set(
        os.path.basename(f).split("_seed")[0]
        for f in glob.glob(os.path.join(results_dir, "*.json"))
    ))
    rows = []
    for c in cases:
        agg = aggregate_seeds(results_dir, c)
        if agg:
            rows.append(agg)
    df = pd.DataFrame(rows)

    # baseline: data-driven MLP (no physics) — load from baseline_results.json
    baseline_path = os.path.join(results_dir, "baseline_data_driven.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        df["data_mlp_rel_l2"] = df["case"].map(
            lambda c: baseline.get(c, {}).get("rel_l2_final", np.nan))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print("\nReported as mean ± std over seeds. "
          "Data-MLP column is the no-physics baseline.")
>>>>>>> 3beac0a4ac344e129a93c385751debf2fcbf07dc

def main(results_dir, output):
    # Find unique base cases
    cases = sorted(set(os.path.basename(f).split("_seed")[0] for f in glob.glob(os.path.join(results_dir, "*_seed*.json"))))
    
    rows = []
    for c in cases:
        agg = aggregate_seeds(results_dir, c)
        if agg: rows.append(agg)
        
    df = pd.DataFrame(rows)
    
    # Add a verdict column based on the mean error
    def verdict(r):
        if r["rel_l2_final_mean"] > 0.2 or r["norm_drift_max_mean"] > 5e-2: return "FAIL"
        if r["rel_l2_final_mean"] > 5e-2: return "BORDERLINE"
        return "PASS"
    
    if not df.empty:
        df["verdict"] = df.apply(verdict, axis=1)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)
        print(df.to_string(index=False))
        print(f"\nWrote aggregate table to {output}")
    else:
        print("No seed-swept checkpoints found. Did you run scripts/run_all.sh?")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/checkpoints")
    p.add_argument("--output", default="results/tables/master_results.csv")
    a = p.parse_args()
    main(a.results_dir, a.output)
