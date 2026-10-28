"""Build the master results table from all checkpoint JSONs and update README."""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def aggregate_seeds(results_dir, case):
    """Collect all JSONs matching `case_seed*.json` and return mean/std metrics."""
    rows = []
    for f in glob.glob(os.path.join(results_dir, f"{case}_seed*.json")):
        with open(f) as fh:
            rows.append(json.load(fh))
    if not rows:
        return None

    return {
        "case": case,
        "n_seeds": len(rows),
        "rel_l2_final_mean": float(np.mean([r["rel_l2_final"] for r in rows])),
        "rel_l2_final_std": float(np.std([r["rel_l2_final"] for r in rows])),
        "norm_drift_max_mean": float(np.mean([r["norm_drift_max"] for r in rows])),
        "norm_drift_max_std": float(np.std([r["norm_drift_max"] for r in rows])),
        "energy_drift_max_mean": float(np.mean([r["energy_drift_max"] for r in rows])),
        "energy_drift_max_std": float(np.std([r["energy_drift_max"] for r in rows])),
        "train_time_mean": float(np.mean([r["train_time_s"] for r in rows])),
    }


def verdict(row):
    """Classify the outcome based on mean error and norm drift."""
    if row["rel_l2_final_mean"] > 0.2 or row["norm_drift_max_mean"] > 5e-2:
        return "FAIL"
    if row["rel_l2_final_mean"] > 5e-2:
        return "BORDERLINE"
    return "PASS"


def update_readme_table(csv_path, readme_path):
    """Replace the auto-generated results block inside README.md."""
    if not os.path.exists(csv_path):
        print(f"[warn] {csv_path} not found — skipping README update.")
        return
    if not os.path.exists(readme_path):
        print(f"[warn] {readme_path} not found — skipping README update.")
        return

    df = pd.read_csv(csv_path)
    try:
        md_table = df.to_markdown(index=False)
    except ImportError:
        # Fallback if `tabulate` is not installed
        md_table = df.to_string(index=False)

        with open(readme_path, encoding="utf-8") as f:
            content = f.read()

    start_marker = "<!-- AUTO-GENERATED-RESULTS-START -->"
    end_marker = "<!-- AUTO-GENERATED-RESULTS-END -->"

    if start_marker not in content:
        print(f"[warn] {start_marker} not found in README — skipping update.")
        return

    new_content = re.sub(
        f"{start_marker}.*?{end_marker}",
        f"{start_marker}\n\n{md_table}\n\n{end_marker}",
        content,
        flags=re.DOTALL,
    )

    with open(readme_path, "w") as f:
        f.write(new_content)
    print(f"[ok] README results table updated in {readme_path}")


def main(results_dir, output, readme_path=None):
    # --- 1. Aggregate PINN results over seeds ---
    cases = sorted(set(
        os.path.basename(f).split("_seed")[0]
        for f in glob.glob(os.path.join(results_dir, "*_seed*.json"))
    ))

    rows = []
    for c in cases:
        agg = aggregate_seeds(results_dir, c)
        if agg:
            rows.append(agg)

    if not rows:
        print("No seed-swept checkpoints found. Did you run scripts/run_all.sh?")
        return

    df = pd.DataFrame(rows)

    # --- 2. Attach data-driven baseline column (no physics) ---
    baseline_path = os.path.join(results_dir, "baseline_data_driven.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        df["data_mlp_rel_l2"] = df["case"].map(
            lambda c: baseline.get(c, {}).get("rel_l2_final", np.nan)
        )
        print(f"[ok] Attached data-driven baseline from {baseline_path}")
    else:
        print(f"[note] No baseline found at {baseline_path} (optional).")

    # --- 3. Verdict column ---
    df["verdict"] = df.apply(verdict, axis=1)

    # --- 4. Write CSV ---
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print(f"\n[ok] Wrote aggregate table to {output}")
    print("Reported as mean ± std over seeds. "
          "Data-MLP column is the no-physics baseline.")

    # --- 5. Update README (if requested) ---
    if readme_path:
        update_readme_table(output, readme_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/checkpoints")
    p.add_argument("--output", default="results/tables/master_results.csv")
    p.add_argument("--readme", default=None,
                   help="Optional path to README.md to auto-update.")
    a = p.parse_args()
    main(a.results_dir, a.output, a.readme)
