
import argparse, glob, json, os
import pandas as pd


def main(results_dir: str, output: str):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh:
            m = json.load(fh)
        rows.append({
            "case": m["case"],
            "rel_l2_final": m["rel_l2_final"],
            "rel_l2_max": m["rel_l2_max"],
            "norm_drift_max": m["norm_drift_max"],
            "energy_drift_max": m["energy_drift_max"],
            "n_params": m["n_params"],
            "train_time_s": m["train_time_s"],
            "inference_time_s": m["inference_time_s"],
        })
    df = pd.DataFrame(rows)
    # verdict heuristic
    def verdict(r):
        if r["rel_l2_max"] > 0.2 or r["norm_drift_max"] > 5e-2:
            return "FAIL"
        if r["rel_l2_max"] > 5e-2:
            return "BORDERLINE"
        return "PASS"
    df["verdict"] = df.apply(verdict, axis=1)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/checkpoints")
    p.add_argument("--output", default="results/tables/master_results.csv")
    a = p.parse_args()
    main(a.results_dir, a.output)