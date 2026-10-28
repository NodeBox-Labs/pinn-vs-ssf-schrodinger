"""Windows-native equivalent of scripts/run_all.sh."""
import subprocess
import sys
import os
import glob

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    # 0. Create output directories
    for d in ["results/checkpoints", "results/tables", "results/figures"]:
        os.makedirs(d, exist_ok=True)

    # 1. Solver validation
    print("\n=== 0. Solver validation ===")
    run([sys.executable, "tests/test_reference_convergence.py", "--case", "a"])
    run([sys.executable, "tests/test_solver_against_analytic.py"])

    # 2. Convergence study (optional; failure here is not fatal)
    print("\n=== 1. Convergence study ===")
    ok = run([sys.executable, "scripts/convergence_study.py"])
    if not ok:
        print("[WARN] Convergence study failed. Continuing with benchmark.")

    # 3. PINN benchmark — 3 seeds × 7 cases
    print("\n=== 2. PINN benchmark (3 seeds) ===")
    for cfg in sorted(glob.glob("configs/case_*.yaml")):
        for seed in [42, 43, 44]:
            print(f"\n--- {cfg} (seed {seed}) ---")
            run([
                sys.executable, "src/train.py",
                "--config", cfg,
                "--override", f"seed={seed}",
            ])

    # 4. Data-driven baseline
    print("\n=== 3. Data-driven baseline ===")
    for cfg in sorted(glob.glob("configs/case_*.yaml")):
        run([sys.executable, "src/baseline_data_driven.py", "--config", cfg])

    # 5. Aggregate results
    print("\n=== 4. Aggregate results ===")
    run([
        sys.executable, "src/evaluate.py",
        "--results_dir", "results/checkpoints",
        "--output", "results/tables/master_results.csv",
        "--readme", "README.md",
    ])

    # 6. Failure plots
    print("\n=== 5. Failure plots ===")
    run([
        sys.executable, "scripts/generate_failure_plots.py",
        "--cases", "d1_bright_soliton", "d2_moving_soliton",
                   "d3_two_soliton_collision", "e_long_time",
    ])

    print("\n✓ Pipeline complete.")

if __name__ == "__main__":
    main()