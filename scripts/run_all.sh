#!/usr/bin/env bash
set -e
mkdir -p results/checkpoints results/tables results/figures

echo "=== 0. Solver validation ==="
python tests/test_reference_convergence.py --case a
python tests/test_solver_against_analytic.py

echo "=== 1. Convergence study ==="
python scripts/convergence_study.py

echo "=== 2. PINN benchmark (3 seeds) ==="
for cfg in configs/case_*.yaml; do
    for seed in 42 43 44; do
        python src/train.py --config "$cfg" --override seed=$seed
    done
done

echo "=== 3. Data-driven baseline ==="
for cfg in configs/case_*.yaml; do
    python src/baseline_data_driven.py --config "$cfg"
done

echo "=== 4. Aggregate results ==="
python src/evaluate.py \
    --results_dir results/checkpoints \
    --output results/tables/master_results.csv

echo "=== 5. Auto-generate README ==="
python scripts/generate_report.py

echo "=== 6. Generate failure plots ==="
python scripts/generate_failure_plots.py --cases d1_bright_soliton d2_moving_soliton d3_two_soliton_collision e_long_time

echo "✓ Pipeline complete."