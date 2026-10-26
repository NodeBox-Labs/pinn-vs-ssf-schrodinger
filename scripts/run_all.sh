#!/usr/bin/env bash
set -e
mkdir -p results/checkpoints results/tables results/figures

echo "=== 0. Validating Reference Solver ==="
python tests/test_reference_convergence.py --case a

echo "=== 1. Running PINN Benchmark (3 Seeds) ==="
for cfg in configs/case_*.yaml; do
    for seed in 42 43 44; do
        echo "-> Training $cfg (seed $seed)"
        python src/train.py --config "$cfg" --override seed=$seed
    done
done

echo "=== 2. Running Data-Driven Baseline (No Physics) ==="
for cfg in configs/case_*.yaml; do
    echo "-> Baseline $cfg"
    python src/baseline_data_driven.py --config "$cfg"
done

echo "=== 3. Aggregating Results ==="
python src/evaluate.py --results_dir results/checkpoints --output results/tables/master_results.csv

echo "=== 4. Generating Failure Plots ==="
python scripts/generate_failure_plots.py --cases d1_bright_soliton d2_moving_soliton d3_two_soliton_collision e_long_time

echo "All done! Check results/tables/master_results.csv"