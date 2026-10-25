#!/usr/bin/env bash
set -e
mkdir -p results/checkpoints results/tables results/figures

# 0. solver validation — MUST pass before any training
python tests/test_reference_convergence.py --case a
python tests/test_solver_against_analytic.py

# 1. run every case with 3 seeds
for cfg in configs/case_*.yaml; do
    for seed in 42 43 44; do
        python src/train.py --config "$cfg" \
            --override seed=$seed \
            --override output.ckpt_dir=results/checkpoints
        # rename checkpoint to include seed
        base=$(basename "$cfg" .yaml)
        mv "results/checkpoints/${base}.pt" "results/checkpoints/${base}_seed${seed}.pt"
        mv "results/checkpoints/${base}.json" "results/checkpoints/${base}_seed${seed}.json"
    done
done

# 2. data-driven baseline (no physics)
for cfg in configs/case_*.yaml; do
    python src/baseline_data_driven.py --config "$cfg"
done

# 3. aggregate
python src/evaluate.py --results_dir results/checkpoints \
                       --output results/tables/master_results.csv

# 4. failure plots for the worst cases
python scripts/generate_failure_plots.py --cases d1_bright_soliton \
    d2_moving_soliton d3_two_soliton_collision e_long_time
