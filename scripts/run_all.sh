#!/usr/bin/env bash
set -e
mkdir -p results/checkpoints results/tables results/figures

python tests/test_reference_convergence.py --case a

for cfg in configs/case_*.yaml; do
    echo "=== $cfg ==="
    python src/train.py --config "$cfg"
done

python src/evaluate.py --results_dir results/checkpoints \
                       --output results/tables/master_results.csv

python scripts/generate_failure_plots.py --cases d1_bright_soliton \
    d2_moving_soliton d3_two_soliton_collision e_long_time