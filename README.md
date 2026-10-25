# PINN vs Split-Step Fourier for 1D Schrödinger / Gross-Pitaevskii

An honest, reproducible benchmark comparing Physics-Informed Neural Networks (PINNs)
against a Strang-split-step Fourier reference solver for

    i ∂ψ/∂t = [-½ ∂²/∂x² + V(x) + g|ψ|²] ψ,   ħ = m = 1

**We do not tune only the successful cases.** Solitons and long-time dynamics
are expected to *fail*, and we document the failure with error-vs-time plots,
loss curves, and spectral diagnostics.


## Install

```bash
conda env create -f environment.yml
conda activate pinn-ssf
```

or

```bash
pip install -r requirements.txt
```

## Reproduce

# 1. Verify the reference solver is second-order accurate
python tests/test_reference_convergence.py --case a

# 2. Run every case (successes AND failures)
bash scripts/run_all.sh

# 3. Build the master results table
python src/evaluate.py --results_dir results/checkpoints \
                       --output results/tables/master_results.csv

# 4. Regenerate failure diagnostics
python scripts/generate_failure_plots.py --cases d1 d2 d3 e

## Ablations (documented, not claimed as fixes)

# Fourier features on the bright soliton (expected: partial improvement, still fails)
python src/train.py --config configs/case_d1_bright_soliton.yaml \
    --override use_fourier=true fourier_scale=4.0

# Causal weighting on the long-time free packet (expected: slower failure, still fails)
python src/train.py --config configs/case_e_long_time.yaml \
    --override causal_weights=true

    