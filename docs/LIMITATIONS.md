# Limitations and Threats to Validity

This document is written to be read by a skeptical reviewer. Every claim in
the README is contextualized here.

## 1. Scope

- **1D only.** The benchmark does not address 2D or 3D GPE, where PINNs
  may be more competitive because classical spectral solvers become
  memory-bound.
- **Periodic boundaries.** Non-periodic BCs (Dirichlet, absorbing) are
  not tested and may change the PINN/SSF trade-off significantly.
- **Fixed time horizons.** Case (e) is 10× longer than (a) but still
  bounded. Longer times will worsen PINN failure modes.

## 2. Baseline fairness

- **PINN hyperparameters are not exhaustively tuned.** We sweep
  (network size, Fourier scale, causal weights, L-BFGS) but do not
  perform a full Bayesian optimization. A hostile reviewer could claim
  the PINN "wasn't given a fair chance."
- **SSF is at machine precision.** The SSF solver reaches 1e-8 relative
  error at ~0.05s. Any PINN result is compared against this. Small PINN
  errors (1e-3) are still 5 orders of magnitude above SSF accuracy.
- **Data-driven baseline.** The "no-physics MLP" baseline is trained on
  sparse reference snapshots. If the reference solver is wrong, both the
  PINN and the baseline inherit the error.

## 3. Statistical validity

- **Seeds 42/43/44 only.** Three seeds provide a rough estimate of
  variance but not a tight confidence interval. Reported std values
  should be read as lower bounds on real variance.
- **Single hardware class.** All training runs were on CPU (Intel i7)
  and one GPU (RTX 3060). Results on other hardware may shift by 5–10%
  due to floating-point non-associativity, though verdicts (PASS/FAIL)
  are stable.

## 4. Theoretical claims

- **"PINNs fail on solitons."** This is true *for the tested
  architecture and training budget*. Hard-constraint PINNs with L-BFGS
  may succeed. The README does not claim a fundamental impossibility.
- **"Spectral bias causes the failure."** The spectral-error plots are
  *consistent with* spectral bias but do not prove it is the sole
  cause. The optimizer's local-minima behavior and phase accumulation
  also contribute.

## 5. Reproducibility caveats

- **Checkpoints are ignored.** Because of `.gitignore`, the raw
  `.pt` files are not committed. Re-running the benchmark takes
  ~2.5 hours on CPU.
- **Package versions are pinned in `requirements.txt` but not hashed.**
  Future installs may resolve slightly differently. A full lock file
  (`pip freeze > requirements.lock`) is recommended for archival."" 
