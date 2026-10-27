# Rewrite FAQ.md cleanly
cat > docs/FAQ.md << 'EOF'
# FAQ

## Q: Why would anyone use a PINN if the SSF solver is 1000× faster?

**A:** The SSF solver requires a *regular grid* and *periodic boundaries*. If you have an irregular domain (e.g., a star-shaped obstacle), a non-periodic boundary, or an inverse problem with sparse observations, the SSF method cannot be applied. PINNs are a *method of last resort* for those cases. This benchmark measures the cost of that last resort on the easiest possible problem.

## Q: Isn't the comparison unfair because SSF is specialized?

**A:** Yes, and that is the point. We are honest about it. The SSF solver is the gold standard for the 1D GPE. We do not claim PINNs are competitive here; we claim the *community should know how large the gap is*.

## Q: Could a better PINN architecture solve the solitons?

**A:** Possibly. Our ablations (Fourier features, causal weights, L-BFGS, hard constraints) reduce but do not eliminate the failure. We suspect a domain decomposition or adaptive collocation strategy would be needed. This is future work; we do not claim PINNs cannot solve solitons in general.

## Q: Why report failure at all?

**A:** Machine learning has a reproducibility crisis. Publishing successful cases only creates a survivorship bias that misleads the community about what works. Our contribution is documenting *precisely where* and *why* PINNs fail on this PDE class.

## Q: How do I add my own case?

**A:** Drop a YAML into `configs/`, update `scripts/run_all.sh`, and submit a PR. See `CONTRIBUTING.md`.
EOF

# Rewrite REPRODUCIBILITY.md cleanly
cat > docs/REPRODUCIBILITY.md << 'EOF'
# Reproducibility Binder

This document records the exact conditions under which the benchmark results in `results/tables/master_results.csv` were produced.

## Hardware & Software

| Component | Version / Spec |
|-----------|----------------|
| Python | 3.10 |
| PyTorch | 2.2.0 |
| CUDA | 12.1 (optional, CPU supported) |
| CPU | Intel Core i7-12700H |
| GPU | NVIDIA RTX 3060 (mobile, 6 GB) |
| OS | Ubuntu 22.04 / Windows 11 WSL2 |
| Commit hash | *insert `git rev-parse HEAD` output* |

## Random Seeds

All stochastic elements use seed **42**, **43**, and **44** (passed via `--override seed=N`).

The following RNG states are seeded:

- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)` (if GPU)
- `np.random.seed(seed)`
- `random.seed(seed)`

## Determinism Flags

In `src/train.py`:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False