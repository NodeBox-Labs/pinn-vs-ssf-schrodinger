# Reproducibility Binder

This document records the exact conditions under which the benchmark results
in `results/tables/master_results.csv` were produced.

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