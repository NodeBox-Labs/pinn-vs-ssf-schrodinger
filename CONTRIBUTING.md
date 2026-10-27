# Contributing to PINN vs SSF Benchmark

We welcome contributions that extend the failure analysis, improve the PINN, or add new baselines.

## The Golden Rule

**Do not cherry-pick.** If you submit a PR that improves the PINN, you must run the full 7-case benchmark with 3 seeds and include the auto-generated `master_results.csv`. We will not merge changes that only report successful cases.

## How to Submit a PR

1. Fork the repo and create a branch (`git checkout -b feat/improved-pinn`).
2. Run `make install` and `pre-commit install`.
3. Make your changes. Ensure `make test` passes.
4. Run the full benchmark: `bash scripts/run_all.sh`.
5. Commit your changes **including the updated `master_results.csv` and `README.md`**.
6. Open a Pull Request. In the description, state whether the failing cases (d1–d3, e) passed or failed, and why.

## Reporting a Failure

If you find a mathematical error in our solver or initial conditions, please open an issue with the tag `bug:physics`. Include the exact config file and the expected analytic solution.
