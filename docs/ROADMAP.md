# Roadmap

What this repository could grow into, prioritized by scientific value.

## Near-term (3 months)

- [ ] **2D GPE** on a disk / annulus. SSF becomes memory-bound;
      PINNs may become competitive on irregular domains.
- [ ] **Vortex dynamics** in a 2D rotating BEC. The PINN would need
      to be trained on a rotating-frame GPE with angular momentum.
- [ ] **Inverse problems.** Recover `V(x)` or `g(x)` from partial
      observations of ψ. This is where PINNs are genuinely useful.

## Medium-term (6–12 months)

- [ ] **Physics-informed neural operators (PINO / FNO).** Train a
      Fourier Neural Operator to map initial conditions to final
      states. Benchmark against our SSF solver for many ICs.
- [ ] **Hybrid solvers.** Use the SSF solver in smooth regions and the
      PINN only in regions with high-frequency structure.
- [ ] **Dimensional scaling.** Test whether PINNs scale to 3D GPE
      (numerically infeasible for SSF at Nx=1024³).

## Long-term (research paper)

- [ ] Submit to *Journal of Computational Physics* or *Machine Learning:
      Science and Technology*.
- [ ] Include a full "threats to validity" section (see LIMITATIONS.md).
- [ ] Release a 2D/3D extension with a rotating-frame BEC dataset.

## Paper outline (arXiv / workshop)

**Title.** "When Do PINNs Beat Spectral Solvers? A Rigorous Benchmark
on the 1D Gross-Pitaevskii Equation."

1. **Introduction.** PINNs are popular; are they actually better?
2. **Setup.** SSF solver, PINN architectures, 7 test cases, seed sweep.
3. **Results.** PINNs succeed on smooth cases, fail on solitons and
   long horizons. Quantified: 10⁵× worse error, 10³× slower.
4. **Ablations.** Fourier features, causal weights, hard constraints —
   none rescue the failing cases at the tested budget.
5. **Why?** Spectral bias, phase accumulation, non-convex residual.
6. **Implications.** PINNs are useful for inverse problems and
   irregular geometry, not for spectral benchmarks in 1D.
7. **Reproducibility.** All code, configs, seeds in the public repo.