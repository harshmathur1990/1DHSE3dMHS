# Equilibrium adjoint, Phases 4–6 (Poisson modes, API, cost) — 24 Nov 2026

Branch `equilibrium-adjoint`. Plan: `design/adjoint/`. Phases 1–3: `adjoint_phase1-3.md`.

## Poisson modes

| Case | 1 rank | 3 ranks |
| --- | --- | --- |
| mhs-poisson-z (30 Gauss–Newton steps + polish) | 6.4e-8 | 3.5e-8 |
| mhs-poisson-ltau (tight solve) | 1.5e-9 | 1.3e-9 |
| mhs-poisson-ltau μ 0.8 inclined | 5.8e-9 | 5.8e-9 |
| mhs-poisson-ltau block preconditioner | 2.3e-9 | 2.0e-9 |
| mhs-poisson-logm (tight solve) | 1.3e-7 | 1.1e-7 |
| mhs-poisson-ltau default solver + polish | 3.0e-8 | 1.4e-9 |
| mhs-poisson-logm default solver + polish | 3.5e-9 | 3.9e-9 |

The dot-product check against finite differences of `solve_embedded!` (Richardson, best over h). Target 1e-5. The τ-grid
residual's reverse pass alone (`resid_vjp`) agrees with differences of Φ to < 2e-7 in every input on both axes and both
geometries.

## What was found on the way (deviations from the plan)

- **The Gauss–Newton solves do not reach stationarity.** On the height-grid fixture, `solve_pressure` after 80, 300 and
  1000 steps gives finite differences 4.4, 7.8 and 10.0 against the adjoint's 10.2–10.3 (‖G‖ hardly falls). Its steps
  are damped (`ereg` on every slot, the anchor penalty) and its decrement is measured with that damping, so slow
  directions stay unconverged while the decrement looks small; the anchor mean drifts by 5e-3 in ln P (the design says
  it is held fixed). New option `polish = n` (both Poisson option types, default 0): Newton steps with the exact
  Hessian after the solve — 4 steps to a decrement of 1e-25 here — which make the returned state the one the adjoint
  differentiates. `adj.status` reports the undamped Newton decrement, not the solver's damped one.
- **Null directions of Φ in the cut heights (τ and mass axes).** Besides the uniform shift (removed by the gauge), the
  Nyquist modes of the centred horizontal differences ((−1)^y on the 11 × 12 fixture) leave Φ exactly unchanged; the
  solve keeps the start's value, and the output heights carry it. The adjoint projects them out of its solve and passes
  their cotangent to the start's cut heights.
- **Preconditioning.** Without damping the Hessian's curvature along c is ~1e10 below that along ln P per unit; the
  solver's preconditioners (built for its damped operator, `ereg` on the c slot) made CG break down. The adjoint builds
  its own block-Jacobi preconditioner: the exact Gauss–Newton column blocks with no damping on ln P and the measured
  curvature on c.
- **The τ polish weights** must be formed at the true start: `solve_tau` updates its start ln P in place (fixed).
- Hessian–vector and mixed products on the τ grid use a relative difference step of 1e-5 (`PoissonTau.HVP_STEP`); 1e-7
  was dominated by round-off along c.

## Cost (one rank, 64 threads, hyades5, 252² columns)

| Mode | solve | recorded solve | adjoint | adjoint / solve |
| --- | --- | --- | --- | --- |
| column-hse-ltau (96 levels, FALC start + perturbation) | 0.37 s | 1.24 s | 0.60 s | 1.6 |
| column-hse-logm (96 levels) | 0.36 s | 0.79 s | 0.63 s | 1.8 |
| column-hse-z (177 levels, tiled fixture) | 0.81 s | 1.74 s | 2.51 s | 3.1 |
| mhs-poisson-ltau (95 levels, polish 3) | 631 s | 649 s | 555 s | 0.88 |
| mhs-poisson-z (177 levels, polish 3) | 281 s | 275 s | 9.5 s | 0.03 |

The column times vary by ~30 % between runs (a first run: ltau 0.52 / 0.44 s, logm 0.49 / 0.57 s). Two changes after
the first run: the adjoint's Newton decrement (a status number, not part of the gradient) is solved to CG tolerance 1e-3,
and column-hse-z reuses the recorded ln P instead of replaying the column (adjoint 2.91 → 2.51 s). Before them
mhs-poisson-ltau's adjoint took 1199 s and mhs-poisson-z's 22 s; the default adjoint CG tolerance is 1e-8. Accuracies
above are unchanged by them.

The equilibrium part of one gradient of the current inversion (column-hse-ltau) goes from 1280 reconstructions (two per
control coordinate) to 0.6 s, independent of the number of control coordinates. `test/scripts/adjoint_benchmark.jl`.

## FFNOInversion (Phase 7, branch `equilibrium-adjoint` of FFNOML)

`HybridAdjointObjectiveGradient(equilibrium = :adjoint)` (default): `distributed_regularization_vjp!` (analytic, with
the halo transpose) and `equilibrium_vjp!` (→ `adjoint_embedded!`) before the control accumulation; the per-coordinate
loop only for `:finite_difference` or backends without an adjoint. Tests (`test/equilibrium_adjoint.jl`): the
regularization gradient against differences (both views, orders 1 and 2, τ and height grids) < 1e-7; the equilibrium VJP
for column-hse-ltau, -logm, -z with and without B < 1e-6; the objective gradient with a population model that depends on
ρ: adjoint against the finite-difference equilibrium 2e-10 / 1e-11 and against the full finite-difference oracle 3e-10
(τ and height grids). `mpi_equilibrium_worker.jl` + `mpi_equilibrium_compare.jl`: the equilibrium, its adjoint and the
regularization gradient equal on 1, 2 and 3 ranks to 1e-10 (66 comparisons).
