# Equilibrium adjoint, Phases 1–3 (EOS, output layer, column modes) — 24 Nov 2026

Branch `equilibrium-adjoint`. Plan: `design/adjoint/`. Tests: `test/test_adjoint.jl`, `test/mpi/mpi_adjoint.jl`, run in
parallel over hyades Zen 3 nodes with `test/scripts/adjoint_parallel.sh` (~50 s for all column groups).

## What exists

- WittBifrostEOS (branch `equilibrium-adjoint`, 8e5d56a): `state_d`, the X and Y derivatives of the table interpolant,
  with the values of `state` bit for bit. HSEMHS uses the local checkout as a path source until that branch is pushed.
- `tau_column_vjp!` (Tau.jl), `inclined_tau_vjp!` and the transposed bilinear shear (ViewTau.jl).
- `PoissonTau`: `HeunTape`, `hse_tau_replay!`, `hse_tau_vjp!`, `logtau_mass_vjp!`, `align_ptot_vjp!` (with the median
  reference, `gquantile_vjp`), `gauge_vjp!`, `interp_lt_vjp!` (PoissonTauAdjoint.jl).
- `RK4Tape`, `hse_column_replay!`, `hse_column_vjp!` (ColumnHSEAdjoint.jl).
- API (Adjoint.jl): `solve_embedded!(...; record = true)`, `EquilibriumCotangent`, `adjoint_embedded!` → `EquilibriumAdjoint`
  (cotangents of ln T and B), for column-hse-ltau, -logm, -z. Refusals: no record, a later solve, given nₑ, wrong shapes.

## Results (dot-product check against central differences of `solve_embedded!` itself, Richardson, best over h)

| Case | 1 rank | 3 ranks |
| --- | --- | --- |
| column-hse-ltau (ptot, B) | 3.8e-10 | 3.8e-10 |
| column-hse-ltau align none, B = 0 | 1.8e-9 | 1.8e-9 |
| column-hse-ltau μ 0.8 | 9.4e-9 | 9.4e-9 |
| column-hse-ltau μ 0.8 inclined | 9.4e-9 | 9.4e-9 |
| column-hse-logm (ptot, B) | 9.3e-10 | 9.4e-10 |
| column-hse-logm align none, B = 0 | 3.9e-10 | 3.9e-10 |
| column-hse-logm μ 0.8 inclined | 2.6e-10 | 2.5e-10 |
| column-hse-z anchor bottom | 3.0e-8 | 3.0e-8 |
| column-hse-z anchor top | 5.7e-10 | 5.7e-10 |
| column-hse-z anchor height | 4.9e-9 | 4.9e-9 |
| column-hse-z adapt off, half_cell | 8.7e-9 | 8.7e-9 |
| column-hse-z τ top zero | 2.3e-8 | 2.3e-8 |
| column-hse-z μ 0.8 inclined | 2.3e-9 | 2.3e-9 |

Targets 1e-6 (τ, mass) and 1e-7 (height). The 2- and 3-rank adjoints equal rank 0 alone to 1e-10. Building blocks
(state_d, τ trapezoid with each top closure, inclined τ at 0/30/90°, log τ on the mass axis, Heun and RK4 tapes with
every anchor) pass at 1e-8 or better. Locality: a cotangent at one node reaches only its column (align none).

## Deviations from the plan

- Heun step sizes are differentiated, not frozen (README A1, updated): frozen lengths left 6e-4 against the solver.
- The height fixture has a node exactly on a table slope break (T = 4365.2 K, a pair of table nodes 1e-9 apart): there
  the adjoint gives the slope of the cell `locate` picks and every finite difference averages both sides (1e-5 in the
  pairing from that one node). The check moves such nodes 5e-6 off the break (`off_breaks`); a measure-zero set.
- `z_tau1` is not a cotangent field (a diagnostic; FFNOInversion does not use it).

## Forward unchanged

The recording hooks are off by default. On hyades (Julia 1.12.3, 16 threads): test_embedded (parity with the stored
standalone results at 1e-12), test_logm, test_view, test_warm, test_ne_given, test_block, test_tau_height, test_nlte all
pass, except `allocations per warmed-up solve` for column-hse-ltau (148.5 volumes against a budget of 7), which fails
identically on unmodified main in the same environment (the budget was fitted on Julia 1.13.1; standalone the solve
allocates 4.8 volumes on main and on this branch). The column loop of `column_hse_tau` sits behind a function barrier
(`hse_columns!`) so the optional tape does not add allocations.
