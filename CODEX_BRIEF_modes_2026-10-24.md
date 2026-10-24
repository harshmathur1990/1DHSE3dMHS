# CODEX brief — five run modes merged into main (2026-10-24)

## What changed

At the user's request, everything is now on `main` (fast-forward to 16618b7: merges of `m5i-phi-notrust` and
`poisson-tau-direct`, the latter containing `poisson-step` and `poisson-tau`). One entry point, MPI + threads in every mode:

    [mpiexec -n R] julia --project=. -t T HSEMHS/scripts/run.jl HSEMHS/config/modes/<mode>.toml

| run.mode | implementation | decomposition |
|---|---|---|
| column-hse-z | existing `run_column_hse` (alias `column_hse`) | 2-D blocks |
| column-hse-ltau | `src/PoissonTau.jl` `column_hse_tau` | x slabs |
| mhs-poisson-z | `src/PoissonZ.jl` (port of scripts/poisson/poisson_core.jl), column HSE run in-process | x slabs |
| mhs-poisson-ltau | `src/PoissonTau.jl` `solve_tau` (port of poisson_tau_direct.jl) | x slabs |
| full-mhs-z | scripts/run.jl sets the M5i switches from `[full_mhs]` and calls `m5h_fullres.jl` `main(args)` | 2-D blocks |

New `src/Slab.jl`: x-slab context on `init_parallel(px = R, py = 1)`, periodic x halos for centred differences, global
sums, exact global quantiles (bisection on order-preserving Int64 keys of the Float64 values, ≤ 64 Allreduce rounds),
a distributed 2-D FFT (rfft along y per x plane, Alltoallv transpose to y-wavenumber slabs, FFT along x) used by both
preconditioners (tridiagonal per mode for z; the diagonalised dense (n+1)² operator for τ), cooperative HDF5 writes.
Config schema: new sections `[poisson]`, `[tau_grid]`, `[full_mhs]`; τ inputs must carry `x`, `y` datasets (the user's
IO/input-mhs-ltau.nc does not yet; the user will add them).

Solver change in both Poisson modes: stop on ΔΦ/Φ < rtol on 3 consecutive steps (ltau: or one such step moving no
column height by more than dz_tol_m = 100 m). Before, a single small step (ΔΦ/Φ 1.7e-5) stopped an ltau run while
strong-field columns were still moving 80 km per step; whether that happened depended on the thread count.

## Executed (hyades11; hyades4/5/15/21 for multi-node, all idle)

* 96² crops (z: input-mhs-hse.nc x 172:268, y 248:344; τ: same columns of input-mhs-ltau.nc, x/y added in a test copy):
  column modes identical on 1 vs 4 ranks; mhs-poisson-z 1 vs 4 ranks |Δ ln P| ≤ 1.5e-6, vs the original script ≤ 5.8e-6;
  mhs-poisson-ltau with the new stop rule, 1×8 / 1×32 / 4×8 ranks×threads: z(τ=1) |Δ| p50 ≈ 1 m, p99 12–24 m, max 150 m.
* full-mhs-z smoke (42 × 42) through run.jl on 4 ranks; log confirms σ_B = 5 G uniform, no trust region, Φ stop 1e-6.
* Full cube, 8 ranks × 32 threads on 4 nodes: column-hse-ltau 95 s; mhs-poisson-ltau 8.3 min (14 min on one node,
  same z(τ=1) p1/p50/p99 −53 / −11 / +109 km); column-hse-z 5.3 min; mhs-poisson-z 4.7 min (hit the then 60-step cap at
  ΔΦ/Φ 1.8e-4; cap now 100).
* test/runtests.jl on hyades11: all pass (exit 0), incl. MPI ranks.

## Problems and uncertainties

* mhs-poisson-ltau: CG hits its 80-iteration cap in every step on large grids (rel residual 1e-2–2e-1); the last steps
  still move a few strong-field columns by ~7 km while Φ changes < 1e-4 — those heights are weakly determined.
* mhs-poisson-z on the magnetic crop creeps (ΔΦ/Φ ≈ 1.5e-3 per step for 100 steps): the known low-β pressure inflation.
* Rank invariance is to rounding only (reduction order); the Poisson modes amplify it along the iteration path.

## Please review

1. `Slab.gkth`/`gquantiles` exactness and the Alltoallv pack/unpack order in `transpose_fwd!/bwd!`.
2. The Lorentz linearisation in `PoissonTau.evaluate!/Jop!/JTop!` (adjoint test exact, FD to 1e-9 on a 32² crop).
3. Whether the new stop rule is the right criterion, or whether a height-change criterion should be primary.
4. The full-mhs-z ENV hand-off in scripts/run.jl (constants are read at include time).
