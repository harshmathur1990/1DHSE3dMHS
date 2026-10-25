# Embedding API gate (FFNO integration, Phase 1)

Branch `embedding-api` from `b2401a3`. Specification: `FFNOML/FFNOInversion.jl/docs/hsemhs-integration/phase-1-hsemhs-embedding-api.md`.
Environment for every number below: hyades login node, Julia 1.12.3 (node-local copy in `/tmp/julia-1.12.3`), MPI.jl's
bundled MPICH 5.0.2, a separate environment (`/mn/stornext/d9/data/harshm/hsemhs_env112`) that `develop`s this package,
so the committed 1.13.1 Manifest is untouched.

## What changed

| File | Change |
|---|---|
| `src/Parallel.jl` | `init_parallel(comm, NX, NY, NZ; …)` takes the parent communicator, duplicates it (`Comm_dup`) and builds the Cartesian communicator from the duplicate; both are owned by the `ParCtx` (`owned_comms`) and released by `free_parallel!(par)` (never `MPI.Finalize`). The ownership check is computed for every rank from `block_range` on all ranks (no reduction needed). `init_parallel(NX, NY, NZ)` remains the standalone entry (`ensure_mpi` + `COMM_WORLD`). |
| `src/Slab.jl` | `slab_par(comm, NX, NY, NZ)`; the x-slab condition is the exported pure `slab_check(NX, nr)` (`NX ÷ nr ≥ 2`, which is exact: the smallest `block_range` slab is ⌊NX/nr⌋), raised as `InvalidInput` identically on all ranks. `slab_par(NX, NY, NZ)` remains the standalone entry. |
| `src/ColumnHSE.jl` | The per-column state (P, ρ, nₑ, χ₅₀₀, flags, τ, z(τ=1)) is the shared `column_state!`, used by `column_hse` (2-D blocks) and by the embedded slab path. |
| `src/PoissonZ.jl` | `solve_pressure` accepts a cached `SlabFFT` (`S`), `rms_atol` (zero-step acceptance of an already balanced start) and returns, besides the old fields, `reason` (:converged/:no_descent/:max_steps), `gn_decrement`, `grad_norm`, `cg_its`, `cg_rel`, `rms0`, `rms`, `phi0_zero` and `cont_bad` (non-finite pressures left by the hydrostatic continuation above the cut). A cut below the grid is `InvalidInput`, not `error`. |
| `src/PoissonTau.jl` | `column_hse_tau` returns `nbad` (columns that left the EOS table) instead of calling `error` (alignment and gauge are skipped then); `solve_tau` accepts `S`, `rms_atol`, returns a 7th value `info` with the same diagnostics plus `lm_mu`, and returns `reason = :eos_out_of_envelope` instead of `error` for a start outside the table. |
| `src/Embedded.jl` (new) | The in-memory API (below). |
| `src/RunModes.jl` | `run_column_hse_ltau`, `run_mhs_poisson_ltau`, `run_mhs_poisson_z` are now: read file → `prepare_embedded(COMM_WORLD, …)` → `solve_embedded!` → write file, with unchanged datasets (two attributes added: `stop_reason`, `gn_decrement`). |
| `src/HSEMHS.jl` | Includes `Embedded.jl`, exports the API; **`__init__` removed** (it set `BLAS.set_num_threads(1)` on load). `scripts/run.jl` now sets BLAS to one thread itself; the τ-Poisson driver keeps its `min(32, nthreads)` toggle around the solve (standalone path only). |

`run_column_hse` (column-hse-z with its full diagnostics output: Pe, column mass, residuals, layer tables) still runs
on the 2-D block decomposition. It is not rewired onto the slab API because its extra diagnostics are not part of the
embedded result; it shares `hse_column!` and `column_state!` with the embedded path, and the equivalence test below
shows the slab path reproduces its fields bitwise.

## API

```julia
ws  = prepare_embedded(comm, mode; axis, x, y, eos, g, column, poisson = nothing, logf = s -> nothing)
res = solve_embedded!(ws, lnT, Bx = nothing, By = nothing, Bz = nothing)
free_embedded!(ws)
snapshot(res)                       # owned deep copy
```

`mode` ∈ `(:column_hse_z, :column_hse_ltau, :mhs_poisson_z, :mhs_poisson_ltau)` (named wrappers `prepare_column_hse_z`
etc.). `axis`: height z [m], strictly decreasing (top first), or log10 τ500, strictly increasing. `x`, `y`: global,
uniform, increasing [m]. `eos`: `EOSTable` or table path. Arrays: rank-local x slab, (vertical, y, x); rank r of `comm`
owns `slab_range(NX, nranks, r)` and all of y. B is required for the Poisson modes and for column-hse-ltau with
`align = "ptot"`.

**Collective protocol.** `prepare_embedded` and `solve_embedded!` run all rank-local checks (mode, axes, options,
slab size, EOS table load; array shapes and finiteness) inside `agree(f, comm, stage)`: one `Allreduce` of a failure
flag on the caller's (already valid) communicator, and on failure an `Allgatherv` of the messages, after which every
rank throws the same `InvalidInput` naming each failing rank. Only then are communicators, FFT plans or solvers entered.

**Result** (`EquilibriumResult`, workspace buffers, overwritten by the next solve; `snapshot` for an owned copy):
`lnP`, `Pgas` [Pa], `rho` [kg m⁻³], `ne` [m⁻³], `chi500` [m⁻¹, extinction per length at 5000 Å, absorption + scattering;
`eos/Opacity500.jl:chi500_m` multiplies the CGS cm⁻¹ value by 100], `tau500`, `logtau500` (3-D), `z` [m] (3-D: the
height axis broadcast in height modes; the aligned and gauged heights in τ modes), `z_tau1` (y, x), `eos_flags`
(UInt8 mask), `Pgas_start` (column-HSE start of the Poisson modes). ρ, nₑ and χ₅₀₀ of the final state come from one
`state(tab, ln T, ln P)` call per node on one table.

**Status** (`EquilibriumStatus`, identical on all ranks): `status::SolverStatus`, `reason`, `valid`, `steps`, `phi0`,
`phi`, `rms0`, `rms` (= √(Φ/ΣW)), `phi0_zero`, `gn_decrement`, `grad_norm`, `cg_its`, `cg_rel`, `lm_mu`,
`eos_violations`, `counts` (per failed check), `stats` (column-stage statistics, cut level, alignment, gauge shift),
`messages`, `seconds`.

* `valid` = EOS clean ∧ all fields finite ∧ P, ρ, χ₅₀₀ > 0 ∧ z not increasing and τ not decreasing down
  every column (amended in `a5cdc9e`: equal heights of adjacent levels are valid — the τ-grid column HSE ends a
  capped `cap_per_dex` interval without advancing z; 1683 such columns on `IO/input-mhs-ltau.nc` were wrongly
  rejected by the original strict check; counted in `stats["equal_height_intervals"]`) ∧ (Poisson z) the continuation above the cut succeeded. `reason` is the first failed check
  (`:eos_out_of_envelope`, `:continuation_failed`, `:nonfinite`, `:nonpositive`, `:folded`), else the solver's stopping
  reason (`:converged`, `:no_descent`, `:max_steps`; column modes `:converged`).
* **Stationarity.** The Poisson solvers stop on `n_consec` consecutive relative decreases of Φ below `rtol`; that does
  not prove stationarity. After the iteration each solver solves one more undamped Gauss–Newton system at the returned
  state, (JᵀWJ + ereg) δ = −JᵀWR, and reports `gn_decrement` = δ·(−JᵀWR)/Φ — the relative decrease of Φ a further full
  step would predict (scale-free; 0 at a stationary point) — and `grad_norm` = ‖JᵀWR‖. The returned fields are not
  changed by this. `lm_mu` is the final Levenberg–Marquardt damping (τ grid): it shapes the steps, it is not in the
  stationary equation (`phi` has no damping term).

## Option mapping

All option structs mirror the standalone config sections (same names, defaults, units). Validation is the pure
`option_errors(opts, axis)`; FFNO can call it at configuration time.

| Struct | Field | Unit | Default | Valid | Standalone key |
|---|---|---|---|---|---|
| `ColumnZOptions` | `anchor` | — | `:bottom` | `:bottom`, `:top`, `:height` | `column_hse.anchor` |
| | `P_ref` | Pa | required | finite > 0 (resolved number; `"from_input"` is a file-driver feature: callers read the attribute themselves) | `column_hse.P_ref` |
| | `z_ref` | m | NaN | inside the axis when `anchor = :height` | `column_hse.z_ref_m` |
| | `n_sub`, `adapt`, `rk_tol`, `n_sub_max` | — | 4, true, 1e-6, 64 | ≥ 1, —, > 0, ≥ n_sub | `column_hse.*` |
| | `tau_top`, `H_max`, `mu` | —, m, — | `:exp_tail`, 5e6, 1 | `:exp_tail`/`:half_cell`/`:zero`, > 0, > 0 | `tau.top`, `tau.H_max_m`, `tau.mu` |
| `ColumnTauOptions` | `cap_per_dex` | ln P per dex | 50 | > 0 | `tau_grid.cap_per_dex` |
| | `align` | — | `"ptot"` | `"ptot"`, `"none"` | `tau_grid.align` |
| `PoissonZOptions` | `z_cut_m` | m | 1e6 | ≥ 3 solved levels at or below the cut (`POISSON_Z_MIN_LEVELS`) | `poisson.z_cut_m` |
| | `max_steps`, `rtol`, `n_consec` | — | 100, 1e-4, 3 | ≥ 1, > 0, ≥ 1 | `poisson.*` |
| | `anchor_w` | — | 1 | > 0 | `poisson.anchor_w` |
| | `cg_tol`, `cg_maxit` | — | 1e-3, 80 | (0, 1), ≥ 1 | `poisson.*` |
| | `rms_atol` | — | 0 | ≥ 0 | (new; 0 = standalone behaviour) |
| `PoissonTauOptions` | `logtau_cut` | dex | −4 | inside the axis, cut level k_c ≥ 2, ≥ 3 solved levels (`POISSON_TAU_MIN_LEVELS`) | `poisson.logtau_cut` |
| | `max_steps`, `rtol`, `n_consec`, `cg_tol`, `cg_maxit` | — | 100, 1e-4, 3, 1e-3, 80 | as above | `poisson.*` |
| | `dc_cap_m`, `du_cap`, `lm_mu0` | m, —, — | 3e5, 2, 0 | > 0, > 0, ≥ 0 | `poisson.dc_cap_m`, (internal), `poisson.lm_mu0` |
| | `rms_atol` | — | 0 | ≥ 0 | (new) |

Minimum retained levels: `solve_pressure` indexes `h[1]` and `h[end]` and the anchor face; with fewer than 2 solved
levels it fails, so 3 is required. `solve_tau` evaluates heights at k_c from level k_c − 1 (`zden`), so k_c ≥ 2, and its
face vectors need ≥ 3 solved levels.

## Semantics callers rely on

* **PoissonZ anchor and damping.** `Aop!` = JᵀWJ + `ereg`·I + `Aanc`·(anchor-cell sum)ᵀ(anchor-cell sum); `phi` has only
  force residuals and the right-hand side is −JᵀWR. The anchor (`anchor_w`, default bottom plane) and `ereg` therefore
  damp the increment of each Gauss–Newton step; they are not a constraint and the mean ln P over the anchor cells is
  not held fixed. Measured drift: see "Measurements". Decision (user, 24 October 2026): HSEMHS is the authority; this
  is the intended behaviour.
* **PoissonTau damping.** `ereg` (1e-2 κ² weights) plus LM damping μ·`Dlm` on the increment system; Φ has no damping
  term. μ is raised ×4 on a failed step and lowered after full steps; its final value is reported.
* **τ-grid alignment and gauge.** `align_ptot!`: the reference is the global median (exact order statistic, `gquantiles`)
  of ln(P + B²/2μ₀) on the bottom level; each column is shifted so that its total pressure equals the reference at one
  height (linear interpolation between levels in total pressure, hydrostatic extrapolation below the column when the
  reference is below it). `gauge!`: subtract the global mean of z(log10 τ = 0) (linear in log τ, extrapolated from end
  segments) so that mean z(τ = 1) = 0. Both are global reductions: every column's z depends on all columns.

## Tests

`test/test_embedded.jl` (in `runtests.jl`; also `test/run_embedded.jl` alone) and `test/mpi/mpi_embedded.jl` (2, 3, 4
ranks; in `runtests.jl`). Fixtures: `test/fixtures/embedded/` (windows of the production inputs, 24×18 and 11×12
columns, made by `oracles/python/make_embedded_fixtures.py`); pre-refactor outputs of the standalone drivers at
`b2401a3` for all four modes on both: `test/fixtures/embedded_ref/` (configs included).

### Equivalence with the pre-refactor drivers

`test/scripts/compare_embedded_refs.jl` re-runs the eight reference configs (four modes × two fixtures) through the
rewired file drivers and compares every dataset with the `b2401a3` output: **all bitwise identical** (column-hse-z 29
datasets, mhs-poisson-z 12, τ modes 11 each). In `test_embedded.jl` the embedded API on the same inputs reproduces the
stored results to ≤ 1e-12 for all four modes and both fixtures, including column-hse-z through the slab path against
the 2-D-block `column_hse` output.

### Single rank (`test_embedded.jl`): 75/75 pass

Equivalence (above); layout on a 7×5×40 grid with T varying independently along all axes (columns equal direct
`hse_column!` + `column_state!` calls bitwise); χ₅₀₀/τ units (τ equals the trapezoid of χ [m⁻¹] over z [m] plus the
exp-tail closure to 1e-12); option validation (retained levels at the top, bottom, next-to-bottom and minimum-accepted
cut for both Poisson modes, cut below the axis, missing `P_ref`, `z_ref` outside the axis, invalid `align`, wrong or
missing Poisson options, missing table, reversed axis, non-uniform x); input validation (wrong slab shape, missing B for
a Poisson mode, NaN in ln T); physics failures as status (out-of-envelope T in mhs-poisson-z and column-hse-ltau →
`:eos_out_of_envelope`, next valid solve recovers; the field checker on injected non-finite, non-positive and folded
columns); already balanced start (horizontally uniform T, B = 0, `rms_atol` 1e-3: zero steps, `:converged`, GN
decrement 0, solved region unchanged to rounding); stationarity measure (1 step vs 30 steps: `:max_steps` and a larger GN
decrement after 1 step); buffer contract (`res` is the workspace buffer, `snapshot` independent); allocations (below);
BLAS thread count unchanged by loading and solving.

### Multi-rank (`mpi/mpi_embedded.jl`): n = 2: 30/30, n = 3: 30/30, n = 4: 32/32

Rank-count independence against the single-rank result (max relative difference over all returned fields):

| Mode | fixture | n = 2 | n = 3 | n = 4 | asserted tolerance |
|---|---|---|---|---|---|
| column-hse-z | 24×18 / 11×12 | 0 / 0 | 0 / 0 | 0 / 0 | 0 (bitwise) |
| column-hse-ltau | 24×18 / 11×12 | 1.5e-12 / 0 | 3.0e-12 / 3.5e-13 | 1.5e-12 / 0 | 1e-11 |
| mhs-poisson-z | 11×12 (converged, 5 steps) | 1.5e-12 | 1.7e-12 | 1.8e-15 | 1e-9 |
| mhs-poisson-ltau | 11×12 (converged, 5 steps) | 4.6e-11 | 8.9e-11 | 3.3e-11 | 1e-9 |
| mhs-poisson-z | 24×18 (stops at max_steps = 30) | 8.8e-6 | 1.2e-6 | 1.4e-6 | recorded |
| mhs-poisson-ltau | 24×18 (24 steps) | 4.1 | 9.7 | 5.3 | recorded |

The 24×18 window is a non-periodic crop of the production atmosphere solved as periodic. Poisson z does not converge
there in 30 steps (GN decrement 5.6e-3), so rounding-level differences persist. **Poisson τ is step-path sensitive
there:** the same solve on one rank also changes by O(1) between 1 and 4 BLAS threads (the standalone driver's
`min(32, nthreads)` toggle); different accepted Levenberg–Marquardt steps lead to different final heights in a few
strong-field columns, while Φ/Φ₀ agrees (8.473e-3 vs 8.474e-3). This matters for Phase 4/5 (finite differences through
mhs-poisson-ltau and reproducibility across rank counts) and is passed on to those phases.

Collective safety (all pass on 2, 3, 4 ranks): missing EOS table on the last rank only, a non-table HDF5 file on rank 0
only (stands in for a corrupt table), a malformed slab on rank 1 only inside `solve_embedded!` — every rank throws the
same `InvalidInput` naming the failing rank, and a following parent-communicator `Barrier` completes; out-of-envelope
T on the last rank only — identical `:eos_out_of_envelope` status and counts everywhere, and the next solve is valid;
NX = 5 over 3 and 4 ranks rejected identically; 50 prepare/solve/free cycles; on 4 ranks, two 2-rank sub-communicators
solving different atmospheres concurrently give results **bitwise equal** to the same solves done one pair at a time
(no hidden COMM_WORLD use or cross-talk), followed by a world `Barrier`.

### Existing suite

Single-rank files, unmodified (`MHS_modes`, identical source to `b2401a3`) vs this branch, same Julia 1.12.3
environment, each file included in order with failures caught (`hsemhs_continue.jl`): every file has the same
outcome, except the pre-existing allocation-budget tests, which fail intermittently on Julia 1.12.3 with and without
these changes (`test_gradient.jl:108`: 10.9 MB unmodified / 7.1 MB here vs a 1 MB budget; `test_stag_gradient.jl:165`:
103–108 MB vs 2 MB in both): they measure a single `@allocated` call that includes compilation, and the code they test
is unchanged. `test_operators.jl`/`test_gate1.jl` need the Python oracle (`PY`), not available in that harness, in both.
`mpi/mpi_tests.jl`: n = 2, 4, 6: 34/34 each.

### Measurements (hyades login node, 4 threads per rank, warmed-up second solve)

| fixture | mode | 1 rank | 4 ranks | steps | reason | Φ/Φ₀ | GN decrement | anchor-cell mean Δln P |
|---|---|---|---|---|---|---|---|---|
| 24×18×177 | column-hse-z | 0.060 s | 0.021 s | — | converged | — | — | — |
| 24×18×95 | column-hse-ltau | 0.015 s | 0.007 s | — | converged | — | — | — |
| 24×18×177 | mhs-poisson-z | 1.43 s | 1.21 s | 30 | max_steps | 2.45e-2 | 5.6e-3 | +0.338 |
| 24×18×95 | mhs-poisson-ltau | 7.10 s | 4.80 s | 24 | converged | 8.47e-3 | 6.7e-6 / 8.3e-6 | — |
| 11×12×177 | mhs-poisson-z | 0.115 s | 0.096 s | 5 | converged | 1.37e-1 | 4.5e-6 | +2.5e-4 |
| 11×12×95 | mhs-poisson-ltau | 0.307 s | 0.304 s | 5 | converged | 4.44e-2 | 2.0e-11 | — |

`prepare_embedded` with a table already in memory: 0.2–1.0 s on first use (compilation), negligible afterwards.
The anchor drift column is the change of the bottom-plane mean ln P from the column-HSE start: the anchor damps but does
not hold it (0.34 in ln P on the unconverged crop, 2.5e-4 on the converged one).

**Allocations per warmed-up solve** (minimum over three calls, units of one local volume of Float64): column-hse-z 1.8,
column-hse-ltau 4.7, mhs-poisson-z 1114, mhs-poisson-ltau 20282 (6.7 GB for the 24×18×95 crop). The Poisson kernels
allocate temporaries every CG iteration (`similar`, broadcasts in `Jop!`, `JTop!`, `evaluate!`, `pcg!`). Test budgets
are the measured values × 1.5.

## Deviations from the phase document

* **Allocation target not met for the Poisson modes** (target "≤ 1 volume per solve"; measured 1114 and 20282
  volumes). Removing the per-iteration temporaries means rewriting `PoissonZ`/`PoissonTau` kernels and risks the
  bitwise equivalence that this phase is gated on; it belongs with the Phase 6 performance work. The column modes meet
  it approximately (1.8 and 4.7 volumes, mostly result copies).
* **Rank-count tolerance ≤ 1e-10 holds for converged solves only.** It is asserted on the converged 11×12 fixture; on
  the 24×18 crop the differences are recorded (Poisson z unconverged; Poisson τ step-path sensitive, see above).
* `run_column_hse` (column-hse-z file driver) keeps its 2-D block decomposition and full diagnostics output; it shares
  `hse_column!`/`column_state!` with the embedded slab path instead of calling the embedded API (equivalence shown
  bitwise).
* The stationarity measure is the Gauss–Newton decrement plus ‖JᵀWR‖, computed with one extra preconditioned CG solve
  after the iteration (diagnostic only); a second independent computation of ‖JᵀWR‖ was not added.
* "Hash mismatch on one rank" is exercised with a non-table HDF5 file on that rank (`load_table` fails before its hash
  check); the dedicated SHA mismatch path is the same code path (`InvalidInput` inside `agree`).
* Multi-rank tests ran on the hyades login node only (MPI.jl's bundled MPICH): the system OpenMPI needs `libhcoll`,
  absent on most nodes, and multi-node runs are left for Phase 3 with the environment set-up.
* Reference outputs total 9.9 MB (target < 5 MB): after the full-output bitwise comparison above, they were reduced to
  the fields the tests read (`test/scripts/trim_embedded_refs.jl`, values verified unchanged) and deflated; Float64
  fields of the 24×18 crops do not compress below that. Fixture inputs are 3.2 MB.
* Not changed: the Manifest (1.13.1) was not regenerated; `Project.toml` resolves and the suite runs on Julia 1.12.3 in
  a separate environment that `develop`s this package.

## Environment notes for Phase 3

* Run Julia from a node-local copy (`/tmp/julia-1.12.3`): loading it from stornext intermittently fails ("failed to map
  segment"), which kills MPI ranks and hangs the others.
* System OpenMPI 5.0.5 (`openmpi/gcc13/5.0.5`) links `libhcoll.so.1`, present only on hyades2/3 and eagle8; the `hpcx`
  RPM provides it (to be installed into `~/packages`).
