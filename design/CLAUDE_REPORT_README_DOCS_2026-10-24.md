# Completion report: root README and code-matched documentation

Date: 24 October 2026. Brief: `design/CLAUDE_BRIEF_README_DOCS_2026-10-24.md` (inspected HEAD cb90a4f). Implementer: Claude.

## Brief check

Every code-level statement in the brief was checked against the source before any change; none was found incorrect.
One error of mine surfaced in the check: `technical/column-hse-z` said the column-HSE output is Julia (x, y, z); it is
h5py (x, y, z) = Julia (z, y, x). Fixed.

## Code change (user decision, option b)

`mhs-poisson-ltau` now stops on the objective only: the height shortcut (`nsmall ≥ 1 && max|δc| < dz_tol`, default
100 m) and the key `poisson.dz_tol_m` were removed (`src/PoissonTau.jl`, `src/RunModes.jl`, `src/Config.jl`; commit
813a3af). Test: 96² crop, 1 rank × 32 threads, `scripts/run.jl` on hyades11: 37 steps, `converged: ΔΦ/Φ < 1e-04 on 3
consecutive steps`, exit 0 (same step count as before the change). The `read_tau_input` docstring was corrected (only x, y
honour a units attribute). No other code was changed.

## Documentation changed

* New `README.md` (root): purpose and conventions, the five modes and auxiliary modes (legacy `mhs` not executable),
  methods, installation chain, data and EOS-table chain, input contracts, serial/threads/MPI/hybrid commands, the
  decomposition and `[parallel]` table, thread controls (Julia/BLAS/FFTW), I/O and memory, a site-specific multi-node
  appendix, per-mode honoured/ignored keys, the stopping table (Φ-only, exact definitions), exit status versus
  convergence, full-mode run directory/STOP/restart/overwrite behaviour, output layouts, tests, a dated performance table
  with commits and log locations, limitations, code follow-ups, the documentation table, a repository map, licence status.
* `HSEMHS/config/modes/README.md`: reproducibility claim scoped; pointer to the stopping/status section.
* Physics PDFs: `column-hse-z` (τ_top non-zero at the box top; lines form over τ ranges); `column-hse-ltau` (no external
  P_ref but modelling choices; capped intervals are not hydrostatic and miss thickness; alignment an approximation, tension
  neglected, gas-scale-height extrapolation; Wilson depression a tendency; ∇·B not enforced); `mhs-poisson-z` (Helmholtz
  uniqueness needs boundary conditions; the solver is a weighted non-linear least squares, not a projection; the anchor is
  a penalty that the step clamp can shift; "in general" for the no-static-state statement); `mhs-poisson-ltau` (new section
  on what is and is not determined: global gauge, fixed layers above the cut, weakly constrained columns, no ∇·B = 0);
  `full-mhs-z` (geometric-mean gauge built into the unknowns; schematic objective with term-specific normalisations;
  residual sign; obstruction "in general"; stop definition; no equilibrium guarantee).
* Technical PDFs: exit status ≠ convergence and how to read the termination (all three solver modes); `mhs-poisson-ltau`
  Φ-only stop, the actual LM damping rule (`α == 1`, not `α == s_c`), the meaning of `max|δc|`, no ∇·B, BLAS/FFTW threads,
  reproducibility scoped, crop and full-cube records with commits and logs; `mhs-poisson-z` anchor as a penalty, termination,
  reproducibility, output order; `column-hse-ltau` actual input contract (no T/B unit conversion, fixed names, non-strict
  checks), cap behaviour, extrapolation approximation, exit status; `column-hse-z` output order, exit status, dated record;
  `full-mhs-z` two entry points and their defaults, forwarded/ignored keys, restart check scope, output-directory
  behaviour, termination/status, replicated raw input memory, dated M5i record (not converged).

## Commands executed

* `bash docs/build.sh`: all 10 PDFs built; then each TeX twice with `pdflatex`: 0 errors, 0 overfull boxes, 0 LaTeX
  warnings (physics 3/3/3/3/4 pages, technical 2/2/2/3/2). Pages inspected visually: physics `column-hse-ltau` p1,
  technical `full-mhs-z` p1.
* On hyades11 (worktree `MHS_modes` at 813a3af): `HSEMHS.load_config` on the five mode configs and the gate0 and
  EOS-table configs referenced in the README: all parse, modes as expected; `MPI.install_mpiexecjl` exists.
* The 96² crop run above.

## Evidence reused, not re-run

All timings and comparison numbers are from runs earlier on 24 Oct (and the M5i run of 22–23 Oct), with the commits
stated in the README and PDFs; logs are in the cluster worktrees (`MHS_modes/HSEMHS/out/modes{,_test}/`,
`MHS_m5i/HSEMHS/out/m5i/run_fullz`), not in git. No production solve was run. The current Φ-only rule has not been run
on a full cube.

## Unresolved code issues (for the user's decision)

1. Poisson drivers return `converged` (exit 0) regardless of termination; `scripts/run.jl` returns 0 after the full driver.
2. `hse_tau!`: a capped interval breaks before integrating its remaining geometric thickness; capped locations are not stored.
3. `solve_tau`: damping lowered only when `α == 1`, not after a full scaled step.
4. τ reader: no unit conversion for T/B; non-strict log τ check; no uniform-grid check.
5. No automated tests for `Slab.jl` and the two Poisson modes.
6. `eos.build_if_missing` and the legacy `mhs` mode exist in the schema without an implementation.
7. No licence file; third-party terms of `wittman-eos/` not recorded.
