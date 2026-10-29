# Release candidate: τ-grid viewing angle and opt-in solver acceleration (`solver-time`, 29 Oct 2026)

Prepared after Codex rounds 6–8 (`design/CODEX_REVIEW_solver_time_round{6,7,8}_2026-10-29.md`). **Not merged.**
`main` and `embedding-api` are used in production by FFNOML. Merging, deployment and acceptance of the reduced scope are
the user's decisions.

## Candidate

- Branch `solver-time`, commit: the commit that adds this file (see `git log -1 -- design/RELEASE_CANDIDATE_solver_time_2026-10-29.md`).
- It is 54 commits ahead of `main` (5e901b5). It includes the viewing-angle work and the solver-time rounds.
- Environment: Julia 1.13.1 (the committed Manifest), MPI.jl's MPICH (`mpiexecjl`), the EOS table `cache/eos_table_fullz.h5`.
- On hyades, use node-local copies of Julia for multi-rank runs; see the root README.
- `main`'s own Manifest does not currently precompile here (ProgressMeter). That fix is pending the user's decision and
  is not part of this candidate.

## Supported settings

**Defaults.** These are unchanged from before this work:

| setting | default |
|---|---|
| `[view]` | μ = 1, plane-parallel |
| `poisson.cut_frame` | `"vertical"` |
| `poisson.max_steps` | 25 |
| `stop` | `"gn"` (gn_tol 1e-3 on 2 steps, then the qualified stop check) |
| `lm` | `"alpha"` |
| `forcing` | `"fixed"` (cg_tol 1e-3, cg_maxit 80) |
| `precond` | `"fourier"` |

**Supported opt-in settings.**

| setting | what it is |
|---|---|
| `[view]` μ in (0, 1], any azimuth, `plane_parallel` or `inclined` | the viewing angle |
| `lm = "gain"`, `forcing = "ew"`, `cg_cap = 200` | the D2 benchmark settings |
| `precond = "switch"` | Fourier, then column blocks once a CG solve hits its limit |
| `precond = "block"`, `"hybrid"`, `damping = "marquardt"` | available, not benchmarked for this release |

**Experimental, not on the supported path.**
- The two-level preconditioners in `scripts/viewing/ml_lab.jl`.

## Changes for users (since `main`)

- **Viewing angle for the τ-grid modes.** Input contract: one line of sight per pixel, on a regular deprojected footprint
  grid, with B in the local frame. Plane-parallel and inclined geometries are supported. The cut is at the vertical depth
  by default. μ = 1 reproduces the vertical view bitwise.
- **`column-hse-ltau`** balances each line of sight on its own. An inclined input gives the plane-parallel result.
- **`mhs-poisson-ltau`.**
  - New stop rule: qualified Gauss–Newton decrement (`stop = "gn"`, 25 steps).
  - Round-off acceptance and an EOS-domain stop (`domain_limited`).
  - PCG breakdown outcomes are reported.
  - The stop check is solved on column blocks, with `gn_decrement_source`.
  - The solver controls above are exposed through config, CLI and the embedding API.
  - Work statistics are in the status and the output.
- **Exit codes (breaking).** The `column-hse-ltau`, `mhs-poisson-ltau` and `mhs-poisson-z` drivers exit:
  - 0 when converged;
  - 2 for a valid result that did not converge (it used to be 0);
  - 3 for an invalid result, which is not written; an existing output is kept, and `<output>.failed.txt` records the
    failure.

  The README has a migration note and a shell example.
- **Fix:** the `mhs-poisson-z` and `column-hse-z` CLIs crashed while logging their status.

## Verification (Julia 1.13.1, 29 Oct 2026)

Single-rank tests run from `HSEMHS/test` with the `runtests.jl` preamble, for example
`julia --project=.. -t 8 -e 'using Test, HSEMHS, MPI, HDF5, OffsetArrays, Random, Statistics, LinearAlgebra; const H = HSEMHS; const FIX = joinpath(pwd(), "fixtures"); H.ensure_mpi(); include("test_embedded.jl")'`.
MPI tests run from `HSEMHS` with `mpiexecjl --project=. -n R julia --project=. -t 2 test/mpi/<file>.jl`.

| suite | result |
|---|---|
| units/config | 18 / 18 |
| test_block (preconditioners, PCG outcomes, qualification, round-off, switch) | 116 / 116 |
| test_warm | 47 / 47 |
| test_view (angles, Jacobians, column HSE at μ 1–0.2) | 133 / 133 |
| test_embedded (interface parity, validation, status, CLI exit codes in process and as a subprocess, invalid rerun, allocations) | 103 / 103 on hyades18 (Xeon E5-2695 v4) and hyades20 (Xeon 8160) |
| mpi_tests, mpi_embedded, mpi_view on 2 / 3 / 4 ranks | all pass: mpi_tests 34 / 34 at 2, 3, 4 ranks; mpi_embedded 30 / 30, 30 / 30, 32 / 32; mpi_view 34 / 34, 28 / 28, 34 / 34 (including `switch` on R ranks) |

**Known-equilibrium controls through the CLI.** All three qualify their stop with both `fourier` and `switch`:

| control | qualified at step |
|---|---|
| rotated, μ1, φ 30° | 6 |
| rotated, μ0.3, φ 30° | 11 |
| μ0.3 at 256² | 13 |

## Requirement status

| requirement | status |
|---|---|
| Column HSE at an angle (CLI and embedding) | **delivered** |
| MHS at an angle: validity, correctness coverage, status reporting | **delivered** |
| Faster valid budget-limited MHS output | **delivered**: `switch` (opt-in) is 1.9–4× faster for 25 steps on the Bifrost cube |
| Qualified convergence of `mhs-poisson-ltau` on the Bifrost cube | **not met**: every μ tested ends `max_iter` (exit 2) |
| A few minutes on 8 nodes / 256 CPUs | **unmeasured**: 6.5–11.4 min with `switch` on one node (96 cores); four nodes give 1.4–1.9× at fixed work |

Numbers: root README, "Release result table"; `design/tau/SOLVER_TIME_DESIGN.md` §6.14–6.16.

## Merge proposal (for the user's decision)

1. Review this file, the root README sections "Viewing angle", "Stopping, status…", "Release result table" and "Known
   limitations", and the round-8 review.
2. Decide on scope: either accept this as a limited release (angle support plus faster valid budget-limited MHS), or
   keep the branch open for the outstanding convergence and runtime work as a separately scoped task.
3. Before any merge into `main`:
   - check FFNOML's callers against the exit-code change (exit 2 for a valid `max_iter` result) and the new status
     fields;
   - resolve `main`'s Manifest;
   - rerun the suites above on the merge result.
4. Merge and deploy only on the user's instruction.
