# CODEX brief, round 8: release candidate for the τ-grid viewing angle (2026-10-29)

Branch `solver-time`, HEAD after this brief. This round carries out the final work package of your round-7 review
(`design/CODEX_REVIEW_solver_time_round7_2026-10-29.md`). The details are in `design/tau/SOLVER_TIME_DESIGN.md` §6.15.
The user-facing record is the root `README.md`, sections "Viewing angle (τ-grid modes)" and "Release result table".
Equations, Φ, boundaries, gn_tol 1e-3 and the 25-step budget are unchanged. Main and embedding-api are untouched.

## 1. Selected configuration

**Defaults are unchanged:**
- `lm alpha`;
- `forcing fixed` (cg_tol 1e-3, cg_maxit 80);
- `precond fourier`;
- `stop gn`;
- max_steps 25.

**Opt-in options:**
- the D2 benchmark settings `lm gain`, `forcing ew`, `cg_cap 200`;
- `precond switch` on top of them.

All of these are exposed through `PoissonTauOptions`, `[poisson]`, the CLI and the embedding API. The README separates
the configured defaults from the benchmark settings.

**Changed for every preconditioner.**
- The stop qualification and the final decrement diagnostic are solved with the column blocks where the grid has them,
  with a Fourier fallback. `qualify_stop(precond = :block)` restores the outer preconditioner afterwards.
- `gn_decrement_approx` flags a decrement taken from an unconverged final solve or from an unqualified stop check.

## 2. The bounded outer-step experiment: negative, so the trust-region trial was not run

Cap probe at saved μ0.3 states, run loose and tight and at damping μ 0 and 1:

| state | binding cap | gain at the capped step | 2 × capped | full step |
|---|---|---|---|---|
| after 20 steps | ln P (0.31–0.35) | 0.27–0.31 | Φ rises 5.6–16 % | Φ rises 39–72 % |
| after 10 steps | ln P (0.36–0.41) | 0.86–0.88 | Φ rises 52 % to ×8 | Φ ×36–132 |

- The binding cap is the ln P cap, not the height cap; round 7 was wrong on this. The worst cell asks for
  |δ ln P| 4.8–6.4.
- Along the Gauss–Newton direction, the capped step is already as long as the model allows.
- By your rule, radial scaling is not shown to be wasteful, so the trust-region trial is not justified. Scope stops
  here.
- That single cells set the cap suggests localised nonlinearity. We record it as a candidate for a separately scoped
  task, not pursued.

## 3. Resolved test failures (your finding 4)

All five also fail on main's source, built with this branch's Manifest (main's own Manifest does not precompile: the
open ProgressMeter issue).

| failure | class | action |
|---|---|---|
| mhs-poisson-z 24×18, 1.45e-5 | Julia 1.12 → 1.13 rounding (passes on 1.12.3) | reference regenerated |
| mhs-poisson-ltau 24×18, 7.42 rel (68 m in z near 0) | solver change since the pre-refactor reference | reference regenerated |
| mhs-poisson-ltau 11×12, 9.5e-12 | rounding | reference regenerated |
| allocations column-hse-z 3.34 > 3, mhs-poisson-z 2054 > 1700 | per-thread scratch (Profile.Allocs: column chunks, FFT line buffers); 1/4/8/16 threads give 1.2/1.6/2.2/3.3 and 709/1118/1387/2054; the gate's 1.8 and 1114 were at about 4 threads | budget a + b·threads, fitted, × 1.5 |

**Parity before replacement.** Each reference was regenerated with the current standalone CLI, and the embedding path
reproduces all 8 at ≤ 1e-12.

**Portability.** The iterative results also depend on the BLAS thread count (0.86 rel at 4 vs 16 threads) and on the
CPU (1.97 rel between hyades18 and hyades20), on the not-converged 24×18 crop. The Poisson equivalence test therefore
compares against the standalone driver run in-process on the same machine and threads. Column modes keep the stored
files.

**Two bugs found and fixed.**
- The `mhs-poisson-z` and `column-hse-z` CLIs crashed when logging status stats that hold strings. `sort` on Pairs
  compares the values too.
- The τ, column-τ and z drivers returned `converged` whatever happened. They now return the solver status:
  - exit 0 when converged;
  - exit 2 at max_iter or stalled;
  - exit 3 for an invalid result, which is logged and not written.

  A test checks the exit codes.

## 4. Verification

- **Single-rank suites.** Block 114, warm 47, view 133 (including column HSE at μ 1, 0.8, 0.5, 0.3, 0.2), units/config
  18. Embedded 91 on hyades18 and hyades20.
- **MPI suites on 2, 3 and 4 ranks.** `mpi_tests` 34, `mpi_embedded` 30–32, `mpi_view` 28–34. They include
  `switch` on R ranks: the same switch step on every rank and on one rank, and the stop check on blocks.
- **Controls through the CLI** (rotated μ1 and μ0.3 at φ 30°, gs μ0.3 at 256²) qualify with both preconditioners, at
  steps 6, 11 and 13. The rotated μ0.3 control switches after step 9 and takes 64 s against 78 s.
- **Embedding cold/warm/cold with switch** at μ0.3 on Bifrost: 10.3 / 13.8 / 10.1 min (Fourier 26 / 38 / 24), all
  max_iter. Warm and cold minimise different weights, so no endpoint equivalence is claimed and no speedup is claimed.

## 5. Release candidate: requirement status

| item | status |
|---|---|
| Column-HSE delivery at an angle (CLI and embedding) | **met** |
| MHS validity and correctness coverage (angle Jacobians, controls, MPI, status) | **met** |
| Fixed-budget speedup (switch, opt-in) | **met**: 1.9–4× on the Bifrost cube |
| Qualified convergence of mhs-poisson-ltau on the Bifrost cube | **not met**: every μ ends `max_iter` in 25 steps; it qualifies on the consistent controls |
| Measured deployment runtime (8 nodes / 256 CPUs) | **unmeasured**: 6.5–11.4 min with switch on one node (96 cores); four nodes give 1.4–1.9× at fixed work |

The request is not complete. Qualified full-cube convergence and the 8-node runtime remain outstanding unless the user
accepts the reduced scope.

## 6. Questions

1. Is this a reviewable release candidate as it stands? Is anything missing before the user decides on merging?
2. The reference policy: Poisson equivalence against a same-machine standalone run, stored files as a record. Acceptable?
3. Exit code 2 for a valid `max_iter` atmosphere is new for the τ and z drivers, which used to exit 0. Callers that treat
   any non-zero exit as failure will see a change. Is that the right default, or should valid budget-limited output
   exit 0 with the status in the file?
