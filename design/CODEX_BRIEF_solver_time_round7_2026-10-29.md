# CODEX brief, round 7: delivery decision and result table for `mhs-poisson-ltau` (2026-10-29)

Branch `solver-time`. This round follows your round-6 plan (`design/CODEX_REVIEW_solver_time_round6_2026-10-29.md`):
- baseline matrix through the production interfaces;
- scaling at fixed work;
- the inner-accuracy decision experiment;
- one selected implementation;
- the three code findings.

Details and every number are in `design/tau/SOLVER_TIME_DESIGN.md` §6.14. The equations, Φ, the boundaries, gn_tol
1e-3 and the 25-step budget are unchanged.

## 1. Selected implementation

**`precond = :switch`** (opt-in; default still `:fourier`).
- It uses Fourier until the first step whose CG ends at its iteration limit, then column blocks for the rest of the solve.
- It is exposed through `PoissonTauOptions`, `[poisson]`, the CLI and the embedding API.
- It is chosen from the decision experiment below. No new preconditioner was written.

The additive branch two-level preconditioner is not selected. With 16² aggregates it reaches 1e-5 1.7–3× faster than
block on 128² crops. But it needs 17–23 s of setup per outer step, more than a whole loose (η ≤ 0.1) block solve, and its
coarse solve is serial. It stays experimental in `ml_lab.jl`.

## 2. Decision experiment (μ0.3 full cube, 10 steps per variant)

| start | variant | Φ after 10 steps (relative to its start) | CG its | solve s |
|---|---|---|---|---|
| column HSE | Fourier, η ≤ 0.1, cap 200 (baseline) | 1.8e-4 | 548 | 242 |
| | block, same forcing | 0.60 | 10 | 25 |
| | block, η = 1e-3, cap 3000 | 0.19 | 112 | 65 |
| baseline step 10 | Fourier (baseline) | 0.413 | 1384 | 557 |
| | block, same forcing | 0.255 | 345 | 124 |
| | block, η = 1e-3, cap 3000 | 0.242 | 3864 | 1116 |

- **Early steps.** Block steps (loose or tight) ask for cut-height changes far beyond the 300 km cap, so the whole-step
  scaling cuts them to 2–11 %. The truncated Fourier CG acts as implicit damping of those flat large-scale height
  modes.
- **Tail.** Block at the baseline forcing gives lower Φ at 4.5× less time. Tighter inner accuracy adds nothing per
  second.
- **Answer to your question.** Inexact inner solves are not what limits the tail. The cheaper, better-matched
  preconditioner is what helps, and only after the early phase.

## 3. Completion table

All runs below go through the production CLI on one node with 3 ranks × 32 threads. The cube is 504² × 95, with the
vertical-depth cut at −4, vertical balance, the D2 forcing, 25 steps and gn_tol 1e-3. Time is process wall time;
memory is the peak per rank.

| case | method | stop / status | Φ/Φ₀ | A products | time | memory |
|---|---|---|---|---|---|---|
| column HSE μ1, 0.5 φ45, 0.3, 0.2 | – | converged / converged | – | – | 53–54 s | 2.9 GB |
| MHS μ1 | fourier | max_steps / max_iter | 2.585e-2 | 2988 | 21.4 min | 6.0 GB |
| | switch | max_steps / max_iter | 2.579e-2 | 1421 | 11.4 min | 7.8 GB |
| MHS μ0.5 φ45 | fourier | max_steps | 2.217e-3 | 2780 | 20.2 min | 6.4 GB |
| | switch | max_steps | 2.152e-3 | 1022 | 8.0 min | 7.9 GB |
| MHS μ0.3 | fourier | max_steps | 6.34e-5 | 3137 | 19.8 min | 6.3 GB |
| | switch | max_steps | 4.33e-5 | 1302 | 8.9 min | 7.7 GB |
| MHS μ0.2 | fourier | max_steps | 2.85e-4 | 3590 | 26.1 min | 6.2 GB |
| | switch | max_steps | 2.14e-4 | 765 | 6.5 min | 7.5 GB |
| embedding μ0.3, cold / 2 % ln T warm / same cold | fourier | max_iter ×3 | – | – | 26 / 38 / 24 min | 6.3 GB |

**Against the user's targets.**

| target | status |
|---|---|
| Finite, valid output at every μ and azimuth tested | met: no abnormal CG exits, no EOS-limited steps, every field valid |
| Qualified gn_tol convergence on the Bifrost cube | **not met**: every MHS case ends `max_steps` (status `max_iter`) |
| A few minutes on 8 nodes / 256 CPUs | **unmeasured**: 6.5–11.4 min with switch on one node (96 cores); about 6–8 min estimated on four nodes for μ0.3 from the scaling below; 8 nodes not available |
| Time independent of μ | met for the budget-limited runs (switch 6.5–11.4 min across μ); not meaningful while no run converges |
| Column HSE delivered | met: about 1 min on one node; its own tests at μ 1, 0.8, 0.5, 0.3, 0.2 (§4) |

**Notes on the MHS rows.**
- The final decrement diagnostic is an 80-iteration solve with the run's last preconditioner. It reads 1–3e-2 for
  Fourier and 2e-2 to 1.3e-1 for switch. The more accurate block solve shows a larger decrement, so the Fourier values
  understate the distance from stationarity, and the two are not comparable.
- In the switch tails the steps are cut by the 300 km cap (α 0.15–0.4) with gain ratios 0.1–0.7. What remains is
  limited by the outer step.
- The warm start gives no saving on this input: the warm and cold solutions differ by 6.1 km rms in z(τ_LOS = 1).

**Scaling at fixed work.** μ0.3, 3 steps × exactly 100 CG iterations; seconds spent in the steps.

| nodes × ranks × threads | Fourier | block |
|---|---|---|
| 1 × 1 × 32 | 164 | 164 |
| 1 × 3 × 32 | 128 | – |
| 2 × 1 × 32 | 164 | 123 |
| 2 × 2 × 16 | 104 | – |
| 4 × 1 × 32 | 116 | 87 |

- Four nodes give 1.4× (Fourier) to 1.9× (block) over one node at 32 cores per node.
- Two ranks × 16 threads beat one rank × 32 threads per node.
- The Fourier transposes are the likely limit; this is not profiled.
- The hardware is mixed: the fourth node is an EPYC 7742; the others are 7543s.
- hyades16 could not host a rank (no InfiniBand, no node-local Julia).

## 4. Code findings and integration (commits 08484e0 … this round)

- **PCG finiteness.** The check now counts non-finite entries in the existing ‖r‖² collective. There is a test with a
  finite x whose sum overflows.
- **Round-off floor.**
  - Relabelled a numerical-precision acceptance heuristic: grid and gauge dependent, K untuned, never used for stalled
    runs.
  - Tests cover finite Φ and floor, and a +1 Mm translated gauge, which is accepted at once with the floor +0.7 %.
- **Smoother.** The power estimate is replaced by a B-Rayleigh lower bound (2.80 against a dense-spectrum 2.83) plus the
  provable block-stencil bound λmax ≤ 13. `omega_safe=1` gives the proven ω = 1/13, which raises κ of `mult` from 71
  to 272. `add` is SPD for any ω. Lab only.
- **Tiled tests.** The caveat is noted in §6.13/6.14: `ereg` depends on the domain, so the operator is not identical.
- **Public interfaces.**
  - `PoissonTauOptions`/`[poisson]` gain lm, forcing, cg_cap, eta_min, eta_max, precond (`switch` included), klow and
    damping.
  - Validation is collective. Defaults are unchanged, and the config hash is unchanged unless these are set.
  - The status and CLI output now carry:
    - the work (A and M⁻¹ products, CG iterations, abnormal exits, switch step, seconds);
    - roundoff and phi_floor;
    - the qualification outcome and bounds;
    - ranks and threads.
- **Column HSE.** Its own test set covers:
  - valid and converged at μ 1, 0.8 (φ 90), 0.5 (φ 45), 0.3 and 0.2 (φ 200);
  - inclined equal bitwise to plane-parallel (each line of sight balanced on its own, no horizontal force balance);
  - finite fields, positive P, ρ and χ, z decreasing and τ increasing down every ray, mean z(τ_LOS = 1) = 0;
  - an out-of-table temperature returning `eos_failure` / `eos_out_of_envelope`.

  MPI rank independence is in `test/mpi/mpi_view.jl` (not rerun this round).
- **Tests.** Block 106/106, warm 47/47, view 133/133, units/config 18/18, option validation 30/30.
  - The embedded API has 82/87. The 5 failures are the stored-reference equivalence of the Poisson modes and the
    allocation budgets. They fail identically on 08484e0 (6 there): the references predate this branch's solver
    changes.

## 5. Not done (release candidate blockers)

1. Regenerate the stored standalone references and allocation budgets for the embedded tests (only after the default
   is decided).
2. Rerun the MPI suites (`mpi_tests.jl`, `mpi_embedded.jl`, `mpi_view.jl`) on 2–4 ranks with the final options.
3. README and run examples: supported geometry, the input conventions, the resource settings, the stop semantics, and the
   measured limitations with this table.

## 6. Decisions for the user (proposed) and questions for you

1. **Default preconditioner.** Make `switch` the default on this release candidate? It is 1.9–4× faster for the same
   step budget at every μ tested, with equal or lower Φ, at +1.5 GB per rank.
2. **Budget-limited output.** On this Bifrost cube no μ qualifies in 25 steps with either method. Options:
   - (a) deliver as is: `max_iter` atmospheres, valid and labelled, with the decrement reported;
   - (b) a larger step budget (the user's decision);
   - (c) one outer-step adjustment.
3. **Is (c) justified, and which?** The tail is limited by the whole-step scaling to the 300 km cut-height cap, with gain
   ratios 0.1–0.7. A Steihaug–Toint truncated CG (stop the inner solve at the trust radius rather than scale a converged
   step) is the principled version of what Fourier truncation does implicitly early on. Is that the single adjustment
   to try, or should the cap and gain policy be left alone for delivery?
4. **Stop test preconditioner.** The decrement diagnostic and the qualification currently use the run's preconditioner.
   Should the qualification always use the column blocks? On this data the Fourier solve understates the decrement.
5. **Scaling.** Given 1.4–1.9× on four nodes, is profiling the Fourier transposes (or a pure-block path after the switch
   point, which scales better) in scope for delivery, or should the measured numbers simply be documented?

## 7. Where to look

- `design/tau/SOLVER_TIME_DESIGN.md` §6.14.
- `HSEMHS/src/PoissonTau.jl` `solve_tau` (`:switch`, `work.switch_step`).
- `HSEMHS/src/Embedded.jl` `PoissonTauOptions`, `option_errors`, `poisson_ltau!`.
- `HSEMHS/src/RunModes.jl` (CLI attributes).
- `HSEMHS/src/Config.jl` (`HASH_IF_SET`).
- `HSEMHS/scripts/delivery/`: `run_case.sh`, `scaling.sh`, `inner_accuracy.sh`, `embed_seq.jl`.
- Logs on the cluster: `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/delivery/` (`mhs_*`, `sw_*`, `col_*`, `scale_*`,
  `inner_*`, `embed_mu0.3.*`) and `out/frozen/*.labadd.log`.
