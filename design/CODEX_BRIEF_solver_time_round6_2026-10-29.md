# CODEX brief, round 6: μ-independent solve time for `mhs-poisson-ltau` (2026-10-29)

Branch `solver-time`. This round carries out your round-5 review
(`design/CODEX_REVIEW_solver_time_round5_2026-10-29.md`): A (baseline, exact start), B (two-level prototype),
C (measured against the pass criteria as far as the prototype allows), D (narrow correctness checks) and findings 1–5.
The details are in `design/tau/SOLVER_TIME_DESIGN.md` §6.13. Nothing in the objective, the equations or the boundary
conditions changed.

## 0. The user's framing (new, for your review)

The user's production target is 8 nodes with 256 CPUs: "as long as the convergence is finite and can be done within a
few minutes, it's okay; when the domain size increases, convergence time will increase."

The full-cube numbers we have (504² × 95, one node, 3 ranks × 32 threads, Fourier, D2 settings):

| | 25 steps | qualified? |
|---|---|---|
| μ1 | about 20 min | no |
| μ0.3 vertical | about 15 min | no |
| μ0.2 | about 43 min | no |

- On the Bifrost input no μ qualifies the gn_tol stop within 25 steps; the outcome is `max_steps`.
- At μ0.3 every full-cube step hits the CG cap of 400 with relative residual 0.3–0.7, so the steps are inexact.
- On the consistent control, the same solver qualifies in 4 steps (μ1) and 8 steps (μ0.3) within seconds.
- Strong scaling beyond 4 nodes is not measured; 4 is our limit.

## 1. Changes

| commit | content |
|---|---|
| 8f193cf | `phi_floor` round-off stationarity in `solve_tau`, with regressions (block 97/97, warm 47/47, view 87/87) |
| 6ce44d2 | `pcg!` finiteness folded into the ‖r‖² reduction (3 collectives per iteration); abnormal PCG exits logged and counted in `solve_tau` (`work.cg_abnormal`) and stop refinement in `qualify_stop` (`stop_check.pcg`); `ml_lab.jl` (frozen archive, two-level lab) |
| 1692c57 | branch coarse spaces; `defect_parts` (fixed-unit defect) at s0; the rotated control (`waves_x`, `waves_y`); `control_refine` compares B components and checks grids; summary labels; restart content fingerprint |
| this round | analytic Galerkin assembly, `rhs=` and `tile=` in `ml_lab`, the closure fix, design §6.13 |

**Exact start (A).**
- The round-off floor of Φ is each residual with its terms replaced by their magnitudes, times 64 eps, summed with Φ's
  own weights.
- A start at or below the floor returns `converged` with `info.roundoff = :start`.
- A no-decrease step from such a state ends `converged` (`:no_descent`) instead of `no_descent`.
- The gn_tol test, the budget and every other path are unchanged.
- Field-free controls now end `converged`: Φ ≈ 1e-19 against floors ≈ 5e-14, so the floor is conservative by about
  10⁵.
- No default `rms_atol` was chosen.

**Finding 4 (PCG cost).** The trial-x finiteness test now shares the residual-norm collective, as one `gsum` of
[‖r‖², Σx]. Single-rank PCG overhead beyond A and M⁻¹ is within timing noise; multi-rank collective cost is not
measured.

**Finding 5.**
- `control_refine.py` compares B_x, B_y, B_z and asserts the axes, x, y and the control parameters.
- `control_summary`'s "hit" is now labelled a retrospective scalar-error diagnostic.
- The restart metadata is described accurately (path, size, Σ log τ) and gains a content fingerprint: global sums of
  T, B and the coordinates. It is not a cryptographic checksum.

**Bugs found and fixed.** Two closure-capture bugs in prototype scripts: `fields` in `make_mhs_control` assigned an
outer variable from threads, and `frozen()` in `ml_lab` rebound `pr`. The earlier probe-based two-level numbers are
unaffected, because the rebound problem had the identical operator and column blocks.

## 2. Correctness checks (D, finding 1)

**Defect in fixed units** (`PoissonTau.defect_parts`). R_x, R_y and R_z/Δz are made dimensionless with a fixed
H_ref = 100 km, volume-weighted, and split into cut / interior / bottom layers. The defect is still about first order
in Δlog τ: at μ1 the interior vertical part goes 2.6e-2 → 9.3e-3 → 4.3e-3.
- At μ1 the interior dominates, not the end layers.
- At μ0.3 the bottom vertical layer dominates (0.08–0.13 at Δlog τ 0.1) and grows with nx. The one-sided bottom
  derivative is suspected.
- The solutions converge at second order (§6.12), so the first-order defect component appears not to drive the
  solution error. Not resolved.

**Rotated control.**
- Waves (2, 1), 6 Mm box, 128², φ 30°, so t = (2.75, 1.59) at μ0.3.
- The y-invariant output is bit-identical to the round-5 generator.
- Truth-boundary errors: 0.7 km (μ1) and 0.1 km (μ0.3). Production defaults: 5 steps at μ1.

**Perturbation return.** Smooth perturbations at the matched boundary (ln P 0.05, cut height 5 km; 27–32 km in
z(τ=1)) return to within 4 m (μ1) / 2 m (μ0.3), with ln P within 9e-5.

## 3. Two-level prototype (B, C)

**Method.**
- Damped symmetric column-block smoothing, with ω = 1/λmax(B⁻¹A) from power iteration.
- Horizontal a × a aggregates of whole columns. Galerkin A_c = PᵀAP from the actual A (ereg and Marquardt included).
  Sparse Cholesky. Applied as symmetric multiplicative (`mult`) or additive (`add`).

**Checks.**
- A_c by probes and A_c by the new analytic assembly agree to 1e-16 in four cases: μ1 ray, μ0.3 vertical, and the
  rotated control (nonzero t_y) in both balances. Both match PᵀAP from the dense A.
- M⁻¹ is symmetric (1e-15) and SPD on 8 × 8 systems.
- The analytic assembly forms J_RPᵀ W J_RP′ for every residual column R and its 5-column stencil, then scatters with
  the branch signs.

**Your Nyquist point was decisive.** Piecewise-constant aggregates leave the (−1)ⁱ, (−1)ʲ, (−1)ⁱ⁺ʲ branches of the
centred differences unrepresented. With modulated copies (`multb`) on the 8 × 8 μ0.3 crop:
- κ(M⁻¹A) falls from 1.4e4 to 59;
- iterations fall from 18/149 to 10/27;
- all four gauge modes are reduced to 0.015.

**Resolution independence** (control, fixed domain, nx 128 → 256 → 512, iterations to 1e-3):

| | block | multb:4 | multb:8 |
|---|---|---|---|
| μ1 | 9 → 15 → 31 | 5 → 6 → 6 | 7 → 10 → 11 |
| μ0.3 | 371 → 84 → 103 | 34 → 25 → 21 | 42 → 35 → 29 |

The control states differ slightly per grid (5 steps each), which explains block's irregular μ0.3 row.

**Domain independence.** A 32² Bifrost crop is tiled 1×, 2× and 4×, so the local state is identical and only the domain
grows. The GN right-hand side is tile-periodic and keeps every Krylov iterate periodic, so the right-hand side is a
seeded random one. Iterations to 1e-3:

| rhs | block | multb:8 | multb:16 |
|---|---|---|---|
| μ0.3, smooth | 993 → 1636 → 2718 | 184 → 206 → 211 | 225 → 248 → 259 |
| μ0.3, white | 982 → 1822 → 3007 | 184 → 195 → 196 | 225 → 231 → 236 |
| μ1, smooth | 342 → 635 → 1004 | 60 → 63 → 65 | 113 → 162 → 174 |

**Cost on the real GN systems** (128² crops, step 10; one rank, 32 threads, shared nodes so timings are ±30 %).
Iterations, then setup + solve seconds:

| | target | block | multb:16 |
|---|---|---|---|
| μ1 | 1e-3 | 123 its, 7.2 s | 30 its, 8.9 + 5.6 s |
| | 1e-5 | 845 its, 49 s | 64 its, 8.9 + 11.9 s |
| μ0.2 | 1e-5 | 1126 its, 46 s | 207 its, 6.4 + 23.5 s |
| μ0.3 | 1e-3 | 163 its, 8.2 s | 49 its, 8.1 + 9.6 s |
| | 1e-5 | 1605 its, 83 s | 384 its, 8.1 + 83 s |

- Analytic assembly takes 0.5–6.7 s; the sparse factorisation of A_c dominates setup for small aggregates.
- Total time wins at 1e-5 for μ1 (about 2.4×) and μ0.2 (about 1.5×).
- At μ0.3 there is no time gain: iterations fall 3–4×, but each iteration costs 3 A applications plus the coarse solve.
- The late μ0.3 Bifrost system still needs 223–384 iterations to 1e-5 with the branch coarse space, so part of its
  slow band lies outside it.

**Against your pass criteria.** The prototype does not yet pass: no μ0.3 total-time improvement on the representative
difficult case, and no end-to-end trajectory. Positives: resolution and domain independence on clean tests, exact
assembly, and a setup cost now comparable to tens of block iterations.

**Known gaps.**
- The full-cube coarse problem, 4·n₁·(504/a)² (800 k unknowns at a = 8, 200 k at a = 16), is too large for one direct
  factorisation. It needs a third level or a compact basis.
- Our compact basis (the k smallest eigenvectors per group) loses the gauge modes: κ 7.9e3 on 8 × 8.
- The coarse solve is single-rank.

## 4. Questions

1. **Given the user's framing (§0), what should the next step be?**
   - (a) Strong scaling of the current solver on 4 nodes, to predict 8-node production time.
   - (b) The multilevel work, which on Bifrost helps μ1 and μ0.2 more than μ0.3.
   - (c) Neither: the obstacle on real data is the unqualified stop (a flat Φ tail on a non-equilibrium snapshot), and
     faster linear algebra does not shorten it.

   Is inexact full-cube stepping at μ0.3 (CG cap, relative residual 0.3–0.7) a plausible cause of extra Gauss–Newton
   steps that a better preconditioner would remove? How would you test that cheaply?
2. **The μ0.3 slow band outside the branch coarse space.** What would you add?
   - Per-level horizontal shapes rather than whole-column constants?
   - Smoothing along the slant direction (line or column-pair blocks along t)?
   - An additive version with cheaper iterations (one A per iteration, at about 2× the iterations)?
3. **Hierarchy for the full cube.** Three levels (recursively aggregating A_c with the same branch construction), or a
   compact basis that keeps the branches and the cut-height modes explicitly? How should the coarse solve be
   distributed over the x slabs?
4. **Defect order.** The fixed-unit defect is first order in Δlog τ, interior at μ1 and bottom at μ0.3, while the
   solutions converge at second order. Is a first-order, solution-neutral residual pattern plausible here, for example
   from linear interpolation of the truth onto τ nodes or from the centred secant on non-uniform heights? Does it matter
   for this work?
5. **The round-off floor.** It is about 10⁵ above the Φ reached on exact starts. Is K = 64 with neighbour differences
   estimated from local values acceptable, or should it be tightened?

## 5. Where to look

- `design/tau/SOLVER_TIME_DESIGN.md` §6.13.
- Code:
  - `HSEMHS/src/PoissonTau.jl`: `phi_floor`, `PCG_ABNORMAL`, `defect_parts`, `qualify_stop`, `solve_tau`.
  - `HSEMHS/src/PoissonZ.jl` `pcg!`.
  - `HSEMHS/scripts/viewing/ml_lab.jl`: `galerkin`, `galerkin_analytic`, `jac_RP!`, `twolevel`, `ml_apply!`.
  - `HSEMHS/scripts/viewing/make_mhs_control.jl`.
- Logs on the cluster, `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/`:
  - `frozen/*.lab*.log`, which also holds the frozen systems;
  - `control/rot_*`, `df_*`, `ro_*`.
