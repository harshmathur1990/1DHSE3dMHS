# CODEX brief, round 2: μ-independent solve time for `mhs-poisson-ltau` (2026-10-28)

Follow-up to `design/CODEX_BRIEF_solver_time_2026-10-28.md` and your review
`design/CODEX_REVIEW_solver_time_2026-10-28.md`. Branch `solver-time`. main and embedding-api are untouched; they are
production for FFNOML. I followed your priority list.

- **Done:** steps 1–3, and the stop-qualification part of step 4.
- **Partly done:** step 5. Its coarse-space lab is still running; the results so far are in §3.3.
- **Not started:** the step-4 reference runs and step 6 (full-cube benchmarks, D4b).

The goal is unchanged. The user wants the time to converge to be about the same at any μ as at μ = 1. The defaults gn_tol
1e-3 and 25 steps stay; FFNOML may override them.

## 1. What changed

### Commit 2c3d99b: groundwork for review steps 1–3
- **Marquardt damping in the Fourier part of `:hybrid`** (`pr.fdamp`, default true). The layer matrix is now
  `A = GzᵀWfGz + diag(ereg + mlm · median_layer(diag JᵀWJ))`, so the Fourier part sees the same damping as the operator
  and the column blocks.
- **A common objective for different starts.** `solve_tau(...; weights = (wc, wf, ereg))` replaces the weights
  `tp_weights!` would derive from the start. The weights a solve used come back in `info.weights`. Cold and warm starts
  can now minimise the same Φ.
- **`measure_warm.jl` (your finding 3).**
  - It keeps immutable copies of the column-HSE baseline and errors if the baseline changes.
  - It evaluates the real step-0 heights of cold and warm starts with `tp_setup` + `evaluate!`.
  - `common_weights=1` passes the cold solve's weights to the warm solve.
  - The D4a numbers in the first brief came from the aliased script. Treat them as D4a only. D4b has not been run yet.
- **`scripts/viewing/frozen_lab.jl`** (new). It runs STEPS D2 steps, then freezes the linearisation with that
  trajectory's weights and regularisation. Its parts:
  - **checks:** adjoint; finite differences for ε = 1e-2 … 1e-6; symmetry and positivity of A and of each M⁻¹; the
    Marquardt column block against operator probes.
  - **gauge:** the cut-height modes 1, (−1)^i, (−1)^j, (−1)^(i+j).
  - **tight:** tight PCG with each preconditioner, and the pairwise differences between the resulting δ, including the
    |k| ≤ 2 part of δc.
  - **direct:** on a crop only, a dense direct solve, Cholesky, and the smallest eigenpairs.
  - **marq:** the Marquardt ablation.
- **Tests:** weights reuse, the baseline copy, and API/script agreement (`test_block.jl`, `test_warm.jl`).

### Commit cc60256: the stop certificate (your finding 1, step 4)
When `stop = :gn` would trigger (pred < gn_tol on gn_consec steps), `qualify_stop` solves the undamped system at the
current state:
- it uses ereg only and mlm = 0;
- it solves to `qual_tol` (1e-2) with at most `qual_maxit` (400) iterations;
- it checks the true residual, ‖b − Aδ‖ from `Aop!`.

The outcome:
- the true residual exceeds 2·qual_tol: reason `:unqualified`, reported as status `stalled` by Embedded;
- the qualified pred is below gn_tol: `:converged`;
- otherwise the counter resets and the solve continues.

`info.stop_check` records (pred, model decrease, ereg term, its, true_rel). The threshold, count and budget are the
user's and are unchanged.

Regression tests:
- heavy damping gives a small pred but a large qualified pred;
- `qual_maxit = 1` is not certified;
- `solve_tau` returns `:unqualified`.

Test status: block 44/44, warm 47/47, view 87/87. test_embedded on main still has the older failures noted earlier; they
need the user's approval to fix on main.

### Not yet committed: `coarse_lab` in `frozen_lab.jl` (step 5)
1. **Ritz pairs from the Lanczos process.** It runs 300 iterations of block PCG at mlm = 0, using
   q_j = z_j/√(r_jᵀz_j) with T from (α, β). For the 32 smallest Ritz pairs it reports θ, the residual
   ‖M⁻¹Ax − θx‖/(θ‖x‖), the cut-height share, the |k| ≤ 2 share, the share in the 4 gauge modes, and the mean depth of
   the ln P part. All norms are scaled, with c/1e5 m.
2. **Balanced deflation** M⁻¹ = Q + (I − QA)B(I − AQ), with Q = Z(ZᵀAZ)⁺Zᵀ and B the exact column blocks. It is tried
   for these coarse spaces Z:
   - the 4 gauge modes;
   - cut-height Fourier modes with |k| ≤ 2 (16 modes, gauge included) and |k| ≤ 4 (52);
   - per-level Fourier modes on every level with |k| ≤ 1 (5n₁ + 3) and |k| ≤ 2 (13n₁ + 3);
   - the 8 and the 32 smallest Ritz vectors.

   AZ is formed by explicit operator products. The pseudo-inverse uses an eigenvalue cutoff of 1e-12. The projections
   are dense matrix products with one Allreduce each; this is not tuned for speed.

## 2. Where it ran
Three full-cube cases ran on hyades4/5/6, one node each, with 3 MPI ranks × 32 threads. The 16 × 16 crops ran on
hyades20 with 1 rank. All nodes were free before launch.

| Case | μ | Geometry and balance | log τ cut (line of sight) | Grid | Unknowns |
|---|---|---|---|---|---|
| full_mu1 | 1 | inclined, ray | −4.0 | 504² × 95 | 13.97 M (n₁ 55) |
| full_mu0.3v | 0.3 | inclined, vertical | −3.5 | 504² × 95 | 12.70 M (n₁ 50) |
| full_mu0.2r | 0.2 | inclined, ray | −3.3 | 504² × 95 | 12.19 M (n₁ 48) |
| crop16_mu1, crop16_mu0.3v | 1 and 0.3 | as above | | 16² × 95 | 12 800 |

In every case the frozen state is the one after 15 D2 steps with the `:fourier` preconditioner, `lm = :gain` and
`forcing = :ew`. Logs are in `out/solver/lab/*.log` on the cluster worktree `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS`.

## 3. Results

### 3.1 Algebra and gauge (review steps 2 and 5a): sound
- **Adjoint:** 4e-14 (μ1), 1e-15 (μ0.3v), 2e-15 (μ0.2r).
- **Finite differences:** smallest error 5e-11 to 6e-10 at ε = 1e-3 … 1e-4. The 5-levels-below-the-cut probe behaves
  the same.
- **Symmetry:** A and every M⁻¹ (fourier, block and hybrid; mlm 0 and 1; fdamp false and true) are symmetric to 2e-13
  and positive on random vectors.
- **Marquardt column block:** equal to operator probes to 6e-16.
- **Gauge modes:** on the full 504² grid and on the crops, the constant, (−1)^i, (−1)^j and (−1)^(i+j) cut-height modes
  give ‖W½Jv‖ = 0 exactly. Only ereg holds them. For comparison, ‖W½Jv‖/‖v‖ is 25–225 for cos(kx = 1) and about 2.7e3
  for a random v.
- **The 16² crops agree with the direct solve.**
  - Tight PCG matches the dense direct solve: fourier 1e-7 and block 8e-8 (μ0.3v); hybrid 7e-8 at μ1.
  - Hybrid at μ0.3v stopped at its 3000-iteration cap at 7e-5 and differs by 2e-3.
- **Crop eigenvalues** (scaled H, μ0.3v):
  - 4 exact zero modes, entirely cut height. With ereg added they sit at λ/λmax 1.1e-6.
  - The next modes are deep ln P modes (log τ 1.3–1.4) at λ/λmax 5–6e-6.
  - ereg spans 70 … 190 in scaled units.

Conclusion: the linear algebra is correct. The earlier D3b-versus-D2 height difference was a trajectory effect.

### 3.2 Full-cube frozen systems (review steps 2–3)

**Tight solves** (tol 1e-8, at most 2000 iterations):

| Case | fourier | block | hybrid | Exact step: max\|δu\|, max\|δc\|, cap scale | δc difference between solves, rms (\|k\| ≤ 2 part) |
|---|---|---|---|---|---|
| μ1 | 1.2e-3 | 6.3e-9 (1872 its) | 5.1e-6 | 13, 388 km, 0.15 | 0.03–0.10 km (0.000) |
| μ0.2 ray | 2.0e-2 | 9.2e-5 | 8.7e-4 | 12, 330 km, 0.16 | 15–20 km (14–18) |
| μ0.3 vertical | 0.44 | 5.1e-3 | 5.8e-2 | 3.7, 232 km, 0.54 | 11–15 km (8–12) |

The residuals are true relative residuals. At small μ no preconditioner reaches 1e-8 in 2000 iterations. The
unresolved part is the large-scale (|k| ≤ 2) cut height.

**Marquardt ablation**, CG iterations to tol 1e-3 with a cap of 600 (an asterisk means the cap was hit; the value
reached is in brackets):

| mlm | μ1 block | μ1 hybrid, fdamp off → on | μ0.2r block | μ0.2r hybrid, off → on | μ0.3v block | μ0.3v hybrid, off → on |
|---|---|---|---|---|---|---|
| 0 | 131 | 205 → 205 | 275 | 600* → 600* | 600* (1.1e-2) | 600* → 600* |
| 0.01 | 66 | 208 → 102 | 91 | 600* → 559 | 141 | 600* → 508 |
| 0.1 | 38 | 319 → 51 | 44 | 600* → 266 | 68 | 600* → 244 |
| 1 | 14 | 444 → 28 | 15 | 600* → 141 | 26 | 600* → 108 |
| 10 | 5 | 564 → 15 | 5 | 600* → 74 | 7 | 600* → 46 |

The cap scale at mlm = 0, 0.01, 0.1, 1 and 10:
- μ1: 0.15, 0.16, 0.19, 0.43, 1;
- μ0.2r: 0.16, 0.18, 0.25, 0.71, 1;
- μ0.3v: 0.54, 0.80, 1, 1, 1.

What this shows:
- Damping in the Fourier part fixes the hybrid's deterioration as mlm grows (your finding 2 confirmed).
- Block alone is still faster than any hybrid at every mlm and μ. I consider the additive hybrid retired.
- With mlm ≥ 0.01, block converges in ≤ 141 iterations at every μ.
- At mlm = 0.01, pred is 0.107 (μ1), 0.32 (μ0.2r) and 0.385 (μ0.3v). The undamped values are 0.115, 0.42 and 0.51. So
  mild damping keeps 75–95% of the predicted decrease and costs a third or less of the CG work.

### 3.3 Slow modes and balanced deflation (review step 5): partial; the lab is still running

**Lanczos** (block PCG, 300 iterations, mlm = 0):

| Case | κ(M⁻¹A), Ritz estimate | θ_min … θ_max | Smallest θ |
|---|---|---|---|
| μ1 | 2.5e4 | 1.9e-4 … 4.7 | 1.9e-4, 5.5e-4, 1.2e-3, 1.9e-3, 3.0e-3, … |
| μ0.2r | 3.6e4 | 1.3e-4 … 4.5 | 1.3e-4, 5.5e-4, 1.1e-3, 2.0e-3, … |
| μ0.3v | 3.4e4 | 1.5e-4 … 5.0 | 1.5e-4, 5.8e-4, 1.2e-3, 2.1e-3, … |

- The small end is a dense ladder, growing roughly as n^1.7 (θ_n/θ_1 ≈ 2.9, 6.3, 10, 16 for n = 2…5). That is the shape
  of a Laplacian-like large-scale spectrum rather than a few isolated outliers.
- The smallest Ritz vector:
  - μ1: cut-height share 1.00, |k| ≤ 2 share 0.56;
  - μ0.2r: 0.93 and 0.68;
  - μ0.3v: 0.67 and 0.35, ln P mean depth log τ −0.9.

  Its share in the 4 gauge modes is at most 0.06.
- The Ritz residuals are large (17–58 relative to θ), so the vectors are only rough.
- Puzzle: κ is about the same at all three μ, yet the CG work is not. My guess is that the right-hand side at small μ
  loads the low end much more heavily. I have not checked this.

**Balanced deflation**, CG iterations to 1e-3 and 1e-5 (caps 600 and 1200):

| Z (dimension) | μ1, 1e-3 / 1e-5 | μ0.2r, 1e-3 / 1e-5 | μ0.3v, 1e-3 / 1e-5 |
|---|---|---|---|
| none (block, §3.2) | 131 / – | 275 / – | 600* (1.1e-2) / – |
| gauge (4) | 135 / 587 | 272 / 1200* (3.1e-4) | 600* (9.9e-3) / 1200* (6.6e-3) |
| c, \|k\| ≤ 2 (16) | 133 / 548 | 226 / 1200* (4.1e-5) | 600* (1.4e-2) / 1200* (4.4e-3) |
| c, \|k\| ≤ 4 (52) | 108 / 563 | 229 / 1200* (3.2e-5) | running |
| all levels, \|k\| ≤ 1 (≈ 5n₁) | running | running | running |
| all levels, \|k\| ≤ 2 (≈ 13n₁) | running | running | running |
| Ritz 8, Ritz 32 | running | running | running |

- ZᵀAZ conditioning for the cut-height sets: 2.7e2 (μ1), 1.8e4 (μ0.3v), 4.2e4 to 4.0e5 (μ0.2r).
- Coarse spaces of cut height alone do not remove the small-μ cost. They save at most about 20% at μ1 and μ0.2r and
  nothing at μ0.3v.
- My working hypothesis is that the soft directions are cut height coupled to ln P through the column. The per-level
  and Ritz rows test this. I can append them when they finish if you want them before replying.

## 4. Interpretation, uncertainties and things that failed
1. **The bottleneck is the linear solve at mlm = 0, not the algebra.** At small μ the undamped GN system has a
   large-scale near-null space that block Jacobi cannot propagate.
2. **The undamped GN step is far outside the trust region.** At 15 steps the caps cut it to 15–55%. So solving it
   tightly is wasted work, and the trajectories decide how well each method does. The earlier D3c Marquardt runs were
   slow (μ1 4082 s versus D2/D3h about 1200 s) even though the damped CG is cheap here. So the cost there must have come
   from the outer loop: retries, more steps and the line search. I have not separated these yet.
3. **The qualified stop has a gap.** `qualify_stop` solves the undamped system to 1e-2 in at most 400 iterations. At
   μ0.3v block does not reach 1e-2 in 600 iterations at the 15-step state. Near convergence the right-hand side is
   smaller, but its large-scale content may still make the check fail and return `:unqualified` → `stalled`. I have not
   measured this near a converged state yet.
4. **The lab deflation is unoptimised.** Each iteration takes up to 2.5× longer with 52 vectors. Compare the iteration
   counts, not the times.
5. **Nothing is end-to-end yet.** There are no step-4 reference runs and no step-6 benchmarks. Every number above is at
   one frozen state per case.

## 5. Questions for you
1. **Coarse space.** Given the dense, roughly quadratic low-θ ladder and the failure of cut-height-only Z:
   - Should the coarse space couple cut height and ln P per column, as aggregation or "column-shape" vectors? For
     example, the low-k c modes extended down each column by the column-HSE response, or a two-level method whose
     coarse grid is the 2-D Poisson-like operator for c with ln P eliminated per column (a Schur complement through
     the column blocks).
   - Or is the right move multigrid in the horizontal (column-block smoother, coarse 2-D grid) rather than deflation?
2. **Damping instead of deflation.** Is a small fixed Marquardt term (mlm ≈ 0.01–0.1) in the outer loop scientifically
   acceptable as the main remedy? It leaves the slow large-scale correction to the GN iteration and makes CG
   μ-independent (≤ 141 iterations). If so, what outer strategy should replace the D3c gain-ratio and cap logic, which
   was slow end to end? Options: mlm fixed and decreasing with gain; a trust region on the scaled norm; a dogleg with a
   Cauchy point from the block-preconditioned gradient.
3. **The stop certificate at small μ.** If the undamped qualification solve cannot reach 1e-2 cheaply at μ0.3, what
   should certify the stop? A damped qualification with the damping reported, a Ritz-based bound, or a cap on the
   certificate work, after which the run is reported as budget-limited?
4. **The puzzle.** Is it plausible that κ(M⁻¹A) is μ-independent while the CG work grows because of how the
   right-hand side is spread over the spectrum? What diagnostic would you want: the spectral distribution of b from the
   Ritz pairs, or residual projections onto the low-k subspace per iteration?
5. **Gauge.** The four null cut-height modes are held only by ereg (scaled 70–430). Should they be removed explicitly
   (projection, or pinning the mean and the three checkerboards of δc) rather than regularised? Pinning them changes
   nothing scientifically: Φ is invariant along them.

## 6. Where to look
- `HSEMHS/src/PoissonTau.jl`:
  - `setprec!` (fdamp), `Minv!`, `colblocks!`;
  - `qualify_stop` and the `stop = :gn` branch of `solve_tau` (keywords `weights`, `qualify`, `qual_tol`,
    `qual_maxit`).
- `HSEMHS/scripts/viewing/frozen_lab.jl`: `main` holds the parts; `coarse_lab` holds the Lanczos/Ritz analysis and the
  BNN runs.
- `HSEMHS/scripts/viewing/measure_warm.jl`: the baseline copies, `zstart` and `common_weights`.
- `HSEMHS/test/test_block.jl`, `HSEMHS/test/test_warm.jl`.
- Logs: `out/solver/lab/{full_mu1,full_mu0.3v,full_mu0.2r,crop16_mu0.3v,coarse_*}.log` in the cluster worktree.
