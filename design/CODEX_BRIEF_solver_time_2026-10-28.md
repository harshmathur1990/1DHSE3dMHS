# CODEX brief: μ-independent solve time for `mhs-poisson-ltau` (2026-10-28)

Branch `solver-time` (HEAD af4d565). Design and earlier measurements: [`tau/SOLVER_TIME_DESIGN.md`](tau/SOLVER_TIME_DESIGN.md).
Please answer in `design/CODEX_REVIEW_solver_time_2026-10-28.md`.

**Goal (user):** a cold `mhs-poisson-ltau` solve of an inclined view (μ = cos θ down to 0.2, any azimuth) should
converge in a time comparable to μ = 1, and that time should not depend on μ. Targets: about 10–20 steps and ≤ about
15 min on one node for 504² × 95, with z(τ=1) within about 1 km rms of the converged state. Warm starts (inversion loop,
`solve_embedded!(…; start = :previous)`) should need a few steps.

**Constraints:** `main` and `embedding-api` are in production (FFNOML), so everything new is opt-in on this branch.
Defaults the user fixed: stop when the predicted decrease is < 1e-3 on 2 steps, or after 25 steps, both overridable.
Runs use 3 MPI ranks (x slabs) × 32 threads on one node; there is no parallel HDF5.

---

## 1. The problem being solved

Code: `HSEMHS/src/PoissonTau.jl` (`evaluate!`, `Jop!`, `JTop!`, `Aop!`, `setprec!`, `colblocks!`, `solve_tau`).

**Grid.** The grid is the τ grid along the line of sight: 504 × 504 columns and 95 levels, with log10 τ from about −6 to 2.
Inclined columns are straight rays with slant t = tan θ (cos φ, sin φ). Above the cut (log10 τ_LOS = cut − log10 μ,
"vertical-depth cut"), the column structure is fixed relative to the cut height c(x, y). Below the cut there are
n₁ − 1 unknowns ln P per column plus c: n₁ is about 55–75 depending on μ.

**State.** Heights follow from the τ-scale: z_{k+1} = z_k − Δlτ/2 (a_k + a_{k+1}), with a = μ τ / χ(T, P). So δz at level m
is δc plus a lower-triangular (cumulative) sum of aχ·δu.

**Residuals per node.**
- Horizontal force balance per unit mass:

      R_x = −∂_x ln P|_τ − W ∂_x z|_τ + L_x / P
      R_y = (the same in y)
      W = g ρ/P − L_n / P

  L is J × B on the τ levels and L_n is its component along the ray. Horizontal derivatives are centred differences
  with periodic x halos between ranks.
- Column balance on faces:

      R_z = Δu − Δlτ/2 (W a)_f + 2

  The "+ 2" means two terms:
  - "ray balance": as written;
  - "vertical balance" (D0): minus Δz · t·R̄_h.

**Objective.** Φ = Σ wc (R_x² + R_y²) + Σ wf R_z². The weights are layer medians of H² Δz. The Lorentz terms depend on
z through the ∂_x|_z = ∂_x|_τ − ∂_x z ∂_ray transformation. This gives the px/pXx/zζ… coefficients: a δz map couples
into the 4 horizontal neighbours.

**Gauss–Newton step.** (JᵀWJ + ereg + μ_LM D) δ = −JᵀWR, solved with PCG.
- ereg = 1e-2 κ² (wc W̄²; wc), with κ = 2π/L the box wavenumber.
- The step is scaled as a whole to |δu| ≤ 2 and |δc| ≤ 300 km ("caps"), then a backtracking line search (4 halvings).
- LM damping uses the gain ratio (D2).
- CG tolerance uses Eisenstat–Walker forcing (η = clamp(pred, 0.005, 0.1), cap 200 iterations).
- pred = δ·(−JᵀWR)/Φ is the stationarity measure (D1 stop).

**Preconditioners in the code.**
- `:fourier`: per horizontal mode, the layer-median field-free operator A + |k|² B (A from R_z, B from R_h), diagonalised
  once per solve.
- `:block` (D3b): the exact per-column n₁ × n₁ diagonal blocks of JᵀWJ + ereg, assembled analytically from the column's
  own rows and its 4 neighbours' centred-difference rows. They are verified against operator probes to 1e-10 and
  Cholesky-factored per column. Building them costs about a quarter of a step.
- `:hybrid` (D3c): block Jacobi + the Fourier preconditioner restricted to |k| ≤ klow = 8, added (additive two-level).
- `damping = :marquardt` (D3c): D = diag(JᵀWJ), and a step the caps would scale below 0.5 counts as outside the trust
  region (μ is raised and the step solved again).

## 2. Measurements (504² × 95 synthetic Bifrost cubes, 40 steps, one hyades node each)

Truth for z(τ=1) is the Bifrost atmosphere. "Strong−quiet" is the mean z(τ=1) difference between strong-field and
quiet columns (Wilson depression; truth −76.6 km at μ 1, −48.8 at μ 0.5, −36.4 at 0.3, −29.9 at 0.2).
The column-HSE start has an error of 20–25 km rms at every μ.

### 2.1 Wall time and quality, cold start, 40 steps

| Case | D2 (`:fourier`) | D3b (`:block`) | D3c (`:hybrid` + Marquardt) | D3h (`:hybrid`, LM as D2) |
|---|---|---|---|---|
| μ 1 | 2421 s, 22.5 km, −74.0 | 416 s, 33.3 km, −38.4 | 4082 s, 35.4 km, −8.8 | **1210 s, 22.5 km, −75.0** |
| μ 0.5 ray | 2109 s, 26.4, −36.8 | 662 s, 33.1, −16.1 | 2030 s (conv. at 36), 26.4, −36.5 | – |
| μ 0.5 φ45 ray | 2497 s, 31.6, −40.2 | 836 s, 32.0 | 2794 s, 32.0, −40.4 | – |
| μ 0.3 ray | 2473 s, 24.1, −27.1 | 361 s, 36.8 | 6261 s, 40.2, +19.2 | – |
| μ 0.3 vertical | 1891 s, 21.0, −12.1 | 237 s (no descent at 29), 25.9 | > 6600 s at step 33 | – |
| μ 0.2 ray | 3556 s, 32.2, −1.5 | 271 s, 26.5 | > 6800 s at step 31 | – |
| μ 0.2 vertical | 4864 s, 19.6, +2.3 | 195 s, 22.2 | 5855 s, 49.0, +22.6 | – |

Each cell gives wall time, z(τ=1) error in km rms, and strong−quiet in km.

What the table shows:

- **D3b** makes CG cheap (1–10 iterations to rel 0.1; precond lab at μ 1: 10 iterations to 1e-1 and 104 to 1e-3, against
  147 / >600 with `:fourier`). Φ falls as far as with D2 (μ 1: Φ/Φ₀ 2.56e-2 against 2.55e-2), but **the solution is wrong**:
  z(τ=1) differs from D2 by 24 km rms. 74 % of that power is at |k| < 2 and 26 % at 2–8; the error grows with |B|; and the
  Wilson depression is half of the truth. Block Jacobi cannot propagate large scales. Their curvature in Φ is ∝ k², so they
  hardly show in the residual, and the iteration drifts along a flat valley.
- **D3h** (hybrid, same LM as D2) at μ 1 removes the drift: the same answer as D2 (22.5 km, −75.0) in half the time, with
  CG at 7–60 iterations. Its trajectory still reaches a state 1 km from final only at step ~30 (D2: step ~15), so it does
  not converge in fewer steps.

  | step | 5 | 10 | 15 | 20 | 25 | 30 | 40 |
  |---|---|---|---|---|---|---|---|
  | D2: cum s / Δ from final (km rms) | 111 / 6.2 | 271 / 2.6 | 564 / 0.8 | 866 / 0.5 | 1191 / 0.24 | 1624 / 0.12 | 2421 / 0 |
  | D3h: cum s / Δ from final | 85 / 14.7 | 176 / 11.1 | 285 / 7.6 | 469 / 2.2 | 645 / 1.5 | 846 / 1.1 | 1210 / 0 |

- **D3c (Marquardt)** is worse everywhere. CG goes to the 200 cap with relative residual 0.5–2.4 (> 1 at μ ≤ 0.3), so
  steps take 100–300 s. At μ 1 the Wilson depression collapses to −8.8 km.
- **μ ≤ 0.3, every variant:** not converged after 40 steps. The strong-field depression is still moving, pred stays at
  1e-2 – 1. step_lab at μ 0.3 vertical (after 15 D3b steps):
  - the accurate block-preconditioned GN step has max|δu| = 44 and |δc| about 1000 km, concentrated at log τ −3.4…−3.0 (just below
    the cut −3.5), with pred 0.999;
  - the Fourier-preconditioned step has max|δu| 0.4 and pred 0.997;
  - the old damping shape (ereg ∝ κ² wc) does not tame the large step even at μ_LM = 100.

  So the undamped GN problem has near-null directions just below the cut, and the caps then cut α to 0.02–0.1.
- CG with `:fourier` at μ 0.2 (state after 15 steps): rel 0.2–0.5 after 400 iterations (ray), and it does not reach 3e-2 in 600
  iterations (vertical). `:block` reaches 1e-2 in 51–105 iterations. The "add" variant (block + full Fourier, additive) is as bad
  as Fourier alone, which hints that the additive combination over-corrects where both parts act.

### 2.2 Warm start (D4a: ln P below the cut shifted by the change at the cut; D2 solver)

T is perturbed by ε (1 % or 3 %, smooth). Each entry is z distance from the cold 40-step solution of the perturbed cube
(km rms), with the cumulative time.

| Case | start | cold, 10 steps | warm, 5 steps | warm, 10 | warm, 25 |
|---|---|---|---|---|---|
| μ 1, ε 3 % | 16.8 (both) | 2.9, 300 s | 27.2, 152 s | 4.5, 552 s | 2.5, 1583 s |
| μ 0.5, ε 1 % | 23.3 | 6.8, 274 s | 2.1, 266 s | 1.8, 555 s | 1.7 |
| μ 0.5, ε 3 % | 24.5 | 4.1, 266 s | 36.7, 140 s | 15.3, 262 s | 2.3 |
| μ 0.5 φ45, ε 1 % | 29.1 | 11.1, 594 s | 1.8, 373 s | 1.2, 761 s | 1.1 |
| μ 0.3, ε 1 % | 17.0 | 13.7 | 14.4 | 16.2 | 17.4 |
| μ 0.2, ε 1 % | 14.4 | 10.1 | 11.5 | 12.8 | 14.8 |

The warm start moves away from the answer in the first steps at ε 3 %: the first GN steps blow up z_max to 400–870 km.
The column-HSE start of the warm (D4b, implemented, not yet measured) adds the column response of the new T. At μ ≤ 0.3 the
cold reference itself is not converged, so the warm numbers there say nothing.

## 3. Questions

1. **Diagnosis.** Is the μ ≤ 0.3 behaviour (near-null directions just below the cut, a huge GN step, pred near 1, tiny α)
   a property of the problem? For example, ln P below the cut weakly determined once the column above the cut is fixed and
   the ray is long in x, or the Lorentz terms ∝ tan θ making JᵀWJ nearly singular in some column-coupled direction. Or is it
   a defect of the formulation that should be fixed in the equations, for example:
   - the cut boundary condition (c free, ln P at the cut fixed);
   - the weights;
   - the vertical balance coupling Δz t·R̄_h, which makes R_z depend on the horizontal residuals.

   What would you compute to tell these apart (e.g. the smallest eigenpairs of a few column blocks, or Lanczos on JᵀWJ from
   CG)?
2. **Regularisation instead of damping.** If those directions are physically undetermined, what is the right prior? Options:
   - Tikhonov on δu weighted by depth below the cut;
   - smoothness of ln P − ln P_colHSE along the ray;
   - a penalty on c − c_colHSE at low k;
   - a trust region in the H̄-scaled norm instead of the hard caps.

   Should the regularisation be in Φ itself, so that the converged answer and the stop rule see it, rather than only in
   the step?
3. **Preconditioner.** For an operator that is "column-local blocks + a weak, large-scale coupling":
   - Is additive block + low-k Fourier the right two-level form, or should it be multiplicative / deflated / a Galerkin coarse
     space (coarse space = low-k modes × layer-median vertical eigenvectors, coarse operator = projected JᵀWJ, not the
     field-free median)?
   - Which klow and how many vertical vectors?
   - The cost of a coarse Galerkin projection with the existing distributed FFT is the question for the 3-rank layout.
4. **Why Marquardt failed.** diag(JᵀWJ) scaling plus "raise μ if the caps would bite" made CG much worse (rel > 1 at 200
   iterations) and made the answers worse at μ 1. Is that expected? Is our PCG still correct when the block preconditioner is
   rebuilt for every μ retry, or is something off (e.g. the Fourier part of `:hybrid` not including μ·dlm)?
5. **Outer iteration.** GN with gain-ratio LM still needs ~30 steps at μ 1 to get within 1 km, with ρ ≈ 0.1–0.5 in the tail.
   Is the tail nonlinearity (the z-dependence of the Lorentz terms through ∂_x z) better handled by:
   - a continuation in B (scale L by s → 1);
   - in tan θ (solve at μ 1, then step the slant);
   - a Newton–Krylov with second-order terms;
   - a nonlinear preconditioner (column-HSE relaxation sweeps between GN steps)?

   Which of these would give a step count independent of μ?
6. **Stop rule and accuracy.** Given that Φ is flat along the large-scale modes, is pred < 1e-3 a meaningful convergence
   test, or should the stop also bound the low-k component of δ (e.g. ‖P_low δc‖ < 1 km)?
7. **Warm start.** Why would the warm start from a converged neighbour state make the first GN steps blow up at ε 3 %? Is
   this the same near-null issue? What warm start would you use?
8. **What to try first.** Rank the options by expected gain over cost; we will implement and measure in that order. Please
   say what you expect each to show if it works, so that the measurement can falsify it.

## 4. Where to look

- `HSEMHS/src/PoissonTau.jl`:
  - `evaluate!` (residuals and Jacobian coefficients);
  - `Jop!`/`JTop!`/`Aop!`;
  - `setprec!`/`fourier_minv!` (layer-median operator);
  - `colblocks!`/`block_solve!` (D3b);
  - `tp_weights!` (wc, wf, ereg);
  - `solve_tau` (LM, line search, caps, stop rules).
- `HSEMHS/src/Slab.jl`: distributed FFT, halos, global reductions.
- `HSEMHS/src/Embedded.jl`: `solve_embedded!` warm start (D4a/D4b).
- `HSEMHS/scripts/viewing/`:
  - `measure_steps.jl`, `measure_warm.jl` (per-step snapshots);
  - `precond_lab.jl` (CG convergence per preconditioner at a frozen state);
  - `step_lab.jl` (step size, pred and |δu| per damping);
  - `stop_rules.py`.
- Tests: `HSEMHS/test/test_block.jl` (exact blocks, block solve, solves), `test_warm.jl`.
- Run outputs (cluster): `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/solver/{d2,d3,d3c,d3h,d4}_*/` (`steps.txt`,
  `run.log`), `out/solver/lab/`.
