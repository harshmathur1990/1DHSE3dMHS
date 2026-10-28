# CODEX brief, round 3: μ-independent solve time for `mhs-poisson-ltau` (2026-10-29)

Branch `solver-time`. This round covers your round-2 review (`design/CODEX_REVIEW_solver_time_round2_2026-10-28.md`),
priorities 1–4. The measurements are in `design/tau/SOLVER_TIME_DESIGN.md` §6.5–6.10.

The goal is unchanged: the time to converge should not depend on μ, and should be comparable to μ = 1. The defaults
`gn_tol` 1e-3 and 25 steps stay; FFNOML may override them. There is no height-based production stop and no physical
prior.

## 1. What changed

| commit | content |
|---|---|
| b2229c5 | Stop qualification by the interval L = 2bᵀx − xᵀAx ≤ bᵀA⁻¹b ≤ L + rᵀD⁻¹r. PCG refines from the current x (`pcg!` keyword `x0`) until gn_tol is decided or the budget is spent. Outcomes are qualified / not_stationary / undecided / invalid. An `:unresolved` stop reason maps to embedding status `stalled`, with `stats["stop_check"]` recorded. Also: `check_weights`, `lm = :fixed`, per-step and total work counters, and tests against the dense crop solve (block tests 57/57). |
| 1674b8f | `measure_steps`: `start=DIR` (start from another run's final state) and `common_weights` (the column-HSE start's objective, whatever the start). |
| 92d029e | Line search: `evaluate!` returns the number of cells outside the EOS table; trials outside the table backtrack up to 12 halvings; failed tries are logged. |
| e735b81 | `frozen_lab.jl parts=spectral`: dense generalised problem on a crop; Ritz checks; weights of b; deflation with c-only, column-block-extended and exactly extended cut-height modes, and with exact eigenvectors. |

The qualification interval behaved as intended on the full cube: it refused a false convergence (pred 1e-9 while
L/Φ ≥ 0.985). U is very pessimistic (up to 1e9 × gn_tol) because ereg is small, so in practice the outcome is decided
by L, or it is undecided.

## 2. Results

### 2.1 Damped block end-to-end (priority 2; §6.7)

- Damped block runs are 2–8× cheaper per 40 steps than D2 and reach the same Φ at μ1.
- Their heights are 10–18 km worse, and strong−quiet is far from the truth (−13 to −26 km against −76.6 km).
- Φ is nearly flat along large-scale height modes. Block steps do not move those modes; Fourier steps do.

### 2.2 Common-objective references (priority 3; §6.8)

**μ1.**
- From the block endpoint, Fourier steps bring the heights back to the column-start result (22.6 km, strong−quiet
  −73.6 against −75.5).
- The block start even reaches a slightly lower Φ, so the column run is not converged at 80 steps either.

**μ0.3, ray and vertical.**
- Nothing converged in 80 steps (5–4.5 h, 28–30 k CG).
- **Φ keeps decreasing while the heights get worse.**
  - The z(τ=1) error grows from 21–24 km at 40 D2 steps to 26–30 km.
  - The z(τ=1) p1/p99 spread grows from −22/+54 km to −47/+80 km.
  - In the vertical balance, strong−quiet moves away from the truth (−26.1 against −36.4).
- The two starts reach Φ within 1.3 % of each other, with heights 1.6 km apart.

**The μ0.3 vertical block-endpoint `no_descent`.**
- Every trial point of step 2 left the EOS table.
- The default damping (a multiple of ereg) barely shortens the step.
- With continued backtracking, the run creeps at α ≈ 1e-3 with ρ ≈ 1. That is a feasibility barrier, not a bad model.

### 2.3 Spectrum (priority 4; §6.9, 16 × 16 crops, dense)

**Separated modes.**
- The four checkerboard cut-height modes sit far below the rest of the spectrum.
- b has essentially no weight in them (≤ 1e-8).

**The dense band.**
- Above them is a dense band: λ < 0.1 λmax holds 182 eigenvalues at μ1, 675 at μ0.3 ray and 2031 at μ0.3 vertical,
  out of about 13 k.
- At μ0.3 vertical, 10 % of b lies in that band.

**Ritz extraction.**
- At μ1 it is accurate.
- At μ0.3 no Ritz pair converges in 300 iterations, even with full reorthogonalisation. The loss of orthogonality is
  real (max |VᵀMV − I| ≈ 0.5) but secondary.
- The earlier full-cube Ritz analysis and deflation therefore used unconverged vectors.

**Deflation.**
- Cut-height |k| ≤ 2 modes halve the iterations at μ1 and do nothing at μ0.3 vertical.
- Energy extension (column-block or exact) lowers the coarse vectors' Rayleigh quotients 5–10× but changes the
  iterations by only a few.
- Even the exact 64 smallest eigenvectors give at most a factor of 3.

### 2.4 Block CG against crop size (§6.10)

- Undamped, the μ0.3 iterations to 1e-3 grow with the crop: 54 → 107 → 89 → 208 → 200 (vertical) and
  146 → 67 → 86 → 448 → > 600 (ray) for 8² … 128².
- With Marquardt damping 0.1 they stay at 22–50 at every size and μ.
- Caveat: each crop has its own 10-step state, and the larger crops are further from converged.

## 3. Interpretation

1. At small μ the Gauss–Newton problem itself is the obstacle, not only the linear algebra.
   - The objective has a wide, nearly flat valley in large-scale height and pressure directions.
   - At μ0.3, going further down that valley makes the heights worse.
   - A solver that converges faster at μ0.3 will reach a worse answer faster. "Time to converge" is only meaningful
     with respect to an objective whose minimiser we want.
2. Damped column blocks give size- and μ-independent linear solves but do not move the valley.
   - Fourier steps do move it, at 400 capped iterations per step.
   - The two preconditioners are complementary. The retired hybrid combined them with inconsistent damping.
3. A fixed coarse space cannot hold a band of this size. Horizontal multigrid is the option that would scale, but it
   would make it cheaper to go further down a valley we may not want to go down.

## 4. Questions

1. **Is the μ0.3 behaviour a property of the objective or of our weights?** The weights are wc, wf and ereg from the
   column-HSE start, fixed for the solve.
   - Could the balance of Rx/Ry against Rz, or the size of ereg relative to JᵀWJ, be making the large-scale height
     directions nearly flat at small μ?
   - Is there a principled rescaling, not a prior, that would make the minimiser well determined? An example is
     weights from the inclined geometry.
   - How would you test that without a height-based stop?
2. **Given §2.2, what should "converged" mean at μ0.3?** The user's requirement is time comparable to μ = 1.
   - Is it legitimate to stop on a criterion that is μ-independent in the right sense, such as a relative decrease of
     Φ per unit work, or the decrement measured in a damped metric?
   - Or must we first fix the objective (question 1)?
3. **Preconditioner.**
   - Would you now recommend a consistent two-level additive preconditioner: damped column blocks plus the Fourier
     layer-median solve on the residual, with the same damping in both parts?
   - Or go straight to horizontal multigrid with column-block smoothing?
   - What crop-size scaling test would you accept as evidence, given that the crop states differ?
4. **EOS feasibility.** The default damping (multiples of ereg) does not shorten steps that leave the table.
   - Should the step instead be limited by a fraction-to-boundary rule on ln P per cell?
   - Or should Marquardt damping be the default when a trial leaves the table?
5. **Next experiments.** Are the reference runs enough to proceed, given they did not converge at μ0.3?
   - Run longer (160 steps)?
   - Or first run a perturbation test: start from the truth-derived state and see whether the solver walks away from it?

## 5. Where to look

- Measurements: `design/tau/SOLVER_TIME_DESIGN.md` §6.5–6.10.
- Code:
  - `HSEMHS/src/PoissonTau.jl`: `qualify_stop`, `solve_tau` line search and work counters.
  - `HSEMHS/scripts/viewing/frozen_lab.jl`: `spectral_lab`, `coarse_lab`.
  - `HSEMHS/scripts/viewing/measure_steps.jl`.
- Logs on the cluster worktree `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/solver/`:
  - `ref_*`, `diag_mu0.3v_blk`;
  - `lab/spec16_*.log`, `lab/scale*_*.log`.
