# CODEX brief, round 4: μ-independent solve time for `mhs-poisson-ltau` (2026-10-29)

Branch `solver-time`. This round covers your round-3 review (`design/CODEX_REVIEW_solver_time_round3_2026-10-29.md`).

- **Done:** your priority order through step 3b.
- **Started, not finished:** step 3c (the known-equilibrium control). The user asked for this brief now; §3 records
  where step 3c stands.
- **Not done:** steps 4 (the interpretation corrections, listed in §4 for you to check) and 5.

The goal and constraints are unchanged:
- Converging should take about as long at small μ as at μ = 1.
- The defaults `gn_tol` 1e-3 and 25 steps stay; FFNOML may override them.
- No height-based production stop and no physical prior.
- "Converged" is not redefined to meet a wall-time target.
- main and embedding-api are untouched.

## 1. What changed (commit 9f9fd44)

**Step 1: PCG restarts and work accounting.**
- `pcg!` with `x0 = true` now computes the explicit residual first:
  - an exact start, or one already within tol, returns `k = 0`;
  - b = 0 returns x = 0;
  - a breakdown stops at the last good iterate.
- `pcg!` counts its A and M⁻¹ applications (`cnt`).
- `solve_tau` reports `info.work` as:
  - CG iterations, linear solves, and A and M⁻¹ applications;
  - evaluations and restorations;
  - qualification work, with restarts and the explicit interval residuals included;
  - final-diagnostic work, which is zero when the stop check is reused;
  - seconds per part.
- Nothing is double-counted.
- `measure_steps` checks restart metadata on every rank: ranks, slab, grid, cut, view, balance and input size/checksum.
  It also:
  - evaluates the start state for s0;
  - saves a Float64 checkpoint (`final64`) and the objective (`weights/`).

**Step 2: EOS feasibility.**
- `feas_scale` scales every trial step by the fraction to the boundary on ln P against the table range, less two probe
  widths (`EPS_PROBE` = 1e-4, the opacity-derivative probe).
- `evaluate!` returns the global count of bad cells.
- `eos_cells` separates real out-of-table states from probe-only failures (cells where only ln P ± EPS_PROBE leaves
  the table). It also locates the worst cell.
- A failed try logs μ, CG, pred, cap and feasibility scales, and either Φ_trial/Φ or the bad-cell breakdown.
- Tests pass: block 71/71, warm 47/47, view 87/87.

## 2. Results of step 3 (truth consistency and truth starts)

### 2.1 How far the synthetic truth is from the solver's model (`scripts/viewing/truth_consistency.jl`)

Cut at log τ −4 for μ1 and −3.5 for μ0.3.

| | μ1 | μ0.3 |
|---|---|---|
| ln ρ_EOS(T, P_truth) − ln ρ_truth | 0.011 everywhere | 0.011 |
| Solver Δz per level against truth Δz, log τ > 0 (rel. rms) | 17 % | 38 % (strong field, log τ [−1, 0): 49 %) |
| z(τ_LOS = 1) rebuilt by the solver map from the truth cut height − truth | 23.6 km rms (strong 12.9, quiet 24.0), mean −6.7 | 55 km rms (strong 70), mean −8 |
| Strong − quiet shift produced by the map alone | +0.6 km | −15.1 km |
| Cut ln P, column HSE − truth | rms 0.124 | rms 0.166 |
| Column-HSE ln P − truth, below the cut | 0.05–0.1 | 0.09–0.12 |

**Cause.** `make_slanted_tau.jl` takes χ₅₀₀ from Bifrost T and P on the geometric grid, shears it, integrates τ along
the ray, and then interpolates T, P and z onto the τ axis. The solver instead takes χ from the interpolated T and the
solved P, and uses the trapezoid dz = −μ τ/χ d ln τ. Since interp(χ(T, P)) ≠ χ(interp T, interp P), the truth's heights
are not representable on the τ grid.

**Consequences.**
- At μ1 the 23.6 km matches the 22.6 km floor that every μ1 solve reaches.
- At μ0.3 the mismatch is more than twice as large, and it biases strong − quiet by −15 km.

### 2.2 Solves started from the truth (20 steps, common weights, Φ in units of the column-HSE start's Φ₀)

Column-HSE Φ₀ for each case: μ1 1.38e11; μ0.3 vertical balance (v) 9.91e13; μ0.3 ray balance (r) 1.33e12.

| Start | Φ/Φ₀ start → 20 steps | z(τ=1) error, start → 20 steps | strong − quiet (truth) |
|---|---|---|---|
| μ1, truth shifted per column to the column-HSE cut pressure | 22 → 0.024 (column reference 0.026) | 248 → 22.4 km | −75.2 (−76.6) |
| μ0.3 v, truth unshifted (`truth_raw=1`) | 131 → 5.3e-5 (column reference at 80 steps 4.0e-5) | 52 → 13.6 at step 5 → 32.8 at step 10 → 35.6 km | −34.2 (−36.4) |
| μ0.3 v, truth shifted | 3890 → 1.8e-4 | 213 → 104 km | −87 |
| μ0.3 r, truth shifted | 622 → 0.011 (column reference 4.6e-3) | 213 → 65 km | −69 |
| μ0.3 v, shifted + smooth |k| = 1 perturbation | — | tracks the unperturbed shifted run (102 km) | — |

**Reading.**
1. **μ1: well determined.** Started from the truth, the solve ends at the same heights as from column HSE, at slightly
   lower Φ.
2. **μ0.3: the truth is far from balance in the solver's objective.**
   - Most of the imbalance is in the vertical (along-ray) residual.
   - The unshifted truth start passes the best heights seen so far (13.6 km at step 5). It then drifts to 35.6 km while
     Φ falls 60×.
   - It ends at about the column reference's Φ, but with heights about 10 km apart.
   - So Φ is nearly flat in large-scale height directions at μ0.3, and part of its minimiser is set by the generator
     mismatch of §2.1.
3. **The "admissible" construction was a poor choice.** Shifting each truth column to the column-HSE cut pressure moves
   every level by the cut mismatch (rms 0.12–0.17 in ln P), so those starts begin 200+ km off. The unshifted variant is
   the meaningful one, even though it has a pressure jump at the cut.

## 3. Step 3c, known-equilibrium control: state when stopped

### Script

`HSEMHS/scripts/viewing/make_mhs_control.jl` is uncommitted in this round and is attached to this brief's commit as a
draft. It builds an exact 2-D magnetostatic atmosphere and maps it onto rays the way `make_slanted_tau.jl` does, but
from analytic fields on a fine height grid (dz 250 m).

**Reference.**
- T₀(τ) is grey Eddington–Hopf (T_eff 5772 K).
- A column HSE in τ gives T₀(z).
- P₀(z) is then integrated exactly in z by RK4 with dP₀/dz = −ρ(T₀, P₀) g.

**Field.** Grad–Shafranov with B = ∇ × (A ŷ):
- A = A₀ cos(kx) f(z) and p(A, z) = P₀ − m(z) A²/2, so ∇²A = μ₀ m A.
- m is chosen so that P = P₀(1 − D cos² kx).
- This gives:
  - f'' = k² f + k² D P₀/(P_m f);
  - ρ = ρ₀(1 − D cos²) − 2 D P₀ f'/(f g) cos²;
  - T = EOS⁻¹(ρ, P).
- D = 0 gives the field-free hydrostatic control.

**Rays.**
- x = ξ + t_x z, with χ = χ(T, P) at the fine points.
- τ comes from `Tau.tau_column!`.
- Values are interpolated in log τ onto lt_min:dlt:lt_max.

**Refinement plan.**
- Box 12 Mm, period 3 Mm, NY = 8 (the atmosphere is y-invariant).
- Grids (NX, Δlog τ) = (128, 0.2), (256, 0.1), (512, 0.05).
- μ1, plus μ0.3 with both balances.
- 80-step trajectories with the reference options, and a gn_tol 1e-3 stop run for the converged time.

**Status.**
- Construction check: centred differences on a fine (x, z) grid give max |R_x|/ρg 7e-5 and |R_z|/ρg 5e-4, as expected
  at O(h²). At z = 0 the maximum |B| is 1.2 kG and P₀ is 1.4e4 Pa.
- The first ray mapping failed:
  - **Upward-decaying start f' = −k f.** f grows too slowly, so the plasma β rises with depth and f'/f becomes large.
    Then the density term 2 D P₀ |f'/f|/g ≫ D ρ₀, so T = EOS⁻¹(ρ, P) falls below the table minimum (2170 K) in the
    middle atmosphere.
  - **Self-similar start f ∝ √P₀ (where ρ = ρ₀).** This removed most of the failures. 333 fine points near z ≈ 2 Mm
    (ln P ≈ −9.4) still fail, at the scale-height change where 2kH varies.
- Fixes not yet tried:
  - cap the top of the fine grid lower, e.g. τ_vert ≈ 1e-9; the axis needs τ_LOS ≤ 1e-8 at μ0.3;
  - make D taper upward (D' ≠ 0 is allowed in the formula);
  - use a longer period, which lowers kH.
- No solver run on the control exists yet.

## 4. Corrections to the round-3 brief and design §6, for your check (step 4)

1. **ereg is not part of Φ.**
   - Φ = Σ wc (Rx² + Ry²) + wf Rz².
   - ereg enters only the step operator A = JᵀWJ + D(ereg).
   - Round 3, question 1, wrongly suggested that ereg's size could flatten the objective. It can only flatten the
     linear model.
2. **Coarse-space gains at both tolerances** (16² crop, μ0.3 vertical, PCG iterations):

   | Coarse space | tol 1e-3 | tol 1e-5 |
   |---|---|---|
   | block alone | 107 | 385 |
   | c, \|k\| ≤ 2 | 96 | 174 |
   | c, \|k\| ≤ 2, column-extended | 95 | 171 |
   | smallest 16 eigenvectors | 45 | 83 |
   | smallest 64 eigenvectors | 33 | 51 |

   - The round-3 statement "at most a factor of 3" holds only at 1e-3. At 1e-5 the cut-height modes alone give 2.2×,
     and exact eigenvectors give 7.5×.
   - The λ < 0.1 λmax counts of the "dense band" depend on that threshold. They are not a property of the operator.
3. **"Block steps do not move the valley" is an observation of 40-step runs, not an algebraic property.** Block-Jacobi
   with an exact solve would move every mode; the damped, capped solves did not.
4. **The hybrid ablations already exist (D3c).** Round 3 proposed "block + Fourier with consistent damping" as if it
   were new.
   - Applying a Fourier correction after the block correction on the residual is a multiplicative combination.
   - It is not symmetric, so PCG would need the symmetrised form (pre- and post-smoothing).
   - An additive version would need a consistent shared damping.
5. **The large μ0.3 height error of the reference runs is not, as stated, mostly a solver failure.** §2.1 shows it is
   largely the generator mismatch plus a flat objective.

## 5. Questions

1. **The generator.** Should the synthetic inputs be rebuilt to be consistent with the solver model before more
   solver-time work? Options:
   - (a) z of the truth rebuilt with the solver's map, which changes only the error metric;
   - (b) χ computed after interpolation, with τ re-integrated on the τ axis;
   - (c) keep the inconsistent inputs, because real inputs (FFNOML's) are not model-consistent either.

   Which input should define "time to converge at μ0.3"?
2. **The known-equilibrium control.**
   - Is a y-invariant Grad–Shafranov arcade enough, or do you want a 3-D one? A 3-D exact MHS is harder; one option is
     an axisymmetric self-similar flux tube.
   - What refinement evidence do you need? Our plan is the height error and ΔlnP falling by about 4× per refinement at
     μ1 and μ0.3. Is that enough to attribute the μ0.3 drift?
   - How should the remaining EOS-range failure in §3 be fixed without making the case trivial?
3. **Flat valley at μ0.3.** If the control also shows Φ decreasing while the heights get worse at μ0.3, the objective
   itself is the problem.
   - Is a derived physical residual metric (Φ in force units per volume along the ray rather than per τ level) the
     right next proposal?
   - Or should it be a bound-aware step, or a crop multilevel prototype (your step 5)?
   - Which would you test first, and with what pass criterion?
4. **Stop rule.** The truth-start μ0.3 run passes 13.6 km at step 5 and then drifts. Is there any principled
   stationarity measure, without heights or priors, that would distinguish "early on a flat valley" from "converged"?
   If not, should the default 25-step budget at μ0.3 be reported as is?

## 6. Where to look

- **Code:**
  - `HSEMHS/src/PoissonZ.jl`: `pcg!`.
  - `HSEMHS/src/PoissonTau.jl`: `feas_scale`, `eos_cells`, `qualify_stop`, `solve_tau` work counters.
  - `HSEMHS/scripts/viewing/`: `truth_consistency.jl`, `measure_steps.jl` (start=truth, perturbations),
    `make_mhs_control.jl` (draft).
- **Design:** `design/tau/SOLVER_TIME_DESIGN.md` §6.5–6.10. It has not yet been updated with §2–4 of this brief.
- **Logs** on the cluster, `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/solver/`:
  - `truth_mu1`, `truth_mu0.3v`, `truth_mu0.3r`, `truthraw_mu0.3v`, `truthpert_mu0.3v`;
  - `ref_*`;
  - `lab/spec16_*.log`.
