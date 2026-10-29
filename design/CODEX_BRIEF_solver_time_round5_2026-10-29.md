# CODEX brief, round 5: μ-independent solve time for `mhs-poisson-ltau` (2026-10-29)

Branch `solver-time`. This round carries out your round-4 review
(`design/CODEX_REVIEW_solver_time_round4_2026-10-29.md`), priorities 1–4, and gives a step-5 proposal for the user's
decision. The measurements are in `design/tau/SOLVER_TIME_DESIGN.md` §6.11–6.12.

The goal and constraints are unchanged:
- Converging should take about as long at small μ as at μ = 1.
- The defaults `gn_tol` 1e-3 and 25 steps stay.
- No height-based stop and no physical prior.
- "Converged" is not redefined.
- main and embedding-api are untouched.

## 1. Changes

| commit | content |
|---|---|
| 793c0be | `feas_scale` (finding 3) and `pcg!` (finding 4); see the details below. Tests: block 92/92, warm 47/47, view 87/87, MPI view on 2/3/4 ranks. |
| b443be6 | `make_mhs_control.jl` (the known-equilibrium control), `measure_steps boundary=truth:FILE`, `phi_v3`, `control_refine.py`, `measure_analyse.py` control classes; details below. |
| 3e594dd, dfc9d11 | design §6.7–6.12 corrections and results, `control_summary.py`, `error_maps.py`. |

**793c0be, `feas_scale` (finding 3).** Each moving cell is bounded against the edge it moves toward:
- before the safety bound (two probe widths): the fraction to that bound;
- in the strip between one and two probe widths: the fraction to the evaluator bound;
- at or beyond the evaluator bound: 0;
- non-finite entries: 0;
- inward from outside: limited by the far edge only.

A failed step whose feasible fraction was 0 now stops with `:domain_limited` (embedding status `stalled`).

**793c0be, `pcg!` (finding 4).**
- Each update is formed out of place and kept only if x and r stay finite.
- An abnormal exit returns the true residual of the last good iterate.
- A `flag` keyword reports the outcome: converged, maxit, zero_rhs, start_within_tol, nonfinite_start, breakdown_rz,
  breakdown_pAp, nonfinite_step or nonfinite_update.

**b443be6, the control.**
- `make_mhs_control.jl`:
  - T is inverted by a bracket inside the valid EOS domain. The round-4 failure came from bracketing at the table's T
    extremes, which lie in invalid table corners at low P; it was not a physical failure.
  - Inversion failures are classified.
  - The audit covers every interior fine level over one period: inversion, ln P margins, T range, Cartesian force
    balance and ∇·B. It is stored as truth attributes.
  - Any azimuth is allowed, D(z) can be tapered with the D′ density term, and a thread race is fixed.
- `measure_steps boundary=truth:FILE`: experimental, the truth's fixed levels.
- `phi_v3`: the vertical residual on the first 3 faces below the cut.
- `control_refine.py`, and control classes in `measure_analyse.py`.

## 2. Findings 1–2 (Bifrost)

**Finding 1.** The μ1 solve error (22.6 km) and the solver-map discrepancy (23.6 km) are unrelated fields
(`error_maps.py`, gauge-aligned):
- full-resolution correlation +0.015; rms(e_s − e_m) is 31.7 km;
- at scales > 1 Mm, e_s is 14.6 km and e_m 1.5 km;
- the correlation of e_s with the cut ln P mismatch is −0.03;
- the same holds for the block and truth starts.

So the μ1 error is explained by neither the map discrepancy nor the cut. At μ0.3 vertical, the large-scale error in
strong-field columns is anti-correlated with e_m (−0.49). The statements "floor", "cause" and "largely explained" are
withdrawn in §6.11.

**Finding 2.** The Bifrost runs with the truth boundary (20 steps, common weights; the truth's fixed levels, from the
exact truth and from column HSE) are in §3 below.

## 3. Bifrost with the truth boundary (finding 2; 20 steps, common weights, CG cap 400)

The truth boundary is experimental: the truth's ln P and heights on the fixed levels, so the cut boundary is matched.
"Column start" is column HSE shifted per column to the truth cut pressure. z(τ=1) error is rms with the mean removed.

| run | z(τ=1): step 0 → best (step) → 20 | strong−quiet (truth) | Φ/Φ₀ at 20 | Φ_v3/Φ_v at step 0 |
|---|---|---|---|---|
| μ1, exact truth | 22.6 → 11.4 (3) → 22.4 km | −73.0 (−76.6) | 7.3e-3 | 0.87 |
| μ1, column start | 184 → 22.4 km | −73.1 | 5.4e-4 | 0.12 |
| μ0.3 v, exact truth | 54.3 → 24.2 (5) → 31.8 km, still rising | −47.3 (−36.4) | 3.7e-6 | 0.65 |
| μ0.3 v, column start | 767 → 99 km (start corrupted: up to 3.2 in ln P shift; not informative) | −89 | 7.4e-10 | – |

**μ1 ends in the same place with any start or boundary.** With the column boundary, the column, block and truth starts
gave 22.6, 22.6 and 22.4 km; with the truth boundary, the truth and column starts give 22.4 and 22.4 km. Matching the
boundary therefore changes nothing at μ1. From the exact truth the heights first improve and then move away to that
same state. The 22.4 km is therefore the distance between the snapshot and the solver's nearby discrete equilibrium,
and it is uncorrelated with the map discrepancy (§2).

**μ0.3 drifts even with the boundary matched,** while the consistent control does not (§4).

**Defects at the exact truth** (step 0, truth boundary):
- 87 % (μ1) and 65 % (μ0.3) of the vertical residual sits on the first 3 faces below the cut. Those faces are at log τ
  −4 … −3.7, where the snapshot is most dynamic.
- The step-0 heights rebuilt by the solver's map from the truth pressures are 22.6 km (μ1) and 54.3 km (μ0.3) off.

**Conditional reading.** On Bifrost the snapshot is not a discrete equilibrium of the solver's model. The non-static
forces of the snapshot are the leading untested candidate, then the EOS offset and the discretisation. The cut boundary
is not the explanation. The drift at μ0.3 is consistent with a flat objective around an equilibrium that is far from
the snapshot, not with a solver defect. None of this is proven until the snapshot's own force residual is measured.

## 4. Known-equilibrium control (§6.12)

**Case.**
- Exact 2-D Grad–Shafranov atmosphere: P = P₀(1 − 0.3 cos² kx), |B| up to 1.2 kG, period 3 Mm, box 12 Mm.
- y-invariant, azimuth 0.
- Mapped onto rays as `make_slanted_tau.jl` does.

**Construction.**
- Audit: force balance ≤ 5e-4 ρg (rms 2e-5), ∇·B 3e-5, ln P margins ≥ 12.
- Generator refinement (dz 250 → 62.5 m): second order, ≤ 5 m in z(τ=1).
- Field-free case: exact at μ1 and μ0.3.

**Refinement.** Tracks:
- **bt**: the truth boundary, started from the exact truth;
- **col**: the column-HSE boundary;
- **prod**: the production defaults.

μ0.3 uses cut −3.6, which lies on every grid.

| | μ1 bt z1 / ΔlnP / defect | μ0.3 bt z1 / ΔlnP / defect |
|---|---|---|
| (128, 0.2) | 2.68 km / 0.019 / 2.0e-7 | 0.79 km / 0.022 / 4.8e-7 |
| (256, 0.1) | 0.72 / 0.003 / 7.1e-8 | 0.14 / 0.006 / 3.2e-7 |
| (512, 0.05) | 0.13 / 0.0004 / 3.1e-8 | 0.03 / 0.001 / 1.3e-7 |

**What converges.**
- Solution errors converge at second order. μ1 is limited by Δlog τ alone; at μ0.3 the horizontal and τ terms partly
  cancel, so single-direction refinements are not monotone.
- The truth's discrete defect converges more slowly, about first order. It does not fall with nx at μ0.3, in either
  balance. One-sided column derivatives at the ends are suspected but unchecked.
- From the exact truth or from column HSE, the solve reaches the same state.

**The column-HSE boundary.**
- It adds a uniform height offset (−51 km at μ1, −120 km at μ0.3) and ΔlnP 0.05–0.06 near the cut. Neither changes
  under refinement.
- The interior errors match bt.
- At μ0.3 it also creates a slow Φ tail with unchanged heights; half of the vertical residual sits on the first 3
  faces.

**No height drift.** Heights reach their final error by step 2–5 and stay there for up to 80 steps, with both
balances.

**Time to converge with the production defaults.**

| track | μ1 | μ0.3 |
|---|---|---|
| column boundary | 5 steps, qualified, on all 7 grids (15–27 s) | 6–11 steps, qualified, on 5 grids; 25 steps (max_steps) unqualified on (256, 0.1) and (256, 0.05), although the heights were final by step 3 |
| truth boundary, those two grids | 4 steps (16–17 s) | 8 steps (25–34 s) |

- Every μ0.3 step hits `cg_maxit` 80 (relative residual 0.01–0.04), against 5–9 iterations at μ1.
- Stop-rule edge case: an exact start (field-free) reaches Φ at round-off and ends `no_descent`, which the embedding
  reports as `stalled`.

## 5. Reading

1. **On a known equilibrium consistent with the solver's model, the objective is not the obstacle at μ0.3.** The
   solver converges to the truth at second order without drift, in about 2× the Gauss–Newton steps of μ1. The
   remaining cost difference is linear: about 10× the CG iterations per step.
2. **The production cut boundary is the largest systematic error on the control.** On two grids it is also the reason
   the stop did not qualify at μ0.3.
3. **On Bifrost, neither the map discrepancy nor the cut boundary explains the error.**
   - All μ1 runs, from any start and with either boundary, end at 22.4–22.6 km.
   - The drift at μ0.3 persists with the boundary matched.
   - The snapshot is not a discrete equilibrium of the model. Measuring its own Cartesian force residual (non-static
     forces) is the next diagnostic.

## 6. Step-5 proposal (for the user's decision; nothing implemented)

**(a) No change to Φ now.**
- The control shows Φ's minimiser converging to the truth at second order, with heights fixed early.
- A derived physical metric is not motivated by this evidence.

**(b) Linear cost at small μ: a crop multilevel prototype.**
- On the control and on Bifrost, μ0.3 steps need about 10× the CG iterations of μ1 and hit the cap.
- This is where the μ-dependence of time now sits.
- Pass criterion: on the control grids and on Bifrost crops, CG iterations per Gauss–Newton step at μ0.3 within
  about 2× of μ1, at both 1e-3 and 1e-5, with unchanged endpoints.

**(c) The cut boundary (a scientific decision).**
- The column-HSE cut pressure biases heights by −51/−120 km on the control and creates the slow tail that blocks
  qualification.
- Options:
  - free the cut pressure, as one more unknown per column with the upper structure recomputed;
  - place the cut deeper;
  - leave it, documented.
- Each changes the problem, so this is not proposed as a default.

**(d) Stop-rule edge case.** An absolute floor (`rms_atol` exists, default 0) would let an exact start report
`converged`. A default change is the user's decision.

The bound-aware step is not proposed: no control or Bifrost run since the round-3 fixes has been domain-limited.

## 7. Questions

1. Is the refinement evidence in §4 sufficient to attribute the absence of drift to model consistency? Or do you want
   the rotated case `A(k_x x + k_y y, z)` with nonzero t_y and perturbation-return tests first?
2. The truth's defect converges at about first order while the solutions converge at second. Is the one-sided end
   derivative the likely cause, and does it matter?
3. Of (b)–(d), which first, and with what pass criterion?
4. For the Bifrost 22.4 km at μ1: is measuring the snapshot's own Cartesian force residual (`force_diag.jl` on the
   truth) and its share of Φ the right first ablation? The alternatives are separating the EOS offset (re-deriving ρ
   from the EOS) and the horizontal discretisation (a crop at half spacing).

## 8. Where to look

- `design/tau/SOLVER_TIME_DESIGN.md` §6.11–6.12.
- Code:
  - `HSEMHS/src/PoissonZ.jl` `pcg!`;
  - `HSEMHS/src/PoissonTau.jl` `feas_scale`, `phi_parts`, `solve_tau`;
  - `HSEMHS/scripts/viewing/`: `make_mhs_control.jl`, `measure_steps.jl` (`boundary=`), `control_summary.py`,
    `control_refine.py`, `error_maps.py`.
- Logs on the cluster, `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/`:
  - `control/` (`gs_*`, `ff_*`, `sw_*`, `dr_*`);
  - `solver/bt_*`.
