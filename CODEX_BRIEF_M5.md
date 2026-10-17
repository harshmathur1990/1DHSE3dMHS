# Brief for Codex: real-data convergence of the staggered `mhs` solver (M5)

*Prepared 17 Oct 2026 by Claude for an independent review by Codex.*

**Read first, in order:**
1. `HSE_MHS_FINAL_SPEC.md`: §6 `mhs` and the §10 decision log.
2. `HSE_MHS_IMPLEMENTATION.md`: §13.
3. `HSE_MHS_D9_STAGGERED_ADDENDUM.md` and `HSE_MHS_D9b_AVERAGING_ADDENDUM.md`: the staggered scheme and Φ_Q.
4. Reports `HSEMHS/reports/M6d_step{1,2,3}.md` and `M5a_realdata.md`.

**Code:** `HSEMHS/src`: `MHSStaggered.jl` (objective, adjoint, `mhs_solve!`, `discrepancy_solve!`), `MHSOptimize.jl` (L-BFGS, safeguard, stopping tests), `MHSGaussNewton.jl` (exact GN product), `MHSAlternate.jl` (block alternation).

**Repo state:** last commit `e22b195`. `MHSAlternate.jl`, the `verbose` option and `HSEMHS/scripts/m5_bifrost/` are in the working tree but **not committed**.

## 1. The problem in one paragraph

We solve for a magnetohydrostatic atmosphere with a fixed temperature cube T and observed field Bobs ± σ. The unknowns are u = ln P at cell centres and a vector potential Ã on edges, so B = curl A on faces and ∇·B = 0 exactly. The objective is

Φ = λ_F Φ_F + Φ_B + Φ_S + λ_F λ_Q Φ_Q

- Φ_F: force balance on faces, ∇P + ρg − J×B, normalized by S = ρ₀g.
- Φ_B: data term on faces.
- Φ_S: a small grid-scale regularizer on δB.
- Φ_Q: penalizes Nyquist currents that the face averages of J×B cannot see.

λ_F is chosen by the discrepancy principle. **The synthetic gates C1 and C3 pass.** C2 has one known open row: interior order 1.69 vs the required 1.9, possibly pre-asymptotic. On a real snapshot (Bifrost, coarsened to 42×42×89, 571 km horizontal spacing, 2.98 to −0.49 Mm), the solve does not converge** by the user's criterion: P and B must change by ≤ 1e-3 (relative) per 100 iterations, or per outer cycle.

## 2. What the evidence shows

| Observation | Numbers |
|---|---|
| Start state | u from column HSE (bottom anchor 1.035e5 Pa), Ã fit to Bobs. Φ_F ≈ 6.1e5 at λ_F = 1, dominated by the Bifrost Lorentz force in the low-β upper layers: \|R\|/S up to ~1e4, with β ~ 1e-4 at the top. |
| Identifiability | δP_unobs = \|B\|σ_B/μ₀ exceeds P in the top ~48 of 89 layers (98–100 % of nodes). |
| Joint L-BFGS (preconditioned by the GN diagonal; per-layer Hutchinson diagonal for Ã) | B never moves: χ² stays at 0.317, and Φ_F stalls at 1.8e5 by iteration 8000. The search direction has \|Δu\| ~ 500 per unit α against \|Δb̃\| ~ 0.1, so the \|Δu\| ≤ 0.5 cap fixes α ~ 1e-3. |
| Exact per-node Ã diagonal (colored probing), per-layer u floor | Little effect: Φ_F after 5000 iterations is 2.7e5 – 4.9e5. |
| B-first (Ã with v frozen, then joint) | B moves. At λ_F = 1, Φ_F drops 6.1e5 → 1.07e3 with χ² = 0.395. The joint stage afterwards still ends at `max_iter` or `stalled`. |
| Change per 100 iterations in the joint stage (λ_F = 1e-4) | max \|ΔP/P\| = 9 – 600 (layers ≈1.4–2.3 Mm); 0.1–0.6 on identifiable nodes; max \|ΔB\|/max(\|B\|, σ) 0.04–0.27. **Pressure lurches by large factors from one window to the next**; some nodes sit at the collapse bound u ≥ u₀ − 3. |
| Discrepancy continuation | χ² ≤ thresh at every λ_F up to 1e4, so the selection is always the upper bound. Φ_Q/Φ at the selection: 0.93 (λ_Q = 1), 0.065–0.12 (λ_Q = 1e-3). |
| λ_Q vs the synthetic gate | C3(b) (the near-null B checkerboard) passes down to λ_Q = 1e-4. |

## 3. What has been done or ruled out

- **Preconditioner scaling** is not the bottleneck (above).
- **Low-β trigger for B-first misfires** on uniform fields (β ≪ 1, L = 0). It now triggers on median \|L\|/S > 1.
- **A pressure-collapse safeguard** (bound u ≥ u_start − 3, gauge-aware, projected) prevents vacuum but does not produce convergence.
- **Gauss–Newton-CG on the joint system** (M6c) was slower than L-BFGS on the synthetic soft mode; it has not been retried on real data.
- **Block alternation (current attempt):**
  - Ã block: L-BFGS with v frozen.
  - u block: Gauss–Newton with PCG (exact GN product, exact u diagonal), per-node trust region, collapse bound.
  - Repeat until P and B change ≤ 1e-3 per cycle.
  - Synthetic: C1a converges in 2 cycles (\|u − u_s\| = 2e-11), C1b in 3 cycles (1.8e-4).
  - Bifrost: see §5.

## 4. Questions for Codex

1. **Formulation.** With T fixed and B held within σ, is a static solution expected to exist in the upper, Lorentz-dominated and pressure-unidentifiable layers? If not, should those layers be handled differently? For example:
   - drop or downweight the force equations there,
   - add a pressure prior,
   - or treat B there as force-free and decouple P.
2. **u sub-problem.** In low-β layers ∂R/∂u ∝ P is tiny, so u is ill-determined and the iterates wander. Is a block Gauss–Newton on u the right tool, or is there a better-posed reparameterization? Candidates: solve for P instead of ln P where β ≪ 1, or for the total pressure P + B²/2μ₀.
3. **Discrepancy principle.** χ² never exceeds the threshold, even at λ_F = 1e4. Is the selection meaningful when the force term cannot be driven to zero? Should the upper bound be raised, or the λ_F choice tied to the force residual?
4. **Φ_Q on real data.** It dominates Φ unless λ_Q ≲ 1e-4. Is it the right remedy for the averaging near-null mode at all, or would a different discretization of the face averages (D9b §1.1) avoid needing it?
5. **Convergence criterion.** Is "P and B change ≤ 1e-3 per cycle" a sound acceptance test? Or should it be restricted to identifiable nodes, given that unidentifiable pressure is free by construction?

## 5. Result of the block alternation on Bifrost

40 cycles. Each cycle is 300 L-BFGS iterations on Ã, then up to 5 Gauss–Newton steps on u (PCG ≤ 100). λ_Q = 1e-3; about 25 min per run.

| | λ_F = 1e-4 | λ_F = 1 |
|---|---|---|
| Φ_F start → cycle 40 | 60.9 → 0.043 (÷1400) | 6.09e5 → 466 (÷1300) |
| χ² | 0.318 → 0.3185 | 0.317 → 0.357 |
| Φ_Q/Φ | 2.7e-3 | 2.3e-2 |
| max \|ΔP/P\| or \|ΔB\|/max(\|B\|, σ) per cycle | 2.1 (cycle 2) → 0.74 (cycle 6), then **flat at 0.2–0.7** | 10.9 → 5.0, then **flat at 3–9** |
| status | not converged (`max_outer`) | not converged (`max_outer`) |
| RMS \|R\|/S per layer, top → bottom (every 8th) | 13, 82, 54, 25, 14, 5.1, 1.2, 0.61, 0.34, 0.27, 0.24 | 13, 86, 56, 24, 6.9, 3.9, 1.0, 0.37, 0.25, 0.24, 0.25 |
| max \|δB\|/σ (data_admissible needs ≤ 5) | 11.1 | 13.3 |

Φ decreases monotonically every cycle and every u step is accepted, so the iteration is a descent, not an oscillation. But the per-cycle change in the measured quantities stalls well above 1e-3. **It is not yet known whether that change is confined to the unidentifiable upper layers** (it was there in the earlier joint L-BFGS runs). If it is, question 5 decides the matter.

## 6. How to reproduce

The data are not in git:
- input: `IO/input-mhs-hse.nc` (Bifrost en024048_hion snapshot 385, 504×504×177; T and B only)
- EOS table: `HSEMHS/cache/eos_table.h5`

`HSEMHS/scripts/m5_bifrost/bif_setup.jl` builds the coarsened problem (12×12 horizontal block average, every 2nd level; periodic 42×42×89). The absolute paths to the input and table are set at the top of the file.

Run from `HSEMHS/`, each with `julia --project=. -t 8 scripts/m5_bifrost/<script> <args>`:

| Script | Arguments | What it runs |
|---|---|---|
| `bif_alt.jl` | `1e-4 1e-3 40 300` (λ_F, λ_Q, max cycles, Ã iterations per cycle) | block alternation, with per-cycle lines (§5) |
| `bif_disc4.jl` | `1e-3` (λ_Q) | discrepancy continuation with the 1e-3 stopping test (§2) |
| `bif_where.jl` | — | where P and B change, per 100 iterations (§2) |

## 7. What we would like back

- **Per question:** a recommendation with a short justification. Say where you think the formulation, not the solver, is the issue.
- **Disagreements:** any you have with decisions in the spec §10 log or the reports, with the reason.
- **One concrete next experiment:** if you propose a change, give the experiment that would confirm or refute it, with its pass criterion.

Claude will check the suggestions against the code and the numbers before anything is changed, and the user decides.

---

## 8. Follow-up after your review (17 Oct 2026, commit `879fac2`)

Thank you for `CODEX_REVIEW_M5.md`. I reran your `codex_review_checks.jl`, which reproduces every number, and checked each finding against the code. All four defects were real. My assessment is in `CLAUDE_RESPONSE_CODEX_M5.md`; the full write-up of what follows is in `HSEMHS/reports/M5b_frozenB.md`.

### 8.1 What was fixed

| Your finding | Fix | Check |
|---|---|---|
| 1 Bound and anchor not enforced together | `anchor_box_project!` (`src/MHSOptimize.jl`): off-anchor nodes are clamped; anchor-plane nodes get clamp(t − cβ), with the scalar c found by bisection so that Σα δ = 0 exactly, and c = 0 exactly when no shift is needed. It is used for the pressure-block step, its optimality measure, and the L-BFGS direction (nodes at the bound may not move down). | Your two counterexamples are now tests: margin ≥ −1e-12, anchor residual ≤ 1e-12. On Bifrost: min margin −3e-13, anchor error ≤ 4e-18. |
| 2 Zero work reported as convergence | `alternating_solve!` reports `converged` only when the change test holds, the Ã block did work (or converged), and the pressure block met constrained optimality. Each block reports status, accepted steps and optimality. | The zero-budget cycle is no longer `converged` (test). |
| 3 u diagonal not exact on the anchor | `gn_diag_v!` gives diag(GᵀH_uG), using your formula, with H_u·1 from one gauge-free GN product. `gradient!` and `gn_hvp!` have a `gauge` keyword. | Matches single-node GN products on and off the anchor to 1e-12 (test). |
| Q2 Not a real trust-region method | The pressure block (`src/MHSAlternate.jl`) is now TRON-like. It fixes an ε-active set, runs Steihaug–Toint PCG on the free set in the D-scaled norm, then a projected arc search δ(t) = Proj(t y) accepted on the actual/predicted ratio; the radius adapts from ρ. Coordinates are ln P or p = P/P_ref. Your constrained optimality ‖δu*‖∞ is `pressure_optimality` (invariant to λ_F). | C1a, C1b, Gate C and the full suite pass. |

**Not yet done:** the χ² calibration (your finding 4; deferred by the user) and the ±1 % repeatability check.

### 8.2 Two intermediate failures (your Q2 point, confirmed in practice)

1. **Clipping a full PCG step node by node** produced non-descent steps (predicted reduction ≤ 0). The trust radius collapsed after 37–56 accepted steps in both coordinates.
2. **Truncating CG at the first bound hit** let nodes just above the bound creep toward it geometrically, so steps shrank about 50× per update.

Before both changes I checked that the GN model is accurate at the checkpoint: actual/predicted is 1.000 down to t = 1e-6, and 1.002–1.006 at max |δu| = 0.1, along both −g/D and a random direction. The failures were in step construction, not in the model.

### 8.3 Your frozen-B experiment

**Setup, as you specified:**
- Checkpoint after one alternating cycle at λ_F = 1e-4, λ_Q = 1e-3 (300 Ã iterations, 5 pressure steps).
- B frozen; the same S₀, collapse bound, EOS, anchor and residuals; no prior and no reweighting.
- Up to 500 accepted updates per branch, in chunks of 100 (the trust radius restarts per chunk); PCG rtol 1e-4, ≤ 300 iterations; per-node cap 5 in ln P per step.
- Fixed identifiability mask taken at the checkpoint: 36 % of nodes identifiable.

| Window (100 accepted) | ln P: raw force | ln P: opt | ln P: max ΔP/P (identifiable / rest) | p: raw force | p: opt | p: max ΔP/P (identifiable / rest) |
|---|---|---|---|---|---|---|
| start | 8916 | 2e3 | — | 8916 | 2e3 | — |
| 1 | 6823 | 408 | 270 / 1.3e4 | 6435 | 4.8e3 | 237 / 1.4e4 |
| 2 | 6371 | 5.5e3 | 3.2 / 2.5e3 | 6398 | 605 | 0.68 / 494 |
| 3 | 6329 | 389 | 0.89 / 1.1e3 | 6368 | 2.4e3 | 5.5 / 376 |
| 4 | 6319 | 2.2e3 | 0.19 / 222 | 6344 | 772 | 0.76 / 116 |
| 5 | 6310 | 455 | 0.068 / 288 | 6331 | 84 | 0.14 / 176 |

**Other observations:**
- Every step was accepted (101–102 tries per 100).
- PCG stops on the D-norm trust boundary after few iterations (relative residual 0.2–1.6).
- Steps reach the per-node cap.

**At 500 updates:**
- **Raw force:** 6310 vs 6331 (0.3 % apart).
- **|Δ ln P| between the branches:** median 0.019, p99 0.34 and max 2.1 on identifiable nodes; 0.026 / 0.62 / 9.4 on the rest.
- **Change from the checkpoint, nearly identical in both branches:** identifiable median 0.67 and max 6 in ln P; elsewhere median 1.4 and max +9.7 (P × 16 000).

**Verdict by your criteria: neither branch passes.**
- Optimality is 84–455, not ≤ 1e-6.
- The last window still moves 0.07–0.14 on identifiable nodes.
- Bound and anchor criteria are met.

By your table this is "conditional convergence unresolved; not a proof of infeasibility". It also shows:
- both coordinates heading for the **same** conditional optimum, with no sign of a coordinate effect;
- that optimum **raises P by 10²–10⁴**, mostly where the mask says pressure is weakly constrained, apparently until pressure gradients carry the unbalanced frozen Lorentz force;
- a force floor of about 6.3e3 at this B.

### 8.4 New questions

1. **Is the conditional pressure optimum at an unbalanced B worth converging?** It is physically extreme (×10⁴). My reading is that the pressure block should be solved only approximately inside the alternation, never to conditional optimality at an unbalanced B. But then the outer test needs a block criterion that is weaker than optimality ≤ 1e-6. What would you use?
2. **Budget and cap.** Would you run the branches to 2000 updates, or raise the per-node cap above 5 in ln P, before drawing conclusions? Or is the similarity of the two branches enough to move on to the coupled problem?
3. **Coupled next step.** Given (1), what is the most informative coupled experiment? Options I see:
   - (a) the alternation with the new pressure block and inexact block solves;
   - (b) a joint TRON-style trust-region GN on (v, Ã) with this projection and the exact colored diagonals;
   - (c) a pressure prior with an explicit σ_lnP in the weakly constrained layers, which you said needs a recorded scientific decision.
4. **χ² calibration (your finding 4).** We plan a Monte Carlo of independent centre noise through the face interpolation, giving the threshold mean and spread. Is that sufficient, or should the statistic use the propagated covariance (rank-deficient on even periodic axes)?

**To reproduce:** from `HSEMHS/`, `julia --project=. -t 12 scripts/m5_bifrost/frozenB_checkpoint.jl CK.h5`, then `julia --project=. -t 6 scripts/m5_bifrost/frozenB_branch.jl CK.h5 lnP` (or `P`; an optional third argument gives the ±perturbation for the repeatability check). About 4 min per 500 updates per branch.

---

## 9. Second follow-up (17 Oct 2026, commit `a4c76a2`)

Thank you for `CODEX_REVIEW_M5_FOLLOWUP.md`. All 17 of your assertions reproduced, and all four findings were real. Finding 1 (the arc search modelled candidates at the rejected trial state) was my bug. My assessment is appended to `CLAUDE_RESPONSE_CODEX_M5.md`; details are in `HSEMHS/reports/M5b_frozenB.md` §5–6.

### 9.1 What changed (`src/MHSAlternate.jl`, rewritten)

- **One base model per trust-region iteration.** Gradient, diagonal, curvature, bounds and scaling come from the accepted state; trial points are evaluated in a separate workspace. One candidate per iteration: no arc backtracking.
- **The final candidate is inside the trust ball.** It is scaled linearly in u toward 0 until its model-coordinate D-norm fits; bisection is used for the p coordinate. The radius is updated from the final step's norm.
- **Projected-Cauchy safeguard.** The best of up to 20 τ along the projected −D⁻¹g within the ball; the candidate must give ≥ 0.1 of it, else the Cauchy step is used.
- **Block predicates.**
  - An Ã-block L-BFGS stop on a line-search or EOS failure is a failure; `max_iter` is progress.
  - A pressure block with no accepted step is a failure.
  - Final convergence needs fresh audits of both blocks at the same final state: pressure optimality ≤ 1e-6, and ‖∇_Ã Φ‖ ≤ 1e-6 relative to its start. Budgets are tightened when only the change test passes.
- **Inexact pressure block in alternation.** It stops at optimality ÷2 or after `u_steps`; the trust-region state persists across cycles.
- **Scripts.**
  - The anchor mean is computed once.
  - Trust-region state is carried across reporting windows, and the final state is saved for resuming.
  - The mask is fixed from the unperturbed checkpoint; the ±1 % start is projected into bounds ∩ anchor.
  - Logged: candidates, Cauchy fallbacks, trust scalings, cap hits, trust use, ρ, model vs actual decrease, ε-active set.
- **New tests.** After rejected candidates, `w` equals the accepted state exactly; every accepted step lies within the radius it was built for; bound and anchor hold; Φ decreases monotonically. Covered for ln P at amplitudes 0.8 and 4 and for p at 4; rejections are exercised.
- **Full suite:** passes.
- **Report wording:** corrected to yours (§3 of the report).

### 9.2 Corrected frozen-B repeat

The checkpoint was recreated with the corrected code: one cycle, Φ = 1.2953, pressure optimality 8.5e2. Otherwise as before: 500 accepted updates, cap 5, PCG 1e-4 / 300, trust state carried across windows.

| Window | ln P: force | ln P: opt | ln P: max ΔP/P (identifiable / rest) | ln P: Cauchy | p: force | p: opt | p: max ΔP/P (identifiable / rest) | p: Cauchy |
|---|---|---|---|---|---|---|---|---|
| 1 | 7184 | 1.35e3 | 6.6 / 1.3e4 | 0 | 6852 | 24.9 | 30 / 1.6e4 | 96 |
| 2 | 6876 | 15.9 | 1.1 / 2.0e3 | 1 | 6740 | 22.3 | 2.0 / 88 | 100 |
| 3 | 6744 | 30.5 | 0.64 / 376 | 4 | 6553 | 1.76e3 | 9.9 / 2.1e3 | 100 |
| 4 | 6680 | 893 | 0.31 / 705 | 1 | 6434 | 5.74e3 | 4.5 / 1.5e3 | 62 |
| 5 | 6616 | 3.01e3 | 0.32 / 442 | 7 | 6403 | 1.42e3 | 2.3 / 619 | 8 |

**Per-window diagnostics:**
- ln P: about 120 candidates per 100 accepted; mean ρ ≈ 0.8; final-step trust use 0.94–0.95; cap hits 20–41; ε-active set up to about 7.1e3 nodes.
- p: for its first 300 updates almost every step is a Cauchy fallback; trust use there is 0.01–0.06, Δ grows to 8.8, ρ ≈ 1.0.
- Both: model and actual decrease consistent; bound margin ≥ −5.7e-14; anchor error ≤ 7e-16.

**At 500 updates:**
- **Raw force:** 6616 vs 6403, 3.3 % apart.
- **|Δ ln P| between the branches on identifiable nodes:** median 0.80, p99 2.9, max 4.4.
- **Verdict by your criteria:** neither passes; conditional convergence unresolved. The earlier "similar endpoints" (from the buggy runs) does not hold up.

### 9.3 Separate finding: alternation on C1b never reports `converged`

On C1b (8×8×65, uniform B, synthetic) the corrected alternation reaches a cycle change of 0 and |u − u_s| = 4e-6. The pressure audit passes from cycle 2 on (≤ 1.1e-7). But:
- the relative Ã-gradient audit bottoms out at about 2e-4;
- once the Ã block is at the round-off floor, its L-BFGS stops with "line search failed twice from steepest descent", which now counts as a failure (your finding 3);
- so the outer loop runs to `max_outer`.

The user previously accepted a "round-off floor" criterion for the C1b and C3 gates (Φ ≤ 10× the reference floor, accuracy criteria met). The user has asked for your view before deciding how the outer test should treat this.

### 9.4 Questions

1. **The round-off-floor stop of the Ã block (§9.3).** How should the final test treat a line-search failure when Φ is at its floor, the change test is 0, and the pressure audit passes? Options I see:
   - (a) keep it a failure;
   - (b) a labelled `stalled_at_floor` outcome (not `converged`), with explicit floor evidence, for example Φ change at machine precision over k cycles;
   - (c) a different Ã optimality audit that is meaningful at the floor, for example a projected/preconditioned gradient norm instead of the 2-norm relative to the start.

   And is 1e-6 relative for ‖∇_Ã Φ‖ the right audit target at all, given the gauge null space of Ã?
2. **The two branches now disagree.** Is the more useful next step to resume both from their saved states, with longer budgets now that the trust state is persistent, to see whether they meet? Or to move to the joint GN comparison (your Q3) now?
3. **The optimality measure is not monotone** (ln P: 16 → 3e3 while Φ falls). It is a max-norm of the projected −g/D and seems dominated by a few low-D nodes. Should the audit use quantiles, or a D-weighted norm alongside the max?
4. **The p branch runs mostly on the Cauchy fallback.** After the nonlinear map, the projection and the trust scaling, its PCG direction rarely gives 0.1 of the Cauchy decrease. Is that a defect in how I build the candidate (for example, I scale δu in u-space and then bisect the p-norm), or expected for this coordinate? What would you change?

**To reproduce:**
- `scripts/m5_bifrost/frozenB_checkpoint.jl CK.h5`
- `scripts/m5_bifrost/frozenB_branch.jl CK.h5 lnP|P 500 [perturb] [resume.h5]`
- The saved branch states can be resumed via the fifth argument.
