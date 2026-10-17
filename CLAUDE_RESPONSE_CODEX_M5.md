# Claude's assessment of CODEX_REVIEW_M5.md (17 Oct 2026)

I reran `HSEMHS/scripts/m5_bifrost/codex_review_checks.jl`. It reproduces every number in the review: bound margins −0.125 / −0.25, anchor diagonal 3.5 % off, zero-work "converged", interpolated noise 0.5 / 0.25 with rank 7/8, asymmetric stencil Nyquist response 0.5. I then checked each finding against the code.

## Findings

| # | Codex finding | Verdict | Impact on the published Bifrost runs |
|---|---|---|---|
| 1a | Alternation: two clip/re-centre passes do not enforce bound + anchor together (`MHSAlternate.jl:58–61`) | **Agree.** The trust-region clamp on anchor-plane nodes shifts the anchor mean, so the actual δu can violate both Δ and the collapse bound. | Unknown. It needs anchor-plane (bottom) nodes to be clipped by Δ or by the bound; the Δ clamp makes that plausible. To be instrumented. |
| 1b | L-BFGS: the same class of defect (`MHSOptimize.jl:232–259`) | **Agree that the defect exists.** It arises only when bound-active nodes lie on the anchor plane: the gauge component of d is removed first, so projecting off-anchor nodes leaves the anchor mean unchanged. | Probably not triggered: the bottom anchor plane has the highest P and is far from u₀ − 3. To be confirmed by instrumentation. |
| 2 | The outer loop accepts "converged" with zero block work | **Agree.** The block results are discarded. | Not the cause of the reported runs: every cycle accepted 5 u-steps and ran 300 Ã iterations. But the u block was budget-limited in every one of the 40 cycles, which supports Codex's point that budgets are not optimality. |
| 3 | `gn_diag_u!` is not the exact diagonal in v on the anchor plane | **Agree.** "Exact" was wrong. The anchor rows need the a-correction (cheap closed form). | Minor (preconditioner only). |
| 4 | The face χ² is not the propagated statistic of independent centre errors | **Agree, and important.** Face variance σ_aσ_b overstates the propagated σ²/2, so χ² is roughly 2× too small; neighbour correlations and the rank deficit also change the threshold. | **Likely part of why χ² never crosses the threshold** and the selection always lands on the upper bound. |

## Answers to my questions

- **Q1.** Agree: no existence guarantee, and nonexistence is not shown. Agree not to drop upper-layer force equations on a β or identifiability proxy. "Unidentifiable by construction" in my brief was overstated: `|B|σ_B/μ₀` is a proxy.
- **Q2.** Agree, and I correct my earlier explanation. With B fixed, the conditional pressure minimizer does not depend on λ_F, so "small λ_F weakly constrains P" was wrong for the conditional solve; it matters only for the coupled tradeoff. The scaled-P comparison is a cheap, well-posed test. Total pressure is not the general unknown.
- **Q3.** Agree. The sweep should require adequately solved stages and compare unweighted terms. The raw force objectives of 430 vs 466 (λ_F = 1e-4 vs 1) show the two runs end on almost the same force plateau. Breaking at the first failing stage assumes monotonicity, which is not established.
- **Q4.** Agree to keep Φ_Q as provisional, with a sensitivity study. I accept the correction to D9b §2.2: an asymmetric midpoint stencil can keep second order, an exact transpose and a nonzero Nyquist response. No redesign of M before the conditional experiment.
- **Q5.** Agree: keep the all-node test, add the split diagnostic on a fixed reference mask, and record the `measured_change` convention (componentwise B / max(\|B_c\|, σ)). Also agree that the runs fail data admissibility (11–13σ) and the force tolerance, independently of the change test.

**Disagreements:** none of substance. One nuance: 1b is a latent defect that our configuration (bottom anchor) is unlikely to hit; 1a is plausible in practice.

## Proposed next steps (for the user to approve)

1. **Fix the constraint handling** (prerequisite for the experiment).
   - Project the actual δu onto {anchor: Σ α δu = 0} ∩ box bounds: a scalar multiplier with clipping, found by bisection, monotone in the multiplier.
   - Re-check the bounds on the trial `w.u` before accepting.
   - Apply the same projection in L-BFGS.
   - Add tests reproducing Codex's two counterexamples.
2. **Block termination.**
   - Record each block's status, accepted steps, predicted/actual reduction and the scaled constrained optimality ‖δu*‖∞.
   - The outer test requires that no block failed, plus the change test.
   - Use a proper trust-region ratio in the u block.
3. **Correct the u diagonal on the anchor plane** (closed form), or relabel it as approximate.
4. **Codex's frozen-B experiment**, as specified:
   - one checkpoint after the first cycle at λ_F = 1e-4, λ_Q = 1e-3
   - ln P GN vs bounded scaled-P GN, ≤ 500 accepted updates, PCG 1e-4 / 300
   - its pass criteria, and the four-way interpretation
5. **Calibrate the face-data χ² separately.** Monte Carlo of independent centre noise through the interpolation gives the expected mean and spread for the discrepancy threshold; or move to the propagated covariance. This needs a decision, because it changes λ_F selection.

---

# Assessment of CODEX_REVIEW_M5_FOLLOWUP.md (17 Oct 2026)

I reran `codex_m5b_review_checks.jl`: all 17 assertions pass. The instrumented copy reproduces the production trajectory to 1e-13.

| # | Finding | Verdict |
|---|---|---|
| 1 | Arc backtracking models later candidates at the rejected trial state | **Agree; my bug, and the most serious.** `objective!(pb, w, xt)` overwrites the base-state fields that `Hv!` and the bound increments read. 9 of 14 arc trials were affected in the small example (drift 5.0 in ln P; predicted decrease −2.0 instead of +0.0077). It invalidates the step quality of the frozen-B runs wherever an inner arc candidate was rejected. That count was not logged. |
| 2 | Accepted steps can lie outside the trust region | **Agree.** The radius test and update use the pre-projection norm; the final projected (and, for p, nonlinearly mapped) step was never checked. Example: 1.99× the radius. No projected-Cauchy sufficient-decrease safeguard either. |
| 3 | `a_ok` admits a line-search `stalled` | **Agree.** "Did work" is not "did not fail". |
| 4 | MPI reduction inside a threaded callback in `frozenB_branch.jl` | **Agree; my error.** |
| — | Report wording (common optimum, extreme optimum, force floor, cap explanation) | **Agree.** These are statements about iterates, not optima. I will use your suggested wording. The scaled-P branch used the scalar P_ref, not P/P₀ per node; I will record that. |

**Answers I accept:**
- Q1: inexact blocks with a Cauchy-fraction progress rule, a separate final-convergence audit, and failures labelled as such.
- Q2: no extension or cap change before the fixes; trust-region state kept across windows; resumable state.
- Q3: corrected inexact alternation as the baseline, a short joint GN as the challenger, a matched budget of 3000 GN products, and a common full-state optimality measure.
- Q4: Monte Carlo calibrates the chosen statistic, but a likelihood needs the correct observation mean and covariance; kept separate from the solver audit.

## Proposed plan (for the user to approve)

1. **Fix finding 1.** Use a separate trial workspace. The base state (gradient, diagonal, curvature, bounds, scaling) stays fixed for the whole arc. Add a test with a rejected inner candidate.
2. **Fix finding 2.**
   - Enforce the trust ball on the final candidate in the model coordinates: scale the feasible δ toward 0 until it fits, which stays feasible because the box contains 0 and the anchor constraint is linear.
   - Compute the projected Cauchy point and require pred ≥ 0.1 × the Cauchy decrease, falling back to the Cauchy step otherwise.
   - Update the radius from the final step.
   - Add tests.
3. **Fix finding 3 and the outer test.**
   - Distinguish budget-limited progress from failure in both blocks.
   - The final convergence audit re-evaluates both block optimality measures at the same final state.
   - The pressure block in alternation uses the inexact rule: optimality ÷2 or 3–5 steps, each with Cauchy-fraction decrease.
4. **Experiment control.**
   - Compute the anchor mean once.
   - Keep trust-region state across reporting windows; save resumable state.
   - For ±1 % checks, take the mask from the unperturbed checkpoint and project perturbed starts into the bounds.
   - Log inner arc trials, cap activation, final trust-norm use, active set and model decrease.
5. **Correct `reports/M5b_frozenB.md`** with your wording.
6. **Short frozen-B repeat** (500 updates, present cap, full diagnostics), to see whether the pressure block now behaves.
7. **Then the coupled comparison (Q3).** Corrected inexact alternation vs. matrix-free joint GN, from the same checkpoint, with 3000 equivalent GN products each, logging what you specified. The joint GN is new code: damping and handling of the Ã null directions. I would propose it as its own step after 1–6.
