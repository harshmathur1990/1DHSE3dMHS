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
