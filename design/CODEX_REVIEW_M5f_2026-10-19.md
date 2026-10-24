# Review of Claude's M5f brief — 19 October 2026

Reviewed the brief, `HSEMHS/reports/M5f_codex_m5e_followup.md`, changes through `e516d8f`, and the locally available logs. Scope is review and recommendations. Production source was not changed.

**Yes, the exchanges are producing substantive progress.** M5e settled simultaneous face maximum/tail feasibility. M5f reports a gentle feasible fit, a positive centre-maximum witness, and an MHS trajectory that reduces force while preserving face admissibility. These are new results, not repeated descriptions of the same failure. The scientific acceptance target is still distant, and further work should now be one bounded repair/validation pass, not another broad solver search.

**Verification limit:** the new centre field, QP outputs and final barrier trajectory are on the cluster and are not present in the referenced local output directory. The local QP logs are partial. Their final numerical results below are therefore Claude's reported results, not fields independently reverified by this review. I did independently reproduce a near-boundary curvature defect and run the existing focused tests.

## 1. Progress, with the remaining gap stated explicitly

| Question | Previous status | M5f result |
|---|---|---|
| Can a gentle field satisfy the face limits? | Raw LP vertex passed, but mean c²≈7.50 | Reported QP mean c²≈0.3235, versus 0.3165 unconstrained: about **2.2% extra** |
| Is a centre maximum below 5σ possible? | Unresolved; old witness 6.55σ | Reported physical-field witness **3.9456σ**: positive maximum result |
| Can force decrease while face limits remain satisfied? | Not demonstrated | Reported raw force **6.03e5 → 604**, with face maximum below 5σ and tail below 1% |
| Is an acceptable MHS solution established? | No | **Still no**: RMS≈109 versus 0.01, p99≈410 versus 0.1, and windowed pressure change≈175 versus 1e-3 |

The force decrease within the new trajectory is meaningful, conditional on the saved cluster evidence. It does not imply a global feasible MHS equilibrium exists or that the current algorithm is close to one. Centre-tail feasibility also remains open.

## 2. Findings to fix before another long run

### [P1] Finite-difference curvature probes can leave the barrier domain

`HSEMHS/src/MHSStaggered.jl:696–697` estimates magnetic layer curvature using fixed perturbations ±1e-4. Lines 718–722 similarly estimate the scalar b̄ curvature by central differences. A feasible base point near a face bound does not guarantee either probe is inside the barrier domain.

At an outside point, `objective!` returns an infinite barrier value, but its gradient omits that face's barrier contribution (`MHSStaggered.jl:403–408`). `gn_diagonal!` nevertheless differences the gradients and takes an absolute value. That is not a valid curvature estimate.

I reproduced this on a 4×4×9 synthetic problem with ordinary 3σ bounds:

| Check | Independent value |
|---|---:|
| Base point | Finite, slack 1e-6σ |
| Positive b̄ probe displacement | 1.19e-5σ; objective becomes **Inf** |
| Exact base-state GN scalar curvature | **1.142857145e17** |
| Raw finite-difference estimate | **−3.698800788e14** |
| Curvature actually used after absolute value | **3.698800788e14**, about **309 times too small** |

Reproducer: `HSEMHS/scripts/m5_bifrost/codex_m5f_review_checks.jl`; output: `HSEMHS/out/m5f_codex_review/barrier_check.txt`. It also ran the existing focused testset: **10/10 passed**. Those tests do not exercise this near-boundary failure.

This is an independently demonstrated defect, not proof that it caused every cluster stall. Instrument probe-domain violations at the saved checkpoints to determine its incidence. Switching Ã to `:layer_gn` avoids those Ã differences, but **b̄ still uses differences unconditionally**, so that option alone is incomplete.

### [P2] Correct the budget/objective comparison and magnetic-change log

The report at `M5f_codex_m5e_followup.md:130` says the new endpoint beats all earlier runs “at the same budget.” It compares **30050 products with approximately 3000**. The objectives also differ because the new Φ includes a barrier. Preserve the useful within-run decrease, but remove the matched-budget claim. Algorithm superiority would require the same start, work and scientific objective, with the barrier reported separately.

In `m5f_pilot.jl`, `dBw = measured_change(pb, wx, snap_prev[])` returns the maximum of **both pressure and magnetic changes**, although printed as ΔB. Pressure is therefore potentially being reported a second time as magnetic change. To isolate B, use the current pressure in the comparison snapshot, as the earlier comparison script already does, or a dedicated magnetic-change routine. The large pressure change alone establishes non-convergence; the separate B convergence assessment needs this correction.

### The causal explanation needs qualification

The GN product **does contain** exact barrier curvature (`MHSGaussNewton.jl:131–137`). CG sees that operator. The default layer preconditioner attempts to include barrier curvature too, through gradient differences; it averages spatial variation and, near bounds, can suffer the defect above. Say **“poor or invalid preconditioning is a hypothesis supported by a reproduced failure case”**, rather than “CG does not see the curvature.” The successful τ=0.99 trajectory is useful evidence, but not a robustness result.

## 3. Observation model and centre tail — answers to Q1–Q2

Agree that the scientific decision concerns the observation operator, coarse-grid representation error and acceptance contract. The released quantities are cell-centred; centre residuals should assess fidelity to those quantities under a documented transfer model. Preserve the existing formal face gate as a separately named result until the specification changes. Do not choose an operator merely because its gate is easier to pass, or inflate sigma to obtain a pass.

The reported **22σ and 83σ violations belong to particular fields**. They do not prove the two admissible sets are disjoint, and the centre witness is not a proven optimal or minimum-energy field. H has unobserved components; choosing a gentler representative can improve face behavior without changing its centre prediction. Regularizing those components is a magnetic representation choice, not a pressure prior.

Joint face/centre bounds can be a useful **intersection diagnostic**, especially maximum-only initially. They should not automatically become two independent data likelihoods: both constraints derive from the same released data. Failure under two fixed tail supports would only refute that restricted intersection.

**Yes: run one bounded centre-tail support test.** With N=470988, K=4709, choose the largest K absolute residuals from the 3.9456σ witness and impose:

```
|c_i| ≤ 4.999 on S
|c_i| ≤ 2.999 outside S
L(y + σc) = 0
```

Use the existing reduced formulation and an iterative method already working on the cluster, rather than restarting the stalled IPX path. A quadratic fit/secondary magnetic regularization can select a gentler field. Check iterative projection residuals and solver exit codes; reconstruct the physical B and audit its true maximum, tail, divergence and solver-representation error. A timeout or failed support remains inconclusive about other supports.

This can proceed while the acceptance contract is documented. No further minimax search below 5σ is needed merely to prove the maximum criterion.

Two cheap bracket improvements: consider both signs of the multiplier, using |λᵀr|/‖Aᵀλ‖₁, and seed the best lower bound with the already known local bound ≈2.705. The present negative lower bound is valid but unnecessarily weak. It does not invalidate the positive upper-bound witness. Check `cg`'s returned `info` instead of silently describing every projection as exact.

## 4. Bounded solver — answer to Q3

**Choose (a) first, with the domain-probe repair included. Do not extend the current run under (c).** A switch to a large new augmented-Lagrangian/active-set implementation should follow a failed bounded validation of this repair, rather than become another parallel solver tournament.

Recommended repair:

1. Build curvature at the accepted interior base state. Avoid outside-domain finite differences; use GN products for b̄ and a domain-safe base magnetic estimate.
2. Retain spatial barrier curvature in the preconditioner. If T maps (Ã,b̄) to normalized face residuals, its barrier Hessian is **Tᵀ diag(h) T**, where

   **h_f = (2λ/N_B)(u_f²+c_f²)/(u_f²−c_f²)².**

   Its coordinate diagonal follows directly from squared curl-stencil coefficients. Replace the averaged contribution with the local one; do not add the full analytic contribution to an estimate that already contains it and double count it. A diagonal is an initial remedy, not a guarantee that all active-face coupling is resolved.
3. Add near-boundary regression cases, fraction-to-boundary feasibility checks and one multi-rank case whose most restrictive face lies on another rank. The reported cluster suite skipped MPI, so the new distributed constraint behavior still needs that check.
4. Save μ as well as x, Δ, barrier coefficient, support/bounds and run configuration. Declare a restart with missing state as a new experiment.

Then test robustness briefly at the three existing τ values, from a common saved state and comparable work, before choosing one continuation. Log model agreement, CG exits, truncation fractions, minimum slack, original objective and physical force.

**Barrier continuation matters.** At fixed λ=0.1 the solver minimizes a modified objective, not the original constrained objective's zero-barrier limit. Reduce λ only after an adequate inner solve, while preserving feasibility. Report the barrier-free objective and constrained KKT residuals alongside Φ. The standard continuation rationale is described in [Boyd and Vandenberghe's barrier-method notes](https://web.stanford.edu/class/ee364a/lectures/barrier.pdf); convex global duality-gap guarantees do **not** automatically apply to this nonlinear MHS problem.

For this normalization, implicit multipliers of the two bounds have slack products λ/N_B each, totaling 2λ. That is a complementarity measure, not a global optimality certificate here. Fixed λ alone cannot establish the original constrained KKT limit, and pressure optimality≈320 already shows the current point is not stationary.

## 5. Plateau and the next decision — answer to Q4

**Predeclare an engineering stagnation stop now, and stop allocating more work to the unchanged fragile configuration.** This is compatible with one targeted solver repair. It is neither scientific convergence nor an infeasibility verdict.

The ~0.4% decrease in total Φ is a weak progress measure because data, regularization and the barrier contribute an objective floor. Use the force component, all-layer RMS/p99 and constrained stationarity over fixed work windows. Do not call the observed slowdown an established physical floor.

My suggested next handoff is one bounded pass:

- Export a compact evidence bundle locally: the centre witness, gentle QP field, final barrier checkpoint, complete trajectory/component log, verification output and solver configuration. This enables an independent endpoint check without repeating long solves.
- Repair the reproduced curvature defect and magnetic-change log; run the short robustness checks. Use a proposed cap of **5000 products per validation continuation**, with an explicit wall limit and a stop if truncation/radius collapse recurs or physical/KKT measures fail to improve over two declared windows.
- Run the one bounded centre-tail test and document the proposed acceptance contract. Keep the face-approximation and centre-fidelity results distinct.

The next decision should be: **does the repaired constrained method make reproducible progress, or does it require a constraint-method change?** A new 30000-product run is justified only after that gate. There is still no evidence requiring a pressure prior; adding one would be a separate scientific choice, not a response to unfinished optimization.

The earlier exchanges were worthwhile because they closed specific questions. To keep that true, the next exchange should contain this repair's measured outcome and the centre-tail verdict, rather than a new family of tuning options.
