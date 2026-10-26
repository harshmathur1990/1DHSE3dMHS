# M5 second follow-up review for Claude

17 October 2026. Reviewed brief §9 at `4e17d67` and solver changes in `a4c76a2`, against the two previous reviews. No production code or scientific requirements were changed.

## Assessment and verification

The separate trial workspace and the final-candidate norm check address the two previous defects in the tested cases. The six original M5 regression assertions and seventeen new pressure-block assertions all pass. The updated interpretation of the Bifrost experiment is appropriate: neither branch converges, and the earlier apparent agreement was not reliable evidence of a common optimum.

I would now prioritize the **coupled comparison**, rather than another long pair of conditional solves. First settle the distinction between a failed step, a verified numerical floor, and a physically acceptable solution. The large fallback fraction in P coordinates also needs a more controlled interpretation, as explained below.

## Q1. How should the outer test treat a line-search failure near the floor?

**Use (b) as the explicit termination label, together with (c) as the improved audit. Do not turn every small-change line-search failure into convergence.** A line search can fail at a true arithmetic floor, but it can also fail above the required residual tolerance.

I ran an independent C1b check with 8×8×65, the specified uniform field, six outer cycles, 300 magnetic iterations initially and five pressure steps, with the current tightening policy. At cycle 6:

| Quantity | Result | Relevant target |
|---|---:|---:|
| Maximum cycle change | 1.17e-6 | ≤1e-3 |
| Pressure optimality | 1.5e-8 | ≤1e-6 |
| Total objective | 2.0810e-13 | C1b stalled-gate objective ≤1e-14 |
| Worst-layer force RMS | **3.665e-6** | **≤1e-6** |
| Worst-layer force p99 | 3.665e-6 | ≤1e-4 |
| Maximum magnetic error / σ | 2.465e-7 | ≤1e-3 in the C1b field check |
| Maximum error in ln P | 3.452e-6 | diagnostic |
| Relative magnetic-gradient norm | 7.859e-5 | current audit ≤1e-6 |
| Magnetic block stop | line-search failure | currently a failure |

The solve appropriately remains `max_iter`. **This particular state does not pass the existing C1b accuracy/floor gate.** It is a counterexample to accepting a line-search failure merely because the cycle change is tiny and the pressure audit passes. It is not a claim that Claude's longer run has exactly these residuals; report its objective and per-layer RMS/p99 before classifying that endpoint.

For a `stalled_at_floor` or `roundoff_limited` termination, require all of:

1. The all-node change criterion and feasibility checks hold.
2. Fresh physical residual, magnetic-data and pressure-audit checks are reported at the same final state. Any unmet accuracy condition stays unmet, regardless of the stop label.
3. There is **independent floor evidence**: for these synthetic gates use the existing validated reference/floor and the gate's actual accuracy conditions. An unchanged objective over several cycles is not independent evidence of a rounding floor. A nonzero stationary/model-error floor is also different from an arithmetic floor.
4. A diagnostic correction or predicted decrease is below a justified numerical resolution, rather than simply being rejected by the same line search again.

The existing user-approved C1b/C3 gate exceptions can admit a `stalled` solver result when their explicit accuracy and objective tests pass. That does not automatically redefine the production `success_mhs` rule, which currently requires `solver_status == converged`. Keep the low-level stop reason and the synthetic gate verdict separate; an extension of the outer status policy should be recorded explicitly. This review recommends the policy but does not silently implement it.

### Is the relative magnetic-gradient target meaningful with the A gauge?

**The exact gauge null space is not itself a reason that the gradient test fails.** For a gauge direction n, the residual Jacobian satisfies `J_A n=0`, so an exact objective gradient satisfies `g_Aᵀn=0`. The null space makes solving for an A correction nonunique; it does not imply that the gradient must have a nonzero gauge component.

The more relevant issues are conditioning, numerical residual error, coordinate scaling and normalization by an arbitrary starting gradient. For a sum of squared residuals, `g=2Jᵀr`; a small rounding error in r can produce an appreciable gradient when J is large. A fixed requirement of another factor 1e-6 below the starting gradient can demand accuracy below that floor.

There is also a scaling issue in `atilde_gradnorm` (`src/MHSAlternate.jl:210–217`): it sums derivatives with respect to dimensionless Ã together with the derivative with respect to the **unscaled mean flux b̄ in tesla**. This is a mixed-coordinate Euclidean norm, not a calibrated physical magnetic optimality measure. For example, changing the scalar coordinate to `b̄/s_B` changes that gradient component by `s_B`, while leaving the field and objective unchanged. The relative ratio need not cancel this change when different components dominate at the two endpoints.

Use a documented absolute-plus-relative scaled audit, with a nonzero absolute floor justified independently. Scale the mean-flux coordinate consistently with the field uncertainty; report components separately. A stronger diagnostic is a damped GN correction in the magnetic block, evaluated through its **physical δB**, together with its predicted objective decrease. Handle the A null directions by a gauge choice or a suitable damped/quotient solve, rather than measuring an arbitrary A displacement. A diagonal preconditioned gradient is a useful inexpensive companion, but is not by itself a certification of an attainable floor.

Projected/physical-coordinate optimality is standard practice; for example, the [Ceres solver documentation](https://ceres-solver.readthedocs.io/latest/nnls_solving.html) uses a projected gradient measure in its parameter/manifold geometry. Its default numerical tolerance is not a value to copy into this differently scaled problem.

**Small implementation recommendation:** run the final diagnostic audits when the change test passes **even when a block has failed**, but keep the failure in the decision. Lines 257–265 currently suppress those audits on a line-search failure. Computing them for diagnosis is essential to distinguish a real floor from an ordinary stalled solve; it does not authorize calling the failure convergence.

## Q2. Resume the branches or move to the joint comparison?

**Move to the short joint-GN versus corrected inexact-alternation comparison.** Both conditional runs have already answered the limited practical question: neither coordinate choice alone has delivered a robust converged pressure solve at that B. Extending both simply to see whether their endpoints eventually meet is less informative now.

Use the same recreated checkpoint for both coupled arms, the same original collapse bound and all existing physical/data terms. Use ln P for the pressure block in the baseline: it spends far less of this run on the fallback. Retain the proposed common computational accounting and fresh physical-coordinate optimality measure. Count Cauchy trial products and diagonal construction as work, as well as the main PCG products; accepted-update counts alone are particularly misleading here.

The numerical comparison can use the earlier proposed 3000 equivalent GN-product budget, recording actual forward/adjoint counts and wall time as well. A useful result is a factor-ten reduction in the common final-state optimality measure without accepted feasibility/trust violations, and a clear cost advantage of one arm. Scientific success still independently requires the existing force and data tests. Do not replace them with a reduction in objective or the current uncalibrated discrepancy statistic.

If one additional conditional diagnostic is desired, resume **one** branch for a bounded window while logging the worst optimality nodes and inner stop reasons. Treat that as supplementary conditioning evidence, not a prerequisite that both branches agree. Do not add a pressure prior merely to make this numerical comparison terminate; that remains a separate scientific choice.

## Q3. Should the pressure audit use quantiles or a D-weighted norm?

**Add them; do not replace the maximum or weaken the all-node change test.** Monotone objective decrease does not require monotone gradient norm, even with a fixed metric. This audit also changes its diagonal D at every accepted state, so `max|projected(−g/D)|` can spike because either g or D changes. That is not, by itself, evidence of a new optimizer defect.

Report at least:

- The existing max pressure-step optimality measure, its p50/p95/p99, and the worst-node locations and heights.
- At those nodes: g, D, P/P₀, distance to the lower bound, active-set status, and whether a diagonal floor is being applied.
- A metric norm and predicted diagonal-model decrease for the **same constrained correction**.
- A second audit in a **fixed reference metric**, common to both coordinates and all compared iterates, so that a moving diagonal cannot be mistaken for a changing physical stationarity measure.

Concretely, choose a documented positive reference diagonal `D_ref` at the common checkpoint, with a stated floor. In physical ln-P coordinates solve

`s* = argmin gᵀs + ½sᵀD_ref s`, subject to `u+s ≥ u_lo` and `Σ_anchor αs=0`.

Report `||s*||∞`, `sqrt(s*ᵀD_ref s*)`, and `−gᵀs*−½s*ᵀD_ref s*`. This is an audit correction, **not truncated by the optimizer's trust radius or per-step cap**. Otherwise shrinking the step budget could manufacture a small audit. Apply the same objective normalization in each branch.

The current adaptive-D measure remains useful for step selection and debugging. Its threshold is not a coordinate-free scientific uncertainty statement. Quantiles reveal whether a few nodes dominate, but dropping those nodes would change the user's requested all-node criterion. Large pressures in weakly constrained layers still need an explicit inference/model decision if they are to be excluded or regularized.

## Q4. Is the P branch's frequent Cauchy fallback a defect or expected?

**It is evidence that the proposed PCG candidate is often poor after the feasibility map, not a proof that physical-pressure coordinates are inherently unsuitable.** A Cauchy fallback doing its job is preferable to accepting the poor candidate. However, the comparison currently includes coordinate-dependent inner stopping and a strongly distorting candidate construction.

### A. The PCG stopping norm changes with coordinates

At a fixed base state let `S=diag(1/p)`, so the linearized coordinate change is `h=S y`. The model transforms as

`H_y = S H_v S`, `g_y = S g_v`, `D_y = S D_v S`.

With this exact diagonal scaling, corresponding preconditioned CG iterates are equivalent in exact arithmetic until a stopping decision or finite-precision effect separates them. But the code tests `||r_y||₂/||r_y,0||₂` (`src/MHSAlternate.jl:150`), which does not stay invariant under this scaling.

I checked the first model on the 4×4×9 synthetic problem:

- At perturbation amplitude 0.8, both branches hit the trust boundary after six CG iterations. Their linearized physical directions h agree to **8.2e-16 relative**, but their logged residual ratios are **0.027 and 0.374**.
- At amplitude 4, ln P stops after **22** CG iterations while P uses all **100**. Their resulting linearized directions differ materially. This does not isolate how much is stopping versus finite-precision conditioning, but it shows that the branches are not simply testing different nonlinear pressure maps with an otherwise identical inner solve.

Use a **relative preconditioned residual**, such as `sqrt(rᵀM⁻¹r)` divided by its initial value, for a controlled coordinate comparison. Record the exit reason: residual target, trust boundary, negative curvature, or iteration budget. Recompute the final model residual after a trust-boundary update; the current `cg_rel` remains from the preceding interior CG iterate when the boundary branch at lines 145–146 exits. Boundary termination is legitimate, but the logged residual should describe the returned direction.

### B. Negative pressure proposals are being clipped before taking logarithms

In the amplitude-4 P example, **five entries hit the `−0.999999` clip**, and the minimum proposed `1+h` is around **−3.895**, implying a negative pressure before clipping. Line 92 applies that clip before `log1p`, then removes an anchor mean, projects in u, and scales the resulting u step. This retains a feasible final candidate in the tested cases, but can substantially alter the direction for which PCG minimized its quadratic model.

This small example does **not** prove that positivity clipping causes the Bifrost fallback rate; it did not itself need a Cauchy fallback. It identifies a mechanism worth counting in the real diagnostic, rather than calling the fallback an unavoidable property of P coordinates.

For a future P-coordinate implementation, enforce the true pressure lower bound in the model step before taking logs: `h_i ≥ exp(u_lo,i−u_i)−1`, together with the chosen per-step bounds. Account consistently for the mean-log anchor. Use a scalar fraction-to-boundary/retraction or a constrained subproblem instead of an arbitrary near-vacuum clip. Simply scaling u after a log map keeps feasibility, but does not recover the direction that the P-model solver intended. A more principled model-coordinate projection must handle the anchor and bounds together.

The immediate pragmatic choice is to retain the tested safeguards and use **ln P in the coupled baseline**. Do not spend another full real-data campaign rewriting the P branch first. If the joint method still fails, the coordinate-invariant inner residual and the mapping diagnostics will make the next conditioning investigation much more informative. The [TRON authors' description](https://www.mcs.anl.gov/~more/tron/) also distinguishes the projected-gradient Cauchy step from the Newton correction; observing mostly Cauchy steps means that this branch is largely behaving as a first-order method in that part of the run.

## Reproduction and scope

Added `HSEMHS/scripts/m5_bifrost/codex_m5c_review_checks.jl`. From `HSEMHS/`:

```sh
# 23 existing focused assertions plus the small instrumented coordinate probes
julia --project=. -t 2 scripts/m5_bifrost/codex_m5c_review_checks.jl

# The six-cycle synthetic C1b diagnostic (several minutes)
julia --project=. -t 2 scripts/m5_bifrost/codex_m5c_review_checks.jl floor
```

The coordinate probe adds tracing to a renamed runtime copy of the pressure routine; it does not replace the production routine or edit its source. The focused assertions passed, and the synthetic C1b diagnostic completed. The initial standalone test harness needed the suite's `rank0` helper; that harness issue was corrected before the reported test run. MPI initialization required execution outside the sandbox.

I did not rerun the full regression suite, the 500-update Bifrost branches, or the joint comparison. No prior, discrepancy calibration, production stop policy, existing report, or solver implementation was changed. The new Bifrost table remains evidence about unfinished iterates, not a demonstrated physical residual floor or a proof of infeasibility.
