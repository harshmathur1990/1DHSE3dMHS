# Independent M5 review for Claude

17 October 2026. Reviewed commit `e22b195` **plus the existing uncommitted working tree**, including `MHSAlternate.jl`. This is a review and proposed experiment, not approval to change the scientific requirements. No existing solver, specification, or Claude work was edited.

## Main finding

Do not decide that this is exclusively a formulation problem yet. There are reproducible safeguard and stopping defects to address first. Conversely, fixing numerical convergence would not establish a successful MHS reconstruction: the reported layer residuals and maximum magnetic deviations already fail §6.7 independently of the pressure-change criterion.

My recommended order is: **correct the feasibility checks; isolate and converge the conditional pressure problem; then reassess the coupled fit and physical feasibility**. Do not lower the acceptance standard simply because the remaining pressure is described as unidentifiable.

## Findings verified against the code

### 1. The pressure bound is not reliably enforced after the anchor projection

`src/MHSAlternate.jl:58–64` clips a proposed pressure change twice, but the actual change is `d - anchor_mean(d)`. Clipping changes that mean. Two passes do not solve the intersection of the bounds and the anchor constraint.

A concrete example with equal-area anchor nodes:

- Raw anchor direction: `[-2, 2]`; permitted change magnitude: `0.5`.
- The first node is already at its lower bound, so its permitted change is nonnegative.
- After the two passes in the code, the stored direction is `[0.25, 0.5]`.
- Its anchor mean is `0.375`, giving the **actual** pressure-log change `[-0.125, 0.125]`.
- The first node violates the lower bound by **0.125 in ln P**.

The independent reproduction uses the production field types and `anchor_mean`, on a 4×4×9 synthetic problem.

The L-BFGS safeguard has the same class of issue (`src/MHSOptimize.jl:232–259`): `project_lower!` changes the anchor mean, the code computes the resulting actual `du`, but `step_to_lower` skips nodes already at the bound. Their newly negative `du` need not be round-off. The reproduction obtains an allowed step with an actual bound violation of **0.25 in ln P**.

**Recommendation:** construct the actual `δu` in the intersection of the anchor tangent constraint and the box bounds, using a constrained projection or active-set subproblem. On the anchor plane, enforce `Σ α δu = 0` together with the bounds; then set `δv = δu`. For a weighted projection this can be solved with a scalar multiplier and clipping. Check the bounds again on the actual trial `w.u` before acceptance. Apply the same principle to L-BFGS. Repeating a fixed number of re-centering/clipping passes is insufficient.

These are counterexamples to the claimed safeguard, **not evidence that these violations caused the published Bifrost runs**. Record their occurrence and size in a real run to establish impact.

### 2. Zero work can be reported as convergence

`src/MHSAlternate.jl:95–105` discards the magnetic-block result, receives no pressure optimality result, and accepts convergence from endpoint change alone. On a deliberately perturbed synthetic state, `max_outer=1, a_iter=0, u_steps=0` returns `converged`, change `0`, with objective `6.510455e-3`.

Zero budgets expose the logic clearly; failed line searches, exhausted pressure retries, or tiny trust radii can produce the same stationary endpoints with ordinary budgets. Monotone descent also does not exclude these failures, or alternating motion along a shallow coupled valley.

**Recommendation:** retain each block's status, actual/predicted reduction, accepted steps and constrained optimality measure. Small measured change is a useful stopping observation, but insufficient evidence that the blocks have been solved. A failed block must not silently satisfy the outer convergence test. Bound-aware optimality and change tests measure different properties; see the [SciPy least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

### 3. The advertised exact pressure diagonal is not exact for the optimizer coordinates

`gn_diag_u!` (`src/MHSStaggered.jl:554`) constructs the diagonal in physical `u` before applying the anchor parameterization. `gn_hvp!` acts in `v` and includes `Gᵀ H_u G`, where `G = I - 1 aᵀ` and `a` contains the anchor weights.

Thus

`diag(Gᵀ H_u G)_i = (H_u)_ii - 2 a_i (H_u 1)_i + a_i² (1ᵀ H_u 1)`.

Off the anchor, `a_i=0`, so the diagonals agree. At the anchor they need not. With a nonuniform pressure perturbation, the reproduction finds **3.50% relative disagreement** at the selected anchor node and exact agreement at the selected non-anchor node. Floors make the result a preconditioner rather than an exact diagonal in any case.

This does not make PCG invalid: a positive approximate diagonal can still precondition it. It does mean the claim of an exact coordinate diagonal needs qualification or correction. The small example does not establish this as the main real-data bottleneck.

### 4. The face-data statistic is not the original independent-observation chi-square

`src/MHSStaggered.jl:113–123` interpolates centre observations onto faces and uses variance `σ_i σ_j`, with a diagonal weight matrix and the count of faces as `N_B`.

For genuinely independent centre errors, the propagated covariance is instead

`C_f = I C_c Iᵀ`.

For equal centre variances `σ²`, an interior two-point average has variance `σ²/2`; adjacent interpolated faces share covariance `σ²/4`. On an even periodic axis the interpolation even loses a Nyquist degree of freedom. The reproduction gives diagonal `0.5`, adjacent covariance `0.25`, and covariance rank **7/8** for eight unit-variance centre observations.

This is not repaired by changing every denominator by a factor of two: the correlations, rank, interpolation bias and observation locations also matter. Treating the face residual as an engineering penalty is possible, but its discrepancy threshold then needs calibration. The setup currently assigns synthetic uncertainty scales to a simulation snapshot; it has not established an independent Gaussian measurement-error model.

## Answers to the five questions

### Q1. Is a static solution expected to exist with fixed T and constrained B?

**There is no existence guarantee, but the current runs do not prove nonexistence.** Bifrost evolves the time-dependent momentum equation, including momentum transport and viscous stress, rather than requiring instantaneous static balance. This is explicit in equation (2) of the [Bifrost code paper](https://arxiv.org/html/1105.6306). My inference is that a fixed-T static fit to this snapshot can have irreducible model error. Replacing its thermodynamics with an LTE EOS and coarse-graining T and B can add mismatch. In particular, averaging magnetic stress is not equivalent to forming stress from the averaged B.

The continuum compatibility conditions are stronger than a small plasma beta. Along the field,

`B · (∇P + ρ(T,P) g zhat) = 0`.

Horizontally, `∂x P = Lx`, `∂y P = Ly`, so an exact solution requires `∂y Lx = ∂x Ly`, plus zero horizontal circulation around periodic cycles. Closed field lines impose further compatibility of the pressure ODE with the fixed temperature. Use these as diagnostic constraints, with discrete operators appropriate to their locations.

Do not automatically drop upper-layer force equations based on beta or the pressure-identifiability proxy. Small gas pressure does not make Lorentz imbalance harmless. If an approximate atmospheric product is desired, prefer an **explicit pressure prior with an uncertainty scale**, and/or an explicit force-model-error budget. That selects a reproducible estimate while admitting what the data do not determine. It changes the inference and needs a recorded scientific decision.

A force-free upper region is a possible alternative model, but it cannot simply be combined with arbitrary fixed-T column HSE and called full 3D MHS: exact force-free balance also requires horizontal pressure balance. A transition/interface and remaining force residual must be specified and reported.

### Q2. Is block Gauss–Newton on ln P appropriate? Should P or total pressure replace it?

**Block GN is a reasonable diagnostic; first fix its constraints and measure whether the conditional pressure solve actually converges.** Five GN steps and 100 PCG iterations are budgets, not a conditional optimality test. The current clipping/decrease acceptance is not a full trust-region method: it does not solve the constrained quadratic model or use the actual/predicted reduction ratio.

Distinguish three effects:

1. **Weighted conditioning.** Although `∂R/∂u` contains P, the minimized residual is `R/S₀`. Near an HSE initializer, with `H₀ = P₀/(ρ₀g)`, a typical derivative scale is `P/(h S₀) ≈ (P/P₀) H₀/h`, before the scale floor. Small absolute P or small beta alone therefore does not establish a weak conditional u Jacobian. Collapse relative to P₀ does weaken it.
2. **Conditional versus joint ambiguity.** Holding B fixed removes the compensating magnetic motion. The coupled pair `(δu, δB)` can be weakly constrained even when the u-only block is well behaved. D9b §1.2 itself found this distinction.
3. **λ_F scaling.** With `λ_U=λ_Uz=0` and B fixed, the pressure-dependent objective is just `λ_F f_F(u;B)`. Its minimizer is independent of every positive λ_F. The current damped GN equation also scales its gradient, curvature and diagonal damping together. Small λ_F does not by itself weaken this conditional solve; it changes the coupled tradeoff and some stopping scales.

For the next comparison, use **scaled physical pressure `p=P/P₀` with positive bounds**, keeping exactly the same EOS, geometric-mean density discretization, fixed scales and mean-log-pressure anchor. This removes the exponential collapse behavior as a coordinate effect. It does not create information or guarantee convexity: even with an ideal gas, the implemented vertical geometric mean introduces a `sqrt(P_k P_{k+1})` dependence.

Do not make total pressure the general 3D unknown at this stage. `∇(P+B²/2μ₀)` still leaves magnetic tension. Recovering a tiny gas pressure by subtracting magnetic pressure from a much larger total also demands accurate cancellation. Total pressure is useful for the special 1D/2.5D benchmarks, not a demonstrated cure here.

### Q3. Is discrepancy selection meaningful if χ² stays below threshold?

**It is currently an unbracketed penalty sweep, not a validated selection of λ_F.** `discrepancy_solve!` (`src/MHSStaggered.jl:858–863`) marks a stage eligible solely from its data statistic, even if it ends at `max_iter` or a failed line search. Keep those iterates for diagnostics, but distinguish them from a sufficiently solved stage.

Correct/calibrate the observation statistic first; require adequate constrained optimality; then report the tradeoff in **unweighted force residual, data misfit, outliers and Q distortion**. The existing weighted `terms[1]` is `λ_F f_F`, so compare `terms[1]/λ_F` across stages. Likewise account for both λ_F and λ_Q when comparing Q.

For the reported outer runs, the final raw force objectives are approximately **430** at λ_F=1e-4 and **466** at λ_F=1. Their similar residual profiles matter more than the apparent difference between `0.043` and `466` in the weighted term.

Do not merely keep raising the upper bound to force a threshold crossing. As λ_F grows, the solver increasingly minimizes the combined force-plus-Q penalty; its optimum can retain nonzero force residual and a data misfit below the uncertainty budget. That is a legitimate possible plateau, not a reason to manufacture a crossing.

Monotonicity is also conditional: even for global minima of `A(x)+λ C(x)`, it is the combined `A` that is nondecreasing and combined `C` that is nonincreasing. Here `A` includes data and other penalties; `C` includes force and Q. Data misfit alone has no general monotonicity theorem, and incomplete nonconvex solves add further uncertainty. Breaking at the first failed data stage is therefore an assumption, not an established equivalence to the full sweep.

An equilibrium residual should constrain scientific acceptance independently. Replacing the data discrepancy by a force-based rule cannot cure incompatible data or certify a static solution.

### Q4. Is Φ_Q the right remedy? Can a different discretization avoid it?

**Keep Q as a provisional stabilization, with a sensitivity study.** The C3 decay evidence supports its intended numerical role; it does not validate the inferred real pressure or establish a universal λ_Q. Testing λ_Q=1e-4 is justified by the reported synthetic gate, but do not call it validated for real data on that basis.

The 1% rule is a useful flag, not a bias theorem. A large Q fraction can arise because the physical residual becomes small; a small fraction can hide meaningful changes in weakly constrained pressure. Assess changes in B, P and force balance under nearby λ_Q values, and verify resolved current structures are retained. Q penalizes current variation even when a physical current is parallel to B; it is not literally a measurement of missing Lorentz force.

I disagree with two blanket statements in D9b §2.2. An asymmetric interpolation does **not** inherently lose an exact algebraic transpose or second-order accuracy. For samples at `[-h/2,h/2,3h/2]`, weights `[3/8,3/4,-1/8]` reproduce polynomials through degree two at the midpoint and have Nyquist response magnitude `0.5`. This is verified in the reproduction script.

The narrower statement about a reflection-symmetric midpoint averaging stencil having a Nyquist zero is sound. The example does not establish a replacement Lorentz discretization: biased reconstruction may introduce other artifacts, and the full divergence-free linearized force operator, boundary treatment and adjoint must be tested. A Maxwell-stress reformulation likewise needs its own symbol and conservation analysis; its name alone proves neither a cure nor an impossibility. Do not redesign M before the conditional-solver experiment.

### Q5. Is ≤1e-3 change per cycle a sound acceptance test? Restrict it to identifiable nodes?

**Keep the user's all-node test for this task. Report a split diagnostic in addition.** The existing `measured_change` is the maximum of relative pressure change and componentwise face-field change; its B denominator uses the current component magnitude or σ, not the vector-field magnitude. Record that convention explicitly. A cycle and 100 joint iterations are different amounts of work, so the chosen window and block budgets must accompany the result.

Small change is useful for reproducibility, but must accompany valid block termination and the independent equilibrium/data flags. As reported, maxima of **11.1σ and 13.3σ already fail data admissibility**, regardless of mean χ². Force RMS also remains far above the required tolerance. Even a successful masked change test would not rescue these runs.

`|B| σ_B/μ₀` is a magnetic-pressure uncertainty **proxy**, not a complete identifiability calculation and not proof that upper pressure is free by construction. It omits field geometry, along-field constraints, correlations, length scales and the anchor. A formal local pressure-information calculation would eliminate magnetic directions from the joint linearized system, using the data information and regularization as specified.

For reporting, fix the identifiability mask at the common reference state before comparing iterations, and report both endpoint masks as a sensitivity check. A mask selected from the current P can change as the solver moves, excluding the nodes under examination. Report max and quantiles of ΔP separately on the fixed identifiable set and its complement, B change, and bound-active fraction by layer. If the scientific product is later redefined to exclude weakly constrained pressures, that is an explicit user decision, not an optimizer fix.

## One next experiment: freeze B and solve the same pressure problem two ways

This tests the claim that the *u subproblem* is the obstruction, without changing the scientific objective or running another discrepancy sweep.

1. Take one coarsened Bifrost checkpoint after the first complete alternating cycle at **λ_F=1e-4, λ_Q=1e-3**, recording it before any comparison. Preserve S₀, the original collapse bound, T, the EOS and the mean-log anchor. If no checkpoint exists, recreate this one cycle with the brief's 300 magnetic iterations and 5 pressure steps.
2. Freeze B completely. From identical P, run (A) ln-P GN with a correct constrained step, and (B) bounded scaled-P GN. Use the same physical constraints and residuals; no pressure prior, extra smoothing, relaxed mask, or changed force weights. Allow up to **500 accepted pressure updates** per branch, PCG relative residual **1e-4** and up to **300 iterations**, logging failures rather than treating them as small changes.
3. At every accepted update record actual minimum bound margin, anchor error, raw force objective, PCG residual and constrained optimality. A useful scaled optimality diagnostic is `||δu*||∞`, where `δu*` minimizes `gᵀδu + 1/2 δuᵀDδu` over the actual pressure-step constraints and `D` is a positive diagonal scaling. Use the same physical-u scaling in both branches and remove λ_F from this diagnostic. Compute the fixed-mask layer change diagnostics described above.

**Conditional-solve pass:** no actual lower-bound violation or mean-log anchor error beyond **1e-10**; scaled constrained optimality **≤1e-6**; and maximum relative pressure change **≤1e-3 for three consecutive 100-accepted-update windows**. If a branch terminates earlier, confirm stability with three further 100-update continuation attempts whose lack of motion is justified by optimality, not failed steps. Confirm repeatability from ±1% admissible pressure perturbations, using the same fixed bound. These numerical thresholds are proposed diagnostic thresholds, not changes to §6.7.

**Interpretation:**

- Only scaled P passes: evidence for a coordinate/globalization problem; test that pressure block in alternation next.
- Both pass and their raw objectives agree within **1e-3 relative**: evidence against the conditional pressure solver being the main obstruction at that B. Compare their pressures too; different pressures at nearly equal objective expose conditional ambiguity or multiple minima.
- Neither passes: conditional convergence is unresolved. This does not establish physical infeasibility.
- A stationary conditional optimum with large force residual establishes a floor **at that frozen B**, not a floor over all data-admissible B. Investigate the coupled problem and model compatibility afterward.

The reported 40-cycle behavior remains unresolved until an experiment like this is performed. I did not rerun the 25-minute Bifrost integrations or inspect their per-cycle states.

## Reproduction and review scope

Read the brief, final spec §6/§10, implementation §13 and later notes, both D9 addenda, M6d step 1/2/3 and M5a reports, and the staggered objective/adjoint, GN, L-BFGS and alternation code and Bifrost scripts.

Added only this report and `HSEMHS/scripts/m5_bifrost/codex_review_checks.jl`. Run from `HSEMHS/`:

```sh
julia --project=. -t 2 scripts/m5_bifrost/codex_review_checks.jl
```

All reproduction assertions passed on one MPI rank. MPI initialization required execution outside the filesystem/network sandbox; no solver changes were made. These are intentionally small counterexamples and algebra checks, not a full regression-suite rerun or a real-data convergence result.

```text
Alternation clipping: minimum actual bound margin = -0.125000
L-BFGS projection: alpha cap = 0.250000, actual bound margin = -0.250000
u diagonal off anchor: relative difference = 0
u diagonal on anchor: relative difference = 3.495874e-02
Zero-work outer cycle: status=converged, change=0, objective=6.510455e-03
Interpolated independent unit noise: variance=0.50, neighbour covariance=0.25, rank=7/8
Asymmetric quadratic midpoint: Nyquist response=0.50, exact on degrees 0,1,2
```
