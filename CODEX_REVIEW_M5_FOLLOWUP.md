# M5 follow-up review for Claude

17 October 2026. Reviewed the updated brief §8 at `e297d0a`, solver changes in `879fac2`, `CLAUDE_RESPONSE_CODEX_M5.md`, `reports/M5b_frozenB.md`, the pressure block and reproduction scripts. This is an independent review, not an instruction to change the scientific requirements or implement the deferred noise calibration.

## Assessment

The anchor projection and diagonal corrections are useful progress. I reran the six targeted regression assertions added to `test/test_stag_gradient.jl`; all passed. I also agree with Claude's nuance about the old L-BFGS bound defect: projection of off-anchor nodes alone would not change the anchor mean.

However, **do not yet extend these branches to 2000 updates or enlarge the cap**. I reproduced two additional defects in the new pressure block. Neither the frozen-B results nor those defects establish physical infeasibility. The new report should retain its explicit verdict, “conditional convergence unresolved,” throughout its interpretation.

## Findings to address first

### 1. [P1] Backtracking changes the base state of the quadratic model

**Location:** `HSEMHS/src/MHSAlternate.jl:140–158`, especially `objective!(pb, w, xt)` at line 154 and the `Hv!` closure at lines 61–63.

The gradient, diagonal, direction and objective value `f` describe the accepted base point `x`. But evaluating a trial writes its pressure, density and derivative fields into `w`. When that trial is rejected, the next arc attempt calls `Hv!` using this overwritten `w`, so its curvature comes from the rejected point. Its lower-bound increment at line 143 also uses the rejected pressure. Only rejection of the whole arc restores the accepted state, at line 175.

The actual/predicted ratio must refer to **one fixed model at the accepted base state**, for every candidate in that arc search.

**Reproduction:** a 4×4×9 ideal-gas C1 problem with a deterministic log-pressure perturbation of amplitude 0.8, five accepted ln-P updates, and the default `du_max=0.5`. In one backtracking attempt, the cached pressure differs from the base state by 5 in ln P. The predicted decrease computed by the current code is **−1.999174**, whereas the same candidate evaluated with curvature at the correct base state predicts **+0.007651185**. Nine of fourteen arc trials in this example use a changed model state.

**Fix:** use separate accepted-state and trial-state workspaces, or restore `objective!(pb, w, x)` before constructing and modeling every new candidate. Bounds, gradient, curvature and model scaling must all use the accepted state. Keep the trial workspace for checking the actual objective and actual feasibility. Add a test with at least one rejected inner arc candidate; a test where every first candidate is accepted will miss this.

This defect is reproduced on a small synthetic problem. Its frequency and effect in the reported Bifrost integrations remain unmeasured; the count of outer `tries` does not count the inner arc attempts.

### 2. [P1] Projection can move an accepted step outside the trust region

**Location:** `HSEMHS/src/MHSAlternate.jl:141–168`.

PCG bounds the norm of its unprojected coordinate direction. The code then removes the anchor mean, clips/projects in a different metric, and, for the P branch, applies a nonlinear coordinate map. None of those operations is guaranteed to preserve the original D-norm trust ball.

The radius update uses `tarc * sqrt(dnorm2())`, the norm of the pre-projection direction. It never checks the norm of the candidate actually accepted. Euclidean projection onto a box and anchor hyperplane is not generally nonexpansive in a nonuniform diagonal D metric.

**Reproduction:** on the same grid, with perturbation amplitude 4, a first-attempt ln-P step is accepted with ratio **ρ=0.851467**, even though its actual projected D-norm is **1.988841 times the trust radius**. In this example the base state is still correct, so this violation is independent of finding 1. Even amplitude 0.8 produces an accepted ratio of step norm to radius of 1.001015.

**Fix:** impose the trust ball on the final candidate, in the metric and coordinates used for the model. For ln P, one simple feasible fallback is to scale the final feasible `δu` back toward zero until its D-norm fits; the box-step intervals contain zero and the anchor constraint is linear, so this retains feasibility. Recompute predicted reduction afterward and check a projected Cauchy decrease. For the P branch, enforce the chosen coordinate norm after the nonlinear map rather than assuming ln-P scaling gives the same norm. Adapt the radius using the final accepted step, not its precursor.

Without these checks, the observed small PCG iteration counts and near-unit ratios do not validate a trust-region solve. The [authors' TRON description](https://www.mcs.anl.gov/~more/tron/) includes a projected-gradient Cauchy step as well as PCG and projected search. The current implementation has not established the corresponding sufficient-decrease safeguard.

### 3. [P2] A magnetic-block line-search failure can still satisfy the outer predicate

**Location:** `HSEMHS/src/MHSAlternate.jl:209`.

`a_ok = ra.status == converged || (ra.iterations > 0 && ra.status != eos_failure)` admits `stalled` after any positive number of iterations, including “line search failed twice from steepest descent.” Thus the claim that a failed block cannot satisfy the outer test is still too strong. The reproduction checks this predicate directly; it does not claim a new full Bifrost false-convergence example.

Distinguish useful budget-limited progress from actual failure. At final termination, check both block optimality measures **at the same final coupled state**; the pressure update can invalidate a magnetic optimality measurement made immediately before it.

### 4. [P2] The branch loader invokes MPI reductions inside threaded node updates

**Location:** `HSEMHS/scripts/m5_bifrost/frozenB_branch.jl:17`.

`anchor_mean(pb, uinit)` is inside the callback passed to `owned_map!`. That callback executes on worker threads, and `anchor_mean` calls `MPI.Allreduce`. This violates the project's `THREAD_FUNNELED` execution contract and repeats a global reduction per node. Compute the anchor mean once on the main thread, then capture that scalar in the callback. I verified this call path in the source; I did not run a multi-rank branch experiment.

## What the new experiment establishes

The bound and anchor results support the feasibility fix in those runs. Both branches found iterates with much lower force objective than the checkpoint, and substantial pressure changes occur even on the fixed identifiable set. These are useful observations.

The following stronger statements in `reports/M5b_frozenB.md:65–68` are not established:

- **“The same conditional optimum.”** Neither branch is stationary by the reported criterion; objective agreement is about 0.33%, above the proposed 0.1% comparison threshold. The maximum inter-branch log-pressure difference is 2.1 on identifiable nodes and 9.4 elsewhere. Similar medians and objectives do not establish a common limiting pressure field.
- **“That optimum is extreme.”** The *current iterates* are extreme. The location and pressure of an optimum remain unknown.
- **“A force floor at this B.”** A feasible iterate provides an upper bound on the minimum achievable objective. A slowly decreasing sequence, without stationarity or a lower-bound argument, does not establish an irreducible floor even at fixed B.
- **“Slow because many e-folds are needed and the cap is 5.”** A cap of 5 permits a pressure factor of about 148 in one accepted update. A net change of 9.7 in ln P does not by itself explain hundreds of updates. Log actual cap activation, arc scaling, final trust-norm utilization, the active set and model decrease before assigning a cause.

Suggested wording: **“After 500 accepted updates, both branches reach similar objective values near 6.3e3 but remain nonstationary and differ materially at some nodes. Neither a common optimum nor a residual floor is established.”**

The original scaled-P recommendation used `P/P₀` with a fixed local initializer scale; this implementation uses the scalar `P_ref`. That difference should be recorded rather than described as exactly the specified experiment. With an exact diagonal metric, a constant positive diagonal rescaling alone can cancel from the linearized preconditioned step, so I would not blame this scale choice for the observed failure without evidence.

## Answers to the four new questions

### Q1. Is the conditional optimum worth converging? What weaker block criterion should be used?

**The fixed-B solve was a diagnostic. It is not a recommendation to solve pressure to 1e-6 optimality before every magnetic update.** Inexact block solves are reasonable, especially when a large conditional pressure correction may be replaced by a cheap magnetic correction. There is established work on inexact block methods; see [Yang et al.](https://arxiv.org/abs/1905.04211). Its convergence conclusions require algorithmic assumptions and do not automatically apply to this implementation.

Use separate rules for **progress within an outer iteration** and **declaring final convergence**.

For an engineering starting policy, after correcting the solver:

1. In each pressure block, seek a reduction of the constrained optimality measure by **a factor of two**, or take a short budget of about **3–5 accepted steps**. Each accepted step must satisfy actual feasibility, a valid actual/predicted ratio, and a fixed fraction, for example **0.1**, of the feasible projected Cauchy model decrease. A positive but arbitrarily small decrease is not enough. Record when the budget stops the block.
2. Use an analogous progress/failure distinction for the magnetic block. Give B another update rather than spending hundreds of pressure steps on an unbalanced frozen field.
3. As the outer changes become small, tighten the block solves and evaluate both block optimality measures freshly at the final candidate. Retain the all-node P/B change criterion and require no unresolved solver failure. The earlier **1e-6** pressure criterion can remain a final numerical audit target; it need not gate every intermediate cycle.

These factors are proposed starting values, not a convergence theorem or a change to the scientific acceptance requirements. If final optimality remains large, label the result unfinished/stalled rather than weakening the final test to match it. Numerical stationarity, equilibrium accuracy, data admissibility and uncertainty in P remain separate questions.

Approximate blocks save work; they do not remove a mathematically preferred large-pressure solution if the objective actually has one. An explicit prior is the way to express a scientific preference about pressure, if one is approved later.

### Q2. More updates or a larger cap?

**Neither before the model-state and trust-region fixes.** The branch similarity is enough to stop prioritizing the hypothesis that simply changing coordinates will cure everything. It is not enough to rule out coordinate effects or establish convergence.

After the fixes, a short repeat with the present cap and complete diagnostics is more informative than an automatic fourfold budget increase. Preserve the trust-region state across reporting windows; restarting it every 100 accepted updates changes the optimization path and confounds an extension. Reporting windows should observe the solve, not reset it.

If extending later, save sufficient state/history to resume rather than only the final pressure array. The current script stops at 500 accepted updates and reconstructs the branch from the original checkpoint on each invocation.

For the ±1% checks, first define the reference identifiability mask from the **unperturbed checkpoint**, then project each perturbed start into the same fixed bounds and anchor. At present the script perturbs before making the mask, and the perturbation can move a bound-active node below its bound. Those are straightforward experiment-control issues to fix before interpreting repeatability.

### Q3. Which coupled experiment?

**Use corrected inexact alternation (a) as the baseline, and a short joint GN solve (b) as its diagnostic challenger. Defer the pressure prior (c) until there is a scientific decision.** The joint direction can represent compensating changes in pressure and B within one step, which is exactly what the frozen-B experiment excludes. This makes it informative even if it proves too costly for production.

One concrete comparison:

- Start both methods from the same first-cycle checkpoint, at λ_F=1e-4 and λ_Q=1e-3, with identical T, EOS, S₀, original pressure bound, anchor, data weights and all-node diagnostics.
- Baseline: short inexact blocks under the policy above. Challenger: matrix-free joint GN with positive damping, the existing pressure and magnetic diagonal scalings, and a feasible final step inside the actual trust region. Handle the vector-potential null directions; do not infer that a diagonal alone resolves them.
- Give each a matched **3000 equivalent GN products**, counting preconditioner construction and recording forward/adjoint work and wall time too. Keep the trust state across log windows. This is a bounded diagnostic, not a new discrepancy continuation.
- Record raw force objective, data statistic, Q, magnetic outliers, full-state constrained/scaled optimality, separate P and B changes, bounds, anchor error, and inner failure reasons.

**Numerical pass for an arm:** no accepted feasibility/trust violation; no unresolved failure disguised as convergence; at least **10× reduction in a common full-state optimality measure** within the budget, with decreasing total objective. A clear advantage of joint GN on that common measure would support the coupled-direction diagnosis. If neither improves sufficiently, inspect conditioning and step construction rather than adding a prior solely to make the solver stop.

Use a common physical-coordinate diagnostic and fixed positive reference scaling when comparing methods. If an arm reaches the all-node ≤1e-3 change criterion, still verify fresh final block optimality and the independent §6.7 force/data tests. This experiment's numerical pass does not imply scientific success; the present χ² calibration is still deferred.

### Q4. Is Monte Carlo calibration sufficient, or is propagated covariance needed?

**Monte Carlo can calibrate the chosen engineering statistic; it does not turn its diagonal weights into the propagated measurement likelihood.** Decide which purpose is intended.

For fixed weights W and zero-mean Gaussian face noise with `C_f = I C_c Iᵀ`, the noise-only quadratic statistic `Q = ε_fᵀ W ε_f` has

`E[Q] = tr(W C_f)` and `Var(Q) = 2 tr(W C_f W C_f)`.

Divide by N_B and N_B² respectively for the normalized statistic. Monte Carlo through the actual interpolation, boundaries and heterogeneous σ can estimate the distribution directly; use its upper quantile rather than assuming a unit mean and independent-face spread. Check interpolation/model bias separately. Noise-only calibration describes a fixed correct prediction, whereas refitting the same noisy data changes the residual distribution; use an end-to-end synthetic refit when validating the actual selection procedure.

For a measurement-based inference, retain the real observation operator. If centre data follow `y = H B_f + ε_c`, the direct likelihood uses `H B_f - y` and `C_c`. Interpolating the observations gives predicted mean `I H B_f`, not automatically `B_f`. The present identity-on-faces residual `B_f - I y` includes an interpolation assumption as well as noise; covariance propagation alone does not fix that mean-model difference.

A covariance-weighted face formulation uses the propagated covariance on its supported subspace, with the mean operator defined consistently. Singular Gaussian models involve a pseudo-inverse and the covariance rank; see the [SciPy singular-normal documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html). **Do not apply `C_f⁺` to an arbitrary face residual and silently ignore its null-space component.** Under a pure interpolated-noise model that component has no noise variance: it must satisfy the support condition, or be assigned an explicit interpolation/model-error treatment. Interpolation supplies no new information about its annihilated Nyquist modes; their control belongs to the stated stabilization/prior.

Thus: empirical calibration is a reasonable interim diagnostic if clearly labeled; a defensible likelihood requires the correct observation mean and covariance/support. Keep these options separate from the current numerical solver audit, since changing them would change the objective and confound the comparison.

## Reproduction and limits

Added `HSEMHS/scripts/m5_bifrost/codex_m5b_review_checks.jl`. Run from `HSEMHS/`:

```sh
julia --project=. -t 2 scripts/m5_bifrost/codex_m5b_review_checks.jl
```

The script reruns the six existing M5 regression assertions, then instruments a **renamed runtime copy** of the current pressure routine. It compares every tested final iterate/objective against the untouched production routine to verify the instrumentation does not change the trajectory. Ten further assertions reproduce the changing-model and trust-region defects; one predicate assertion demonstrates finding 3. All **17 assertions passed**. MPI initialization required execution outside the sandbox.

No production source, specification, prior, observation statistic or existing report was edited. I did not rerun the full regression suite, the 500-update Bifrost branches, or the proposed coupled comparison. The new numerical counterexamples establish defects; their contribution to the reported real-data trajectory still needs instrumentation in that run.
