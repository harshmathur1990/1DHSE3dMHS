# Round 6 review and completion plan — 2026-10-29

Reviewed HEAD `ce485e0`, the round-6 brief, changes since `da96fe7`, solver/roundoff changes, `ml_lab.jl`, embedding options and existing view tests. No tests, scientific jobs or cluster access performed. Numerical results are Claude's reported measurements. Scientific source is unchanged. Recommendations remain for the user's decision.

## Decision: switch from exploratory development to delivery

There has been real progress: angle geometry and Jacobians have verification, the controls recover stable solutions, numerical failures are reported more honestly, and the branch-aware coarse space addresses an identified stencil weakness. However, the full input is still not qualified within the default budget, and the multilevel prototype has no demonstrated end-to-end gain. Neither is ready to be called the finished production solution.

**Next: measure the existing solver's practical scaling, perform one bounded inner-accuracy experiment, select one implementation, and integrate/validate it through the production-facing interfaces.** Do not start another broad sweep of coarse shapes, scientific objectives, boundaries or snapshot interpretation. Column HSE must be included in delivery, not left implicit while MHS research continues.

The latest brief records a revised target of a few minutes on 8 nodes/256 CPUs, with only 4 nodes available for testing. Treat that as the reported target; document exact cores/ranks/threads per node rather than assuming an allocation from the phrase “256 CPUs.” Measure up to available hardware and label any 8-node estimate as an estimate. Do not claim the deployment target met without a measured run there.

## What to do next — a bounded sequence

### 1. Lock a baseline and completion matrix

Choose a single current configuration for the existing solver, with all damping, forcing, cut, CG limits and qualification options recorded. Use the actual CLI and embedding paths, not just the standalone lab. Keep equations, Φ, boundaries and the configured gn_tol/count/25-step budget unchanged; diagnostic reference runs may explicitly override budgets.

Use a small matrix covering μ=1, 0.3 and 0.2; one non-axis-aligned azimuth at small μ; column HSE and MHS; cold and one representative warm start. Reuse existing controls and inputs. Do not generate another family of equilibria.

Record valid output, stopping reason, decrement bounds when available, true residual, abnormal exits, steps, A products, setup/solve/qualification time, peak memory and exact resources. Heights are diagnostic only. A valid max_steps atmosphere is not a qualified solution. This table is the completion dashboard and should replace repeated open-ended briefs.

### 2. Measure scaling before committing to a larger algorithm

First compare one identical frozen full-size system and a short fixed outer-step sequence on 1, 2 and 4 nodes. Use the same state/RHS, tolerances and total work; include MPI reductions, FFT communication, setup and memory. Then run the chosen end-to-end configuration on the largest available allocation. This separates hardware scaling from different nonlinear trajectories.

Strong scaling cannot cure convergence by itself. A 25-step max_steps run that becomes faster is a faster budget-limited run, not success. Conversely, the pessimistic current single-node timings do not establish failure on the intended deployment.

Use this measured baseline as the fallback. Do not implement an untested 8-node communication design solely from extrapolation.

### 3. One controlled experiment: does better inner accuracy reduce total nonlinear work?

Yes, CG residuals of 0.3–0.7 plausibly cause extra outer steps, poor directions and misleading step predictions. That is not yet established as the dominant cause, and a flat snapshot objective is not a reason to dismiss faster linear algebra.

From one archived representative μ0.3 state, run 5–10 outer steps with identical objective/start and:

- the baseline inner policy;
- a reliably attained tighter inner tolerance using block solves, with a diagnostic work budget;
- the existing branch-aware additive candidate, if it attains the same tighter tolerance more cheaply.

Track actual/model decrease, cap/line-search scales, qualification lower bound, total products and **total elapsed time**. Compare the returned fields to each other only at matched numerical convergence. If tighter inner solves improve progress per second enough to matter, pursue the cheap candidate. If they improve direction accuracy but cost more overall, keep the baseline/forcing policy and document the result. If neither makes meaningful progress, use this evidence to propose one outer-step adjustment; do not launch arbitrary new preconditioners.

This is the decision gate. Report the selected method and why; do not extend the experiment indefinitely.

### 4. At most one final preconditioner implementation

The current symmetric multiplicative cycle requires two additional A applications inside M⁻¹, explaining why lower iteration counts do not yield a μ0.3 time gain. The **existing additive branch-aware correction** is the lowest-effort final comparison, not a new research project. Compare setup plus solve, with identical accuracy and gauge treatment.

If that wins, integrate it as an opt-in candidate and address the coarse size with a compact basis that explicitly retains mandatory gauge/branch height modes, plus only the pressure shapes demonstrated necessary. Do not retain local eigenvectors alone when they are already known to lose required modes. A third level is justified only if the selected candidate wins and a concrete memory/communication estimate shows a compact coarse problem is insufficient. Recursive branch modulation is not automatically correct: Nyquist branches are tied to the fine stencil; coarse transfers must preserve the represented space and Galerkin consistency.

A serial coarse solve at 200k–800k unknowns is not a full-size solution. A delivery-ready design must specify bounded coarse memory, factorisation/rebuild cost and MPI ownership. Start with distributed restriction/prolongation and an explicitly small coarse problem; gathering a small coarse RHS to a designated rank can be acceptable if measured. Do not gather a huge coarse matrix to one rank by default.

If neither additive nor multiplicative has a practical time win, leave the prototype experimental and finish delivery with the best measured existing solver. Report any unmet time/convergence requirement explicitly rather than hiding it in another lab.

### 5. Integrate and close

Expose only the selected supported solver controls through `PoissonTauOptions`/configuration, CLI and embedding API, with collective validation and consistent defaults. The embedding wrapper currently does not expose the full laboratory preconditioner/damping/forcing controls, so laboratory wins do not yet improve the caller's solver. Ensure qualification, roundoff/domain/breakdown outcomes and work statistics survive through public status/output.

For column HSE, verify the existing path independently: μ1 equivalence, plane-parallel optical-depth shift, inclined-ray interpretation, finite positive EOS fields, monotone z/τ, MPI agreement and a documented failure outcome. Column HSE balances the one-dimensional column/ray relation; it does not claim full horizontal MHS force balance. No new solver algorithm is needed unless these checks fail.

Run the existing meaningful regression/view/MPI suites plus the small delivery matrix through both interfaces. Validate one cold/warm embedding sequence with the selected options. Update README/run examples, supported geometry and input conventions, resource settings, stop semantics, measured limitations and a final result table. Prepare a concrete reviewable merge/release candidate; leave the user's final merge/deployment decision to the user.

## Code findings to resolve within that plan

### Smoother stability is not guaranteed by the current power estimate

`lmax_block` uses a finite power iteration on B⁻¹A and `ω=1/estimate`. This is an estimate, not a proven upper bound. B⁻¹A is generally not Euclidean symmetric; the reported norm ratio need not bound its spectral radius. For a symmetric Richardson-based cycle, an overly large ω can compromise positive definiteness despite the small dense tests passing.

Use a conservative justified spectral bound, or a safeguarded estimate and positivity checks with a clear fallback, before PCG production integration. Do not describe random-vector positivity or an 8² test as a proof for all grids. This is an integration blocker for the new cycle, not for the existing solver.

### The roundoff floor is a heuristic with gauge/grid dependence

`phi_floor` uses absolute local z, u and inverse grid spacing. Uniformly translating the height gauge changes the floor without changing the mathematical residual. It also omits some cancellation inside W/Lorentz construction; K=64 is not a rigorous global error bound. Sharing Φ's weights does not make it grid independent. A floor 1e5 larger than one observed exact-start Φ does not by itself establish that K is too large or acceptable.

Keep the exact-start fix, but label it numerical precision acceptance rather than a stationarity theorem. Add a bounded regression with gauge-translated equivalent inputs, a genuinely nonstationary small perturbation, and finite Φ/floor checks. Prefer a canonical gauge and actual operand-magnitude estimates if the current heuristic makes acceptance depend materially on translation. Do not spend another campaign tuning K or use the floor to accept ordinary stalled MHS runs.

### PCG finiteness via signed sum needs care

The new `[dot(r,r),sum(x)]` reduction detects nonfinite entries in ordinary cases but can overflow for a large finite vector through accumulation. It is not equivalent to a robust finite-entry predicate. Combine a locally accumulated finite-entry indicator with the existing collective instead, or otherwise document the conservative false-breakdown behaviour. Retain collective agreement across ranks. Address this as a small robustness fix, not an excuse to rework all CG reductions.

### Tiled tests are useful but not a pure unchanged-operator experiment

`ml_lab make` tiles the input and reruns a trajectory on the larger domain. `ereg` also depends on κ²=(2π/L)², so domain growth changes the step penalty. The tests support promising scalability but do not isolate only propagation across identical frozen local operators. State that caveat; preserve exact coefficients/penalty when the completion comparison requires an identical tiled operator. No repeat of the entire study is needed to select the next bounded trial.

## Answers to the remaining research questions

- **Slow modes:** do not add per-level shapes, slant-line blocks and multiple new smoothers at once. Try the already implemented cheaper additive cycle first. Enrich one compact basis only if measured residual components justify it.
- **First-order defect versus second-order solution:** possible because the linear inverse can suppress particular residual components, and mapped/interpolated truth is not an exact discrete state. The observation is not a proof of solution neutrality. Record it as a numerical limitation; investigate further only if it breaks refinement, field recovery or the delivery tests.
- **Four-node versus eight-node time:** measure four-node scaling now; do not promise linear extrapolation to eight nodes. Communication and the coarse bottleneck can dominate.
- **Column HSE:** give it its own explicit delivery result. Working MHS tests are not sufficient evidence for its public angle/API semantics, and column-HSE success does not certify MHS convergence.

## Definition of done

1. Both column HSE and MHS run through documented CLI and embedding interfaces at the supported μ/azimuths, with correct input/output conventions and reliable global status.
2. Existing correctness/MPI tests and the narrow delivery matrix pass. No abnormal PCG/domain failure is silently called converged.
3. The selected solver is measured by end-to-end time and memory at a stated numerical convergence standard on available hardware. Budget-limited cases are identified explicitly.
4. A final table states whether the requested production runtime and convergence targets are met, partly met or unmeasured. The 8-node target is not declared achieved from an estimate.
5. The chosen code, documentation and results form a reviewable release candidate. Unselected experimental methods remain opt-in or outside the supported path.

**Next brief should be a delivery decision and result table, not another open menu of research directions.** Keep the scientific problem fixed and bring one supported implementation to a clear, testable finish.
