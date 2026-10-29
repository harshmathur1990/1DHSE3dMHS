# Round 7 review: selected solver and final delivery gate

Reviewed HEAD `a60d959`, round-7 brief, changes since `ce485e0`, switch implementation, public options/configuration, delivery scripts and new tests. No tests, scientific runs or cluster access performed. Timings and test results are Claude's reported measurements. Scientific source unchanged; recommendations remain for the user's decision.

## Decision

**Keep `switch` as the selected opt-in acceleration and close out the multilevel research.** It is a useful measured improvement: 1.9–4× faster for the same step budget on the tested cubes, with valid output. It has not demonstrated convergence of those cubes, so it is not yet evidence that the full feature request is complete.

Do not change the default or scientifically reinterpret the objective to finish the project. Finish API/MPI/regression/docs work now, and allow **one bounded outer-step experiment** to address the remaining convergence bottleneck. After that, deliver a concrete candidate with an explicit met/unmet requirements table. No further open-ended algorithm menu.

## What is ready, and what is not

- Column HSE has its own angle checks and full-size valid CLI results. It is ready for final MPI/interface verification and documentation; it does not need further algorithm research.
- Angle MHS has correctness coverage, useful controls and a faster budget-limited implementation. The production-sized Bifrost cases all still end `max_iter`.
- Eight-node runtime is unmeasured. The 6.5–11.4 min single-node figures are genuine progress but cannot be called “time independent of μ”: they are roughly comparable **fixed-budget times**, spanning about 1.75×. Qualified convergence time remains unknown.
- The reported embedding sequence used Fourier. The selected `switch` method still needs the same cold/warm/cold sequence through the actual embedding path.

## Findings and necessary corrections

### 1. Fixed-budget improvement does not justify a default change yet

`switch` changes preconditioning only between PCG solves, so that choice is algebraically legitimate. Its trajectory depends on the iteration budget and tolerance because the switch occurs when the accepted step's final PCG exits `:maxit`. This is an empirical phase-selection policy, not a convergence theorem or a guaranteed identification of the nonlinear tail. The small test verifies switching/no-switching, not every production regime.

Keep it opt-in for the candidate. Verify the switch point/status is consistent across ranks and repeat the control/mixed-azimuth test through CLI and embedding. Document that warm calls restart the phase policy and can immediately switch on a capped Fourier solve. No need to add more phase-selection heuristics now.

### 2. Qualification must be distinguished from the approximate final decrement

An 80-iteration final solve that misses tolerance does not produce a comparable accurate decrement across preconditioners. Do not infer that block is “more accurate” solely because its decrement is larger. The true residual and, preferably, the existing regularised-decrement interval provide the evidence.

For final acceptance diagnostics, use a consistent block-based qualification policy where block support exists, on the same H+ereg system, with bounded work and the existing interval outcome. This changes the numerical method used to evaluate the metric, not the metric itself. Retain a documented Fourier fallback for unsupported small grids. Preserve the outer preconditioner separately and restore it if a rejected qualification is followed by more outer steps.

Do not require a costly fully qualified diagnostic every step. A lower bound already above gn_tol proves the threshold is not reached. An inconclusive interval remains inconclusive. Label the ordinary final `gn_decrement` approximate whenever the solve is capped.

### 3. Warm/cold disagreement is not evidence against the warm-start implementation

The public embedding path still derives weights from each starting atmosphere. Thus warm and cold runs on the perturbed cube can minimise different weighted objectives, in addition to both being budget limited. The 6.1 km field difference cannot isolate a warm-start defect or establish a common numerical endpoint.

For release, test warm validity/status/performance through the selected solver. If claiming equivalent endpoints, run the diagnostic comparison with common weights and matched numerical convergence. Do not introduce a new public weighting default solely for this check, and do not promise warm-start speedup without evidence.

### 4. Do not erase the failing tests by regenerating everything

Five embedding failures remain a release blocker even if older commits also fail. Classify each before updating fixtures:

- If the stored reference is obsolete, compare current standalone and embedding paths at identical settings, input and stopping outcome; update the reference only after this parity is established.
- If allocations exceed budgets, profile the increase and distinguish expected added work from an avoidable regression. An allocation budget is not a golden field reference. Do not raise it simply to make the test green.

Record each changed reference/budget with a specific explanation. The current Fourier default can be preserved while fixing these tests; this work need not wait for a default-preconditioner decision.

## The one remaining numerical experiment

The tail results support keeping loose inner solves: tighter block CG is much more expensive and barely improves ten-step progress. They do **not** prove that inexact inner solves never limit the tail in other states. Stop that investigation here and focus on the observed whole-step cap.

First, at one saved late μ0.3 state, compare the actual objective/gain at the current capped step, twice that step, and the full direction wherever EOS-feasible. This requires only a few residual evaluations and determines whether merely relaxing the 300 km cap has an obvious benefit. Do not change the production cap from this single probe.

If whole-direction radial scaling is demonstrably wasteful, **one opt-in scaled trust-region GN step is a justified final algorithm trial**. Steihaug–Toint truncated CG is a reasonable implementation: terminate at the trust boundary rather than solve a large direction and shrink every component. Define a dimensionless pressure/height norm, use the appropriate preconditioned trust-region formulation, and retain EOS feasibility safeguards. Use physical model decrease from H, actual decrease and gain-based radius updates. Do not reinterpret a small trust-limited step as stationarity.

Keep the equations, Φ, boundaries and qualification threshold fixed. Compare baseline switch and the new step from the same checkpoint and then the same cold start. Bound this trial to one representative μ0.3 case and a μ1 regression before expanding to the delivery matrix. Pass only if total time to a common numerical stationarity standard improves, with no abnormal exits or material regression. Merely lowering Φ faster for another 25 steps is encouraging but does not fulfil the convergence requirement.

If the initial cap probes show no useful direction, or the trust-region trial does not improve the outcome, stop expanding scope. Deliver the faster valid budget-limited candidate as such, with **qualified full-cube convergence explicitly outstanding**. That is an honest partial delivery, not a claim of completion. Any further solver redesign should be a separately scoped task with the user's decision.

## Final work package for Claude

1. Keep Fourier default, `switch` opt-in. Freeze selected settings and run the switch embedding sequence.
2. Complete the single cap/trust-region decision above; do not restart multilevel development or snapshot-accuracy work.
3. Make final diagnostics comparable and preserve honest unresolved/budget outcomes.
4. Resolve the five embedding failures individually; rerun existing MPI suites on 2–4 ranks with final options, including switch and column HSE.
5. Write README/run examples and a release result table: supported μ/azimuth/input conventions, units, MPI/thread layouts, memory, exact configuration, qualification policy, cold/warm results and known limitations. Distinguish configured defaults from the D2 benchmark settings.
6. Prepare one reviewable release candidate. State separately: column-HSE delivery, MHS validity/correctness coverage, fixed-budget speedup, qualified convergence, and measured deployment runtime. Do not mark the whole request complete unless its convergence/runtime acceptance criteria are actually demonstrated or the user explicitly accepts reduced scope.

Profiling Fourier communication is optional only if needed to explain a measured deployment bottleneck; it is not a new delivery prerequisite. Mixed-hardware fixed-work scaling does not justify a numerical eight-node promise. Record available measurements and defer that claim until the target allocation is available.

**Next brief should contain the final selected configuration, resolved test failures, the bounded outer-step result and the release candidate—not another collection of candidate algorithms.**
