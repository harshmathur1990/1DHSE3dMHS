# Solver-time review, round 5 — 2026-10-29

Reviewed HEAD `da96fe7`, round-5 brief, changes since `5d4f353`, PCG/feasibility code, boundary/start construction, control refinement and summary scripts. No tests, scientific jobs or cluster access performed. Results/test counts are Claude's reported evidence, not independently reproduced. No scientific source edited.

The user's latest direction governs this review: **implement the correct solver faster and more efficiently, with the equations, objective and boundary conditions held fixed**. Bifrost height disagreement is a separate scientific diagnostic, not a reason to reject a converged numerical solution. Recommendations remain for the user's decision.

## Recommendation

Proceed with **step 5(b): a modest horizontal multilevel prototype using column-block smoothing**, beginning with frozen systems. Fix the exact-start/no-descent classification in parallel. Do not change Φ, free the cut pressure, move the cut, or make snapshot force diagnosis a prerequisite for the solver work.

The control provides encouraging evidence of correct recovery and refinement for the tested geometry. It is sufficient to begin performance work; a rotated case and perturbation-return tests should accompany the prototype as coverage, not delay it with another broad science-validation campaign. The original full-size time target remains unproven.

## Findings affecting interpretation and validation

### 1. High: the reported defect norm is not resolution invariant

`control_summary.py` calls `sqrt(phi(s0)/Wtot)` the truth's discrete defect, where `Wtot=(2sum(wc)+sum(wf))*NX*NY`. The weights are not commensurate residual counts: wc scales like H² Δz, while wf scales like H²/Δz because Rz is an integrated face residual. When vertical spacing h decreases, sum(wf) typically grows like h⁻², while a consistent volume-weighted force objective need not. Thus the reported RMS can fall like h even for a fixed physical residual. It cannot establish that the actual consistency defect is first order.

**Action:** report horizontal and vertical defects separately in explicitly defined physical/nondimensional units, with Rz divided by its face thickness and a common volume normalisation, or use independent Cartesian force residuals. Hold the normalising scales fixed across refinements. Then inspect cut-adjacent, interior and bottom layers separately. Only after that should one-sided end derivatives be blamed. They are a plausible contributor: `dzcol!` uses endpoint secants and an interior secant on a generally nonuniform height grid, whose local order also needs checking.

The same normalisation issue means a nonzero default `rms_atol` should not be selected from the current RMS values as if it were a grid-independent physical tolerance.

### 2. Medium: a uniform height offset is a gauge, not an identifiable boundary error

This formulation is invariant under a uniform shift of all heights, and the embedding path explicitly fixes the mean z(τ=1). The measured −51/−120 km uniform offsets therefore cannot, by themselves, establish physical boundary bias. The gauge-aligned shape and pressure differences can establish boundary sensitivity; the pressure mismatch near the cut is a separate observation.

**Action:** qualify the brief's “largest systematic error” claim. Compare height shapes after the same gauge, pressure profiles, objective/gradient and cost. For this request, optimise the production boundary problem as specified. Changing cut pressure or cut location is a separate scientific proposal and should not be bundled into solver acceleration.

### 3. Medium: control results support correctness in this case, not identification of Bifrost dynamics

Agreement across starts and refinement supports the control implementation. It does not prove the Bifrost trajectory is close to a stationary discrete equilibrium: capped late CG and unresolved qualification still limit that inference. Low correlation between two error maps rules out a simple map-equality explanation, not every indirect or nonlinear influence.

Measuring the snapshot Cartesian residual is reasonable optional science work, but a residual alone cannot distinguish dynamics from mapping, EOS and discretisation effects. Do not divert the next solver sprint into explaining the 22.4 km number. Use identical frozen Bifrost systems as realistic conditioning benchmarks without requiring their endpoints to match snapshot heights.

### 4. Medium: new safety checks add communication in the PCG hot path

`pcg!` now forms trial x/r safely and calls `gfin(zv)` via a global maximum on every iteration, in addition to residual and curvature reductions. This protects the last good iterate, but a full-array scan and extra MPI collective can be material on the target grid. The flag is optional; audit callers so breakdown is not silently interpreted as an ordinary tolerance miss.

**Action:** profile before optimising the checks. Where practical accumulate local finiteness information during existing update kernels and combine compatible reduction data, retaining collective agreement across ranks and the last-good-iterate guarantee. Do not remove safety checks merely to improve a timing. Include this cost in the baseline for the prototype.

### 5. Low: refinement and restart diagnostics need precise labels

`control_refine.py` compares magnetic magnitude, not vector components; a directional interpolation error can be hidden. Compare each B component and validate grid/coordinate metadata. `control_summary.py` finds when scalar RMS error reaches its final magnitude; equal scalar errors do not imply equal fields. Call that a retrospective scalar-error diagnostic, not numerical convergence.

Restart metadata uses file size and sum(logτ), not a content checksum of T/B/coordinates. Correct that claim in the brief and add a stable content identity if strict restart verification is required. These are diagnostics fixes, not reasons to postpone frozen-system optimisation.

## Concrete implementation sequence for Claude

### A. Baseline and exact-start handling

Archive a small set of frozen states, exact weights/ereg, inputs, cut, geometry and RHS: μ1 and μ0.3 control states, and representative early/late Bifrost states. Include μ0.2 when available. Test the same A and b with all candidate preconditioners. Keep damping values fixed for each linear comparison and report them.

Before a no-descent exit, distinguish roundoff-scale stationarity from a failed direction or inaccurate inner solve. Prefer an explicitly documented numerical-roundoff check at the starting/returned state, using residual/gradient and meaningful scales; retain the configured decrement threshold and 25-step budget. Do not accept convergence merely because Φ fails to decrease or CG returns a zero direction. A nonzero default absolute tolerance needs a scale study and separate review. Add exact-start, small-perturbation, and nonstationary-no-descent regressions.

### B. Small SPD multilevel prototype

Prototype on crops and controls before the full cube:

1. Use whole columns as blocks; coarsen only horizontal coordinates initially. Reuse the analytic column blocks as the smoother, but choose a damped/symmetric smoothing scheme whose stability and positive definiteness are checked. Exact block inversion by itself is not proof that an arbitrary multigrid sweep is SPD.
2. Build Galerkin coarse operators from the **actual current A**, including ereg and any LM term. Start with one or two levels and a direct coarse solve on small problems. Keep the operator/preconditioner fixed during each PCG solve; rebuild between outer iterations as needed.
3. Preserve a compact set of coupled pressure/height column shapes. Do not assume c alone is enough or form dense full-cube coarse bases. Compare a compact aggregation/column-shape basis with a simple geometric baseline.
4. Treat constant/checkerboard gauge modes consistently. Centred horizontal derivatives have Nyquist branches; standard low-frequency coarsening may miss slow error near those branches. Measure error after smoothing and the transfer/coarse-space representation of those modes before selecting restriction/prolongation. Project only declared gauges, not physical large-scale structure.
5. Use symmetric transfers/cycles compatible with PCG, and verify numerical symmetry/positivity against explicit small matrices. Compare the correction with a direct solution; test true residuals and direction/model-decrease accuracy. A nonsymmetric or varying cycle requires an appropriate Krylov method rather than assuming PCG remains valid.

Keep the previously corrected hybrid as a baseline. Do not rerun it as a newly proposed algorithm. A fixed-work coarse prototype can be useful even if the final implementation later needs a hierarchy or a flexible outer linear solver.

### C. Pass criteria

The proposed “μ0.3 within 2× μ1 iterations” is a useful secondary target, not the primary pass criterion. A multilevel iteration may cost much more than a block iteration, and worsening μ1 would trivially improve the ratio.

For each identical frozen system, compare:

- Time to true relative residual 1e-3 and 1e-5, total A/preconditioner work, setup, memory and reductions; independently recompute the final residual.
- Agreement of δ and physical model decrease with a well-resolved reference where attainable. Label unresolved reference systems honestly.
- Scaling across fixed-domain resolution and fixed-resolution domain-growth tests separately, at prescribed comparable states.

Advance only if total time improves at the same accuracy on representative difficult cases, with no material μ1 regression. Then compare outer trajectories with identical starts, weights, stopping policy and boundaries. Log qualification work and time separately and include it in end-to-end time. Compare converged fields with one another after the same gauge and against a tight solve of the **same numerical problem**; do not use Bifrost truth proximity as the acceptance test.

Finally measure the requested 504²×95 workload, fixed node/rank/thread/BLAS settings, several azimuths and μ1/0.5/0.3/0.2. Include cold setup and warm-start cases. Report failures and budget stops; do not infer full-size performance from the y-invariant control's 15–34 s timings.

### D. Narrow accompanying correctness checks

Add one rotated control with nonzero tx and ty and a smooth perturbation-return check at matched boundaries. Recompute refinement defects with the corrected norm and localise endpoint contributions. These can run alongside the prototype and should remain limited to checking correctness of the implemented equations.

## Answers to the brief's priorities

- **(b) first:** build and measure the small multilevel linear prototype.
- **(d) alongside it:** fix exact-start classification, with scale-aware criteria and regressions rather than a casually chosen default floor.
- **(c) defer:** preserve the production cut boundary; freeing pressure or moving the cut changes the task.
- **Bifrost residual analysis optional:** useful for a future scientific interpretation, not a gate for the requested solver work.

No change to Φ is justified or needed for the next implementation step. The control results now support proceeding with numerical acceleration while retaining bounded, focused correctness checks. Success is a faster, reliably converged solution of the same stated problem—not a solution chosen for agreement with a particular snapshot.
