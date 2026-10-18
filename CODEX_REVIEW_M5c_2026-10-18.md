# Review of Claude's M5c results — 18 October 2026

Reviewed `CODEX_BRIEF_M5c_2026-10-18.md`, commit `ef2d86f`, the new audit and joint-GN code, the comparison script/report, and the actual saved checkpoint and endpoints in `HSEMHS/out/m5c/`. This review adds diagnostics only; it changes no production solver, objective, prior, observation model or acceptance policy.

## Recommendation

**Keep corrected alternation as the current baseline, but do not launch another broad preconditioner search or add a pressure prior solely because the original audit increased.** The next priorities are:

1. Resolve the magnetic-data admissibility obstruction and the observation/preprocessing model.
2. Retain the original fixed-metric audit result, but add the common-metric sensitivity results below; do not use it as the sole progress verdict.
3. Check the joint step's allocation of its trust radius and use actual GN products for its diagonal probes before paying for a colored diagonal at every rebuild.
4. Discuss the scientific pressure prior explicitly, with the understanding that it selects an estimate and cannot repair an incompatible magnetic-data requirement.

The comparison is useful progress. The new small tests pass, the saved objective/residual numbers reproduce, and both methods substantially reduce force imbalance. Neither produces a converged, data-admissible equilibrium.

## 1. The pressure-audit verdict is strongly reference-dependent

### What I measured

I reconstructed the model and recomputed **every saved state with each of three common metrics**, using the existing `reference_metric` and `full_audit`. Each table row holds its metric fixed across all three states:

| Common reference metric | Checkpoint m | Alternation endpoint m | Joint endpoint m |
|---|---:|---:|---:|
| Original checkpoint | 0.289716 | 8.25608 | 48.8382 |
| Alternation endpoint | 0.717140 | 0.00775268 | 0.387519 |
| Joint endpoint | 0.820555 | 0.107654 | 0.0745891 |

With the original metric, neither meets the preregistered tenfold reduction. With the common alternation-end metric, alternation improves about **92.5×**, joint about **1.85×**. With the common joint-end metric, joint improves about **11.0×**, alternation about **7.62×**, and their endpoint ordering by m reverses.

These later metrics are a **post-hoc sensitivity analysis**, not replacements chosen to award a pass. The original recorded numerical criterion still fails. Nor do the alternative reductions imply physical success: the force and magnetic-data requirements still fail independently.

At alternation's worst original-audit node `(24,40,13)`, the checkpoint diagonal is `5.336e-12`, while the current reference diagonal is `4.339e-5`: a factor **8.13 million**. The original audit proposes a change of about `3.77e5` in ln P there. That is a mathematical diagonal-model correction, far outside a useful local linearization. In fact the reported m can exceed the entire nonnegative objective, so it cannot be interpreted as an attainable objective decrease.

### Answer to question 1

The fixed positive metric still defines a valid stationarity diagnostic: its constrained correction vanishes at the corresponding first-order conditions. But **a fixed metric does not make that diagnostic a monotone global measure of progress**, particularly through many orders of pressure change.

I proposed this diagnostic and the tenfold test. The new evidence shows that using that test alone as a general progress verdict was too restrictive. Preserve its failed result; distinguish it from the incorrect conclusion that neither method made progress, or that complement pressure must be physically unidentifiable.

A simple example explains the behavior without an implementation defect: for `f(u)=(exp(u)-a)^2`, increasing a very small pressure toward a reduces f while initially increasing `|df/du|`. A metric based on the very small starting pressure amplifies that rise. Here the metric itself is fixed; the nonlinear Jacobian and gradient are changing.

Do **not** promote a pressure audit restricted to the proxy-identifiable set into the sole acceptance test. Keep the original common metric and all-node maximum, split the diagnostics on the fixed mask, and add a clearly declared sensitivity metric. A symmetric pooled diagonal, for example an elementwise maximum of the three saved-state diagonals with a stated floor, could be used for a subsequent comparison; evaluate all states with that same metric and label the analysis post-hoc. Do not refresh the metric separately for each method and then compare the resulting numbers as though they used the same ruler.

For solver decisions, also report actual pressure/field changes, force residuals, data admissibility, current constrained optimality and a local feasible model-decrease test. No single diagonal audit should overrule them. Pressure quantiles are descriptive; they do not exempt the worst nodes from the user's convergence criterion.

## 2. A relevant preconditioner distinction precedes the colored-diagonal experiment

**Code finding:** `src/MHSStaggered.jl:635–649` estimates the default magnetic layer diagonal by finite differences of the **nonlinear objective gradient**, not by `gn_hvp!`. Thus it samples the full Hessian, including the residual-curvature term. It is described as a GN diagonal in several new comments/reports, which is inaccurate for nonzero force residuals. The exact colored option does use GN products.

For a squared-residual objective, the full Hessian contains both `2 Jᵀ W J` and `2 Σ r_i W_i ∇²r_i`. The second term need not be positive and is potentially relevant on the real-data checkpoint. The subsequent absolute values/floors make a usable positive **preconditioner**, but do not turn it into the GN diagonal.

This does not invalidate joint GN or prove the preconditioner caused its poorer result. It does create a concrete, inexpensive hypothesis that is especially relevant to the C1b/Bifrost reversal: near a synthetic zero-residual solution the two Hessians nearly agree, whereas far from equilibrium they need not.

**First test:** estimate the same layer-averaged diagonal with the same probe vectors and actual `gn_hvp!` products. Compare the estimates and a short joint continuation from the same checkpoint, accounting for construction work. This is a more controlled change than jumping immediately to hundreds of colored products per rebuild. It also makes the preconditioner match the advertised model. Retain positivity safeguards for noisy finite-probe estimates.

## 3. What the solver reversal does and does not establish

Alternation is the better baseline for reducing the force objective at this checkpoint and budget: raw force falls from 9353 to **2675**, versus **4354** for joint GN, with correspondingly smaller maximum layer RMS and smaller magnetic-gradient/audit components.

However, “better on every measure” should be narrowed. Joint GN has **slightly lower data misfit** (`0.317500` versus `0.317812`) and a shorter recorded wall time. The common-metric table above also shows that full-state audit ranking is not invariant to the reference choice. Neither arm is close to physical equilibrium: the real-data layer RMS limit is 0.01, whereas the reported maxima are 243 and 312.

The C1b comparison is suggestive but was not actually matched at 3000 work: alternation used 6734 and joint used 3054, before the block-budget fix. Joint's better result with less work is interesting, but this is not a clean universal ranking. Its C1b RMS of about `1.5e-6` also remains above the synthetic gate.

### Answer to question 2

The reversal shows **problem-dependent performance of these implementations**, not that joint directions cannot help, that all complement pressures defeat every method, or that exact colored diagonals are the necessary remedy.

All 81 joint CG exits being on the trust boundary is not evidence that the residual tolerance was missed incorrectly. Boundary exit is normal for a truncated trust-region solve. It does make radius geometry worth examining. The joint radius is initialized from **the pressure diagonal alone** (`src/MHSJointGN.jl:70`), and pressure and magnetic corrections then share that one ball. The alternation gives the magnetic block an independent opportunity to move.

Before a long preconditioner campaign, log at every proposed and accepted joint step:

- D-norm fractions in pressure, Ã and mean flux, before and after projection/capping;
- actual maximum relative pressure and magnetic changes;
- pressure, magnetic and cross-term contributions to the quadratic model;
- the part of an A step removed by a consistent gauge projection, if one is available;
- final model residual and inner exit reason.

The new joint routine still records the preceding `cg_rel` when it exits at the boundary (`MHSJointGN.jl:90–92`); it does not recompute the returned direction's residual there, unlike the revised pressure block. Correct that diagnostic before interpreting it. This is a logging issue, not proof of a faulty accepted direction.

A **short frozen-model comparison** at the existing checkpoint is then sufficient to test the radius-allocation hypothesis: compare a magnetic-only direction, the current joint direction, and a joint direction under a declared alternative block scaling/separate block step limits. Keep the objective and physical feasibility constraints fixed, and compare attainable predicted reduction and the actual trial reduction at equal work. Support for the hypothesis would be a dominant pressure share suppressing a useful magnetic correction, with a better validated step after changing the step geometry. If that evidence is absent, investigate a block/Schur preconditioner. A pressure-block inverse can address spatial/coupled modes that a more exact magnetic diagonal still misses.

Colored probing costs roughly 375 products **per build**, not just once. Rebuilding it at all 65 accepted steps would exceed 24,000 products before the main solve, against the current total budget of 3000. If tested later, count that cost and consider controlled reuse. Do not treat “exact diagonal” as a cure for off-diagonal coupling or gauge directions.

## 4. The unchanged outlier is a bottom-boundary magnetic problem

### What I measured

The largest normalized residual in all four states is **Bz at face `(i,j,k)=(35,32,89)`**, the bottom boundary face:

| Quantity | Value |
|---|---:|
| Interpolated observed Bz | 34.7185 G |
| Face σ | 7.18005 G |
| Initializer fitted Bz | 114.9009 G |
| Checkpoint fitted Bz | 114.8787 G |
| Alternation endpoint Bz | 114.8343 G |
| Joint endpoint Bz | 114.8578 G |

The maximum residual is 11.1674σ in the initializer, 11.1643σ at the checkpoint, 11.1581σ after alternation and 11.1614σ after joint GN. It is not literally unchanged; the small movement is hidden by printing two decimals. A second large outlier lies just above it, at `(35,32,88)`, also in Bz.

This is strong evidence that the 11σ obstruction **predates the nonlinear MHS comparison**. It is not caused solely by pressure wandering in the upper complement. It does not prove that a data-admissible divergence-free field is impossible: an unconstrained least-squares projection can concentrate error in a few faces, and the initializer also includes a regularizer and a finite linear-solve budget.

I checked two necessary conditions for a field satisfying `|B−Bobs_f|≤t σ_f` and discrete `div B=0`:

- A cellwise divergence bound gives **t ≥ 1.9200**, with the worst cell `(13,19,88)`.
- Constancy of the horizontally averaged Bz gives only **t ≥ 4.63e-5**. Global vertical-flux mismatch therefore does not explain this outlier.

The first is a bound on the *maximum* normalized correction, not on mean χ². Neither necessary condition rules out t=5, and neither proves a feasible field exists.

I also tried a sparse linear feasibility problem for divergence-free face corrections bounded by **4.9σ**, using the same block-averaging/interpolation construction. It has 472,752 variables and 156,996 equalities. HiGHS reached the explicit **60-second limit without a feasibility verdict**. This run is inconclusive; it is not a certificate of infeasibility.

### Answer to question 4, and the next experiment

**Yes: settle the observation/admissibility issue before another large nonlinear solver search.** The first question is not yet the calibrated mean chi-square threshold. It is whether the declared data requirement is attainable for a divergence-free field on this grid at all.

Run the convex, data-only problem

`minimize t, subject to div B=0 and |B_f−Bobs_f|≤t σ_f`.

It needs no pressure, EOS, force weight or Q penalty. Use an adequately converged sparse optimization method and verify the final constraint residual or an infeasibility/dual certificate; the time-limited trial above is not enough.

**Decision criterion:**

- A certified minimum **t>5** means no solver can meet the current maximum-data-error requirement, even before force balance. Audit the input locations, transfer/coarsening, boundary interpolation and uncertainty/model-error budget before attempting MHS convergence.
- A feasible witness at **t≤5** establishes that the maximum-error requirement alone is attainable. Also evaluate the required fraction above 3σ; a witness meeting both conditions is magnetic-data-admissible. Failure of that fraction in one witness is not a proof that no other witness meets it.
- If feasible, compare the present initializer with a data-only **bound-constrained** fit. A small mean least-squares penalty does not enforce a 5σ maximum or a 1% tail constraint. Continuing to hope those post-hoc tests will pass can be an objective/acceptance mismatch rather than a solver defect.

In parallel, confirm whether the original magnetic data are native face values or centre values and whether coarsening preserves magnetic flux on the intended staggered grid. The present workflow block-averages component values and then interpolates to faces; this is not automatically a divergence-preserving transfer between the native scheme and the new one. A representation error should be measured and documented, not absorbed by increasing σ until a run passes.

The previously deferred covariance/mean-model calibration remains important. If the observations are centre data, `HB_f−y` is their observation residual; interpolating y gives mean `IHB_f`, not automatically `B_f`. Merely retuning the threshold of the present face-identity penalty does not resolve that distinction. Keep any changed likelihood in a separately identified experiment, rather than silently changing it during another solver comparison.

## 5. Is it time to decide on a pressure prior?

### Answer to question 3

**It is reasonable to put the scientific choice on the agenda now, but the increased audit is not sufficient evidence to choose a prior or its strength.** Its interpretation is demonstrably metric-sensitive, and these finite-budget runs have not established uniqueness, nonexistence or formal pressure identifiability.

The magnetic feasibility and observation-transfer experiment should precede adopting a prior as the solution to M5. A pressure prior cannot repair an 11σ magnetic outlier or make incompatible magnetic constraints feasible.

The substantive question for the user is whether the product should return only what the fixed-T force/data constraints determine, with unresolved pressure reported, or select a reproducible atmospheric estimate using stated prior information. If the latter, document the reference pressure, its uncertainty in ln P, spatial correlations and how it enters the objective. HSE is a possible reference, but it is scientific information, not a neutral numerical gauge. Specify the support independently of the drifting inferred pressure; do not let a moving proxy mask decide which nodes receive a prior.

Validate a proposed prior by sensitivity of P, B and force balance to plausible prior strengths/references, with all original residual and data flags retained. Do not set its strength by the condition that the optimizer finally reports convergence. Continuing to use the all-node change test is compatible with a prior-selected estimate, but the interpretation of the resulting pressure must be explicit.

## Verification, code notes and reproducibility

- Independently reran the **nine new M5c assertions**: all pass. I read the supplied full-suite log (`EXIT 0`); the known C2 expected failure is still present. I did not independently rerun the full suite.
- Recomputed the saved-state objectives, three-by-three audit table, magnetic outliers and necessary divergence/flux bounds. No nonlinear comparison was rerun.
- Added `scripts/m5_bifrost/codex_m5c_result_review.jl` for those saved-state checks. From `HSEMHS/`, run `julia --project=. -t 4 scripts/m5_bifrost/codex_m5c_result_review.jl`. It reconstructs the original initializer and requires the existing `out/m5c` files; setup can take several minutes.
- Added `scripts/m5_bifrost/codex_m5c_face_feasibility.py` for the separate 60-second feasibility diagnostic. It reconstructs the preprocessing from the input in Python; Float32 summation order may differ slightly from Julia. Its necessary local bound reproduces 1.92000070484. The recorded LP run timed out and supplies no witness or certificate. The [SciPy linear-programming documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) distinguishes a time limit from optimality and infeasibility.
- MPI initialization required execution outside the sandbox. Existing production files and reports were not edited.

Two implementation cautions are worth retaining for later production work: `reference_metric`/`full_audit` use `MPI.Allgather` with equal-sized local buffers and gather large arrays on every rank, so uneven horizontal decompositions need coverage; and `full_audit.worst` currently sorts local nodes, not a globally selected worst-node list. Neither affects these one-rank comparisons.
