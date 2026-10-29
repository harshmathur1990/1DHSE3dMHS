# Solver-time review, round 3 — 2026-10-29

Reviewed `CODEX_BRIEF_solver_time_round3_2026-10-29.md`, design §6.5–6.10, HEAD `d40f9de`, diff from `fd0cc20`, solver/qualification changes, spectral lab, restart measurement code, EOS table interpolation, and the slanted-input generator. No tests, scientific runs or cluster access performed. All numerical results cited are Claude's reported measurements. No scientific source changed. Recommendations remain for the user's decision.

## Decision and priority

**Do the truth-derived consistency/perturbation experiment next, before another 160-step run or a new production preconditioner.** The 80-step runs establish that the current methods have not met the time/accuracy goal and that objective decrease can diverge from Bifrost height accuracy. They do not establish the converged minimiser, prove physical non-identifiability, or justify changing weights to favour the known truth.

The decrement interval and common-objective comparisons are meaningful improvements. Preserve the user-selected threshold/count/budget and report unresolved or budget-limited outcomes honestly. No height-based production stopping rule, state prior, or new main/embedding-api default is recommended here.

## Code findings

### 1. High: restarted PCG does not check whether its initial residual already satisfies tolerance

`PoissonZ.pcg!` now supports `x0=true`, but sets `rel=1` and enters the recurrence without testing `‖b−Ax0‖/‖b‖`. For nonzero b and an exact initial solution, r=p=0 and the first step computes 0/0. This new path is used by qualification refinement. Approximate initial solutions already within tolerance also incur unnecessary iterations.

**Claude action:** check the actual initial residual before preconditioning/iteration. Define zero-b behaviour consistently for a nonzero x0; do not claim a zero residual solely because b=0. Return cleanly for an already-converged initial guess and handle nonfinite/nonpositive curvature or inner-product breakdown explicitly. Add focused tests with an exact nonzero initial solution, an approximate solution already inside tolerance, and zero b with nonzero x0. Keep the existing zero-start behaviour covered because this helper is shared with PoissonZ.

### 2. High: the effective EOS feasibility domain is narrower than the physical table domain

`PoissonTau.evaluate!`, lines 297–302, requires `state(T,u±1e-4)` to be finite to estimate the opacity derivative. A valid physical state within 1e-4 of either pressure-table edge therefore fails evaluation. The returned bad-cell count combines failed physical evaluation with failed derivative probes; it is not specifically the number of physical cells outside the table.

The Hermite table already supports a pressure derivative (`hermite(..., true)`), used by `lnrho_dY`; opacity is field 3. An analytic derivative of the table interpolant would avoid this artificial restriction. Alternatively use a validated in-domain derivative scheme. This is not authorization to extrapolate the EOS or extend its scientific validity.

**Claude action:** first distinguish physical out-of-table cells from derivative-probe failures. Log global location, log τ, T, u, distance to the nearest pressure boundary, proposed δu and which boundary limits α. Test valid states near both edges. Changing the derivative implementation requires its own comparison against interior finite differences and near-edge checks.

Use a global fraction-to-boundary scale for pressure updates before evaluating a trial:

`α_feas = min_cells { (u_hi−u)/δu if δu>0; (u_lo−u)/δu if δu<0 }`,

then start at `min(existing_scale, 0.99 α_feas)` when finite, with robust handling of invalid starting points and zero increments. Use the true valid domain of the evaluator and a documented numerical margin. This avoids repeatedly paying for predictably invalid trials; it does not remove a constrained optimum or guarantee progress when one cell limits the whole step.

### 3. Medium: work counters are not yet counts of actual products/evaluations

`work.final_cg=its` includes reused qualification iterations when the final state reuses `qinfo`, so summing qualification and final-diagnostic work double-counts them. `qualify_stop` calls restarted PCG and evaluates extra A products for true residuals, neither counted by `q.its`. Its `maxit` budget limits recurrence iterations rather than all products despite the docstring. `neval` counts trial evaluations but omits initial evaluations and linearisation restorations. Per-step timing ends before qualification.

**Claude action:** distinguish recurrence iterations, actual A applications, preconditioner applications, trial/restore evaluations, qualification, final diagnostic and end-to-end time. Set final diagnostic work to zero when reused. Do not label the present counters product counts. At minimum correct the labels before using them to compare efficiency per unit work.

### 4. Medium: restart checks and step-zero snapshots remain incomplete

`measure_steps start=DIR` checks kc and local i0, but not full grid, axis values, input identity, viewing angle, balance or units. It also saves the old column-HSE heights below the cut at step zero after pressure has been replaced by the restart. The evaluated atmosphere generally has different heights, the same issue previously fixed in `measure_warm`.

**Claude action:** validate restart metadata/shapes collectively before solving, and evaluate the actual starting state before saving s0. For the reference study, save the exact weights and a Float64 checkpoint alongside the Float32 visualisation snapshots; quantify whether Float32 pressure/height rounding affects very weak directions. Mark prior restarted s0 height diagnostics as stale where applicable.

## Interpretations that need correction

1. **ereg is not in Φ.** It changes the step, conditioning, decrement metric and finite-budget trajectory, but it does not define the objective minimiser. It cannot make the physical least-squares minimiser better determined unless it is added as an explicit state penalty, which would be a new scientific assumption. Answer the weights question separately from the ereg question.
2. **The coarse results are stronger than the brief says.** At μ0.3 vertical, c-only deflation changes the 1e-5 count from 385 to 174, and exact 64-vector deflation changes it to 51: about 2.2× and 7.5×. “Nothing” and “at most a factor of 3” apply only approximately to the looser 1e-3 comparison. State the target tolerance whenever comparing methods.
3. A dense spectral band does not prove a fixed-dimensional correction is useless or that multigrid is the only option. Relevant error/RHS components and application costs matter. Conversely, a good crop iteration reduction does not establish full-cube wall-time improvement.
4. `λ < 0.1 λmax` counts are a chosen spectral threshold, not a count of physically undetermined variables. The reported RHS fraction is squared weight in the preconditioned spectral metric, not an ordinary Euclidean “fraction of b.” Report both that weight and its inverse-eigenvalue-weighted contribution to the decrement.
5. Accurate gain ratios near a tiny feasible step support a local feasibility restriction. They do not prove that the model/objective is scientifically correct or that the EOS-table boundary is a physical constraint. Determine whether it is a numerical validity boundary, insufficient table coverage within an already validated EOS range, or genuinely unsupported thermodynamics.
6. Round 2 already measured a damping-consistent hybrid. Do not describe the retired additive approach solely by its earlier inconsistency or rerun the same experiment under a new label. A Fourier correction applied **to the residual after a block correction** is multiplicative, not simply additive; ordinary PCG needs an SPD composition.

## Answers to the scientific questions

### 1. Objective, weights and the inclined geometry

Variable scaling/right preconditioning can improve numerical conditioning without changing Φ. Changing residual weights generally changes the nonzero-residual minimiser. Neither operation creates missing physical information. A principled residual metric may be warranted, but must be derived rather than selected because it gives the best Bifrost height error.

Derive the map between the implemented `(Rx,Ry,Rray)` and physical Cartesian force components at a common point. The implemented horizontal residuals eliminate a ray derivative using the ray balance; they coincide with the desired force balance at a zero, but need not have the same least-squares metric away from a zero. Ray/vertical residual mixing introduces cross terms and geometry-dependent factors. If the target is an integral of Cartesian force error normalised by a specified physical scale, derive its metric and volume Jacobian explicitly, including off-diagonal residual weights when needed. Verify ray and vertical representations give the same physical diagnostic under that metric before comparing minima.

Currently wc uses layer-median H²Δz and wf uses Hf Hf+1/Δz; the face residual is integrated while horizontal residuals are nodal. Units, face-to-node approximation, slant geometry and the fixed-reference volume choice all need to be stated. Do not replace these with arbitrary factors of μ. State-dependent weights would add derivative terms if they are part of the target objective; freezing them is a different defined problem.

For a noisy/inconsistent atmosphere, an uncertainty-derived residual covariance is another principled metric, but it requires an error model. A Bifrost snapshot need not satisfy static MHS. Therefore a least-squares MHS solution can have lower Φ and worse snapshot height agreement without a solver bug.

### 2. What should convergence mean?

Keep three distinct statements: convergence of the chosen objective/metric; a work-budget stopping decision; agreement with a synthetic physical reference. Relative decrease per unit work is an efficiency or stagnation indicator, not stationarity. A damped decrement can be made small by increasing damping and cannot silently become the convergence certificate. A consistent nondimensional gradient or decrement can support angle comparisons, but does not guarantee equal cost or physical accuracy.

Use the current interval as a qualified statement about H+ereg; explicitly report that metric and the configured budget. If EOS bounds are intended as constraints on the optimisation problem, a bound-constrained projected-gradient/KKT condition may be appropriate; an unconstrained decrement is then a different test. If bounds are merely table-validity limits, report **domain-limited** rather than claiming the physical problem is solved. Standard active-set methods handle persistent bound constraints; [PETSc's bound-constrained solver documentation](https://petsc.org/release/manual/snes/) illustrates this distinction. Adding such a solver is not the first proposed change.

### 3. Preconditioner and a credible scaling test

Do not implement another generic hybrid before the objective/domain checks. If they pass, compare the already-corrected additive baseline with a symmetric balanced or symmetric multiplicative correction on the same systems; compare actual time to a specified true residual and decrement accuracy. Neither “block never moves the valley” nor “only Fourier can move it” is an algebraic property: sufficiently accurate solves of the same SPD system give the same direction.

For scaling, use a smooth periodic manufactured field over a fixed physical domain, sampled at several resolutions with consistent vertical grid, EOS and physical parameters. Freeze each system at a prescribed common physical state (or perturbation), rather than after a different number of nonlinear steps. Separately test a tiled fixed-resolution domain-growth sequence; it answers a different scalability question. Cover both balances and several angles/azimuths. Record defect/residual spectra, setup, products, preconditioner cost, memory, communication and true residual. Size-independent iterations are useful evidence; only total time at the same physical accuracy addresses the user's goal.

Horizontal multigrid remains a candidate if coupled pressure/height smooth error survives block smoothing and grows with domain size. Build a crop prototype first; handle the centred-stencil checkerboard branches and use an SPD cycle for PCG. [SciPy's least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) also distinguishes variable scaling from the objective and describes bound-aware trust-region strategies; these are methodological options, not a request to replace Julia or introduce a new dependency.

### 4. EOS feasibility: fraction-to-boundary first, damping second

Yes, diagnose and apply the global fraction-to-boundary safeguard before blind trial evaluation. Then test whether the small step is caused by one cell or a broad region, and whether the direction has feasible descent alternatives. Marquardt damping is a numerical control, not a substitute for domain information; making it the production default requires a successful end-to-end comparison. Repeated feasible α≈1e-3 is a reason to investigate active bounds/scaling, not simply raise damping indefinitely.

### 5. Truth-derived perturbation test: first establish that the state belongs to the model

`make_slanted_tau.jl` computes χ on the original geometric grid, shears χ separately from lnT/lnP, integrates τ from that sheared opacity, then interpolates T/P/z onto the τ axis separately. The solver instead recomputes χ from interpolated T/P and reconstructs z by trapezoidal integration. In general

`interpolate(χ(T,P)) ≠ χ(interpolate(T),interpolate(P))`,

and interpolated z need not satisfy the solver's discrete τ-to-height equation. This mismatch can grow with slant and remapping. The generator also carries snapshot density rather than enforcing the solver EOS density, and the solver fixes pressure/relative structure above the cut from column HSE. These are potential consistency differences to measure, not proven explanations of the observed drift.

**Priority experiment for Claude:**

1. On the existing input, measure the truth state's EOS/opacity consistency, discrete τ–z defect, cut-pressure mismatch, upper-column relative-height mismatch and original Cartesian MHS residual. Report by depth, field strength and angle, with extrapolated cells identified separately.
2. Construct an **admissible truth-derived state**: retain input T/B, initialise solved pressures from truth, enforce the actual fixed cut pressure and upper structure, and reconstruct z through the solver's own map. State exactly what was altered; report its distance from raw truth before solving. Keep the common weights fixed. If truth cut data are used instead, that is a separate boundary-condition experiment.
3. Evaluate residual components, Φ, gradient/decrement bounds and force diagnostics there, then take a few controlled feasible steps. Record which terms decrease as height error increases. Repeat from small smooth pressure/c perturbations. Walking away shows a disagreement between this constrained objective and this truth-derived state; it alone does not identify which assumption is wrong.
4. Run the same check on an independent known equilibrium: start with a smooth field-free hydrostatic case, then an analytic/manufactured magnetostatic atmosphere consistently mapped to inclined rays. Check discretisation defects decrease under refinement. Do not generate the only validation case by solving the same residual equations and call that independent physical verification.
5. If known equilibria recover correctly while the Bifrost experiment drifts, quantify dynamical/model/remapping discrepancy before proposing a new metric or prior. If they fail, investigate geometry, quadrature and boundary treatment before more production runs.

The existing references are enough to justify this diagnostic work, but they remain unconverged trajectories, not certified physical targets. Extend to 160 steps only after these checks, with a specific unresolved question and a stopping/work budget. A longer unconverged run by itself will not separate objective mismatch, active bounds and linear-solve error.

## Concrete next actions, in order

1. Fix and test restarted-PCG initial residual handling; correct work-counter labels/double counting and restart s0/metadata.
2. Split physical EOS failures from derivative-probe failures; add boundary diagnostics and trial feasibility scaling on the branch.
3. Execute the truth-consistency and short admissible-truth perturbation experiment above, plus a known-equilibrium control. This is the next scientific priority.
4. Correct the brief/design interpretations and report coarse gains at both tolerances. Preserve existing hybrid ablations as baselines.
5. Based on those diagnostics, propose either a derived physical residual metric, a bound-aware numerical step, or a crop multilevel prototype. Present any change to the scientific objective for the user's decision before treating it as the production target.

The central question is now whether the discrete, boundary-constrained MHS problem represents the desired atmosphere closely enough at small μ. Establish that while repairing the remaining numerical edge cases; do not redefine “converged” to meet a wall-time target.
