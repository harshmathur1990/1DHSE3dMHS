# Independent review: solver time, 2026-10-28

Reviewed brief `CODEX_BRIEF_solver_time_2026-10-28.md`, HEAD `e3c329c` (implementation through `af4d565`), relevant source, tests, measurement scripts, and implementation diff from `097571e`. Read-only inspection; no tests, scientific jobs, or cluster access performed. Measurement numbers below are Claude's reported results, not independently reproduced. Untracked report/figure files were left untouched. Recommendations remain for the user's decision; no scientific source was edited.

## Assessment

The column blocks and low-frequency correction are useful progress. However, the evidence does **not** yet establish physical non-identifiability or a wrong block-Jacobi solution. Different approximate linear solves can follow different nonlinear trajectories, and comparable objective values do not certify the same stationary state. Neither D2 nor a 40-step endpoint is presently a certified reference. Fix the convergence interpretation and measurement issues before introducing a physical prior or launching another broad production-size sweep.

The fastest useful path is: trustworthy frozen-system comparisons; isolate the Marquardt/Fourier mismatch; preserve the warm baseline; then test an SPD coarse correction on identical systems. No method can presently promise a step count independent of viewing angle.

## Findings requiring action

### 1. High: `pred` can declare convergence because damping or an incomplete solve made the step small

`PoissonTau.jl:919–925,969–972`: `pred = rhs'δ/Φ` uses the actual damped system and whatever PCG returned, without checking its residual. Increasing LM damping drives this quantity towards zero even when the undamped gradient remains large. Two consecutive values do not cure systematic bias. The accepted step is then taken and the convergence decision describes its **previous** linearisation state. The final undamped diagnostic is not used to qualify the result and itself defaults to only 80 CG iterations.

For H = JᵀWJ and D the total step penalty, an exact solve `(H+D)δ=b` gives

`bᵀδ = δᵀHδ + δᵀDδ`,

whereas the full physical residual model decrease is `2bᵀδ − δᵀHδ`. These are equal only for an exact unregularised GN solve. The implemented `model` correctly evaluates the latter for the scaled step, but `pred` is not an upper bound on the decrease available to an undamped step. Even the nominally undamped diagnostic retains `ereg`.

**Action:** retain the user-selected threshold/count/budget, but distinguish a step prediction from a stationarity certificate. When the threshold would trigger, qualify it with a reliably solved diagnostic system at the returned state, with its penalty and residual reported. If qualification fails, report budget/linear-solve limitation rather than certified convergence. Add a regression where large damping produces small `pred` despite a large gradient, and another where CG exhausts its cap. Do not add a height-based production stop without user approval; height accuracy is a benchmark diagnostic here.

The advertised Eisenstat–Walker rule is currently a heuristic `clamp(previous_pred,...)`, not a residual-based forcing derivation. Label it accordingly unless its relationship to nonlinear residual progress is established. A successful inexact-Newton forcing condition is also not guaranteed when PCG misses the requested tolerance.

### 2. High: Marquardt damping enters the operator and column blocks, but not the Fourier correction

`Aop!` adds `pr.mlm * pr.dlm * v`; `colblocks!` includes that diagonal. In contrast, `setprec!` builds Fourier A from Gz and `pr.ereg` alone. Under Marquardt, `ereg` stays at its base value. Thus the additive Fourier inverse can dominate low modes while ignoring arbitrarily large LM curvature. This is a concrete mechanism for poor hybrid conditioning; it is not evidence against Marquardt damping itself.

This does **not** inherently invalidate PCG: rebuilding a fixed SPD preconditioner between independent, zero-started PCG solves is valid. The concern is spectral quality. A sum of SPD/positive-semidefinite corrections is not intrinsically an invalid “over-correction.” Also, Euclidean CG residuals need not decrease monotonically; `rel > 1` alone does not prove an algebraic bug.

**Action:** at one identical frozen state and each LM value, compare block-only Marquardt, existing hybrid, and hybrid whose Fourier approximation includes a positive layerwise approximation to `μ diag(H)`. Check numerical symmetry/positivity and true residuals. Keep coefficients/preconditioner fixed inside each PCG call. Expected result if this diagnosis is right: block-only or damping-aware hybrid avoids the deterioration as μ rises. If block-only also fails, investigate scaling, block/operator consistency with nonzero μ, and roundoff before redesigning the outer method.

### 3. High: the D4b measurement script loses its previous column-HSE pressure baseline

`measure_warm.jl`: `ua,... = solve(lT, u0, z0,...)` passes `u0` directly into `solve_tau`, which updates `u .= ut` in place. Consequently `u0` now equals the previous solved pressure. Later D4b passes `ua[...]` as `uprev` and `u0[...]` as `ucol`. The update

`uprev + (u_new_column − u_old_column)`

therefore cancels to `u_new_column`, losing the intended previous MHS pressure correction. The height baseline is not overwritten in this same way. This affects the pending D4b benchmark, not the reported D4a measurements. The embedding implementation copies the solver input and saves ordinary slicing copies of the column baseline, so the same aliasing defect is not apparent there.

**Action:** preserve immutable copies of the original column pressure/height and pass a separate pressure copy into the initial solve. Assert that the baseline remains unchanged, and compare the constructed D4b state against the embedding API. Evaluate warm step zero through `evaluate!` before reporting its height: the script explicitly records stale column-HSE heights below the cut. Thus the table's step-zero height distances do not describe the actual warm starting atmosphere.

### 4. High: cold and warm runs generally minimise different weighted objectives

`tp_weights!` is called from each solve's starting atmosphere and freezes medians of height spacing, density scale height, and W. A warm start therefore changes wc, wf and ereg relative to the cold run on the **same** new T/B cube. At a nonzero-residual minimum, changing these weights can change the minimiser. Warm/cold endpoint disagreement is not solely a solver or initialisation effect.

The frozen-state labs also recompute weights at the saved late state rather than retaining those of the trajectory. Useful as separate local experiments, but their systems are not precisely the trajectory's original systems.

**Action:** provide an experimental way to reuse explicitly saved weights/penalty. Compare starts and preconditioners with identical objectives. Alternatively define weights from a canonical new-input column-HSE state for both starts, subject to user review before changing production semantics. Archive exact weights, state, residuals, cut index, μ, azimuth, balance and penalty for each lab.

### 5. Medium: centred periodic derivatives have additional exact height gauge modes

For even horizontal sizes, the derivative symbol used here is `sin(k Δ)/Δ`; it vanishes at both zero and Nyquist. A cut-height perturbation constant down each column with horizontal pattern `1`, `(-1)^i`, `(-1)^j`, or `(-1)^(i+j)` has zero centred horizontal derivative and zero column-spacing perturbation. Consequently J cannot see these height shifts, including through the Lorentz-height terms. On 504×504 there are four such height modes. Removing mean z fixes only the constant mode. `ereg` makes the step system positive definite but does not add a state penalty to Φ or fix arbitrary initial amplitudes of these modes.

This does not explain the reported **low-frequency** discrepancy by itself, but it makes a generic “smallest eigenmode” investigation misleading unless these known modes are identified first.

**Action:** explicitly probe these four c modes and report their amplitudes in reference comparisons. Test odd and even grids. Choose a documented gauge/projection or a different discretisation only with scientific justification; changing the stencil is a scientific change. Include derivative-null modes in the numerical analysis rather than assuming all poorly determined modes are low k.

### 6. Medium: current tests do not certify the claimed new regimes

`test_block.jl` checks valuable analytic blocks against operator probes, but only with `mlm = 0`, and its nonlinear tests mainly check objective decrease in 15 steps. They would accept the reported large height discrepancy. `test_view.jl` already supplies adjoint/finite-difference checks, including μ=0.25: extend those rather than duplicating them, using late near-cut states and an epsilon sweep. `test_warm.jl` tests construction and identical-input repetition, not perturbed-input D4b or benchmark/API equivalence.

Shared `PoissonZ.pcg!` has no explicit nonfinite, nonpositive-curvature or preconditioned-inner-product breakdown status, and reports recursive residual only. Because this is shared production code, first add diagnostic checks in the branch/lab and propose any shared API change separately. At termination compute `‖b−Aδ‖/‖b‖`; do not silently treat a capped solve as a converged linear solve.

## Answers to the scientific/design questions

### Diagnosis and physical assumptions

Large δ plus tiny allowed α is evidence of a poorly scaled or weakly constrained **local model**, not proof of a physical null space. Compare tight solves of the same system before attributing differences to the preconditioner. In exact arithmetic, converged PCG with different SPD preconditioners gives the same δ. The reported block/Fourier directions with `pred ≈ 1` but radically different amplitudes make that comparison the first priority.

Use dimensionless variables `δc/Href` and `δu`, with a fixed documented Href. On a manageable representative crop, form the weighted Jacobian explicitly and inspect singular vectors (normal equations square its condition number). On larger problems use matrix-free, scaled Ritz/Lanczos diagnostics, with genuine residual checks on the resulting eigenpairs. Small column-block eigenpairs reveal local weakness but cannot establish global low-frequency weakness. Analyse H separately from H+ereg and H+LM; project the known height gauge modes first.

For each candidate soft direction v report its depth support, spatial spectrum, `‖W½Jv‖`, pressure/height amplitudes, and contributions from horizontal, ray and vertical-balance residuals. Verify `R(x+εv)−R(x)` against εJv over several ε. Then sample the actual objective along v. A mode soft only in the linearisation may be nonlinear curvature; a mode disappearing under refinement may be a discretisation artefact.

Ray and vertical balance are equivalent at an exact zero-force solution, but mixing their residuals changes the least-squares metric when residuals cannot vanish. In D0 the mixing depends on Δz and tan θ, so it can amplify conditioning and alter the fitted compromise. Compare transformed residuals and consistently transformed weights at the **same** state; do not call the two least-squares objectives interchangeable. Vary cut location while controlling weights and align the physical comparison region. Cut/μ changes alter the actual problem as well as its conditioning.

### Priors versus damping

Do not introduce a pressure or Wilson-depression prior solely to recover the D2 endpoint. If a state prior is scientifically justified, put it explicitly in the objective, include its gradient/Jacobian/model prediction, choose its strength from an uncertainty or sensitivity study, and report the bias. A penalty on the step δu is damping; it is not a state prior and does not define the same stationary point as a penalty on `u−u_HSE`. A low-k c prior can directly suppress the Wilson-depression signal of interest. Smoothness relative to column HSE can suppress real magnetic structure as well.

A scaled trust region is a useful numerical first experiment because it need not change the target objective. Use a global scaled norm with representative height/pressure scales; compare with existing component caps as safety bounds. It cannot supply information absent from the equations.

### Coarse preconditioning

Keep additive hybrid as a baseline. Prefer a Galerkin correction using the **current** operator, including current damping, over assuming the field-free median is accurate in soft modes. With coarse basis Z, use `E=ZᵀAZ`, `Q=Z E⁻¹ Zᵀ`. One SPD balanced form with SPD block inverse B is

`M⁻¹ = Q + (I−QA) B (I−AQ)`.

Use an independent coarse basis and a stable SPD coarse solve. Arbitrary one-way multiplicative composition is generally nonsymmetric and unsuitable for ordinary PCG. Even a symmetric block→Fourier→block sweep needs positivity analysis; symmetry alone does not guarantee SPD without appropriate smoothing bounds.

Start with the measured discrepancy's modes: a small low-k set and 2–4 scaled vertical vectors (c plus pressure/height response shapes, enriched by observed slow Ritz vectors). Sweep klow=2,4,8 and increase vertical rank only when projected residuals demonstrate a need. There is no justified universal klow or vertical rank yet. A dense coarse operator costs quadratically in coarse dimension: klow=8 times many vertical vectors can already make setup expensive. Prototype on a crop, measure setup/apply/reduction costs and amortisation, and consider a small coarse grid or adaptive basis rather than projecting every Fourier mode times every vertical degree of freedom on the full cube.

### Outer iteration and warm starts

First isolate linear accuracy, damping consistency and step scaling. If tight, correctly scaled steps still have small gain ratios, compare the exact Hessian action with GN on a crop; the omitted residual-weighted second derivatives matter when the residual is nonzero. Only then consider Newton–Krylov with negative-curvature safeguards and a trust region. A generic switch to Newton is not a guarantee of faster or angle-independent convergence.

B continuation changes the intermediate physics but is a plausible optional cold-start aid: Lorentz force scales as B², so scaling B by sqrt(s) scales L by s. Count every stage in total time. Angle continuation is invalid if it merely changes μ on an unchanged inclined-view input cube: different viewing angles sample different rays. It requires a consistent data remapping/homotopy. Column-HSE sweeps can erase magnetic corrections and need an objective-decrease safeguard; they are lower priority than the linear-system fixes.

D4b's column-response update is a sensible first predictor after fixing the benchmark. A 3% T perturbation need not be small in opacity or cumulative height response. Evaluate the candidate, EOS validity and residual before stepping. Compare cold and warm candidates with the same weights and choose a safe candidate; do not assume smaller Φ alone certifies better height accuracy in a flat valley. Once the linear model is dependable, a sensitivity predictor for changed T/B is a stronger option. Cache metadata must distinguish last **valid** from last **converged**: the current API intentionally caches valid max_steps/no_descent results too.

## Priority prompt for Claude: implement and measure in this order

1. **Repair evidence first.** Preserve D4b baseline copies; evaluate real step-zero heights; save/reuse a common weight/penalty definition. Add regressions for baseline aliasing and API/script agreement. Expected: D4b retains the previous magnetic pressure correction and identical-input warm starts reproduce the actual previous atmosphere up to gauge. Report earlier D4a results separately; do not relabel them D4b.
2. **Freeze one μ=1 and one μ=0.2–0.3 troublesome state.** On a tractable crop, test adjointness/finite differences, operator and preconditioner symmetry/positivity, nonzero-Marq block equality, and true residuals. Compare tight block/Fourier/hybrid solutions to a direct solve where feasible. Expected: the same system yields the same step; persistent disagreement identifies algebra/precision trouble. Separate this from nonlinear trajectories and report any cap-limited linear solve explicitly.
3. **Ablate Marquardt at those states.** Compare block-only, existing hybrid, damping-aware hybrid across LM values; log actual damping used, cap scale, retry counts, all CG work and build time. Expected: corrected coarse scaling removes μ-dependent hybrid deterioration. Otherwise investigate block/scaling defects before outer-method changes.
4. **Qualify the stop and establish references.** Retain configured 1e-3/two-step/25-step policy, but mark its convergence limitations; use a reliable final diagnostic for the study. Extend reference runs until objective, stationarity and gauge-aligned z/pressure are stable under tighter inner tolerance and more steps, preferably from two starts. A 40-step endpoint is a provisional reference. Report solver error relative to this reference separately from Bifrost model/truth discrepancy. Height diagnostics do not become a new production stopping rule.
5. **Identify soft modes, then add one coarse experiment.** Remove known gauge modes, distinguish physical sensitivity from cut/weight/discretisation effects, and trial a small balanced Galerkin correction. Expected: it reduces both true residual work and low-k step discrepancy at fixed accuracy. Reject it if setup cost outweighs saved products or if it merely changes the trajectory without improving time to the common reference.
6. **Only then benchmark full cubes and D4b.** Use identical 3-rank ×32-thread allocation/binding, fixed BLAS/FFTW settings and node type, common input/cut/objective, several azimuths and μ=1,0.5,0.3,0.2. Include setup, rejected retries, column-HSE work, final diagnostics and output overhead in end-to-end time; also report solver-only time. Existing snapshot callbacks are outside the per-step `seconds` timer, and the final diagnostic adds work outside accepted-step totals. Report maximum rank elapsed time, memory, stop reason and time to a common accuracy, not just 40-step wall time. Warm tests should include unchanged input, small/3% T changes, B changes and clearly identified converged versus budget-limited parents.

Defer state priors, full Newton and physical continuation until these cheaper experiments discriminate their need. Success means comparable **time to a verified common accuracy**, not simply fewer iterations or a forced stop at 25. The current results support pursuing the hybrid approach, but do not yet support the requested μ-independent accuracy/time claim.
