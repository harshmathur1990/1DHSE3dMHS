# Solver-time review, round 2 — 2026-10-28

Reviewed HEAD `fd0cc20`, both solver-time briefs, changes since `e3c329c`, `PoissonTau.jl`, `frozen_lab.jl`, `measure_warm.jl`, and the new tests. No tests, scientific calculations, or cluster jobs were run. Cluster results and test counts are Claude's reports. This supplements the first review; scientific source files are untouched. Recommendations are subject to the user's decision.

## Recommendation

The baseline-copy, common-weight and Fourier-damping fixes address the earlier concrete issues. The crop direct comparisons and derivative checks are useful evidence. The next priority should be a **small end-to-end damped-block experiment with complete cost accounting**, alongside a sound qualification diagnostic. Do not wait for an expensive full-cube coarse-space sweep before establishing whether the cheaper approach solves the actual nonlinear problem efficiently.

Keep the configured threshold, count and step budget. Do not introduce a height stopping rule or a physical prior. Report accuracy versus a verified reference separately from whether the configured stopping policy fires.

## Findings and corrections

### 1. High: a relative linear residual is not a certificate of decrement accuracy

`qualify_stop` computes an approximate solution x of `A x = b`, where `A = H + D`, `H = JᵀWJ`, and `D = diag(ereg0)`. A true Euclidean residual of 1–2% does not imply that `bᵀx/Φ` is within 1–2% of `bᵀA⁻¹b/Φ`, especially with the reported conditioning. Residual components along weak eigenvectors can leave substantial decrement error. The implementation is a useful qualification attempt, but calling the threshold comparison certified overstates its guarantee. Moreover, this is a metric based on **H+D**, not the unregularised H problem; disclose that distinction.

There is an inexpensive conservative bound using the positive diagonal already present. For `r = b−Ax`, define

`L = 2 bᵀx − xᵀAx`,

`U = L + rᵀD⁻¹r`.

Then, in exact arithmetic, `L ≤ bᵀA⁻¹b ≤ U`, because the missing term is `rᵀA⁻¹r` and `A ⪰ D > 0`. Divide by Φ for the decrement interval. This is an algebraic bound for the **regularised metric**, not a bound on physical height error or nonlinear global optimality. Account for roundoff, use a small safety margin, and verify all diagonal entries are finite and strictly positive. Derivation is local to this review; no external theorem implementation is assumed.

**Action:** report the interval. If U/Φ is below the configured threshold, the regularised decrement qualifies; if L/Φ is above it, continue; if it straddles, refine within a work budget or report unresolved. The upper bound may be pessimistic, but a pessimistic explicit bound is preferable to an unjustified certificate. Validate interval containment against the dense crop solve. A damped decrement is a legitimate reported diagnostic but cannot replace this bound silently.

### 2. Medium: qualification needs explicit zero/nonfinite handling and clear returned diagnostics

In `qualify_stop`, a nonzero-residual stationary point can have `‖b‖=0`. The PCG helper handles zero b, but qualification then computes 0/0 for `true_rel`. NaN makes `q.true_rel > 2*qual_tol` false, permitting the following `q.pred < gn_tol` branch to accept without a finite diagnostic. Handle exact zero b explicitly, and reject all nonfinite metrics/negative-curvature breakdowns. Also handle near-zero Φ deliberately rather than relying on ratios.

`info.stop_check` contains the detailed check, but the embedding wrapper currently does not propagate it into its public status/stats. The later final stationarity solve can overwrite the visible decrement/CG diagnostics with a less accurate 80-iteration result. A rejected check can also remain in `qinfo` after later outer steps, so tag it with the state/step where it was evaluated.

**Action:** return the qualification outcome, evaluated step, bounds, tolerance, true residual and work counters through the embedding status. Reuse an applicable final qualification diagnostic instead of silently replacing it with a weaker duplicate solve. Add zero-gradient, nonfinite rejection and stale-check metadata tests.

### 3. High: do not infer the full low-end spectrum from the present Ritz analysis

The Ritz residuals of 17–58 relative to θ are too large to identify the reported vectors as accurate low eigenvectors. The ratio of the extreme Ritz values is not a certified condition number; an incomplete Krylov space generally misses extreme eigenvalues. A handful of such values cannot establish a dense Laplacian-like ladder or μ-independent conditioning.

`coarse_lab` stores 300 recursively generated preconditioned residual directions but does not reorthogonalise them. In finite precision these need not remain M-orthonormal; the tridiagonal reconstructed from CG coefficients need not equal the projection in the stored basis. It can still be a useful diagnostic, but it does not establish reliable eigenvectors merely by multiplying V by the tridiagonal eigenvectors. The negative off-diagonal convention is not by itself a defect.

**Action:** on a crop, compare the reconstructed Ritz vectors/eigenvalues with the explicit symmetric transformed problem, using the block factorisation `M=LLᵀ`, `Ahat=L⁻¹ A L⁻ᵀ`. Check the M-Gram matrix of V, true Rayleigh quotients and residuals. For a modest retained basis, orthonormalise globally and solve the actual projected generalised eigenproblem, removing dependent directions. Use restarted/reorthogonalised symmetric Lanczos if spectral diagnosis remains necessary. Treat current Ritz deflation as an empirical trial space, not a validated representation of the true slow eigenspace.

### 4. Medium: revise conclusions that exceed the measurements

- At small μ none of the full-cube “tight” solves converged to the requested tolerance. Label the “Exact step” column **best available approximate step**, with the solver/residual identified. The crop results support algebraic correctness but do not establish exact full-cube directions.
- Differences in trajectory are plausible and expected with inexact solves, but do not yet exclude all precision/conditioning effects on full cubes. Keep the conclusion conditional.
- The original D3c poor end-to-end result used the old damping-inconsistent hybrid and showed capped CG with large residuals. It does **not** imply its cost “must” have come from the outer loop. Measure the corrected implementation before assigning the cause.
- “μ-independent CG” from three frozen systems and a bound of 141 iterations is too strong. This is encouraging performance over three cases at one point in each trajectory; a 26-to-141 range is still material. Different balance choices also confound a pure μ comparison.
- Dropping additive hybrid as the next performance candidate is reasonable. Keep it as a diagnostic baseline; its inferiority here does not establish that all coarse corrections are ineffective.

### 5. Medium: dense full-cube trial spaces need a memory gate

For μ=1, 13.97 million unknowns × about 718 per-level low-k vectors × 8 bytes is about **80 GB for one global distributed matrix**. `Zm` and `AZ` alone require about 160 GB across the node; constructing a list of full arrays before concatenation can add another matrix-sized peak. The 300-vector Lanczos basis adds about 34 GB while it exists. These are decimal GB, before operator, state, block factors and work arrays. Distribution across three ranks does not reduce total node memory.

**Action:** estimate and record peak memory before any additional full-cube trial-space run; first use crops, reduced ranks of vertical shapes, implicit Fourier prolongation or streaming construction. Ensure current allocations fit the actual node. Never judge a method solely by iteration count when each iteration already costs 2.5× more. The final target is time and memory at the same accuracy.

## Answers to Claude's questions

### Coarse space: couple pressure and height, then consider a horizontal hierarchy

Yes: c-only vectors need not represent an energy-soft joint pressure/height displacement. With variables split `(c,u)`, ideal extension of a coarse c vector is

`u = −Auu⁻¹ Auc c`,

and the exact Schur operator is `Acc − Acu Auu⁻¹ Auc`. However, **Auu is horizontally coupled**. Eliminating pressure using independent column blocks is an approximation, not an exact Schur reduction. Distinguish the exact formula from its affordable surrogate.

First trial a few c modes with an approximate energy-minimising pressure extension from column/subdomain blocks, plus a small number of representative pressure shapes. Or smooth initial aggregate vectors with the current A while maintaining gauge control. Compare with c-only at the same coarse dimension and setup budget. Do not assume column-HSE response is the energy-minimising shape; test its residual/energy.

If this improves a crop but convergence still grows with horizontal grid size, horizontal multigrid with column blocks is the stronger long-term option. Use symmetric smoothing and Galerkin coarse operators for PCG. Centred-stencil Nyquist/alias branches need explicit treatment; a naive coarse grid representing only low Fourier frequencies can miss other slow modes. Coarsening and redistribution across the three x slabs need to be costed. Near-null-space-aware aggregation is standard practice; [PETSc's GAMG guidance](https://petsc.org/main/manual/manual.pdf) describes supplying problem-specific near-null vectors. This is methodological context, not a recommendation to add PETSc to the repository now.

### Damping: scientifically acceptable as a step choice, but assess total progress

A small fixed positive Marquardt term is acceptable as an opt-in numerical step penalty: it does not add a state prior to Φ, and exact zero steps still correspond to zero gradient. It may nevertheless stall practical progress along weak directions, so neither fewer CG iterations nor a small damped decrement is enough.

First compare block-only `mlm=0.01` and `0.1`, plus one adaptive variant, with identical objectives/starts. Fixed damping removes retry overhead and provides a clean baseline. Log achieved physical model decrease `2bᵀδ−δᵀHδ`, actual decrease, gain ratio, cap scale and time. Do not compare candidates only by `bᵀδ`, which is affected by damping.

For the adaptive variant use a coherent scaled trust-region/LM strategy: shrink or reject a poor-gain trial, expand for a good-gain step near the trust boundary, and count rejected solves. Raising damping merely because a global max cap scales the direction below 0.5 can let a few outliers dictate cost. Keep component caps as safety limits while studying a dimensionless norm. A dogleg using the block-preconditioned gradient is a later comparison if LM still wastes retries. Standard trust-region implementations explicitly separate acceptance and radius updates; see [PETSc SNESNEWTONTR](https://petsc.gitlab.io/petsc/release/manualpages/SNES/SNESNEWTONTR/). None guarantees angle-independent outer convergence.

### Stopping: bounded work with an honest unresolved status

Use the interval above for the chosen H+D metric. Set a separate qualification product/time budget and report unresolved if the interval does not decide the threshold. If a lower bound already exceeds the threshold, expensive extra qualification is unnecessary: the state has not qualified. A damped check should remain labelled damped. The present inaccurate Ritz values cannot support a rigorous missing-energy bound.

For production, distinguish valid returned atmosphere, configured stopping outcome, qualification outcome, and total work. A useful budget-limited atmosphere need not be called a converged solution. A final unresolved check should not be disguised as EOS failure or physical invalidity.

### Similar apparent κ but different CG work: plausible, not established here

Even equal true condition numbers do not imply equal convergence: spectral clustering, initial residual weights, stopping norm and finite precision matter. Here the condition numbers are not reliably estimated yet. On the crop symmetric transformed system, measure the weights `|qᵀ L⁻¹b|²` in eigenvectors and their cumulative low-eigenvalue share. On the full cube, monitor true and preconditioned residuals and projections onto a **fixed, normalised** physically interpretable coarse space over iterations. Raw b projections onto nonorthogonal, inaccurately reconstructed Ritz vectors are not trustworthy spectral weights. This diagnostic should follow, rather than delay, the simple damped-block end-to-end trial.

### Gauges: project deliberately and document effects on diagnostics

Projecting out the four cut-height gauge components is preferable to interpreting them as recoverable physics on an even×even grid. Use only the valid modes: constant always, x checkerboard only for even NX, y only for even NY, mixed only when both are even. Verify compatibility `Gᵀb≈0`.

For a constrained solve, use `PAP` and `P M⁻¹ P` on the projected space and project initial residuals/search directions; pinning arbitrary cells would constrain additional combinations. Restrict all coarse spaces consistently. Do not merely project δ at the end and assume it equals a constrained solve. Choose whether to retain initial gauge amplitudes or canonicalise the state; these are different policies.

Φ is invariant, but raw height maps, interpolation and Wilson-depression diagnostics can change under checkerboard shifts. Thus “changes nothing scientifically” is too broad: the data cannot determine these components with this discretisation, and the output convention must be explicit. Report gauge-aligned comparisons, and investigate discretisation changes separately if the patterns contaminate physical outputs.

## Priority handoff

1. Fix qualification edge cases/metadata; validate decrement intervals against the direct crop solve. Check positive finite reused weights, dimensions and matching grid/cut metadata before treating `weights` as a safe shared-objective API.
2. Run the small corrected **block-only damped** outer comparison at μ=1 and the troublesome small-μ case, logging all products, retries, qualification and wall time. Compare balances separately. This can falsify the cheapest proposed remedy.
3. Establish at least one extended, common-objective reference per case; vary tolerance and initialisation. Current 40-step and frozen endpoints remain provisional. Keep height error diagnostic, separate from the user's stopping policy.
4. Validate the spectral extraction on a crop, then compare c-only against a compact energy-extended coarse basis. Advance to horizontal multigrid only if scaling evidence justifies it. Gate full-cube dense bases by memory and measured time.
5. Benchmark corrected D4b with common weights and a clearly labelled parent convergence status, then expand to the full viewing-angle/azimuth suite. Include embedding-path tests: the standalone labs expose newer solver controls and common weights that the public embedding options do not yet expose.

No physical prior or broad new production default is justified by the current evidence. The reported work makes damped block solves a credible next experiment; it has not yet demonstrated the requested accuracy within a μ-independent wall-time budget.
