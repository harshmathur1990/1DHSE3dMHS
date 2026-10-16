# M4a gate report: MHS residual, objective, adjoint gradient, Gate C1–C3 (16 Oct 2026)

Code: `src/MHSResidual.jl` (objective, adjoint, Gauss–Newton diagonal), `src/MHSOptimize.jl` (L-BFGS, strong-Wolfe line search, Jacobi preconditioning, step cap), `src/Synthetic.jl`. Tests: `test/test_gradient.jl`, `test/test_gateC.jl`.

## Gradient (doc §13.6): pass

| Check | Result |
|---|---|
| Transposes of all six stencil operators | ≤ 1e-13 (M0 tests, 1–6 ranks) |
| Directional FD vs adjoint, all fields, 16×14×15, ideal gas / Wittmann table | 2.7e-10 / 3.1e-10 |
| Per field: b̃x, b̃y, b̃z | ≤ 1.9e-10 |
| v alone, weak-field state (\*) | 3.5e-9 / 5.5e-9 |
| Per term alone: force, divergence, regularizer on δB, regularizer on u, data | ≤ 3.1e-10 |
| Thread invariance (1 vs 4 threads) | bitwise |
| Rank invariance (1, 2, 4 ranks) | ≤ 1e-12 relative |

(\*) On the strong-field test state, Φ ≈ 1.7e11 is dominated by Lorentz terms, so the v-only derivative (~2e3) sits below the FD round-off floor.

## Gate C

| ID | Result | Pass |
|---|---|---|
| C1a (uniform T(z), B = 0) | Residual within the E3 floor rule at NZ = 33/65/129 (RMS/floor 0.92 / 0.57 / 0.33); floor converges at order 2.0; \|u − u_exact\| = 1.40 / 0.41 / 0.11 (orders 1.76, 1.93); max \|b̃\| = 0 | yes |
| C1b (uniform B = 20 G, 50 G) | Floor rule met; discrete L of a uniform field ≤ 1e-12; converges in 331 iterations with preconditioning (none: no convergence in 3000) | yes |
| C2 (potential field) | max\|L\| and max\|∇·B\| fall monotonically; orders 1.90 and 1.95 on 128 → 256 (pre-asymptotic below 64, where the field varies by e⁷ over the box) | yes (last-pair gate) |
| C3 (a): objective penalizes checkerboards | Φ and the regularizer terms increase | yes |
| C3 (b): modes decay by ≥ 10³ | With B = 0: u mode down to 1.1–3.0e-3 of its initial amplitude, b̃ mode to 0.5–0.9e-3 (marginal). With the C1b background field: b̃ mode down to 2.5e-3; the u mode **grows** ×630. | **no** |

## Why C3 (b) fails

1. **The modes are penalized.** The curvature of Φ is positive along every checkerboard tested. But the spread is enormous. With B₀ = (20, 0, 50) G, curvatures are 5.8e6 (b̃ checkerboard with a z component), 2.1e3 (smooth b̃), 2.2e-3 (u, xyz), 2.6e-5 (smooth u) and **3.8e-6 (u, horizontal-only)**. That is about 1e12 in total. The horizontal pressure odd–even mode is felt only through λ_U = 1e-6; with B = 0 its curvature is 5e-8.
2. **The top of the synthetic atmosphere has β ≈ 3e-5.** A 1e-3 σ checkerboard in B there gives Lorentz forces ~10³ × S. The descent direction then builds an alternating pressure checkerboard (ln P swinging by ±15) to balance them, instead of only relaxing B. Since ∂Φ/∂u ∝ P, the gradient vanishes once P has collapsed, and the relative gradient test reports "converged" at Φ = 0.07 (equilibrium: 1e-4).
3. **The safeguards added in M4a help but don't fix it.** The Gauss–Newton Jacobi preconditioner gives a large speedup on C1b. A step cap (|Δ ln P| ≤ 0.5, |Δb̃| ≤ 5 per iteration) does not prevent the drift, and tighter caps stall B because L-BFGS uses one step length for all variables.

So the failure combines optimizer conditioning (extreme low β, log-pressure variable) with one genuinely weak mode (horizontal pressure checkerboard, controlled only by λ_U). It does not show a spurious equilibrium with lower Φ. Per D9, though, a C3 failure triggers the staggered-B / vector-potential fallback, and that is the user's call.

## Follow-up: cheap remedies (`reports/M4a_c3_remedies.md`)

- **B-first continuation** fixes the collapsed-pressure trap; the u checkerboard then decays by ≥ 10³ at the default λ_U. Kept as the default solve path.
- **Raising λ_U** does not avoid the trap on its own and is not needed with continuation.
- **Raising λ_S** barely helps and breaks the 1 % rule from λ_S = 0.1.
- **The b̃ checkerboard does not decay** (b̃z keeps 30 %). It is a null mode of the collocated curl, held only by the data and regularizer terms (conditioning ~1e11). It also turns C1b's status into `stalled`.
- **Recommendation:** invoke D9 (vector potential / staggered B) and write a design addendum.
