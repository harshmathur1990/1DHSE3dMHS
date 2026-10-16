# M6c gate report: Gate C rerun on the staggered scheme (16 Oct 2026)

**Tests:** `test/test_gateC_staggered.jl`. **Helpers:** `sc1_problem`, `layer_residual_stats`, `rms_diff`, `potential_A!` and the face-Lorentz helpers `lor_x/y/z` (all in `src/MHSStaggered.jl`). The criteria are from the D9 addendum §10: E3 is withdrawn, so RMS ≤ 1e-6 and p99 ≤ 1e-4, with no floor.

## Summary

| Gate | Result | Pass |
|---|---|---|
| C1a: uniform T(z), B = 0, from column_hse u | `converged` in 99 / 236 / 483 iterations (NZ = 33 / 65 / 129). RMS = p99 = 3.1e-9 / 5.2e-9 / 9.3e-10. \|u − u_stag\| = 1.2e-8 / 4.4e-8 / 6.2e-9. \|u − u_colhse\| = 0.37 / 0.097 / 0.024 (orders 1.93, 1.98) | **yes** |
| C1b: uniform B = (20 G, 0, 50 G), NZ = 65 | `converged` (2181 iterations). RMS = p99 = 2.6e-7; \|u − u_stag\| = 5.7e-7; \|B − Bobs\| = 1.9e-7 σ | **yes** |
| C2: potential field from an analytic A, ∇·B | ∇·B ≤ 5e-15 (exact) | yes |
| C2: L on interior layers 2…NZ−1 (N = 32 / 64 / 128) | 6.1e-3 / 2.4e-3 / 7.4e-4; orders 1.36, 1.69 | **no** (≥ 1.9 required; may be pre-asymptotic, see below) |
| C2: L on the top and bottom layers | 0.229 / 0.233 / 0.230: **does not converge** | **no** |
| C3(a): the objective sees both checkerboards | yes | yes |
| C3(b): pressure checkerboard (ε = 1e-3 in v) | δu 1e-3 → 1.3e-9 (÷7e5) | **yes** |
| C3(b): Ã checkerboard (ε = 1e-5, linear regime) | δB 6.0e-8 → 7.1e-9 T (÷8.5), reported as `converged` at gtol 1e-10 | **no** |

C2 (both rows) and C3(b) for B are recorded as `@test_broken`. Everything else in the suite passes.

## Findings

**1. Boundary-layer Lorentz force (C2).**
- The addendum (§4) states that the one-sided boundary-edge currents (k = 0, NZ) "only enter the force at boundary faces, which carry no equation". That is not true for the x- and y-faces of layers 1 and NZ.
- Those faces carry equations, and their 2-point J averages use one boundary edge. On that edge J_x and J_y lack the ∂_z B term.
- As a result, L has an O(1) relative error there (0.23 of B₀²k/μ₀), independent of resolution.
- C1/C3 do not see this because ∂_z B = 0 for a uniform field. Real data will.

**2. Interior C2 order.** 1.69 on 64 → 128. The collocated scheme was also pre-asymptotic on this field up to 128 → 256, where it reached 1.9. A 256³ staggered run would settle whether the order reaches 1.9; I have not done it.

**3. The B checkerboard is still nearly unconstrained (C3).** The measurement is against the analytic C1b solution (uniform B, the exact staggered HSE), and the reference solve is accurate to 2e-13 T.
- **Structure of the remaining error:** it projects entirely onto the horizontal checkerboard (−1)^(i+j) in Bx and By, with a Bz partner, at all levels.
- **Mechanism:** the 2-point averages kill this pattern in both places it could be seen.
  - H (data) averages Bx along x and By along y.
  - The Lorentz averages M kill its currents.
  - A matching pressure checkerboard can balance the remainder; δu grows from 1e-10 to 1.4e-7 during the solve.
- **Hessian Rayleigh quotient** along the remaining error is 1.5e-5, against 3.1e2 along a smooth Ã direction. The mode is not an exact null mode, but it is about 2×10⁷ softer.
- **Not a tolerance issue:** 40 000 iterations at gtol 1e-14 do not reduce it (÷13).

So the staggered scheme removed the exact curl null mode. A near-null mode remains in the averaging operators H and M.

**4. Pressure-collapse trap for larger B perturbations.**
- An Ã checkerboard with ε = 1e-3 (δB ≈ 4e-3 σ, about 2e-6 T) gives Φ₀ ≈ 3e4. The reason is low β at the top: P_top ≈ 1e-3 Pa and β ≈ 1e-4, so the Lorentz term dwarfs S = ρg.
- L-BFGS then drives u to vacuum: δu ≈ 100–500 in ln P above the bottom layer. It stops at Φ = 3.4e-2, a stationary point because ∂R/∂u ∝ P → 0, instead of the equilibrium at Φ ≈ 1e-15.
- The same trap appeared in M4a. B-first continuation does not prevent it here (its first stage converges in 2 iterations, then the joint stage collapses).

**5. B-first continuation is counterproductive for C1b on this scheme.**
- With v frozen at the column_hse pressure, the Ã stage tries to balance the pressure error with currents. It hits its 500-iteration cap, and the joint stage then ends `stalled`.
- A plain joint solve from A₀ converges.
- So the C1 tests use the plain joint solve.

## Remedies tried (options behind default-off flags, FD-verified ≤ 1.5e-10)

| Remedy | C3 B decay (ε = 1e-5, gtol 1e-10) | Vacuum trap at ε = 1e-3 |
|---|---|---|
| None | ÷15 (÷8.5 vs the analytic solution) | yes |
| `lambda_N = 0.01`: penalize ¼δ²B along each face normal (the part H averages away) | ÷15 | yes |
| `lambda_N = 1` | ÷36 (8392 iterations) | yes |
| `data_on_faces = true`: data term on faces against Bobs interpolated to faces | ÷15 | yes (`stalled`) |
| Field ×0.1 (β ×100), as a diagnostic | ÷3.9 (worse) | milder (δu 2.3) |

None of the cheap remedies works. The near-null mode survives even when the data are compared on faces. That points to the coupled (δB Nyquist, δP Nyquist) force-balanced pair rather than to H alone, and to the diagonal-preconditioned L-BFGS being unable to resolve a 10⁷ stiffness ratio.

## Options (decision needed)

1. **Boundary-layer J (finding 1). Two candidate fixes:**
   - (a) Drop the horizontal force equations on layers 1 and NZ, consistent with "no equations on boundary faces". This is simplest; those layers are then constrained by the data and the vertical equations only.
   - (b) Use a linearly extrapolated ghost tangential B for the boundary-edge J. This keeps the equations, with a first-order error confined to those layers.

   I recommend (a).
2. **C3 B mode (findings 3–4).** Options in order of preference:
   - (i) **Solver change, not discretization.** Gauss–Newton with an inner preconditioned CG on the full (v, Ã) system. Nonlinear least squares with ~10⁷ stiffness ratios is GN-CG territory. Add a pressure-collapse safeguard: reject steps that lower u by more than a cap below u₀, i.e. a trust region on u.
   - (ii) **Discretization change.** Replace the 4-point B averages and 2-point J averages in M with a form whose transpose does not annihilate the (−1)^(i+j) pattern, for example the energy-conserving Yee interpolation through cell-centred B and J. This is a larger redesign of addendum §4 and needs its own short addendum.
   - (iii) **Relax the C3(b) B criterion** to a decay of ≥ 10 and rely on M5's discrepancy principle to flag affected regions. This is not recommended, because the mode is grid-scale and would show up in real-data force residuals.

## Code status (uncommitted)

- **`src/MHSStaggered.jl`:**
  - the face Lorentz helpers `lor_x/y/z`, used by the objective
  - the options `lambda_N` and `data_on_faces` (default off; term slot 2 now reports Φ_N)
  - the Gate C helpers
- **`test/test_gateC_staggered.jl`:** the new tests above.
- **`test/test_stag_gradient.jl`:** FD tests for both new terms.
