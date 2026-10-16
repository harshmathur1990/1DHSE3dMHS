# M6c follow-up: boundary-layer fix (a), Gauss–Newton-CG, C3 rerun (16 Oct 2026)

**Follows** `reports/M6c_gate.md`. **Code:** `horeq` in `src/MHSStaggered.jl`, and `src/MHSGaussNewton.jl` (new). **Tests:** `test/test_stag_gradient.jl` (GN product) and `test/test_gateC_staggered.jl`.

## 1. Boundary-layer fix, option (a)

- The x- and y-face force equations now exist on layers 2…NZ−1 only (`horeq`), so N_F = NX·NY·(3NZ − 5). The GN diagonal and the layer statistics follow.
- **Effect on C3.**
  - The Ã checkerboard (ε = 1e-3) now costs Φ₀ = 2.9 instead of 3.4e4. Almost all of the old cost came from the low-β top layer's inconsistent boundary-edge current.
  - **The vacuum trap is gone:** no run below collapsed the pressure.
- **Side effect.** In the 16×16×33 C1b reference solve, L-BFGS now stalls (`stalled`, 900 iterations) with δu ≈ −4.8e-4, +3.3e-4, −2.2e-4 in the top three layers and ≈ 0 below.
  - The error is horizontally uniform and grows upward like 1/P: an almost additive pressure constant, which the low-P top barely constrains.
  - Before (a), the same solve converged to 1.2e-10.
  - **Consequence:** the C3 pressure-checkerboard check, which is measured against the analytic solution, now fails (marked `@test_broken`). The checkerboard itself still decays.
  - C1a (all NZ) and C1b at 8×8×65 still pass and report `converged`.

## 2. Gauss–Newton / Levenberg–Marquardt with inner PCG (`gncg!`)

- **Exact GN product** H_GN v = Jᵀ W J v:
  - J v is the analytic linearization (δu, δB = C(A_ref δÃ) + δb̄ ẑ, δJ, δP = Pδu, δρ̄, and δL = M(δJ)×M(B) + M(J)×M(δB)).
  - Jᵀ W is the existing adjoint, driven by seeds built from J v.
- **Verification:**

  | Check | Result |
  |---|---|
  | Quadratic terms (data and regularizers): H_GN v vs Δ∇Φ/ε | 7e-13 (centre data), 7e-13 (face data) |
  | Symmetry ⟨c, H a⟩ vs ⟨a, H c⟩ | 6e-16 |
  | Linearized force seeds vs FD of the nonlinear seeds | ≤ 6e-6 relative |
  | vᵀ H_GN v vs d²Φ/dε² at the C1b equilibrium | matches (5.4644) |

- **Solver.**
  - Each step solves (H_GN + μD)s = −∇Φ by PCG, with preconditioner (1+μ)D (the GN diagonal) and η = 1e-3, max 300 inner iterations.
  - Steps are capped at |Δu| ≤ 0.5.
  - **Pressure-collapse safeguard:** a step is rejected if u drops more than 3 below its starting value anywhere.
  - Acceptance uses ρ = actual/predicted > 1e-4, and μ adapts.

## 3. C3 rerun after (a): B checkerboard decay against the analytic solution

| Solver, data term | ε = 1e-5 Ã | ε = 1e-3 (v and Ã) | Status / cost |
|---|---|---|---|
| L-BFGS, centre data (H) | ÷1.6 | ÷1.6 | stalled / converged at gtol 1e-10 |
| L-BFGS, face data | **÷160** | **÷120** | max_iter (8000), ~95 s |
| GN-CG, centre data | ÷1.6 | ÷1.6 | stalled (step rejection), 278 outer / 67k inner, 440 s |
| GN-CG, face data | ÷26 | ÷11 | max_outer 100 (30k inner), 198 s |

None reaches the ÷10³ gate.

- **GN-CG does not help.** Its inner PCG cannot resolve the soft directions any better than L-BFGS, and it costs far more per unit of progress.
- The soft mode is in the GN Hessian itself (Rayleigh quotient ~1e-7 relative to the stiff directions), not a weakness of the solver.
- **The only change that helps materially is comparing the data on faces** (×100 better). That confirms H's averaging as a main contributor; the Lorentz averages M are the rest.

## Status

- **Full suite:** everything passes except the `@test_broken` entries:
  - collocated C3(b) for B
  - staggered C2 interior order
  - staggered C3(b) for B
  - staggered C3(b) for u (new; side effect of (a))
- **Kept, default-off and verified:** `gncg!`, `lambda_N`, `data_on_faces`.
- **Nothing is committed** after `3233d47` (M6c).
