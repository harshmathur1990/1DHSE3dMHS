# M6b gate report: staggered objective, adjoint, preconditioner and initializers (16 Oct 2026)

**Code:** `src/MHSStaggered.jl` (new), plus the following changes:
- `src/MHSOptimize.jl`: L-BFGS is generic over state layout and problem type.
- `src/Parallel.jl`: corner halos are now exchanged.

**Tests:** `test/test_stag_gradient.jl`, `test/scripts/sgrad_ranks.jl`. **Design:** `HSE_MHS_D9_STAGGERED_ADDENDUM.md` §5–8.

## What is implemented

- **State.** x = (v, Ãx, Ãy, Ãz, b̄).
  - u = v − v̄ + ln P_ref at the centres.
  - A = A_ref·Ã on the edges, with A_ref = median σ × mean horizontal spacing.
  - b̄ is a distributed scalar stored on rank 0 only.
- **Forward model.**
  - P = eᵘ, and ρ from the EOS.
  - B = curl A + b̄ẑ on the faces, and J = C\*B/μ₀ on the edges.
  - The Lorentz force on each face comes from the 2-point averages of J and the 4-point averages of B named in the addendum.
  - The force residual is evaluated on x-faces, y-faces and the NZ−1 interior z-faces, with g√(ρ_k ρ_{k+1}) on the z-faces. Its weights are w_f/S_aS_b, normalized by N_F = NX·NY·(3NZ−1).
  - Data term on HB at the centres, normalized by N_B = 3N.
  - λ_S term on δB = HB − Bobs.
  - λ_U term (default 0).
- **Adjoint.** Hand-written, in gather form:
  - Gᵀ and the ρ̄ chain for u.
  - The transposes of the face averages, gathered back to the edges (J) and the faces (B).
  - C\*ᵀ/μ₀, Hᵀ, then Cᵀ, with Σ g_Bz giving ∂Φ/∂b̄.
  - The gauge chain for v.
- **Preconditioner (`gn_diagonal!`).**
  - u: exact Gauss–Newton diagonal.
  - Ã: per-layer Hutchinson mean of the Hessian–vector products (4 Rademacher probes, deterministic hash).
  - b̄: exact second derivative.
- **Vertical initializer (`staggered_hse!`).**
  - Exact discrete balance on the z-faces for B = 0, bottom-anchored.
  - Each step is solved by Brent's method.
  - The bracket is found by stepping down from u_{k+1} until F < 0, which keeps it inside the EOS envelope (a fixed [u − 60, u] bracket failed for every step with the Wittmann table).
- **A₀ (`init_A!`).**
  - With v frozen and λ_F = λ_U = 0, Φ is quadratic in (Ã, b̄).
  - It is solved by CG on the normal equations, with Qp = ∇Φ(p) − ∇Φ(0) (one gradient per iteration).
  - Starting from 0, CG returns the minimum-norm solution in the curl gauge null space.

## Results

| Check | Ideal gas | Wittmann table | Pass |
|---|---|---|---|
| FD vs adjoint, all fields | 2.0e-10 | 9.2e-11 | yes (≤ 1e-6) |
| FD, v / Ãx / Ãy / Ãz / b̄ alone | 2.4e-7 / 3.4e-10 / 2.7e-10 / 2.2e-11 / 9.8e-14 | 6.4e-8 / 2.0e-10 / 2.3e-10 / 8.5e-11 / 2.6e-14 | yes |
| FD, each term alone: force / reg δB / reg u / data | 2.0e-10 / 5.1e-13 / 6.5e-13 / 5.8e-13 | 6.8e-11 / 5.1e-13 / 5.2e-13 / 5.8e-13 | yes |
| Vertical initializer, B = 0: max \|R_z\|/(gρ̄) | 8.6e-15 | — | yes (≤ 1e-10) |
| GN diagonal strictly positive on every node | yes | — | yes |
| A₀ CG, representable Bobs = H(curl A\* + b̄\*): Φ_B 0.208 → 1.5e-9 (128 iterations, ‖r‖/‖r₀‖ = 7.7e-11) | 7.1e-9 relative | — | yes (≤ 1e-7) |
| A₀ CG, generic smooth Bobs (not solenoidal at centres): Φ_B 12.4 → 0.181 (101 iterations) | least-squares floor, 1.5 % | — | yes (converged, rtol 1e-10) |
| Φ, ⟨∇Φ, p⟩, ‖∇Φ‖² on 1×1, 1×4 threads, 2×2, 4×1 ranks×threads | 1×4 bitwise equal to 1×1; ranks ≤ 1e-12 (measured ≈ 2e-15) | — | yes |
| Allocations per objective + gradient (16×14×15) | 0.39 MB | — | yes (< 2 MB) |
| Full suite (M0–M6b) | all pass; collocated C3(b) is still `@test_broken` | — | yes |

The FD error for v alone (2e-7) is at the round-off floor of this strong-field test state, where Φ ≈ 3e8 is dominated by Lorentz terms. The collocated M4a test has the same floor.

The A₀ fit stops at Φ_B/Φ_B0 ≈ 7e-9 rather than at machine precision for two reasons:
- Φ_B scales as the error squared.
- κ(Q) is large: H averaging has weakly constrained checkerboard combinations of B.

That residual is ~1e-4 rms in δB, far below σ, so it is adequate for an initial state. If needed, a tighter rtol or a Jacobi-preconditioned CG would reduce it.

## Bug found and fixed: corner halos

`exchange_halos!` filled the x halos and then the y halos for i ∈ 1:nx only, so the diagonal corners (0/nx+1, 0/ny+1) were never filled. The collocated stencils never read corners. The staggered 4-point face averages do read them, for example By at (i+1, j−1) on an x-face.

The adjoint was consistent with the forward model, so every FD test passed. The bug showed up only in the rank-invariance test: Φ_F differed by 9e-6 between 1, 2 and 4 ranks, because on a single rank the corners held zeros.

The fix: the y pass now carries i ∈ 0:nx+1, so the corners arrive with it. `halo_errors` now also checks the corners. The collocated results are unchanged.

## Next (M6c)

- Rerun Gate C1–C3 on the staggered scheme with the preconditioned L-BFGS and B-first continuation, starting from `staggered_hse!` + `init_A!`.
- C3(b) is the decisive check that the B checkerboard null mode is gone.
