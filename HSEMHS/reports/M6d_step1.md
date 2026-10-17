# M6d step 1: Φ_Q gate experiment (17 Oct 2026)

**Follows** `HSE_MHS_D9b_AVERAGING_ADDENDUM.md` §5 step 1. **Code:** Φ_Q forward and adjoint in `src/MHSStaggered.jl` (`lambda_Q`, default 0); `gn_hvp!` refuses λ_Q > 0 until step 2. **Setup:** C3 of `test/test_gateC_staggered.jl` (16×16×33, uniform B = (20, 0, 50) G), `data_on_faces = true`, L-BFGS with the GN-diagonal preconditioner (the per-layer Hutchinson Ã diagonal includes Φ_Q automatically), gtol 1e-10, max 8000 iterations. Scripts: scratch `c3_lamQ.jl`, `c3_diag.jl` (not committed).

## Verdict

**Go for Φ_Q** (linear regime). The B checkerboard at ε = 1e-5 decays ≥ 10³ with status `converged` for λ_Q ∈ {0.1, 0.3, 1}. The ε = 1e-3 row fails for a different reason: the pressure-collapse (vacuum) trap of M6c finding 4, not the H/M near-null. That needs a solver safeguard (decision below).

## 1. Verification

| Check | Result |
|---|---|
| FD gradient, Φ_Q alone (all fields; Ã), ideal gas / table | 5e-15, 7e-15 / 3e-14, 3e-14 (Φ_Q ≈ 1.3e8 at the test state, so the term is exercised) |
| FD gradient, all terms + Φ_Q, ideal gas / table | 4.6e-11 / 3.2e-10 |
| Rank/thread layouts (1×1, 1×4, 2×2, 4×1), λ_Q = 1, face data | bitwise across threads; ≤ 3e-15 relative across ranks |

## 2. Implementation correction found by the experiment

The first prototype restricted the J_z terms to layers 2…NZ−1 and the x/y-edge horizontal differences to edge levels 1…NZ−1. The addendum excludes boundary edges only for the **z** second difference. With those extra exclusions the experiment gave ÷154 / ÷47 / ÷33 at λ_Q = 0.1 / 1 / 10. A parity/layer decomposition showed why: the interior cleared to 1e-11, but the (−1)^(i+j) mode survived in the top layer (6.9e-8 T). Under option (a) no force equation sees the top-layer J_z or the boundary-edge currents, so Φ_Q is the only grid-scale control there.

**Now:** horizontal second differences on every edge level, J_z on all layers. The z second difference is still interior only (2 ≤ k ≤ NZ−2 for x/y edges). Everything below uses this version.

## 3. C3 results (face data)

B error relative to the analytic C1b solution. "Ref" is the tightened C1b reference solve.

| λ_Q | Ref: \|u−u_s\|, \|B−B_ex\| | Φ_Q/Φ at ref | ε = 1e-5 (Ã) | ε = 1e-3 (v + Ã) | ε = 1e-3 (Ã only) |
|---|---|---|---|---|---|
| 0 | 8.4e-7, 4.6e-13 T | 0 | ÷91, max_iter | ÷96, max_iter | — |
| 0.1 | 7.6e-9, 9.4e-12 T | 2e-4 | **÷5.9e3, converged** | ÷570, max_iter, δu 1.7e-5 | ÷201, stalled, **δu = 34** |
| 0.3 | 9.7e-9, 1.1e-11 T | 6e-4 | **÷2.3e3, converged** | ÷3.5, **δu = 3.3** | ÷3.4, **δu = 3.2** |
| 1 | 2.8e-9, 2.7e-12 T | 2.8e-3 | **÷2.2e3, converged** | ÷134, **δu = 42** | ÷136, **δu = 45** |
| 10 | 8.9e-8, 1.1e-10 T | 1.3e-2 | ÷316, converged | ÷135, **δu = 28** | — |

δu is the maximum ln P deviation from the reference after the solve. δu ≫ 1 means the top layers collapsed toward vacuum (Φ stationary because ∂R/∂u ∝ P → 0), while L-BFGS still reports `converged`.

**Observations**
- Φ_Q does what §1.1 predicted: it makes the mode force-stiff. At λ_Q = 1 the post-solve error is down to 3.7e-9 T, mostly a horizontally uniform δBx in the lower layers, and no longer the checkerboard.
- λ_Q = 10 is too stiff: conditioning limits the decay to ÷316, and Φ_Q exceeds the 1 % rule at the reference.
- With Φ_Q the references converge more tightly than at λ_Q = 0 (\|u − u_s\| 3e-9 vs 8e-7). With face data the top additive-pressure stall of M6c_gn does not occur in this setup.
- **ε = 1e-3:** every λ_Q > 0 eventually enters the vacuum trap. λ_Q = 0 does not, but it also fails the decay (÷96). The trap is a globalization failure: a large initial checkerboard current in the low-β top layers drives u downward. It is not a stiffness problem.
- The Rayleigh-quotient criterion of §4 is moot where the decay passes: the remaining error is at round-off scale relative to the smooth direction (1e-14).

## 4. Decisions requested

1. **λ_Q default.** The data support λ_Q ∈ [0.1, 1]. I recommend keeping the addendum's λ_Q = 1, which passes the linear gate and the 1 % rule at the reference. The C2 and Bifrost 1 % checks in step 3 may lower it.
2. **The ε = 1e-3 row of C3(b).** Options:
   - (a) Add the `gncg!` pressure-collapse safeguard to L-BFGS: reject or shorten any line-search step that lowers u more than a cap (e.g. 3) below the start value anywhere. This is cheap, and it matches what `gncg!` already does. **Recommended.**
   - (b) Continuation in λ_Q (0 → 1) or in ε for large perturbations.
   - (c) Gate the B decay at ε = 1e-5 only, and treat ε = 1e-3 as a robustness check (no collapse, decay ≥ 10).
3. **Proceed to step 2** (Φ_Q in GN seeds, λ_Uz, the dP_unobs diagnostic, `data_on_faces` default on, config keys), with the boundary coverage of §2 above recorded as an addendum clarification.
