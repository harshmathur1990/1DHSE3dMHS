# M6d step 3: 1 % rule and a first real-data solve (17 Oct 2026)

**Follows** `reports/M6d_step2.md` (committed as `64fb278`). **Setup:** Bifrost en024048_hion snapshot 385 (`IO/input-mhs-hse.nc`), block-averaged 12×12 horizontally (periodic 42×42, 571 km), every 2nd level (89 levels, 2.98 to −0.49 Mm). Wittmann table, P_ref = 1.035e5 Pa at the bottom, synthetic σ (5 % of |B| ⊕ 5 G), staggered scheme with the D9b defaults, λ_F = 1, `mhs_solve!`. Script: scratch `bifrost2.jl` (not committed).

## Verdict

**The 1 % rule cannot be evaluated on Bifrost yet: the real-data solve does not converge.** Along the way it exposed and fixed a bug in the step-2 collapse safeguard. The synthetic 1 % checks are degenerate: on the noise-free C1/C3 solutions Φ → 1e-18, so any term's fraction is round-off. The C1b value of Φ_Q/Φ = 3e-3 is meaningless for the same reason.

## 1. Safeguard bug fixed (two parts)

Before the fix, the solve stopped `stalled` (ftol window) after 200 iterations, with the last iterations accepting α = 0.

- **Rounding at the bound.** A node 1.8e-14 below the bound was projected (d = 0). The gauge re-centring then left it a round-off Δu < 0, and `step_to_lower` returned 0. Fix: `step_to_lower` skips nodes within `tol` of the bound.
- **Tolerance.** `project_lower!` now projects nodes within tol = 1e-3·`max_du` of the bound, instead of 1e-12. Every remaining breakpoint is then ≥ 1e-3 of the |Δu| step cap.

After the fix the solve runs to max_iter with α ≈ 1e-3 throughout. Gate C and the full suite were rerun (results in §4).

## 2. Bifrost solve (λ_Q = 1, λ_F = 1, 5000 iterations)

| Quantity | Start | After 5000 iterations |
|---|---|---|
| Φ_F | 6.09e5 | 3.09e5 |
| Φ_Q (slot 2) | 8.13e3 | 8.15e3 |
| Φ_B (χ²/N_B) | 0.317 | 0.317 |
| max layer RMS / p99 of \|R\|/S | — | 3.6e3 / 1.4e4 |
| nodes with Δu < −2.5 (bound at −3) | — | 146k of 157k |
| ‖∇Φ‖/‖∇Φ₀‖ | 1 | 0.46 |

Status `max_iter`. Runs with λ_Q = 0, with the safeguard off, and with λ_F = 1e-2 show the same picture: B never moves, and Φ_Q and Φ_B are constant to three digits. Without the safeguard, u falls by up to 14.

**Observations** (diagnosis, not yet tested as fixes):
- **Ã does not move** although its gradient dominates (‖∇_Ã Φ‖ ≈ 5e7 vs ‖∇_v Φ‖ ≈ 3e2). The search direction has \|Δu\| up to 560 per unit α but \|Δb̃\| ≈ 0.1, so the \|Δu\| ≤ 0.5 cap limits α to ~1e-3 and B barely changes. This points to the preconditioner scaling:
  - The u diagonal spans 3.6e-8 … 3.4e3. Low-diagonal nodes get huge D⁻¹∇Φ steps.
  - The per-layer Hutchinson Ã diagonal is large, so Ã steps are tiny.
- **u slides toward vacuum almost everywhere** (to the bound). Most of Φ_F comes from Lorentz forces in the low-β upper layers, where |R|/S ~ 1e3–1e4. The top 48 of 89 layers are > 50 % unidentifiable (δP_unobs > P).
- The spec's real-data path is the λ_F discrepancy continuation from λ_F = 1e-4 (§6.4, M5), with real-data tolerances 1e-2 / 1e-1. One λ_F = 1e-2 run shows the same stall, so continuation alone will not fix it.

## 3. Recommended next step (M5)

Real-data robustness belongs to M5. It needs, in order:
1. **u-preconditioner floor per layer** (e.g. a fraction of the layer median, not 1e-12 of the global maximum), and a check that the Ã Hutchinson diagonal is not dominated by a few stiff nodes.
2. **λ_F continuation per §6.4.**
3. Then **rerun this Bifrost check** for the 1 % rule (Φ_Q, Φ_S) and for whether Φ_Q helps or hurts on real data.

## 4. Status

- **Code:** the safeguard fix in `src/MHSOptimize.jl` (`project_lower!` / `step_to_lower` take `tol`).
- **Tests:** Gate C staggered and the full suite rerun after the fix.
