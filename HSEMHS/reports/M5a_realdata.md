# M5a: real-data preconditioning, λ_F continuation, Bifrost 1 % check (17 Oct 2026)

**Follows** `reports/M6d_step3.md`. **Setup:** the same coarsened periodic Bifrost cube: 42×42×89, 571 km horizontal spacing, Wittmann table, synthetic σ, staggered scheme with face data, λ_Q = 1, `collapse_du = 3`. Scratch scripts: `bif_setup.jl`, `bif_run.jl`, `bif_disc2.jl` (not committed).

**Uncommitted code:**
- `src/MHSStaggered.jl`: `precond_atilde` / `precond_period` / `precond_ufloor`, `gn_diag_colored!`, `discrepancy_solve!`, Φ_Q × λ_F.
- Tests in `test/test_stag_gradient.jl` and `test/test_gateC_staggered.jl`.

## Summary

| Item | Result |
|---|---|
| Preconditioner diagnosis | The Ã layer diagonal was wrong by ~2000× in the lower half (the global `1e-8·max` floor) and ~40× in the upper half (mean ≫ median; a 1e4 spread within a layer). The u diagonal gives \|g_u\|/D_u up to 1.3e4 at low-P nodes. |
| Fixes | Exact per-node Ã GN diagonal by colored probing (period 5: 375 `gn_hvp!` products; exact to 0.0 against single-node products, and period 5 is the smallest exact period). Per-layer u floor (fraction of the layer median). Both are options, default off. |
| Effect on Bifrost | **Small.** Φ_F after 5000 iterations: 3.1e5 (layer), 2.7e5 (layer + u floor), 4.2e5–4.9e5 (colored). **B never moves on the plain path** (χ² = 0.317 throughout). The solve stalls at Φ_F = 1.8e5 by 8000 iterations. |
| **What does work** | **B-first** (Ã relaxed with v frozen, then joint). At λ_F = 1: Φ_F 6.1e5 → 1.07e3 (÷570), χ² 0.317 → 0.395, max layer RMS \|R\|/S 3.6e3 → 148. |
| λ_F continuation | `discrepancy_solve!` follows implementation doc §13.9 (grid 1e-4…1e4, log-bisection to 1.5, upper/lower-bound flags). It stops the grid at the first failing λ, which is equivalent to the full sweep if χ² is monotone in λ_F. |
| **Φ_Q × λ_F (new, needs approval)** | With Φ_Q independent of λ_F, the λ_F = 1e-4 stage made Φ_Q 97 % of Φ: B was bent to kill grid-scale currents and χ² rose to 4.76, so the selection failed at the lower bound. D9b prices Φ_Q "like the force it hides", which only holds if it scales with λ_F. On the synthetic gates λ_F = 1, so nothing changes there; all tests pass. |
| **1 % rule on Bifrost** | **Fails.** With forced B-first, χ² ≤ thresh at every grid λ_F, so λ_F = 1e4 is selected (`lambda_F_at_upper_bound`). **Φ_Q/Φ = 0.93**, and about 0.9 at every λ_F ≥ 1. Φ_S/Φ = 2e-9. |
| Equilibrium (real-data tol 1e-2) | Not met. Per-layer RMS \|R\|/S from top to bottom (every 8th layer): 6, 56, 35, 15, 5.5, 3.2, 2.8, 0.25, 0.07, 0.09, 0.10. The upper 6–7 sampled layers are 98–100 % unidentifiable (δP_unobs > P). Every stage hit `max_iter` (3000). |

## Discrepancy continuation (forced B-first, Φ_Q × λ_F)

| λ_F | status | χ² | Φ_F | Φ_Q |
|---|---|---|---|---|
| 1e-4 | max_iter | 0.319 | 0.115 | 0.64 |
| 1e-2 | max_iter | 0.361 | 4.8 | 38.7 |
| 1 | max_iter | 0.579 | 332 | 3.09e3 |
| 1e2 | max_iter | 0.621 | 2.36e4 | 2.70e5 |
| 1e3 | max_iter | 0.978 | 2.31e5 | 2.50e6 |
| 1e4 | stalled | 0.722 | 1.89e6 | 2.40e7 |

With the policy as approved (B-first only when Φ_Q/Φ_F > 10), every stage takes the plain path, B never moves, and the λ_F ≥ 1 stages stall at iteration 0.

## Interpretation

1. **The real-data obstacle is the coupled solve, not preconditioner scaling.** In the low-β upper layers the residual is the Bifrost Lorentz force, up to ~3e3 ρg. Only B can remove it; the needed change is ~1e-4 of \|B\|, far below σ. The joint L-BFGS direction is dominated by the vacuum direction in u, so α is capped and Ã never moves. Freezing v (B-first) lets Ã do the work.
2. **Φ_Q at λ_Q = 1 is not a small regularizer on real data.** Coarsened Bifrost B has real grid-scale currents that the data supports (χ² stays < 1). Φ_Q prices them about 10× above the residual force, so it dominates Φ and works against equilibrium. The synthetic C3 gates passed at λ_Q = 0.1 with ÷5.9e3; smaller values are untested.

## Decisions requested

1. **Solver policy.** The Φ_Q/Φ_F trigger does not fire on real data, where B-first is essential. Options:
   - **(a) always B-first.** On C1b this reaches the accuracy criteria but ends at `max_iter`.
   - **(b) an adaptive trigger:** B-first whenever the force residual is Lorentz-dominated in some layer, e.g. any layer with median β < 1.
   - **(c) alternate:** Ã-only and joint stages until neither improves.
2. **Φ_Q × λ_F.** Keep it (recommended).
3. **λ_Q for real data.** Lower it until Φ_Q ≤ 1 % of Φ on Bifrost, rechecking C3(b) at that value; or keep λ_Q = 1 for synthetic and flag `regularizer_dominant` on real data.
4. **Iteration budget.** Every Bifrost stage hit 3000 iterations. Real-data convergence probably needs more, or a better solver for the Ã stage (it is an NLFFF-like problem).

## Next

C4–C7, classification and checkpointing remain for M5 proper.

## Update (17 Oct 2026, later): user decisions applied, and what the reruns show

**Applied:**
- Φ_Q × λ_F kept.
- C1b uses the round-off-floor criterion (as C3(b)).
- Real-data convergence test (user): a stage is `converged` when P and B change by ≤ 1e-3 (relative) over 100 iterations: `xtol_rel`, `xtol_window`, config `mhs.xtol_rel = 1e-3`. Measured as max |ΔP/P| and max |ΔB|/max(|B|, σ).
- **B-first trigger.** The user's "median β < 1 (Lorentz-dominated)" is implemented as **median |L|/S > 1 in some layer** (`mhs.lorentz_bfirst`). **Please confirm.** With the β test, the uniform-field C1b/C3 references (β ≈ 1e-4 but L = 0) took B-first. That pushed the column-HSE pressure error into the soft B mode: the reference was 1.27e-9 T off, against 3e-13 T on the plain path. C3 then failed (÷47). Measured directly, the Lorentz test picks plain for C1/C3 references and the ε = 1e-5 checkerboard, B-first for the ε = 1e-3 checkerboard and for Bifrost. Gate C passes in full.

**λ_Q.** The earlier C3 failures at small λ_Q were an artifact of that contaminated reference. With the Lorentz trigger, C3(b) passes at λ_Q = 1e-3, 3e-4 and 1e-4 (B: ÷1.3e5 at ε = 1e-5, ÷1.3e7 at ε = 1e-3; u: ÷2e7). On Bifrost:

| λ_Q | Stopping | Selected λ_F | Φ_Q/Φ at selection |
|---|---|---|---|
| 1 | 3000 it. | 1e4 (upper bound) | 0.93 |
| 1e-3 | 3000 it. | 1e4 (upper bound) | 0.065 |
| 1e-3 | xtol 1e-3, 20 000 it. | 1e4 (upper bound) | 0.118 |
| 1e-4 | xtol 1e-3, 20 000 it. | (running at time of writing) | — |

**Convergence (the user's criterion) is not reached on Bifrost.**
- With xtol 1e-3 and 20 000 iterations per stage, the λ_F ≤ 0.1 stages end at `max_iter`, and the λ_F ≥ 1 joint stages stall after 9–3107 iterations.
- Measured over 100-iteration windows at λ_F = 1e-4: max |ΔP/P| is 9–600, mostly in layers 24–43 (≈1.4–2.3 Mm); every layer changes by > 1e-3. On identifiable nodes it is 0.1–0.6, and max |ΔB|/max(|B|, σ) is 0.04–0.27.
- The pressure at low-P nodes lurches between the collapse bound and much higher values. At small λ_F, P is constrained only through λ_F Φ_F, and a few low-P nodes with large g/D dominate the step.
- **This is a robustness problem of the u sub-problem, not slow convergence.**

**Options for the next step:**
- (a) Block alternation: Ã with v frozen, then u with Ã frozen (a smaller, better-conditioned problem; Gauss–Newton is practical there), repeated until the xtol test holds.
- (b) A per-node trust region on u instead of a single global α, so a few low-P nodes cannot dictate the step.
- (c) Gauss–Newton-CG (`gncg!`) with the exact colored preconditioner for the joint stage.

I recommend (a), with (b) inside the u block.
