# M5b: Codex-review fixes and the frozen-B pressure experiment (17 Oct 2026)

**Follows** `CODEX_REVIEW_M5.md` and `CLAUDE_RESPONSE_CODEX_M5.md`. The user approved steps 1–4 (constraint and termination fixes, exact anchor diagonal, frozen-B experiment); the χ² calibration (step 5) is deferred.

**Code:**
- `src/MHSOptimize.jl`: `anchor_box_project!`, used by the L-BFGS safeguard.
- `src/MHSAlternate.jl`: the trust-region u block, `pressure_optimality`, block-aware outer test.
- `src/MHSStaggered.jl`: `gn_diag_v!`; `gauge` keyword on `gradient!`.
- `src/MHSGaussNewton.jl`: `gauge` keyword on `gn_hvp!`.

**Scripts:** `scripts/m5_bifrost/frozenB_checkpoint.jl`, `frozenB_branch.jl`.

## 1. Fixes

| Codex finding | Fix | Verification |
|---|---|---|
| 1 Bound + anchor not enforced together | `anchor_box_project!`: off-anchor nodes are clamped; anchor-plane nodes get clamp(t − cβ) with the scalar c found by bisection, so Σα δ = 0 exactly. Used for the u-block step, the u-block optimality measure, and the L-BFGS direction (nodes at the bound cannot move down). Exact c = 0 when no shift is needed (keeps inactive projections bit-identical to before). | Codex's two counterexamples: margin ≥ −1e-12 and anchor residual ≤ 1e-12 (new test). Bifrost: min margin −3e-13, anchor error ≤ 4e-18 throughout. |
| 2 Zero work reported as convergence | Outer `converged` needs the change test **and** an Ã block that did work (or converged) **and** a u block at constrained optimality ≤ `opt_tol`. Each block reports status, accepted steps, optimality. | Zero-budget cycle is no longer `converged` (test). |
| 3 u diagonal not exact on the anchor plane | `gn_diag_v!` = diag(GᵀH_uG): D_i − 2α_i(H_u1)_i + α_i²(1ᵀH_u1), with H_u1 from one gauge-free GN product. | Equals single-node GN products on and off the anchor to 1e-12 (test). |
| (Q2) Clip-and-test is not a trust-region method | The u block now: ε-active set, Steihaug–Toint PCG on the free set in the D-scaled norm, then a projected arc search δ(t) = Proj(t y), accepted on the actual/predicted ratio, with the radius adapted from ρ. Coordinates ln P or p = P/P_ref. | See §2. |

**Two intermediate failures, fixed:**
- Clipping a full CG step node by node was not a descent step. The trust radius collapsed (`failed` after 37–56 steps).
- Truncating CG at the first bound hit let nodes creep toward the bound geometrically. Steps shrank about 50× per update.

The GN model itself was verified accurate along the steepest-descent and random pressure directions: actual/predicted ratio 1.000 down to t = 1e-6, and 1.002–1.006 at max |δu| = 0.1.

## 2. Frozen-B experiment (Codex review, "One next experiment")

**Setup:**
- Checkpoint: one alternating cycle at λ_F = 1e-4, λ_Q = 1e-3 (300 Ã iterations, 5 pressure steps), on the coarsened Bifrost cube (42×42×89).
- B frozen; the same S₀, collapse bound (u_init − 3), EOS, anchor and residuals.
- Each branch: up to 500 accepted updates in chunks of 100 (the trust radius restarts per chunk), PCG rtol 1e-4, ≤ 300 iterations, per-node cap 5 in ln P per step.
- Fixed identifiability mask from the checkpoint (δP_unobs ≤ 0.1 P): 36 % of nodes overall. By layer, every 8th from the top: 0, 0, 0, 0, 0, 0, 0.08, 0.45, 0.96, 1, 1, 1.

| Window (100 accepted) | ln P: raw force | ln P: opt | ln P: max ΔP/P (identifiable / complement) | p: raw force | p: opt | p: max ΔP/P (identifiable / complement) |
|---|---|---|---|---|---|---|
| start | 8916 | 2e3 | — | 8916 | 2e3 | — |
| 1 | 6823 | 408 | 270 / 1.3e4 | 6435 | 4.8e3 | 237 / 1.4e4 |
| 2 | 6371 | 5.5e3 | 3.2 / 2.5e3 | 6398 | 605 | 0.68 / 494 |
| 3 | 6329 | 389 | 0.89 / 1.1e3 | 6368 | 2.4e3 | 5.5 / 376 |
| 4 | 6319 | 2.2e3 | 0.19 / 222 | 6344 | 772 | 0.76 / 116 |
| 5 | 6310 | 455 | 0.068 / 288 | 6331 | 84 | 0.14 / 176 |

**Other quantities:**
- Bound margin ≥ −3e-13 and anchor error ≤ 4e-18 in both branches: the pass criteria on bound and anchor are met.
- Every step was accepted (101–102 tries per 100).
- The PCG relative residual is 0.2–1.6: CG stops on the D-norm trust boundary after few iterations.
- Steps reach the per-node cap (5 in ln P).

**Branch comparison at 500 updates:**
- The raw force objectives are 6310 (ln P) and 6331 (p): 0.3 % apart, more than the 1e-3 in Codex's criterion.
- Pressures differ by |Δ ln P| of 0.019 (median), 0.34 (p99) and 2.1 (max) on identifiable nodes; 0.026 / 0.62 / 9.4 on the complement.
- Change from the checkpoint is nearly identical in both branches: identifiable median 0.67 in ln P (a factor 2), max 6 (×400); complement median 1.4, max +9.7 (×16 000).

## 3. Verdict against Codex's criteria

**Neither branch passes.**
- Optimality is 84–455, against a pass value of ≤ 1e-6.
- The change criterion (≤ 1e-3 for three consecutive windows) is not met: the last window still moves 0.07–0.14 on identifiable nodes and 180–290 on the complement.

By Codex's interpretation table this is "**neither passes: conditional convergence is unresolved; this does not establish physical infeasibility**".

**What the run does show:**
1. **The u block now works as an optimizer.** Steps are accepted with ρ ≈ 1, the bound and anchor hold, and the force objective falls monotonically (8916 → ≈6300). The earlier failures were in the step construction, as Codex said, not in the model.
2. **Both coordinates head for the same conditional optimum**: the same force plateau within 0.3 %, and pressures within 2 % (median) on identifiable nodes. Nothing so far indicates a coordinate effect; the scaled-p branch reached lower optimality in its last window, but not consistently.
3. **That optimum is extreme.** With B frozen (and not force-balanced), the force term drives the pressure up by factors of 10²–10⁴, mostly where the mask flags pressure as poorly constrained, until pressure gradients carry the unbalanced Lorentz force. Convergence is slow because those nodes need changes of many e-folds, and the per-step cap (5 in ln P) and trust radius limit them.
4. The force residual at this frozen B stays large (raw 6.3e3, from 8.9e3). By Codex's last row, that is a floor **at this B only**, not over data-admissible B.

## 4. Open items and suggestions

- **Whether the conditional problem converges with more budget** (or with a larger per-node cap): run each branch to 2000 updates. That is cheap: about 4 min per 500 updates.
- **The conditional optimum at an unbalanced B is not a meaningful pressure estimate.** It argues for keeping the Ã and u blocks coupled, as the alternation does, rather than solving pressure to optimality at a fixed B. That is a judgement for the user and Codex.
- The repeatability check from ±1 % perturbations (Codex step 3) is not done yet. The branch script supports it (third argument).
- χ² calibration (deferred step 5).
