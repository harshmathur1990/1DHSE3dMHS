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

## 3. Verdict against Codex's criteria (wording corrected after CODEX_REVIEW_M5_FOLLOWUP.md)

**Neither branch passes; conditional convergence is unresolved.** Optimality is 84–455 against a pass value of ≤ 1e-6. The change criterion (≤ 1e-3 for three consecutive windows) is not met: the last window still moves 0.07–0.14 on identifiable nodes and 180–290 on the complement.

**After 500 accepted updates, both branches reach similar objective values near 6.3e3 but remain nonstationary and differ materially at some nodes. Neither a common optimum nor a residual floor is established.**
- The two objectives agree to 0.33 %. That is above the proposed 0.1 % comparison threshold.
- The inter-branch |Δ ln P| reaches 2.1 on identifiable nodes and 9.4 elsewhere.
- Statements about the size of the pressure changes refer to the **current iterates**, not to an optimum.
- A feasible iterate only bounds the minimum force objective at this B from above.

**These runs are also affected by two defects found in the follow-up review**, so they do not validate the trust-region solve:
- Rejected arc candidates were modelled at the rejected trial state.
- Accepted final steps could lie outside the trust ball.

Both were fixed afterwards (§5).

**The scaled-P branch used the scalar reference P_ref**, not the per-node initializer scale P₀ of Codex's specification. With an exact diagonal metric, a constant rescaling may cancel from the preconditioned step, so this is recorded rather than blamed.

## 4. Open items and suggestions

- No extension and no cap change before the fixes (Codex). After the fixes, a short repeat with full diagnostics (§5).
- The fixed-B solve is a diagnostic. In alternation the pressure block is solved inexactly (Codex Q1).
- The repeatability check from ±1 % perturbations (Codex step 3) is not done yet. The branch script supports it (third argument).
- χ² calibration (deferred step 5).

## 5. Fixes after the follow-up review (CODEX_REVIEW_M5_FOLLOWUP.md)

- **One base model per trust-region iteration.** Gradient, diagonal, curvature, bounds and scaling all come from the accepted state. Trial points are evaluated in a separate workspace. One candidate per iteration (no arc backtracking).
- **Final step inside the trust ball.** It is scaled linearly toward 0 in u, which stays feasible (the box contains 0; the anchor constraint is linear), until its model-coordinate D-norm fits. The radius is updated from the final step's norm.
- **Projected-Cauchy safeguard.** The candidate must give ≥ 0.1 of the best Cauchy model decrease along −D⁻¹g (projected, in the trust ball); otherwise the Cauchy step is used.
- **Block failure.** An Ã-block L-BFGS stop on a line-search or EOS failure is a failure; budget-limited progress is not. A pressure block with no accepted step is a failure.
- **Final convergence.** Requires fresh audits of both blocks at the same final state: pressure optimality ≤ 1e-6 and ‖∇_Ã Φ‖ ≤ 1e-6 relative to the start. Blocks tighten when only the change test passes.
- **Inexact pressure block in alternation.** Stops when optimality halves or after `u_steps` accepted steps. The trust-region state persists across cycles.
- **Scripts.** Anchor mean computed once (no MPI in threaded callbacks). Trust-region state carried across reporting windows. Resumable final state. The ±1 % start is perturbed after the mask is fixed from the unperturbed checkpoint, then projected into bounds ∩ anchor. Logs add candidates, Cauchy fallbacks, trust scalings, cap hits, trust-norm use, ρ, model vs actual decrease, and ε-active set size.
- **Tests (new).** After rejected candidates `w` holds the accepted state exactly; every accepted step lies in its trust ball; bound and anchor hold; Φ decreases monotonically. These cover ln P at amplitudes 0.8 and 4 and p at amplitude 4, with rejections exercised.

**Open (decision needed).** In the alternating solve of C1b (8×8×65), the change test reaches 0 and |u − u_s| = 4e-6, but the result is never `converged`:
- At the round-off floor the Ã block's L-BFGS stops with "line search failed twice from steepest descent", which now counts as a failure (Codex finding 3).
- Its relative gradient audit bottoms out at about 2e-4, not the 1e-6 target.

This is the floor situation previously accepted for the C1b/C3 gates. How the outer test should treat it is for the user to decide; Codex asked that it not be treated silently as success.

## 6. Frozen-B repeat with the corrected pressure block (17 Oct 2026)

**Setup:** a new checkpoint made with the corrected code: one cycle, 300 Ã iterations, 5 pressure steps; Φ = 1.2953, pressure optimality 8.5e2. Otherwise as in §2. The trust-region state is carried across windows; per-node cap 5 in ln P. Fixed mask from the checkpoint.

| Window | ln P: raw force | ln P: opt | ln P: max ΔP/P (identifiable / rest) | ln P: Cauchy fallbacks | p: raw force | p: opt | p: max ΔP/P (identifiable / rest) | p: Cauchy fallbacks |
|---|---|---|---|---|---|---|---|---|
| 1 | 7184 | 1.35e3 | 6.6 / 1.3e4 | 0 | 6852 | 24.9 | 30 / 1.6e4 | 96 |
| 2 | 6876 | 15.9 | 1.1 / 2.0e3 | 1 | 6740 | 22.3 | 2.0 / 88 | 100 |
| 3 | 6744 | 30.5 | 0.64 / 376 | 4 | 6553 | 1.76e3 | 9.9 / 2.1e3 | 100 |
| 4 | 6680 | 893 | 0.31 / 705 | 1 | 6434 | 5.74e3 | 4.5 / 1.5e3 | 62 |
| 5 | 6616 | 3.01e3 | 0.32 / 442 | 7 | 6403 | 1.42e3 | 2.3 / 619 | 8 |

**Per-window diagnostics:**
- ln P: about 120 candidates per 100 accepted; mean ρ ≈ 0.8; final-step trust use 0.94–0.95; cap hits 20–41; ε-active set up to about 7.1e3 nodes.
- p: for its first 300 updates almost every step is a Cauchy fallback; trust use there is 0.01–0.06, Δ grows to 8.8, ρ ≈ 1.0.
- Both branches: model and actual decrease consistent (ratio about 0.6–0.98); min bound margin ≥ −5.7e-14; anchor error ≤ 7e-16.

**At 500 accepted updates:**
- Raw force: 6616 (ln P) vs 6403 (p), 3.3 % apart.
- **|Δ ln P| between the branches on identifiable nodes:** median 0.80, p99 2.9, max 4.4.
- Change from the checkpoint: identifiable median 1.10 (ln P) vs 0.32 (p); complement median 1.3 in both.

**Verdict:** neither branch passes (optimality 1.4e3 – 3.0e3; identifiable nodes still move 0.3–2.3 per window). Conditional convergence is unresolved.

**What this run adds:**
1. The corrected pressure block behaves as a trust-region method: feasible, inside its radius, consistent ratios, monotone decrease.
2. The two coordinates now reach **different** iterates: 3 % in objective and about a factor 2 (median) in pressure on identifiable nodes. The earlier "similar endpoints" (§2, buggy runs) does not hold up.
3. The constrained optimality measure is **not monotone** (ln P: 16 → 3e3 while Φ falls). That is consistent with a max-norm measure dominated by a few low-D nodes. It should be read together with quantiles; logging those is a suggested addition.
4. In p-coordinates the PCG direction, after the nonlinear map, projection and trust scaling, rarely gives 0.1 of the Cauchy decrease. That branch runs mostly on projected scaled steepest descent.
