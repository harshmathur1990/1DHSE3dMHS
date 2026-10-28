# Bounded, μ-independent solve time for `mhs-poisson-ltau`: design for review

*Status: 28 Oct 2026, branch `solver-time`. D1, D0, the vertical-depth cut and D2 are implemented and measured (§6);
D3 is revised in §6.4 and awaits a decision. Follows
[`VIEWING_ANGLE_DESIGN.md`](VIEWING_ANGLE_DESIGN.md) and the P4 results (`HSEMHS/reports/viewing/`).*

**Request (user, 27 Oct 2026):** the time to converge must be comparable to the μ = 1 mode and independent of μ.

---

## 1. Where the time goes (P4 logs, full 504² × 95 cubes, cut −4, 3 ranks × 32 threads)

| Run | Steps (stop) | s/step | Steps to 1.5× / 1.2× final Φ | CG rel. residual, steps 1–10 / later | Median α |
|---|---|---|---|---|---|
| μ = 1 (synthetic) | 100 (max) | 65* | 8 / 22 | 0.05 / 0.29 | 0.32 |
| inclined μ 0.866 | 100 (max) | 43 | 5 / 15 | 0.06 / 0.26 | 0.25 |
| inclined μ 0.707 | 62 (rtol) | 44 | 6 / 9 | 0.06 / 0.25 | 0.55 |
| inclined μ 0.5 | 100 (max) | 67* | 13 / 24 | 0.04 / 0.25 | 0.45 |
| inclined μ 0.5, φ 45° | 100 (max) | 44 | 12 / 19 | 0.07 / 0.19 | 0.50 |
| τ-cut study, production cube, μ = 1 | 19 (rtol) | 43 | 5 / 7 | 0.02 / 0.04 | 1.00 |

\* hyades18; the others ran on hyades20. The difference is the node, not μ.

Findings:

1. **The cost per step does not depend on μ or on the geometry** (43 s on hyades20 in every case). The slant terms are
   local and add no work to the products.
2. **The number of steps that matter is similar at every μ.** 5–13 steps reach 1.5× the final Φ, and 7–24 reach 1.2×.
3. **The total step count is set by the tail and the stop rule, at μ = 1 as well.** After about 10 steps the CG
   solves end at their 80-iteration cap with a relative residual of 0.2–0.3. The resulting Gauss–Newton steps are
   poor, so the line search takes α ≈ 0.2–0.5 and Φ creeps by about 0.1 % per step. The rule "ΔΦ/Φ < 1e-4 on 3
   consecutive steps" then needs a long run of creeping steps, and one noisier step resets the count. Whether it fires
   at step 19 or never is close to chance: that is the 19-step versus 100-step difference between two μ = 1 cubes.
4. **The slant terms are small at the grid scale.** Their size is tan θ · Δz/Δx, with Δz ≈ 10–20 km per level and
   Δx = 48 km: 0.1–0.7 for μ ≥ 0.5. This is why CG quality does not degrade with θ there. It would degrade for
   μ ≲ 0.3 (tan θ ≳ 3), where the preconditioner, which ignores the slant, stops representing the operator.

So making the time independent of μ needs (a) a solve whose length does not depend on a creeping tail, and (b) a
preconditioner that stays accurate at any θ.

## 2. Design

### D1. Stop on stationarity, every step, with a hard step budget

Each step already has the Gauss–Newton direction δ and the right-hand side −JᵀWR. The **predicted relative decrease**
of a full step,

    d_k = δ·(−JᵀWR) / Φ_k,

is scale-free and costs nothing extra. Today it is computed only once, after the loop.

- **Stop when d_k < `gn_tol`** (default 1e-3: a full Gauss–Newton step could lower Φ by at most about 0.1 %). It is
  required on 2 consecutive steps, so that one inaccurate CG solve cannot stop the run.
- **`max_steps` becomes a time budget** (default 25 instead of 100), with the stop reason `:max_steps` reported as
  now.
- The ΔΦ/Φ rule is kept as an option (`stop = "phi"`) for continuity with the documents and FFNOML.

This removes the dependence on the noisy tail: a creeping run stops as soon as no meaningful decrease remains.

### D2. Inexact Gauss–Newton: CG accuracy matched to the outer progress

A fixed 80-iteration cap gives accurate steps early, when they are not needed, and poor ones late, when they are.

- **Eisenstat–Walker forcing:** the CG tolerance is η_k = min(0.1, max(0.01, √d_{k−1})).
- **Cap 200, but stop as soon as η_k is met.** Early steps then use fewer iterations than today, and late steps get the
  accuracy that makes α ≈ 1 possible again.
- The expected net effect: the steps that matter cost about the same, and the tail disappears instead of costing 80
  iterations per creeping step.

### D3. Slant-aware preconditioner (θ-independent CG convergence)

The preconditioner today is, per horizontal Fourier mode k, the layer-median operator A + |k|² B, diagonalised once.
The slant adds first-order couplings i (k·t) C and second-order couplings (k·t)² D, where C and D are n₁ × n₁
layer-median matrices from the ray-direction terms (§4.2 of the viewing-angle design). These depend on the direction of
k, not only on |k|, so the single generalised eigenproblem no longer covers every mode. Two options:

- **(a) A complex n₁ × n₁ Cholesky factorisation per mode.** That is 127 000 modes × n₁³/3 ≈ 7e9 flops per step,
  about 1 s threaded, against 43 s per step today. Exact for the median operator at any θ.
- **(b) Diagonalise in the rotated, sheared Fourier basis.** k is split into the components along and across t; this
  is cheaper but approximate.

**Proposal:** (a), enabled only when t ≠ 0, so the μ = 1 path is unchanged. It removes the expected degradation at
μ ≲ 0.3 and makes the CG iteration count independent of θ.

### D4. Warm start in the embedding API (for inversion loops; μ-independent)

`solve_embedded!(ws, …; start = :previous)` starts from the last converged (ln P, c) of the workspace instead of the
column HSE, when the new T and B are close to the previous ones. Successive inversion iterations then need 1–3 steps.
The default stays `:column_hse`, so FFNOML behaviour is unchanged unless it opts in.

### D5. What stays unchanged

The equations, the cut at −4, the gauge, the outputs, and every default the FFNOML production code relies on until
the user switches. All of this goes on a new branch from `viewing-angle`.

## 3. Validation plan

1. **Accuracy against the number of steps** (measurement before any change): save (ln P, z) at steps 5, 10, 15, 20,
   30 and 100 for μ = 1 and inclined μ = 0.5. If z(τ = 1), ln P and the Wilson depression stop changing by step
   15–20, D1 with a budget of 25 loses nothing.
2. **With D1 + D2** on all five P4 cubes, plus μ = 0.3: the wall time at each μ must be ≤ 1.2 × the μ = 1 time, and
   accuracy must match the 100-step results to within 1 km rms in z(τ = 1) and 0.005 in ln P.
3. **With D3** at μ = 0.3 and 0.2: the CG iteration count per step must stay within 20 % of μ = 1.
4. **D4:** perturb T by the size of an inversion update and count the steps to `gn_tol` from the previous solution.
5. **Regression:** the existing tests, μ = 1 bitwise with the old stop rule selected, adjoint tests of the new
   preconditioner, and MPI tests.

## 4. Expected outcome

At every μ: about 10–20 steps × about 40 s ≈ 7–13 min for a full 504² cube on one node, cold start. Warm-started
inversion iterations take 1–3 steps (≈ 1–2 min). This is the same at μ = 1 and off-centre, which is what the user
asked for. The μ = 1 run also gets 3–10× faster than the 100-step runs.

## 5. Decisions requested

1. Approve D1–D4, or a subset (D3 matters only for μ ≲ 0.3).
2. Defaults: `gn_tol` 1e-3 and `max_steps` 25, or other values. Make the new stop rule the default, or opt-in only?
3. Run measurement §3.1 first (about 1.5 h on one node), then implement.

## 6. Measurements (28 Oct 2026; full 504² × 95 synthetic cubes, 3 ranks × 32 threads, one hyades node per run)

Scripts: `scripts/viewing/measure_steps.jl` (per-step snapshots), `measure_analyse.py`, `stop_rules.py` (replays stop
rules on the snapshots). Runs in `out/solver/` on the cluster worktree.

### 6.1 D1 and the cut at the vertical depth

- Every step until 40 or more, the predicted decrease stays at 0.3–3 %: with `gn_tol` 1e-3 the 25-step budget is what
  ends the solve. 1e-3 was met once in 40 steps (μ = 0.5, step 36).
- With the cut at log10 τ_LOS = −4 the strong inclinations were wrong (μ = 0.3: 38.7 km rms z(τ=1); μ = 0.2: 57.9 km;
  column HSE 25 km): the cut lay physically higher, at low β. **`cut_frame = vertical` (default)** places it at the same
  vertical depth, log10 τ_LOS = cut − log10 μ: μ = 0.3: 21–25 km, μ = 0.2: 20–32 km. μ = 1 is unchanged.

### 6.2 D0 (vertical balance) against the ray balance, with the vertical-depth cut (40 steps, D2)

| μ | truth str−quiet | column HSE: rms / str−quiet | ray | vertical |
|---|---|---|---|---|
| 0.5 | −48.8 | 23.9 / −12.2 | 26.4 / −36.8 | 30.9 / −36.1 |
| 0.5, φ 45° | −50.4 | 23.6 / −13.2 | 31.6 / −40.2 | – |
| 0.3 | −36.4 | 25.3 / −0.3 | 24.1 / −27.1 | 21.0 / −12.1 |
| 0.2 | −29.9 | 24.6 / +4.5 | 32.2 / −1.5 | 19.6 / +2.3 |

Neither formulation is converged at μ ≤ 0.3 after 40 steps (the strong-field depression is still moving); the choice
waits for converged solves. Both are kept (`balance`).

### 6.3 D2 (gain-ratio LM, Eisenstat–Walker CG, cap 200)

Steps until z(τ=1) changes by < 0.5 km rms per step on 2 consecutive steps (time, distance from the 40-step state):

| μ | baseline | D2 |
|---|---|---|
| 1 | 10, 656 s, 3.5 km | 13, 418 s, 1.2 km |
| 0.866 | 9, 392 s, 3.8 km | 12, 389 s, 3.3 km |
| 0.707 | 15, 726 s, 2.4 km | 16, 622 s, 1.3 km |
| 0.5 (ray) | 21, 876 s, 1.7 km | 16, 700 s, 1.2 km |
| 0.5, φ 45° | 21, 1048 s, 2.5 km | 18, 949 s, 1.9 km |
| 0.3 (vertical) | 20, 573 s, 4.2 km | 32, 1243 s, 3.3 km |
| 0.2 (vertical) | 14, 376 s, 4.2 km | 21, 1960 s, 3.4 km |

- μ ≥ 0.5: D2 reaches a given closeness to the final state sooner; early steps take 5–40 CG iterations (4–20 s).
- The damping now rises (to 10⁴–10⁵) when the gain ratio drops; the gain ratio stays 0.1–0.5 in the tail even with
  CG at 0.03, so the tail is partly nonlinearity, not only inexact CG.
- μ ≤ 0.3: no gain. CG ends at the 200 cap with relative residual 0.1–0.5 (at times > 1), so the tail steps cost
  70–150 s and the time is 2–5× that of μ = 1.
- Stop rules replayed on the snapshots: "pred < 1e-2 on 2 steps" stops μ ≥ 0.5 at 17–33 steps within ≤ 2 km of the
  final state; "Δz(τ=1) < 0.5 km rms on 2 steps" at 13–21 steps. Both are candidates; the defaults (1e-3, 25) stay.

### 6.4 D3 revised: the slant is not where the preconditioner fails

In the linearised operator the slant appears only (a) in the Lorentz-force coefficients (tan θ × local B gradients,
different in every column) and (b) in the D0 coupling −Δz t·R̄_h. For the ray balance the field-free operator, which
the layer-median preconditioner represents, has **no slant term**, so D3 as written (§2) would change nothing there;
for the vertical balance it would add only (b). CG is already poor at μ = 1 in the tail (relative residual 0.05–0.26
at 200 iterations) and worse at μ = 0.2 (ray 0.15–0.3, vertical 0.3–0.5): the deficiency is the field-dependent terms,
which tan θ amplifies. Options:

- **D3a** as designed, vertical balance only: per-mode Hermitian factorisation of A + s²B + i(k·t)C, binned in
  (|k|, angle of k to t) to bound memory. Helps only (b).
- **D3b** column-exact two-level preconditioner (both balances, every μ, including the μ = 1 tail): the Fourier
  layer-median solve for the smooth part, then a per-column n₁ × n₁ correction holding the local W, dw, aχ and the
  Lorentz ζ terms (block-Jacobi on the residual). Cost ≈ one extra banded solve per column per iteration.
- **D3c** no new preconditioner: accept the μ ≤ 0.3 time, cap CG at 80 again in the tail (it does not converge anyway)
  to halve the late-step cost.
