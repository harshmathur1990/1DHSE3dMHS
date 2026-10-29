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

### 6.5 D3b, D3c, D3h and D4a (28 Oct; details in `design/CODEX_BRIEF_solver_time_round2_2026-10-28.md`)

- **D3b** (column blocks, `precond = block`), **D3h** (Fourier + column-block correction, `hybrid`) and **D3c**
  (Marquardt damping) are implemented in `PoissonTau.jl`. At μ = 1, D3h reached Φ/Φ₀ 2.62e-2 in 1308 s against D2's
  2472 s; D3b reached 2.56e-2 but with 33 km rms z(τ=1) error against D2's 22.5 km. D3b stopped with `no_descent` at
  step 28 at μ0.3 vertical. D3c with the old hybrid took 4082 s.
- The D4a warm-start numbers in the first brief came from an aliased script; D4b (warm start with common weights) has
  not been run.
- `hybrid` is retired as a candidate (damping inconsistent between its parts); the frozen labs below use `fourier` and
  `block`.

### 6.6 Frozen systems (`scripts/viewing/frozen_lab.jl`, full cubes and crops)

- Algebra sound: adjoint, finite differences, symmetry and positivity of A and of every M⁻¹, column blocks against
  probes. The four checkerboard cut-height modes (constant, (−1)ⁱ, (−1)ʲ, (−1)ⁱ⁺ʲ) are exact null modes of JᵀWJ.
- At small μ the undamped full-cube solves miss tol 1e-3 in 600 block iterations; the unresolved part is the
  large-scale (|k| ≤ 2) cut height. Block with Marquardt damping ≥ 0.01 converges in ≤ 141 iterations (3 cases).
- Balanced deflation with cut-height modes (gauge, |k| ≤ 2, |k| ≤ 4) or per-level |k| ≤ 1 modes: at most 20 % fewer
  iterations, nothing at μ0.3 vertical.

### 6.7 Damped block end-to-end (40 steps, `lm = fixed` / adaptive, 28 Oct)

| run | Φ/Φ₀ | z(τ=1) rms | str−quiet | time |
|---|---|---|---|---|
| μ1 fixed 0.01 / 0.1 / adaptive | 2.59 / 2.85 / 2.94e-2 | 36.7 / 35.2 / 32.7 km | −25.8 / −13.2 / −19.4 | 529 / 284 / 354 s |
| μ0.3 vertical fixed 0.01 / 0.1 / adaptive | 2.58e-2 / 0.309 / 2.92e-2 | 25.3 / 26.4 / 25.9 km | | 142 / 227 / 215 s |
| μ0.3 ray fixed 0.01 / adaptive | 5.30 / 5.44e-3 | 41.7 / 39.4 km | | 599 / 585 s |
| D2 μ1 / μ0.3 vertical / μ0.3 ray | 2.55e-2 / 5.5e-5 / 5.1e-3 | 22.5 / 21.0 / 24.1 km | −74 / – / – | 2472 s (μ1) |

Damped block runs are fast per step but, in these 40-step runs, left the large-scale heights where they were: Φ is flat
along those modes, and the damped, iteration-capped block solves did not move them while the Fourier solves did. This is
an observation of these runs, not an algebraic property: an exact block-Jacobi-preconditioned solve reaches every mode.

### 6.8 Common-objective reference runs (80 steps, Fourier, η ≤ 0.01, CG cap 400, `common_weights`, 29 Oct)

Each case from the column-HSE start and from the damped-block (fixed 0.01) endpoint, with the column-HSE start's
weights for both. Φ in units of the column-HSE start's Φ₀.

| case | start | final Φ/Φ₀ | z(τ=1) rms | str−quiet (truth) | CG / time |
|---|---|---|---|---|---|
| μ1 ray, cut −4 | column HSE | 2.58e-2 | 22.6 km | −75.5 (−76.6) | 13 034 / 5639 s |
| | block endpoint | ≈2.52e-2 | 22.6 km | −73.6 | 23 520 / 7473 s |
| μ0.3 ray, cut −3.5 | column HSE | 4.60e-3 | 28.7 km | −37.4 (−36.4) | 29 770 / 9968 s |
| | block endpoint | ≈4.66e-3 | 30.3 km | −35.9 | 28 181 / 9076 s |
| μ0.3 vertical | column HSE | 3.98e-5 | 25.8 km | −26.1 (−36.4) | 30 455 / 16 262 s |
| | block endpoint | no_descent at step 2 (see below) | | | |

- μ1: both starts reach the same heights; the block endpoint's 36.7 km was a stall in flat large-scale modes.
- μ0.3: none converged (pred 1e-4 – 2e-3 at step 80, CG at the 400 cap with relative residual 0.3–0.7). Φ keeps
  falling slowly while the z(τ=1) spread widens (p1/p99 −22 / +54 km at step 10 → −47 / +80 km at step 79) and the
  height error grows from 21–24 km (D2 at 40 steps) to 26–30 km. On this input, lower Φ at μ0.3 did not mean better
  heights against the Bifrost truth. The Bifrost truth is not a discrete equilibrium of the solver's model (§6.11), and
  on a consistent known equilibrium the same solver does not drift (§6.12).
- The μ0.3 vertical block-endpoint failure: every trial point of step 2 left the EOS table (32 of 32, μ up to 4e3;
  damping by multiples of ereg barely changes the step). With backtracking continued while out of the table (commit
  92d029e) the run proceeds at α ≈ 1e-3, ρ ≈ 1: a feasibility barrier, not a model failure.

### 6.9 Spectrum on 16 × 16 crops (`frozen_lab.jl parts=spectral`, dense generalised A q = λ M q, M = column blocks)

| case | λmin / λmax | λ < 0.1 λmax: count (weight of b) | block CG to 1e-3 / 1e-5 |
|---|---|---|---|
| μ1 ray | 2.0e-4 / 3.1 | 182 of 14 080 (0.037) | 93 / 146 |
| μ0.3 ray | 1.1e-5 / 3.8 | 675 of 12 800 (0.005) | 67 / 176 |
| μ0.3 vertical | 1.8e-5 / 3.7 | 2031 of 12 800 (0.103) | 107 / 385 |

- The 4 smallest λ are the checkerboard cut-height modes, well separated, with weight of b ≤ 1e-8: they cost nothing.
- Ritz pairs: at μ1 the smallest are exact (20 of 32 with residual < 1e-3 without reorthogonalisation, 25 with). At
  μ0.3 none of 32 converge in 300 iterations, with or without reorthogonalisation: the low spectrum is a dense band.
  The earlier Ritz-deflation trials used unconverged vectors.
- Deflation, CG iterations to 1e-3 / 1e-5:

| coarse space | μ1 | μ0.3 vertical |
|---|---|---|
| none | 93 / 146 | 107 / 385 |
| cut height \|k\| ≤ 2 (16), c only | 45 / 81 | 96 / 174 |
| the same, column-block energy extension | 44 / 79 | 95 / 171 |
| the same, exact energy extension | 44 / 82 | 109 / 199 |
| exact smallest 64 eigenvectors | 15 / 24 | 33 / 51 |

  The energy extension lowers the coarse vectors' Rayleigh quotients (exact 5–10× lower) but changes the iterations
  by a few. The gains depend on the tolerance: at μ0.3 vertical the cut-height modes give 1.1× at 1e-3 but 2.2× at
  1e-5, and the exact 64 eigenvectors 3.2× at 1e-3 and 7.5× at 1e-5. The "λ < 0.1 λmax" counts depend on that
  threshold: they describe this operator in this metric (block M), not an invariant number of undetermined physical
  modes.

### 6.10 Block CG against crop size (10 D2 steps on each crop; tol 1e-3; Marquardt damping 0 / 0.01 / 0.1)

| crop | μ1 | μ0.3 vertical | μ0.3 ray |
|---|---|---|---|
| 8² | 101 / 65 / 34 | 54 / 51 / 29 | 146 / 81 / 37 |
| 16² | 93 / – / – | 107 / – / – | 67 / – / – |
| 32² | 49 / 43 / 23 | 89 / 52 / 37 | 86 / 53 / 27 |
| 64² | 33 / 50 / 22 | 208 / 88 / 42 | 448 / 93 / 42 |
| 128² | 154 / 56 / 29 | 200 / 74 / 38 | > 600 / 107 / 50 |
| 504² (full) | 131 / – / – | > 600 / ≤ 141 / – | – |

Each crop has its own state (the larger crops are further from converged: pred 0.07–0.34 against ≤ 0.05), so this is
not a clean scaling test. Undamped, the μ0.3 iterations grow with the crop; with damping 0.1 they stay at 22–50 at
every size and μ.

Note on the regularisation: Φ = Σ wc (Rx² + Ry²) + wf Rz² does not contain ereg. ereg enters only the step operator
A = JᵀWJ + D(ereg), where it adds curvature to the linear model of each step; it does not change Φ or its minimisers.

### 6.11 The Bifrost truth against the solver's model (`truth_consistency.jl`; truth starts, 20 steps; 29 Oct)

These measure the synthetic Bifrost inputs, not the solver. They hold the truth pressures and the truth cut heights
fixed, while a solve may change both, so they are not lower bounds on the error a solve can reach.

| at the truth (T, P), cut −4 (μ1) / −3.5 (μ0.3) | μ1 | μ0.3 |
|---|---|---|
| ln ρ_EOS(T, P_truth) − ln ρ_truth | 0.011 | 0.011 |
| solver Δz per level − truth Δz, log τ > 0 (relative rms) | 17 % | 38 % |
| z(τ_LOS = 1) by the solver map from the truth cut height − truth (rms; strong, quiet) | 23.6 km (12.9, 24.0) | 55 km (70, –) |
| strong − quiet shift from that map | +0.6 km | −15.1 km |
| cut ln P, column HSE − truth (rms) | 0.124 | 0.166 |

Candidate sources, not yet separated: the snapshot density against the EOS; the generator's interpolation of χ from the
geometric grid against the solver's χ(interpolated T, solved P); the trapezoid τ–z quadrature; the fixed column-HSE cut
pressure and upper geometry; and forces the snapshot does not balance (it is not static). **Spatial comparison** (`error_maps.py`, gauge-aligned): the 23.6 km map discrepancy e_m and the 22.6 km μ1 solve
error e_s are similar in size but are unrelated fields.

| μ1, ref column start (block start and truth start alike) | e_s rms | e_m rms | corr(e_s, e_m) | rms(e_s − e_m) | corr(e_s, cut ΔlnP) |
|---|---|---|---|---|---|
| full resolution | 22.6 km | 22.6 km | +0.015 | 31.7 km | −0.03 |
| scales > 1 Mm (21 × 21 box-car) | 14.6 | 1.5 | +0.06 | 14.6 | −0.04 |
| scales < 1 Mm | 14.0 | 22.6 | +0.01 | 26.4 | −0.02 |

- At μ1 the solved error is not the solver-map discrepancy, and not the cut-pressure mismatch. Against the
  model-projected truth (z_truth + e_m) it is larger (31.7 km). So the 22.6 km is not explained by either. The
  remaining candidates (non-static snapshot forces, the EOS density offset, horizontal discretisation) are untested.
- At μ0.3 vertical, e_s (25.8 km) is mostly large-scale (21.6 km). In strong-field columns that large-scale part is
  anti-correlated with e_m (−0.49, slope −1.1); elsewhere |corr| ≤ 0.1.

Truth starts (common weights; Φ in units of the column-HSE start's Φ₀): from the truth shifted per column to the
column-HSE cut pressure, μ1 ends at the column start's heights (22.4 km, Φ 0.024 against 0.026), which shows
repeatability for these starts, not uniqueness. At μ0.3 vertical the unshifted truth (with the column-HSE cut, so a
pressure jump at the first solved face) passes 13.6 km at step 5 and ends at 35.6 km with Φ 5.3e-5 (column start 4.0e-5
at 80 steps). Neither start is the original equilibrium with its own boundary, and a non-stationary trajectory passing a
low-error point does not identify a stopping time. The shifted starts move every level by the cut mismatch (rms
0.12–0.17 in ln P) and begin 200+ km off.

### 6.12 Known-equilibrium control (`make_mhs_control.jl`, 29 Oct; logs `out/control/` on the cluster)

**Case.** An exact 2-D Grad–Shafranov atmosphere, x-periodic and y-invariant:
- box 12 Mm, period 3 Mm, NY 8;
- pressure deficit P = P₀(1 − 0.3 cos² kx), |B| up to 1.2 kG at z = 0;
- Eddington–Hopf reference, fine dz 250 m;
- mapped onto rays like `make_slanted_tau.jl`.

The truth depression contrast between the lowest and highest 20 % of z(τ=1) is −65 km (μ1) and −53 km (μ0.3).

**Construction checks.**
- Centred-difference force balance: max |R_x|/ρg 7e-5 and |R_z|/ρg 5e-4 (rms 2e-5); the maximum is at z ≈ −70 km, where T₀(z) has a kink.
- ∇·B: 3e-5 of the largest term.
- The EOS inversion succeeds everywhere, with ln P at least 12 inside the table.
- Generator refinement (dz 250 → 125 → 62.5 m) changes z(τ=1) by ≤ 5 m and ln P by ≤ 3e-5, both second order. That is negligible against the solver errors below.
- The field-free case (D = 0) is recovered exactly at μ1 and μ0.3: Φ reaches round-off in 2 steps.

**Refinement.** μ1 uses ray balance with cut −4; μ0.3 uses vertical balance with cut −3.6, which lies on every grid. The
tracks are:
- **bt**: the experimental truth boundary (the truth's fixed levels), started from the exact truth;
- **col**: the production column-HSE boundary;
- **prod**: column boundary with the production defaults (gn_tol 1e-3, 25 steps).

z1 is the rms z(τ_LOS = 1) error with the mean removed.

| grid (nx, Δlog τ) | μ1 bt: defect / z1 / ΔlnP [0,1.5) | μ0.3 bt: defect / z1 / ΔlnP [0,1.5) | col: mean z1 offset; ΔlnP [−4,−2) (μ1 / μ0.3) |
|---|---|---|---|
| 128, 0.2 | 2.0e-7 / 2.68 km / 0.019 | 4.8e-7 / 0.79 km / 0.022 | −51 / −119 km; 0.054 / 0.063 |
| 256, 0.2 | 1.9e-7 / 2.91 / 0.014 | 5.3e-7 / 0.32 / 0.018 | −50 / −119; 0.054 / 0.063 |
| 128, 0.1 | 7.6e-8 / 0.57 / 0.005 | 2.7e-7 / 1.11 / 0.015 | −51 / −121; 0.049 / 0.056 |
| 256, 0.1 | 7.1e-8 / 0.72 / 0.003 | 3.2e-7 / 0.14 / 0.006 | −51 / −120; 0.049 / 0.056 |
| 512, 0.1 | 7.1e-8 / 0.78 / 0.002 | 4.2e-7 / 0.11 / 0.007 | −51 / −120; 0.049 / 0.056 |
| 256, 0.05 | 3.2e-8 / 0.15 / 0.0006 | 9.7e-8 / 0.19 / 0.002 | −51 / −120; 0.046 / 0.052 |
| 512, 0.05 | 3.1e-8 / 0.13 / 0.0004 | 1.3e-7 / 0.03 / 0.001 | −51 / −120; 0.046 / 0.052 |

The defect is the rms residual of the exact truth on the τ grid, √(Φ/ΣW), with the run's own weights. It is almost all
in the vertical (along-column) residual.

**What converges.**
- **μ1:** the error is set by Δlog τ: about 4× per halving in height and ln P (second order). Horizontal refinement
  changes nothing.
- **μ0.3:** both directions matter, and single-direction refinements are not monotone; the horizontal and τ error
  terms partly cancel. Along the joint path (128, 0.2) → (256, 0.1) → (512, 0.05), z1 goes 0.79 → 0.14 → 0.03 km and
  ΔlnP 0.022 → 0.006 → 0.001, roughly 4–5× per step.
- **The truth's defect converges more slowly than the solution:** about 1.2–1.4 orders per halving at μ1. At fixed
  Δlog τ it does not fall with nx at μ0.3, in either balance; the ray balance gives 3.6–4.0e-7 at nx 128–512. The
  one-sided column derivatives at the ends are the likely reason; this is not yet checked.

**From the truth or from column HSE, the bt solve reaches the same state.** Starts compared: the exact truth, and column
HSE shifted to the truth cut pressure, at μ1 and μ0.3 (nx 256, dl 0.1).

**Boundary bias (col against bt).** The column-HSE cut boundary adds:
- a near-uniform height offset of −51 km at μ1 and −120 km at μ0.3;
- ΔlnP 0.05–0.06 near the cut.

Neither changes under refinement, as expected for a boundary model that differs from this magnetostatic equilibrium.
The interior errors (z1 with the mean removed, the depression contrast) match bt. With the column boundary, Φ also has
a slow tail at μ0.3: Φ/Φ₀ 1.34e-3 → 1.28e-3 over 15 steps with unchanged heights, half of the vertical residual on the
first 3 faces. With the truth boundary Φ converges in a few steps.

**No drift.**
- In every run the heights reach their final error by step 2–5 and stay there for up to 80 steps.
- Both balances behave the same at μ0.3: on the 256/0.1 grid the ray balance gives 0.2 km and the vertical balance
  0.1 km.
- Perturbations of the start were not run on the control.

**Time to converge with the production defaults** (gn_tol 1e-3, 25 steps, cg_maxit 80; one rank, 32 threads; small
grids, so setup is a large part of the wall time):

| grid | μ1 col | μ0.3 col | μ1 bt | μ0.3 bt |
|---|---|---|---|---|
| 128, 0.2 / 256, 0.2 | 5 steps, qualified, 20 / 24 s | 6 / 8 steps, qualified, 23 / 30 s | | |
| 128, 0.1 / 512, 0.1 | 5, qualified, 25 / 21 s | 6 / 11, qualified, 28 / 38 s | | |
| 256, 0.1 | 5, qualified, 15 s | **25, max_steps, 38 s** | 4, qualified, 16 s | 8, qualified, 25 s |
| 256, 0.05 | 5, qualified, 27 s | **25, max_steps, 81 s** | 4, qualified, 17 s | 8, qualified, 34 s |
| 512, 0.05 | 5, qualified, 18 s | 10, qualified, 63 s | | |

- Every μ0.3 step uses the full 80 CG iterations (relative residual 0.01–0.04), against 5–9 at μ1.
- With the truth boundary μ0.3 qualifies in 8 steps against 4 at μ1.
- With the column boundary, the slow Φ tail of the boundary compromise keeps the inexact (capped) steps' decrement near
  0.04 on two grids, so the stop never qualifies in 25 steps, although the heights are final by step 3.

**Stop-rule edge case.** When the start is already exact (field-free), Φ reaches round-off, the relative decrease is
undefined, and the solve ends `no_descent`. The embedding reports that as status `stalled` for a perfect solve.

**Reading.** On a known equilibrium consistent with the solver's model:
- the solver converges at μ0.3 to the truth at second order, without the height drift seen on the Bifrost input, in
  about 2× the Gauss–Newton steps of μ1;
- each μ0.3 step costs about 10× the linear iterations;
- the remaining systematic error is the column-HSE cut boundary.

This supports, but does not prove, that the Bifrost μ0.3 drift comes from the input and boundary (§6.11) rather than
from the objective. The control is 2-D (y-invariant, azimuth 0). Mixed derivatives and nonzero t_y, for example with a
rotated A(k_x x + k_y y, z), are not yet tested.
