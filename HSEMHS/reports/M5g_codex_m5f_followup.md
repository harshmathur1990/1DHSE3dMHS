# M5g: Codex's M5f review — bounded repair and validation pass (21 Oct 2026)

**Follows** `CODEX_REVIEW_M5f_2026-10-19.md` and the consolidated `REVIEW_M5f_2026-10-21.md`. The user asked for the
review to be implemented and run. Code: commit `ae47d13`. Runs: UiO cluster, repository copy
`/mn/stornext/d9/data/harshm/MHS` (fast-forwarded to `ae47d13`), outputs in `HSEMHS/out/m5g/` there (ignored by git).
Julia 1.13.1 (the Manifest's version), Python `~/env` (numpy 2.1.3, scipy 1.16.2).

## 0. Summary

| Review item | Done | Outcome |
|---|---|---|
| 1. Outside-domain curvature probes | Repaired; audited on every M5f checkpoint | The defect was active throughout M5f's bounded runs (see §2) |
| 2. Cauchy safeguard lost after truncation | Both candidates truncated before comparison; diagnostics on the final step | Safeguard now correct, but **never selected** in the validation runs (0/146 candidates) |
| 3. PDHG dual sign | Fixed; both signs; seeded with the local bound | Centre bracket now **t\* ∈ [3.629, 3.913]** (was [−0.85, 3.946]) |
| 4. Matched-budget claim | Corrected in the M5f report | — |
| 5. ΔB log included ΔP/P | `measured_change_P` / `measured_change_B`; pilot, solver diagnostics fixed | — |
| 6. Bracket / verification checks | CG status, ‖Ac − r‖∞, reconstruction mismatch, finiteness, divergence gate; incumbent saved | All UB fields pass (div_rel ≤ 1e-12) |
| 7. Tests and API validation | Parameter/feasibility validation; 55 new Julia assertions, MPI limiting-face case, 16 Python checks | All pass; MPI n = 2, 4 pass (local Julia copy, §6) |
| §4 short robustness validation | τ = 0.9, 0.99, 0.999, common start, ≤ 5000 products, predeclared stops | All three complete, no collapse; 2–8× lower Φ than M5f at matched work |
| §3 fixed-support centre-tail test | `centre_tail.py` (PDHG) | **Inconclusive** at the 2 h wall: best field max 5.375σ, tail 0.776 % (both criteria not met: 307 cells > 5σ); no infeasibility certificate |
| Evidence bundle | Local copy, `HSEMHS/out/evidence_m5g/` (§8) | — |

## 1. Code changes

`src/MHSStaggered.jl`
- `gn_diagonal!(…; barrier_safe = true)`: with a face hinge or barrier set, every probe-based estimate (the Ã layer
  means — `:layer`, `:layer_gn` and `:colored` alike — and the b̄ central difference) is computed with the face term
  switched off, and the exact local diagonal of the face term is added from the base state:
  `face_curvature!` (h_f = (λ/N_B) φ''(c_f)/σ_f², φ'' = 2(u² + c²)/(u² − c²)² for the barrier, 2 on active hinge faces —
  the weight `gn_hvp!` already uses) and `face_curvature_diag!` (squared curl-stencil coefficients of `curlA_T!`,
  × A_ref²; b̄: Σ over z-faces). This **replaces** the averaged contribution (no double counting). The pre-M5g estimator
  is kept behind `barrier_safe = false` for the audit only.
- `set_face_hinge!`: λ finite (> 0 for a barrier, ≥ 0 for a hinge); thresholds finite (> 0 barrier, ≥ 0 hinge).
- `measured_change_P`, `measured_change_B`; `measured_change` = their maximum (unchanged behaviour for its callers).

`src/MHSJointGN.jl` (`joint_gn_solve!`)
- `face_fraction` / `fraction_to_boundary`: largest feasible fraction from the exact linear map
  δB = A_ref curl δÃ + δb̄ ẑ (no objective evaluation, no work charged).
- The GN candidate and **every** Cauchy trial are truncated (τ × fraction) **before** their model decreases are
  computed and compared; the Cauchy direction −D⁻¹g uses the barrier-inclusive D (barrier-scaled, so it moves little in
  coordinates that push active faces outward). Radial truncation alone guarantees no tangent motion; this is stated in
  the docstring.
- Diagnostics (`share_final`, model split, ΔP, ΔB — now magnetic only) are computed for the final selected step.
- Validation: 0 < `bound_tau` < 1, finite positive bounds, a feasible base (strictly inside with a barrier); the trial
  state is re-checked against the bounds (`bound_check_tol` = 1e-9σ) before acceptance.
- `callback` returning `:stop` ends the solve (status `stalled`, "callback stop"); `cand_sink` exposes the live candidate
  list; the result carries μ and Δ; candidate records carry `pred_gn`, `predc`, both fractions and `bound_viol`.

Scripts (`scripts/m5_bifrost/`)
- `centre_bracket.py`: restructured (`CentreSystem`); LB = |λᵀr|/‖Aᵀλ‖₁, seeded with the local bound (certificate: its
  row); `project` records CG info and ‖Ac − r‖∞ and has an absolute floor (a rounding-level residual is not iterated on);
  `physical` gates a UB on finiteness and relative divergence ≤ 1e-9; the incumbent field is saved at each improvement;
  `--reeval`, `--warm`; wording "numerically verified" (no rigorous rounding bounds).
- `centre_tail.py` (new): fixed-support tail test, PDHG (§5).
- `verify_field.py`: finiteness and divergence gates, criteria lines, exit code.
- `m5g_bounded.jl` (new): validation driver — predeclared stops, Φ₀ / Φ_H split, stationarity, slack, separate ΔP/ΔB,
  window candidate statistics, full restart state in every checkpoint (x, work, Δ, μ, λ, τ, margin, paths, git).
- `m5g_probe_audit.jl` (new): pre-M5g probe incidence and curvature comparison at saved checkpoints.
- `m5f_pilot.jl`: ΔB column magnetic only. `data_qp.py`: "KKT gap" wording removed.

Tests
- `test/test_face_bounds.jl` (in `runtests.jl`), 55 assertions: parameter validation; barrier gradient and isolated
  barrier Hessian vs finite differences at slack 0.5 and 1e-3 (steps within 1e-3 of the fraction to the boundary along
  ±d); the Codex reproducer asserted (b̄ curvature exact to 1e-3, pre-M5g < 0.1 × exact); face-term part of the Ã diagonal
  equal to the exact GN diagonal node by node (1e-8); fraction to the boundary; bounded joint GN at slack 1e-6 (invalid
  τ and infeasible start rejected, Φ decreases, strictly feasible, post-truncation model decrease > 0, divergence ≤ 1e-12
  relative, anchor preserved, `:stop`); separate ΔP / ΔB.
- `test/mpi/mpi_tests.jl`: limiting face on a non-zero rank — global fraction, violation and a bounded joint GN agree
  on all ranks.
- `scripts/m5_bifrost/test_centre_bracket.py`: scalar (λ = −1; corrected bound 1, M5f sign −1), two LPs with known
  optima (one above the local bound), an end-to-end known divergence-free field, an invalid field, a NaN field, a failed
  projection (CG info ≠ 0 recorded).

## 2. Incidence of the curvature defect at the M5f checkpoints (`out/m5g/probe_audit.log`)

All 20 saved checkpoints of the three M5f barrier runs (λ = 0.1), with the pre-M5g probes replicated:

| Checkpoints | min face slack | pre-M5g Ã probes outside the domain | b̄ probes outside |
|---|---|---|---|
| 3 starts | 4.9e-2 | 0/8 | 0/2 |
| 8 near-bound checkpoints (τ 0.9: 2; τ 0.99: 4; τ 0.999: 2) | 6.7e-10 – 1.4e-5 | **4–8 of 8** | 0–2 of 2 |
| 9 later τ = 0.99 checkpoints | 6.5e-4 | 0/8 | 0/2 |

Curvature against the exact GN value (unit-vector GN products):
- b̄: pre-M5g ratio as low as **1.8e-6** (τ 0.999, checkpoint 2) and 7.5e-3 – 7.5e-2 elsewhere near bounds; M5g 1.000 at all 20.
- Ã at the 4 nodes touching each checkpoint's minimum-slack face (80 nodes): pre-M5g median ratio **0.023**, minimum
  **3e-308** (the estimate fell to the 1e-300 floor while the exact value was 3.3e7: infinite objectives in the probes);
  M5g median **1.01** (range 1 – 4.2; the 4.2 cases are the barrier-free layer mean, identical in both estimators).

So the reproduced defect was not a corner case: it was active at every M5f checkpoint with a face within ~1e-5σ of its
bound, which is where M5f's collapses happened. The τ = 0.99 run escaped it once the minimum slack recovered to 6.5e-4.

## 3. Short robustness validation (`out/m5g/bounded_tau*.log`)

Common start (the M5f pilot start, QP m = 0.05 field, v of the M5e ×100 endpoint), barrier λ = 0.1, hard bounds 5/3 − 1e-3
on the M5f support, Δ0 = 1.2412 (the M5f common radius), budget 5000 products, windows of 1250 (chosen so that windows
fall at M5f's 2500/5000 checkpoints). **Predeclared stops:** wall 7200 s; radius collapse; two consecutive windows without
a ≥ 1 % decrease of raw force, max layer RMS or ‖g_mag‖_{D⁻¹}; ≥ 50 % of a window's candidates truncated below 1e-3.

| τ | work | Φ (M5g / M5f) | raw force (M5g / M5f) | max layer RMS (M5g / M5f) | outcome (M5g / M5f) |
|---|---|---|---|---|---|
| 0.9 | ~2500 | **0.617** / 1.445 | 2.45e3 / 1.07e4 | 235 / 514 | |
| 0.9 | ~5000 | **0.511** / 1.268 | 1.40e3 / 8.98e3 | 174 / 472 | budget / **collapse at 5034** |
| 0.99 | ~2500 | **1.377** / 2.851 | 1.01e4 / 2.48e4 | 492 / 834 | |
| 0.99 | ~5000 | **0.624** / 2.056 | 2.53e3 / 1.69e4 | 240 / 670 | budget / budget (30000) |
| 0.999 | ~2500 | **0.572** / 5.451 | 2.01e3 / 5.08e4 | 212 / 1279 | |
| 0.999 | ~5000 | **0.508** / 4.158 (at 4408) | 1.37e3 / 3.79e4 | 174 / 1090 | budget / **collapse at 4408** |

Same start, same settings, same Δ0, matched work: the comparison isolates the M5g solver changes. Further per run:

| τ | accepted / candidates | Cauchy | truncated (median fraction) | min slack at end | ‖g_mag‖_{D⁻¹} start → end | windowed max ΔP/P, last window | ΔB (magnetic only) | wall |
|---|---|---|---|---|---|---|---|---|
| 0.9 | 47/47 | 0 | 21 (0.38) | 6.7e-4 | 124.5 → 0.28 | 64 | 0.54 | 236 s |
| 0.99 | 51/51 | 0 | 49 (0.11) | 1.0e-8 | 124.5 → 0.34 | 62 | 0.48 | 231 s |
| 0.999 | 48/48 | 0 | 39 (0.30) | 4.5e-10 | 124.5 → 0.41 | 1.0e4 | 1.1 | 262 s |

(M5f's truncation fractions: median 0.031 (τ 0.9), 0.163 (0.99), **6.6e-139** (0.999).) Throughout: face max ≤ 4.999σ,
face tail 0.564–0.567 % > 3σ (budget 1 %, K = 4727), 0 faces > 5σ, Φ_H ≈ 4.46e-3, Φ_B 0.3249–0.3254, centre max
21.7–22.0σ (unchanged by construction), anchor error ≤ 1e-19, μ at its floor 1e-12 in every window.

**Reading.**
- With the preconditioner repaired, all three τ values complete the budget, with every candidate accepted and no
  truncation stall. M5f's τ = 0.9 and 0.999 collapsed in the same interval from the same start. τ is no longer the
  deciding setting; τ = 0.999 and 0.9 end level (Φ 0.508 / 0.511), τ = 0.99 is 20 % higher.
- **The Cauchy repair is untested in practice:** the GN candidate won every comparison (0 Cauchy selections in 146
  candidates). The improvement is attributable to the preconditioner. The step-selection fix is covered only by the
  synthetic regression.
- **Not converged:** the windowed |ΔP/P| is still 10¹–10⁴, pressure optimality 10²–10³, and max layer RMS 174 vs the
  0.01 target (p99 609 vs 0.1). μ sits at its floor, so damping does not limit the steps. The stagnation rule never fired
  (each window improved force or RMS by ≥ 1 %), but this is 5000 products only.
- For scale only (different work, so not a matched comparison): M5f's single surviving τ = 0.99 run needed 30050
  products to reach Φ 0.432 / RMS 109 / force 604; M5g reached Φ 0.51 / RMS 174 / force 1.4e3 in 5000.
- The barrier λ = 0.1 was not continued (fixed-λ solve, Φ includes Φ_H ≈ 4.5e-3); the barrier-free Φ₀ is logged.

## 4. Centre-model maximum: corrected bracket (`out/m5g/centre_bracket.log`)

- Re-evaluating the saved M5f multiplier with the corrected bound: −0.851 → **+0.851** (λᵀr = −3.675; the M5f script
  saved the multiplier maximizing the wrong-sign bound, so this one is weak).
- The saved M5f witness re-verified: UB 3.945587, CG info 0, ‖Ac − r‖∞ 2.2e-15, mismatch 9e-13σ, div_rel 6.8e-13.
- PDHG warm-started from the M5f iterate, 2500 iterations (3000 s wall):

| it | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|
| LB | 3.318 | 3.462 | 3.560 | 3.602 | **3.629** |
| UB | 3.946 | 3.946 | 3.944 | 3.913 | **3.913** |

Final, re-verified: **t\* ∈ [3.6292, 3.9134]** (LB certificate: the averaged PDHG multiplier at it 2500; UB field: CG
info 0, ‖Ac − r‖∞ 2e-15, mismatch 1.2e-12σ, div_rel 6.4e-13; tail 1.45 % > 3σ). The centre maximum criterion is
feasible (UB < 5), and **no divergence-free field has a centre maximum below 3.63σ**. Numerical, not rigorous rounding.

## 5. Centre-model tail on one fixed support (`out/m5g/centre_tail.log`)

Support S: the K = 4709 largest |c| of the M5f 3.9456σ witness (min |c| on S 3.110); bounds |c| ≤ 4.999 on S, ≤ 2.999
elsewhere; the clipped witness violates them at 1926 entries. Method: PDHG on min (γ/2)‖c‖² + ι_box s.t. A c = r
(γ = 1e-2, diagonal preconditioning, restart to the average every 2000), started from the witness clipped into the box;
physical check (re-projection, reconstruction, H B recomputed) every 10000 iterations; weak-duality ratio
|λᵀr| / Σ ub|Aᵀλ| on the multipliers every 1000. (A first ADMM version with an LU-factored projection was abandoned:
LU of A Aᵀ, 162 708², took 23 min with 1.3e9 fill, then ~160 s per projection.)

| it | 10k | 30k | 60k | 90k | 120k | 130k |
|---|---|---|---|---|---|---|
| ‖Ac − r‖∞/‖r‖∞ (PDHG iterate) | — | 1.5e-3 | — | — | — | ~4e-4 |
| physical centre max | 6.612σ | 6.015σ | 5.699σ | 5.520σ | 5.405σ | **5.375σ** |
| physical > 3σ (K = 4709) | 0.782 % | 0.796 % | 0.783 % | 0.781 % | 0.777 % | **0.776 %** |
| physical > 5σ | 314 | 342 | 320 | 312 | 308 | 307 |

Every physical field passed the checks (CG info 0, ‖Ac − r‖∞ ≤ 2e-14, mismatch ≤ 2e-12σ, div_rel ≤ 1e-12). The best
field (`centre_tail_incumbent.npz`, it 130000), independently re-verified by `verify_field.py`: centre max 5.375σ,
3655 > 3σ (0.776 %, within the tail budget), 307 > 5σ, mean c² 0.362, div_rel 9.5e-13; its face residual is 58σ
(9.7 % > 3σ). The weak-duality ratio never exceeded the local value 0.678 (needs > 1 to prove infeasibility).

**Verdict: inconclusive for this support** (stopped at the 7200 s wall; the final in-script check was interrupted on
request, the incumbent is the result). The maximum fell slowly (≈ 0.03σ per 10k iterations at the end) and the iterate
was still converging; this is neither a feasibility witness nor evidence of infeasibility, and says nothing about other
supports. A field meeting the TAIL criterion alone exists (0.776 %), but not jointly with max ≤ 5σ so far.

## 6. Tests

- `HSEMHS_SKIP_MPI=1 julia -t 8 test/runtests.jl` (hyades11): **EXIT 0**, no failures; the two known broken tests (Gate C3
  collocated, Gate C2 staggered); all 55 M5g assertions pass. At `-t 32` (hyades7) the pre-existing allocation test
  fails (3.2 MB per `fg!` vs its 2 MB budget; 0.22 MB at `-t 8`): the threaded loops allocate per task. `objective!` and
  `gradient!` were not changed; the budget is thread-count dependent.
- Multi-rank MPI tests (n = 2, 4; 32/32 each, including the new limiting-face case): **pass**, when Julia runs from a
  copy on the node's local `/tmp`. From the shared StorNext installation, a rank fails with `failed to map segment from
  shared object` (libgcc_s / libstdc++) and `mpiexec` hangs — the M5f failure; no ulimit explains it. Concurrent Julia
  start-ups from StorNext also killed two of the first three validation launches.
- `test_centre_bracket.py`: 16/16, locally and on the cluster.

## 7. Observation contract — options for the user (no decision taken)

Unchanged facts: the face gate is the specification's; σ and thresholds unchanged.
- **A. Face operator (current formal gate).** Gentle admissible field exists (M5f QP: mean c² 0.3235); the bounded MHS
  run keeps it admissible. Its centre residual is ~22σ.
- **B. Centre operator with a representation-error model.** The released quantities are cell-centred; centre maximum
  t\* ∈ [3.63, 3.91]σ; the tail on one fixed support is unresolved (best: max 5.375σ with tail 0.776 %). Needs a declared coarse-grid transfer model (how face
  values relate to released centre values) before it can become a gate.
- **C. Joint face + centre bounds as an intersection diagnostic** (not a likelihood: both come from the same data). Not
  run. The two witnesses' cross-residuals (22σ, 83σ) are properties of those fields, not proof the sets are disjoint.

## 8. Evidence bundle (local, for independent checks)

`HSEMHS/out/evidence_m5g/` on the local machine (ignored by git), with `MANIFEST.txt` (sizes, sha256):
- `m5g/`: all M5g logs; validation checkpoints `bounded_tau*/ck_*.h5` (restart state in attributes) and `candidates.h5`;
  `probe_audit.log`; `centre_bracket.npz` (+ `_incumbent.npz`, best_lam, LB certificate) and log; `centre_tail_incumbent.npz`
  and log; `verify_m5f_fields.txt`, `verify_m5g_fields.txt`; test logs (`full_test*.log`, `mpi_test*.log`, `test_centre_bracket.log`).
- `m5f/`: centre witness (`centre_bracket.npz`, `centre_witness.h5`, `state_centre_witness.h5`), gentle QP fields
  (`qp_m0.05.npz/.h5`, `qp_m0.001.npz`, `state_qp_m0.05.h5/.txt`), the final τ = 0.99 barrier checkpoint
  (`long_barrier_l0.1_tau0.99_final_barrier_14.h5`), the three long-run logs, the M5f test log.
- `m5d/observations.h5`: the shared observation artifact all Python checks read.

## 9. Reproduce (from `HSEMHS/`, cluster copy, Julia 1.13.1)

1. Tests: `HSEMHS_SKIP_MPI=1 julia --project=. -t 8 test/runtests.jl`; MPI with a node-local Julia copy:
   `julia --project=. -e 'using MPI; …mpiexec -n 2/4 … test/mpi/mpi_tests.jl'`; `python3 -I scripts/m5_bifrost/test_centre_bracket.py`.
2. Audit: `julia --project=. -t 32 scripts/m5_bifrost/m5g_probe_audit.jl out/m5f/qp_m0.05.h5 0.1 1e-3 out/m5f/long_barrier_l0.1_tau*/barrier_*.h5`
3. Validation: `julia --project=. -t 32 scripts/m5_bifrost/m5g_bounded.jl out/m5f/state_qp_m0.05.h5 out/m5f/qp_m0.05.h5 out/m5g/bounded_tauT T 5000 1250 0.1 1e-3 7200` for T ∈ {0.9, 0.99, 0.999}
4. Bracket: `python3 -I scripts/m5_bifrost/centre_bracket.py --obs out/m5d/observations.h5 --out out/m5g/centre_bracket.npz --reeval out/m5f/centre_bracket.npz --warm out/m5f/centre_bracket.npz --wall 3000 --report 500`
5. Tail: `python3 -I scripts/m5_bifrost/centre_tail.py --obs out/m5d/observations.h5 --witness out/m5f/centre_bracket.npz --out out/m5g/centre_tail.npz --wall 7200` (run under an external wall guard)
6. `python3 -I scripts/m5_bifrost/verify_field.py out/m5d/observations.h5 FIELD.npz …`
