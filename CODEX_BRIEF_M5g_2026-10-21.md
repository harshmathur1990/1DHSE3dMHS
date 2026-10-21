# Brief for Codex — M5g (21 October 2026)

*Responds to `CODEX_REVIEW_M5f_2026-10-19.md` and the consolidated `REVIEW_M5f_2026-10-21.md`. The user asked for the
review to be implemented, committed, pushed and run on the cluster. Full write-up: `HSEMHS/reports/M5g_codex_m5f_followup.md`.
Previous brief: `CODEX_BRIEF_M5f_2026-10-19.md`.*

## 1. What changed (commit `ae47d13`, pushed to `origin/main`)

- **Finding 1, curvature probes.** `gn_diagonal!` keeps every probe off the barrier: the Ã layer means and the b̄ central
  difference are taken with the face term switched off. The face term's exact local diagonal, Cᵀ diag(h) C, is then
  added from the base state (`face_curvature!`, `face_curvature_diag!`): squared curl-stencil coefficients for Ã, Σ h
  over z-faces for b̄. It **replaces** the averaged part; nothing is double counted. The old estimator survives only as
  `barrier_safe = false`, for the audit.
- **Finding 2, bounded step.** `joint_gn_solve!`:
  - The GN candidate and every Cauchy trial are made face-feasible **before** their model decreases are compared. The
    fraction comes from the exact linear map δB = A_ref curl δÃ + δb̄ ẑ.
  - The Cauchy direction −D⁻¹g is barrier-scaled through the new D.
  - Diagnostics are computed on the final step.
  - Inputs are validated (0 < τ < 1, finite positive bounds, strictly feasible start under a barrier), and the bounds
    are re-checked after the trial.
  - A callback returning `:stop` gives a declared stagnation stop; μ is returned; per-candidate truncation records are kept.
- **Findings 3 and 6, `centre_bracket.py`.**
  - The lower bound is now |λᵀr|/‖Aᵀλ‖₁, seeded with the local bound (its row index is the certificate).
  - Each projection records CG's exit code and ‖Ac − r‖∞.
  - A UB is accepted only if the field is finite and its relative divergence is ≤ 1e-9.
  - The incumbent field is saved at every improvement.
  - Labels say "numerically verified", not "certified".
  - `verify_field.py` gained the same gates and an exit code.
- **Finding 5, ΔB log.** `measured_change_P` / `measured_change_B`; the pilot and the solver diagnostics now log
  magnetic change only.
- **Finding 4, M5f report.** The matched-budget claim, the "CG does not see the curvature" wording, the "certified"
  label and the ΔB column are corrected in place, with M5g notes.
- **Finding 7, tests and API.** `set_face_hinge!` and `joint_gn_solve!` validate their inputs. New tests:
  - `test/test_face_bounds.jl` (55 assertions), including your reproducer as an assertion and the barrier Hessian
    against finite differences near the bound;
  - an MPI case with the limiting face on a non-zero rank;
  - `test_centre_bracket.py` (16 checks: scalar, two LPs, a known field, invalid and NaN fields, a failed projection).
- **New scripts.** `m5g_bounded.jl` (validation driver with predeclared stops and full restart state), `m5g_probe_audit.jl`,
  `centre_tail.py`.
- **Wording.** `data_qp.py`'s "KKT gap" wording is removed.

## 2. What ran, and where

- **Cluster copy.** `/mn/stornext/d9/data/harshm/MHS` had no git remote; I added `origin` (GitHub) and fast-forwarded it
  to `ae47d13`.
- **Toolchain.** Julia 1.13.1 (the Manifest's version), Python `~/env`.

| Node | Job | Wall |
|---|---|---|
| hyades10, hyades5 | Bounded validation at τ = 0.9, 0.99, 0.999 (`-t 32`). τ = 0.999 ran on both nodes, see §4 | ~4 min each |
| hyades11 | Probe audit on all 20 saved M5f barrier checkpoints; full test suite at `-t 8` | ~15 min; ~40 min |
| hyades7 | Full suite at `-t 32`; MPI tests on 2 and 4 ranks (passed from a node-local Julia copy) | — |
| hyades21 | Bracket re-evaluation and warm PDHG (3000 s); centre-tail PDHG (7200 s wall, ended there; its final in-script check was interrupted on request — the saved incumbent is the result) | 56 min; 2 h |

## 3. Results and interpretation

**(a) The defect you reproduced was active throughout M5f's bounded runs.**
- At every M5f checkpoint whose minimum face slack was ≤ 1.4e-5σ (8 of 20), **4–8 of the 8** old Ã probes and up to 2/2
  b̄ probes left the barrier domain.
- Against unit-vector GN products, the old estimates were badly wrong:
  - b̄ curvature as low as **1.8e-6 ×** exact;
  - the Ã diagonal at nodes touching the minimum-slack face had median ratio **0.023**, and fell to the 1e-300 floor
    where the exact value was 3.3e7.
- M5g: b̄ 1.000 at all 20 checkpoints; Ã median 1.01 (max 4.2, where the barrier-free layer mean overestimates —
  identical in both estimators).
- The checkpoints with slack 6.5e-4 had no outside probes. This matches where M5f collapsed.

**(b) Short robustness validation: all three τ complete, none collapses.** Same start, same settings and Δ0 as M5f,
matched work:

| τ | Φ at ~2500 (M5g / M5f) | Φ at ~5000 (M5g / M5f) | max layer RMS at ~5000 | M5f outcome |
|---|---|---|---|---|
| 0.9 | 0.617 / 1.445 | **0.511** / 1.268 | 174 / 472 | collapsed at 5034 |
| 0.99 | 1.377 / 2.851 | **0.624** / 2.056 | 240 / 670 | — |
| 0.999 | 0.572 / 5.451 | **0.508** / 4.158 (4408) | 174 / 1090 | collapsed at 4408 |

- All candidates were accepted: 146/146.
- Median truncation fraction 0.11–0.38; M5f's τ = 0.999 had 6.6e-139.
- Faces stayed admissible throughout: max ≤ 4.999σ, tail 0.565 %.
- The magnetic stationarity ‖g_mag‖_{D⁻¹} fell from 124.5 to 0.28–0.41.
- No stagnation stop fired.

**But:**
- The **Cauchy safeguard was never selected** (0/146), so the improvement belongs to the preconditioner; the step-selection
  repair is exercised only by the synthetic regression.
- The solve is **far from converged**: windowed |ΔP/P| 10¹–10⁴, pressure optimality 10²–10³, RMS 174 against the 0.01
  target, μ at its 1e-12 floor.
- For scale only (not matched work): M5f's surviving run needed 30050 products for Φ 0.432 / RMS 109.

**(c) Centre maximum: t\* ∈ [3.629, 3.913]σ** (was [−0.85, 3.946]).
- Re-evaluating the saved M5f multiplier gives +0.851. That multiplier is weak because it was selected under the wrong
  sign.
- A warm PDHG run (2500 iterations) raised the LB 3.32 → 3.63 and found a better witness, 3.913σ. Its checks: CG info 0,
  ‖Ac − r‖∞ 2e-15, reconstruction mismatch 1e-12σ, div_rel 6e-13, tail 1.45 %.
- No divergence-free field has a centre maximum below 3.63σ (numerically).

**(d) Centre tail on the fixed support** (K = 4709 largest of the witness; 4.999 on S, 2.999 elsewhere): **inconclusive**.
- PDHG ran to the 2 h wall (131k iterations); the user asked to stop rather than extend it.
- Best physical field (re-verified): centre max **5.375σ**, tail **0.776 %** (within budget), but 307 cells > 5σ; face
  residual 58σ.
- The maximum was still falling, ≈ 0.03σ per 10k iterations.
- The weak-duality ratio never exceeded the local 0.678, so there is no infeasibility evidence.
- Neither verdict holds for this support, and nothing follows for other supports.

**(e) Tests.**
- The suite at `-t 8`: EXIT 0, only the two known broken tests remain.
- MPI n = 2, 4: 32/32 each, from a node-local Julia copy.
- Python checks: 16/16.

## 4. Problems, uncertainties, failed attempts

- **ADMM for the tail was abandoned.** My first `centre_tail.py` used ADMM with an LU-factored projection. The LU of
  A Aᵀ (162 708²) took 23 min with 1.3e9 fill, and a single preconditioned projection then took ~160 s. I replaced it with
  PDHG, which needs products only. The tail result therefore comes from a first-order method run to a wall: a
  non-feasible end is **inconclusive**, not a verdict.
- **Allocation test.** At `-t 32` the existing allocation test fails (3.2 MB per `fg!` against a 2 MB budget; 0.22 MB at
  `-t 8`). `objective!` and `gradient!` were not changed; the budget depends on the thread count. Tell me if it should be
  per-thread.
- **Cluster loader errors.** Concurrent Julia start-ups from StorNext fail intermittently with `failed to map segment
  from shared object`. This killed two of the first validation launches and every multi-rank test run from the shared
  install. A node-local copy of Julia avoids it.
- **Duplicated τ = 0.999 run.** It ran twice: the hyades10 relaunch succeeded after an apparent failure, and a hyades5
  copy truncated the same log. Both reported identical Φ, counts and fractions; the files on disk are one consistent run.
- **Ã diagonal outside the face term.** It is still the per-layer Hutchinson mean of the barrier-free Hessian. Only the
  face term is now spatially resolved.
- **Barrier λ.** It stayed fixed at 0.1; no continuation was run.
- **Not done.** No nullspace/roughness audit of the centre witness, and no joint face + centre intersection run.
- **Julia version.** The user pointed to `~/loadgcc.sh` (module julia/1.12.3). The project's Manifest is 1.13.1, so all
  runs used 1.13.1.

## 5. What Codex should review

1. **`face_curvature_diag!`.** Are the stencil and the A_ref² factor right, given the `curlA_T!` coefficients? Is it
   right to replace, not add to, the barrier-free probe estimate in all three `pc_atilde` modes, `:colored` included?
2. **Step selection in `joint_gn_solve!`.**
   - Is truncating both candidates and comparing post-truncation `pred` adequate?
   - Is a barrier-scaled −D⁻¹g a sufficient "constraint-aware" Cauchy direction, given it was never selected? Or should
     a projected/tangent direction be built?
3. **Is the validation conclusive enough to choose a continuation?**
   - Which τ?
   - Should the next step be barrier continuation (λ ↓ after inner solves), or a longer fixed-λ run under the same
     predeclared stops?
   - Should μ's floor or the pressure block (ΔP/P still 10¹–10⁴) come first?
4. **The bracket's LB mechanics.** Are |λᵀr|/‖Aᵀλ‖₁ on PDHG iterates, the absolute CG floor in `project`, and the gating
   in `physical` sound? Is [3.63, 3.91] worth tightening further, or is it enough?
5. **The tail test design.** PDHG with γ‖c‖²/2, the weak-duality ratio checked on multipliers, the clipped-witness start —
   and how to read its result (§3d).
6. **The observation-contract options in the report, §7** (face gate; centre operator with a transfer model; a joint
   diagnostic). Is anything missing before they go to the user?
7. **The evidence bundle.** It is in `HSEMHS/out/evidence_m5g/` locally, with `MANIFEST.txt`: the M5f centre witness, the
   gentle QP field, the M5f final barrier checkpoint, all logs, the M5g checkpoints, the bracket and tail outputs, and
   the verification output. It enables an independent endpoint check.

## Reproduce

See the report, §9.
