# M5d: follow-up to Codex's M5c review (18 Oct 2026)

**Follows** `CODEX_REVIEW_M5c_2026-10-18.md`. The user asked for all of it to be implemented.

**Code:**
- `src/MHSStaggered.jl`: `precond_atilde = :layer_gn`, and corrected labels for the default estimator.
- `src/MHSJointGN.jl`: residual recomputed after a boundary exit; per-candidate block diagnostics; `Δ_init`, `Δ0`, `block_scale`, `magnetic_only`, `max_accepted`, `max_candidates`.
- `src/MHSAudit.jl`: variable-size gathers; globally selected worst nodes.

**Scripts:** `scripts/m5_bifrost/{stagger_test,data_feasibility,pooled_audit,geometry_experiment}.jl`.

**Outputs:** `out/m5c/` (ignored by git).

## 1. Codex's saved-state review reproduces exactly

`codex_m5c_result_review.jl` reproduces Codex's 3×3 common-metric table, the outliers (11.17σ at Bz face (35, 32, 89); 9.39σ at (35, 32, 88)) and both necessary bounds (local divergence t ≥ 1.920; plane mean t ≥ 4.6e-5).

## 2. Code findings, all confirmed and addressed

| Finding | Fix |
|---|---|
| The default Ã layer diagonal differences ∇Φ: it is the **full** Hessian, not GN | Labels corrected; `:layer_gn` added (same probes, `gn_hvp!`). **Measured at the checkpoint:** the GN and full layer means agree to 0.1 % (Ã_x, Ã_y) and 0.8 % (Ã_z); build cost is 7 vs 11 products. **The preconditioner is not the cause of the joint/alternation gap** (see §4). |
| Joint radius initialized from the pressure diagonal alone | `Δ_init = :full` option (and `Δ0`); tested in §4 |
| Joint `cg_rel` stale after a boundary exit | The residual of the returned direction is recomputed |
| Joint lacks block diagnostics | Per-candidate D-norm shares (pressure / Ã / b̄) before and after projection, actual max ΔP/P and ΔB, and the pressure / magnetic / cross split of the model; not charged to the work count |
| `Allgather` with equal-size buffers; worst nodes selected per rank | `Allgatherv`; worst nodes selected globally |

**Tests (new, pass):**
- The GN layer diagonal is positive, differs from the full estimate, and leaves the pressure part unchanged.
- `magnetic_only` leaves P exactly unchanged.
- The D-norm shares sum to 1.
- Block-scaled and full-radius steps stay feasible and inside their ball.

## 3. Post-hoc common-metric sensitivity (declared as such; the original verdict stands)

| Metric (one per row) | checkpoint | alternation end | joint end |
|---|---|---|---|
| checkpoint (original, preregistered) | 0.290 | 8.26 | 48.8 |
| alternation end | 0.717 | 0.00775 | 0.388 |
| joint end | 0.821 | 0.108 | 0.0746 |
| **pooled (elementwise max of the three)** | **0.281** | **0.00720 (÷39)** | **0.0739 (÷3.8)** |

- **Under the original metric neither arm meets the tenfold test (recorded failure).**
- Under the symmetric pooled metric, the alternation improves ÷39 and the joint GN ÷3.8.
- As Codex showed, the original audit's correction (up to 3.8e5 in ln P) lies far outside any local linearization: the checkpoint diagonal at the worst node is 8.1e6 times smaller than the current one. So it cannot be read as an attainable decrease.

## 4. Step geometry (frozen model at the checkpoint, common radius Δ = 1.105)

| One candidate | predicted | actual | ρ | D-norm share u → final | max ΔP/P | model: pressure / magnetic / cross |
|---|---|---|---|---|---|---|
| joint (default) | 0.1187 | 0.0149 | 0.13 | 0.53 → 0.63 | 85 | 0.041 / 0.062 / 0.015 |
| joint, GN layer diagonal | 0.1186 | 0.0088 | 0.07 | 0.53 → 0.62 | 88 | 0.041 / 0.062 / 0.016 |
| **magnetic only** | 0.1179 | **0.1164** | **0.99** | 0 | 0 | 0 / 0.118 / 0 |
| pressure stiffer ×10 | 0.0983 | 0.0989 | 1.01 | 0.15 → 0.72* | 16 | 0.009 / 0.088 / 0.001 |
| pressure stiffer ×100 | 0.0586 | 0.0586 | 1.00 | 0.02 → 0.92* | 3.1 | 0.002 / 0.057 / 0.0002 |
| Δ from the full D-norm (own, larger radius) | 0.188 | −15.0 | −80 | 0.56 → 0.49 | 147 | — (rejected) |

\* Shares are measured in the scaled norm, so the ×10 and ×100 rows weight pressure up by those factors.

| Continuation, 1000 equivalent products | Φ | raw force | max layer RMS \|R\|/S | accepted / candidates |
|---|---|---|---|---|
| joint (default) | 0.961 | 6014 | 366 | 17 / 23 |
| joint, GN layer diagonal | 0.941 | 5814 | 358 | 21 / 26 |
| alternation (ln P) | 0.828 | 4679 | 326 | 6 cycles |
| **pressure stiffer ×100** | **0.724** | **3636** | **290** | 8 / 8 |
| **magnetic only** | **0.722** | **3624** | **289** | 9 / 9 |

**This supports Codex's radius-allocation hypothesis.**
- In the default joint step the pressure part takes over half the trust ball (0.53 → 0.63 after projection), proposes pressure changes of ×85, and is badly modelled (ρ = 0.13). It suppresses a magnetic correction that alone achieves ρ = 0.99 and 8× the actual decrease.
- Making pressure stiffer in the step geometry (×10–×100) restores ρ ≈ 1. At 1000 products the stiffened joint method matches magnetic-only and beats both the default joint method and the alternation.
- A larger shared ball (the full-norm rule) is much worse.
- The diagonal estimator (full vs GN) makes no material difference.

**Not established:**
- Over longer budgets, magnetic-only progress must eventually be followed by pressure updates: it is a diagnostic, not a solver.
- The best block scale and its dependence on the state are open.

## 5. Magnetic data feasibility

### 5.1 The input B is cell-centred

`stagger_test.jl` gives the discrete divergence of the full-resolution input, as |div B| h/|B|:

| Reading of the data | p50 | p99 | max |
|---|---|---|---|
| Staggered faces (either vertical convention) | 4.7–4.8e-3 | 0.41 | 34–40 |
| **Cell-centred (central differences)** | **1.95e-3** | **0.17** | **16** |

The data are cell-centred, apparently interpolated by the data release, and **not** discretely divergence-free on either mesh. The present pipeline then block-averages (12×12 horizontally, every 2nd level) and interpolates to faces. At the bottom and top z-faces it takes the nearest centre value. That is not a flux-preserving transfer.

### 5.2 How small can max |B − Bobs|/σ be for a divergence-free field?

`data_feasibility.jl` minimizes the L_p mean of the normalized residual over (Ã, b̄) (B = curl A + b̄ ẑ, so div B = 0 exactly), with p continued 2 → 128 at 400 iterations per stage. Each stage gives a **witness**, an upper bound on the minimum t. The lower bound is 1.92 (Codex's local divergence bound).

| Observation model | least-squares initializer: t / > 3σ / # > 5σ | witness p = 128: t / > 3σ / # > 5σ | worst location |
|---|---|---|---|
| face, `B_f − I·y` (current) | 11.17 / 0.61 % / 341 | **5.13** / 0.70 % / 47 | moves from the bottom outlier (35, 32, 89) to (33, 14, 88) |
| centre, `H B_f − y` | 21.1 / 2.25 % / 2660 | 6.55 / 1.61 % / 1140 | (17, 26, 84), interior |

(A longer face run, 1500 iterations per stage up to p = 256, is in §5.3.)

- **The 11σ outlier is not intrinsic to the data.** It is an artifact of fitting by least squares: a divergence-free field exists with max error ≤ 5.13σ and 0.70 % > 3σ under the face model. The tail limit (≤ 1 %) is met; whether the maximum (≤ 5σ) can be met is not yet decided (5.13 > 5).
- **Under the centre model** (the observation residual for cell-centred data) the witnesses are much worse: 6.55σ and 1.6 % > 3σ, failing the 1 % tail limit. Face interpolation of the data smooths a centre-level inconsistency; the data are not discretely divergence-free.
- **A bound-constrained data fit** (L_∞-type, as here) is required to reach admissibility at all. The least-squares A₀ does not enforce the maximum or the tail.

### 5.3 Exact LP (added 15:20)

`scripts/m5_bifrost/data_feasibility_lp.py` solves Codex's face-variable formulation exactly (SciPy/HiGHS interior point). It reproduces the necessary bound t ≥ 1.920000, and every solution is verified from the returned field.

| LP | result | solve time | div residual | max \|c\| | > 3σ | # > 5σ |
|---|---|---|---|---|---|---|
| feasibility at t = 5 | **feasible** | 16 s | 1.1e-15 | 5.000 | 88.9 % (vertex) | 0 |
| min Σ\|c\| subject to \|c\| ≤ 5 | optimal | 71 s | 5.6e-16 | 5.000 | **2.04 %** | 0 |
| exact min t | running | | | | | |

- The 5σ maximum is attainable under the face model.
- No field found so far meets both the maximum and the 1 % tail. The L_p witness had 0.70 % above 3σ but a 5.13σ maximum.
- The longer L_p run was stopped as superseded.

## 6. What this changes

1. **Data admissibility is attainable or nearly attainable under the face model, but only with a max-norm fit, not with least squares.** Before the next solver step, the data term or initializer should be decided: keep least squares and accept that outliers are a model or objective mismatch; or add a bound-constrained initializer or data term.
2. **Under the centre model the data are not admissible at 5σ.** This is a statement about the data and its transfer (cell-centred, not divergence-free, block-averaged), not about the solver.
3. **The joint GN's weakness is step geometry, not preconditioning.** A pressure-stiffened block scaling fixes the single-step quality and the short-budget ranking.
4. **The pressure prior remains a scientific decision for the user.** Nothing here forces it; the data question comes first.
