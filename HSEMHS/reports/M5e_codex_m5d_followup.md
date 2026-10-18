# M5e: Codex's M5d review — steps 1–5 (18 Oct 2026)

**Follows** `CODEX_REVIEW_M5d_2026-10-18.md`. The user asked for Codex's five steps to be carried out and for the old min-t calculation to be stopped (stopped).

**Code:**
- `src/MHSJointGN.jl`: `anchor_tangent` (projected PCG).

**Scripts:** `scripts/m5_bifrost/`
- `export_observations.jl`, `data_lp.py`, `make_support.py`;
- `geometry_experiment_v2.jl`, `long_comparison.jl`;
- `data_feasibility.jl` now saves its per-stage witnesses.

**Outputs:** `out/m5d/` (ignored by git).

## Step 1: corrections (in `reports/M5d_review_followup.md`)

- The centre model is **unresolved**, not refuted. The 6.55σ witness is only an upper bound, and face and centre values don't rank the data's consistency.
- The preconditioning conclusion is qualified: at this checkpoint the layer-diagonal estimator is not the cause; other states and scales are not ruled out.
- Cell-centring is backed by provenance as cited by Codex: release `DATA_LEV = 2`, Carlsson et al. 2016, A&A 585, A4, §4–5, and the Bifrost FITS documentation. I have not read those sources independently.

## Step 2: shared observation artifact

- `export_observations.jl` writes exactly the arrays the Julia solver uses: face observations and σ, centre observations and σ, and the metrics.
- `data_lp.py` reads only that artifact and verifies every solution independently of its LP matrix. It rebuilds the physical field, recomputes its divergence from the metrics, recomputes the residuals, and reports the margins.
- It reproduces Codex's check: face maximum **4.9999σ feasible** (16 s; margin +1.0e-4; divergence 4e-13 relative).
- At exactly t = 5, 9429 faces lie on the bound by round-off (−2.7e-15). Strict thresholds need margins.

## Step 3: data tests

**Saved witnesses.** `data_feasibility.jl` now saves every stage's field (`out/m5d/witness_face_p*.h5`). The p = 128 face witness has max 5.126σ, 0.704 % > 3σ, 47 faces > 5σ (as before).

**Face model: maximum and tail together (Codex's fixed-support LP).**
- N = 472 752 faces; K = ⌊0.01 N⌋ = 4727.
- Support S: the p = 128 witness's |c| > 3 set (|S| = 3328 ≤ K).
- Bounds: |c| ≤ 5 − m on S and |c| ≤ 3 − m elsewhere, plus div B = 0.

| Support | margin m | LP | max \|c\| | > 3σ | # > 5σ | divergence (relative) |
|---|---|---|---|---|---|---|
| p = 128 \|c\| > 3 set, 3328 faces | 1e-6 | **feasible**, 13 s | 4.999999σ | **0.607 %** | 0 | 3.4e-13 |
| same | **1e-3** | **feasible**, 16 s | **4.999σ** | **0.615 %** | 0 | 3.5e-13 |
| p = 128 top-K, 4727 faces | 1e-6 | feasible, 16 s | 4.999999σ | 0.846 % | 0 | 3.8e-13 |

**Under the face model, a divergence-free field meeting both data-admissibility criteria exists:** maximum ≤ 5σ and ≤ 1 % above 3σ, with a 1e-3 margin. It is a numerically verified witness (HiGHS interior point; verified from the physical field), not an exact-arithmetic certificate. The witness field is saved in `out/m5d/support_above3_m1e-3.npz`.

Codex's earlier top-K support from the min-L1 field was infeasible; the p = 128 support is what works. The same field has a centre-model maximum of 23σ: the two observation models disagree strongly about what "close to the data" means.

**Centre model, maximum only (t = 5).** Still running after more than 1.5 h, with no verdict. The face LP took 16 s; this one has about 9.4e5 two-sided residual rows. Centre-model feasibility therefore remains **unresolved**.

**Hinge search.** Not needed for the face model: the fixed-support LP gave a constructive positive result.

## Step 4: step geometry with an explicit common radius (Δ0 = 1.105 for every joint arm)

| One candidate | predicted | actual | ρ | shares u, y → final | max ΔP/P |
|---|---|---|---|---|---|
| joint (default) | 0.1187 | 0.0149 | 0.13 | 0.53 → 0.63 | 85 |
| magnetic only | 0.1179 | 0.1164 | 0.99 | 0 | 0 |
| ×10 | 0.0983 | 0.0989 | 1.01 | 0.15 → 0.72 | 16 |
| ×100 | 0.0586 | 0.0586 | 1.00 | 0.02 → 0.92 | 3.1 |
| joint + anchor-tangent | 0.1070 | −0.0743 | −0.70 | 0.52 → 0.32 | 147 |
| **×100 + anchor-tangent** | 0.1168 | **0.1152** | **0.99** | **0.02 → 0.01** | 147 |

| Continuation, 1000 products, common Δ0 | Φ | raw force | max RMS | accepted / candidates | work |
|---|---|---|---|---|---|
| joint (default) | 0.9610 | 6014 | 366 | 17/23 | 1010 |
| joint + anchor-tangent | 1.0689 | 7094 | 411 | 21/27 | 1013 |
| ×10 | 0.8121 | 4522 | 306 | 9/12 | 1106 |
| magnetic only | 0.7224 | 3624 | 289 | 9/9 | 1106 |
| ×100 + anchor-tangent | 0.7088 | 3489 | 283 | 9/9 | 1106 |
| **×100** | **0.6838** | 3237 | 273 | 9/9 | 1099 |

- The ×100 continuation **reproduces Codex's common-radius rerun exactly** (Φ = 0.683823). The earlier ×100 result (0.724) also had a 10× larger starting radius.
- **Anchor-tangent PCG removes the projection inflation** Codex identified: with ×100, the pressure share stays 0.02 → 0.01 instead of 0.02 → 0.92, and the single-step actual decrease doubles (0.059 → 0.115).
- In the default geometry it does **not** help: the step is rejected. The pressure step is then simply too large, and per-node caps and bounds clip it.
- Over a continuation, ×100 alone is slightly better than ×100 + anchor-tangent.
- Codex's caution holds: even accepted steps can change P by ×147 at individual nodes.

## Step 5: longer matched comparison (3000 equivalent products, three starting states)

**Starts:**
- the M5c checkpoint (Φ 1.2953);
- the alternation's M5c endpoint (0.6275);
- the joint GN's M5c endpoint (0.7949).

All joint arms use that start's default-rule radius as a common Δ0. Actual work was 3019–3115.

| Arm | checkpoint | alternation endpoint | joint endpoint |
|---|---|---|---|
| alternation (L-BFGS Ã, ln P blocks) | 0.6275 | 0.5284 | 0.5555 |
| alternation, magnetic trust-region block + pressure blocks | 0.5737 | 0.5033 | 0.5298 |
| **joint ×100** | **0.5218** | 0.4848 | 0.5018 |
| joint, per-node \|δu\| ≤ 0.5 | 0.5616 (stalls from ~2000 work; 10 Cauchy) | **0.4690** | **0.4939** |
| joint, anchor-tangent | 0.8700 | 0.5783 | 0.6968 |
| joint ×100 + anchor-tangent | 0.5669 | 0.5011 | 0.5239 |

Other quantities at the endpoints, over all arms and starts:

| Quantity | Value |
|---|---|
| max layer RMS \|R\|/S | 151–345 (real-data target 0.01) |
| data statistic | 0.3175–0.3183 |
| max \|δB\|/σ | 11.149–11.162 |
| bound margin | ≥ −5e-14 |
| block failures (alternation) | 0 |
| fresh pressure optimality | 93 – 5160 |

The lowest pressure optimality came from the magnetic-trust-region alternation and from ×100 + anchor-tangent.

**Reading:**
- **Joint ×100 is the most consistent arm:** best from the checkpoint and second from the other two starts.
- The per-node pressure cap wins from two starts, but stalls from the checkpoint with a large Ã gradient.
- Both alternations trail every pressure-limited joint arm at this budget. Of the two, the magnetic-trust-region alternation is better.
- Anchor-tangent alone is worst everywhere.
- **No arm converges.** The pressure audits are not decreasing consistently, and the force residual is two to four orders above the real-data tolerance.
- The magnetic outlier does not move in any arm, as expected: these solvers keep the least-squares data term.
