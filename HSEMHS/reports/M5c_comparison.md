# M5c: diagnostics, floor-label policy, joint GN vs alternation (18 Oct 2026)

**Follows** `CODEX_REVIEW_M5_SECOND_FOLLOWUP.md`. The user approved all three steps: diagnostics; `stalled_at_floor` with Codex's four conditions, `success_mhs` unchanged; and the matched-budget comparison.

**Code:**
- `src/MHSAudit.jl`: `reference_metric`, `full_audit`.
- `src/MHSAlternate.jl`: `atilde_audit`; PCG preconditioned residual and exit reasons; audits always computed; `stalled_at_floor`; work budget; callback.
- `src/MHSJointGN.jl`: `joint_gn_solve!`.
- `src/MHSStaggered.jl`, `src/MHSGaussNewton.jl`: `WORK_COUNTS`, `work_equivalent`.

**Script:** `scripts/m5_bifrost/compare_solvers.jl`. **Outputs:** `out/m5c/` (ignored by git).

## 1. Diagnostics and policy

| Item | Implementation |
|---|---|
| Work accounting | objective + gradient = 1, GN product = 1, bare objective = ½. Cauchy trials, diagonal builds and the residual recomputation are all charged. Diagnostics in callbacks are not charged. |
| Fixed-metric audit | D_ref at the common checkpoint: GN diagonal, floored at 1e-6 × median per field. Pressure correction s* = argmin gᵀs + ½sᵀD_ref s subject to the bounds and the anchor, **without** trust radius or cap. Ã and b̄ diagonal corrections, with b̄ in σ_B units. The full-state measure m is the predicted diagonal-model decrease, in units of Φ. Reported with quantiles, the fixed-mask split, and the worst nodes (g, D, P/P_ref, bound distance, active status). |
| Magnetic audit | `atilde_audit`: ‖∇_Ã Φ‖ and σ_B·|∂Φ/∂b̄| reported separately. This replaces the mixed-unit norm. |
| PCG | Relative preconditioned residual (coordinate-invariant). Exit reasons recorded (residual / boundary / negcurv / budget). The residual is recomputed for the returned direction after a boundary exit. |
| Outer audits | Computed whenever the change test passes, also after block failures. |
| `stalled_at_floor` | Only if all four conditions hold (spec §10, 18 Oct 2026). Production supplies no floor evidence, so it cannot obtain the label. |
| Joint GN | Damped trust-region GN on (v, Ã, b̄): base-state model, separate trial workspace, exact pressure projection, whole final step scaled into the D-ball, projected-Cauchy fraction 0.1, radius updated from the final step. The Ã gauge null space is handled by Levenberg damping. |

**Tests (new, pass):**
- The reference metric is positive; the audit is finite and non-negative.
- Joint GN: accepted steps lie in their ball and are feasible; Φ is monotone; the work budget holds to within one iteration; the audit decreases on the small problem.
- No `stalled_at_floor` without floor evidence; the label is obtained when all conditions are forced.
- Full suite: see the commit.

## 2. C1b (8×8×65, synthetic), 3000 equivalent products

| Arm | Φ | Max layer RMS \|R\|/S | \|u − u_s\| | Audit m start → end |
|---|---|---|---|---|
| Alternation (ln P), before the per-block budget cap (overshot to 6734 work) | 1.6e-12 | 1.0e-5 | 3.3e-6 | ÷1.9e7 |
| **Joint GN** (3054 work) | **2.5e-13** | **1.5e-6** | **1.4e-6** | **÷7.5e10** |

With a gate floor check (RMS ≤ 1e-6, Φ ≤ 1e-14), the alternation was correctly **not** labelled `stalled_at_floor`. On this synthetic case the joint GN is clearly better.

## 3. Bifrost (coarsened 42×42×89), 3000 equivalent products each

**Setup:**
- The same recreated checkpoint (λ_F = 1e-4, λ_Q = 1e-3; one corrected alternating cycle; Φ = 1.2953).
- Collapse bound u_init − 3; D_ref and the identifiability mask from the checkpoint.
- Alternation: Ã block 100 L-BFGS iterations, then ≤ 5 accepted pressure steps or optimality ÷2, in ln P.
- Joint GN: CG ≤ 100, preconditioned rtol 1e-2.

| | Start | Alternation (ln P) | Joint GN |
|---|---|---|---|
| work / wall time | — | 3052 / 136 s | 3014 / 120 s |
| counts | — | objective 2198, adjoint 3004, GN 902; 17 cycles; 0 Ã-block and 0 pressure-block failures | objective 873, adjoint 2941, GN 2215; 65 accepted of 81 candidates; 4 Cauchy fallbacks; CG exits all on the trust boundary |
| Φ | 1.2953 | **0.6275** | 0.7949 |
| raw force (Φ_F / λ_F) | 9353 | **2675** | 4354 |
| max layer RMS \|R\|/S | 476 | **243** | 312 |
| data statistic | 0.3180 | 0.3178 | 0.3175 |
| Q | 9.49e-4 | 1.00e-3 | 9.55e-4 |
| max \|δB\|/σ | 11.16 | 11.16 | 11.16 |
| **audit m** (full state) | 0.290 | 8.26 | 48.8 |
| — pressure part | 0.0115 | 8.25 | 48.8 |
| — Ã part | 0.278 | **6.5e-3** (÷43) | 7.3e-2 (÷3.8) |
| — b̄ part | 8.6e-6 | 6.7e-9 | 2.5e-7 |
| u* max, identifiable / rest | 0.078 / 846 | 0.072 / 3.8e5 | 5.9 / 9.8e5 |
| ‖∇_Ã Φ‖; σ_B·\|∂Φ/∂b̄\| | 71.5; 0.075 | 9.1; 2.1e-3 | 19.9; 1.3e-2 |
| min bound margin; anchor error | 4.3e-2; 0 | 1.3e-4; 5e-15 | −3.2e-14; 4e-15 |

**Trends:**
- Φ decreases monotonically in both arms.
- In both, m first rises by 1–4 orders of magnitude, through its pressure part, and then fluctuates. Alternation: 6.9 → 26 → 48 → 60 → 32 → 12 → 8.3. Joint: 8.7e3 → 373 → 205 → 462 → 49.

## 4. Verdict

**Numerical pass (Codex Q3: ≥ 10× reduction in a common full-state measure, decreasing objective, no violations): neither arm passes.**
- Both keep feasibility and the trust ball, and neither has a solver failure disguised as convergence.
- But the full-state measure m **increases** in both, entirely through the pressure part on the weakly constrained complement of the identifiability mask. There the uncapped correction −g/D_ref reaches 10⁵–10⁶ in ln P.

**At matched work on Bifrost, the alternation is better on every measure:** Φ, raw force, layer RMS, the Ã part of the audit (÷43 vs ÷3.8), and the identifiable pressure audit (flat, against a ×75 increase for the joint GN). This is the **opposite of C1b**.

So on this case the coupled direction alone does not appear to be the missing ingredient. One interpretation, not yet established: the complement pressures are poorly conditioned in any method, and a joint step spends its trust region on them.

**Not established:**
- whether the alternation would continue to improve;
- whether a different joint preconditioner (for example the exact colored Ã diagonal) would change the order;
- whether the rise in the fixed-metric pressure audit reflects the problem or the metric. D_ref is frozen at a checkpoint where P was up to 10⁴ times smaller on those nodes.

**Unchanged throughout:** the data statistic (0.318) and the magnetic outlier (11.16σ, which fails `data_admissible` independently).
