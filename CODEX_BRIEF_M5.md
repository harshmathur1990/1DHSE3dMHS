# Brief for Codex: real-data convergence of the staggered `mhs` solver (M5)

*Prepared 17 Oct 2026 by Claude for an independent review by Codex.*

**Read first, in order:**
1. `HSE_MHS_FINAL_SPEC.md`: §6 `mhs` and the §10 decision log.
2. `HSE_MHS_IMPLEMENTATION.md`: §13.
3. `HSE_MHS_D9_STAGGERED_ADDENDUM.md` and `HSE_MHS_D9b_AVERAGING_ADDENDUM.md`: the staggered scheme and Φ_Q.
4. Reports `HSEMHS/reports/M6d_step{1,2,3}.md` and `M5a_realdata.md`.

**Code:** `HSEMHS/src`: `MHSStaggered.jl` (objective, adjoint, `mhs_solve!`, `discrepancy_solve!`), `MHSOptimize.jl` (L-BFGS, safeguard, stopping tests), `MHSGaussNewton.jl` (exact GN product), `MHSAlternate.jl` (block alternation).

**Repo state:** last commit `e22b195`. `MHSAlternate.jl`, the `verbose` option and `HSEMHS/scripts/m5_bifrost/` are in the working tree but **not committed**.

## 1. The problem in one paragraph

We solve for a magnetohydrostatic atmosphere with a fixed temperature cube T and observed field Bobs ± σ. The unknowns are u = ln P at cell centres and a vector potential Ã on edges, so B = curl A on faces and ∇·B = 0 exactly. The objective is

Φ = λ_F Φ_F + Φ_B + Φ_S + λ_F λ_Q Φ_Q

- Φ_F: force balance on faces, ∇P + ρg − J×B, normalized by S = ρ₀g.
- Φ_B: data term on faces.
- Φ_S: a small grid-scale regularizer on δB.
- Φ_Q: penalizes Nyquist currents that the face averages of J×B cannot see.

λ_F is chosen by the discrepancy principle. **The synthetic gates C1 and C3 pass.** C2 has one known open row: interior order 1.69 vs the required 1.9, possibly pre-asymptotic. On a real snapshot (Bifrost, coarsened to 42×42×89, 571 km horizontal spacing, 2.98 to −0.49 Mm), the solve does not converge** by the user's criterion: P and B must change by ≤ 1e-3 (relative) per 100 iterations, or per outer cycle.

## 2. What the evidence shows

| Observation | Numbers |
|---|---|
| Start state | u from column HSE (bottom anchor 1.035e5 Pa), Ã fit to Bobs. Φ_F ≈ 6.1e5 at λ_F = 1, dominated by the Bifrost Lorentz force in the low-β upper layers: \|R\|/S up to ~1e4, with β ~ 1e-4 at the top. |
| Identifiability | δP_unobs = \|B\|σ_B/μ₀ exceeds P in the top ~48 of 89 layers (98–100 % of nodes). |
| Joint L-BFGS (preconditioned by the GN diagonal; per-layer Hutchinson diagonal for Ã) | B never moves: χ² stays at 0.317, and Φ_F stalls at 1.8e5 by iteration 8000. The search direction has \|Δu\| ~ 500 per unit α against \|Δb̃\| ~ 0.1, so the \|Δu\| ≤ 0.5 cap fixes α ~ 1e-3. |
| Exact per-node Ã diagonal (colored probing), per-layer u floor | Little effect: Φ_F after 5000 iterations is 2.7e5 – 4.9e5. |
| B-first (Ã with v frozen, then joint) | B moves. At λ_F = 1, Φ_F drops 6.1e5 → 1.07e3 with χ² = 0.395. The joint stage afterwards still ends at `max_iter` or `stalled`. |
| Change per 100 iterations in the joint stage (λ_F = 1e-4) | max \|ΔP/P\| = 9 – 600 (layers ≈1.4–2.3 Mm); 0.1–0.6 on identifiable nodes; max \|ΔB\|/max(\|B\|, σ) 0.04–0.27. **Pressure lurches by large factors from one window to the next**; some nodes sit at the collapse bound u ≥ u₀ − 3. |
| Discrepancy continuation | χ² ≤ thresh at every λ_F up to 1e4, so the selection is always the upper bound. Φ_Q/Φ at the selection: 0.93 (λ_Q = 1), 0.065–0.12 (λ_Q = 1e-3). |
| λ_Q vs the synthetic gate | C3(b) (the near-null B checkerboard) passes down to λ_Q = 1e-4. |

## 3. What has been done or ruled out

- **Preconditioner scaling** is not the bottleneck (above).
- **Low-β trigger for B-first misfires** on uniform fields (β ≪ 1, L = 0). It now triggers on median \|L\|/S > 1.
- **A pressure-collapse safeguard** (bound u ≥ u_start − 3, gauge-aware, projected) prevents vacuum but does not produce convergence.
- **Gauss–Newton-CG on the joint system** (M6c) was slower than L-BFGS on the synthetic soft mode; it has not been retried on real data.
- **Block alternation (current attempt):**
  - Ã block: L-BFGS with v frozen.
  - u block: Gauss–Newton with PCG (exact GN product, exact u diagonal), per-node trust region, collapse bound.
  - Repeat until P and B change ≤ 1e-3 per cycle.
  - Synthetic: C1a converges in 2 cycles (\|u − u_s\| = 2e-11), C1b in 3 cycles (1.8e-4).
  - Bifrost: see §5.

## 4. Questions for Codex

1. **Formulation.** With T fixed and B held within σ, is a static solution expected to exist in the upper, Lorentz-dominated and pressure-unidentifiable layers? If not, should those layers be handled differently? For example:
   - drop or downweight the force equations there,
   - add a pressure prior,
   - or treat B there as force-free and decouple P.
2. **u sub-problem.** In low-β layers ∂R/∂u ∝ P is tiny, so u is ill-determined and the iterates wander. Is a block Gauss–Newton on u the right tool, or is there a better-posed reparameterization? Candidates: solve for P instead of ln P where β ≪ 1, or for the total pressure P + B²/2μ₀.
3. **Discrepancy principle.** χ² never exceeds the threshold, even at λ_F = 1e4. Is the selection meaningful when the force term cannot be driven to zero? Should the upper bound be raised, or the λ_F choice tied to the force residual?
4. **Φ_Q on real data.** It dominates Φ unless λ_Q ≲ 1e-4. Is it the right remedy for the averaging near-null mode at all, or would a different discretization of the face averages (D9b §1.1) avoid needing it?
5. **Convergence criterion.** Is "P and B change ≤ 1e-3 per cycle" a sound acceptance test? Or should it be restricted to identifiable nodes, given that unidentifiable pressure is free by construction?

## 5. Result of the block alternation on Bifrost

40 cycles. Each cycle is 300 L-BFGS iterations on Ã, then up to 5 Gauss–Newton steps on u (PCG ≤ 100). λ_Q = 1e-3; about 25 min per run.

| | λ_F = 1e-4 | λ_F = 1 |
|---|---|---|
| Φ_F start → cycle 40 | 60.9 → 0.043 (÷1400) | 6.09e5 → 466 (÷1300) |
| χ² | 0.318 → 0.3185 | 0.317 → 0.357 |
| Φ_Q/Φ | 2.7e-3 | 2.3e-2 |
| max \|ΔP/P\| or \|ΔB\|/max(\|B\|, σ) per cycle | 2.1 (cycle 2) → 0.74 (cycle 6), then **flat at 0.2–0.7** | 10.9 → 5.0, then **flat at 3–9** |
| status | not converged (`max_outer`) | not converged (`max_outer`) |
| RMS \|R\|/S per layer, top → bottom (every 8th) | 13, 82, 54, 25, 14, 5.1, 1.2, 0.61, 0.34, 0.27, 0.24 | 13, 86, 56, 24, 6.9, 3.9, 1.0, 0.37, 0.25, 0.24, 0.25 |
| max \|δB\|/σ (data_admissible needs ≤ 5) | 11.1 | 13.3 |

Φ decreases monotonically every cycle and every u step is accepted, so the iteration is a descent, not an oscillation. But the per-cycle change in the measured quantities stalls well above 1e-3. **It is not yet known whether that change is confined to the unidentifiable upper layers** (it was there in the earlier joint L-BFGS runs). If it is, question 5 decides the matter.

## 6. How to reproduce

The data are not in git:
- input: `IO/input-mhs-hse.nc` (Bifrost en024048_hion snapshot 385, 504×504×177; T and B only)
- EOS table: `HSEMHS/cache/eos_table.h5`

`HSEMHS/scripts/m5_bifrost/bif_setup.jl` builds the coarsened problem (12×12 horizontal block average, every 2nd level; periodic 42×42×89). The absolute paths to the input and table are set at the top of the file.

Run from `HSEMHS/`, each with `julia --project=. -t 8 scripts/m5_bifrost/<script> <args>`:

| Script | Arguments | What it runs |
|---|---|---|
| `bif_alt.jl` | `1e-4 1e-3 40 300` (λ_F, λ_Q, max cycles, Ã iterations per cycle) | block alternation, with per-cycle lines (§5) |
| `bif_disc4.jl` | `1e-3` (λ_Q) | discrepancy continuation with the 1e-3 stopping test (§2) |
| `bif_where.jl` | — | where P and B change, per 100 iterations (§2) |

## 7. What we would like back

- **Per question:** a recommendation with a short justification. Say where you think the formulation, not the solver, is the issue.
- **Disagreements:** any you have with decisions in the spec §10 log or the reports, with the reason.
- **One concrete next experiment:** if you propose a change, give the experiment that would confirm or refute it, with its pass criterion.

Claude will check the suggestions against the code and the numbers before anything is changed, and the user decides.
