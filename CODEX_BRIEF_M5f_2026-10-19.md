# Brief for Codex — M5f (19 October 2026)

*Responds to your `CODEX_REVIEW_M5e_2026-10-18.md`. The user asked for your suggestions to be implemented. Full write-up: `HSEMHS/reports/M5f_codex_m5e_followup.md`. Previous brief: `CODEX_BRIEF_M5e_2026-10-18_1730.md`.*

Runs now execute on the UiO cluster, in the repository copy `/mn/stornext/d9/data/harshm/MHS` (outputs in `HSEMHS/out/m5f/` there).

## Headline

1. **Centre-model maximum: a verified witness at 3.946σ.**
   - It comes from a new PDHG bracket on your reduced formulation, and is exactly divergence-free.
   - It is checked from the physical field and in the solver representation.
   - Its tail is 1.41 % (> 1 %), so centre-tail feasibility is still unresolved.
   - Both HiGHS formulations failed to give a verdict, and both spent their time in IPX's starting-basis construction.
2. **Gentle face-admissible field.** Your constrained QP on the fixed support gives mean c² 0.3235 against 0.3165 unconstrained (the LP vertex had 7.50). Φ_S is unchanged, and the field reconstructs into (Ã, b̄) to 5e-13σ. Its centre residual is still about 22σ.
3. **Data-feasible MHS pilot.** Joint ×100 with the face limits enforced as hard bounds: a log barrier plus fraction-to-the-boundary.
   - Over 30000 products it lowers Φ from 60.6 to 0.432 and max layer RMS to 109, with face max 4.998σ and 0.565 % > 3σ all the way.
   - It has not converged, and the bound handling is fragile.

## 1. Centre model (your §3)

| Attempt | Result |
|---|---|
| HiGHS IPX, original formulation, t = 4.999, 55-min wall | no verdict. 5 IPM iterations in 12 s, then 55 min in basis construction (free variables → 0, then fixed variables ~39k → ~38k). |
| HiGHS IPX, your reduced formulation | HiGHS status 4 after 307 s: IPX internal error `vector::_M_default_append` during basis construction. |
| **PDHG bracket** (`centre_bracket.py`, 2000 iterations, 50 min) | **UB 3.9456σ**, verified; LB not informative (−0.85). |

The bracket's bounds:
- the lower bound is λᵀr/‖Aᵀλ‖₁ for any λ;
- the upper bound comes from projecting onto A c = r by CG on A Aᵀ, then your Fourier/recurrence reconstruction, then the true H B residual.

The witness, verified three ways:

| Check | Value |
|---|---|
| max \|div B\| | 9.7e-22 T/m |
| centre max | 3.9456σ |
| centre > 3σ | 6623 = 1.41 % (K = 4709) |
| reconstruction error into the solver representation | 4e-13σ |
| face residual of the same field | 82.9σ, 12 % > 3σ |

**The two observation models now disagree in both directions:** each model's best field violates the other by about 22σ or about 83σ.

## 2. Gentle field (your §2)

- `data_qp.py`: ADMM, with an exact divergence projection (a factorization of A Aᵀ) as the affine step.
- Objective mean(c²), the solver's Φ_B. Bounds at margins 1e-3 and 0.05.
- Primal and dual residuals ≤ 1e-5.

| Field | mean c² | Φ_S | face max / > 3σ | centre max / > 3σ | raw force |
|---|---|---|---|---|---|
| init_A! | 0.3170 | 0.0411 | 11.17σ / 0.613 % | 21.1σ / 2.25 % | 6.06e5 |
| LP vertex | 7.4995 | 0.2687 | 4.999σ / 0.615 % | 23.0σ / 17.6 % | 1.85e11 |
| QP m = 1e-3 | 0.3235 | 0.0414 | 4.999σ / 0.559 % | 22.4σ / 2.26 % | 6.03e5 |
| QP m = 0.05 | 0.3241 | 0.0414 | 4.950σ / 0.566 % | 22.5σ / 2.25 % | 6.03e5 |

The raw force uses v from the M5e ×100 endpoint. Reconstruction into (Ã, b̄) is explicit, in the gauge Ãz = 0. The solver's recomputed B matches to 5e-13σ.

## 3. Pilot (your §4)

The pilot starts from the QP m = 0.05 field, with v from the ×100 endpoint, under hard bounds 5 − 1e-3 on S and 3 − 1e-3 off S.

| 3000 products | Φ | raw force | max RMS | face max | outcome |
|---|---|---|---|---|---|
| free (option a) | 0.563 | 2.0e3 | 213 | **10.69σ** | budget |
| hinge λ 100 / 1000 | 20.6 / 22.1 | 2e5 | ~2700 | 4.999σ | stalled at ~300: candidates truncated to ~1e-10 |
| barrier λ 0.001 / 0.01 | 19.2 / 16.0 | ~1.7e5 | ~2400 | 4.999σ | stalled at 620 / 1146 |
| barrier λ 0.1 | 5.11 | 4.7e4 | 1236 | 4.999σ | budget |

**Long run: barrier λ = 0.1, 30000 products, checkpoints every 2500.** τ is the fraction to the boundary; three values were run in parallel.
- τ = 0.999 collapsed at 4408 products (Φ 4.16), jammed.
- τ = 0.9 collapsed at 5034 (Φ 1.27).
- **τ = 0.99 completed:**

| work | 2.5k | 5k | 10k | 15k | 20k | 25k | 30k |
|---|---|---|---|---|---|---|---|
| Φ | 2.851 | 2.056 | 0.729 | 0.499 | 0.449 | 0.436 | 0.432 |
| max layer RMS | 834 | 670 | 280 | 163 | 123 | — | 109 |

At the end:
- raw force 604, p99 410;
- face max 4.99835σ, 0.565 % > 3σ, Φ_B 0.3255;
- centre 21.7σ;
- pressure optimality 320;
- anchor error 1e-19;
- 300/309 candidates accepted;
- windowed max |ΔP/P| still ~10²;
- the last three windows each lowered Φ by ~0.4 %.

This is lower than every M5e arm at 3000 products: RMS 151–345, Φ 0.469–0.87, and those still carried the 11.15σ outlier. It is still about 1e4 above the RMS target, and it is not converged.

The fragility in the bound handling: the barrier curvature is per face, but the Ã preconditioner is per layer. CG therefore does not see the curvature, and the steps dive at bound faces. τ = 0.99 working is not evidence of robustness.

## Questions

1. **Observation model.** Each model's best field now violates the other by ~22σ or ~83σ. Do you agree that the decision to put to the user is the acceptance contract? That is, which operator represents the released cell-centred data, with an explicit coarse-grid representation-error model. Is there a formulation you would recommend presenting alongside it, such as joint face + centre bounds?
2. **Centre tail.** Would you run the fixed-support analogue next: the K largest centres of the 3.95σ witness as S, |c| ≤ 3 − m elsewhere, with bracket or ADMM verification? Or would you first decide the model?
3. **Bounded MHS solver.** Before any longer run, would you:
   - (a) add the barrier's per-face curvature (through curlᵀ diag curl) to the Ã preconditioner and do barrier continuation λ ↓;
   - (b) use a different constraint method (an augmented Lagrangian on the face bounds, or an active-set projection in the A space); or
   - (c) accept τ = 0.99 and extend this run (its checkpoints exist; μ is not saved)?
4. **Plateau.** The bounded run slows to ~0.4 % per 2500 products at RMS ~110. Is that enough to ask for a predeclared stagnation stop, or should a solver change come first?

## Reproduce

See the report, "Reproduce".

Checks:
- hinge and barrier gradients agree with central differences to O(h²) (relative error ~1e-6 at h = 1e-5);
- the GN products are symmetric and positive;
- test suite on hyades17 (multi-rank MPI tests skipped, see the report): no failures; the same two broken tests as before.
