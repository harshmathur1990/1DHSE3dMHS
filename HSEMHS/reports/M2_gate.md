# M2 gate report: column_hse + τ₅₀₀ (16 Oct 2026)

| Gate 2 row (spec §5) | Result | Threshold | Pass |
|---|---|---|---|
| Isothermal ideal gas, 251 jittered levels, n_sub = 64: bottom / top anchor | 2.5e-13 / 7.9e-13 rel | 1e-12 | yes |
| Height anchor (z_ref = 0.3 Mm) | ≤ 1e-12 rel | 1e-12 | yes |
| ln T-linear ideal gas vs analytic per-interval integral (E13), n_sub = 64 | ≤ 1e-8 in ln P | 1e-8 | yes |
| RK4 order under substep halving (n_sub 1→8) | 4.00, 4.00, 4.00 | ≥ 3.9 | yes |
| Top-first indexing: P and τ strictly increase with k; `Pgas[:,:,anchor_k] === P_ref` | holds (ideal gas and Wittmann table) | — | yes |
| Mass identity \|P − P_top − g m\|/P, Bifrost columns, Wittmann table, bottom / top | 6.1e-7 / 3.8e-9 | 1e-6 | yes |
| τ convergence on exponential χ (E1 closure exact) | order ≥ 1.95 | ≥ 1.95 | yes |
| E1 fallback (χ increasing upward → half-cell, flagged); `:zero`; μ = 0.5 | as specified | — | yes |
| Bitwise identity: 1 rank × 1 thread vs 1×8, 4×2, 6×1 | identical Pgas, log τ, m | bitwise | yes |
| Bifrost 385, both anchors | `reports/gate2_anchor.md` | report | done |
| Matched-EOS LTE comparison with RH on 3 columns | 6.8e-4 (RH given our n_H, profile refined ×64; converges ~2nd order from 1.3e-1 on the native grid) | 1e-3 | yes |

Bifrost 385 production runs (504×504×177, 12 threads, 1 rank): integration 11 s, total 110 s, mostly HDF5 compression.
- Bottom anchor (`out/bifrost385_colhse_bottom.h5`): 0 EOS failures; 22 127 columns needed substep doubling (all met 1e-6); 194 522 columns use the exp-tail τ closure and 59 494 fall back to half-cell; median z(τ₅₀₀ = 1) = 63 km (Bifrost header: 61 km).
- Top anchor (`out/bifrost385_colhse_top.h5`): `solver_status = eos_failure` because 12 759 columns (5 %) exceed the table envelope (P > 3×10¹⁴ Pa at depth). This is reported, not extended silently (doc §7.3).

Erratum E13 (spec Gate 2): the analytic `∫dz/T = Δz·ln(T₂/T₁)/(T₂−T₁)` holds for T linear in z, but the solver interpolates ln T linearly (spec §5.2). The test uses the matching formula `Δz (1/T₁ − 1/T₂)/ln(T₂/T₁)`.
