# M3 gate report: force-budget screening and column-HSE residual (16 Oct 2026)

| Check | Result | Pass |
|---|---|---|
| Hand case (a): uniform P, B = 0 → M = 0, \|δL\| ≥ M | M ≤ 1e-12 | yes |
| Hand case (b): P = P₀(1 + ε cos kx) → M = P₀εk\|sin kx\| | order ≥ 1.9 | yes |
| Hand case (c): B = (0, b cos kx, 0) → \|Lₕ\| = M = (b²k/2μ₀)\|sin 2kx\|; δL formula (E10) | order ≥ 1.9; δL to 1e-12 rel | yes |
| Bifrost 385 run (2 ranks × 6 threads, 22 s) | `reports/screening_bifrost385.md` / `.h5` | done |

Findings (screening, not a feasibility certificate; P from bottom-anchored `column_hse` with the Wittmann table; synthetic σ):

| Height | Fraction of cells with \|δL\| ≥ mismatch \|∇ₕP − Lₕ(B_obs)\| | Interpretation |
|---|---|---|
| z ≥ 1.5 Mm | ≥ 0.998 | A field correction within σ_B can balance the horizontal mismatch, scale-wise. |
| z = 1.0 Mm (β ≈ 1) | 0.85 | Mostly balanceable. |
| z = 0.0–0.5 Mm | 0.001–0.03 | Median mismatch is 15–860 × \|δL\|. Fixed-T MHS cannot balance the photosphere with B changes inside σ_B, so `approximate_only` is expected there. |
| −0.49 Mm (anchor plane) | — | ∇ₕP = 0 by construction (uniform scalar anchor). |

These numbers match the Gate 0 proxy screening (for example 0.85 vs 0.87 at 1 Mm, 0.031 vs 0.033 at 0.5 Mm).

**Column-HSE residual.** The discrete vertical residual of the RK4 solution, measured with the collocated operator, is |R_z|/ρg ≈ 4–7e-3 (median) in the photosphere and chromosphere. This is the truncation level of the 3-point stencil at Δz/H ≈ 1/8, and it sits just under the real-data `tol_rms = 1e-2` that MHS will use. The horizontal residual |R_h|/ρg is 0.12 (median) at 0–0.5 Mm and rises to O(10) at the top, where gas forces are tiny compared with Lorentz-force scales.
