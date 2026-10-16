# Gate 2 anchor report: column_hse (Wittmann table) vs Bifrost native pressure

Statistics on every 4th column in x and y. log₁₀(P_column_hse / P_Bifrost), p1 / p50 / p99. NaN columns (left the EOS envelope) are excluded and counted.

- bottom anchor run: `out/bifrost385_colhse_bottom.h5`, 0 of 15876 sampled columns failed
- top anchor run: `out/bifrost385_colhse_top.h5`, 805 of 15876 sampled columns failed (P beyond the table envelope at depth)

| z [Mm] | bottom anchor (P_ref = 103 509 Pa at −0.494 Mm) | top anchor (0.1 Pa at +2.98 Mm) | proxy (Gate 0) bottom / top p50 |
|---|---|---|---|
| 2.98 | -3.74 / -0.63 / +1.53 | +0.32 / +1.46 / +2.23 | -1.02 / +1.47 |
| 2.00 | -3.06 / -0.35 / +1.51 | +0.43 / +1.73 / +3.52 | -0.60 / +1.81 |
| 1.50 | -2.14 / -0.19 / +1.47 | +0.43 / +2.03 / +4.50 | -0.30 / +2.17 |
| 1.00 | -0.70 / +0.10 / +1.23 | +0.07 / +2.43 / +5.37 | +0.04 / +2.55 |
| 0.50 | -0.33 / +0.05 / +0.44 | -0.21 / +2.45 / +6.97 | +0.02 / +2.50 |
| -0.00 | -0.24 / +0.02 / +0.11 | -0.33 / +2.67 / +8.44 | +0.01 / +2.50 |
| -0.49 | -0.02 / -0.00 / +0.06 | -0.35 / +2.74 / +9.21 | +0.00 / +2.51 |

## τ₅₀₀ = 1 height (Gate 0 item 3)

| run | z(τ=1) p1 / p50 / p99 [km] |
|---|---|
| bottom anchor | -29.5 / 63.2 / 127.1 |
| top anchor | 7.8 / 630.9 / 1302.6 |

Bifrost header ZTAU51 (mean τ₅₀₀ = 1 height) = 61 km.

Implied opposite-boundary pressure: bottom-anchored top-plane log₁₀P p1/p50/p99 = -6.18 / -3.1 / -0.71; top-anchored bottom-plane = 4.65 / 7.75 / 14.21 (finite columns only).
