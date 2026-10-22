# Gate 1: EOS table report

- Table: `cache/eos_table_fullz.h5`, 1831 × 1402 nodes, sha256 `4e19ad6aa75cc252921fa00036da062c5798ec06e60089ff1add160668d3ab8e`
- Envelope: T = 2172 – 1.781e+06 K, log₁₀P = -13.00 – 15.00 Pa (eos.logP_range override)
- Direct solver: prec = 1.0e-10, maxit = 5000, FIX_GASC_H2 = true
- Coverage scan (400 × 200): 14824 direct calls did not converge; fully-ionized test: false
- Hot branch: `robust_witt`; T_sw = 1.502e+04 K; max branch difference in [T_sw, 1.1 T_sw] = 2.81e-10
- Below T_sw, 312 scan points needed the robust fallback; 11368 table nodes are flagged EOS_ROBUST_FALLBACK
- Breakpoints inserted: 244 temperatures (partition-function grid and opacity switches)

## Refinement

- level 0: 1831 × 1402 nodes, midpoint max |Δ| = 7.12e-07 / 1.99e-06 / 4.32e-05 (ln ρ / ln nₑ / ln χ); marked 0 X and 0 Y intervals

## Gate 1 table rows

| Check | Value | Threshold | Pass |
|---|---|---|---|
| max \|Δ ln ρ\| (1000000 withheld points) | 7.14e-07 | 1e-5 | true |
| max \|Δ ln nₑ\| | 2.02e-06 | 1e-4 | true |
| max \|Δ ln χ₅₀₀\| | 3.51e-05 | 1e-4 | true |
| p99.9 \|Δ\| (ρ / nₑ / χ) | 1.5e-07 / 4.2e-07 / 2.8e-07 | — | — |
| ∂ln ρ/∂ln P vs FD sweep (10000 points) | max 1.94e-06 | 1e-3 abs | true |
| table nodes from failed model calls | 0 | 0 | true |
| model failures at withheld points | 0 | 0 | true |

Worst points (T [K], P [Pa]): lnchi: (31530.0, 0.001167); lnne: (2509.0, 1.246e-13); lnrho: (3307.0, 1.08e-13)
