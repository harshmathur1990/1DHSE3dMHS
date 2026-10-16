# Gate 1: EOS table report

- Table: `cache/eos_table.h5`, 1733 × 1254 nodes, sha256 `0623d31e4f47bd808e613455e588864ef3571e7f4cf5edf4838db149eabb3a89`
- Envelope: T = 2172 – 1.613e+06 K, log₁₀P = -10.50 – 14.51 Pa (Gate 0 proxy anchor runs ±1 dex (reports/gate0_bifrost385.h5))
- Direct solver: prec = 1.0e-10, maxit = 5000, FIX_GASC_H2 = true
- Coverage scan (400 × 200): 10372 direct calls did not converge; fully-ionized test: false
- Hot branch: `robust_witt`; T_sw = 1.509e+04 K; max branch difference in [T_sw, 1.1 T_sw] = 2.82e-10
- Below T_sw, 28 scan points needed the robust fallback; 714 table nodes are flagged EOS_ROBUST_FALLBACK
- Breakpoints inserted: 244 temperatures (partition-function grid and opacity switches)

## Refinement

- level 0: 1732 × 1252 nodes, midpoint max |Δ| = 5.19e-07 / 1.43e-06 / 5.48e-05 (ln ρ / ln nₑ / ln χ); marked 1 X and 2 Y intervals
- level 1: 1733 × 1254 nodes, midpoint max |Δ| = 5.19e-07 / 1.43e-06 / 3.35e-05 (ln ρ / ln nₑ / ln χ); marked 0 X and 0 Y intervals

## Gate 1 table rows

| Check | Value | Threshold | Pass |
|---|---|---|---|
| max \|Δ ln ρ\| (1000000 withheld points) | 5.20e-07 | 1e-5 | true |
| max \|Δ ln nₑ\| | 1.44e-06 | 1e-4 | true |
| max \|Δ ln χ₅₀₀\| | 3.17e-05 | 1e-4 | true |
| p99.9 \|Δ\| (ρ / nₑ / χ) | 1.1e-07 / 2.7e-07 / 2.3e-07 | — | — |
| ∂ln ρ/∂ln P vs FD sweep (10000 points) | max 6.23e-07 | 1e-3 abs | true |
| table nodes from failed model calls | 0 | 0 | true |
| model failures at withheld points | 0 | 0 | true |

Worst points (T [K], P [Pa]): lnchi: (23270.0, 0.000641); lnne: (2871.0, 2.35e-10); lnrho: (3797.0, 6.661e-11)
