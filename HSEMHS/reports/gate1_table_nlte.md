# Gate 1: EOS table report

- Table: `cache/eos_table_nlte.h5`, 4092 × 3714 nodes, sha256 `462cf9a33e1900c1adb56be439fe3373ecd5d63c5a97523398baad715d9e77f5`
- EOS model: nlte (hydrogen ionization from `/mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS/data/nlte/nlte_hion_correction.h5`, en024048_hion snapshots [500, 600, 700] stride 2; ch012012_hion snapshots [900, 930, 960] stride 4; 5565370 NLTE nodes)
- Envelope: T = 2172 – 1.781e+06 K, log₁₀P = -13.00 – 15.00 Pa (eos.logP_range override)
- Direct solver: prec = 1.0e-10, maxit = 5000, FIX_GASC_H2 = true
- Coverage scan (400 × 200): 14824 direct calls did not converge; fully-ionized test: false
- Hot branch: `robust_witt`; T_sw = 1.502e+04 K; max branch difference in [T_sw, 1.1 T_sw] = 2.81e-10
- Below T_sw, 312 scan points needed the robust fallback; 44307 table nodes are flagged EOS_ROBUST_FALLBACK
- Breakpoints inserted: 244 temperatures (partition-function grid and opacity switches)

## Refinement

- level 0: 1831 × 1402 nodes, midpoint max |Δ| = 4.75e-03 / 3.56e-02 / 2.04e-02 / 1.09e-01 / 3.05e+00 (lnrho / lnne / lnchi / lnnhi / lnnp); marked 762 X and 836 Y intervals
- level 1: 2593 × 2238 nodes, midpoint max |Δ| = 5.95e-03 / 2.51e-02 / 2.49e-02 / 5.05e-02 / 1.52e+00 (lnrho / lnne / lnchi / lnnhi / lnnp); marked 1499 X and 1476 Y intervals
- level 2: 4092 × 3714 nodes, midpoint max |Δ| = 3.31e-03 / 1.93e-02 / 1.43e-02 / 3.25e-02 / 8.71e-01 (lnrho / lnne / lnchi / lnnhi / lnnp); marked 2927 X and 2777 Y intervals

## Gate 1 table rows

| Check | Value | Threshold | Pass |
|---|---|---|---|
| max \|Δ ln ρ\| (1000000 withheld points) | 1.99e-03 | 1e-5 | false |
| max \|Δ ln nₑ\| | 1.74e-02 | 1e-4 | false |
| max \|Δ ln χ₅₀₀\| | 1.42e-02 | 1e-4 | false |
| p99.9 \|Δ\| (ρ / nₑ / χ) | 3.8e-07 / 2.4e-05 / 2.0e-05 | — | — |
| ∂ln ρ/∂ln P vs FD sweep (10000 points) | max 1.35e-03 | 1e-3 abs | false |
| table nodes from failed model calls | 0 | 0 | true |
| model failures at withheld points | 0 | 0 | true |

Worst points (T [K], P [Pa]): lnchi: (3846.0, 4.396e-7); lnne: (2982.0, 8.392e-10); lnnhi: (6265.0, 0.0001387); lnnp: (2585.0, 1.207e-11); lnrho: (5387.0, 0.0001912)
