# RH cross-check (Gate 1 opacity row, Gate 2 RH row)

RH source commit `c95bdf66ff6bc1c6cea0ed04dc688100170e8ee6`, unmodified; inputs in `/Users/harshm/Documents/WorkRepo/MHS/HSEMHS/cache/rh_runs/H_20`.
RH hydrogen model atom: `H_20.atom` (LTE populations).
Matched: T, nₑ (SOLVE_NE = NONE), abundances (Wittmann vector), m_top (= P_top/g of the column_hse solution), g = 274.0 m s⁻², molecules H₂ and H₂⁺ only, no line opacity (KURUCZ_DATA = none, ALLOW_PASSIVE_BB = FALSE).

## Column 1 (typical): global (i, j) = (37, 181); T range 3082 – 12465 K

- ρ: RH algebraic n_H (metals neutral, no H₂) vs HSEMHS Wittmann: max |Δρ/ρ| = 1.14e-02 (median 3.73e-06)
- Height scale (RH trapezoid of dm/ρ vs HSEMHS RK4): bottom-node offset 4.864 km, max |Δz| 5.184 km
- ρ: largest difference at z = 463 km, T = 3494 K, P = 488 Pa
- P at the same height, RH on the native grid: max |ΔP/P| = 6.53e-02 (median 1.61e-02)
- **Matched EOS (RH given our n_H), profile refined ×1 / ×4 / ×16 / ×64, heights aligned at the bottom anchor, z ≤ 2981 km: max |ΔP/P| = 4.82e-02 / 3.51e-03 / 5.90e-04 / 4.07e-04** (orders 1.89, 1.29, 0.27)
- Matched EOS including the TR and corona: 4.82e-02 / 3.51e-03 / 5.90e-04 / 4.07e-04
- RH's own n_H (P = g m, no H₂), same refinement: 3.28e-02 / 1.41e-02 / 1.68e-02 / 1.70e-02 (EOS-difference floor)
- Mass per H nucleus: RH (AMU·wght_per_H) / HSEMHS − 1 = +1.50e-05
- log τ₅₀₀ (RH tau_ref vs HSEMHS, below the top node): median +0.0162, p1/p99 -0.0306 / +0.3103 dex

## Column 2 (cool_chromosphere): global (i, j) = (141, 457); T range 2426 – 222133 K

- ρ: RH algebraic n_H (metals neutral, no H₂) vs HSEMHS Wittmann: max |Δρ/ρ| = 1.47e-03 (median 3.70e-06)
- Height scale (RH trapezoid of dm/ρ vs HSEMHS RK4): bottom-node offset 13.369 km, max |Δz| 13.498 km
- ρ: largest difference at z = 248 km, T = 4641 K, P = 3.13e+03 Pa
- P at the same height, RH on the native grid: max |ΔP/P| = 1.41e-01 (median 8.74e-02)
- **Matched EOS (RH given our n_H), profile refined ×1 / ×4 / ×16 / ×64, heights aligned at the bottom anchor, z ≤ 2121 km: max |ΔP/P| = 1.32e-01 / 1.03e-02 / 1.29e-03 / 6.78e-04** (orders 1.84, 1.50, 0.46)
- Matched EOS including the TR and corona: 1.32e-01 / 1.03e-02 / 1.29e-03 / 6.78e-04
- RH's own n_H (P = g m, no H₂), same refinement: 1.26e-01 / 3.47e-03 / 5.62e-03 / 6.29e-03 (EOS-difference floor)
- Mass per H nucleus: RH (AMU·wght_per_H) / HSEMHS − 1 = +1.50e-05
- log τ₅₀₀ (RH tau_ref vs HSEMHS, below the top node): median +0.0964, p1/p99 -0.0313 / +1.0539 dex

## Column 3 (low_transition_region): global (i, j) = (161, 253); T range 4588 – 806322 K

- ρ: RH algebraic n_H (metals neutral, no H₂) vs HSEMHS Wittmann: max |Δρ/ρ| = 1.21e-03 (median 2.92e-06)
- Height scale (RH trapezoid of dm/ρ vs HSEMHS RK4): bottom-node offset 9.391 km, max |Δz| 9.477 km
- ρ: largest difference at z = 307 km, T = 4627 K, P = 2.49e+03 Pa
- P at the same height, RH on the native grid: max |ΔP/P| = 9.06e-02 (median 2.75e-02)
- **Matched EOS (RH given our n_H), profile refined ×1 / ×4 / ×16 / ×64, heights aligned at the bottom anchor, z ≤ 1301 km: max |ΔP/P| = 1.67e-02 / 1.25e-03 / 2.65e-04 / 2.03e-04** (orders 1.87, 1.12, 0.19)
- Matched EOS including the TR and corona: 1.67e-02 / 1.25e-03 / 2.65e-04 / 2.03e-04
- RH's own n_H (P = g m, no H₂), same refinement: 1.45e-02 / 1.93e-03 / 2.22e-03 / 2.25e-03 (EOS-difference floor)
- Mass per H nucleus: RH (AMU·wght_per_H) / HSEMHS − 1 = +1.50e-05
- log τ₅₀₀ (RH tau_ref vs HSEMHS, below the top node): median +0.4903, p1/p99 -0.0316 / +2.7224 dex

## Gate 1 row: 500 nm background opacity at 10 photospheric points (column 1)

RH `backgrcontr` at λ = 500.000 nm, all absorption and scattering contributions summed; HSEMHS χ₅₀₀ from the table.

| z [km] | T [K] | P [Pa] | χ HSEMHS [m⁻¹] | χ RH [m⁻¹] | RH/HSEMHS − 1 | RH, artifacts removed / HSEMHS − 1 | n_H RH/HSEMHS − 1 |
|---|---|---|---|---|---|---|---|
| 404 | 4200 | 944.6 | 1.095e-07 | 1.106e-07 | +1.02 % | +1.02 % | -1.58e-03 |
| 346 | 4505 | 1667 | 3.048e-07 | 3.07e-07 | +0.73 % | +0.73 % | -1.14e-03 |
| 268 | 4782 | 3401 | 1.052e-06 | 1.057e-06 | +0.45 % | +0.45 % | -1.13e-03 |
| 209 | 5115 | 5625 | 2.447e-06 | 2.449e-06 | +0.09 % | +0.09 % | -8.78e-04 |
| 132 | 5930 | 1.029e+04 | 9.294e-06 | 9.079e-06 | -2.32 % | -2.31 % | -3.63e-04 |
| 75 | 7707 | 1.481e+04 | 0.0001138 | 0.0001084 | -4.75 % | -4.30 % | -6.72e-05 |
| -0 | 10613 | 2.024e+04 | 0.002209 | 0.002062 | -6.64 % | -1.86 % | -2.10e-05 |
| -57 | 10897 | 2.492e+04 | 0.003548 | 0.00331 | -6.68 % | -1.61 % | -2.18e-05 |
| -133 | 11245 | 3.255e+04 | 0.006342 | 0.005915 | -6.74 % | -1.35 % | -2.31e-05 |
| -190 | 11489 | 3.948e+04 | 0.00951 | 0.008865 | -6.78 % | -1.20 % | -2.43e-05 |

Max |Δχ/χ| vs RH as is = 6.78 %; vs RH with its two artifacts removed = 4.30 % (threshold 5 %): **pass (artifact-corrected)**

RH contributions at the deepest of these points (m⁻¹): HYDROGEN_BF = 0.00471, HMINUS_BF = 0.00233, HYDROGEN_FF = 0.00102, HMINUS_FF = 0.000646, H2PLUS_FF = 0.000141, METAL_BF_SI = 4.2e-06, METAL_BF_C  = 2.63e-06, METAL_BF_MG = 2.53e-06, METAL_BF_O  = 1.62e-06, THOMSON = 1.61e-06, METAL_BF_AL = 1.27e-06, METAL_BF_S  = 1.68e-07, METAL_BF_NA = 3.95e-08, METAL_BF_HE = 3.73e-08, RAYLEIGH_H = 2.26e-08, RAYLEIGH_HE = 8.16e-11


## Gate 2 row summary

max |ΔP/P| over the 3 columns, matched EOS, RH on the ×64 refined grid = 6.78e-04 (threshold 1e-3): **pass**
