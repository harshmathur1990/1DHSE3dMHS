# gasc H₂ correction: derivation and effect on Bifrost 385 (16 Oct 2026)

## Derivation

Wittmann's EOS works with fractions of p_H′ (the pressure all hydrogen nuclei would exert as atoms): f₁ = H, f₂ = H⁺, f₃ = H⁻, f₄ = H₂⁺, f₅ = H₂. Its equations are:

| Relation | Form in the code |
|---|---|
| Saha H⁺ | f₂ = g₂ f₁, g₂ ∝ 1/pe |
| H⁻ | f₃ = g₃ f₁, g₃ ∝ pe |
| H₂ dissociation, p(H)²/p(H₂) = K(T) | f₅ = f₁² p_H′ / K |
| H₂⁺ | f₄ = e f₅, e = g₂ g₄/g₅ (pe cancels) |
| nuclei | f₁+f₂+f₃+2f₄+2f₅ = 1 → a f₁ + b f₅ = 1, a = 1+g₂+g₃, b = 2(1+e) |
| charge | pe / p_H′ = fe = f₂ − f₃ + f₄ + g₁ (g₁: metal donors) |

Combining the H₂ and charge equations gives f₅·fe = f₁²·pe/K ≡ c f₁², with **c = pe/K**. Substituting f₅ = (1 − a f₁)/b reproduces the code's quadratic exactly: c₁ = c b² + a d b − e a², c₂ = 2ae − db + a b g₁, c₃ = −(e + b g₁). The f₅ refinement loop, f₅ = p_H′·(g₅/pe)·f₁² = g₅ f₁²/fe, also requires g₅ = pe/K. `molecb` returns log₁₀(1/K) in CGS, so:

- `pe_pg` (P → pe): `g5 = pe·10^cmol1`, so c = pe/K. **Consistent.**
- `gasc` (pe → P, ρ, partials): `g5 = 10^cmol1`, so c = 1/K. **Missing the factor pe** (it implicitly sets pe = 1 dyn cm⁻²). The same applies to `g4`.

Check: with the correction, `gasc` at the converged pe reproduces the requested P to ≤ 1e-10 at every Bifrost cell (max 9.6e-11). Without it, the mismatch is up to 44 %.

`gasc` feeds `rho_from_pe` (ρ through fe), `background_partials` (the H, H⁺, H⁻ densities used by the opacity) and `pg_from_pe`. nₑ does not depend on it, because pe comes from `pe_pg`.

## Effect on Bifrost 385

2.8 M cells (4×4 subsample, −0.5 to 3 Mm), at the cube's own T and P; log₁₀(original / corrected):

| Quantity | Effect |
|---|---|
| nₑ | identical (0 dex) |
| ρ | max 0.0011 dex (0.26 %), at T ≈ 3700 K, P = 1e-3 Pa; no cell differs by more than 1e-3 dex |
| χ₅₀₀, T < 2500 K | median −1.22 dex, max 1.36 dex (original too low by ×17–23) |
| χ₅₀₀, 2500–3000 K | median −0.72 dex |
| χ₅₀₀, 3000–3500 K | median −0.11 dex |
| χ₅₀₀, 3500–4000 K | median −0.008 dex |
| χ₅₀₀, T > 4000 K | ≤ 0.006 dex |
| χ₅₀₀, whole cube | 5.2 % of cells differ by more than 0.01 dex, 2.3 % by more than 0.1 dex |

Mechanism: at low pe, the original `gasc` puts far too much hydrogen into H₂, so it returns too few H atoms and H⁻ ions, and H⁻ (the dominant 500 nm opacity at these temperatures) is too weak.

Outside the cube's (T, P) range, ρ is also wrong at high pressure:

| T, P | ρ original / corrected | μ̄ original | μ̄ corrected |
|---|---|---|---|
| 3000 K, 1e6 Pa | 1.013 | 2.345 | 2.315 |
| 3000 K, 1e8 Pa | 1.94 | 4.58 | 2.36 |
| 5000 K, 1e8 Pa | 185 | 402 | 2.17 |

Pressures like these are reached only by top-anchored column HSE.

**Correction to an earlier statement:** I had described the bug as "up to ~40 % density error below 3500 K". That 40 % was the pressure-closure mismatch, not the density. On the Bifrost cube the bug's practical effect is on χ₅₀₀ in the cool chromospheric pockets, not on ρ or nₑ.

Reproduce: build the variant with `HSEMHS_FIX_GASC_H2=0 HSEMHS_WITT_BUILD_DIR=<dir> julia deps/witt/build.jl`, then run the comparison with `HSEMHS_LIBWITT=<dir>/libwitt.dylib`.
