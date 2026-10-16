# 500 nm opacity: HSEMHS (Wittmann/ATLAS) vs RH at 7.7–11.5 kK (16 Oct 2026)

**Question.** At the 7.7–11.5 kK points of the Gate 1 RH check, HSEMHS's χ₅₀₀ is 4.8–6.8 % above RH's. Why, and should we align with RH?

**Method.** The test points are Bifrost column (37, 181), z = −190 … +75 km, with the same T, nₑ and abundances in both codes and RH's `H_20` hydrogen atom. Each contribution was recomputed separately: RH's formulas evaluated in Julia with HSEMHS populations, and RH's own LTE populations read from `atmos.out`. Scripts: `scripts/rh_crosscheck.jl` (table below) plus the diagnostic snippets quoted in this file.

## Findings

### 1. Hydrogen bound-free: RH drops levels n ≥ 5 at 500 nm (RH artifact)

- **Cross sections agree.** Fed the same populations, RH's formula (hydrogenic σ₀·n_eff·g_bf·(λ/λ_edge)³ with Seaton's Gaunt factor) and Wittmann's `HOP` agree level by level to ≤ 1 %, and to 0.4 % in total, including the n ≥ 9 tail (explicit levels in RH, merged integral in Wittmann).
- **Populations agree.** RH's LTE neutral-hydrogen density matches HSEMHS to 0.5 %.
- **The cause.** `Hydrogen_bf` only uses a continuum whose model-atom wavelength grid reaches λ. In `H_20.atom` the grid minimum is 569.9 nm for n = 5, 820.6 nm for n = 6, and longer for higher n (the same holds for `H_6.atom`). At 500 nm, RH therefore counts only n = 3 and n = 4, although photons at 500 nm ionize every level n ≥ 3. Our n = 3 + n = 4 sum (4.74e-3 m⁻¹) reproduces RH's `HYDROGEN_BF` (4.71e-3 m⁻¹), and the dropped n ≥ 5 levels are the 19 %.

### 2. H⁻ density: RH uses all atomic hydrogen, including protons (RH artifact)

- **Cross sections agree.** H⁻ bound-free: Wittmann's fit gives 2.832e-21 m² at 500 nm, against 2.84e-21 m² from Geltman in RH (0.3 %).
- **The cause.** In `chemequil.c`, the iteration uses `fHmin = nₑ·fn0[0]·Φ` (neutral fraction), but the stored density is `nHmin = nₑ·n[0]·Φ`, where `n[0]` is all atomic H (neutral plus ionized). H⁻ is therefore too high by (n_HI + n_p)/n_HI: 1.13 at 11.5 kK, 1.09 at 10.6 kK, 1.004 at 7.7 kK and 1.000 in the photosphere. With this factor, RH's `HMINUS_BF` is reproduced exactly (2.331e-3 m⁻¹).
- **Free-free.** H⁻ ff agrees within 0.3–3 % up to 10 kK. Above 10 kK, RH's Stilley–Callaway table is held at θ = 0.5 (T = 10 080 K), which adds +7 % to H⁻ ff at 11.5 kK (0.5 % of the total).

### 3. H₂⁺: different physics data (genuine model choice)

Wittmann/ATLAS `H2PLOP` (Kurucz's fit for H + H⁺, including photodissociation) gives 3–6 × RH's `H2PLUS_FF` (Bates 1952 free-free table, tabulated only up to 12 kK). That is +4.3 % of the total χ at 7.7 kK, +2.5 % at 5.9 kK, +1.3 % at 11.5 kK, and < 0.3 % at T ≤ 5100 K. **This accounts for all of the remaining difference.**

## Result after removing RH's two artifacts

Opacity relative to HSEMHS, at the 10 points (z = 404 … −190 km):

| T [K] | 4200 | 4505 | 4782 | 5115 | 5930 | 7707 | 10613 | 10897 | 11245 | 11489 |
|---|---|---|---|---|---|---|---|---|---|---|
| RH as is | +1.0 % | +0.7 % | +0.5 % | +0.1 % | −2.3 % | −4.8 % | −6.6 % | −6.7 % | −6.7 % | −6.8 % |
| RH, artifacts removed | +1.0 % | +0.7 % | +0.5 % | +0.1 % | −2.3 % | −4.3 % | −1.9 % | −1.6 % | −1.4 % | −1.2 % |
| H₂⁺ data difference alone | | | | −0.2 % | −2.5 % | −4.3 % | | | | −1.3 % |

The residual equals the H₂⁺ difference. With the artifacts removed, all 10 points are within 4.3 %, inside the 5 % threshold.

## Proposal

**Do not align HSEMHS with RH.** The two large differences are RH artifacts, and both make RH's 500 nm continuum physically too low or too high:

- the hydrogen continua truncated by their model-atom wavelength grids;
- the H⁻ density computed with protons included.

Copying them would make the code less correct. The Wittmann/ATLAS opacities are also the ones the SIR/FIRTEZ inversion family uses, which matters if FIRTEZ is a consumer.

**Gate 1 RH row:** count it as passed, at ≤ 4.3 % against RH with its artifacts removed (≤ 2.3 % at photospheric T against RH as is). Record the H₂⁺ data difference as known.

**Optional, user's choice:** if the products are meant for RH 1.5D synthesis and exact consistency with RH's background continuum matters, add an opacity option `h2plus = "bates1952"` (RH's table). It would change χ₅₀₀ by 0.2–4 % at 5–8 kK. I would not make it the default.

**For the RH maintainers (outside this project):** artifacts 1 and 2 look like genuine RH issues and may be worth reporting upstream:
- `Hydrogen_bf` could evaluate hydrogenic continua at any λ ≤ λ_edge regardless of the model-atom grid's λmin;
- `chemequil.c` line 342 should multiply by `fn0[0]`.
