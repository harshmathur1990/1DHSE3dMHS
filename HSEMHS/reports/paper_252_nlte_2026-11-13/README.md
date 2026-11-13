# The method paper's tests with the experimental NLTE EOS (13 Nov 2026)

The cases of `../paper_252_2026-11-10` (Table 1 of the method paper) rerun with `eos.model = "nlte"` (the experimental NLTE
equation of state, root `README.md`, off by default), against the paper's LTE results. Branch `nlte-eos`; NLTE table
`cache/eos_table_nlte.h5`, sha256 `462cf9a3…` (`../gate1_table_nlte.md`). Outputs in
`/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13/`, inputs in `/mn/stornext/d9/data/harshm/MHS/IO/tau252_nlte/`.

## Inputs

`IO/tau252_nlte` is `IO/tau252` rebuilt by `make_tau_input.jl` (bit-identical to the paper's inputs and truths) with the
snapshot's nₑ added to each truth (`truth/lnne`), plus one new input: `input_eosnlte_mu1.0`, the μ = 1 τ integrated with the
**NLTE** table's χ500 (`build_logs/build_nlte.sh`). The paper's own-τ inputs (`input_eos_*`) are integrated with the LTE table's
χ500 and the M3D input with MULTI3D's `H/tau500`; both are reused unchanged.

## Runs (`configs/`)

The paper's settings, 3 ranks × 32 threads per hyades node (24 on hyades17/18), `launch.sh` and `start_all.sh`. `nlte_*`: the
paper's cases with `eos.model = "nlte"`; `nlteB_*`: μ = 1 on the NLTE-built τ; `lte_mhs_eos_mu1.0`: the paper's μ = 1 MHS on
this branch with the default LTE model (bit-identical to the paper's). `add_ne.jl` adds ln nₑ of each output from its own
table at (T, P); `compare.py` → `compare.txt`: the paper's metrics (same masks, gauge and layers) plus nₑ.

## Results (`compare.txt`)

| case | EOS | MHS steps | z(τ=1) rms [km] | depression [km] (Bifrost) | ΔlnP solved / above cut | Δlog nₑ solved / above cut [dex] |
|---|---|---|---|---|---|---|
| μ = 1, NLTE-built τ | NLTE | 7 | 22.5 | −67.5 (−77.1) | 0.073 / 0.62 | 0.045 / 0.28 |
| μ = 1, own (LTE-built) τ | LTE | 16 | 22.6 | −71.2 (−76.7) | 0.073 / 0.45 | 0.063 / 0.52 |
| | NLTE | 7 | 22.5 | −66.0 | 0.125 / 1.23 | 0.050 / 0.33 |
| μ = 1, M3D τ | LTE | 8 | 22.9 | −71.1 (−76.8) | 0.116 / 1.55 | 0.037 / 0.51 |
| | NLTE | 8 | 22.8 | −68.9 | 0.126 / 0.92 | 0.026 / 0.21 |
| μ = 0.707, inclined | LTE | 33 | 22.0 | −53.5 (−58.7) | 0.088 / 0.52 | 0.072 / 0.54 |
| | NLTE | 27 | 21.3 | −53.4 | 0.153 / 1.21 | 0.056 / 0.34 |
| μ = 0.5, inclined | LTE | 34 | 28.4 | −50.4 (−49.3) | 0.101 / 0.58 | 0.084 / 0.55 |
| | NLTE | 18 | 29.5 | −51.2 | 0.184 / 1.19 | 0.064 / 0.35 |
| μ = 0.5, plane-parallel | LTE | 25 | 47.2 | +4.5 (−49.3) | 0.098 / 0.58 | 0.083 / 0.55 |
| | NLTE | 19 | 39.9 | +1.5 | 0.180 / 1.19 | 0.064 / 0.35 |
| column mass | LTE / NLTE | 6 / 6 | 19.5 / 19.4 | −73.2 / −73.5 (−76.7) | 0.080 / 0.266 (both) | — |

Every run converged. Solved layers log τ ≥ −4 (mass grid: log m ≥ −0.5); "above cut" log τ −6 … −4.

* **nₑ** is better with the NLTE EOS in every case: by 20–30 % in the solved layers and by ≈ 2× above the cut.
* **z(τ=1) and the magnetic depression** are unchanged within ≈ 5 km. The depression is 2–5 km shallower with NLTE.
* **P and ρ depend on whether τ is consistent with the EOS.**
  * On the NLTE-built τ, NLTE matches LTE in the solved layers (0.073 vs 0.073).
  * On MULTI3D's τ, NLTE is 9 % worse in the solved layers (0.126 vs 0.116) and 40 % better above the cut (0.92 vs 1.55).
  * On τ integrated with the LTE χ500 (the paper's own-τ inputs), NLTE is 1.6–1.8× worse: the τ axis then carries the LTE
    opacity, which is 1–2 dex too low in cool chromospheric gas.
  * The loss is concentrated in log τ −4 … −3, next to the top boundary.
* **MHS steps** are the same or fewer, e.g. 16 → 7 and 34 → 18.
* On the mass grid τ does not enter, so the column-mass cases are identical.

**Use:** pair `eos.model = "nlte"` with a τ axis computed with NLTE opacities: the NLTE table's χ500 (`make_tau_input.jl …
table=cache/eos_table_nlte.h5`) or an NLTE radiative-transfer τ500. Don't use it with τ integrated from the LTE table.

## History of this report

The NLTE table was rebuilt three times before these results (earlier rounds in the `out/` folder, `v1_unlimited/`, `v2_lipschitz/`):

1. **Correction with cliffs in P.** The correction had jumps of up to 160 per dex of log P. MHS ran off them: one column was
   pushed to ln P 15.8, and the NLTE-τ MHS stalled. The fix was wider smoothing in P plus a slope limit of 2 per dex.
2. **Correction grid too narrow.** The grid ended at 10⁻⁴ Pa with |D| still up to 15 dex, then dropped to 0. The fix was to
   widen the grid to 10⁻¹² … 10⁸ Pa.
3. **Excited levels by Boltzmann from the NLTE ground state.** χ500 was 1.4–1.7 dex too high in 8–25 kK gas. That put the
   hot layer of the NLTE-built τ at log τ ≈ −4, and 2.7 % of its columns were 0.5–1.5 Mm off at the default cut. The fix was
   to put hydrogen's excited levels in Saha with the actual n_p and nₑ.

   `chi_vs_m3d.jl` and `chi_model_vs_m3d.jl` compare χ500 cell by cell with MULTI3D's `H/tau500` for snapshot 385. After the
   fix the NLTE χ500 is within ±0.2 dex of MULTI3D's (median) in every T, P bin. With the fix the default cut (−4) works;
   `nlteB_mhs_eosnlte_cut3` (cut −3, 5 steps, 23.1 km) is no longer needed.
