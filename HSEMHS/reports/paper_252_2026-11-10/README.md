# Consistent τ inputs and the tests of the method paper (10 Nov 2026)

Source of Table 1 and Figs. 1–5 of the A&A method paper (`HSEMHS-Paper`, `hsemhs.tex`). Runs on `main` at 6e0a52b; outputs in
`/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_2026-11-10/`, inputs in `/mn/stornext/d9/data/harshm/MHS/IO/tau252/`.

## Why

The former τ input `IO/input-mhs-ltau.nc` had been interpolated on MULTI3D's `ltau500` with x and y swapped: each column's T
sat on the τ scale of the mirrored column (18 % rms in T at τ = 1 against a consistent input). Every τ-grid result built on it
(`../ltaucut`, the `out/modes` τ runs, the first paper draft) was invalid; the files were deleted on the user's request.

## Inputs (`IO/tau252`, builders in `HSEMHS/scripts/inputs/`)

All on the grid of the MULTI3D run `/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385` (`mesh`, `atm3d`; `H/tau500`
at μ = 1): 252² columns (95.24 km) and the snapshot's top 467 levels (14.37 … −1.02 Mm). The run's columns are found by
matching atm3d's T against the netCDF (exactly one 2-fold subsampling, among offsets, flips and a swap, agrees to 1e-5):
run column (i, j) = netCDF (x = 2i, y = 2(251 − j) + 1), 0-based; arrays are written with x, y increasing. T from atm3d;
B, P, ρ from the netCDF. Each input is written with its truth (snapshot z, ln P, ln ρ, T on the same levels and rays).

| input | τ | build log |
|---|---|---|
| `input_m3d_mu1.0` | MULTI3D `H/tau500` (vertical) | `build_logs/build_m3d_mu1.0.log`: z(τ=1) vs own τ: median −2.94 km, rms 1.12 km, corr 0.9996 |
| `input_eos_mu1.0`, `input_eos_mu0.707_az0`, `input_eos_mu0.5_az0` | integrated along the (inclined) rays with the EOS-table χ500 | `build_logs/` |
| `input_logm_mu1.0` | column mass (`make_logm_input.py --multi3d`) | `build_logs/build_logm_mu1.0.log` |

Checks: the own-τ build at μ = 1 reproduces the 504² truth subsampled on the same columns exactly (corr 1.0000, 0.00 km), and
is bit-identical with 1 and 64 threads. (A thread race in an intermediate version of `make_tau_input.jl` — a function-level
variable shared by a threaded loop — was found by this check and fixed before these builds.)

## Runs (`configs/`)

Release settings (`stop = "gn"`, `gn_tol` 1e-3, `lm = "gain"`, `forcing = "ew"`, `precond = "switch"`, `cg_cap` 200,
`max_steps` 100), 3 ranks × 32 threads per hyades node (24 on hyades18). `ms.sh`: `scripts/viewing/measure_steps.jl` with every
step saved (μ = 1, own τ) → `history_eos_mu1.0.csv` (`step_history.py`). `diag.sh`: `scripts/ltaucut/force_diag.jl`.
`metrics.py` → `metrics.txt` (Table 1); `make_figures.py` → `HSEMHS-Paper/images/`.

## Results (`metrics.txt`)

| case | z(τ=1) rms: col HSE → MHS | depression: col HSE → MHS (Bifrost) | rms ΔlnP, solved layers | MHS steps |
|---|---|---|---|---|
| μ = 1, own τ | 20.7 → 22.6 km | −56.8 → −71.2 (−76.7) km | 0.080 → 0.073 | 16, converged |
| μ = 1, MULTI3D τ | 20.7 → 22.9 km | −60.1 → −71.1 (−76.8) km | 0.127 → 0.116 | 8, converged |
| μ = 0.707 inclined | 21.6 → 22.0 km | −31.9 → −53.5 (−58.7) km | 0.093 → 0.088 | 33, converged |
| μ = 0.5 inclined | 23.2 → 28.4 km | −15.4 → −50.4 (−49.3) km | 0.104 → 0.101 | 34, undecided |
| μ = 0.5 plane-parallel | – → 47.2 km | – → +4.5 (−49.3) km | 0.098 | 25, converged |
| column mass | 56.6 → 19.5 km | −64.6 → −73.2 (−76.7) km | 0.103 → 0.080 | 6, converged |

Cuts at μ = 1 (own τ): −5 ends invalid (one folded column, no output); −6 and no cut stall (z(τ=1) rms 267 and 837 km;
median |F_h|/ρg at log τ −3…0 0.64 and 2.24, against 0.018 for −4, 0.072 column HSE and 0.058 Bifrost). Stop rules
(`history_eos_mu1.0.csv`): the qualified stop at step 16 (232 s of steps); the Borrero et al. (2021) rule (mean |Δz(τ=1)| <
6 km) holds at step 1 (mean 3.0 km, max 421 km, depression −65 km).

Follow-ups: the embedded-API test fixture `test/fixtures/embedded/ltau_*.h5` was cropped from the deleted input (it is a
numerical-regression fixture and still valid as such, but `oracles/python/make_embedded_fixtures.py` now points at a missing
file); column HSE on column mass places τ = 1 much worse than on τ (57 vs 21 km), not yet explained.

## Input-error test and mask sensitivity (added after the Codex review of 10 Nov 2026)

`make_perturbed_inputs.py` → `IO/tau252/perturbed/input_{small,large}_s{1,2,3}.nc` from `input_eos_mu1.0`: T·exp(σ_T ε),
B_i·(1 + σ_B ε_i), ε unit-variance fields Gaussian-correlated over 2 pixels and 0.5 dex; small σ_T 0.02, σ_B 0.10; large
0.05, 0.25; azimuth ambiguity assumed resolved. Column HSE and MHS (release settings) on each; `diag_pert.sh` force
diagnostics; `perturbed_metrics.py` → `perturbed_metrics.txt` (mask from the unperturbed field):

| errors | column HSE z(τ=1) rms / depression | MHS z(τ=1) rms / depression | MHS status |
|---|---|---|---|
| none | 20.7 km / −56.8 km (74 %) | 22.6 km / −71.2 km (93 %) | converged, 16 steps |
| small | 23.6 ± 0.1 / −56.5 ± 1.0 (74 %) | 23.2 ± 0.2 / −72.1 ± 0.3 (94 %) | converged, 17–19 |
| large | 35.9 ± 0.3 / −59.3 ± 5.1 (77 %) | 26.9 ± 0.5 / −68.9 ± 2.0 (90 %) | converged, 16–37 |

`depression_sensitivity.py` → `depression_sensitivity.txt`: strong thresholds 300 / 500 / 1000 G give MHS 86 / 93 / 96 %
(μ 1), 87 / 91 / 90 % (0.707), 100 / 102 / 102 % (0.5) against column HSE 74–75, 52–56, 26–37 %; 6 × 6 block bootstrap
(16–84 %): MHS 87–97, 86–95, 89–115 %; column HSE 71–77, 52–57, 28–35 %.

Force diagnostic (log τ −3 … 0, μ = 1): column HSE |F_h|, |F_z| = 0.072, 0.001 ρg; MHS 0.018, 0.014 ρg; Bifrost 0.058, 0.058.
