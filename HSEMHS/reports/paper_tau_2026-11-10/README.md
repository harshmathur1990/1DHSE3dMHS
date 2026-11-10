# Numbers and figures of the method paper (10 Nov 2026)

Source of Table 1 and Figs. 1–4 of the A&A method paper (`HSEMHS-Paper`, `hsemhs.tex`). Runs on `main` at 6e0a52b,
outputs in `/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_tau_2026-11-10/` (configs `*.toml`, `launch.sh`, `diag.sh`,
logs, `metrics.txt`, `ms100/history.csv`).

**Input and reference.** τ grid: `out/ltaucut/input-ltau-xy.nc` (504² × 95, log τ −8 … 1.4) and `out/ltaucut/truth.h5`
(Bifrost en024048 snapshot 385 on that grid), the pair of the cut study [`../ltaucut`](../ltaucut/README.md). Column mass:
the runs of [`../logm_2026-11-02`](../logm_2026-11-02/README.md) (column HSE) and
[`../logm_cut_2026-11-03`](../logm_cut_2026-11-03/README.md) (cut −0.5), not rerun. The 29 Oct release runs
(`out/delivery/`) are not used: their input `out/view/input_mu1.0_az0.nc` is built from slanted rays of the height cube and
differs from `input-ltau-xy.nc` (T by up to 1.5 MK), so `truth.h5` is not its reference.

**Runs.** One hyades node each (EPYC 7543), 3 ranks × 32 threads. MHS settings: the release settings (`stop = "gn"`,
`gn_tol` 1e-3, `lm = "gain"`, `forcing = "ew"`, `precond = "switch"`, `cg_cap` 200).

| run | node | config | |
|---|---|---|---|
| `colhse` | hyades10 | column-hse-ltau | 42 s wall |
| `mhs` | hyades11 | cut −4, 25 steps | converged (qualified) in 16 steps, 731 s wall |
| `ms100` | hyades12 | `scripts/viewing/measure_steps.jl`, cut −4, 100 steps, every step saved | same iterates as `mhs` (converged in 16) |
| `cut5`, `cut6`, `cutnone` | hyades15, 21, 20 | cut −5, −6, −7.9; 100 steps | running |

**Scripts** (copies run from `out/paper_tau_2026-11-10/scripts/`):
`paper_metrics.py` → `metrics.txt` (Table 1); `step_history.py` → `ms100/history.csv` and the step of the
Borrero et al. (2021) rule; `make_figures.py` → `HSEMHS-Paper/images/{ztau1,profiles,force,convergence}.pdf`;
`diag.sh` runs `scripts/ltaucut/force_diag.jl` (`diag_truth.h5` linked from `out/ltaucut`).

**Common mask and gauge** (both grids): strong / quiet = |B| > 500 G / < 100 G at log τ = 0 of the τ input (1.2 % /
91.5 %); every state, the truth included, shifted to mean z(τ500 = 1) = 0. This replaces the log10 m = 1.5 mask of the
column-mass reports, which gave a truth depression of −77.9 km instead of −74.2 km.

## Results (`metrics.txt`)

Solved band: log τ ≥ −4 (τ) or log10 m ≥ −0.5 (m). f2, f10: fraction within a factor 2, 10 of the truth.

| run | steps | z(τ=1) rms | corr | depression | rms ΔlnP | rms Δlnρ | rms Δz | f2 P | f2 ρ | f10 P, ρ |
|---|---|---|---|---|---|---|---|---|---|---|
| τ column HSE | – | 84.7 km | 0.611 | −105.5 km | 0.255 | 0.373 | 83.0 km | 0.966 | 0.909 | 1.000 / 0.999 |
| τ MHS −4 | 16 | 27.4 km | 0.685 | −77.6 km | 0.234 | 0.348 | 34.5 km | 0.972 | 0.920 | 1.000 |
| m column HSE | – | 56.5 km | 0.575 | −62.4 km | 0.103 | 0.106 | 59.6 km | 0.999 | 0.999 | 1.000 |
| m MHS −0.5 | 5 | 19.3 km | 0.778 | −76.9 km | 0.080 | 0.082 | 23.5 km | 1.000 | 1.000 | 1.000 |
| Bifrost | | 0 | 1 | −74.2 km | | | | | | |

* The factor-10 criterion of Borrero et al. (2019) (84 % MHS vs 47 % HSE in their sunspot) is met by every point of the
  solved band by both methods here; it does not discriminate in this weak-field snapshot.
* On the τ grid, P and ρ errors grow below τ = 1 (rms ΔlnP 0.6 at log τ 1.4) for column HSE and MHS alike; on the mass
  grid they stay near 0.1. Not yet explained (τ-axis discretisation across the photospheric T gradient, or the opacity).
* **Stop rules** (`ms100/history.csv`): the qualified stop at step 16 (499 s of steps, Φ/Φ0 0.0177); the Borrero et al.
  (2021) rule, mean |Δz(τ=1)| < dz/2 = 6 km, holds at step 6 (108 s, Φ/Φ0 0.0210): z(τ=1) error 27.4 km and depression
  −77.6 km there equal the final ones; max |Δz(τ=1)| at step 6 is 145 km.
