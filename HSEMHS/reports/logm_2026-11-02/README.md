# Log10 column-mass modes: check on the Bifrost cube (2 Nov 2026)

**Modes:** `column-hse-logm`, `mhs-poisson-logm` (branch `log10m-mode`), the column-mass counterparts of the `-ltau` modes.
**Input:** `IO/input-mhs-logm.nc`, Bifrost snapshot 385 on a common axis log10 m = −4.9 … 2.6 (m in kg m⁻², 0.1 dex, 76
levels), built by `make_input_mhs_logm.py`: m(z) = P_top/g + ∫ρ dz from the snapshot, T and B interpolated in log10 m.
**Truth:** the snapshot's P and z on the same axis (`truth_logm.h5`, made with `--truth`); z(τ = 1) from `out/ltaucut/truth.h5`.
**Comparison:** `scripts/logm/compare_truth.py`; full numbers in `summary.txt`.

## Runs

| run | where | wall | outcome | z(τ = 1) p1 / p50 / p99 |
|---|---|---|---|---|
| `column-hse-logm` | hyades10, 1 rank × 64 threads | 14 s | exit 0 | −167.1 / 15.0 / 104.9 km |
| `mhs-poisson-logm`, `logm_cut = −0.5`, defaults | hyades13, 1 rank × 96 threads | 187 s | exit 0, **`converged`, qualified**, 5 steps, Φ/Φ₀ 1.09e-2, decrement in [4.6e-7, 4.2e-4] | −41.6 / −1.0 / 40.0 km |

`logm_cut = −0.5` is the median log10 (P/g) at log τ = −4 of `out/modes/mhs-poisson-ltau.h5` (p1 −0.95, p99 −0.30).

## Against the Bifrost truth

Heights after removing the mean difference at log10 m = 1.5.

| log10 m | column HSE: rms Δ ln P, rms Δz | Poisson: rms Δ ln P, rms Δz |
|---|---|---|
| [−5, −1) (above the cut) | 0.536, 1068 km | 0.536, 1065 km |
| [−1, 0) | 0.155, 80 km | 0.148, 42 km |
| [0, 1) | 0.106, 66 km | 0.072, 24 km |
| [1, 2) | 0.085, 59 km | 0.071, 19 km |
| [2, 2.6] | 0.097, 40 km | 0.071, 28 km |

z(τ = 1) (truth std 30.6 km): column HSE rms 56.5 km, corr 0.575, std 68.3 km; Poisson rms 19.3 km, corr 0.778, std 21.8 km.

Bifrost's P/(g m) has p1 / p50 / p99 of 0.35 / 0.89 / 1.95 at log m = −2 and 0.92 / 1.04 / 1.35 at log m = 1: the region
above the cut (P = g m there) is not a hydrostatic atmosphere in the snapshot, which is the large error of the first row.

## How far this comparison goes

* **Not like for like with the τ modes.** The `reports/ltaucut` run with cut −4 has z(τ = 1) rms 27.4 km, corr 0.685, and the
  same cube on the τ axis ended `max_iter` at 25 steps in the release table, against 5 qualified steps here. But the axes
  hold different information: this input's m comes from the snapshot's true ρ and P, and an inversion does not deliver
  that. The grids (76 vs 95 levels), solver settings and code versions differ too. Whether the mass axis is really
  easier needs a controlled comparison; this run does not show it.
* **Only μ = 1 on the full cube.** μ < 1 is tested on the 11 × 12 fixture only (plane-parallel shift, inclined Jacobian,
  rank independence), not on the cube: this input is a vertical column mass.
* One rank on the full cube; the rank-count checks are on the fixture (2, 3, 4 ranks, `test/mpi/mpi_view.jl`).

## Tests

`test/runtests.jl` on hyades20 (8 threads): exit 0, including `test_logm.jl` (338 passed) and the multi-rank suites. The τ modes'
stored-reference and parity tests pass unchanged. On one uniform column the τ and mass axes approach each other with level
refinement ×1, ×4, ×16, ×64: max |Δz| 279 km, 11 km, 1.0 km, 88 m; max |Δ log τ| 0.24, 0.033, 0.0027, 0.00019.

## Data

`/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/logm_2026-11-02/`, linked from `data/` on the cluster.
