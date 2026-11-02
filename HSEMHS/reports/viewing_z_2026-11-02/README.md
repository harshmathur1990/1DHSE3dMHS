# Viewing angle in the height modes: check on the Bifrost cube (2 Nov 2026)

**Input:** `IO/input-mhs-hse.nc`, Bifrost 504 × 504 × 177 (dx = dy = 47.61 km).
**Code:** `main` at `f28f132`; the description is in the top-level README, section "Viewing angle" (`0e4344a`).
**Question:** does `[view]` work in `column-hse-z` and `mhs-poisson-z`, and does it change only τ?

## Runs

| run | mode | view | wall | z(τ₅₀₀ = 1) p1 / p50 / p99 |
|---|---|---|---|---|
| `zview_col_mu1` | column-hse-z | μ = 1 | 308 s | −29.2 / 63.5 / 127.4 km |
| `zview_col_mu05_pp` | column-hse-z | μ = 0.5 plane_parallel | 308 s | 11.3 / 103.1 / 156.7 km |
| `zview_col_mu05_inc` | column-hse-z | μ = 0.5 inclined, φ = 0 | 183 s | 22.3 / 103.3 / 154.9 km |
| `zview_poi_mu1` | mhs-poisson-z | μ = 1 | 608 s | −2.6 / 69.9 / 125.7 km |
| `zview_poi_mu05_inc` | mhs-poisson-z | μ = 0.5 inclined, φ = 0 | 357 s | 46.2 / 108.1 / 162.7 km |

The Poisson runs used 4 ranks × 32 threads. Their logs end in an MPI teardown error
**after** the output was written; the files are complete.

## Results

1. P, ρ and χ₅₀₀ are **bitwise identical** for μ = 1, 0.5 plane-parallel and 0.5 inclined, in both modes. The view
   changes only `tau500`/`logtau500` and z(τ = 1).
2. Plane-parallel τ(μ = 0.5) / τ(μ = 1) = 2 exactly at every node.
3. Inclined: an exact re-implementation of `src/ViewTau.jl` (shear χ to footprints with periodic bilinear interpolation,
   trapezoid dz/μ, exp-tail top closure, shift back) reproduces τ to 4e-13 and z(τ = 1) to 7e-11 m.
   A first check that interpolated along the exact ray instead showed differences up to 13 % in τ and 2.6 km in
   z(τ = 1); that is the difference between two interpolation schemes, not a code error.
4. z(τ = 1) rises by about 40 km at μ = 0.5 (std 37.6 → 35.0 km plane-parallel, 31.3 km inclined: the slanted rays
   smooth the surface).

## Files

- `configs/`: the run TOMLs; `logs/`: run logs and `run_one.sh`.
- Check script: `HSEMHS/scripts/viewing/check_zview.py`.
- `data/` (not in git): links to the outputs in
  `/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/viewing_z_2026-11-02/` (about 19 GB).
