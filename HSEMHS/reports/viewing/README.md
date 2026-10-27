# Viewing-angle experiments (P4, 27 Oct 2026)

Design: `design/tau/VIEWING_ANGLE_DESIGN.md` §7. Code: branch `viewing-angle` (291fea3).

Synthetic off-centre τ cubes from Bifrost en024048_hion 385 (`scripts/viewing/make_slanted_tau.jl`): one line of sight
per footprint at μ = 1, 0.866, 0.707, 0.5 (φ = 0°) and μ = 0.5 (φ = 45°), on the 95-level log τ axis of
`IO/input-mhs-ltau.nc`; input and truth from the same calculation. Runs: column-hse-ltau, mhs-poisson-ltau plane-parallel
and inclined (cut −4, max_steps 100, rtol 1e-4), 3 ranks × 32 threads on hyades18/20. Analysis:
`scripts/viewing/analyse_view.py` → `summary.txt`, `ztau1_vs_mu.png`, `profiles.png`, `maps_*.png`.
Cluster run directory: `/mn/stornext/d9/data/harshm/MHS_view/HSEMHS/out/view/`.

| μ (φ) | z(τ_LOS=1) rms error: column HSE / plane-parallel / inclined | strong − quiet height, Bifrost: column HSE / pp / inclined |
|---|---|---|
| 1 | 20.7 / 22.6 / 22.6 km | −77: −57 / −74 / −74 km |
| 0.866 | 21.4 / 24.1 / 22.5 km | −68: −42 / −42 / −51 km |
| 0.707 | 22.4 / 33.5 / 22.5 km | −60: −29 / −9 / −46 km |
| 0.5 | 23.9 / 69.3 / 27.0 km | −49: −12 / +22 / −38 km |
| 0.5 (45°) | 23.6 / 81.5 / 33.2 km | −50: −13 / +21 / −46 km |

Plane-parallel Poisson degrades fast off-centre (wrong sign of the strong-field depression at μ = 0.5); the inclined
geometry keeps the rms error near the μ = 1 value and recovers most of the depression. Column HSE has the lowest rms
error at every μ but misses most of the depression. Poisson runs take 30–110 min (40–70 s per step, CG at its 80-iteration
cap, 41–100 steps): too slow for use inside an inversion loop as it stands.
