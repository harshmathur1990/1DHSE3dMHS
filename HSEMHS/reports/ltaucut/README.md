# mhs-poisson-ltau: choice of the τ cut (26 Oct 2026)

Input `IO/input-mhs-ltau.nc` (504², 95 levels, log τ −8 … 1.4; x, y added from the Bifrost cube, dx = 47.62 km),
truth = Bifrost en024048 snapshot 385 mapped onto the τ grid (`scripts/ltaucut/truth_on_tau.jl`, τ500 from Bifrost T, P
with our EOS table; input T agrees to 0.1–1 % median at log τ −4.5 … −0.5). Force balance from one geometric formula for
all states (`scripts/ltaucut/force_diag.jl`); comparison `scripts/ltaucut/analyse.py` → `summary.txt`, figures here.
Runs: cluster worktree `MHS_ltaucut/HSEMHS/out/ltaucut/` (configs `cut*.toml`, logs, outputs).

| run | steps / stop | Φ/Φ₀ | wall | z(τ=1) err rms | corr | Wilson (strong − quiet) |
|---|---|---|---|---|---|---|
| column HSE | – | – | 78 s | 85 km | 0.61 | −106 km |
| cut −4 | 19 / converged | 1.8e-2 | 16 min | **27 km** | **0.69** | **−78 km** |
| cut −5 | 100 / max_steps | 2.2e-3 | 89 min | 40 km | 0.62 | −104 km |
| cut −6 | 100 / max_steps | 1.1e-1 | 64 min | 1137 km | 0.09 | −180 km |
| no cut (−7.9) | 100 / max_steps | 1.8e-1 | 81 min | 20184 km | 0.06 | −16 Mm |
| Bifrost | | | | 0 | 1 | −74 km |

Bifrost β (median): 21 at log τ −4 (5 % of points < 1), 3 at −5 (35 %), 0.15 at −5.5 (73 %), < 0.02 above −6.
Once the solved region contains low-β layers, the unbalanced Lorentz force of a non-force-free field is carried by the
pressure and, through the opacity, by the heights of the whole column below: the τ=1 surface is pushed by Mm in large
coherent patches and the residual force over the photosphere ends worse (median |F_h|/ρg ≈ 1 for −6, ≈ 4 for no cut)
than column HSE (≈ 0.2). Cut −4 converges, lowers |F_h|/ρg below Bifrost's own dynamic floor (0.02–0.05 vs 0.05–0.1),
reduces the z(τ=1) error 3× and recovers the Wilson depression. Cut −5 is close but slower, not converged in 100
steps, and overestimates the Wilson depression. Above the cut the τ-hydrostatic upper layer is off Bifrost by
1.5–1.9 rms in ln P for every run. Keep logtau_cut = −4.
