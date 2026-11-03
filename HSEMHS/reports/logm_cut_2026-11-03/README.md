# Which `logm_cut` for `mhs-poisson-logm`? (3 Nov 2026)

**Result: keep −0.5.** z(τ = 1) barely depends on the cut from −2 to +0.5; only −3 is clearly worse.

**Input:** `IO/input-mhs-logm.nc` (Bifrost snapshot 385, 76 levels, μ = 1). **Runs:** defaults except the cut, 1 rank × 64 threads,
one node each. **Compared with:** the snapshot on the mass axis and z(τ = 1) of `out/ltaucut/truth.h5`, by
`scripts/logm/cut_study.py`. Numbers in `summary.txt`.

| cut | solved levels | steps | status | wall | z(τ=1) rms | corr | Wilson depression | band rms Δ ln P | band rms Δz |
|---|---|---|---|---|---|---|---|---|---|
| column HSE | 0 | – | – | 14 s (2 Nov run) | 56.5 km | 0.575 | −61.7 km | 0.091 | 54.7 km |
| −3.0 | 57 | 25 | `max_iter` | 1356 s | 33.7 km | 0.535 | −119.8 km | 0.070 | 35.9 km |
| −2.0 | 47 | 11 | converged | 597 s | 19.6 km | 0.769 | −81.6 km | 0.070 | 22.8 km |
| −1.5 | 42 | 7 | converged | 290 s | 19.2 km | 0.781 | −81.1 km | 0.069 | 22.4 km |
| −1.0 | 37 | 6 | converged | 231 s | 19.2 km | 0.781 | −81.8 km | 0.069 | 22.4 km |
| **−0.5** | 32 | 5 | converged | 167 s | 19.3 km | 0.778 | −82.5 km | 0.069 | 22.6 km |
| 0.0 | 27 | 5 | converged | 169 s | 19.5 km | 0.776 | −82.8 km | 0.070 | 22.8 km |
| +0.5 | 22 | 5 | converged | 147 s | 19.5 km | 0.776 | −82.9 km | 0.074 | 23.2 km |

Truth: z(τ = 1) std 30.6 km, Wilson depression (|B| > 500 G minus < 100 G at log m = 1.5) −77.9 km. "Converged" is the qualified
stop. The band is log10 m in [0.5, 2.6], solved by every cut, with heights after removing the mean difference at log m = 1.5.
Wall is `solve_seconds`.

## Reading

* **−3 is too high.** There 89 % of the cube has β < 1 and the median Bifrost P/(g m) is 0.81. It does not converge in 25 steps,
  and z(τ = 1) is worse than at every lower cut, with the Wilson depression overdone by 40 km.
* **−2 to +0.5 is a plateau.** z(τ = 1) rms spans 19.2–19.6 km. Those differences are small against the 19 km error itself and
  this is one snapshot, so −1.5 and −1.0 are not shown to be better than −0.5.
* **The cost is not flat.** −1.0 takes 38 % longer than −0.5, −1.5 74 %, −2.0 3.6×, for at most 0.1 km of z(τ = 1) rms.
* **Going deeper has a limit we did not reach.** +0.5 is still fine on these metrics, with the band Δ ln P starting to rise
  (0.074). It leaves the layers between log m −0.5 and +0.5 (β roughly 20 to 2000) unsolved, where the force balance still
  matters on physical grounds, so it is not preferred on this evidence alone.
* Φ/Φ₀ and the error over the solved levels are not compared across cuts: the weights and levels differ with the cut.

## Limits

One snapshot, μ = 1. The input's m comes from the snapshot's true ρ and P, which an inversion does not have, so the values transfer
to inversion output only as far as its m(T, B) is close to this one. A different atmosphere (e.g. the Hα input) may move the
edge of the plateau.

## Data

`/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/logm_cut_2026-11-03/`, linked from `data/` on the cluster.
