"""Per-step history of a scripts/viewing/measure_steps.jl run against the Bifrost truth, and the step at which the
stopping rule of Borrero et al. (2021) would have stopped it.

usage: python step_history.py TRUTH.h5 INPUT.nc RUNDIR
Writes RUNDIR/history.csv and prints it. Columns: step, cumulative solve time (s), Φ/Φ0, predicted relative decrease of
the next Gauss–Newton step, mean |Δz1|, rms Δz1 and max |Δz1| between consecutive steps (km; z1 = z(τ500 = 1) on the
log τ = 0 level, mean removed per step, i.e. the gauge), the z1 rms error against the truth (km) and the depression of
strong minus quiet columns (km; |B| > 500 G / < 100 G at log τ = 0).
Borrero et al. (2021): converged once the mean change of z(τ = 1) between two iterations is below dz/2 = 6 km (dz 12 km).
"""
import glob
import sys

import h5py
import numpy as np

truth, inp, run = sys.argv[1:4]
with h5py.File(truth) as f:
    lt = f["logtau500"][:]
    z1t = f["truth/z_tau1_m"][:]
k0 = int(np.argmin(abs(lt)))
with h5py.File(inp) as f:
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
strong, quiet = B1 > 0.05, B1 < 0.01
z1t = z1t - z1t.mean()


def rms(a):
    return np.sqrt(np.mean(a ** 2))


hs = [h5py.File(p) for p in sorted(glob.glob(f"{run}/snap_r*.h5"), key=lambda p: h5py.File(p).attrs["i0"])]
kc = int(hs[0].attrs["kc"]) - 1
steps = sorted(int(n[1:]) for n in hs[0].keys() if n.startswith("s") and n[1:].isdigit())


def z1(name):
    z = np.concatenate([h[name]["z"][:] for h in hs], axis=0).transpose(2, 1, 0)[k0 - kc]
    return z - z.mean()


phi0 = hs[0]["s0"].attrs["phi"]
rows = []; zprev = None; cum = 0.0; borrero = None
for s in steps:
    a = hs[0][f"s{s}"].attrs
    z = z1(f"s{s}")
    if s > 0:
        cum += a["seconds"]
    d = (z - zprev) if zprev is not None else np.full_like(z, np.nan)
    r = (s, cum, a["phi"] / phi0, a.get("pred", np.nan), np.mean(np.abs(d)) / 1e3, rms(d) / 1e3, np.max(np.abs(d)) / 1e3,
         rms(z - z1t) / 1e3, (z[strong].mean() - z[quiet].mean()) / 1e3)
    rows.append(r)
    if borrero is None and s > 0 and r[4] < 6.0:
        borrero = s
    zprev = z

hdr = "step,cum_s,phi_rel,pred,mean_dz1_km,rms_dz1_km,max_dz1_km,err_z1_km,depression_km"
with open(f"{run}/history.csv", "w") as f:
    f.write(hdr + "\n")
    for r in rows:
        f.write(",".join(f"{v:.6g}" for v in r) + "\n")
print(hdr)
for r in rows:
    print(",".join(f"{v:.4g}" for v in r))
print(f"Borrero et al. (2021) rule (mean |Δz1| < 6 km): step {borrero}")
