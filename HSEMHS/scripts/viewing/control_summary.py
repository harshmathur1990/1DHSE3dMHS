"""Summary of the known-equilibrium control refinement sweep (out/control/sw_<grid>_mu<μ>_<track>, written by
out/control/sweep.sh and analysed by measure_analyse.py into steps.csv).

usage: python control_summary.py [GRID ...]   (default: every grid found)
Per grid and μ, for the tracks col (production column-HSE boundary, reference solver options), bt (experimental truth
boundary, started from the exact truth) and prod (column boundary, production defaults):
  defect   rms residual of the exact truth on the τ grid, √(Φ/ΣW) with the run's own weights (bt, step 0)
  z1       rms z(τ_LOS = 1) error after removing the mean, and the mean offset (km)
  ΔlnP     rms ln P error by log τ band [-4,-2) [-2,0) [0,1.5)
  low−high truth depression contrast and the solution's (km)
  steps to the final height error (first step within 10 % or 0.05 km of it), the stop reason, Φ/Φ₀, solve seconds
"""
import glob
import re
import sys

import h5py
import numpy as np


def run(name):
    rows = list(open(f"{name}/steps.csv"))
    keys = rows[0].strip().split(",")
    rec = [dict(zip(keys, r.strip().split(","))) for r in rows[1:]]
    f = h5py.File(sorted(glob.glob(f"{name}/snap_r*.h5"))[0])
    a = f.attrs
    W = None
    if "weights" in f:
        wc = f["weights/wc"][:]; wf = f["weights/wf"][:]
        W = (2 * wc.sum() + wf.sum()) * a["NY"] * a["NX"]
    s0 = f["s0"].attrs["phi"] if "s0" in f else np.nan
    fin = rec[-1]
    e = float(fin["err"])
    steps = [r for r in rec if r["step"] not in ("final",)]
    hit = next((int(r["step"][1:]) for r in steps if float(r["err"]) <= max(1.1 * e, e + 0.05)), -1)
    out = dict(err=e, off=float(fin["off"]), wil=float(fin["wil"]), b=[float(fin[k]) for k in ("b0", "b1", "b2")],
               reason=a["reason"].decode() if isinstance(a["reason"], bytes) else a["reason"], steps=int(a["steps"]),
               phi=float(fin["phi"]), sec=float(a["solve_seconds"]), hit=hit,
               defect=np.sqrt(s0 / W) if W else np.nan, qual=a.get("qual_outcome", b"-"))
    f.close()
    return out


def truth_wil(name):
    t = open(f"{name}/steps.txt").readline()
    m = re.search(r"truth \S+ ([-+0-9.]+) km", t)
    return float(m.group(1)) if m else np.nan


grids = sys.argv[1:] or sorted({re.match(r"sw_(nx\d+_dl[0-9.]+)_", d).group(1) for d in glob.glob("sw_*")})
print("grid           μ    track | defect   | z1 rms  mean off | ΔlnP [-4,-2) [-2,0) [0,1.5) | low−high (truth) | hit steps reason       Φ/Φ0     s     qual")
for g in grids:
    for m in ("1", "0.3"):
        for tr in ("col", "bt", "prod"):
            n = f"sw_{g}_mu{m}_{tr}"
            try:
                r = run(n)
            except Exception as ex:                       # not finished or failed
                print(f"{g:14s} {m:4s} {tr:5s} | {type(ex).__name__}")
                continue
            q = r["qual"].decode() if isinstance(r["qual"], bytes) else r["qual"]
            print(f"{g:14s} {m:4s} {tr:5s} | {r['defect']:.2e} | {r['err']:6.2f} {r['off']:+8.1f} | "
                  f"{r['b'][0]:.4f} {r['b'][1]:.4f} {r['b'][2]:.4f} | {r['wil']:+6.1f} ({truth_wil(n):+6.1f}) | "
                  f"{r['hit']:3d} {r['steps']:5d} {r['reason']:12s} {r['phi']:.2e} {r['sec']:5.0f} {q}")
