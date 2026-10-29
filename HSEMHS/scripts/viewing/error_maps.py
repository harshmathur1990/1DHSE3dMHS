"""Spatial comparison of a solve's z(τ_LOS = 1) error with the candidate input defects of the Bifrost truth
(design/CODEX_REVIEW_solver_time_round4_2026-10-29.md, finding 1): is the solved error the solver-map discrepancy?

usage: python error_maps.py RUNDIR TRUTH.h5 CONS.h5 INPUT.nc [STEP]
CONS.h5: truth_consistency.jl output (z1_rebuilt_err: the solver's τ–z map on the truth (T, P) from the truth cut height
minus the truth z(τ=1); cut_dlnP: column-HSE − truth ln P at the cut). STEP: snapshot (default final).

All maps gauge-aligned (mean removed). Reports, overall and by |B(τ=1)| class (strong > 500 G, quiet < 100 G):
rms of the solved error e_s and of the map discrepancy e_m; their correlation and regression slope of e_s on e_m; the
rms of e_s − e_m (the error against the model-projected truth z_truth + e_m); and the correlation of e_s with the cut
ln P mismatch. Also the same at large scales (box-car 21 × 21 columns ≈ 1 Mm) and the residual small scales.
"""
import glob
import sys

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

run, truth, cons, inp = sys.argv[1:5]
step = sys.argv[5] if len(sys.argv) > 5 else "final"
with h5py.File(truth) as f:
    lt = f["logtau500"][:]
    z1t = f["truth/z_tau1_m"][:]                                    # (y, x)
k0 = int(np.argmin(abs(lt)))
with h5py.File(cons) as f:
    em = f["z1_rebuilt_err"][:]; dP = f["cut_dlnP"][:]
with h5py.File(inp) as f:
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
files = sorted(glob.glob(f"{run}/snap_r*.h5"), key=lambda p: h5py.File(p).attrs["i0"])
hs = [h5py.File(p) for p in files]
kc = int(hs[0].attrs["kc"]) - 1
z = np.concatenate([h[step]["z"][:] for h in hs], axis=0).transpose(2, 1, 0)    # (τ, y, x)
z1 = z[k0 - kc]


def g(a):
    return a - np.nanmean(a)


es = g(z1 - z1t); em = g(em); dP = g(dP)
cls = [("all", np.ones_like(es, bool)), ("strong", B1 > 0.05), ("quiet", B1 < 0.01)]


def report(tag, a, b, c):
    print(tag)
    for name, m in cls:
        x = a[m]; y = b[m]; w = c[m]; ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]; w = w[ok]
        r = np.corrcoef(x, y)[0, 1]; slope = np.dot(x - x.mean(), y - y.mean()) / np.dot(y - y.mean(), y - y.mean())
        rp = np.corrcoef(x, w)[0, 1]
        rms = lambda v: np.sqrt(np.mean(v ** 2))
        print(f"  {name:6s}: rms e_s {rms(x) / 1e3:6.2f} km, e_m {rms(y) / 1e3:6.2f} km | corr {r:+.3f}, slope e_s on e_m {slope:+.3f} | "
              f"rms(e_s − e_m) {rms(x - y) / 1e3:6.2f} km | corr(e_s, cut ΔlnP) {rp:+.3f} | mean e_s {x.mean() / 1e3:+6.2f} km")


print(f"{run} [{step}] against {truth}; map discrepancy {cons}")
report("full resolution", es, em, dP)
L = lambda a: uniform_filter(np.nan_to_num(a), 21, mode="wrap")
report("large scales (21 × 21 box-car)", L(es), L(em), L(dP))
report("small scales (full − large)", es - L(es), em - L(em), dP - L(dP))
