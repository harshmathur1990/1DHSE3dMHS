"""Compare column-hse-logm and mhs-poisson-logm with the Bifrost snapshot on the log10 m axis, and z(τ = 1) with the same
truth as the log τ cut study.

usage: python compare_truth.py DIR [TRUTH_TAU]   (DIR holds truth_logm.h5, column-hse-logm.h5, mhs-poisson-logm.h5;
TRUTH_TAU = out/ltaucut/truth.h5 with truth/z_tau1_m). Writes DIR/summary.txt. Heights are compared after removing the mean
difference at log10 m = 1.5, as the runs and the snapshot have different zero points.
"""
import sys

import h5py
import numpy as np

D = sys.argv[1]
out = open(f"{D}/summary.txt", "w")


def p(*a):
    print(*a); print(*a, file=out)


def rms(a):
    return np.sqrt(np.mean(a ** 2))


with h5py.File(f"{D}/truth_logm.h5") as f:                # h5py (x, y, m) → (m, y, x)
    lm = f["log10m"][:]
    Pt = f["Pgas"][:].transpose(2, 1, 0); zt = f["z_m"][:].transpose(2, 1, 0)
kr = int(np.argmin(abs(lm - 1.5)))
BANDS = ((-5, -1), (-1, 0), (0, 1), (1, 2), (2, 3))
p("truth P/(g m): " + "; ".join(f"log m {lm[k]:.1f} p1/p50/p99 " + "/".join(f"{v:.3f}" for v in np.percentile(Pt[k] / (274 * 10 ** lm[k]), [1, 50, 99]))
                                 for k in (int(np.argmin(abs(lm - L))) for L in (-2, 0, 1, 2))))
runs = {}
for name in ("column-hse-logm", "mhs-poisson-logm"):
    with h5py.File(f"{D}/{name}.h5") as f:
        g = f["mass_grid"]
        P = g["Pgas"][:]; z = g["z_m"][:]; zt1 = g["z_tau1_m"][:]; lt = g["logtau500"][:]
        at = {k: f.attrs[k] for k in f.attrs if k in ("status", "stop_reason", "steps", "phi_start", "phi_end", "gn_decrement", "solve_seconds")}
    runs[name] = zt1
    dl = np.log(P / Pt)
    dz = z - zt
    dz = dz - dz[kr].mean()
    p(f"\n{name}: {at}")
    p(f"  median log τ at log m = 1.5: {np.median(lt[kr]):.2f}; z(τ=1) p1/p50/p99 " + " / ".join(f"{v / 1e3:.1f}" for v in np.percentile(zt1, [1, 50, 99])) + " km")
    for lo, hi in BANDS:
        s = (lm >= lo) & (lm < hi)
        p(f"  log m [{lo:2d}, {hi:2d}): rms Δ ln P {rms(dl[s]):.3f}, median Δ ln P {np.median(dl[s]):+.3f}, rms Δz {rms(dz[s]) / 1e3:7.1f} km")
if len(sys.argv) > 2:
    with h5py.File(sys.argv[2]) as f:
        z1t = f["truth/z_tau1_m"][:]
    z1t = z1t - z1t.mean()
    p(f"\nz(τ=1) against the Bifrost truth ({sys.argv[2]}), std truth {z1t.std() / 1e3:.1f} km:")
    for name, zz in runs.items():
        d = (zz - zz.mean()) - z1t
        p(f"  {name}: rms {rms(d) / 1e3:.1f} km, corr {np.corrcoef(zz.ravel(), z1t.ravel())[0, 1]:.3f}, std {zz.std() / 1e3:.1f} km")
