"""mhs-poisson-logm at several logm_cut against the Bifrost snapshot on the log10 m axis.

usage: python cut_study.py DIR TRUTH_LOGM TRUTH_TAU INPUT   (DIR holds cut_<value>.h5 and column-hse-logm.h5)
Writes DIR/summary.txt. Errors are also given on one fixed band, log10 m in [0.5, 2.6], which every cut solves, so the rows
compare. Heights after removing the mean difference at log10 m = 1.5. Strong/quiet: |B| > 500 G / < 100 G at log10 m = 1.5.
"""
import glob
import re
import sys

import h5py
import numpy as np

D, TL, TT, INP = sys.argv[1:5]
out = open(f"{D}/summary.txt", "w")


def p(*a):
    print(*a); print(*a, file=out)


def rms(a):
    return np.sqrt(np.mean(a ** 2))


with h5py.File(TL) as f:                                   # (x, y, m) → (m, y, x)
    lm = f["log10m"][:]
    Pt = f["Pgas"][:].transpose(2, 1, 0); zt = f["z_m"][:].transpose(2, 1, 0)
with h5py.File(TT) as f:
    z1t = f["truth/z_tau1_m"][:]
z1t = z1t - z1t.mean()
kr = int(np.argmin(abs(lm - 1.5)))
with h5py.File(INP) as f:
    B = np.sqrt(sum(f[k][:, :, kr] ** 2 for k in ("B_x", "B_y", "B_z"))).T
strong = B > 0.05; quiet = B < 0.01
band = (lm >= 0.5) & (lm <= 2.6)
p(f"strong {strong.mean() * 100:.1f} %, quiet {quiet.mean() * 100:.1f} %; truth z(τ=1) std {z1t.std() / 1e3:.1f} km, Wilson (strong − quiet) {(z1t[strong].mean() - z1t[quiet].mean()) / 1e3:.1f} km\n")
p("run         levels steps status     stop       Φ/Φ0     upper    wall |  z(τ=1) rms  corr   std  Wilson | fixed band rms ΔlnP  Δz km | solved rms ΔlnP")
files = sorted(glob.glob(f"{D}/cut_*.h5"), key=lambda s: float(re.search(r"cut_(.*)\.h5", s)[1]))
for fn in [f"{D}/column-hse-logm.h5"] + files:
    with h5py.File(fn) as f:
        g = f["mass_grid"]
        P = g["Pgas"][:]; z = g["z_m"][:]; z1 = g["z_tau1_m"][:]
        A = f.attrs
        dec = lambda k: (A[k].decode() if isinstance(A[k], bytes) else str(A[k])) if k in A else "-"
        cut = float(A["logm_cut"]) if "logm_cut" in A else None
        ph = A["phi_end"] / A["phi_start"] if "phi_end" in A else np.nan
        row = (dec("steps"), dec("status"), dec("qual_outcome"), ph, A.get("qual_upper", np.nan), A.get("solve_seconds", np.nan))
    dl = np.log(P / Pt); dz = z - zt; dz = dz - dz[kr].mean()
    a = z1 - z1.mean()
    W = a[strong].mean() - a[quiet].mean()
    s = lm >= cut if cut is not None else np.zeros_like(lm, bool)
    p(f"{'col HSE' if cut is None else f'cut {cut:+.1f}':10s} {s.sum():5d} {row[0]:>5s} {row[1]:10s} {row[2]:10s} {row[3]:8.2e} {row[4]:8.1e} {row[5]:6.0f} | "
      f"{rms(a - z1t) / 1e3:8.1f} {np.corrcoef(a.ravel(), z1t.ravel())[0, 1]:6.3f} {a.std() / 1e3:6.1f} {W / 1e3:7.1f} | "
      f"{rms(dl[band]):14.3f} {rms(dz[band]) / 1e3:7.1f} | {rms(dl[s]) if s.any() else np.nan:8.3f}")
