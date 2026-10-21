#!/usr/bin/env python3
"""A field on surfaces of constant log10 tau500 from saved m5h atmospheres (linear in log10 tau between the bracketing
levels of each column, as z_tau1). Columns without a bracketing pair (the level lies above the top of the column's tau)
are NaN (white) and counted.
  FIELD = T   : temperature [K], log colour scale.
  FIELD = Bz  : vertical field at cell centres (H B) [G], diverging scale symmetric about 0; with --input OBS.nc the
                observed input B_z, evaluated on the FIRST file's tau, is added as the first column.
usage: python3 m5h_isotau.py OUT.png FIELD LOGTAU1[,LOGTAU2,...] FILE.h5 [FILE2.h5 ...] [--input IO/input-mhs-hse.nc]"""
import sys
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

args = sys.argv[1:]
inp = None
if "--input" in args:
    i = args.index("--input"); inp = args[i + 1]; del args[i:i + 2]
out, field, levels, files = args[0], args[1], [float(v) for v in args[2].split(",")], args[3:]
assert field in ("T", "Bz")

def on_level(lt, f, L):
    """lt: (nz, ny, nx) log10 tau, top-first (increasing with k); f: field on the same grid -> f at log10 tau = L."""
    k = (lt <= L).sum(axis=0) - 1                     # last level with lt <= L (tau is monotone down each column)
    ok = (k >= 0) & (k < lt.shape[0] - 1)
    kc = np.clip(k, 0, lt.shape[0] - 2)
    a = np.take_along_axis(lt, kc[None], 0)[0]; b = np.take_along_axis(lt, kc[None] + 1, 0)[0]
    fa = np.take_along_axis(f, kc[None], 0)[0]; fb = np.take_along_axis(f, kc[None] + 1, 0)[0]
    w = np.where(b > a, (L - a) / np.where(b > a, b - a, 1.0), 0.0)
    return np.where(ok, fa + w * (fb - fa), np.nan)

INK, SURF = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "font.size": 10})
seq = LinearSegmentedColormap.from_list("seq", ["#fbe6dc", "#eb6834", "#5c1f08"])     # one hue, light -> dark
div = LinearSegmentedColormap.from_list("div", ["#174a8b", "#2a78d6", "#e4e3df", "#eb6834", "#8a2e0b"])   # two hues, gray midpoint
cols = []                                             # (label, {L: map})
first_lt = None
for fn in files:
    h = h5py.File(fn, "r")
    lt = np.log10(h["atmosphere/tau500"][()])
    f = h["atmosphere/T"][()] if field == "T" else 1e4 * h["atmosphere/Bz_centre"][()]
    acc = int(h.attrs["accepted"])
    if first_lt is None:
        first_lt = lt
    cols.append((f"accepted {acc}" if acc > 0 else "initial (fit)" if field == "Bz" else "initial", {L: on_level(lt, f, L) for L in levels}))
if inp is not None and field == "Bz":
    with h5py.File(inp, "r") as g:
        bz = 1e4 * np.transpose(g["B_z"][()], (2, 1, 0))      # file (x, y, z) -> (z, y, x), T -> G
    cols.insert(0, ("observed input (initial τ)", {L: on_level(first_lt, bz, L) for L in levels}))
unit = "K" if field == "T" else "G"
for lab, d in cols:
    for L in levels:
        m = d[L]
        print(f"{lab}: {field} at log10 tau500 = {L}: min {np.nanmin(m):.4g} {unit}, median {np.nanmedian(m):.4g}, max {np.nanmax(m):.4g}; "
              f"columns without the level {int(np.isnan(m).sum())} of {m.size}")
fig, axs = plt.subplots(len(levels), len(cols), figsize=(5.2 * len(cols), 4.7 * len(levels)), squeeze=False)
for r, L in enumerate(levels):
    vals = np.concatenate([d[L].ravel() for _, d in cols])
    if field == "T":
        lo, hi = np.nanpercentile(vals, [0.5, 99.5]); norm = LogNorm(lo, hi); cmap = seq; scl = "log scale"
    else:
        lim = np.nanpercentile(np.abs(vals), 99.5); norm = TwoSlopeNorm(0.0, -lim, lim); cmap = div; scl = "±99.5th pct"
    for c, (lab, d) in enumerate(cols):
        im = axs[r, c].imshow(d[L], origin="lower", cmap=cmap, norm=norm)            # h5py order: (y, x)
        axs[r, c].set_title(f"{field} at log10 τ500 = {L} [{unit}, {scl}], {lab}\n(white: level above the column top)", loc="left", fontsize=9.5)
        axs[r, c].set_xlabel("x index"); axs[r, c].set_ylabel("y index")
        fig.colorbar(im, ax=axs[r, c], shrink=0.85)
fig.tight_layout(); fig.savefig(out, dpi=120)
print("wrote", out)
