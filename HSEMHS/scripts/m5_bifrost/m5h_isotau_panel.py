#!/usr/bin/env python3
"""2 x 2 panel from one saved m5h atmosphere: T (top row, log scale) and Bz at cell centres (bottom row, diverging,
symmetric) on two constant log10 tau500 surfaces (linear in log10 tau, as z_tau1; white = level above the column top).
usage: python3 m5h_isotau_panel.py OUT.png FILE.h5 [L1=-5.1] [L2=-5.7]"""
import sys, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

def on_level(lt, f, L):
    """lt: (nz, ny, nx) log10 tau, top-first (increasing with k); f on the same grid -> f at log10 tau = L (as m5h_isotau.py)."""
    k = (lt <= L).sum(axis=0) - 1
    ok = (k >= 0) & (k < lt.shape[0] - 1)
    kc = np.clip(k, 0, lt.shape[0] - 2)
    a = np.take_along_axis(lt, kc[None], 0)[0]; b = np.take_along_axis(lt, kc[None] + 1, 0)[0]
    fa = np.take_along_axis(f, kc[None], 0)[0]; fb = np.take_along_axis(f, kc[None] + 1, 0)[0]
    w = np.where(b > a, (L - a) / np.where(b > a, b - a, 1.0), 0.0)
    return np.where(ok, fa + w * (fb - fa), np.nan)

out, fn = sys.argv[1], sys.argv[2]
levels = [float(sys.argv[3]) if len(sys.argv) > 3 else -5.1, float(sys.argv[4]) if len(sys.argv) > 4 else -5.7]
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "text.color": "#0b0b0b", "font.size": 10})
seq = LinearSegmentedColormap.from_list("seq", ["#fbe6dc", "#eb6834", "#5c1f08"])
div = LinearSegmentedColormap.from_list("div", ["#174a8b", "#2a78d6", "#e4e3df", "#eb6834", "#8a2e0b"])
h = h5py.File(fn, "r")
lt = np.log10(h["atmosphere/tau500"][()]); T = h["atmosphere/T"][()]; Bz = 1e4 * h["atmosphere/Bz_centre"][()]
acc = int(h.attrs["accepted"])
fig, axs = plt.subplots(2, 2, figsize=(11, 9.6))
for c, L in enumerate(levels):
    t = on_level(lt, T, L); b = on_level(lt, Bz, L)
    lo, hi = np.nanpercentile(t, [0.5, 99.5]); lim = np.nanpercentile(np.abs(b), 99.5)
    im = axs[0, c].imshow(t, origin="lower", cmap=seq, norm=LogNorm(lo, hi)); fig.colorbar(im, ax=axs[0, c], shrink=0.85, label="K")
    axs[0, c].set_title(f"T at log10 τ500 = {L} (log scale)\nmedian {np.nanmedian(t):.0f} K", loc="left", fontsize=10)
    im = axs[1, c].imshow(b, origin="lower", cmap=div, norm=TwoSlopeNorm(0.0, -lim, lim)); fig.colorbar(im, ax=axs[1, c], shrink=0.85, label="G")
    axs[1, c].set_title(f"Bz at log10 τ500 = {L}\nrange {np.nanmin(b):.0f} … {np.nanmax(b):.0f} G", loc="left", fontsize=10)
    for r in (0, 1):
        axs[r, c].set_xlabel("x index"); axs[r, c].set_ylabel("y index")
    print(f"L = {L}: T median {np.nanmedian(t):.0f} K (min {np.nanmin(t):.0f}, max {np.nanmax(t):.0f}); Bz {np.nanmin(b):.1f} .. {np.nanmax(b):.1f} G; "
          f"columns without the level {int(np.isnan(t).sum())}")
fig.suptitle(f"Full-resolution MHS state, accepted iteration {acc} (white: level above the column top)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=120)
print("wrote", out)
