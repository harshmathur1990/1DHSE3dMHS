#!/usr/bin/env python3
"""M5h progress figures from a running or finished m5h_fullres.jl output directory.
  1. height_change.png : max / p99 / median |Δz(τ500 = 1)| between adjacent accepted iterations [m], log scale, with the
                         user's threshold 1e-3 m marked;
  2. force.png         : max layer RMS of |R|/S per accepted iteration (physical MHS progress, separate axis by rule);
  3. ztau1_maps.png    : z(τ500 = 1) initial and latest [km] and their difference [m].
usage: python3 m5h_plots.py OUTDIR"""
import sys, os, glob
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

out = sys.argv[1]
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})

h = np.genfromtxt(os.path.join(out, "history.csv"), delimiter=",", names=True, dtype=None, encoding=None)
h = np.atleast_1d(h)
acc = h["accepted"]; m = acc > 0
fig, ax = plt.subplots(figsize=(8, 4.2))
if m.any():
    for col, c, lab in (("dz_max_m", S1, "max over columns"), ("dz_p99_m", S2, "p99"), ("dz_median_m", S3, "median")):
        y = np.where(h[col][m] > 0, h[col][m], np.nan)
        ax.plot(acc[m], y, color=c, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (acc[m][-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.axhline(1e-3, color=INK, lw=1.2, ls="--")
ax.annotate("threshold 1e-3 m", (0.01, 1e-3), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points", color=INK, fontsize=9)
ax.set_yscale("log"); ax.set_xlabel("accepted iteration"); ax.set_ylabel("|Δz(τ500 = 1)| between adjacent accepted iterations [m]")
last = h[-1]
ax.set_title(f"Height change of τ500 = 1 — 504 × 504 columns; latest max {last['dz_max_m']:.3g} m at accepted {int(last['accepted'])}",
             loc="left", fontsize=10.5)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(out, "height_change.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(acc, h["rms_max"], color=S1, lw=2, marker="o", ms=4)
ax.set_yscale("log"); ax.set_xlabel("accepted iteration"); ax.set_ylabel("max layer RMS |R|/S")
ax.set_title(f"Force residual (target 0.01) — latest {last['rms_max']:.3g}; raw force {last['raw_force']:.3g}", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(out, "force.png"), dpi=150); plt.close(fig)

files = sorted(glob.glob(os.path.join(out, "ztau1", "ztau1_*.h5")))
if files:
    rd = lambda p: (h5py.File(p, "r")["z_tau1_m"][()].T, h5py.File(p, "r")["flag"][()].T)
    z0, f0 = rd(files[0]); z1, f1 = rd(files[-1])
    n_last = int(os.path.basename(files[-1]).split("_")[1].split(".")[0])
    seq = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#2a78d6", "#0d2d57"])
    div = LinearSegmentedColormap.from_list("div", [S1, "#e4e3df", S2])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    vmin, vmax = np.nanpercentile(np.r_[z0.ravel(), z1.ravel()] / 1e3, [1, 99])
    for a, zz, t in ((axs[0], z0, "initial"), (axs[1], z1, f"accepted {n_last}")):
        im = a.imshow(zz.T / 1e3, origin="lower", cmap=seq, vmin=vmin, vmax=vmax); a.set_title(f"z(τ500 = 1) [km], {t}", loc="left")
        fig.colorbar(im, ax=a, shrink=0.85)
    d = z1 - z0; lim = np.nanpercentile(np.abs(d), 99) or 1.0
    im = axs[2].imshow(d.T, origin="lower", cmap=div, norm=TwoSlopeNorm(0, -lim, lim))
    axs[2].set_title(f"latest − initial [m] (max |Δ| {np.nanmax(np.abs(d)):.3g} m)", loc="left"); fig.colorbar(im, ax=axs[2], shrink=0.85)
    for a in axs:
        a.set_xlabel("x index"); a.set_ylabel("y index"); a.grid(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ztau1_maps.png"), dpi=130); plt.close(fig)
print("wrote figures in", out)
