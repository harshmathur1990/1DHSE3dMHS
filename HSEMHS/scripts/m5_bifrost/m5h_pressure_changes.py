#!/usr/bin/env python3
"""Pressure and density changes of an m5h run.
  (a) per accepted iteration, from history.csv: max |ΔP/P| (the solver caps |Δ ln P| at 5 per step, so the cap is
      e^5 − 1 = 147.4) and max |ΔB|/max(|B|, σ);
  (b) between saved states (init.h5 and the checkpoints): per height level, max and p99 of |Δ log10 P| and
      |Δ log10 ρ| over the 504 × 504 columns, and the fraction of nodes on the pressure floor (u − ulo < 1e-6).
usage: python3 m5h_pressure_changes.py RUNDIR OUT_PREFIX STATE0.h5 STATE1.h5 [STATE2.h5 ...]"""
import sys, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

run, pre, states = sys.argv[1], sys.argv[2], sys.argv[3:]
S1, S2, S3, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID})
h = np.atleast_1d(np.genfromtxt(os.path.join(run, "history.csv"), delimiter=",", names=True, dtype=None, encoding=None))
m = h["accepted"] > 0
fig, axs = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
axs[0].plot(h["accepted"][m], h["dPmax"][m], color=S1, lw=2, marker="o", ms=3)
axs[0].axhline(np.e ** 5 - 1, color=INK, ls="--", lw=1.1)
axs[0].annotate("per-step cap |Δ ln P| = 5  (e⁵ − 1 = 147.4)", (0.01, np.e ** 5 - 1), xycoords=("axes fraction", "data"), xytext=(0, -12),
                textcoords="offset points", fontsize=9)
axs[0].set_yscale("log"); axs[0].set_ylabel("max |ΔP/P| per accepted iteration")
axs[0].set_title("Pressure change between adjacent accepted iterations (all 45 M cells)", loc="left", fontsize=10.5)
axs[1].plot(h["accepted"][m], h["dBmax"][m], color=S1, lw=2, marker="o", ms=3)
axs[1].set_yscale("log"); axs[1].set_ylabel("max |ΔB| / max(|B|, σ)"); axs[1].set_xlabel("accepted iteration")
axs[1].set_title("Magnetic change between adjacent accepted iterations", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(pre + "_per_iteration.png", dpi=130); plt.close(fig)

def load(fn):
    with h5py.File(fn, "r") as f:
        ulo = f["solver/ulo"][()][:, 1:-1, 1:-1]
        return (np.log10(f["atmosphere/Pgas"][()]), np.log10(f["atmosphere/rho"][()]), np.log(f["atmosphere/Pgas"][()]) - ulo,
                int(f.attrs["accepted"]), f["coords/z_m"][()] / 1e3)
data = [load(fn) for fn in states]
z = data[0][4]
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
cols = [S1, S2, S3]
for q in range(1, len(data)):
    lp0, lr0, _, a0, _ = data[q - 1]; lp1, lr1, fl1, a1, _ = data[q]
    dp = np.abs(lp1 - lp0).reshape(len(z), -1); dr = np.abs(lr1 - lr0).reshape(len(z), -1)
    lab = f"accepted {a0} → {a1}"; c = cols[(q - 1) % 3]
    axs[0].plot(dp.max(1), z, color=c, lw=2, label=f"{lab}: max"); axs[0].plot(np.percentile(dp, 99, 1), z, color=c, lw=1.2, ls="--", label=f"{lab}: p99")
    axs[1].plot(dr.max(1), z, color=c, lw=2, label=f"{lab}: max"); axs[1].plot(np.percentile(dr, 99, 1), z, color=c, lw=1.2, ls="--", label=f"{lab}: p99")
    floor = (fl1 < 1e-6).reshape(len(z), -1).mean(1) * 100
    axs[2].plot(floor, z, color=c, lw=2, label=f"accepted {a1}")
    k = np.argmax(dp.max(1))
    print(f"{lab}: max |Δ log10 P| {dp.max():.3f} dex at z = {z[k]:.0f} km; max |Δ log10 ρ| {dr.max():.3f} dex at z = {z[np.argmax(dr.max(1))]:.0f} km; "
          f"median |Δ log10 P| {np.median(dp):.4f} dex; nodes on the pressure floor at accepted {a1}: {int((fl1 < 1e-6).sum())} "
          f"({(fl1 < 1e-6).mean():.3%}), at z = {z[(fl1 < 1e-6).reshape(len(z), -1).any(1)].min() if (fl1 < 1e-6).any() else float('nan'):.0f} … "
          f"{z[(fl1 < 1e-6).reshape(len(z), -1).any(1)].max() if (fl1 < 1e-6).any() else float('nan'):.0f} km")
    lpz = lp1.reshape(len(z), -1); lp0z = lp0.reshape(len(z), -1)
    for zz in (0, 500, 1000, 2000):
        kk = np.argmin(np.abs(z - zz))
        print(f"   z = {z[kk]:.0f} km: median log10 P {np.median(lp0z[kk]):.3f} → {np.median(lpz[kk]):.3f}")
axs[0].set_xscale("log"); axs[1].set_xscale("log")
axs[0].set_xlabel("|Δ log10 P| [dex]"); axs[1].set_xlabel("|Δ log10 ρ| [dex]"); axs[2].set_xlabel("% of nodes on the pressure floor (ln P₀ − 3)")
axs[0].set_ylabel("height z [km]")
axs[0].set_title("Pressure change per height between saved states", loc="left", fontsize=10.5)
axs[1].set_title("Density change per height (ρ from the EOS at fixed T)", loc="left", fontsize=10.5)
axs[2].set_title("Pressure floor", loc="left", fontsize=10.5)
for a in axs:
    a.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(pre + "_by_height.png", dpi=130)
print("wrote", pre + "_per_iteration.png", pre + "_by_height.png")
