#!/usr/bin/env python3
"""T and Bz on constant log10 tau500 surfaces of a column_hse output (tau from /output/logtau500, T and B from the copied
/input fields; h5py order (x, y, z), z top first), as m5h_isotau_panel.py does for an m5h MHS atmosphere.
  OUT_panel.png : 2 x 2, T (top, log scale) and Bz (bottom, diverging) at L1, L2 on the column-HSE tau;
  OUT_cmp.png   : with MHS.h5, rows T@L1, T@L2, Bz@L1, Bz@L2; columns column HSE | MHS | MHS - column HSE.
usage: python3 m5h_colhse_isotau.py OUT_PREFIX COLHSE.h5 [MHS.h5] [L1=-5.1] [L2=-5.7]"""
import sys
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

args = sys.argv[1:]
pre, chf = args[0], args[1]
mhs = args[2] if len(args) > 2 and args[2].endswith(".h5") else None
rest = [float(v) for v in args[3 if mhs else 2:]]
levels = (rest + [-5.1, -5.7][len(rest):])[:2]

zyx = lambda d: np.transpose(d[()], (2, 1, 0))          # h5py (x, y, z) -> (z, y, x)
with h5py.File(chf, "r") as f:
    lt = zyx(f["output/logtau500"]); T = zyx(f["input/temperature"]); Bz = 1e4 * zyx(f["input/B_z"])
    zt1 = f["diagnostics/z_tau500_1"][()].T
assert np.all(np.diff(lt[:, 0, 0]) >= 0), "log10 tau must increase with k (top first)"
H = {L: (on_level(lt, T, L), on_level(lt, Bz, L)) for L in levels}
M = None
if mhs:
    with h5py.File(mhs, "r") as g:
        ltm = np.log10(g["atmosphere/tau500"][()]); Tm = g["atmosphere/T"][()]; Bm = 1e4 * g["atmosphere/Bz_centre"][()]
        acc = int(g.attrs["accepted"])
    assert np.allclose(Tm, T, rtol=1e-6), "T of the MHS file and the column-HSE input differ (orientation?)"
    M = {L: (on_level(ltm, Tm, L), on_level(ltm, Bm, L)) for L in levels}
for L in levels:
    t, b = H[L]
    print(f"column HSE, log10 tau500 = {L}: T min {np.nanmin(t):.4g} median {np.nanmedian(t):.4g} max {np.nanmax(t):.4g} K; "
          f"Bz {np.nanmin(b):.4g} .. {np.nanmax(b):.4g} G; columns without the level {int(np.isnan(t).sum())}")
print(f"column HSE z(tau500 = 1): min {np.nanmin(zt1)/1e3:.2f} median {np.nanmedian(zt1)/1e3:.2f} max {np.nanmax(zt1)/1e3:.2f} km")

INK, SURF = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "font.size": 10})
seq = LinearSegmentedColormap.from_list("seq", ["#fbe6dc", "#eb6834", "#5c1f08"])
div = LinearSegmentedColormap.from_list("div", ["#174a8b", "#2a78d6", "#e4e3df", "#eb6834", "#8a2e0b"])

def tnorm(*ms):
    lo, hi = np.nanpercentile(np.concatenate([m.ravel() for m in ms]), [0.5, 99.5]); return LogNorm(lo, hi)
def bnorm(*ms):
    lim = np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in ms])), 99.5); return TwoSlopeNorm(0.0, -lim, lim)
def show(ax, m, cmap, norm, title, unit):
    im = ax.imshow(m, origin="lower", cmap=cmap, norm=norm); ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel("x index"); ax.set_ylabel("y index"); plt.colorbar(im, ax=ax, shrink=0.85, label=unit)

fig, axs = plt.subplots(2, 2, figsize=(11, 9.6))
for c, L in enumerate(levels):
    t, b = H[L]
    show(axs[0, c], t, seq, tnorm(t), f"T at log10 τ500 = {L} (log scale)\nmedian {np.nanmedian(t):.0f} K", "K")
    show(axs[1, c], b, div, bnorm(b), f"Bz at log10 τ500 = {L}\nrange {np.nanmin(b):.0f} … {np.nanmax(b):.0f} G", "G")
fig.suptitle("Column HSE (bottom anchor), input T and B on the column-HSE τ500 (white: level above the column top)",
             x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(pre + "_panel.png", dpi=120); plt.close(fig)
print("wrote", pre + "_panel.png")

if M:
    fig, axs = plt.subplots(4, 3, figsize=(16, 19))
    for r, (L, q) in enumerate([(L, q) for q in (0, 1) for L in levels]):
        h, m = H[L][q], M[L][q]
        nm, unit = ("T", "K") if q == 0 else ("Bz", "G")
        nrm = tnorm(h, m) if q == 0 else bnorm(h, m)
        cmap = seq if q == 0 else div
        show(axs[r, 0], h, cmap, nrm, f"{nm} at log10 τ500 = {L}, column HSE\nmedian {np.nanmedian(h):.4g} {unit}", unit)
        show(axs[r, 1], m, cmap, nrm, f"{nm} at log10 τ500 = {L}, MHS accepted {acc}\nmedian {np.nanmedian(m):.4g} {unit}", unit)
        d = m - h; lim = np.nanpercentile(np.abs(d), 99) or 1.0
        show(axs[r, 2], d, div, TwoSlopeNorm(0.0, -lim, lim),
             f"MHS − column HSE [{unit}, ±p99 |Δ|]\nmedian Δ {np.nanmedian(d):.3g}, p99 |Δ| {lim:.3g}", unit)
        print(f"{nm} at {L}: MHS - column HSE median {np.nanmedian(d):.4g} {unit}, p99 |d| {lim:.4g}, max |d| {np.nanmax(np.abs(d)):.4g}")
    fig.tight_layout(); fig.savefig(pre + "_cmp.png", dpi=100); plt.close(fig)
    print("wrote", pre + "_cmp.png")
