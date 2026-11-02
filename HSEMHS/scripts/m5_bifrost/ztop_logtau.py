"""2x2 maps of log10 tau500 at the top cell (z index 0, top-first) of saved atmospheres, on a common colour scale.

usage: python3 ztop_logtau.py OUT.png LABEL1=FILE1.h5 LABEL2=FILE2.h5 ..."""
import sys
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

out, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
seq = LinearSegmentedColormap.from_list("seq", ["#fbe6dc", "#eb6834", "#5c1f08"])     # one hue, light -> dark
maps = []
for lab, fn in pairs:
    with h5py.File(fn, "r") as h:
        if "atmosphere" in h:
            m = np.log10(h["atmosphere/tau500"][0])                                # MHS layout: h5py (z, y, x), top-first
        else:
            m = np.log10(h["output/tau500"][:, :, 0]).T                            # column HSE: h5py (x, y, z), top-first
        acc = int(h.attrs["accepted"]) if "accepted" in h.attrs else None
    lab = f"{lab} (acc {acc})" if acc else lab
    print(f"{lab}: log10 tau500 at z_top min {m.min():.2f}, median {np.median(m):.2f}, max {m.max():.2f}; "
          f"columns > -5.7: {(m > -5.7).sum()}, > -5.1: {(m > -5.1).sum()}")
    maps.append((lab, m))
lo, hi = np.percentile(np.concatenate([m.ravel() for _, m in maps]), [0.5, 99.5])
nr = (len(maps) + 1) // 2
fig, axs = plt.subplots(nr, 2, figsize=(11, 5.2 * nr), squeeze=False, constrained_layout=True)
for a, (lab, m) in zip(axs.flat, maps):
    im = a.imshow(m, origin="lower", cmap=seq, vmin=lo, vmax=hi)
    a.set_title(f"log10 τ500 at z_top, {lab}\nmedian {np.median(m):.2f}, range {m.min():.2f} .. {m.max():.2f}", loc="left", fontsize=9.5)
    a.set_xlabel("x index"); a.set_ylabel("y index")
    fig.colorbar(im, ax=a, shrink=0.85, label="log10 τ500")
for a in axs.flat[len(maps):]:
    a.axis("off")
fig.savefig(out, dpi=110)
print("wrote", out)
