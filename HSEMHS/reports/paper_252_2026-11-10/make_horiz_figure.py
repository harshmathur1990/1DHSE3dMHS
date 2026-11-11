"""Schematic of the horizontal residual (Eq. 4 of the method paper, Fig. 2): a τ surface depressed in a strong-field
region; between two points A and B on it, the change of ln P along the surface equals that at constant height (A→C)
plus the hydrostatic change down the vertical leg (C→B, weight W). Gas pressure and Lorentz force at constant height
balance where R_x = 0. Schematic only, not simulation data.

usage: python make_horiz_figure.py OUTDIR   → OUTDIR/horiz.pdf
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
W1 = 3.46
OUT = sys.argv[1]

X0, WT = 5.0, 1.6                                # depression: centre and half-width
XL, XR, ZB, ZT = 0.0, 10.0, -1.9, 2.3
Z0, DEP = 0.7, 1.4                               # the τ surface of the triangle


def surf(z0, dep, x):
    return z0 - dep * np.exp(-((x - X0) / WT) ** 2)


fig, a = plt.subplots(figsize=(W1, 2.3), constrained_layout=True)
xs = np.linspace(XL, XR, 300)
a.axvspan(X0 - 1.0, X0 + 1.0, color="C0", alpha=0.08, lw=0)
a.text(X0, ZT - 0.12, "strong field", ha="center", va="top", fontsize=7, color="C0")
for z0, dep in ((1.8, 1.0), (-0.6, 1.2)):
    a.plot(xs, surf(z0, dep, xs), color="0.7", lw=0.7)
a.plot(xs, surf(Z0, DEP, xs), color="k", lw=1.0)
a.text(XR - 0.1, Z0 + 0.06, r"$\tau$ surface", fontsize=7, ha="right", va="bottom")

# A and B on the τ surface, C at the height of A above B
xa, xb = 2.2, 4.0
za, zb = surf(Z0, DEP, xa), surf(Z0, DEP, xb)
seg = np.linspace(xa, xb, 50)
a.plot(seg, surf(Z0, DEP, seg), color="k", lw=2.2)
arr = lambda c: dict(arrowstyle="->", lw=1.2, color=c, shrinkA=3, shrinkB=3)
a.annotate("", xy=(xb, za), xytext=(xa, za), arrowprops=arr("C3"))
a.annotate("", xy=(xb, zb), xytext=(xb, za), arrowprops=arr("C2"))
for x, z, s, dx, dz in ((xa, za, "A", -0.12, 0.0), (xb, za, "C", 0.12, 0.06), (xb, zb, "B", 0.12, -0.06)):
    a.plot(x, z, "o", ms=3, color="k", zorder=3)
    a.text(x + dx, z + dz, s, ha="right" if dx < 0 else "left", va="center", fontsize=8)
a.text((xa + xb) / 2, za + 0.1, r"$\partial_x\ln P|_z$", color="C3", ha="center", va="bottom", fontsize=7)
a.text(xb + 0.15, (za + zb) / 2 + 0.05, r"$\Delta\ln P=-W\Delta z$", color="C2", ha="left", va="center", fontsize=7)
a.text(2.85, 0.28, r"$\partial_x\ln P|_\tau$", ha="right", va="top", fontsize=7)

# forces at constant height on the far flank
xf, zf = 7.0, -1.45
a.annotate("", xy=(xf - 0.9, zf), xytext=(xf, zf), arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.2))
a.annotate("", xy=(xf + 0.9, zf), xytext=(xf, zf), arrowprops=dict(arrowstyle="-|>", color="C0", lw=1.2))
a.plot(xf, zf, "o", ms=3, color="k", zorder=3)
a.text(xf - 0.45, zf + 0.1, r"$-\partial_xP|_z$", color="C1", ha="center", va="bottom", fontsize=7)
a.text(xf + 0.45, zf + 0.1, r"$L_x$", color="C0", ha="center", va="bottom", fontsize=7)

a.set_xlim(XL, XR); a.set_ylim(ZB, ZT); a.set_xticks([]); a.set_yticks([])
a.set_xlabel("$x$"); a.set_ylabel("$z$")
fig.savefig(f"{OUT}/horiz.pdf"); plt.close(fig)
print("wrote", f"{OUT}/horiz.pdf")
