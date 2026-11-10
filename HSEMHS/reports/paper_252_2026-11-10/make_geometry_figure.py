"""Schematic of the geometries of the method paper (Fig. 1): (a) a geometric height grid seen at heliocentric angle θ,
(b) optical-depth levels along inclined rays, (c) column-mass levels along the same rays. Schematic only (analytic
surfaces with a depression), not simulation data.

usage: python make_geometry_figure.py OUTDIR   → OUTDIR/geometry.pdf
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
W2 = 7.09
OUT = sys.argv[1]

T = 1.0                                          # slant t = tan θ (θ = 45°, μ = 0.707)
X0, WT = 5.0, 0.9                                # depression: centre and half-width
XL, XR, ZB, ZT = 0.0, 10.0, -0.8, 4.4
XIS = np.arange(0.5, 9.0, 1.2)                   # footprints ξ at z = 0
PIX = 2                                          # the highlighted pixel


def surf(z0, dep, x):
    return z0 - dep * np.exp(-((x - X0) / WT) ** 2)


def node(z0, dep, xi):
    z = z0
    for _ in range(50):
        z = surf(z0, dep, xi + T * z)
    return xi + T * z, z


def frame(a, title):
    a.set_xlim(XL, XR); a.set_ylim(ZB, ZT); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("$x$"); a.set_ylabel("$z$"); a.set_title(title, fontsize=8)


def rays(a, levels, color):
    xs = np.linspace(XL, XR, 300)
    for z0, dep, cut in levels:
        a.plot(xs, surf(z0, dep, xs), color=color, lw=1.2 if cut else 0.8, ls="--" if cut else "-")
    for n, xi in enumerate(XIS):
        hi = n == PIX
        a.plot(xi + T * np.array([ZB, ZT]), [ZB, ZT], color="C3", lw=1.5 if hi else 0.5)
        for z0, dep, _ in levels:
            x, z = node(z0, dep, xi)
            a.plot(x, z, "o", ms=3.5 if hi else 2, color="C3" if hi else "k", zorder=3)
    x, z = node(levels[1][0], levels[1][1], XIS[PIX])
    a.text(x - 0.3, z, "pixel $(i,j)$", color="C3", ha="right", va="center", fontsize=7,
           bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))
    cz = [l for l in levels if l[2]][0]
    a.fill_between(xs, surf(cz[0], cz[1], xs), ZT, color="0.85", alpha=0.5, lw=0)
    a.text(XR - 0.15, ZT - 0.15, "hydrostatic", ha="right", va="top", fontsize=7, color="0.35")
    a.text(XR - 0.15, surf(cz[0], cz[1], XR) - 0.1, "cut", ha="right", va="top", fontsize=7, color=color)


fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), constrained_layout=True)

# (a) geometric grid, line of sight at θ
a = ax[0]; xs = np.linspace(XL, XR, 300)
for z in np.arange(-0.5, 4.4, 0.6):
    a.axhline(z, color="0.75", lw=0.5)
for x in np.arange(0.5, 10, 0.8):
    a.axvline(x, color="0.75", lw=0.5)
a.plot(xs, surf(0.0, 0.6, xs), color="C0", lw=1.2)
a.text(XR - 0.15, 0.1, r"$\tau_{500}=1$", color="C0", ha="right", va="bottom", fontsize=7)
a.annotate("", xy=(2.0 + T * 4.3, 4.3), xytext=(2.0 + T * ZB, ZB), arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
a.plot([4.9, 4.9], [2.9, 4.1], color="C3", lw=0.6, ls=":")
a.text(5.05, 4.05, r"$\theta$", color="C3", fontsize=8, va="top")
a.text(6.6, 3.9, "line of sight", color="C3", fontsize=7, va="top")
frame(a, "(a) geometric height $z$")

# (b) optical depth, (c) column mass, along the rays
rays(ax[1], [(0.0, 0.6, False), (1.0, 0.45, False), (2.0, 0.35, False), (2.9, 0.28, True), (3.8, 0.22, False)], "C0")
frame(ax[1], r"(b) optical depth $\log\tau_{500}$")
rays(ax[2], [(-0.5, 0.45, False), (0.5, 0.38, False), (1.5, 0.3, False), (2.9, 0.22, True), (3.8, 0.18, False)], "C2")
ax[2].plot(xs, surf(0.0, 0.6, xs), color="C0", lw=1.0, ls=":")
ax[2].text(XR - 0.15, -0.15, r"$\tau_{500}=1$", color="C0", ha="right", va="top", fontsize=7)
frame(ax[2], r"(c) column mass $\log m$")

fig.savefig(f"{OUT}/geometry.pdf"); plt.close(fig)
print("wrote", f"{OUT}/geometry.pdf")
