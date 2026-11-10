"""Schematic of the geometries of the method paper (Sect. 2): (a) a geometric height grid seen at heliocentric angle θ,
(b) optical-depth levels along inclined rays, (c) column-mass levels along the same rays. Schematic only: the surfaces
are analytic (a depression in a flux concentration plus a small ripple), not simulation data.

usage: python make_geometry_figure.py OUTDIR   → OUTDIR/geometry.pdf
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42})
W2 = 7.09
OUT = sys.argv[1]

MU = 0.707; T = np.tan(np.arccos(MU))           # slant t = tan θ (φ = 0)
X0, WT = 5.0, 0.9                               # flux concentration: centre and half-width (arbitrary units)
XL, XR, ZB, ZT = 0.0, 10.0, -1.0, 4.6


def surf(z0, dep, x):
    """A level surface: height z0, depressed by dep in the flux concentration, small ripple."""
    return z0 - dep * np.exp(-((x - X0) / WT) ** 2) + 0.04 * np.sin(2.1 * x + z0)


def node(z0, dep, xi):
    """Intersection of the ray x = ξ + t z with the surface (fixed-point iteration)."""
    z = z0
    for _ in range(50):
        z = surf(z0, dep, xi + T * z)
    return xi + T * z, z


def tube(a):
    xs = np.linspace(XL, XR, 400)
    w = WT * (1 + 0.18 * np.linspace(0, 1, 2))
    a.fill_betweenx([ZB, ZT], [X0 - w[0]] * 2, [X0 + w[0]] * 2, color="0.88", lw=0, zorder=0)
    a.text(X0, ZT - 0.08, "flux\nconcentration", ha="center", va="top", fontsize=6, color="0.3",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    return xs


def frame(a, title):
    a.set_xlim(XL, XR); a.set_ylim(ZB, ZT); a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=8)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.annotate("", xy=(XL + 0.02, ZT - 0.05), xytext=(XL + 0.02, ZB + 0.6), arrowprops=dict(arrowstyle="->", lw=0.6))
    a.text(XL + 0.15, ZT - 0.2, "$z$", fontsize=8)
    a.text(XR - 0.25, ZB + 0.08, "$x$", fontsize=8)


def observer(a, x, z, label=True):
    """Line-of-sight arrow toward the observer at angle θ from the vertical (toward +x)."""
    L = 1.0
    a.annotate("", xy=(x + T * L, z + L), xytext=(x, z), arrowprops=dict(arrowstyle="->", lw=0.9, color="C3"))
    a.plot([x, x], [z, z + 0.8], color="C3", lw=0.5, ls=":")
    if label:
        a.text(x + 0.12, z + 0.85, r"$\theta$", color="C3", fontsize=8)
        a.text(x + T * L + 0.1, z + L - 0.1, "to observer\n" + r"$\mu=\cos\theta$", color="C3", fontsize=6.5, va="top",
               ha="left")


fig, ax = plt.subplots(1, 3, figsize=(W2, 2.75), constrained_layout=True)
xis = np.arange(0.4, 9.0, 0.85)                 # footprints ξ (where the rays cross z = 0)

# (a) geometric grid
a = ax[0]; xs = tube(a)
for z in np.arange(-0.5, 4.5, 0.5):
    a.axhline(z, color="0.6", lw=0.4)
for x in np.arange(0.5, 10, 0.75):
    a.axvline(x, color="0.6", lw=0.4)
a.plot(xs, surf(0.0, 0.7, xs), color="C0", lw=1.2)
a.text(9.8, surf(0.0, 0.7, 9.8) + 0.08, r"$\tau_{500}=1$", color="C0", ha="right", va="bottom", fontsize=7)
for xi in (2.0, 3.6):
    zz = np.linspace(ZB, ZT, 2)
    a.plot(xi + T * zz, zz, color="C3", lw=0.8)
observer(a, 0.6, 2.6)
a.text(6.0, 2.9, "levels of constant $z$,\nvertical columns;\n" + r"$\tau$ along a ray needs" + "\ninterpolation across\ncolumns",
       ha="left", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
frame(a, "(a) geometric grid")

# (b) optical-depth levels on inclined rays
a = ax[1]; tube(a)
lv = [(0.0, 0.7, r"$\log\tau=0$"), (0.9, 0.55, ""), (1.8, 0.42, ""), (2.7, 0.32, r"$-4$ (cut)"), (3.6, 0.25, ""),
      (4.3, 0.2, r"$-6$")]
for z0, dep, lab in lv:
    cut = lab.endswith("(cut)")
    a.plot(xs, surf(z0, dep, xs), color="C0", lw=1.2 if cut else 0.7, ls="--" if cut else "-")
    if lab:
        a.text(9.95, surf(z0, dep, 9.95) + 0.06, lab, color="C0", ha="right", va="bottom", fontsize=6.5)
for xi in xis:
    zz = np.linspace(ZB, ZT, 2)
    a.plot(xi + T * zz, zz, color="C3", lw=0.5)
    for z0, dep, _ in lv:
        x, z = node(z0, dep, xi)
        a.plot(x, z, "o", ms=2.2, color="k" if z0 <= 2.7 else "0.55", zorder=3)
    a.plot(xi, 0, "|", color="C3", ms=4)
a.axhline(0, color="0.5", lw=0.4, ls=":")
a.text(0.35, -0.25, r"footprints $\xi$ at $z=0$", fontsize=6, color="C3", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
a.fill_between(xs, surf(2.7, 0.32, xs), ZT, color="C1", alpha=0.08, lw=0)
a.text(0.35, 3.35, "hydrostatic,\nrigid with $c_{ij}$", fontsize=6, color="C1", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
a.text(0.35, 1.95, r"unknowns: $\ln P$, $z$" + "\n" + r"$dz/d\ln\tau=-\mu\tau/\chi$", fontsize=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
a.text(7.0, 0.45, "lines of sight\n" + r"$x=\xi+t\,z$, $\mu=0.707$", fontsize=6.5, color="C3", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
frame(a, r"(b) optical depth $\tau_{500}$ along the rays")

# (c) column-mass levels on the same rays
a = ax[2]; tube(a)
lm = [(-0.6, 0.5, ""), (0.25, 0.45, ""), (1.1, 0.38, ""), (1.95, 0.3, ""), (2.75, 0.24, r"cut"), (3.6, 0.2, ""),
      (4.3, 0.17, "")]
for z0, dep, lab in lm:
    cut = lab == "cut"
    a.plot(xs, surf(z0, dep, xs), color="C2", lw=1.2 if cut else 0.7, ls="--" if cut else "-")
    if lab:
        a.text(9.95, surf(z0, dep, 9.95) + 0.06, lab, color="C2", ha="right", va="bottom", fontsize=6.5)
a.plot(xs, surf(0.0, 0.7, xs), color="C0", lw=1.0, ls=":")
a.text(6.6, 0.45, r"$\tau_{500}=1$ (dotted),"+"\ncomputed afterwards", color="C0", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
for xi in xis:
    zz = np.linspace(ZB, ZT, 2)
    a.plot(xi + T * zz, zz, color="C3", lw=0.5)
    for z0, dep, _ in lm:
        x, z = node(z0, dep, xi)
        a.plot(x, z, "o", ms=2.2, color="k" if z0 <= 2.75 else "0.55", zorder=3)
a.text(0.35, 1.95, r"levels of constant $\log m$" + "\n" + r"$dz/d\ln m=-\mu m/\rho$" + "\n" + r"hydrostatic: $P=\mu g m$",
       fontsize=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
frame(a, r"(c) column mass $m$ along the rays")

fig.savefig(f"{OUT}/geometry.pdf"); plt.close(fig)
print("wrote", f"{OUT}/geometry.pdf")
