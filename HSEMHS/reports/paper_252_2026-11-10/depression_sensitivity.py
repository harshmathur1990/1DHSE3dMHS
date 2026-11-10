"""Sensitivity of the strong-minus-quiet z(τ=1) contrast ("depression") to the mask, and its spatial-block uncertainty.

usage: python depression_sensitivity.py RUNDIR IODIR
Writes RUNDIR/depression_sensitivity.txt: per μ, the number of strong/quiet columns and the contrast of the truth, column
HSE and MHS for strong thresholds 300, 500, 1000 G (quiet < 100 G), and for 500 G the standard deviation over a
block bootstrap (the 252² map cut into 6 × 6 blocks of 42², 2000 resamplings of blocks with replacement) and the
recovered fraction with its bootstrap interval (16th–84th percentile).
"""
import sys

import h5py
import numpy as np

RD, IO = sys.argv[1:3]
out = open(f"{RD}/depression_sensitivity.txt", "w")


def p(*a):
    print(*a); print(*a, file=out)


def z1(fn, grp):
    with h5py.File(fn) as f:
        a = f[grp][:]
    return a - a.mean()


rng = np.random.default_rng(7)
for case in ("eos_mu1.0", "eos_mu0.707_az0", "eos_mu0.5_az0"):
    with h5py.File(f"{IO}/input_{case}.nc") as f:
        lt = f["log10tau500"][:]; k0 = int(np.argmin(abs(lt)))
        B = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
    zs = {"truth": z1(f"{IO}/truth_{case}.h5", "truth/z_tau1_m"), "colhse": z1(f"{RD}/colhse_{case}.h5", "tau_grid/z_tau1_m"),
          "mhs": z1(f"{RD}/mhs_{case}.h5", "tau_grid/z_tau1_m")}
    quiet = B < 0.01
    p(f"== {case}: quiet (<100 G) {quiet.sum()} columns")
    for thr in (0.03, 0.05, 0.10):
        s = B > thr
        d = {k: (v[s].mean() - v[quiet].mean()) / 1e3 for k, v in zs.items()}
        p(f"  strong > {thr * 1e4:4.0f} G: {s.sum():4d} columns | truth {d['truth']:+6.1f}  colhse {d['colhse']:+6.1f} "
          f"({d['colhse'] / d['truth'] * 100:4.0f} %)  mhs {d['mhs']:+6.1f} ({d['mhs'] / d['truth'] * 100:4.0f} %)")
    s = B > 0.05; nb = 6; L = B.shape[0] // nb
    blocks = [(i, j) for i in range(nb) for j in range(nb)]
    boot = {k: [] for k in zs}; frac = {"colhse": [], "mhs": []}
    for _ in range(2000):
        pick = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        idx = np.concatenate([np.ravel_multi_index(np.meshgrid(np.arange(i * L, (i + 1) * L), np.arange(j * L, (j + 1) * L),
                                                               indexing="ij"), B.shape).ravel() for i, j in pick])
        ss = s.ravel()[idx]; qq = quiet.ravel()[idx]
        if ss.sum() == 0:
            continue
        dd = {k: (v.ravel()[idx][ss].mean() - v.ravel()[idx][qq].mean()) / 1e3 for k, v in zs.items()}
        for k in zs:
            boot[k].append(dd[k])
        for k in frac:
            frac[k].append(dd[k] / dd["truth"])
    p("  500 G block bootstrap std (km): " + ", ".join(f"{k} {np.std(v):.1f}" for k, v in boot.items()) +
      " | recovered fraction 16–84 %: " + ", ".join(f"{k} {np.percentile(v, 16) * 100:.0f}–{np.percentile(v, 84) * 100:.0f} %"
                                                   for k, v in frac.items()))
