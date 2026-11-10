"""Why column HSE on column mass has a larger z(τ=1) error than on τ (round-2 review Q2). No new runs: the stored
column-HSE states are re-aligned offline.

usage: python alignment_depth.py RUNDIR IODIR   (output kept in alignment_depth.txt)
The production alignment (PoissonTau.align_ptot!) shifts each column so that P + B²/2μ₀ equals the median total pressure
of the bottom level of the axis, interpolated in the column, or extrapolated hydrostatically below the bottom level
(z = z_bottom − Δln P_tot/(g ρ/P)) for columns whose bottom total pressure is lower. The same policy is applied here with
the reference total pressure taken at other levels (median over columns), so every column is kept at every depth; the
bottom-level row reproduces the stored state. Then gauge mean z(τ=1) = 0 and compare with the truth (rms error, corr).
"""
import sys

import h5py
import numpy as np

RD, IO = sys.argv[1:3]
MU0 = 4e-7 * np.pi; G = 274.0
rms = lambda a: np.sqrt(np.nanmean(a ** 2))
with h5py.File(f"{IO}/truth_eos_mu1.0.h5") as f:
    z1T = f["truth/z_tau1_m"][:]; zT = f["truth/z_m"][:]
off = z1T.mean(); z1T = z1T - off
with h5py.File(f"{IO}/truth_logm_mu1.0.h5") as t:
    zM = t["z_m"][:].transpose(2, 1, 0) - off


def load(run, grid):
    with h5py.File(f"{RD}/{run}.h5") as f:
        g = f[grid]; z = g["z_m"][:]; P = g["Pgas"][:]; rho = g["rho"][:]
        B2 = g["Bx"][:] ** 2 + g["By"][:] ** 2 + g["Bz"][:] ** 2
        ax = g["log10m"][:] if grid == "mass_grid" else g["logtau500"][:]
        lt = g["logtau500"][:] if grid == "mass_grid" else None
    return z, np.log(P + B2 / (2 * MU0)), rho / P, lt, ax


def z1of(z, lt, ax):
    nt, ny, nx = z.shape; out = np.empty((ny, nx))
    for j in range(ny):
        for i in range(nx):
            out[j, i] = np.interp(0.0, lt[:, j, i] if lt is not None else ax, z[:, j, i])
    return out


def zref(z, lpt, q, ref):
    nt, ny, nx = z.shape; zr = np.empty((ny, nx)); nex = 0
    for j in range(ny):
        for i in range(nx):
            v = lpt[:, j, i]
            if v[-1] < ref:                                   # below the column: hydrostatic extrapolation
                zr[j, i] = z[-1, j, i] - (ref - v[-1]) / (G * q[-1, j, i]); nex += 1
            else:
                zr[j, i] = np.interp(ref, v, z[:, j, i])
    return zr, nex


for run, grid, levs in (("colhse_eos_mu1.0", "tau_grid", (1.4, 1.0, 0.5, 0.0, -1.0)),
                        ("colhse_logm_mu1.0", "mass_grid", (2.6, 2.2, 1.9, 1.7, 1.4, 1.0))):
    z, lpt, q, lt, ax = load(run, grid)
    z1 = z1of(z, lt, ax)
    print(f"== {run} (axis bottom {ax[-1]:+.2f})")
    for v in levs:
        k = int(np.argmin(abs(ax - v))); ref = np.median(lpt[k])
        zr, nex = zref(z, lpt, q, ref)
        zz = z1 - zr; zz -= zz.mean()
        zl = z[k] - zr; zl -= zl.mean()
        tl = zM[k] if grid == "mass_grid" else zT[k] - off
        print(f"  reference P_tot = median at level {ax[k]:+.2f} (truth median z {np.median(tl) / 1e3:+6.0f} km): "
              f"z(τ=1) rms {rms(zz - z1T) / 1e3:5.1f} km, corr {np.corrcoef(zz.ravel(), z1T.ravel())[0, 1]:.3f}; "
              f"extrapolated columns {nex}")
