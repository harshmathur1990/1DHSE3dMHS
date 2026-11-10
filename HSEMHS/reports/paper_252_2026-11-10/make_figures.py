"""Figures of the method paper (HSEMHS-Paper/images) from out/paper_252_2026-11-10 and IO/tau252.

usage: python make_figures.py RUNDIR IODIR OUTDIR
Writes OUTDIR/ztau1.pdf, profiles.pdf, force.pdf, convergence.pdf, viewing.pdf and prints the numbers quoted in the text.
Mask, gauge and references as metrics.py.
"""
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD, IO, OUT = sys.argv[1:4]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
W1, W2 = 3.46, 7.09                                          # A&A column and page width (in)
C = {"truth": "k", "col": "#7f7f7f", "mhs": "#1f77b4", "m3d": "#17becf", "cut6": "#ff7f0e", "cutnone": "#d62728",
     "mcol": "#bcbd22", "mmhs": "#9467bd", "pp": "#d62728"}


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


def lvl(a, fn=rms):
    return np.array([fn(a[k][np.isfinite(a[k])]) for k in range(a.shape[0])])


def masks(case):
    with h5py.File(f"{IO}/input_{case}.nc") as f:
        lt = f["log10tau500"][:]; k0 = int(np.argmin(abs(lt)))
        B = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
        x = f["x"][:] / 1e6
    return B, B > 0.05, B < 0.01, x


def truth(case):
    with h5py.File(f"{IO}/truth_{case}.h5") as f:
        z1 = f["truth/z_tau1_m"][:]; off = z1.mean()
        return dict(lt=f["logtau500"][:], u=f["truth/lnP"][:], r=f["truth/lnrho"][:], z=f["truth/z_m"][:] - off,
                    z1=z1 - off, off=off)


def run(name, grid="tau_grid"):
    with h5py.File(f"{RD}/{name}.h5") as f:
        g = f[grid]; z1 = g["z_tau1_m"][:]
        d = dict(u=np.log(g["Pgas"][:]), r=np.log(g["rho"][:]), z=g["z_m"][:] - z1.mean(), z1=z1 - z1.mean())
        if grid == "mass_grid":
            d["lt"] = np.median(g["logtau500"][:].reshape(g["logtau500"].shape[0], -1), axis=1)
        return d


def dep(z1, s, q):
    return (z1[s].mean() - z1[q].mean()) / 1e3


T1 = truth("eos_mu1.0"); B1, s1, q1, x = masks("eos_mu1.0")
col, mhs = run("colhse_eos_mu1.0"), run("mhs_eos_mu1.0")
mcol, mmhs = run("colhse_logm_mu1.0", "mass_grid"), run("mhs_logm_mu1.0", "mass_grid")
with h5py.File(f"{IO}/truth_logm_mu1.0.h5") as f:
    lm = f["log10m"][:]
    TM = dict(u=np.log(f["Pgas"][:]).transpose(2, 1, 0), r=np.log(f["rho"][:]).transpose(2, 1, 0),
              z=f["z_m"][:].transpose(2, 1, 0) - T1["off"])
Tm3 = truth("m3d_mu1.0"); m3c, m3m = run("colhse_m3d_mu1.0"), run("mhs_m3d_mu1.0")

# Fig. 1: z(τ=1) at μ = 1 (own τ): reference, column HSE, MHS on τ and on m; differences
fig, ax = plt.subplots(2, 4, figsize=(W2, 3.75), sharex=True, sharey=True, constrained_layout=True)
ext = [x[0], x[-1], x[0], x[-1]]
kw = dict(origin="lower", extent=ext, cmap="RdBu_r", interpolation="nearest", vmin=-100, vmax=100)
im = ax[0, 0].imshow(T1["z1"] / 1e3, **kw); ax[0, 0].set_title("Bifrost")
for i, (lab, d) in enumerate([(r"column HSE, $\tau$", col), (r"MHS, $\tau$", mhs), (r"MHS, $m$", mmhs)]):
    ax[0, i + 1].imshow(d["z1"] / 1e3, **kw); ax[0, i + 1].set_title(lab)
    ax[1, i + 1].imshow((d["z1"] - T1["z1"]) / 1e3, **kw)
    ax[1, i + 1].set_title(rf"$\Delta z$, rms {rms(d['z1'] - T1['z1']) / 1e3:.0f} km")
ax[1, 0].imshow(B1 * 1e4, origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1500, interpolation="nearest")
ax[1, 0].set_title(r"$|B|$ ($\log\tau_{500}=0$), 0–1.5 kG")
for a in ax.ravel()[1:]:
    if a is not ax[1, 0]:
        a.contour(x, x, s1, levels=[0.5], colors="k", linewidths=0.4)
for a in ax[1]:
    a.set_xlabel("x (Mm)")
for a in ax[:, 0]:
    a.set_ylabel("y (Mm)")
fig.colorbar(im, ax=ax[:, 3], shrink=0.8, label=r"$z(\tau_{500}=1)$, $\Delta z$ (km)")
fig.savefig(f"{OUT}/ztau1.pdf"); plt.close(fig)

# Fig. 2: rms errors per level (μ = 1): own τ, MULTI3D τ, column mass (at the median log τ500 of each m level)
fig, ax = plt.subplots(3, 1, figsize=(W1, 5.2), sharex=True, constrained_layout=True)
sets = [(col, T1, None, "col", "-", r"column HSE, own $\tau$"), (mhs, T1, None, "mhs", "-", r"MHS, own $\tau$"),
        (m3c, Tm3, None, "col", ":", r"column HSE, MULTI3D $\tau$"), (m3m, Tm3, None, "m3d", ":", r"MHS, MULTI3D $\tau$"),
        (mcol, TM, mcol["lt"], "mcol", "--", "column HSE, $m$"), (mmhs, TM, mmhs["lt"], "mmhs", "--", "MHS, $m$")]
for d, T, xl, c, ls, lab in sets:
    xs = T1["lt"] if xl is None else xl
    for i, q in enumerate(("u", "r", "z")):
        ax[i].plot(xs, lvl(d[q] - T[q]) / (1e3 if q == "z" else 1), color=C[c], ls=ls, label=lab)
ax[0].set_ylabel(r"rms $\Delta\ln P$"); ax[1].set_ylabel(r"rms $\Delta\ln\rho$"); ax[2].set_ylabel(r"rms $\Delta z$ (km)")
for a in ax:
    a.set_yscale("log"); a.axvline(-4, color="k", ls=":", lw=0.7); a.grid(alpha=0.3)
ax[2].set_xlabel(r"$\log\tau_{500}$"); ax[2].set_xlim(-8, 1.4); ax[0].legend(ncol=2, fontsize=6)
fig.savefig(f"{OUT}/profiles.pdf"); plt.close(fig)

# Fig. 3: horizontal force imbalance (μ = 1, own τ) and β
fig, ax = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True, constrained_layout=True, height_ratios=[2.2, 1])
ph = (T1["lt"] >= -3) & (T1["lt"] <= 0)
for fn, c, lab in [("diag_truth_eos_mu1.0", "truth", "Bifrost"), ("diag_colhse_eos_mu1.0", "col", "column HSE"),
                   ("diag_mhs_eos_mu1.0", "mhs", "cut $-4$"), ("diag_cut6_eos_mu1.0", "cut6", "cut $-6$"),
                   ("diag_cutnone_eos_mu1.0", "cutnone", "no cut")]:
    with h5py.File(f"{RD}/{fn}.h5") as g:
        fh = g["fh"][:]
        if c == "truth":
            beta = g["beta"][:]
    ax[0].plot(T1["lt"], lvl(fh, np.median), color=C[c], lw=1.6 if c == "truth" else 1.1, label=lab)
    print(f"{lab}: median |F_h|/rho g over log tau -3..0 = {np.median(fh[ph]):.3f}")
ax[1].plot(T1["lt"], lvl(beta, np.median), color="k", label=r"median $\beta$")
ax[1].plot(T1["lt"], lvl(beta, lambda v: np.mean(v < 1)), color="k", ls="--", label=r"fraction $\beta<1$")
for k in (-4, -5, -6):
    i = int(np.argmin(abs(T1["lt"] - k)))
    print(f"log tau {k}: median beta {np.median(beta[i]):.3g}, fraction beta<1 {np.mean(beta[i] < 1):.3f}")
ax[0].set_yscale("log"); ax[0].set_ylabel(r"median $|F_h|/\rho g$"); ax[0].legend(ncol=2); ax[0].grid(alpha=0.3)
ax[1].set_yscale("log"); ax[1].set_ylabel(r"Bifrost $\beta$"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_xlabel(r"$\log\tau_{500}$"); ax[1].set_xlim(-8, 1.4)
fig.savefig(f"{OUT}/force.pdf"); plt.close(fig)

# Fig. 4: convergence of the τ-grid solve (μ = 1, own τ)
h = np.genfromtxt(f"{RD}/ms_eos_mu1.0/history.csv", delimiter=",", names=True)
fig, ax = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, constrained_layout=True)
ax[0].semilogy(h["step"], h["phi_rel"], color="k", label=r"$\Phi/\Phi_0$")
ax[0].semilogy(h["step"], h["pred"], color="C0", label="predicted decrease")
ax[0].axhline(1e-3, color="C0", ls=":", lw=0.8); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].semilogy(h["step"], h["mean_dz1_km"], color="C1", label=r"mean $|\Delta z_1|$ per step")
ax[1].semilogy(h["step"], h["max_dz1_km"], color="C1", ls="--", label=r"max $|\Delta z_1|$ per step")
ax[1].axhline(6, color="C1", ls=":", lw=0.8)
ax[1].set_ylabel("km"); ax[1].set_xlabel("Gauss–Newton step"); ax[1].legend(); ax[1].grid(alpha=0.3)
a2 = ax[1].twinx(); a2.plot(h["step"], -h["depression_km"], color="C2"); a2.axhline(-dep(T1["z1"], s1, q1), color="C2", ls=":")
a2.set_ylabel("depression (km)", color="C2")
fig.savefig(f"{OUT}/convergence.pdf"); plt.close(fig)

# Fig. 5: viewing angle: z(τ=1) error and depression vs μ (inclined geometry; plane-parallel at μ = 0.5)
mus = [1.0, 0.707, 0.5]; cases = ["eos_mu1.0", "eos_mu0.707_az0", "eos_mu0.5_az0"]
res = {k: [] for k in ("Tdep", "cdep", "mdep", "cerr", "merr")}
for m, cs in zip(mus, cases):
    T = truth(cs); _, s, q, _ = masks(cs)
    c = run(f"colhse_{cs}"); d = run(f"mhs_{cs}")
    res["Tdep"].append(dep(T["z1"], s, q)); res["cdep"].append(dep(c["z1"], s, q)); res["mdep"].append(dep(d["z1"], s, q))
    res["cerr"].append(rms(c["z1"] - T["z1"]) / 1e3); res["merr"].append(rms(d["z1"] - T["z1"]) / 1e3)
T5 = truth("eos_mu0.5_az0"); _, s5, q5, _ = masks("eos_mu0.5_az0"); pp = run("mhspp_eos_mu0.5_az0")
fig, ax = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, constrained_layout=True)
ax[0].plot(mus, res["Tdep"], "k-o", ms=3, label="Bifrost")
ax[0].plot(mus, res["cdep"], "-o", color=C["col"], ms=3, label="column HSE")
ax[0].plot(mus, res["mdep"], "-o", color=C["mhs"], ms=3, label="MHS, inclined")
ax[0].plot([0.5], [dep(pp["z1"], s5, q5)], "x", color=C["pp"], label="MHS, plane-parallel")
ax[0].set_ylabel("depression (km)"); ax[0].legend(fontsize=6); ax[0].grid(alpha=0.3)
ax[1].plot(mus, res["cerr"], "-o", color=C["col"], ms=3); ax[1].plot(mus, res["merr"], "-o", color=C["mhs"], ms=3)
ax[1].plot([0.5], [rms(pp["z1"] - T5["z1"]) / 1e3], "x", color=C["pp"])
ax[1].set_ylabel(r"rms error of $z(\tau_{500}=1)$ (km)"); ax[1].set_xlabel(r"$\mu$"); ax[1].grid(alpha=0.3)
ax[1].invert_xaxis()
fig.savefig(f"{OUT}/viewing.pdf"); plt.close(fig)
print("viewing:", {k: [round(v, 1) for v in vals] for k, vals in res.items()})
print("figures written to", OUT)
