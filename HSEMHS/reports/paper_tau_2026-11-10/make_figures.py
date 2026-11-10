"""Figures of the method paper (HSEMHS-Paper/images) from the runs of this report.

usage: python make_figures.py TAUDIR LOGMDIR LOGMCUTDIR OUTDIR
  TAUDIR: out/paper_tau_2026-11-10 (truth.h5, input-ltau-xy.nc, colhse/mhs/cut5/cut6/cutnone .h5, diag_<run>.h5, ms100/history.csv)
  LOGMDIR: out/logm_2026-11-02 (truth_logm.h5, column-hse-logm.h5); LOGMCUTDIR: out/logm_cut_2026-11-03 (cut_-0.5.h5)
Writes OUTDIR/ztau1.pdf, profiles.pdf, force.pdf, convergence.pdf. Mask and gauge as paper_metrics.py.
"""
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TD, LD, LCD, OUT = sys.argv[1:5]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
W1, W2 = 3.46, 7.09                                          # A&A column and page width (in)
C = {"truth": "k", "colhse": "#7f7f7f", "mhs": "#1f77b4", "cut5": "#2ca02c", "cut6": "#ff7f0e", "cutnone": "#d62728",
     "mcol": "#bcbd22", "mmhs": "#9467bd"}

with h5py.File(f"{TD}/truth.h5") as f:
    lt = f["logtau500"][:]
    ut = f["truth/lnP"][:]; rt = f["truth/lnrho"][:]; zt = f["truth/z_m"][:]; z1raw = f["truth/z_tau1_m"][:]
z1off = z1raw.mean(); z1t = z1raw - z1off; zt = zt - z1off
k0 = int(np.argmin(abs(lt)))
with h5py.File(f"{TD}/input-ltau-xy.nc") as f:
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
    x = f["x"][:] / 1e6
strong = B1 > 0.05


def tau_run(r):
    with h5py.File(f"{TD}/{r}.h5") as f:
        g = f["tau_grid"]; z1 = g["z_tau1_m"][:]
        return dict(u=np.log(g["Pgas"][:]), r=np.log(g["rho"][:]), z=g["z_m"][:] - z1.mean(), z1=z1 - z1.mean())


with h5py.File(f"{LD}/truth_logm.h5") as f:
    lm = f["log10m"][:]
    um_t = np.log(f["Pgas"][:]).transpose(2, 1, 0); rm_t = np.log(f["rho"][:]).transpose(2, 1, 0)
    zm_t = f["z_m"][:].transpose(2, 1, 0) - z1off


def m_run(fn):
    with h5py.File(fn) as f:
        g = f["mass_grid"]; z1 = g["z_tau1_m"][:]
        return dict(u=np.log(g["Pgas"][:]), r=np.log(g["rho"][:]), z=g["z_m"][:] - z1.mean(), z1=z1 - z1.mean(),
                    lt=np.median(g["logtau500"][:].reshape(len(lm), -1), axis=1))


R = {r: tau_run(r) for r in ("colhse", "mhs")}
M = {"mcol": m_run(f"{LD}/column-hse-logm.h5"), "mmhs": m_run(f"{LCD}/cut_-0.5.h5")}
LAB = {"colhse": r"column HSE, $\tau$", "mhs": r"MHS, $\tau$", "mcol": "column HSE, $m$", "mmhs": "MHS, $m$"}

# Fig. 1: z(τ=1) maps and differences
fig, ax = plt.subplots(2, 4, figsize=(W2, 3.75), sharex=True, sharey=True, constrained_layout=True)
ext = [x[0], x[-1], x[0], x[-1]]
kw = dict(origin="lower", extent=ext, cmap="RdBu_r", interpolation="nearest")
v = 100
im = ax[0, 0].imshow(z1t / 1e3, vmin=-v, vmax=v, **kw); ax[0, 0].set_title("Bifrost")
for i, (k, d) in enumerate([("colhse", R["colhse"]), ("mhs", R["mhs"]), ("mmhs", M["mmhs"])]):
    ax[0, i + 1].imshow(d["z1"] / 1e3, vmin=-v, vmax=v, **kw); ax[0, i + 1].set_title(LAB[k])
    im2 = ax[1, i + 1].imshow((d["z1"] - z1t) / 1e3, vmin=-v, vmax=v, **kw)
    e = np.sqrt(np.mean((d["z1"] - z1t) ** 2)) / 1e3
    ax[1, i + 1].set_title(f"$\\Delta z$, rms {e:.0f} km")
ax[1, 0].imshow(B1 * 1e4, origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1500, interpolation="nearest")
ax[1, 0].set_title(r"$|B|$ ($\log\tau_{500}=0$), 0–1.5 kG")
for a in ax.ravel():
    a.contour(x, x, strong, levels=[0.5], colors="k", linewidths=0.4) if a is not ax[1, 0] else None
for a in ax[1]:
    a.set_xlabel("x (Mm)")
for a in ax[:, 0]:
    a.set_ylabel("y (Mm)")
fig.colorbar(im, ax=ax[:, 3], shrink=0.8, label=r"$z(\tau_{500}=1)$, $\Delta z$ (km)")
fig.savefig(f"{OUT}/ztau1.pdf"); plt.close(fig)


def lvl(a, fn=lambda v: np.sqrt(np.mean(v ** 2))):
    return np.array([fn(a[k][np.isfinite(a[k])]) for k in range(a.shape[0])])


# Fig. 2: rms errors of ln P, ln ρ, z per level (mass grid plotted at the median log τ500 of each m level)
fig, ax = plt.subplots(3, 1, figsize=(W1, 5.2), sharex=True, constrained_layout=True)
for k, d in R.items():
    for i, (q, t) in enumerate((("u", ut), ("r", rt), ("z", zt))):
        ax[i].plot(lt, lvl(d[q] - t) / (1e3 if q == "z" else 1), color=C[k], label=LAB[k])
for k, d in M.items():
    for i, (q, t) in enumerate((("u", um_t), ("r", rm_t), ("z", zm_t))):
        ax[i].plot(d["lt"], lvl(d[q] - t) / (1e3 if q == "z" else 1), color=C[k], ls="--", label=LAB[k])
ax[0].set_ylabel(r"rms $\Delta\ln P$"); ax[1].set_ylabel(r"rms $\Delta\ln\rho$"); ax[2].set_ylabel(r"rms $\Delta z$ (km)")
for a in ax:
    a.set_yscale("log"); a.axvline(-4, color="k", ls=":", lw=0.7); a.grid(alpha=0.3)
ax[2].set_xlabel(r"$\log\tau_{500}$"); ax[2].set_xlim(-8, 1.4); ax[0].legend(ncol=2)
fig.savefig(f"{OUT}/profiles.pdf"); plt.close(fig)

# Fig. 3: horizontal force imbalance per level, and the truth β
fig, ax = plt.subplots(2, 1, figsize=(W1, 3.9), sharex=True, constrained_layout=True, height_ratios=[2.2, 1])
LC = {"truth": "Bifrost", "colhse": "column HSE", "mhs": "cut $-4$", "cut5": "cut $-5$", "cut6": "cut $-6$", "cutnone": "no cut"}
for r in ("truth", "colhse", "mhs", "cut5", "cut6", "cutnone"):
    try:
        with h5py.File(f"{TD}/diag_{r}.h5") as g:
            fh = g["fh"][:]; beta = g["beta"][:] if r == "truth" else None
    except OSError:
        continue
    ax[0].plot(lt, lvl(fh, np.median), color=C[r], lw=1.6 if r == "truth" else 1.1, label=LC[r])
    if beta is not None:
        ax[1].plot(lt, lvl(beta, np.median), color="k", label=r"median $\beta$")
        ax[1].plot(lt, lvl(beta, lambda v: np.mean(v < 1)), color="k", ls="--", label=r"fraction $\beta<1$")
ax[0].set_yscale("log"); ax[0].set_ylabel(r"median $|F_h|/\rho g$"); ax[0].legend(ncol=2); ax[0].grid(alpha=0.3)
ax[1].set_yscale("log"); ax[1].set_ylabel(r"Bifrost $\beta$"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_xlabel(r"$\log\tau_{500}$"); ax[1].set_xlim(-8, 1.4)
fig.savefig(f"{OUT}/force.pdf"); plt.close(fig)

# Fig. 4: convergence of the τ-grid solve (release settings, 100-step budget)
h = np.genfromtxt(f"{TD}/ms100/history.csv", delimiter=",", names=True)
fig, ax = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, constrained_layout=True)
ax[0].semilogy(h["step"], h["phi_rel"], color="k", label=r"$\Phi/\Phi_0$")
ax[0].semilogy(h["step"], h["pred"], color="C0", label="predicted decrease")
ax[0].axhline(1e-3, color="C0", ls=":", lw=0.8)
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].semilogy(h["step"], h["mean_dz1_km"], color="C1", label=r"mean $|\Delta z_1|$ per step")
ax[1].semilogy(h["step"], h["err_z1_km"], color="C2", label=r"rms error of $z_1$")
ax[1].axhline(6, color="C1", ls=":", lw=0.8)
ax[1].axvline(25, color="k", ls="--", lw=0.7)
ax[1].set_ylabel("km"); ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.savefig(f"{OUT}/convergence.pdf"); plt.close(fig)
print("figures written to", OUT)
