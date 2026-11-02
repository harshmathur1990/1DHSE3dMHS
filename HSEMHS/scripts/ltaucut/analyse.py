"""Compare mhs-poisson-ltau runs with different log10 τ cuts against the Bifrost truth on the τ grid.

usage: python analyse.py DIR      (DIR holds truth.h5, input-ltau-xy.nc, <run>.h5, <run>.log, diag_<run>.h5)
writes DIR/summary.txt, DIR/profiles.png, DIR/ztau1.png, DIR/err_vs_beta.png
"""
import re
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
RUNS = ["colhse", "cut4", "cut5", "cut6", "cutnone"]
LABEL = {"colhse": "column HSE", "cut4": "cut −4", "cut5": "cut −5", "cut6": "cut −6", "cutnone": "no cut (−7.9)",
         "truth": "Bifrost"}
COL = {"colhse": "#7f7f7f", "cut4": "#1f77b4", "cut5": "#2ca02c", "cut6": "#ff7f0e", "cutnone": "#d62728", "truth": "k"}

with h5py.File(f"{D}/truth.h5") as f:
    lt = f["logtau500"][:]
    zt = f["truth/z_m"][:]; ut = f["truth/lnP"][:]; z1t = f["truth/z_tau1_m"][:]
with h5py.File(f"{D}/input-ltau-xy.nc") as f:            # (x, y, τ) → (τ, y, x)
    B = np.sqrt(sum(f[k][:] ** 2 for k in ("B_x", "B_y", "B_z"))).transpose(2, 1, 0)
k0 = int(np.argmin(abs(lt)))
B1 = B[k0]
strong = B1 > 0.05          # > 500 G at log τ = 0
quiet = B1 < 0.01           # < 100 G
z1t = z1t - z1t.mean()      # same gauge as the runs: mean z(τ=1) = 0
zt = zt - np.nanmean(zt[k0])


def gauge(z):
    return z - z[k0].mean()


def lvl(a, fn):
    return np.array([fn(a[k][np.isfinite(a[k])]) if np.isfinite(a[k]).any() else np.nan for k in range(len(lt))])


def rms(a):
    return np.sqrt(np.mean(a ** 2))


def logrun(name):
    try:
        s = open(f"{D}/{name}.log").read()
    except OSError:
        return {}
    m = re.search(r"WALL ([\d.]+) s", s)
    return {"wall_s": float(m.group(1)) if m else np.nan}


out = open(f"{D}/summary.txt", "w")


def p(*a):
    print(*a); print(*a, file=out)


p(f"strong-field columns (|B|>500 G at τ=1): {strong.mean()*100:.1f} %, quiet (<100 G): {quiet.mean()*100:.1f} %")
p(f"truth Wilson depression  z(τ=1) strong − quiet: {z1t[strong].mean()/1e3 - z1t[quiet].mean()/1e3:+.1f} km")

res = {}
with h5py.File(f"{D}/colhse.h5") as f:
    uc = np.log(f["tau_grid/Pgas"][:])
for r in RUNS + ["truth"]:
    if r == "truth":
        u, z, z1, a = ut, zt, z1t, {}
    else:
        try:
            f = h5py.File(f"{D}/{r}.h5")
        except OSError:
            continue
        u = np.log(f["tau_grid/Pgas"][:]); z = gauge(f["tau_grid/z_m"][:]); z1 = f["tau_grid/z_tau1_m"][:]
        z1 = z1 - z1.mean(); a = dict(f.attrs); f.close()
    d = {"u": u, "z": z, "z1": z1, "attrs": a}
    try:
        with h5py.File(f"{D}/diag_{r}.h5") as g:
            d.update(fh=g["fh"][:], fz=g["fz"][:], lor=g["lor"][:], beta=g["beta"][:])
    except OSError:
        pass
    res[r] = d

p("\nrun           steps stop        Φ/Φ0     GN-dec   wall   | z(τ=1) err rms  corr  Wilson(str−q) | ΔlnP vs truth rms: below cut / all levels")
for r in RUNS + ["truth"]:
    if r not in res:
        continue
    d = res[r]; a = d["attrs"]
    e1 = d["z1"] - z1t
    wil = d["z1"][strong].mean() / 1e3 - d["z1"][quiet].mean() / 1e3
    kc = int(np.argmin(abs(lt - a["logtau_cut"]))) if "logtau_cut" in a else 0
    du = d["u"] - ut
    below = rms(du[kc:][np.isfinite(du[kc:])])
    allv = rms(du[np.isfinite(du)])
    steps = a.get("steps", "-"); stop = a.get("stop_reason", "-")
    ph = a["phi_end"] / a["phi_start"] if "phi_end" in a else np.nan
    p(f"{LABEL[r]:13s} {str(steps):>5s} {str(stop):11s} {ph:8.2e} {a.get('gn_decrement', np.nan):8.1e} "
      f"{logrun(r).get('wall_s', np.nan):6.0f} | {rms(e1)/1e3:8.1f} km  {np.corrcoef(d['z1'].ravel(), z1t.ravel())[0,1]:5.3f} "
      f"{wil:+8.1f} km  | {below:.3f} / {allv:.3f}")

# per-level profiles
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
ax = ax.ravel()
for r in RUNS + ["truth"]:
    if r not in res:
        continue
    d = res[r]; kw = dict(color=COL[r], label=LABEL[r], lw=1.6 if r != "truth" else 2.2)
    if r != "truth":
        ax[0].plot(lt, lvl(d["u"] - ut, rms), **kw)
        ax[1].plot(lt, lvl(d["u"] - ut, np.median), **kw)
        ax[2].plot(lt, lvl(d["z"] - zt, rms) / 1e3, **kw)
        ax[3].plot(lt, lvl(d["u"] - uc, lambda v: np.percentile(v, 99)), **kw)
    if "fh" in d:
        ax[4].plot(lt, lvl(d["fh"], np.median), **kw)
        ax[5].plot(lt, lvl(np.abs(d["fz"]), np.median), **kw)
    if "logtau_cut" in d["attrs"]:
        for i in range(6):
            ax[i].axvline(d["attrs"]["logtau_cut"], color=COL[r], ls=":", lw=0.8)
ax[0].set_ylabel("rms ln(P/P_Bifrost)"); ax[1].set_ylabel("median ln(P/P_Bifrost)")
ax[2].set_ylabel("rms z − z_Bifrost [km]"); ax[3].set_ylabel("p99 ln(P/P_column HSE)")
ax[4].set_ylabel("median |F_h|/ρg"); ax[5].set_ylabel("median |F_z|/ρg")
for i in (0, 2, 3, 4, 5):
    ax[i].set_yscale("log")
for a_ in ax:
    a_.grid(alpha=0.3); a_.set_xlabel("log10 τ500")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/profiles.png", dpi=110)

# β profile (truth β per level: median and fraction < 1)
if "beta" in res.get("truth", {}):
    bt = res["truth"]["beta"]
    p("\nlog τ   median β (Bifrost)   frac β<1   median |L|/ρg")
    for k in range(0, len(lt), 5):
        p(f"{lt[k]:5.1f}   {np.median(bt[k]):12.3g}   {np.mean(bt[k] < 1):8.3f}   {np.median(res['truth']['lor'][k]):10.3g}")

# z(τ=1) maps
show = [r for r in ["truth", "colhse", "cut4", "cut6", "cutnone"] if r in res]
fig, ax = plt.subplots(2, len(show), figsize=(3.6 * len(show), 7))
for i, r in enumerate(show):
    im = ax[0, i].imshow(res[r]["z1"] / 1e3, origin="lower", cmap="viridis", vmin=-300, vmax=300)
    ax[0, i].set_title(f"z(τ=1) {LABEL[r]}"); plt.colorbar(im, ax=ax[0, i], shrink=0.8, label="km")
    if r != "truth":
        im = ax[1, i].imshow((res[r]["z1"] - z1t) / 1e3, origin="lower", cmap="RdBu_r", vmin=-200, vmax=200)
        ax[1, i].set_title(f"{LABEL[r]} − Bifrost"); plt.colorbar(im, ax=ax[1, i], shrink=0.8, label="km")
    else:
        im = ax[1, i].imshow(B1 * 1e4, origin="lower", cmap="magma"); ax[1, i].set_title("|B| at τ=1 [G]")
        plt.colorbar(im, ax=ax[1, i], shrink=0.8)
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); fig.savefig(f"{D}/ztau1.png", dpi=100)

# error vs β (truth β), each run
if "beta" in res.get("truth", {}):
    lb = np.log10(res["truth"]["beta"])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, r in enumerate(["cut4", "cut5", "cut6", "cutnone"]):
        if r not in res:
            continue
        du = (res[r]["u"] - ut).ravel(); m = np.isfinite(du) & np.isfinite(lb.ravel())
        ax[i].hist2d(lb.ravel()[m], du[m], bins=[120, 120], range=[[-4, 6], [-4, 4]], cmap="Greys", norm=matplotlib.colors.LogNorm())
        ax[i].axvline(0, color="r", lw=0.8); ax[i].axhline(0, color="b", lw=0.8)
        ax[i].set_title(LABEL[r]); ax[i].set_xlabel("log10 β (Bifrost)")
    ax[0].set_ylabel("ln(P/P_Bifrost)")
    fig.tight_layout(); fig.savefig(f"{D}/err_vs_beta.png", dpi=100)

# levels below each cut: column-HSE start vs result, split strong/quiet
p("\nrms ln(P/P_Bifrost) by band, strong / quiet columns")
bands = [(-8.1, -6), (-6, -4), (-4, -2), (-2, 0), (0, 1.5)]
p("run           " + "  ".join(f"[{a:.0f},{b:.0f})".rjust(15) for a, b in bands))
for r in RUNS:
    if r not in res:
        continue
    du = res[r]["u"] - ut; row = []
    for a, b in bands:
        ks = (lt >= a) & (lt < b)
        s = du[ks][:, strong]; q = du[ks][:, quiet]
        row.append(f"{rms(s[np.isfinite(s)]):6.3f}/{rms(q[np.isfinite(q)]):6.3f}".rjust(15))
    p(f"{LABEL[r]:13s} " + "  ".join(row))
out.close()
