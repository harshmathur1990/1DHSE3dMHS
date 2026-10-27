"""Viewing-angle experiments (design/tau/VIEWING_ANGLE_DESIGN.md §7): plane-parallel and inclined mhs-poisson-ltau (and
the column HSE start) on synthetic off-centre τ cubes from Bifrost, against the Bifrost truth along the same rays.

usage: python analyse_view.py DIR TAG [TAG ...]
DIR holds, per TAG (e.g. mu0.707_az0): input_TAG.nc, truth_TAG.h5, colhse_TAG.h5, pp_TAG.h5, inc_TAG.h5, their .log files
and diag_{truth,colhse,pp,inc}_TAG.h5 (scripts/viewing/force_diag.jl with the cube's true μ, φ). The μ = 1 reference of the
τ-cut study can be included as a TAG whose pp and inc files are the same run.
writes DIR/summary.txt, DIR/ztau1_vs_mu.png, DIR/profiles.png, DIR/maps_TAG.png
"""
import re
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
TAGS = sys.argv[2:]
RUNS = ["colhse", "pp", "inc"]
LABEL = {"colhse": "column HSE", "pp": "plane-parallel", "inc": "inclined", "truth": "Bifrost"}
COL = {"colhse": "#7f7f7f", "pp": "#d62728", "inc": "#1f77b4", "truth": "k"}

out = open(f"{D}/summary.txt", "w")


def p(*a):
    print(*a)
    print(*a, file=out)


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


def lvl(a, fn):
    return np.array([fn(a[k][np.isfinite(a[k])]) if np.isfinite(a[k]).any() else np.nan for k in range(a.shape[0])])


def wall(path):
    try:
        m = re.search(r"WALL ([\d.]+) s", open(path).read())
        return float(m.group(1)) if m else np.nan
    except OSError:
        return np.nan


def load(tag):
    d = {}
    with h5py.File(f"{D}/truth_{tag}.h5") as f:
        lt = f["logtau500"][:]
        mu = float(f.attrs["view_mu"]); az = float(f.attrs["view_azimuth_deg"])
        k0 = int(np.argmin(abs(lt)))
        z1 = f["truth/z_tau1_m"][:]
        zt = f["truth/z_m"][:]
        d["truth"] = dict(u=f["truth/lnP"][:], z=zt - np.nanmean(zt[k0]), z1=z1 - z1.mean(), attrs={})
    with h5py.File(f"{D}/input_{tag}.nc") as f:                 # (x, y, τ) → (τ, y, x)
        B1 = np.sqrt(sum(f[c][:] ** 2 for c in ("B_x", "B_y", "B_z"))).transpose(2, 1, 0)[k0]
    for r in RUNS:
        try:
            f = h5py.File(f"{D}/{r}_{tag}.h5")
        except OSError:
            continue
        z = f["tau_grid/z_m"][:]
        z1 = f["tau_grid/z_tau1_m"][:]
        d[r] = dict(u=np.log(f["tau_grid/Pgas"][:]), z=z - z[k0].mean(), z1=z1 - z1.mean(), attrs=dict(f.attrs),
                    wall=wall(f"{D}/{r}_{tag}.log"))
        f.close()
    for r in list(d):
        try:
            with h5py.File(f"{D}/diag_{r}_{tag}.h5") as g:
                d[r].update(fh=g["fh"][:], fz=g["fz"][:], beta=g["beta"][:])
        except OSError:
            pass
    return dict(lt=lt, mu=mu, az=az, k0=k0, strong=B1 > 0.05, quiet=B1 < 0.01, runs=d)


data = {t: load(t) for t in TAGS}

p("run           μ      φ    | steps stop        Φ/Φ0     wall s | z(τ_LOS=1) err rms  corr  str−quiet (truth) | "
  "rms ΔlnP: [-4,-2) [-2,0) [0,1.5) | median |F_h|/ρg, |F_z|/ρg at log τ [-2, 0]")
summary = {}
for tag, c in data.items():
    t = c["runs"]["truth"]; lt = c["lt"]
    phot = (lt >= -2) & (lt <= 0)
    wt = t["z1"][c["strong"]].mean() / 1e3 - t["z1"][c["quiet"]].mean() / 1e3
    for r in RUNS + ["truth"]:
        if r not in c["runs"]:
            continue
        d = c["runs"][r]; a = d["attrs"]
        e1 = d["z1"] - t["z1"]
        w = d["z1"][c["strong"]].mean() / 1e3 - d["z1"][c["quiet"]].mean() / 1e3
        du = d["u"] - t["u"]
        bands = [rms(du[(lt >= lo) & (lt < hi)]) for lo, hi in ((-4, -2), (-2, 0), (0, 1.5))]
        ph = a["phi_end"] / a["phi_start"] if "phi_end" in a else np.nan
        fh = np.nanmedian(d["fh"][phot]) if "fh" in d else np.nan
        fz = np.nanmedian(np.abs(d["fz"][phot])) if "fz" in d else np.nan
        p(f"{LABEL[r]:13s} {c['mu']:.3f} {c['az']:5.1f} | {str(a.get('steps', '-')):>5s} {str(a.get('stop_reason', '-')):11s} "
          f"{ph:8.2e} {d.get('wall', np.nan):7.0f} | {rms(e1) / 1e3:8.1f} km {np.corrcoef(d['z1'].ravel(), t['z1'].ravel())[0, 1]:6.3f} "
          f"{w:+7.1f} ({wt:+.1f}) km | {bands[0]:.3f} {bands[1]:.3f} {bands[2]:.3f} | {fh:.3f} {fz:.3f}")
        summary[(tag, r)] = dict(mu=c["mu"], az=c["az"], err=rms(e1) / 1e3, wil=w, wil_t=wt)

# z(τ_LOS = 1) error against μ
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for r in RUNS:
    pts = sorted((v["mu"], v["err"], v["wil"] - v["wil_t"], v["az"]) for (t, rr), v in summary.items() if rr == r)
    if not pts:
        continue
    m = np.array(pts)
    for az, mk in ((0.0, "o"), (45.0, "s")):
        s = np.isclose(m[:, 3], az) | (m[:, 0] == 1.0)
        if s.any():
            ax[0].plot(m[s, 0], m[s, 1], marker=mk, color=COL[r], label=f"{LABEL[r]} (φ {az:.0f}°)")
            ax[1].plot(m[s, 0], m[s, 2], marker=mk, color=COL[r])
ax[0].set_ylabel("rms z(τ_LOS=1) − Bifrost [km]"); ax[1].set_ylabel("strong − quiet height difference: run − Bifrost [km]")
for a_ in ax:
    a_.set_xlabel("μ"); a_.grid(alpha=0.3); a_.invert_xaxis()
ax[1].axhline(0, color="k", lw=0.8)
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/ztau1_vs_mu.png", dpi=110)

# per-level errors for every μ
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
cm = plt.get_cmap("viridis")
for n, (tag, c) in enumerate(data.items()):
    t = c["runs"]["truth"]; col = cm(n / max(1, len(data) - 1))
    for r, ls in (("pp", "--"), ("inc", "-")):
        if r not in c["runs"]:
            continue
        d = c["runs"][r]
        ax[0].plot(c["lt"], lvl(d["u"] - t["u"], rms), ls, color=col, label=f"{LABEL[r]} μ {c['mu']:.3f} φ {c['az']:.0f}")
        ax[1].plot(c["lt"], lvl(d["z"] - t["z"], rms) / 1e3, ls, color=col)
        if "fh" in d:
            ax[2].plot(c["lt"], lvl(d["fh"], np.median), ls, color=col)
ax[0].set_ylabel("rms ln(P/P_Bifrost)"); ax[1].set_ylabel("rms z − z_Bifrost [km]"); ax[2].set_ylabel("median |F_h|/ρg")
for a_ in ax:
    a_.set_yscale("log"); a_.grid(alpha=0.3); a_.set_xlabel("log10 τ_LOS"); a_.axvline(-4, color="k", ls=":", lw=0.8)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{D}/profiles.png", dpi=110)

# maps for each tag
for tag, c in data.items():
    show = [r for r in ("truth", "colhse", "pp", "inc") if r in c["runs"]]
    fig, ax = plt.subplots(2, len(show), figsize=(3.6 * len(show), 7))
    t = c["runs"]["truth"]
    for i, r in enumerate(show):
        z1 = c["runs"][r]["z1"]
        im = ax[0, i].imshow(z1 / 1e3, origin="lower", cmap="viridis", vmin=-300, vmax=300)
        ax[0, i].set_title(f"z(τ_LOS=1) {LABEL[r]}"); plt.colorbar(im, ax=ax[0, i], shrink=0.8, label="km")
        if r == "truth":
            ax[1, i].axis("off")
            continue
        im = ax[1, i].imshow((z1 - t["z1"]) / 1e3, origin="lower", cmap="RdBu_r", vmin=-200, vmax=200)
        ax[1, i].set_title(f"{LABEL[r]} − Bifrost"); plt.colorbar(im, ax=ax[1, i], shrink=0.8, label="km")
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"μ = {c['mu']:.3f}, φ = {c['az']:.0f}° (footprint grid)")
    fig.tight_layout(); fig.savefig(f"{D}/maps_{tag}.png", dpi=90)
out.close()
