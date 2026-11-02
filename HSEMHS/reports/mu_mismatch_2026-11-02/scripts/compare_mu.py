"""Compare mu = 1 and mu = 0.5 (inclined) solutions of the same vertical-tau input: column HSE and mhs-poisson-ltau."""
import copy
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out/modes/"
RUNS = {
    ("col", 1.0): OUT + "halpha_colhse_mu1.h5",
    ("col", 0.5): OUT + "halpha_colhse_mu05b.h5",
    ("mhs", 1.0): OUT + "halpha_mu1.h5",
    ("mhs", 0.5): OUT + "halpha_mu05_inclined.h5",
}

d = {}
for key, path in RUNS.items():
    f = h5py.File(path)
    g = f["tau_grid"]
    lt = g["logtau500"][:]
    i0 = int(np.argmin(abs(lt)))
    i2 = int(np.argmin(abs(lt + 2)))
    d[key] = dict(
        zt1=g["z_tau1_m"][:] / 1e3,
        z0=g["z_m"][i0] / 1e3,
        z2=g["z_m"][i2] / 1e3,
        P0=g["Pgas"][i0],
        B0=np.sqrt(g["Bx"][i0] ** 2 + g["By"][i0] ** 2 + g["Bz"][i0] ** 2) * 1e4,
        T0=g["T"][i0],
        status=f.attrs.get("status", b"").decode() if isinstance(f.attrs.get("status", b""), bytes) else f.attrs.get("status", ""),
    )
x = h5py.File(RUNS[("mhs", 1.0)])["coords/x_m"][:] / 1e6
y = h5py.File(RUNS[("mhs", 1.0)])["coords/y_m"][:] / 1e6
B = d[("col", 1.0)]["B0"]


def pct(a):
    return np.percentile(a, [1, 50, 99]).round(1)


print("== spread of z(tau_LOS=1), km: p1/p50/p99, std")
for k in RUNS:
    print(f"  {k[0]} mu {k[1]}: {pct(d[k]['zt1'])}  std {d[k]['zt1'].std():.1f}   status {d[k]['status']}")

print("== P1: column HSE, mu 0.5 vs mu 1")
a, b = d[("col", 1.0)], d[("col", 0.5)]
dz = b["zt1"] - a["zt1"]
print(f"  z(tau=1) difference p1/p50/p99 {pct(dz)} km, rms {dz.std():.2f} km, corr {np.corrcoef(a['zt1'].ravel(), b['zt1'].ravel())[0,1]:.4f}")
th_a, th_b = (a["z2"] - a["z0"]), (b["z2"] - b["z0"])
print(f"  thickness z(logtau -2) - z(logtau 0): median mu1 {np.median(th_a):.1f} km, mu0.5 {np.median(th_b):.1f} km, ratio {np.median(th_b / th_a):.3f}")
r = b["P0"] / a["P0"]
print(f"  P(logtau 0) ratio mu0.5/mu1: p1/p50/p99 {np.percentile(r, [1, 50, 99]).round(3)}")
rm = d[("mhs", 0.5)]["P0"] / d[("mhs", 1.0)]["P0"]
print(f"  (MHS) P(logtau 0) ratio mu0.5/mu1: p1/p50/p99 {np.percentile(rm, [1, 50, 99]).round(3)}")

print("== P2: z(tau=1) minus box median, by |B| at logtau 0 (G)")
bins = [(0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 1e9)]
print("  |B| bin        n     col mu1  col mu.5   mhs mu1  mhs mu.5")
for lo, hi in bins:
    m = (B >= lo) & (B < hi)
    row = [np.mean(d[k]["zt1"][m]) - np.median(d[k]["zt1"]) for k in RUNS]
    print(f"  {lo:>5.0f}-{hi if hi < 1e8 else 'inf':>5}  {m.sum():6d}  " + "  ".join(f"{v:8.1f}" for v in row))

print("== P3: MHS, mu 0.5 minus mu 1")
a, b = d[("mhs", 1.0)], d[("mhs", 0.5)]
dz = b["zt1"] - a["zt1"]
print(f"  dz p1/p50/p99 {pct(dz)} km, rms {dz.std():.1f} km; corr(mu1, mu0.5) {np.corrcoef(a['zt1'].ravel(), b['zt1'].ravel())[0,1]:.3f}")
# x-direction bias: correlation of dz with x- and y-gradients of the mu=1 heights and of T
for name, fld in [("z_mu1", a["zt1"]), ("lnT(logtau 0)", np.log(a["T0"]))]:
    gx = (np.roll(fld, -1, 1) - np.roll(fld, 1, 1)) / 2
    gy = (np.roll(fld, -1, 0) - np.roll(fld, 1, 0)) / 2
    cx = np.corrcoef(dz.ravel(), gx.ravel())[0, 1]
    cy = np.corrcoef(dz.ravel(), gy.ravel())[0, 1]
    print(f"  corr(dz, d/dx {name}) {cx:+.3f}   corr(dz, d/dy {name}) {cy:+.3f}")
# anisotropy of dz: autocorrelation at 3 px lag along x vs y
dd = dz - dz.mean()
ax_ = (dd * np.roll(dd, 3, 1)).mean() / dd.var()
ay_ = (dd * np.roll(dd, 3, 0)).mean() / dd.var()
print(f"  dz autocorrelation at 3 px: along x {ax_:.3f}, along y {ay_:.3f}")
# large-scale part: dz smoothed over 2 Mm boxes, and left/right halves
k = 42
sm = np.real(np.fft.ifft2(np.fft.fft2(dz) * np.fft.fft2(np.pad(np.ones((k, k)) / k**2, ((0, dz.shape[0] - k), (0, dz.shape[1] - k))))))
print(f"  dz smoothed over 2 Mm: rms {sm.std():.1f} km (total {dz.std():.1f}); left half mean {dz[:, :252].mean():+.1f}, right half {dz[:, 252:].mean():+.1f} km")
print(f"  mu=1 MHS: left half mean z(tau=1) {a['zt1'][:, :252].mean():+.1f}, right {a['zt1'][:, 252:].mean():+.1f};"
      f" mu=0.5: left {b['zt1'][:, :252].mean():+.1f}, right {b['zt1'][:, 252:].mean():+.1f} km")

# figure: z(tau=1) maps, column HSE and MHS, mu 1, mu 0.5, difference
cm = copy.copy(plt.get_cmap("RdBu_r"))
fig, axs = plt.subplots(2, 3, figsize=(17, 10.4), dpi=130, constrained_layout=True)
ext = [x[0], x[-1], y[0], y[-1]]
lim = np.percentile(np.abs(np.concatenate([d[k]["zt1"].ravel() for k in RUNS])), 99)
for r_, kind, label in [(0, "col", "column HSE"), (1, "mhs", "mhs-poisson-ltau")]:
    a, b = d[(kind, 1.0)], d[(kind, 0.5)]
    diff = b["zt1"] - a["zt1"]
    dl = max(np.percentile(np.abs(diff), 99), 1.0)
    for c_, fld, title, L in [(0, a["zt1"], f"{label}, μ = 1", lim), (1, b["zt1"], f"{label}, μ = 0.5 inclined", lim),
                             (2, diff, f"{label}: (μ = 0.5) − (μ = 1)", dl)]:
        ax = axs[r_, c_]
        im = ax.imshow(fld, origin="lower", extent=ext, cmap=cm, vmin=-L, vmax=L, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("z(τ_LOS = 1) (km)" if c_ < 2 else "Δz (km)")
        ax.set_title(title + f"   (std {fld.std():.1f} km)", fontsize=10)
        ax.set_xlabel("x (Mm)")
        ax.set_ylabel("y (Mm)")
        for s in ax.spines.values():
            s.set_linewidth(0.5)
fig.suptitle("Same vertical-τ input solved as μ = 1 and as μ = 0.5 (inclined, φ = 0°, tilt toward +x); "
             "heights gauged to mean z(τ_LOS = 1) = 0; footprint grid", fontsize=10)
fig.savefig("out/figs/halpha_mu1_vs_mu05_ztau1.png")
print("wrote out/figs/halpha_mu1_vs_mu05_ztau1.png")
