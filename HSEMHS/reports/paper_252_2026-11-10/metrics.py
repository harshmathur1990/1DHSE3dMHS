"""Table 1 of the method paper: the runs of out/paper_252_2026-11-10 against the Bifrost snapshot on the 252² columns
of the MULTI3D run (inputs and truths: IO/tau252, built by scripts/inputs/make_tau_input.jl and make_logm_input.py).

usage: python metrics.py RUNDIR IODIR
Writes RUNDIR/metrics.txt.

Per case the reference is the truth written with its input (same columns, same rays, same τ). Mask: strong / quiet =
|B| > 500 G / < 100 G at log τ = 0 of the case's τ input (the column-mass case uses the μ = 1 eos input). Gauge: every
state, the truth included, shifted so that its mean z(τ500 = 1) is 0. Depression: mean z(τ500 = 1) of strong minus quiet
columns. Errors of P, ρ, z over the solved band of the default cut (log τ ≥ -4, log10 m ≥ -0.5) and over all levels;
f2, f10: fraction of points with P (or ρ) within a factor 2, 10 of the truth (cf. Borrero et al. 2019).
"""
import re
import sys

import h5py
import numpy as np

RD, IO = sys.argv[1:3]
out = open(f"{RD}/metrics.txt", "w")

CASES = [  # (label, run, input case, grid)
    ("μ=1 own τ: column HSE", "colhse_eos_mu1.0", "eos_mu1.0", "tau"),
    ("μ=1 own τ: MHS -4", "mhs_eos_mu1.0", "eos_mu1.0", "tau"),
    ("μ=1 own τ: MHS -5", "cut5_eos_mu1.0", "eos_mu1.0", "tau"),
    ("μ=1 own τ: MHS -6", "cut6_eos_mu1.0", "eos_mu1.0", "tau"),
    ("μ=1 own τ: MHS none", "cutnone_eos_mu1.0", "eos_mu1.0", "tau"),
    ("μ=1 M3D τ: column HSE", "colhse_m3d_mu1.0", "m3d_mu1.0", "tau"),
    ("μ=1 M3D τ: MHS -4", "mhs_m3d_mu1.0", "m3d_mu1.0", "tau"),
    ("μ=0.707 inc: column HSE", "colhse_eos_mu0.707_az0", "eos_mu0.707_az0", "tau"),
    ("μ=0.707 inc: MHS -4", "mhs_eos_mu0.707_az0", "eos_mu0.707_az0", "tau"),
    ("μ=0.5 inc: column HSE", "colhse_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"),
    ("μ=0.5 inc: MHS -4", "mhs_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"),
    ("μ=0.5 pp: MHS -4", "mhspp_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"),
    ("m: column HSE", "colhse_logm_mu1.0", "logm_mu1.0", "m"),
    ("m: MHS -0.5", "mhs_logm_mu1.0", "logm_mu1.0", "m"),
]


def p(*a):
    print(*a); print(*a, file=out)


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


def frac(d, f):
    d = d[np.isfinite(d)]
    return np.mean(np.abs(d) < np.log(f))


def wall(fn):
    try:
        m = re.search(r"WALL ([0-9.]+) s", open(fn).read())
        return float(m[1]) if m else np.nan
    except OSError:
        return np.nan


def attr(A, k):
    if k not in A:
        return "-"
    v = A[k]
    return v.decode() if isinstance(v, bytes) else v


def masks(case):
    with h5py.File(f"{IO}/input_{case}.nc") as f:
        lt = f["log10tau500"][:]; k0 = int(np.argmin(abs(lt)))
        B = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T      # (y, x)
    return B > 0.05, B < 0.01


def tau_truth(case):
    with h5py.File(f"{IO}/truth_{case}.h5") as f:
        z1 = f["truth/z_tau1_m"][:]; off = z1.mean()
        return dict(lt=f["logtau500"][:], u=f["truth/lnP"][:], r=f["truth/lnrho"][:], z=f["truth/z_m"][:] - off, z1=z1 - off,
                    off=off)


rows = []
truths = {}
for label, run, case, grid in CASES:
    try:
        f = h5py.File(f"{RD}/{run}.h5")
    except OSError:
        p(f"({run}.h5 missing)"); continue
    A = dict(f.attrs)
    if grid == "tau":
        T = truths.setdefault(case, tau_truth(case)); strong, quiet = masks(case)
        g = f["tau_grid"]; band = T["lt"] >= -4.0
    else:
        T0 = truths.setdefault("eos_mu1.0", tau_truth("eos_mu1.0")); strong, quiet = masks("eos_mu1.0")
        with h5py.File(f"{IO}/truth_{case}.h5") as t:                  # (x, y, m) → (m, y, x); same snapshot frame
            T = dict(lt=t["log10m"][:], u=np.log(t["Pgas"][:]).transpose(2, 1, 0), r=np.log(t["rho"][:]).transpose(2, 1, 0),
                     z=t["z_m"][:].transpose(2, 1, 0) - T0["off"], z1=T0["z1"])
        g = f["mass_grid"]; band = T["lt"] >= -0.5
    z1 = g["z_tau1_m"][:]; z = g["z_m"][:] - z1.mean(); z1 = z1 - z1.mean()
    dP = np.log(g["Pgas"][:]) - T["u"]; dr = np.log(g["rho"][:]) - T["r"]; dz = z - T["z"]
    f.close()
    ph = A["phi_end"] / A["phi_start"] if "phi_end" in A else np.nan
    rows.append((label, attr(A, "steps"), attr(A, "status"), ph, attr(A, "qual_outcome"), A.get("solve_seconds", np.nan),
                 wall(f"{RD}/{run}.log"), rms(z1 - T["z1"]) / 1e3, np.corrcoef(z1.ravel(), T["z1"].ravel())[0, 1],
                 (z1[strong].mean() - z1[quiet].mean()) / 1e3, (T["z1"][strong].mean() - T["z1"][quiet].mean()) / 1e3,
                 T["z1"].std() / 1e3, rms(dP[band]), rms(dr[band]), rms(dz[band]) / 1e3,
                 frac(dP[band], 2), frac(dr[band], 2), frac(dP[band], 10), frac(dr[band], 10), rms(dP), frac(dP, 10), frac(dr, 10)))

p("solved band: log τ ≥ −4 | log10 m ≥ −0.5. z1 = z(τ500 = 1). dep = strong − quiet z1 (run / truth).")
p(f"{'case':26s} {'steps':>5s} {'status':10s} {'Φ/Φ0':>8s} {'qual':10s} {'solve':>6s} {'wall':>6s} | {'z1 rms':>6s} {'corr':>5s} "
  f"{'dep':>6s} {'dep T':>6s} {'std T':>5s} | {'ΔlnP':>5s} {'Δlnρ':>5s} {'Δz':>6s} | {'f2P':>5s} {'f2ρ':>5s} {'f10P':>5s} "
  f"{'f10ρ':>5s} | all: {'ΔlnP':>5s} {'f10P':>5s} {'f10ρ':>5s}")
for q in rows:
    p(f"{q[0]:26s} {str(q[1]):>5s} {str(q[2]):10s} {q[3]:8.2e} {str(q[4]):10s} {q[5]:6.0f} {q[6]:6.0f} | {q[7]:6.1f} {q[8]:5.3f} "
      f"{q[9]:+6.1f} {q[10]:+6.1f} {q[11]:5.1f} | {q[12]:5.3f} {q[13]:5.3f} {q[14]:6.1f} | {q[15]:5.3f} {q[16]:5.3f} {q[17]:5.3f} "
      f"{q[18]:5.3f} |      {q[19]:5.3f} {q[20]:5.3f} {q[21]:5.3f}")
