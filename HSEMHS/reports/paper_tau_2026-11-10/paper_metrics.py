"""Metrics for the method paper (HSEMHS-Paper, Table 1): τ-grid and column-mass runs against the Bifrost snapshot.

usage: python paper_metrics.py TAUDIR LOGMDIR LOGMCUTDIR
  TAUDIR      out/paper_tau_2026-11-10: truth.h5, input-ltau-xy.nc, colhse.h5, mhs.h5, cut5.h5, cut6.h5, cutnone.h5, <run>.log
  LOGMDIR     out/logm_2026-11-02: truth_logm.h5, column-hse-logm.h5
  LOGMCUTDIR  out/logm_cut_2026-11-03: cut_-0.5.h5
Writes TAUDIR/metrics.txt.

Both grids use the same pixels, mask and gauge, so their rows compare:
  strong / quiet: |B| > 500 G / < 100 G at log τ = 0 of the τ input;
  heights: every state (truth too) shifted so that its mean z(τ500 = 1) is 0;
  depression: mean z(τ500 = 1) of strong minus quiet columns.
Errors of P, ρ and z are given over the solved band of the default cut (log τ ≥ -4, log10 m ≥ -0.5) and over all levels;
"f2", "f10": fraction of points with P (or ρ) within a factor 2, 10 of the truth (cf. Borrero et al. 2019).
"""
import re
import sys

import h5py
import numpy as np

TD, LD, LCD = sys.argv[1:4]
out = open(f"{TD}/metrics.txt", "w")


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


def attr(A, k, default="-"):
    if k not in A:
        return default
    v = A[k]
    return v.decode() if isinstance(v, bytes) else v


with h5py.File(f"{TD}/truth.h5") as f:
    lt = f["logtau500"][:]
    ut = f["truth/lnP"][:]; rt = f["truth/lnrho"][:]; zt = f["truth/z_m"][:]; z1raw = f["truth/z_tau1_m"][:]
z1off = z1raw.mean()
z1t = z1raw - z1off; zt = zt - z1off
k0 = int(np.argmin(abs(lt)))
with h5py.File(f"{TD}/input-ltau-xy.nc") as f:                       # (x, y, τ) → (y, x)
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
strong, quiet = B1 > 0.05, B1 < 0.01
dep_t = z1t[strong].mean() - z1t[quiet].mean()
p(f"strong {strong.mean() * 100:.1f} %, quiet {quiet.mean() * 100:.1f} % (|B| at log τ = 0); truth z(τ=1) std "
  f"{z1t.std() / 1e3:.1f} km, depression (strong − quiet) {dep_t / 1e3:+.1f} km\n")

rows = []


def row(label, P, rho, z, z1, A, solved, wall_s, truth):
    uP, ur, uz = truth
    z1 = z1 - z1.mean()
    dP = np.log(P) - uP; dr = np.log(rho) - ur; dz = z - uz
    e = z1 - z1t
    st = attr(A, "status"); steps = attr(A, "steps")
    ph = A["phi_end"] / A["phi_start"] if "phi_end" in A else np.nan
    t = A["solve_seconds"] if "solve_seconds" in A else np.nan
    rows.append((label, steps, st, ph, attr(A, "qual_outcome"), t, wall_s,
                 rms(e) / 1e3, np.corrcoef(z1.ravel(), z1t.ravel())[0, 1], (z1[strong].mean() - z1[quiet].mean()) / 1e3,
                 rms(dP[solved]), rms(dr[solved]), rms(dz[solved]) / 1e3,
                 frac(dP[solved], 2), frac(dP[solved], 10), frac(dr[solved], 2), frac(dr[solved], 10),
                 rms(dP), frac(dP, 10), frac(dr, 10)))


# τ grid: runs gauged mean z(τ=1) = 0 by the code; truth shifted above
band_t = lt >= -4.0
for r, lab in [("colhse", "tau col HSE"), ("mhs", "tau MHS -4"), ("cut5", "tau MHS -5"), ("cut6", "tau MHS -6"),
               ("cutnone", "tau MHS none")]:
    try:
        f = h5py.File(f"{TD}/{r}.h5")
    except OSError:
        p(f"({r}.h5 missing)"); continue
    g = f["tau_grid"]; A = dict(f.attrs)
    z1 = g["z_tau1_m"][:]
    z = g["z_m"][:] - z1.mean()
    row(lab, g["Pgas"][:], g["rho"][:], z, z1, A, band_t, wall(f"{TD}/{r}.log"), (ut, rt, zt))
    f.close()

# column mass: truth on the m axis, same gauge (the snapshot frame of truth.h5)
with h5py.File(f"{LD}/truth_logm.h5") as f:                           # (x, y, m) → (m, y, x)
    lm = f["log10m"][:]
    tr = tuple(a.transpose(2, 1, 0) for a in (np.log(f["Pgas"][:]), np.log(f["rho"][:]), f["z_m"][:] - z1off))
band_m = lm >= -0.5
for fn, lab in [(f"{LD}/column-hse-logm.h5", "m col HSE"), (f"{LCD}/cut_-0.5.h5", "m MHS -0.5")]:
    with h5py.File(fn) as f:
        g = f["mass_grid"]; A = dict(f.attrs)
        z1 = g["z_tau1_m"][:]
        row(lab, g["Pgas"][:], g["rho"][:], g["z_m"][:] - z1.mean(), z1, A, band_m, np.nan, tr)

p("solved band (log τ ≥ −4 | log10 m ≥ −0.5)                       z(τ=1): rms err, corr, depression | band: rms ΔlnP Δlnρ Δz | "
  "f2 P f10 P f2 ρ f10 ρ | all levels: rms ΔlnP, f10 P, f10 ρ")
p(f"{'run':13s} {'steps':>5s} {'status':10s} {'Φ/Φ0':>8s} {'qual':10s} {'solve s':>8s} {'wall s':>7s} | "
  f"{'rms km':>7s} {'corr':>5s} {'dep km':>7s} | {'ΔlnP':>5s} {'Δlnρ':>5s} {'Δz km':>6s} | "
  f"{'f2P':>5s} {'f10P':>5s} {'f2ρ':>5s} {'f10ρ':>5s} | {'ΔlnP':>5s} {'f10P':>5s} {'f10ρ':>5s}")
for q in rows:
    p(f"{q[0]:13s} {str(q[1]):>5s} {str(q[2]):10s} {q[3]:8.2e} {str(q[4]):10s} {q[5]:8.0f} {q[6]:7.0f} | "
      f"{q[7]:7.1f} {q[8]:5.3f} {q[9]:+7.1f} | {q[10]:5.3f} {q[11]:5.3f} {q[12]:6.1f} | "
      f"{q[13]:5.3f} {q[14]:5.3f} {q[15]:5.3f} {q[16]:5.3f} | {q[17]:5.3f} {q[18]:5.3f} {q[19]:5.3f}")
p(f"{'Bifrost':13s} {'':>5s} {'':10s} {'':>8s} {'':10s} {'':>8s} {'':>7s} | {0:7.1f} {1:5.3f} {dep_t / 1e3:+7.1f}")
