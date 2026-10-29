"""Accuracy against the step count from scripts/viewing/measure_steps.jl snapshots (design/tau/SOLVER_TIME_DESIGN.md §3).

usage: python measure_analyse.py RUNDIR TRUTH.h5 INPUT.nc [LABEL]
Per saved step: Φ/Φ₀, predicted decrease, CG its/rel, step and cumulative seconds; z(τ_LOS=1) rms error against the truth
and its change from the final state; the strong − quiet height difference (Bifrost value in the header); rms ln P error
by log τ band. Writes RUNDIR/steps.csv and prints the table.
"""
import glob
import sys

import h5py
import numpy as np

run, truth, inp = sys.argv[1:4]
label = sys.argv[4] if len(sys.argv) > 4 else run

with h5py.File(truth) as f:
    lt = f["logtau500"][:]
    zt = f["truth/z_m"][:]; ut = f["truth/lnP"][:]          # h5py (τ, y, x)
    z1t = f["truth/z_tau1_m"][:]
k0 = int(np.argmin(abs(lt)))
with h5py.File(inp) as f:                                   # h5py (x, y, τ)
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
strong, quiet = B1 > 0.05, B1 < 0.01
cls = "strong−quiet"
if not strong.any() or not quiet.any():                     # e.g. the controls: depressed (lowest 20 % of truth z(τ=1)) − raised
    strong, quiet = z1t <= np.quantile(z1t, 0.2), z1t >= np.quantile(z1t, 0.8)
    cls = "low−high"
z1off = z1t.mean()
z1t = z1t - z1t.mean()
files = sorted(glob.glob(f"{run}/snap_r*.h5"), key=lambda p: h5py.File(p).attrs["i0"])
hs = [h5py.File(p) for p in files]
kc = int(hs[0].attrs["kc"]) - 1                              # 0-based
names = [n for n in hs[0].keys()]
order = sorted([n for n in names if n.startswith("s")], key=lambda n: int(n[1:])) + (["final"] if "final" in names else [])


def get(name, ds):                                          # Julia (τ, y, x) slabs → h5py (x, y, τ) per file → (τ, y, x)
    return np.concatenate([h[name][ds][:] for h in hs], axis=0).transpose(2, 1, 0)


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


zf = get("final", "z")[k0 - kc]; zf = zf - zf.mean()
bands = [(-4, -2), (-2, 0), (0, 1.5)]
cum = 0.0
rows = []
print(f"{label}: steps {hs[0].attrs.get('steps')}, {hs[0].attrs.get('reason')}, solve {hs[0].attrs.get('solve_seconds', np.nan):.0f} s; "
      f"truth {cls} {z1t[strong].mean() / 1e3 - z1t[quiet].mean() / 1e3:+.1f} km")
work = {k[5:]: hs[0].attrs[k] for k in hs[0].attrs if k.startswith("work_")}
qual = {k[5:]: hs[0].attrs[k] for k in hs[0].attrs if k.startswith("qual_")}
work and print("work: " + ", ".join(f"{k} {v}" for k, v in work.items()))
qual and print("last stop check: " + ", ".join(f"{k} {v}" for k, v in qual.items()))
print("step    Φ/Φ0   pred     CG  rel   CGstep solves  ρ    cap   s/step   cum s | z1 err rms  Δ from final rms/max | " + cls + " | mean z1 off | Φv/Φ Φv3/Φv | rms ΔlnP " +
      " ".join(f"[{a},{b})" for a, b in bands))
phi0 = hs[0].attrs["phi0"]
for n in order:
    a = hs[0][n].attrs
    if n != "final":
        cum += a["seconds"]
    z1 = get(n, "z")[k0 - kc]; off = (z1.mean() - z1off) / 1e3; z1 = z1 - z1.mean()
    du = get(n, "lnP") - ut[kc:]
    lts = lt[kc:]
    b = [rms(du[(lts >= lo) & (lts < hi)]) for lo, hi in bands]
    w = z1[strong].mean() / 1e3 - z1[quiet].mean() / 1e3
    e = rms(z1 - z1t) / 1e3; dfin = rms(z1 - zf) / 1e3; dmax = np.max(np.abs(z1 - zf)) / 1e3
    row = dict(step=n, phi=a["phi"] / phi0, pred=a["pred"], cg=a["cg_its"], rel=a["cg_rel"], sec=a["seconds"], cum=cum,
               err=e, dfin=dfin, dmax=dmax, wil=w, b0=b[0], b1=b[1], b2=b[2],
               cgstep=a.get("cg_step", np.nan), solves=a.get("solves", np.nan), rho=a.get("rho", np.nan),
               cap=a.get("cap_scale", np.nan), model=a.get("model", np.nan), off=off,
               fv=a.get("phi_v", np.nan) / a["phi"] if "phi_v" in a else np.nan,
               fv3=a.get("phi_v3", np.nan) / a["phi_v"] if "phi_v3" in a else np.nan)
    rows.append(row)
    print(f"{n:>6s} {row['phi']:8.2e} {row['pred']:8.1e} {row['cg']:3.0f} {row['rel']:6.2f} {row['cgstep']:5.0f} {row['solves']:4.0f} "
          f"{row['rho']:5.2f} {row['cap']:5.2f} {row['sec']:7.1f} {cum:7.0f} | "
          f"{e:7.1f} km  {dfin:6.2f} / {dmax:6.1f} km | {w:+6.1f} | {off:+7.1f} | {row['fv']:.2f} {row['fv3']:.2f} | " + " ".join(f"{x:.4f}" for x in b))
with open(f"{run}/steps.csv", "w") as f:
    f.write(",".join(rows[0].keys()) + "\n")
    for r in rows:
        f.write(",".join(str(v) for v in r.values()) + "\n")
