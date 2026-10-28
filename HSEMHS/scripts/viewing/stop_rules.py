"""Stop rules replayed on scripts/viewing/measure_steps.jl snapshots (design/tau/SOLVER_TIME_DESIGN.md §3).

usage: python stop_rules.py TRUTH.h5 INPUT.nc RUNDIR [RUNDIR ...]
Per run and rule: the first step at which the rule holds, the cumulative solve time there, the z(τ_LOS=1) rms error
against the truth, its rms/max distance from the final state and the strong − quiet height difference. Rules:
  pred<X : predicted relative decrease < X on 2 consecutive steps (D1);
  dz<X   : rms change of z(τ_LOS=1) per step < X km on 2 consecutive steps (needs every step saved: steps ≤ 40).
"""
import glob
import sys

import h5py
import numpy as np

truth, inp, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
with h5py.File(truth) as f:
    lt = f["logtau500"][:]
    z1t = f["truth/z_tau1_m"][:]
k0 = int(np.argmin(abs(lt)))
with h5py.File(inp) as f:
    B1 = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
strong, quiet = B1 > 0.05, B1 < 0.01
z1t = z1t - z1t.mean()


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


for run in runs:
    hs = [h5py.File(p) for p in sorted(glob.glob(f"{run}/snap_r*.h5"), key=lambda p: h5py.File(p).attrs["i0"])]
    kc = int(hs[0].attrs["kc"]) - 1
    steps = sorted([int(n[1:]) for n in hs[0].keys() if n.startswith("s")])
    steps = [s for s in steps if s <= 40]

    def z1(name):
        z = np.concatenate([h[name]["z"][:] for h in hs], axis=0).transpose(2, 1, 0)[k0 - kc]
        return z - z.mean()

    Z = {s: z1(f"s{s}") for s in steps}
    zf = z1("final")
    cum = np.cumsum([0.0] + [hs[0][f"s{s}"].attrs["seconds"] for s in steps[1:]])
    pred = {s: hs[0][f"s{s}"].attrs["pred"] for s in steps[1:]}
    dz = {s: rms(Z[s] - Z[s - 1]) / 1e3 for s in steps[1:]}

    def report(rule, s):
        if s is None:
            print(f"  {rule:10s}   not met by step {steps[-1]}")
            return
        i = steps.index(s); z = Z[s]
        w = z[strong].mean() / 1e3 - z[quiet].mean() / 1e3
        print(f"  {rule:10s} step {s:3d}  {cum[i]:6.0f} s | err {rms(z - z1t) / 1e3:5.1f} km | from final {rms(z - zf) / 1e3:5.2f} / "
              f"{np.max(np.abs(z - zf)) / 1e3:5.1f} km | str−quiet {w:+6.1f}")

    def first(cond):
        for s in steps[2:]:
            if cond(s) and cond(s - 1):
                return s
        return None

    print(f"{run}: {len(steps) - 1} steps saved, final err {rms(zf - z1t) / 1e3:.1f} km")
    for x in (1e-2, 3e-3, 1e-3):
        report(f"pred<{x:g}", first(lambda s: pred[s] < x))
    for x in (1.0, 0.5, 0.25):
        report(f"dz<{x:g}km", first(lambda s: dz[s] < x))
