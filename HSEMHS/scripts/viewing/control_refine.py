"""Construction error of the known-equilibrium control (scripts/viewing/make_mhs_control.jl) under refinement of its
fine height step: the differences between consecutive generator outputs on the same τ axis and footprints.

usage: python control_refine.py TAG_A TAG_B [TAG_C ...]   (out/control/{input,truth}_TAG.{nc,h5}, coarse to fine)
Per pair: rms / max |Δ| of the input ln T and each B component (B_x, B_y, B_z), and of the truth ln P, z (km) and
z(τ_LOS = 1) (km), by log τ band. The τ axis, x, y and the control parameters (all but dz_m) must agree.
"""
import sys

import h5py
import numpy as np

bands = [(-8, -4), (-4, -2), (-2, 0), (0, 1.5)]


def load(tag):
    with h5py.File(f"out/control/input_{tag}.nc") as f:          # h5py (x, y, τ) → (τ, y, x)
        lt = f["log10tau500"][:]
        lT = np.log(f["temperature"][:]).transpose(2, 1, 0)
        B = [f[c][:].transpose(2, 1, 0) for c in ("B_x", "B_y", "B_z")]
        x = f["x"][:]; y = f["y"][:]
        par = {k: f.attrs[k] for k in f.attrs if k.startswith("control_") and k != "control_dz_m"}
    with h5py.File(f"out/control/truth_{tag}.h5") as f:          # h5py (τ, y, x)
        P = f["truth/lnP"][:]; z = f["truth/z_m"][:]; z1 = f["truth/z_tau1_m"][:]
        dz = f.attrs.get("control_dz_m", np.nan)
    return dict(lt=lt, lT=lT, B=B, P=P, z=z, z1=z1, dz=dz, x=x, y=y, par=par)


def stat(a):
    return f"{np.sqrt(np.mean(a ** 2)):.2e}/{np.max(np.abs(a)):.2e}"


runs = [load(t) for t in sys.argv[1:]]
for (ta, a), (tb, b) in zip(zip(sys.argv[1:], runs), zip(sys.argv[2:], runs[1:])):
    assert np.allclose(a["lt"], b["lt"]), "τ axes differ"
    assert np.allclose(a["x"], b["x"]) and np.allclose(a["y"], b["y"]), "x or y differ"
    assert a["par"].keys() == b["par"].keys() and all(np.all(a["par"][k] == b["par"][k]) for k in a["par"]), \
        "control parameters other than dz_m differ"
    lt = a["lt"]
    print(f"{ta} (dz {a['dz']} m) → {tb} (dz {b['dz']} m): rms/max |Δ|; z(τ_LOS=1) {stat((a['z1'] - b['z1']) / 1e3)} km")
    for lo, hi in bands:
        k = (lt >= lo) & (lt < hi)
        dB = " ".join(f"{c} {stat(a['B'][i][k] - b['B'][i][k])}" for i, c in enumerate(("Bx", "By", "Bz")))
        print(f"  log τ [{lo}, {hi}): ln T {stat(a['lT'][k] - b['lT'][k])} | {dB} T | "
              f"ln P {stat(a['P'][k] - b['P'][k])} | z {stat((a['z'][k] - b['z'][k]) / 1e3)} km")
