#!/usr/bin/env python3
"""Build the column-mass input of the logm modes (IO/input-mhs-logm.nc) from the Bifrost snapshot.

Column mass per column, from the top: m(z) = P_top/g + integral rho dz (trapezoid; the cube's real rho and P). T (linear in
ln T) and B are interpolated in log10 m to a common uniform axis, which lies inside every column's range. Layout of
input-mhs-ltau.nc, with log10m [m in kg m-2] for log10tau500: h5py (x, y, m), top level first in the last index.
--truth also writes z, P and rho on that axis, for comparing a run with the snapshot.
"""
import argparse, os
import numpy as np, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="IO/BIFROST_en024048_hion_0_504_0_504.nc")
ap.add_argument("--out", default="IO/input-mhs-logm.nc")
ap.add_argument("--truth", default="")
ap.add_argument("--lm-min", type=float, default=-4.9)
ap.add_argument("--lm-max", type=float, default=2.6)
ap.add_argument("--dlm", type=float, default=0.1)
ap.add_argument("--g", type=float, default=274.0)
a = ap.parse_args()

src = h5py.File(a.src, "r")
x = src["x"][:].astype(float); y = src["y"][:].astype(float); z = src["z"][0, :].astype(float)
assert np.all(np.diff(z) < 0), "z must decrease (top first)"
rd = lambda nm: src[nm][0].astype(np.float64)                     # (x, y, z)
rho = rd("density"); P = rd("gas_pressure")
dz = -np.diff(z)
m = np.empty_like(rho)
m[:, :, 0] = P[:, :, 0] / a.g
m[:, :, 1:] = m[:, :, :1] + np.cumsum(0.5 * (rho[:, :, :-1] + rho[:, :, 1:]) * dz, axis=2)
lm = np.log10(m)
nl = int(round((a.lm_max - a.lm_min) / a.dlm)) + 1
axis = a.lm_min + a.dlm * np.arange(nl)
print(f"log10 m: top {lm[:, :, 0].min():.3f} .. {lm[:, :, 0].max():.3f}, bottom {lm[:, :, -1].min():.3f} .. {lm[:, :, -1].max():.3f}; axis {axis[0]:.2f} .. {axis[-1]:.2f}, {nl} levels")
assert axis[0] >= lm[:, :, 0].max() and axis[-1] <= lm[:, :, -1].min(), "the axis must lie inside every column"

def on_axis(v):
    o = np.empty((v.shape[0], v.shape[1], nl))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            o[i, j] = np.interp(axis, lm[i, j], v[i, j])
    return o

T = np.exp(on_axis(np.log(rd("temperature"))))
with h5py.File(a.out, "w") as f:
    d = f.create_dataset("log10m", data=axis); d.attrs["units"] = "kg m-2"
    for nm, v in (("x", x), ("y", y)):
        d = f.create_dataset(nm, data=v); d.attrs["units"] = "m"
    f.create_dataset("temperature", data=T)
    for nm in ("B_x", "B_y", "B_z"):
        f.create_dataset(nm, data=on_axis(rd(nm)))
    f.attrs["gravity_m_s2"] = a.g
    f.attrs["created_by"] = "make_input_mhs_logm.py"
    f.attrs["source"] = os.path.basename(a.src)
print("wrote", a.out)
if a.truth:
    with h5py.File(a.truth, "w") as f:
        f["log10m"] = axis
        f["Pgas"] = np.exp(on_axis(np.log(P)))
        f["rho"] = np.exp(on_axis(np.log(rho)))
        f["z_m"] = on_axis(np.broadcast_to(z, rho.shape))
    print("wrote", a.truth)
