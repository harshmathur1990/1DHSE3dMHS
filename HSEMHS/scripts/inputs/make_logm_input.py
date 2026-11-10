#!/usr/bin/env python3
"""Build the column-mass input of the logm modes (IO/input-mhs-logm.nc) from the Bifrost snapshot.

Column mass per column, from the top: m(z) = P_top/g + integral rho dz (trapezoid; the cube's real rho and P). T (linear in
ln T) and B are interpolated in log10 m to a common uniform axis, which lies inside every column's range. Layout of
input-mhs-ltau.nc, with log10m [m in kg m-2] for log10tau500: h5py (x, y, m), top level first in the last index.
--truth also writes z, P and rho on that axis, for comparing a run with the snapshot.
--multi3d DIR builds the input on the grid of a MULTI3D run of the same snapshot instead (DIR/mesh, DIR/atm3d; as
make_tau_input.jl): the run's nz top levels and its columns, a 2-fold subsample of the netCDF found by matching atm3d's
T (exactly one subsampling offset, with x and y flipped or swapped, must match to 1e-5), ordered with x and y increasing.
T is atm3d's; B, P and rho come from the netCDF.
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
ap.add_argument("--multi3d", default="")
a = ap.parse_args()


def multi3d_grid(d, Tb):
    """(nz, column index arrays cx, cy into the netCDF, the run's T on them (x, y, z)) of the MULTI3D run in d."""
    tok = open(os.path.join(d, "mesh")).read().split()
    nx = int(tok[0]); i = 1 + nx; ny = int(tok[i]); i += 1 + ny; nz = int(tok[i])
    zm = np.array(tok[i + 1:i + 1 + nz], float) / 100
    Tm = np.fromfile(os.path.join(d, "atm3d"), dtype="float32").reshape((nx, ny, nz, 12), order="F")[:, :, :, 1]
    hits = []
    for swap in (False, True):
        for ox in (0, 1):
            for oy in (0, 1):
                for fx in (False, True):
                    for fy in (False, True):
                        MX, MY = (Tb.shape[1], Tb.shape[0]) if swap else Tb.shape[:2]
                        if (2 * nx, 2 * ny) != (MX, MY):
                            continue
                        sx = np.arange(ox, MX, 2); sy = np.arange(oy, MY, 2)
                        sx = sx[::-1] if fx else sx; sy = sy[::-1] if fy else sy
                        ok = True
                        for k in (nz // 4, nz // 2, 3 * nz // 4):
                            tb = Tb[:, :, k][np.ix_(sy, sx)].T if swap else Tb[:, :, k][np.ix_(sx, sy)]
                            ok &= np.max(np.abs(Tm[:, :, k] / tb - 1)) < 1e-5
                        if ok:
                            hits.append((sx, sy, swap, f"swap {swap}, offsets ({ox}, {oy}), flips ({fx}, {fy})"))
    assert len(hits) == 1, f"{len(hits)} subsamplings of the netCDF match atm3d's T (need exactly 1)"
    sx, sy, swap, desc = hits[0]
    assert not swap, "the run's x is the netCDF's y: B would need its components exchanged (not implemented)"
    print("registration:", desc)
    px = np.argsort(sx); py = np.argsort(sy)
    return nz, zm, sx[px], sy[py], Tm[np.ix_(px, py)]


src = h5py.File(a.src, "r")
x = src["x"][:].astype(float); y = src["y"][:].astype(float); z = src["z"][0, :].astype(float)
assert np.all(np.diff(z) < 0), "z must decrease (top first)"
rd = lambda nm: src[nm][0].astype(np.float64)                     # (x, y, z)
Tm = None
if a.multi3d:
    nz, zm, cx, cy, Tm = multi3d_grid(a.multi3d, rd("temperature"))
    assert np.allclose(z[:nz], zm, rtol=0, atol=1e-5 * np.abs(zm).max()), "the run's z is not the netCDF's top nz levels"
    full = rd
    rd = lambda nm: full(nm)[np.ix_(cx, cy)][:, :, :nz]
    x = x[cx]; y = y[cy]; z = z[:nz]
    print(f"grid {len(x)} x {len(y)} x {nz}, dx {(x[1] - x[0]) / 1e3:.2f} km")
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

T = np.exp(on_axis(np.log(rd("temperature") if Tm is None else Tm.astype(np.float64))))
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
    if a.multi3d:
        f.attrs["grid"] = f"MULTI3D run {os.path.abspath(a.multi3d)} (T from atm3d); B, P, rho from the netCDF columns x[{cx[0]}:{cx[-1]}:2], y[{cy[0]}:{cy[-1]}:2] (0-based)"
print("wrote", a.out)
if a.truth:
    with h5py.File(a.truth, "w") as f:
        f["log10m"] = axis
        f["Pgas"] = np.exp(on_axis(np.log(P)))
        f["rho"] = np.exp(on_axis(np.log(rho)))
        f["z_m"] = on_axis(np.broadcast_to(z, rho.shape))
    print("wrote", a.truth)
