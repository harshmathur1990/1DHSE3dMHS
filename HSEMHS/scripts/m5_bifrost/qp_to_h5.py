#!/usr/bin/env python3
"""Write a face field saved by data_qp.py / data_lp.py (.npz, flat face order: x-, y-, z-faces, each C-order (x, y, z))
as HDF5 in Julia layout: Bx, By (nx, ny, nz), Bz (nx, ny, nz+1; index 1 = top boundary face), plus the support mask
and the per-face bounds when present. usage: python3 qp_to_h5.py FIELD.npz OBS.h5 OUT.h5"""
import sys, numpy as np, h5py
src, obs, out = sys.argv[1:4]
d = np.load(src)
with h5py.File(obs, "r") as f:
    nx, ny, nz = f["centre_obs1"].shape[::-1]
n = nx * ny * nz; B = d["B_face"]
parts = {"Bx": B[:n].reshape(nx, ny, nz), "By": B[n:2 * n].reshape(nx, ny, nz), "Bz": B[2 * n:].reshape(nx, ny, nz + 1)}
with h5py.File(out, "w") as f:
    for k, v in parts.items():
        f[k] = np.ascontiguousarray(v.T)                     # h5py reverses on write as Julia reverses on read
    for k in ("support", "ub"):
        if k in d.files:
            a = d[k].astype(np.float64)
            for q, (lo, hi, shp) in enumerate(((0, n, (nx, ny, nz)), (n, 2 * n, (nx, ny, nz)), (2 * n, a.size, (nx, ny, nz + 1))), 1):
                f[f"{k}{q}"] = np.ascontiguousarray(a[lo:hi].reshape(shp).T)
print(f"wrote {out}")
