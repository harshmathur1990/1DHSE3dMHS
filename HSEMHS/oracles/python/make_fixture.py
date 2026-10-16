#!/usr/bin/env python3
"""
Write small NetCDF4-style (HDF5 + dimension scales) input files for the HSEMHS reader tests.

usage: make_fixture.py OUTDIR

Files (C order (x, y, z), top-first z, units attributes as in IO/input-mhs-hse.nc):
  good.nc        NX=10, NY=8, NZ=9; T[i,j,k] = 1000 + i + 100 j + 10000 k (K), B in gauss
                 for B_x (to test unit conversion), periodic attributes, period = N*dx
  good4d.nc      same values stored as (time=1, x, y, depth) with a 2D z(time, depth), units in km,
                 periodicity from a fits_header attribute (Bifrost-like)
  bad_z.nc       z not monotone
  bad_dup.nc     x has a duplicated periodic endpoint
  bad_T.nc       one T <= 0
  bad_units.nc   B_y with units "furlong"
"""
import sys, os
import numpy as np
import h5py

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
NX, NY, NZ = 10, 8, 9
dx = 50.0e3
x = np.arange(NX) * dx
y = np.arange(NY) * dx + 3.0e3
z = np.linspace(2.0e6, -0.4e6, NZ) + np.array([0, 5e3, -3e3, 4e3, 0, -2e3, 1e3, 0, 0])
i, j, k = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
T = (1000.0 + i + 100.0 * j + 10000.0 * k).astype(np.float32)
Bx = (1.0 + i + 0.5 * j - 0.25 * k).astype(np.float32)            # gauss
By = (0.01 * (i - j) + 0.002 * k).astype(np.float32)              # tesla
Bz = (0.02 + 0.001 * i * j - 0.003 * k).astype(np.float32)        # tesla


def write3(path, z=z, x=x, T=T, units_by="T", periodic=True, dup=False):
    with h5py.File(path, "w") as f:
        xs = x.copy()
        if dup:
            xs = np.append(x[:-1], x[0] + NX * dx)  # last point == first + period
        for name, arr in (("x", xs), ("y", y), ("z", z)):
            d = f.create_dataset(name, data=arr)
            d.attrs["units"] = "m"
            d.make_scale(name)
        for name, arr, u in (("temperature", T, "K"), ("B_x", Bx, "G"), ("B_y", By, units_by), ("B_z", Bz, "T")):
            d = f.create_dataset(name, data=arr)
            d.attrs["units"] = u
            for a, s in enumerate(("x", "y", "z")):
                d.dims[a].attach_scale(f[s])
        f.attrs["periodic_x"] = 1 if periodic else 0
        f.attrs["periodic_y"] = 1 if periodic else 0
        f.attrs["period_x_m"] = NX * dx
        f.attrs["period_y_m"] = NY * dx
        f.attrs["vertical_order"] = "top_first"
        f.attrs["P_ref_Pa"] = 1.0e5
        f.attrs["gravity_m_s2"] = 274.0


write3(os.path.join(out, "good.nc"))
zbad = z.copy(); zbad[4] = zbad[2]
write3(os.path.join(out, "bad_z.nc"), z=zbad)
write3(os.path.join(out, "bad_dup.nc"), dup=True)
Tb = T.copy(); Tb[3, 2, 1] = 0.0
write3(os.path.join(out, "bad_T.nc"), T=Tb)
write3(os.path.join(out, "bad_units.nc"), units_by="furlong")

# Bifrost-like 4D layout
with h5py.File(os.path.join(out, "good4d.nc"), "w") as f:
    sn = f.create_dataset("snapshot_number", data=np.array([385], dtype=np.int32)); sn.make_scale("snapshot_number")
    for name, arr in (("x", x), ("y", y)):
        d = f.create_dataset(name, data=arr); d.attrs["units"] = "m"; d.make_scale(name)
    dep = f.create_dataset("depth", data=np.arange(NZ, dtype=np.float32)); dep.make_scale("depth")
    zz = f.create_dataset("z", data=(z / 1e3)[None, :]); zz.attrs["units"] = "km"
    zz.dims[0].attach_scale(sn); zz.dims[1].attach_scale(dep)
    hdr = "SIMPLE  =                    T / x\nPERIOD_X=                    1 / periodic\nPERIOD_Y=                    1 / periodic\n"
    for name, arr, u in (("temperature", T, "K"), ("B_x", Bx, "G"), ("B_y", By, "T"), ("B_z", Bz, "T")):
        d = f.create_dataset(name, data=arr[None, ...]); d.attrs["units"] = u
        d.dims[0].attach_scale(sn); d.dims[1].attach_scale(f["x"]); d.dims[2].attach_scale(f["y"]); d.dims[3].attach_scale(dep)
        if name == "temperature":
            d.attrs["fits_header"] = hdr
print("fixtures written to", out)
