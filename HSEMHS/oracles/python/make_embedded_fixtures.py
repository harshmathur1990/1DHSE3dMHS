#!/usr/bin/env python3
"""
Small inputs for the embedded-API tests (test/test_embedded.jl, test/mpi/mpi_embedded.jl): windows cropped from the
production inputs, so the solvers see realistic T and B. The crop is not periodic; the solvers still treat x and y as
periodic, which is fine for numerical regression (the numbers are compared, not judged).

usage: make_embedded_fixtures.py IO_DIR OUTDIR

Writes:
  z_24x18.nc   height grid, NetCDF-style like IO/input-mhs-hse.nc, x 200:224, y 300:318, every level
  z_11x12.nc   same layout, x 100:111, y 50:62 (uneven splits over 2-4 ranks)
  ltau_24x18.h5  tau grid, IO/input-mhs-ltau.nc layout plus x, y: log10tau500 (nt), x, y (m), temperature and
                 B_x/B_y/B_z as h5py (x, y, tau)
  ltau_11x12.h5  same, x 100:111, y 50:62
"""
import sys, os
import numpy as np
import h5py

io, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def crop_z(name, xs, ys):
    with h5py.File(os.path.join(io, "input-mhs-hse.nc"), "r") as s, h5py.File(os.path.join(out, name), "w") as f:
        x = s["x"][xs]; y = s["y"][ys]; z = s["z"][:]
        dx = x[1] - x[0]; dy = y[1] - y[0]
        for nm, arr in (("x", x), ("y", y), ("z", z)):
            d = f.create_dataset(nm, data=arr)
            d.attrs["units"] = s[nm].attrs["units"]
            d.make_scale(nm)
        for nm in ("temperature", "B_x", "B_y", "B_z"):
            d = f.create_dataset(nm, data=s[nm][xs, ys, :])
            d.attrs["units"] = s[nm].attrs["units"]
            for a, sc in enumerate(("x", "y", "z")):
                d.dims[a].attach_scale(f[sc])
        f.attrs["periodic_x"] = 1
        f.attrs["periodic_y"] = 1
        f.attrs["period_x_m"] = len(x) * dx
        f.attrs["period_y_m"] = len(y) * dy
        f.attrs["vertical_order"] = "top_first"
        f.attrs["P_ref_Pa"] = s.attrs["P_ref_Pa"]
        f.attrs["gravity_m_s2"] = s.attrs["gravity_m_s2"]
        f.attrs["source"] = "input-mhs-hse.nc window x %s, y %s" % (xs, ys)


def crop_tau(name, xs, ys):
    with h5py.File(os.path.join(io, "input-mhs-ltau.nc"), "r") as s, h5py.File(os.path.join(out, name), "w") as f:
        f["log10tau500"] = s["log10tau500"][:]
        nx = xs.stop - xs.start; ny = ys.stop - ys.start
        dx = 2.3999976e7 / 504
        f["x"] = (np.arange(nx) + xs.start) * dx
        f["y"] = (np.arange(ny) + ys.start) * dx
        for nm in ("temperature", "B_x", "B_y", "B_z"):
            f[nm] = s[nm][xs, ys, :]
        f.attrs["source"] = "input-mhs-ltau.nc window x %s, y %s" % (xs, ys)


for tag, xs, ys in (("24x18", slice(200, 224), slice(300, 318)), ("11x12", slice(100, 111), slice(50, 62))):
    crop_z("z_%s.nc" % tag, xs, ys)
    crop_tau("ltau_%s.h5" % tag, xs, ys)
print("wrote", sorted(os.listdir(out)))
