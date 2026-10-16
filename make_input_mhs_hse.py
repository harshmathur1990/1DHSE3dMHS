#!/usr/bin/env python3
"""
Build input-mhs-hse.nc: the minimal input file for the column_hse and mhs modes
(HSE_MHS_FINAL_SPEC.md) from the full Bifrost en024048_hion snapshot.

Contents (and nothing else):
  datasets : x [NX], y [NY], z [NZ]  (m; z strictly decreasing, top first)
             temperature, B_x, B_y, B_z  [NX, NY, NZ]  (K, T), float32 as in source
  attributes (global): P_ref + anchor info, g, periodicity, sigma_B model, provenance

Array convention: C/Python order (x, y, z) with T[:, :, 0] = TOP of the atmosphere.
(Julia HDF5.jl sees the reversed shape (z, y, x); permutedims(A, (3, 2, 1)) on load.)

Run:  /Users/harshm/env/bin/python3 make_input_mhs_hse.py
"""
import argparse, os, re, sys, datetime
import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Users/harshm/Documents/WorkRepo/MHS/IO/BIFROST_en024048_hion_0_504_0_504.nc")
ap.add_argument("--out", default="/Users/harshm/Documents/WorkRepo/MHS/IO/input-mhs-hse.nc")
ap.add_argument("--zmin", type=float, default=-0.5e6, help="lowest height kept [m]")
ap.add_argument("--zmax", type=float, default=3.0e6, help="highest height kept [m]")
args = ap.parse_args()

src = h5py.File(args.src, "r")

def dim_names(ds):
    return [ds.dims[i][0].name.split("/")[-1] if len(ds.dims[i]) else f"dim{i}" for i in range(ds.ndim)]

T = src["temperature"]
names = dim_names(T)
print("source temperature shape", T.shape, "dims", names)
assert T.ndim == 4 and T.shape[0] == 1, "expected (time=1, ., ., depth)"

# identify axes
ax = {}
for i, n in enumerate(names):
    if n in ("x",): ax["x"] = i
    elif n in ("y",): ax["y"] = i
    elif n in ("depth", "z"): ax["z"] = i
assert set(ax) == {"x", "y", "z"}, f"could not identify x/y/depth axes from {names}"

x = np.asarray(src["x"][:], dtype=np.float64)
y = np.asarray(src["y"][:], dtype=np.float64)
zsrc = np.asarray(src["z"][:], dtype=np.float64)
z_full = zsrc.reshape(-1) if zsrc.ndim == 1 else zsrc[0, :]
if zsrc.ndim == 2 and zsrc.shape[0] > 1:
    assert np.allclose(zsrc, zsrc[0:1, :], rtol=1e-6), "z differs between rows: not a common height grid"
for v in ("x", "y", "z"):
    print(f"  units {v}:", src[v].attrs.get("units"))

# height crop (keep source top-first order)
assert np.all(np.diff(z_full) < 0), "source z is not strictly decreasing (top first)"
kz = np.where((z_full >= args.zmin - 1.0) & (z_full <= args.zmax + 1.0))[0]
k0, k1 = int(kz[0]), int(kz[-1]) + 1
z = z_full[k0:k1]
print(f"crop: source levels {k0}..{k1-1}  ({k1-k0} levels)  z = {z[0]/1e6:.4f} .. {z[-1]/1e6:.4f} Mm")

def read_xyz(name):
    ds = src[name]
    sl = [slice(None)] * 4
    sl[0] = 0
    sl[ax["z"]] = slice(k0, k1)
    a = ds[tuple(sl)]                      # 3 remaining axes in source order
    rem = [i for i in range(4) if i != 0]  # source axes kept
    order = [rem.index(ax["x"]), rem.index(ax["y"]), rem.index(ax["z"])]
    a = np.transpose(a, order)             # -> (x, y, z)
    return np.ascontiguousarray(a.astype(np.float32, copy=False))

# --- metadata from the embedded FITS header -------------------------------
hdr = T.attrs.get("fits_header", b"")
hdr = hdr.decode() if isinstance(hdr, (bytes, np.bytes_)) else str(hdr)
def card(key):
    m = re.search(r"%-8s=\s*('?)([^/']*)\1" % key, hdr)
    return m.group(2).strip() if m else None
period_x = int(card("PERIOD_X") or 0); period_y = int(card("PERIOD_Y") or 0)
grav, u_l, u_t = card("GRAV"), card("U_L"), card("U_T")
g = -float(grav) * float(u_l) / float(u_t) ** 2 / 100.0 if grav else 274.0   # cgs code units -> m s^-2
print(f"periodic x,y = {period_x},{period_y};  g = {g:.3f} m s^-2 (from GRAV={grav}, U_L={u_l}, U_T={u_t})")

# --- P_ref: median Bifrost gas pressure on the bottom plane of the crop ----

ds = src["gas_pressure"]
sl = [0, slice(None), slice(None), slice(None)]
sl[ax["z"]] = k1 - 1
Pbot = np.asarray(ds[tuple(sl)], dtype=np.float64)
P_ref = float(np.median(Pbot))
spread = float(np.percentile(np.log10(Pbot), 99) - np.percentile(np.log10(Pbot), 1))
print(f"P_ref = median P at z = {z[-1]/1e6:.4f} Mm : {P_ref:.6e} Pa  (p1-p99 spread {spread:.3f} dex)")

# --- write ----------------------------------------------------------------
if os.path.exists(args.out):
    sys.exit(f"refusing to overwrite existing {args.out}")
comp = dict(compression="gzip", compression_opts=4, shuffle=True)
with h5py.File(args.out, "w") as out:
    nx, ny, nz = len(x), len(y), len(z)
    cx, cy, cz = min(64, nx), min(64, ny), min(16, nz)
    for name, arr, units, ln in (("x", x, "m", "horizontal coordinate x"),
                                 ("y", y, "m", "horizontal coordinate y"),
                                 ("z", z, "m", "height, increasing upward; stored top first (z[0] = top)")):
        d = out.create_dataset(name, data=arr)
        d.attrs["units"] = units; d.attrs["long_name"] = ln
        d.make_scale(name)
    for name, units, ln in (("temperature", "K", "gas temperature"),
                            ("B_x", "T", "magnetic field x component"),
                            ("B_y", "T", "magnetic field y component"),
                            ("B_z", "T", "magnetic field z component (positive upward)")):
        a = read_xyz(name)
        assert a.shape == (nx, ny, nz), (name, a.shape)
        assert np.all(np.isfinite(a)), f"non-finite values in {name}"
        if name == "temperature": assert np.all(a > 0), "non-positive temperature"
        d = out.create_dataset(name, data=a, chunks=(cx, cy, cz), **comp)
        d.attrs["units"] = units; d.attrs["long_name"] = ln
        d.attrs["source_variable"] = name
        for i, s in enumerate(("x", "y", "z")):
            d.dims[i].attach_scale(out[s])
        print(f"  wrote {name:12s} {a.shape} min {a.min():.4g} max {a.max():.4g} {units}")
    A = out.attrs
    A["title"] = "Input atmosphere for column_hse and mhs (HSE_MHS_FINAL_SPEC.md)"
    A["array_order"] = "(x, y, z) C order; [:, :, 0] is the top; Julia HDF5.jl sees (z, y, x)"
    A["vertical_order"] = "top_first"
    A["P_ref_Pa"] = P_ref
    A["P_ref_anchor"] = "bottom"
    A["P_ref_z_m"] = float(z[-1])
    A["P_ref_definition"] = "median of Bifrost gas_pressure on the bottom plane of the crop"
    A["P_ref_log10_spread_p1_p99_dex"] = spread
    A["gravity_m_s2"] = g
    A["periodic_x"] = period_x; A["periodic_y"] = period_y
    A["period_x_m"] = float(nx * (x[1] - x[0])); A["period_y_m"] = float(ny * (y[1] - y[0]))
    A["sigma_B_model"] = "per component: sigma = sqrt((0.05*|B|)^2 + (5e-4 T)^2)  [5 % of |B| in quadrature with a 5 G floor]; synthetic default, no observational meaning"
    A["source_file"] = os.path.basename(args.src)
    A["source_run"] = "Bifrost en024048_hion"; A["source_snapshot"] = 385
    A["crop_z_m"] = [float(z[0]), float(z[-1])]
    A["crop_source_levels"] = [k0, k1 - 1]
    A["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    A["created_by"] = "make_input_mhs_hse.py"

# --- read-back check --------------------------------------------------------
with h5py.File(args.out, "r") as f:
    print("\nread-back:", args.out)
    for k in f: print(f"  {k:12s} {f[k].shape} {f[k].dtype} units={f[k].attrs.get('units')}")
    t = f["temperature"]
    print("  z[0], z[-1] (Mm):", f["z"][0] / 1e6, f["z"][-1] / 1e6)
    print("  median T top plane %.0f K, bottom plane %.0f K" % (np.median(t[:, :, 0]), np.median(t[:, :, -1])))
    print("  global attrs:", sorted(f.attrs.keys()))
print("size: %.1f MB" % (os.path.getsize(args.out) / 1e6))
