"""Test input for poisson_tau.jl: interpolate a height-grid atmosphere (MHS checkpoint layout: /atmosphere/{T, tau500,
Bx_centre, By_centre, Bz_centre, Pgas}, /coords/{x_m, y_m, z_m}) onto surfaces of constant log10 tau500, linear in
log10 tau between the bracketing levels, as an inversion would deliver it.

Output (h5py shapes; Julia sees them reversed): logtau500 (nt, increasing = top first), x_m (nx), y_m (ny),
T [K], Bx, By, Bz [T] (nt, ny, nx); for checking only: ref_z_m, ref_Pgas (nt, ny, nx) of the source atmosphere.

usage: python3 make_tau_input.py IN.h5 OUT.h5 [LOGTAU_TOP=-4.8] [LOGTAU_BOTTOM=1.6] [STEP=0.1]"""
import sys
import numpy as np, h5py

src, out = sys.argv[1], sys.argv[2]
top, bot, step = (float(v) for v in (sys.argv[3:6] + ["-4.8", "1.6", "0.1"][len(sys.argv[3:6]):]))
levels = np.round(np.arange(top, bot + step / 2, step), 6)

with h5py.File(src, "r") as f:
    a = f["atmosphere"]
    lt = np.log10(a["tau500"][()])                                   # (z, y, x), top first: increasing along axis 0
    z = f["coords/z_m"][()]
    fields = {"T": a["T"][()], "Bx": a["Bx_centre"][()], "By": a["By_centre"][()], "Bz": a["Bz_centre"][()],
              "ref_Pgas": a["Pgas"][()]}
    x, y = f["coords/x_m"][()], f["coords/y_m"][()]

nz, ny, nx = lt.shape
res = {k: np.empty((len(levels), ny, nx)) for k in list(fields) + ["ref_z_m"]}
zc = np.broadcast_to(z[:, None, None], lt.shape)
for n, L in enumerate(levels):
    k = (lt < L).sum(axis=0) - 1                                     # last level above L
    if (k < 0).any() or (k >= nz - 1).any():
        sys.exit(f"log10 tau500 = {L} lies outside {((k < 0) | (k >= nz - 1)).sum()} columns; narrow the range")
    a0 = np.take_along_axis(lt, k[None], 0)[0]; a1 = np.take_along_axis(lt, k[None] + 1, 0)[0]
    t = (L - a0) / (a1 - a0)
    for name, arr in list(fields.items()) + [("ref_z_m", zc)]:
        f0 = np.take_along_axis(arr, k[None], 0)[0]; f1 = np.take_along_axis(arr, k[None] + 1, 0)[0]
        if name in ("T", "ref_Pgas"):                                 # positive quantities: interpolate the log
            res[name][n] = np.exp((1 - t) * np.log(f0) + t * np.log(f1))
        else:
            res[name][n] = (1 - t) * f0 + t * f1

with h5py.File(out, "w") as f:
    f["logtau500"] = levels; f["x_m"] = x; f["y_m"] = y
    for name, arr in res.items():
        f[name] = arr
    f.attrs["source"] = src
print(f"wrote {out}: {len(levels)} levels log10 tau500 {levels[0]} .. {levels[-1]}, {ny} x {nx} columns; "
      f"z(tau=1) median {np.median(res['ref_z_m'][np.argmin(abs(levels))]) / 1e3:.1f} km")
