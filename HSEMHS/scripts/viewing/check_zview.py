"""Check the viewing angle in the height modes (column-hse-z, mhs-poisson-z) on the full cube.

1. P and rho do not depend on [view].
2. plane_parallel: tau_LOS = tau_vertical / mu at every node.
3. inclined: tau500 at sample nodes against an independent integration of chi along the slanted ray, and z(tau=1) on
   footprints against the tau = 1 crossing of the interpolated tau field along the ray x = xi + t_x z. This follows the
   exact ray, unlike the code (which interpolates tau back from the footprint columns), so it agrees only to about 1e-4
   at the median, with outliers of ~10 % in tau and ~2 km in z(tau=1) at steep horizontal gradients.
4. inclined: an exact re-implementation of the method (agrees to round-off).
Inputs: out/modes/zview_{col_mu1, col_mu05_pp, col_mu05_inc, poi_mu1, poi_mu05_inc}.h5 (configs
config/modes/zview_*.toml on the run of 2 Nov 2026: the mode configs plus [view]). Run from HSEMHS/."""
import h5py
import numpy as np

O = "out/modes/zview_"


def col(name):
    f = h5py.File(O + name + ".h5")
    o = f["output"]
    return dict(f=f, P=o["Pgas"][:], rho=o["rho"][:], tau=o["tau500"][:], chi=o["chi500"][:],
                zt1=f["diagnostics/z_tau500_1"][:])


def rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


print("== column-hse-z: layout")
f = h5py.File(O + "col_mu1.h5")
for g in ("input", "output", "diagnostics"):
    print(" ", g, {k: f[g][k].shape for k in f[g] if isinstance(f[g][k], h5py.Dataset)})
print("  attrs", {k: f.attrs[k] for k in f.attrs})

a, b, c = col("col_mu1"), col("col_mu05_pp"), col("col_mu05_inc")
print("== 1. P, rho independent of the view (column HSE)")
for n, x in (("mu 0.5 pp", b), ("mu 0.5 inclined", c)):
    print(f"  {n}: max rel dP {rel(x['P'], a['P']):.2e}, drho {rel(x['rho'], a['rho']):.2e}, dchi {rel(x['chi'], a['chi']):.2e}")

print("== 2. plane-parallel: tau(mu 0.5) / tau(mu 1)")
r = b["tau"] / a["tau"]
print(f"  ratio min/max {r.min():.12f} / {r.max():.12f} (expected 2)")

# grid: h5py order (x, y, z) as the input; z top first
inp = h5py.File("../IO/input-mhs-hse.nc")
x = inp["x"][:].astype(float)
y = inp["y"][:].astype(float)
z = inp["z"][:].astype(float)
for v, nm in ((x, "x"), (y, "y"), (z, "z")):
    u = inp[nm].attrs.get("units", b"m")
    u = u.decode() if isinstance(u, bytes) else u
    s = {"m": 1.0, "km": 1e3, "cm": 1e-2, "Mm": 1e6}[u]
    v *= s
dx = x[1] - x[0]
NX = len(x)
mu = 0.5
tx = np.tan(np.arccos(mu))
print(f"== 3. inclined: grid {len(x)}x{len(y)}x{len(z)}, dx {dx:.1f} m, z {z[0]/1e3:.0f} … {z[-1]/1e3:.0f} km, t_x {tx:.4f}")
print(f"  attrs of inclined file: { {k: c['f'].attrs[k] for k in c['f'].attrs if 'view' in k or 'tau' in k} }")
tau, chi = c["tau"], c["chi"]   # (x, y, z)


def xinterp(field_xz_row, xpos):
    """periodic linear interpolation in x of field[:, j, k] at position xpos."""
    s = (xpos - x[0]) / dx
    i0 = int(np.floor(s))
    w = s - i0
    return (1 - w) * field_xz_row[i0 % NX] + w * field_xz_row[(i0 + 1) % NX]


rng = np.random.default_rng(1)
errs = []
for _ in range(400):
    i, j = rng.integers(NX), rng.integers(len(y))
    k = rng.integers(5, len(z))
    tv = tau[i, j, k]
    if not (1e-3 < tv < 30):
        continue
    xs = x[i] + tx * (z[:k + 1] - z[k])                      # ray through node (i, j, k), levels 0..k
    ch = np.array([xinterp(chi[:, j, m], xs[m]) for m in range(k + 1)])
    t_top = xinterp(tau[:, j, 0], xs[0])                     # tau at the ray's top-level crossing
    seg = 0.5 * (ch[1:] + ch[:-1]) * (z[:-1][:k] - z[1:k + 1]) / mu
    t_ind = t_top + seg.sum()
    errs.append((tv, t_ind / tv - 1))
e = np.array(errs)
for lo, hi in ((1e-3, 1e-1), (1e-1, 1), (1, 30)):
    m = (e[:, 0] >= lo) & (e[:, 0] < hi)
    if m.any():
        print(f"  tau in [{lo:g}, {hi:g}): n {m.sum():3d}, independent/code − 1: median {np.median(e[m,1]):+.2e}, max |.| {np.max(np.abs(e[m,1])):.2e}")

# z(tau=1) on footprints: along x = xi + t z, interpolate log tau horizontally at each level, find the crossing
zt = c["zt1"]
print(f"  z_tau500_1 shape {zt.shape}")
d = []
for _ in range(300):
    i, j = rng.integers(NX), rng.integers(len(y))
    xs = x[i] + tx * z
    lt = np.array([np.log(xinterp(tau[:, j, m], xs[m])) for m in range(len(z))])
    kk = np.argmax(lt > 0)
    if kk == 0:
        continue
    w = (0 - lt[kk - 1]) / (lt[kk] - lt[kk - 1])
    zc = z[kk - 1] + w * (z[kk] - z[kk - 1])
    d.append(zc - zt[i, j])
d = np.array(d)
print(f"  z(tau=1) on footprints: independent − code: median {np.median(d):+.1f} m, p99 |.| {np.percentile(np.abs(d), 99):.1f} m, max {np.abs(d).max():.1f} m")

print("== z(tau=1) maps (km): p1 / p50 / p99, mean")
for n, xx in (("mu 1", a), ("mu 0.5 pp", b), ("mu 0.5 inclined", c)):
    q = xx["zt1"] / 1e3
    print(f"  {n:16s} {np.percentile(q, [1, 50, 99]).round(1)}  mean {q.mean():.1f}  std {q.std():.1f}")

print("== mhs-poisson-z: P independent of the view; tau")
p1 = h5py.File(O + "poi_mu1.h5")
p5 = h5py.File(O + "poi_mu05_inc.h5")
print("  attrs mu 0.5:", {k: p5.attrs[k] for k in p5.attrs if "view" in k or k in ("steps", "stop_reason")})
for k in ("Pgas", "rho"):
    print(f"  {k}: max rel diff {rel(p5['atmosphere'][k][:], p1['atmosphere'][k][:]):.2e}")
zp1 = p1["atmosphere/z_tau1_m"][:] / 1e3
zp5 = p5["atmosphere/z_tau1_m"][:] / 1e3
print(f"  z(tau=1) mu 1: {np.percentile(zp1, [1, 50, 99]).round(1)} mean {zp1.mean():.1f}; mu 0.5 inclined: {np.percentile(zp5, [1, 50, 99]).round(1)} mean {zp5.mean():.1f}")

# 4. exact re-implementation of the documented inclined method (src/ViewTau.jl) on sample rows (t_y = 0 here):
#    shear chi to the footprints (periodic linear in x), trapezoid along each footprint column with dz/mu and the
#    exp_tail top closure (src/Tau.jl), z(tau=1) linear in log10 tau, shift tau back to the grid points.
print("== 4. inclined: exact re-implementation of the method")
chi_d = c["f"]["output/chi500"]
tau_d = c["f"]["output/tau500"]
NZ = len(z)


def shift(row, s):
    ii = np.arange(NX) + s
    i0 = np.floor(ii).astype(int)
    w = ii - i0
    return (1 - w) * row[i0 % NX] + w * row[(i0 + 1) % NX]


dtau, dzt = [], []
for j in (0, 77, 150, 251, 333, 420, 503):
    C = chi_d[:, j, :]
    Ct = np.stack([shift(C[:, k], tx * z[k] / dx) for k in range(NZ)], 1)
    T = np.zeros_like(Ct)
    for i in range(NX):
        cc = Ct[i]
        t0 = 0.5 * cc[0] * (z[0] - z[1])
        if cc[0] > 0 and cc[1] > cc[0]:
            Hc = (z[0] - z[1]) / np.log(cc[1] / cc[0])
            if 0 < Hc < 5e6:
                t0 = cc[0] * Hc
        T[i] = np.concatenate([[t0], t0 + np.cumsum(0.5 * (cc[1:] + cc[:-1]) * (z[:-1] - z[1:]))]) / mu
        lt10 = np.log10(T[i])
        k = np.argmax(lt10 >= 0)
        if k > 0:
            w = (0 - lt10[k - 1]) / (lt10[k] - lt10[k - 1])
            dzt.append(z[k - 1] + w * (z[k] - z[k - 1]) - zt[i, j])
    Tg = np.stack([shift(T[:, k], -tx * z[k] / dx) for k in range(NZ)], 1)
    code = tau_d[:, j, :]
    m = code > 1e-6
    dtau.append(np.max(np.abs(Tg[m] / code[m] - 1)))
print(f"  tau at grid points, reproduced/code − 1: max {max(dtau):.1e}; z(tau=1) on footprints: max |Δ| {np.max(np.abs(dzt)):.1e} m")
