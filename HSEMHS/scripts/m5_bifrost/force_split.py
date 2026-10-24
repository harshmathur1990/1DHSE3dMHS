#!/usr/bin/env python3
"""Helmholtz split of the force residual F = -grad P + J x B - rho g z^ of one 3D state into the part a pressure
change can cancel (a gradient, G phi) and the part no pressure can cancel (the rest, orthogonal to every gradient).

Discretisation (consistent, so the split is exactly orthogonal):
  phi, P, rho, B at cell centres; x, y periodic with centred differences; F_x, F_y at centres;
  F_z on the interior z-faces: -(P_k+1 - P_k)/h - g (rho_k + rho_k+1)/2 + (L_z,k + L_z,k+1)/2;
  weights: cell width w_k / S_k^2 for centre values, centre spacing h / S_f^2 for face values, with S = the layer
  median of rho g (S_f the geometric mean of the two adjacent layers), so every layer counts on a relative scale
  (as Phi_F does) instead of the dense bottom layers deciding the split.
The gradient part solves G^T W G phi = G^T W F (FFT in x, y; tridiagonal in z per mode). Because -grad P is
itself a discrete gradient, the non-gradient part does not depend on P except through rho(T, P): it is the floor
left by any pressure-only (Poisson / FIRTEZ-type) correction that keeps rho fixed.

usage: python3 force_split.py OUT_PREFIX KIND FILE
  KIND = bifrost (native P, rho, B of the Bifrost .nc) | colhse (column_hse output: Pgas, rho from mu_bar; input B)
       | mhs (saved MHS atmosphere: Pgas, rho, B at cell centres)"""
import sys, time
import numpy as np, h5py

MU0, G, MU, KB = 4e-7 * np.pi, 274.0, 1.66053906660e-27, 1.380649e-23
out, kind, fn = sys.argv[1:4]
t0 = time.time()
zyx = lambda d: np.ascontiguousarray(np.transpose(np.asarray(d, dtype=np.float64), (2, 1, 0)))   # (x, y, z) -> (z, y, x)
with h5py.File(fn, "r") as f:
    if kind == "bifrost":
        rd = lambda n: zyx(f[n][0])
        P, rho = rd("gas_pressure"), rd("density"); Bx, By, Bz = rd("B_x"), rd("B_y"), rd("B_z")
        z, x, y = f["z"][0].astype(float), f["x"][()].astype(float), f["y"][()].astype(float)
    elif kind == "colhse":
        P = zyx(f["output/Pgas"]); T = zyx(f["input/temperature"])
        rho = zyx(f["output/rho"]) if "output/rho" in f else P * zyx(f["output/mu_bar"]) * MU / (KB * T); del T
        Bx, By, Bz = zyx(f["input/B_x"]), zyx(f["input/B_y"]), zyx(f["input/B_z"])
        z, x, y = f["input/z"][()], f["input/x"][()], f["input/y"][()]
    else:
        a = f["atmosphere"]
        P, rho = a["Pgas"][()], a["rho"][()]; Bx, By, Bz = a["Bx_centre"][()], a["By_centre"][()], a["Bz_centre"][()]
        z, x, y = f["coords/z_m"][()], f["coords/x_m"][()], f["coords/y_m"][()]
        label_acc = int(f.attrs.get("accepted", -1))
if z[0] < z[-1]:                                                  # store top first, as the rest of the project
    P, rho, Bx, By, Bz = (q[::-1].copy() for q in (P, rho, Bx, By, Bz)); z = z[::-1].copy()
nz, ny, nx = P.shape
dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
if np.max(np.abs(z)) < 1e3:                                       # coordinates in Mm -> m
    z = z * 1e6
if dx < 1e3:
    dx, dy = dx * 1e6, dy * 1e6
print(f"{kind}: {fn}\n  grid {nz}x{ny}x{nx}, z {z[0]/1e6:.3f} .. {z[-1]/1e6:.3f} Mm, dx {dx:.4g}, dy {dy:.4g}; "
      f"median P(z~0) {np.median(P[np.argmin(abs(z))]):.4g}, |B| p99 {np.percentile(np.abs(Bz), 99):.3g} (should be T)", flush=True)

Dx = lambda q: (np.roll(q, -1, 2) - np.roll(q, 1, 2)) / (2 * dx)
Dy = lambda q: (np.roll(q, -1, 1) - np.roll(q, 1, 1)) / (2 * dy)
Dz = lambda q: np.gradient(q, z, axis=0)
Jx = (Dy(Bz) - Dz(By)) / MU0; Jy = (Dz(Bx) - Dx(Bz)) / MU0; Jz = (Dx(By) - Dy(Bx)) / MU0
Lx = Jy * Bz - Jz * By; Ly = Jz * Bx - Jx * Bz; Lz = Jx * By - Jy * Bx
del Jx, Jy, Jz, Bx, By, Bz
h = z[:-1] - z[1:]                                                # centre spacing (> 0, top first)
zf = np.concatenate([[z[0] + h[0] / 2], (z[:-1] + z[1:]) / 2, [z[-1] - h[-1] / 2]])
w = zf[:-1] - zf[1:]                                              # cell widths
Fx = -Dx(P) + Lx; Fy = -Dy(P) + Ly
Fz = -(P[:-1] - P[1:]) / h[:, None, None] - G * 0.5 * (rho[:-1] + rho[1:]) + 0.5 * (Lz[:-1] + Lz[1:])
rhog = rho * G
Sc = np.median(rhog, axis=(1, 2)); Sc = np.where(Sc > 0, Sc, 1.0); Sf = np.sqrt(Sc[:-1] * Sc[1:])
wc, wf = w / Sc ** 2, h / Sf ** 2                                 # centre and face weights
k0 = int(np.argmin(abs(z)))
print(f"  sanity at z = {z[k0]/1e6:.3f} Mm: median |dP/dz|/(rho g) {np.median(np.abs((P[k0-1]-P[k0+1])/(z[k0-1]-z[k0+1]))/rhog[k0]):.3f} (~1 in a photosphere)", flush=True)
del Lx, Ly, Lz

# gradient part: per (ky, kx) mode solve (w_k s2) phi_k + vertical Neumann Laplacian = rhs_k (tridiagonal in z)
kx = 2 * np.pi * np.fft.fftfreq(nx, dx); ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
sx = 1j * np.sin(kx * dx) / dx; sy = 1j * np.sin(ky * dy) / dy      # centred-difference symbols
s2 = (np.abs(sy)[:, None] ** 2 + np.abs(sx)[None, :] ** 2)
Fxh = np.fft.fft2(Fx, axes=(1, 2)); Fyh = np.fft.fft2(Fy, axes=(1, 2)); Fzh = np.fft.fft2(Fz, axes=(1, 2))
# G_z phi at face f = (k, k+1) is (phi_k - phi_k+1)/h_f (= d/dz, z decreasing with k), so G_z^T W_f F_z adds
# +F_f to row k and -F_f to row k+1
rhs = wc[:, None, None] * (np.conj(sx)[None, None, :] * Fxh + np.conj(sy)[None, :, None] * Fyh)
rhs[:-1] += Fzh / Sf[:, None, None] ** 2; rhs[1:] -= Fzh / Sf[:, None, None] ** 2
lo = np.zeros(nz); up = np.zeros(nz); di = np.zeros(nz)
c = 1 / (h * Sf ** 2)
up[:-1] = -c; lo[1:] = -c; di[:-1] += c; di[1:] += c
D = di[:, None, None] + wc[:, None, None] * s2[None]
D[:, 0, 0] += di.max()                                                # (0,0) mode: Neumann null space; handled below
cp = np.empty_like(rhs); dp = np.empty_like(rhs)                 # Thomas
cp[0] = up[0] / D[0]; dp[0] = rhs[0] / D[0]
for k in range(1, nz):
    den = D[k] - lo[k] * cp[k - 1]
    cp[k] = up[k] / den if k < nz - 1 else 0; dp[k] = (rhs[k] - lo[k] * dp[k - 1]) / den
ph = dp
for k in range(nz - 2, -1, -1):
    ph[k] = dp[k] - cp[k] * ph[k + 1]
del cp, rhs
ph[:, 0, 0] = 0.0
Gxh = sx[None, None, :] * ph; Gyh = sy[None, :, None] * ph; Gzh = (ph[:-1] - ph[1:]) / h[:, None, None]
Gzh[:, 0, 0] = Fzh[:, 0, 0]                                       # a horizontally uniform dP(z) removes the mean F_z
Gx = np.fft.ifft2(Gxh, axes=(1, 2)).real; Gy = np.fft.ifft2(Gyh, axes=(1, 2)).real; Gz = np.fft.ifft2(Gzh, axes=(1, 2)).real
del Gxh, Gyh, Gzh, Fxh, Fyh, Fzh, ph
Sx, Sy, Sz = Fx - Gx, Fy - Gy, Fz - Gz
# orthogonality check (weighted inner product <S, G>)
ip = lambda a, b, c, d, e, f2: float((wc[:, None, None] * (a * d + b * e)).sum() + (wf[:, None, None] * c * f2).sum())
nF = ip(Fx, Fy, Fz, Fx, Fy, Fz); nG = ip(Gx, Gy, Gz, Gx, Gy, Gz); nS = ip(Sx, Sy, Sz, Sx, Sy, Sz)
print(f"  ||F||^2 {nF:.4e} = ||G||^2 {nG:.4e} + ||S||^2 {nS:.4e} (sum err {abs(nG + nS - nF)/nF:.2e}); "
      f"<S,G>/||F||^2 {ip(Sx, Sy, Sz, Gx, Gy, Gz)/nF:.2e}", flush=True)

# per-layer table (centre layers; face quantities averaged onto the two adjacent centres)
def tocentre(q):
    c = np.zeros((nz,) + q.shape[1:]); c[:-1] += 0.5 * q; c[1:] += 0.5 * q; c[0] *= 2; c[-1] *= 2; return c
mag = lambda a, b, c: np.sqrt(a ** 2 + b ** 2 + tocentre(c ** 2))
aF, aG, aS = mag(Fx, Fy, Fz), mag(Gx, Gy, Gz), mag(Sx, Sy, Sz)
aSh = np.sqrt(Sx ** 2 + Sy ** 2); aSz = np.sqrt(tocentre(Sz ** 2))
rg = Sc
lev = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, -2.0]
rows = []
print(f"\n  {'z [Mm]':>7} {'med rho g':>10} | RMS/med(rho g):  {'|F|':>9} {'|grad|':>9} {'|non-grad|':>10} | "
      f"non-grad share of |F|^2 | non-grad p50 / p99 of |.|/(rho g): horiz, vert")
for L in lev:
    k = int(np.argmin(abs(z - L * 1e6)))
    r = lambda a: float(np.sqrt(np.mean(a[k] ** 2)) / rg[k])
    sh = float(np.sum(aS[k] ** 2) / np.sum(aF[k] ** 2))
    qh = np.percentile(aSh[k] / rhog[k], [50, 99]); qz = np.percentile(aSz[k] / rhog[k], [50, 99])
    rows.append((z[k] / 1e6, rg[k], r(aF), r(aG), r(aS), sh, *qh, *qz))
    print(f"  {z[k]/1e6:7.3f} {rg[k]:10.3e} | {r(aF):25.3e} {r(aG):9.3e} {r(aS):10.3e} | {sh:23.3f} | "
          f"{qh[0]:.3g} / {qh[1]:.3g}, {qz[0]:.3g} / {qz[1]:.3g}")
for lo_, hi_ in [(-2.5, -0.5), (-0.5, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 15.0)]:
    m = (z >= lo_ * 1e6) & (z < hi_ * 1e6)
    e = lambda a: float(np.sum(w[m, None, None] * (a[m] / rg[m, None, None]) ** 2))
    print(f"  band z in [{lo_:+.1f}, {hi_:+.1f}) Mm: weighted |F|^2 {e(aF):.4e}, |grad|^2 {e(aG):.4e}, |non-grad|^2 {e(aS):.4e}; "
          f"non-grad share {e(aS)/e(aF):.3f}")
# per-layer profiles for every k
prof = np.array([[np.sqrt(np.mean(a[k] ** 2)) / rg[k] for a in (aF, aG, aS)] + [np.sum(aS[k] ** 2) / np.sum(aF[k] ** 2)] for k in range(nz)])
np.savez(out + ".npz", z=z, rhog_median=rg, rms_F=prof[:, 0], rms_grad=prof[:, 1], rms_nongrad=prof[:, 2],
         nongrad_share=prof[:, 3], table=np.array(rows), norms=np.array([nF, nG, nS]),
         nongrad_h_map_z0=(aSh / rhog)[int(np.argmin(abs(z)))].astype(np.float32),
         nongrad_h_map_z05=(aSh / rhog)[int(np.argmin(abs(z - 5e5)))].astype(np.float32))
print(f"\n  wrote {out}.npz in {time.time() - t0:.0f} s")
