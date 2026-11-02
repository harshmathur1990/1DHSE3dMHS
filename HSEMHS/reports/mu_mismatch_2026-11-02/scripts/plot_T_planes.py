"""T maps on horizontal planes z = 0, 300, 1200 km from an inclined mhs-poisson-ltau output, on local solar x, y.

Each ray is interpolated in ln T to height z (1-D along the ray); its value sits at x = xi + t_x z, y = eta + t_y z, so the
map at fixed z is the footprint map shifted by t z (the same for all rays). The shift is undone with a periodic linear
sub-pixel roll (t_y = 0 here, so in x only). Run from HSEMHS/ on the cluster (this is the code used on 2 Nov 2026)."""
import copy
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = h5py.File("out/modes/halpha_mu05_inclined.h5")
g = f["tau_grid"]
z = g["z_m"][:]
lnT = np.log(g["T"][:])
lt = g["logtau500"][:]
x = f["coords/x_m"][:]
y = f["coords/y_m"][:]
dx = x[1] - x[0]
tx = f.attrs["view_tx"]
J, I = np.indices(z.shape[1:])


def on_ray(h):
    k = (z >= h).sum(0) - 1                      # z decreases with the tau index
    ok = (k >= 0) & (k < z.shape[0] - 1)
    kk = np.clip(k, 0, z.shape[0] - 2)
    z0 = z[kk, J, I]
    z1 = z[kk + 1, J, I]
    w = (h - z0) / (z1 - z0)
    v = np.exp((1 - w) * lnT[kk, J, I] + w * lnT[kk + 1, J, I])
    return np.where(ok, v, np.nan)               # NaN: the ray does not reach height h


def to_grid(m, h):
    # value of ray xi sits at x = xi + tx h; grid value at x_i = ray value at xi = x_i - tx h (periodic, linear)
    s = tx * h / dx
    n = int(np.floor(s))
    fr = s - n
    return (1 - fr) * np.roll(m, n, axis=1) + fr * np.roll(m, n + 1, axis=1)


hs = [0.0, 300e3, 1200e3]
maps = [to_grid(on_ray(h), h) for h in hs]
cm = copy.copy(plt.get_cmap("Oranges"))
cm.set_bad("#7a7a7a")
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), dpi=140, constrained_layout=True)
ext = [x[0] / 1e6, x[-1] / 1e6, y[0] / 1e6, y[-1] / 1e6]
for ax, m, h in zip(axs, maps, hs):
    lo, hi = np.nanpercentile(m, [1, 99])
    im = ax.imshow(np.ma.masked_invalid(m), origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("T (K)")
    ax.set_title(f"z = {h/1e3:.0f} km   (ray shift {tx*h/1e3:.0f} km in x, undone)", fontsize=10)
    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("y (Mm)")
fig.suptitle("Temperature on horizontal planes, local solar x, y  (mhs-poisson-ltau, μ = 0.5 inclined, φ = 0°; "
             "z gauge: mean z(τ_LOS=1) = 0; grey: no data; colour 1–99 % per panel)", fontsize=10)
fig.savefig("out/figs/halpha_mu05_T_z0_300_1200.png")
