"""Why column HSE on column mass has a larger z(τ=1) error than on τ (round-2 review Q2). No new runs: the stored
column-HSE states are re-aligned offline.

usage: python alignment_depth.py   (paths below; prints the table kept in alignment_depth.txt)
The production alignment shifts each column so that P + B²/2μ₀ equals the median total pressure of the bottom level of
the axis (log τ 1.4, truth median z ≈ −83 km; log m 2.6, ≈ −581 km). Here each column is instead aligned at the median
total pressure of other levels (columns whose range does not contain the reference are left out, unlike production,
which extrapolates hydrostatically), then gauged; and, as an oracle, shifted to the truth height of one mass level.
"""
import h5py, numpy as np
RD="/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_2026-11-10"; IO="/mn/stornext/d9/data/harshm/MHS/IO/tau252"
MU0=4e-7*np.pi
rms=lambda a: np.sqrt(np.nanmean(a**2))
with h5py.File(f"{IO}/truth_eos_mu1.0.h5") as f:
    z1T=f["truth/z_tau1_m"][:]; zTt=f["truth/z_m"][:]; uTt=f["truth/lnP"][:]; ltT=f["logtau500"][:]
off=z1T.mean(); z1T=z1T-off
def load(run, grid):
    with h5py.File(f"{RD}/{run}.h5") as f:
        g=f[grid]; z=g["z_m"][:]; P=g["Pgas"][:]; B2=g["Bx"][:]**2+g["By"][:]**2+g["Bz"][:]**2
        lt=g["logtau500"][:] if "logtau500" in g and g["logtau500"].ndim==3 else None
        ax=g["log10m"][:] if grid=="mass_grid" else g["logtau500"][:]
        z1=g["z_tau1_m"][:]
    return z, np.log(P+B2/(2*MU0)), lt, ax, z1
def z1of(z, lt, ax):
    nt,ny,nx=z.shape; out=np.empty((ny,nx))
    for j in range(ny):
        for i in range(nx):
            l=lt[:,j,i] if lt is not None else ax
            out[j,i]=np.interp(0.0,l,z[:,j,i])
    return out
def realign(z, lpt, ref):
    # height where lpt == ref per column (lpt increases with depth/index); columns not reaching: skip (NaN)
    nt,ny,nx=z.shape; zr=np.full((ny,nx),np.nan)
    for j in range(ny):
        for i in range(nx):
            v=lpt[:,j,i]
            if v[-1]>=ref and v[0]<=ref: zr[j,i]=np.interp(ref,v,z[:,j,i])
    return zr
for run,grid in (("colhse_eos_mu1.0","tau_grid"),("colhse_logm_mu1.0","mass_grid")):
    z,lpt,lt,ax,z1o=load(run,grid)
    z1=z1of(z,lt,ax); e=z1-z1.mean()
    print(f"== {run}: stored z1 rms {rms((z1o-z1o.mean())-z1T)/1e3:.1f}, recomputed {rms(e-z1T)/1e3:.1f} km; bottom axis {ax[-1]:.2f}")
    for lev in (ax[-1], ):
        pass
    # reference total pressure at various depths: median of lpt at chosen axis levels
    for k in sorted(set([len(ax)-1]+[int(np.argmin(abs(ax-v))) for v in ((1.4,1.0,0.5,0.0,-1.0) if grid=="tau_grid" else (2.6,2.2,1.9,1.7,1.4,1.0))])):
        ref=np.median(lpt[k]); zr=realign(z,lpt,ref); zz1=z1-zr; ok=np.isfinite(zz1)
        zz1=zz1-np.nanmean(zz1)
        print(f"  align at median Ptot of level {ax[k]:+.2f}: z1 rms {rms((zz1-z1T)[ok])/1e3:5.1f} km, corr {np.corrcoef(zz1[ok],z1T[ok])[0,1]:.3f}, cols {ok.sum()}")
# oracle: shift colhse logm columns so that z at log m=1.0 matches truth -> remaining z1 error
with h5py.File(f"{IO}/truth_logm_mu1.0.h5") as t:
    lm=t["log10m"][:]; zM=t["z_m"][:].transpose(2,1,0)-off
z,lpt,lt,ax,_=load("colhse_logm_mu1.0","mass_grid"); z1=z1of(z,lt,ax)
for v in (2.6,1.7,1.0,-0.5):
    k=int(np.argmin(abs(ax-v))); zz=z1-z[k]+zM[k]; zz-=zz.mean()
    print(f"  oracle shift (truth height of log m {ax[k]:+.2f}): z1 rms {rms(zz-z1T)/1e3:.1f} km")
# truth: horizontal variation of total pressure at fixed height near bottom of each axis
print("truth z of bottom τ level: median %.0f km; of bottom m level: median %.0f km"%(np.median(zTt[-1])/1e3-off/1e3, np.median(zM[-1])/1e3))
