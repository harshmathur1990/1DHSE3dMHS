"""Response of the reconstructed z(τ=1) to the input errors (round-2 review Q4): rms of z1(perturbed) − z1(unperturbed)
for column HSE and MHS, split by a Gaussian low-pass of σ = 4 px; and the realised statistics of the perturbations
(rms Δ ln T, rms Δ|B|/|B| for |B| > 500 G, 1/e width of the horizontal autocorrelation of Δ ln T at τ = 1).

usage: python perturbation_response.py   (paths below; output kept in perturbation_response.txt)
"""
import h5py, numpy as np
RD="/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_2026-11-10"; IO="/mn/stornext/d9/data/harshm/MHS/IO/tau252"
rms=lambda a: np.sqrt(np.nanmean(a**2))
def z1(run):
    with h5py.File(f"{RD}/{run}.h5") as f: a=f["tau_grid/z_tau1_m"][:]
    return a-a.mean()
with h5py.File(f"{IO}/truth_eos_mu1.0.h5") as f: zt=f["truth/z_tau1_m"][:]
zt-=zt.mean()
n=252; kx=np.fft.fftfreq(n)[:,None]; ky=np.fft.fftfreq(n)[None,:]
def split(a, s=4.0):   # Gaussian low-pass σ = s px
    lo=np.real(np.fft.ifft2(np.fft.fft2(a)*np.exp(-2*(np.pi*s)**2*(kx**2+ky**2)))); return lo, a-lo
with h5py.File(f"{IO}/input_eos_mu1.0.nc") as f:
    lt=f["log10tau500"][:]; k0=int(np.argmin(abs(lt))); T0=f["temperature"][:,:,k0].T
for kind in ("colhse","mhs"):
    base=z1(f"{kind}_eos_mu1.0")
    for tag in ("small","large"):
        rr=[];lo=[];hi=[];ee=[]
        for s in (1,2,3):
            d=z1(f"{kind}_pert_{tag}_s{s}")-base; l,h=split(d)
            rr.append(rms(d)); lo.append(rms(l)); hi.append(rms(h)); ee.append(rms(z1(f"{kind}_pert_{tag}_s{s}")-zt))
        print(f"{kind:6s} {tag:5s}: response rms(z1_pert − z1_unpert) {np.mean(rr)/1e3:5.1f} km (σ>4px part {np.mean(lo)/1e3:4.1f}, <4px {np.mean(hi)/1e3:4.1f}); err vs truth {np.mean(ee)/1e3:.1f}; seeds {', '.join(f'{x/1e3:.1f}' for x in rr)}")
# realised perturbation statistics
for tag in ("small","large"):
    with h5py.File(f"{IO}/perturbed/input_{tag}_s1.nc") as f, h5py.File(f"{IO}/input_eos_mu1.0.nc") as g:
        dT=np.log(f["temperature"][:]/g["temperature"][:])
        B=np.sqrt(sum(g[c][:]**2 for c in ("B_x","B_y","B_z"))); Bp=np.sqrt(sum(f[c][:]**2 for c in ("B_x","B_y","B_z")))
        m=B>0.05
        e=dT[:,:,k0]; 
        # autocorrelation half-width at 1/e horizontally
        F=np.abs(np.fft.fft2(e-e.mean()))**2; ac=np.real(np.fft.ifft2(F)); ac/=ac[0,0]
        r=np.argmax(ac[:,0]<np.exp(-1))
        print(f"{tag}: rms dlnT {rms(dT):.3f}; rms d|B|/|B| (|B|>500G) {rms((Bp-B)[m]/B[m]):.3f}; horizontal ACF 1/e at {r} px")
