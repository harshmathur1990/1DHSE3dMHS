"""Input-error test: column HSE and MHS on the perturbed μ = 1 inputs (make_perturbed_inputs.py) against the unperturbed truth.

usage: python perturbed_metrics.py RUNDIR IODIR
Writes RUNDIR/perturbed_metrics.txt: per run the status, steps, the z(τ=1) rms error and correlation, the depression
(strong/quiet mask from the unperturbed input, so that every row selects the same columns), rms Δ ln P over log τ ≥ −4,
and the median horizontal and vertical force imbalance over log τ −3 … 0 from diag_<run>.h5 when present; then the mean
and spread over the three realisations, next to the unperturbed runs.
"""
import sys

import h5py
import numpy as np

RD, IO = sys.argv[1:3]
out = open(f"{RD}/perturbed_metrics.txt", "w")


def p(*a):
    print(*a); print(*a, file=out)


def rms(a):
    a = a[np.isfinite(a)]
    return np.sqrt(np.mean(a ** 2))


with h5py.File(f"{IO}/truth_eos_mu1.0.h5") as f:
    lt = f["logtau500"][:]; z1t = f["truth/z_tau1_m"][:]; ut = f["truth/lnP"][:]
z1t = z1t - z1t.mean()
with h5py.File(f"{IO}/input_eos_mu1.0.nc") as f:
    k0 = int(np.argmin(abs(lt)))
    B = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
s, q = B > 0.05, B < 0.01
band = lt >= -4; ph = (lt >= -3) & (lt <= 0)
dT = (z1t[s].mean() - z1t[q].mean()) / 1e3


def row(name):
    try:
        f = h5py.File(f"{RD}/{name}.h5")
    except OSError:
        return None
    g = f["tau_grid"]; A = dict(f.attrs)
    z1 = g["z_tau1_m"][:]; z1 = z1 - z1.mean(); u = np.log(g["Pgas"][:]); f.close()
    st = A.get("status", b"-"); st = st.decode() if isinstance(st, bytes) else st
    r = dict(status=st, steps=int(A.get("steps", 0)), err=rms(z1 - z1t) / 1e3, corr=np.corrcoef(z1.ravel(), z1t.ravel())[0, 1],
             dep=(z1[s].mean() - z1[q].mean()) / 1e3, dlnp=rms((u - ut)[band]), fh=np.nan, fz=np.nan)
    try:
        with h5py.File(f"{RD}/diag_{name}.h5") as g:
            r["fh"] = np.median(g["fh"][:][ph]); r["fz"] = np.median(np.abs(g["fz"][:][ph]))
    except OSError:
        pass
    return r


p(f"truth depression {dT:+.1f} km; mask from the unperturbed input ({s.sum()} strong, {q.sum()} quiet)")
p(f"{'run':24s} {'status':10s} {'steps':>5s} | {'z1 rms':>6s} {'corr':>5s} {'dep':>6s} {'frac':>5s} | {'ΔlnP':>5s} | {'F_h':>6s} {'F_z':>6s}")
for kind in ("colhse", "mhs"):
    allr = {"none": [row(f"{kind}_eos_mu1.0")]}
    for tag in ("small", "large"):
        allr[tag] = [row(f"{kind}_pert_{tag}_s{k}") for k in (1, 2, 3)]
    for tag, rs in allr.items():
        for k, r in enumerate(rs):
            if r is None:
                continue
            nm = f"{kind} {tag}" + ("" if tag == "none" else f" s{k + 1}")
            p(f"{nm:24s} {r['status']:10s} {r['steps']:5d} | {r['err']:6.1f} {r['corr']:5.3f} {r['dep']:+6.1f} {r['dep'] / dT * 100:4.0f}% | "
              f"{r['dlnp']:5.3f} | {r['fh']:6.3f} {r['fz']:6.3f}")
        ok = [r for r in rs if r is not None]
        if tag != "none" and ok:
            m = lambda key: (np.mean([r[key] for r in ok]), np.std([r[key] for r in ok]))
            p(f"  {kind} {tag} mean±std: z1 rms {m('err')[0]:.1f}±{m('err')[1]:.1f}, corr {m('corr')[0]:.3f}±{m('corr')[1]:.3f}, "
              f"dep {m('dep')[0]:+.1f}±{m('dep')[1]:.1f} ({m('dep')[0] / dT * 100:.0f} %), ΔlnP {m('dlnp')[0]:.3f}±{m('dlnp')[1]:.3f}")
