"""LTE (paper, out/paper_252_2026-11-10) against NLTE EOS (out/paper_252_nlte_2026-11-13) runs of the paper's cases.
Metrics as reports/paper_252_2026-11-10/metrics.py (same masks, gauge, solved band), plus nₑ and ρ against the snapshot's
in the band above the cut (log τ −6 … −4) and in the solved band. usage: python compare.py → compare.txt"""
import numpy as np, h5py
OLD = '/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_2026-11-10'
NEW = '/mn/stornext/d9/data/harshm/MHS/HSEMHS/out/paper_252_nlte_2026-11-13'
IO = '/mn/stornext/d9/data/harshm/MHS/IO/tau252_nlte'
IO0 = '/mn/stornext/d9/data/harshm/MHS/IO/tau252'
CASES = [("μ=1 own τ: column HSE", "colhse_eos_mu1.0", "eos_mu1.0", "tau"), ("μ=1 own τ: MHS", "mhs_eos_mu1.0", "eos_mu1.0", "tau"),
         ("μ=1 M3D τ: column HSE", "colhse_m3d_mu1.0", "m3d_mu1.0", "tau"), ("μ=1 M3D τ: MHS", "mhs_m3d_mu1.0", "m3d_mu1.0", "tau"),
         ("μ=0.707 inc: column HSE", "colhse_eos_mu0.707_az0", "eos_mu0.707_az0", "tau"), ("μ=0.707 inc: MHS", "mhs_eos_mu0.707_az0", "eos_mu0.707_az0", "tau"),
         ("μ=0.5 inc: column HSE", "colhse_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"), ("μ=0.5 inc: MHS", "mhs_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"),
         ("μ=0.5 pp: MHS", "mhspp_eos_mu0.5_az0", "eos_mu0.5_az0", "tau"),
         ("m: column HSE", "colhse_logm_mu1.0", "logm_mu1.0", "m"), ("m: MHS", "mhs_logm_mu1.0", "logm_mu1.0", "m")]
EXTRA = [("μ=1 NLTE τ: column HSE", "nlteB_colhse_eosnlte_mu1.0", "eosnlte_mu1.0", "tau"),
         ("μ=1 NLTE τ: MHS", "nlteB_mhs_eosnlte_mu1.0", "eosnlte_mu1.0", "tau"),
         ("μ=1 NLTE τ: MHS cut −3", "nlteB_mhs_eosnlte_cut3_mu1.0", "eosnlte_mu1.0", "tau"),
         ("μ=1 own τ: MHS cut −3 (LTE)", "lte_mhs_eos_cut3_mu1.0", "eos_mu1.0", "tau"),
         ("μ=1 own τ: MHS (branch LTE)", "lte_mhs_eos_mu1.0", "eos_mu1.0", "tau")]
out = open(f'{NEW}/compare.txt', 'w')
def p(*a): print(*a); print(*a, file=out)
rms = lambda a: np.sqrt(np.mean(a[np.isfinite(a)] ** 2))
def masks(case):
    with h5py.File(f"{IO}/input_{case}.nc") as f:
        lt = f["log10tau500"][:]; k0 = int(np.argmin(abs(lt)))
        B = np.sqrt(sum(f[c][:, :, k0] ** 2 for c in ("B_x", "B_y", "B_z"))).T
    return B > 0.05, B < 0.01
def truth(case, grid):
    if grid == "tau":
        with h5py.File(f"{IO}/truth_{case}.h5") as f:
            z1 = f["truth/z_tau1_m"][:]; off = z1.mean()
            return dict(ax=f["logtau500"][:], u=f["truth/lnP"][:], r=f["truth/lnrho"][:], ne=f["truth/lnne"][:], z1=z1 - off)
    T0 = truth("eos_mu1.0", "tau")
    with h5py.File(f"{IO0}/truth_{case}.h5") as t:
        return dict(ax=t["log10m"][:], u=np.log(t["Pgas"][:]).transpose(2, 1, 0), r=np.log(t["rho"][:]).transpose(2, 1, 0),
                    ne=None, z1=T0["z1"])
def metrics(fn, nefn, case, grid):
    try: f = h5py.File(fn)
    except OSError: return None
    T = truth(case, grid); strong, quiet = masks(case if grid == "tau" else "eos_mu1.0")
    g = f["tau_grid" if grid == "tau" else "mass_grid"]; A = dict(f.attrs)
    z1 = g["z_tau1_m"][:]; z1 = z1 - z1.mean()
    dP = np.log(g["Pgas"][:]) - T["u"]; dr = np.log(g["rho"][:]) - T["r"]
    band = T["ax"] >= (-4.0 if grid == "tau" else -0.5); upper = (T["ax"] >= -6.0) & ~band if grid == "tau" else (T["ax"] >= -2.0) & ~band
    ne = None
    if T["ne"] is not None:
        try:
            with h5py.File(nefn) as e: ne = (e["grid/lnne"][:] - T["ne"]) / np.log(10)
        except OSError: pass
    st = A.get("status", b"-"); st = st.decode() if isinstance(st, bytes) else st
    return dict(steps=A.get("steps", "-"), status=st, z1=rms(z1 - T["z1"]) / 1e3, dep=(z1[strong].mean() - z1[quiet].mean()) / 1e3,
                depT=(T["z1"][strong].mean() - T["z1"][quiet].mean()) / 1e3, dP=rms(dP[band]), dr=rms(dr[band]),
                dPu=rms(dP[upper]), dru=rms(dr[upper]), ne=rms(ne[band]) if ne is not None else np.nan, neu=rms(ne[upper]) if ne is not None else np.nan)
p("Solved band log τ ≥ −4 (m: log10 m ≥ −0.5); upper band log τ −6 … −4 (m: −2 … −0.5), above the cut (τ-hydrostatic).")
p("z1 = rms z(τ=1) error [km]; dep = strong − quiet z(τ=1) [km] (truth in brackets); ΔlnP, Δlnρ rms; Δlog ne rms [dex].")
hdr = f"{'case':28s} {'EOS':5s} {'steps':>5s} {'status':9s} | {'z1':>5s} {'dep':>6s} {'(T)':>6s} | band: {'ΔlnP':>5s} {'Δlnρ':>5s} {'Δlogne':>6s} | upper: {'ΔlnP':>5s} {'Δlnρ':>5s} {'Δlogne':>6s}"
p(hdr)
def row(label, eos, m):
    if m is None: p(f"{label:28s} {eos:5s} (missing)"); return
    p(f"{label:28s} {eos:5s} {str(m['steps']):>5s} {m['status']:9s} | {m['z1']:5.1f} {m['dep']:+6.1f} {m['depT']:+6.1f} | "
      f"      {m['dP']:5.3f} {m['dr']:5.3f} {m['ne']:6.3f} |        {m['dPu']:5.3f} {m['dru']:5.3f} {m['neu']:6.3f}")
for label, run, case, grid in CASES:
    row(label, "LTE", metrics(f"{OLD}/{run}.h5", f"{NEW}/lte_ne/{run}.ne.h5", case, grid))
    row("", "NLTE", metrics(f"{NEW}/nlte_{run}.h5", f"{NEW}/nlte_{run}.ne.h5", case, grid))
p("")
for label, run, case, grid in EXTRA:
    row(label, "LTE" if run.startswith("lte") else "NLTE", metrics(f"{NEW}/{run}.h5", f"{NEW}/{run}.ne.h5", case, grid))
