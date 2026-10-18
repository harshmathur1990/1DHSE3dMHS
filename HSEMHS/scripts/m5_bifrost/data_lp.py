#!/usr/bin/env python3
"""
Data-only magnetic feasibility LPs on the SHARED observation artifact (Codex M5d review, steps 2–3).

Reads out/m5d/observations.h5 written by scripts/m5_bifrost/export_observations.jl, i.e. exactly the arrays the
Julia solver uses (no re-implemented preprocessing). Unknowns are the face field B_f; div B_f = 0 in every cell.

Modes
  face-max --t T          |B_f − I·y_f| ≤ T σ_f on every face                         (feasibility)
  face-support --support S.npy --t5 5 --t3 3 --margin m
                          |c| ≤ t5 − m on the faces in S (boolean or index array), |c| ≤ t3 − m elsewhere; with
                          |S| ≤ K = floor(0.01 N) this proves BOTH the maximum and the 1 % tail criteria
  face-hinge --t T --knee K3   minimize Σ max(|c| − K3, 0) subject to |c| ≤ T (a relaxation/search tool; a
                          certified optimum > 2·K would prove simultaneous infeasibility; a smaller optimum proves
                          nothing unless its actual count is ≤ K)
  centre-max --t T        |H B_f − y_c| ≤ T σ_c at every centre (H: the two-face average onto each centre)
Every solution is verified INDEPENDENTLY of the LP matrix: the physical face field is rebuilt, its divergence
recomputed directly from the metrics, and the residuals recomputed from the observations; margins to the
thresholds are reported. Results are numerically verified witnesses, not exact-arithmetic certificates.

Requirements: python3, numpy, scipy, h5py (highspy optional, not needed).
"""
import argparse, sys, time
import numpy as np, h5py, scipy.sparse as sp
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True)
ap.add_argument("--mode", choices=("face-max", "face-support", "face-hinge", "centre-max"), required=True)
ap.add_argument("--t", type=float, default=5.0)
ap.add_argument("--t5", type=float, default=5.0)
ap.add_argument("--t3", type=float, default=3.0)
ap.add_argument("--margin", type=float, default=1e-6)
ap.add_argument("--knee", type=float, default=3.0)
ap.add_argument("--support", default="")
ap.add_argument("--solver", choices=("highs-ipm", "highs-ds", "highs"), default="highs-ipm")
ap.add_argument("--time-limit", type=float, default=float("inf"))
ap.add_argument("--out", default="")
args = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

with h5py.File(args.obs, "r") as f:
    # HDF5.jl writes Julia (nx, ny, nz) arrays; h5py reads them reversed: transpose back to (x, y, z)
    rd = lambda k: np.ascontiguousarray(f[k][()].T)
    fo = [rd(f"face_obs{c}") for c in (1, 2, 3)]; fs = [rd(f"face_sigma{c}") for c in (1, 2, 3)]
    co = [rd(f"centre_obs{c}") for c in (1, 2, 3)]; cs = [rd(f"centre_sigma{c}") for c in (1, 2, 3)]
    dxc = f["dxc"][()]; dyc = f["dyc"][()]; hz = f["hz"][()]
nx, ny, nz = co[0].shape
assert fo[2].shape == (nx, ny, nz + 1)
idx = np.arange(nx * ny * nz).reshape(nx, ny, nz); n = idx.size
izf = np.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1) + 2 * n
nf = 3 * n + nx * ny
fobs = np.concatenate([fo[0].ravel(), fo[1].ravel(), fo[2].ravel()])
fsig = np.concatenate([fs[0].ravel(), fs[1].ravel(), fs[2].ravel()])
log(f"grid {nx}×{ny}×{nz}; {nf} faces, {n} cells")

# divergence of a face field, per cell: (Bx[i] − Bx[i−1])/dx + (By[j] − By[j−1])/dy + (Bz[top] − Bz[bottom])/hz
DX = dxc[:, None, None]; DY = dyc[None, :, None]; HZ = hz[None, None, :]
def div_of(Bf):
    Bx = Bf[:n].reshape(nx, ny, nz); By = Bf[n:2 * n].reshape(nx, ny, nz); Bz = Bf[2 * n:].reshape(nx, ny, nz + 1)
    return (Bx - np.roll(Bx, 1, axis=0)) / DX + (By - np.roll(By, 1, axis=1)) / DY + (Bz[:, :, :-1] - Bz[:, :, 1:]) / HZ
# sparse div operator on B_f (same formula)
rows = np.tile(idx.ravel(), 6)
cols = np.concatenate([idx.ravel(), np.roll(idx, 1, axis=0).ravel(), (idx + n).ravel(), np.roll(idx + n, 1, axis=1).ravel(),
                       izf[:, :, :-1].ravel(), izf[:, :, 1:].ravel()])
vals = np.concatenate([np.broadcast_to(1 / DX, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / DX, (nx, ny, nz)).ravel(),
                       np.broadcast_to(1 / DY, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / DY, (nx, ny, nz)).ravel(),
                       np.broadcast_to(1 / HZ, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / HZ, (nx, ny, nz)).ravel()])
Div = sp.coo_matrix((vals, (rows, cols)), shape=(n, nf)).tocsr()
# centre average H (centre ← its two faces)
hrows = np.concatenate([np.repeat(idx.ravel(), 2), np.repeat((idx + n).ravel(), 2), np.repeat((idx + 2 * n).ravel(), 2)])
hcols = np.concatenate([np.stack([np.roll(idx, 1, axis=0).ravel(), idx.ravel()], 1).ravel(),
                        np.stack([np.roll(idx + n, 1, axis=1).ravel(), (idx + n).ravel()], 1).ravel(),
                        np.stack([izf[:, :, :-1].ravel(), izf[:, :, 1:].ravel()], 1).ravel()])
Hop = sp.coo_matrix((np.full(hrows.size, 0.5), (hrows, hcols)), shape=(3 * n, nf)).tocsr()
cobs = np.concatenate([co[0].ravel(), co[1].ravel(), co[2].ravel()])
csig = np.concatenate([cs[0].ravel(), cs[1].ravel(), cs[2].ravel()])
K = int(np.floor(0.01 * nf)); Kc = int(np.floor(0.01 * 3 * n))

def verify(Bf, tag):
    d = div_of(Bf); bscale = np.median(np.abs(fobs)) / np.median(dxc)
    cf = (Bf - fobs) / fsig; cc = (Hop @ Bf - cobs) / csig
    out = dict(div_max=np.max(np.abs(d)), div_rel=np.max(np.abs(d)) / bscale,
               face_max=np.max(np.abs(cf)), face_frac3=np.mean(np.abs(cf) > 3), face_n5=int(np.sum(np.abs(cf) > 5)),
               centre_max=np.max(np.abs(cc)), centre_frac3=np.mean(np.abs(cc) > 3), centre_n5=int(np.sum(np.abs(cc) > 5)))
    log(f"[{tag}] verified from the physical field: max|div B| {out['div_max']:.3e} T/m (relative {out['div_rel']:.1e}); "
        f"FACE max {out['face_max']:.7f}σ (margin to 5: {5 - out['face_max']:+.2e}), >3σ {out['face_frac3']:.4%} (budget 1 % = {K}), #>5σ {out['face_n5']}; "
        f"CENTRE max {out['centre_max']:.7f}σ, >3σ {out['centre_frac3']:.4%}, #>5σ {out['centre_n5']}")
    return out

def run(cost, A_ub, b_ub, A_eq, b_eq, bounds, tag):
    opts = {"primal_feasibility_tolerance": 1e-9}
    if np.isfinite(args.time_limit):
        opts["time_limit"] = args.time_limit
    tm = time.time()
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=args.solver, options=opts)
    log(f"[{tag}] linprog status {res.status} ({res.message}) in {time.time() - tm:.0f} s" + (f"; objective {res.fun:.6g}" if res.status == 0 else ""))
    return res

def save(Bf, tag, extra=None):
    if args.out:
        np.savez(args.out, B_face=Bf, c_face=(Bf - fobs) / fsig, tag=tag, **(extra or {}))
        log(f"saved {args.out}")

# variables scaled as c = (B_f − I·y)/σ_f, so B_f = fobs + fsig·c and the divergence rows read Div·diag(fsig)·c = −Div·fobs
Aeq_c = (Div @ sp.diags(fsig)).tocsr(); beq_c = -(Div @ fobs)
rn = np.asarray(np.abs(Aeq_c).sum(axis=1)).ravel(); Rn = sp.diags(1 / rn)
Aeq_c = (Rn @ Aeq_c).tocsr(); beq_c = beq_c / rn
log(f"necessary local bound (face model): t ≥ {np.max(np.abs(beq_c)):.6f}")

if args.mode == "face-max":
    res = run(np.zeros(nf), None, None, Aeq_c, beq_c, (-args.t, args.t), f"face-max t={args.t}")
    if res.status == 0:
        Bf = fobs + fsig * res.x; verify(Bf, "face-max"); save(Bf, "face-max")
elif args.mode == "face-support":
    S = np.load(args.support)
    mask = S.astype(bool) if S.dtype == bool or S.size == nf else np.isin(np.arange(nf), S)
    log(f"support |S| = {int(mask.sum())} (budget K = {K}); bounds {args.t5}−{args.margin} on S, {args.t3}−{args.margin} elsewhere")
    ub = np.where(mask, args.t5 - args.margin, args.t3 - args.margin)
    res = run(np.zeros(nf), None, None, Aeq_c, beq_c, list(zip(-ub, ub)), "face-support")
    if res.status == 0:
        Bf = fobs + fsig * res.x; o = verify(Bf, "face-support"); save(Bf, "face-support")
        log("BOTH criteria met by this witness" if (o["face_max"] <= 5 and o["face_frac3"] <= 0.01) else "witness does not meet both criteria")
    elif res.status == 2:
        log("infeasible FOR THIS SUPPORT AND THESE BOUNDS ONLY (other supports are not excluded)")
elif args.mode == "face-hinge":
    I = sp.identity(nf, format="csr")
    Aub = sp.vstack([sp.hstack([I, -I]), sp.hstack([-I, -I])]).tocsr(); bub = np.full(2 * nf, args.knee)
    Ae = sp.hstack([Aeq_c, sp.csr_matrix((n, nf))]).tocsr()
    res = run(np.concatenate([np.zeros(nf), np.ones(nf)]), Aub, bub, Ae, beq_c,
              [(-args.t, args.t)] * nf + [(0, None)] * nf, f"face-hinge t={args.t} knee={args.knee}")
    if res.status == 0:
        Bf = fobs + fsig * res.x[:nf]; o = verify(Bf, "face-hinge")
        log(f"hinge optimum {res.fun:.4f} vs 2K = {2 * K}: " + ("> 2K would prove simultaneous infeasibility (needs a certified bound)"
            if res.fun > 2 * K else "≤ 2K: no negative conclusion; positive only if the actual count is ≤ K"))
        save(Bf, "face-hinge")
elif args.mode == "centre-max":
    # variables b = B_f/fsig (conditioning); rows: Div·diag(fsig)·b = 0;  |(H·diag(fsig)·b − y)/σ_c| ≤ t
    Hs = (sp.diags(1 / csig) @ Hop @ sp.diags(fsig)).tocsr(); ys = cobs / csig
    Aeq_b = (Rn @ (Div @ sp.diags(fsig))).tocsr()
    Aub = sp.vstack([Hs, -Hs]).tocsr(); bub = np.concatenate([ys + args.t, -(ys - args.t)])
    res = run(np.zeros(nf), Aub, bub, Aeq_b, np.zeros(n), (None, None), f"centre-max t={args.t}")
    if res.status == 0:
        Bf = fsig * res.x; verify(Bf, "centre-max"); save(Bf, "centre-max")
    elif res.status == 2:
        log(f"centre model: INFEASIBLE at t = {args.t} (HiGHS verdict; numerically, not exact arithmetic)")
