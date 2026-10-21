#!/usr/bin/env python3
"""
Gentle face-admissible field (Codex M5e review §2, deliverable 2): the constrained least-squares fit on a fixed
support S,

    minimize    mean(c²)                              c = (B_f − I·y_f)/σ_f on every face
    subject to  div B_f = 0                           in every cell
                |c_i| ≤ t5 − m on S,  |c_i| ≤ t3 − m elsewhere,

a convex QP in the face field. Reads the shared observation artifact out/m5d/observations.h5 (the arrays the
Julia solver uses), so the objective is the solver's face data term Φ_B = mean(c²) (the λ_S smoothness term is
not included here; it is evaluated afterwards through the solver representation by verify_qp_field.jl).

Solver: ADMM (OSQP-type splitting) with x in the affine set {div B = 0} and z in the box:
    x ← Proj_div( ρ/(1+ρ)·(z − u) ),   x̂ = αx + (1−α)z,   z ← clip(x̂ + u),   u ← u + x̂ − z   (α = 1.6),
Proj_div(v) = v − Aᵀ M⁻¹ (A v − r), A = Div·diag(σ), r = −Div·I·y, M = A Aᵀ (7-point, SPD; factorized once).
Stopped when ‖x − z‖_∞ ≤ tol and the dual residual ρ‖z − z_prev‖_∞ ≤ tol. The returned field is x (exactly
divergence-free to the solve accuracy), and it is verified independently: divergence recomputed from the metrics,
face and centre residuals recomputed, bounds checked against the TRUE thresholds 5 and 3 and against t5−m, t3−m.
Stopping evidence: the ADMM primal/dual residuals and the objective history. These are not a computed KKT gap or a
certificate of optimality (M5f review): the result is an approximate QP optimum; its feasibility is checked separately.

usage: python3 data_qp.py --obs out/m5d/observations.h5 --support out/m5d/support_p128_above3.npy \
           --margin 0.05 --out out/m5f/qp_m0.05.npz [--tol 1e-6] [--max-iter 20000]
"""
import argparse, time
import numpy as np, h5py, scipy.sparse as sp, scipy.sparse.linalg as spla

ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True)
ap.add_argument("--support", required=True)
ap.add_argument("--t5", type=float, default=5.0)
ap.add_argument("--t3", type=float, default=3.0)
ap.add_argument("--margin", type=float, default=0.05)
ap.add_argument("--rho", type=float, default=1.0)
ap.add_argument("--alpha", type=float, default=1.6)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--max-iter", type=int, default=20000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

with h5py.File(args.obs, "r") as f:
    rd = lambda k: np.ascontiguousarray(f[k][()].T)
    fo = [rd(f"face_obs{c}") for c in (1, 2, 3)]; fs = [rd(f"face_sigma{c}") for c in (1, 2, 3)]
    co = [rd(f"centre_obs{c}") for c in (1, 2, 3)]; cs = [rd(f"centre_sigma{c}") for c in (1, 2, 3)]
    dxc = f["dxc"][()]; dyc = f["dyc"][()]; hz = f["hz"][()]
nx, ny, nz = co[0].shape
idx = np.arange(nx * ny * nz).reshape(nx, ny, nz); n = idx.size
izf = np.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1) + 2 * n
nf = 3 * n + nx * ny
fobs = np.concatenate([v.ravel() for v in fo]); fsig = np.concatenate([v.ravel() for v in fs])
cobs = np.concatenate([v.ravel() for v in co]); csig = np.concatenate([v.ravel() for v in cs])
K = int(np.floor(0.01 * nf))
DX = dxc[:, None, None]; DY = dyc[None, :, None]; HZ = hz[None, None, :]

def split(Bf):
    return Bf[:n].reshape(nx, ny, nz), Bf[n:2 * n].reshape(nx, ny, nz), Bf[2 * n:].reshape(nx, ny, nz + 1)
def div_of(Bf):
    Bx, By, Bz = split(Bf)
    return (Bx - np.roll(Bx, 1, axis=0)) / DX + (By - np.roll(By, 1, axis=1)) / DY + (Bz[:, :, :-1] - Bz[:, :, 1:]) / HZ
def centre_of(Bf):
    Bx, By, Bz = split(Bf)
    return np.concatenate([(0.5 * (Bx + np.roll(Bx, 1, axis=0))).ravel(), (0.5 * (By + np.roll(By, 1, axis=1))).ravel(),
                           (0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])).ravel()])

rows = np.tile(idx.ravel(), 6)
cols = np.concatenate([idx.ravel(), np.roll(idx, 1, axis=0).ravel(), (idx + n).ravel(), np.roll(idx + n, 1, axis=1).ravel(),
                       izf[:, :, :-1].ravel(), izf[:, :, 1:].ravel()])
vals = np.concatenate([np.broadcast_to(1 / DX, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / DX, (nx, ny, nz)).ravel(),
                       np.broadcast_to(1 / DY, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / DY, (nx, ny, nz)).ravel(),
                       np.broadcast_to(1 / HZ, (nx, ny, nz)).ravel(), np.broadcast_to(-1 / HZ, (nx, ny, nz)).ravel()])
Div = sp.coo_matrix((vals, (rows, cols)), shape=(n, nf)).tocsr()
A = (Div @ sp.diags(fsig)).tocsr(); r = -(Div @ fobs)
rn = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()); A = (sp.diags(1 / rn) @ A).tocsr(); r = r / rn
tm = time.time()
M = (A @ A.T).tocsc()
lu = spla.splu(M, permc_spec="MMD_AT_PLUS_A")
log(f"grid {nx}×{ny}×{nz}, {nf} faces; M = A Aᵀ {M.shape[0]}² nnz {M.nnz}; LU fill {lu.L.nnz + lu.U.nnz} in {time.time() - tm:.1f} s")
def proj(v):
    lam = lu.solve(A @ v - r)
    lam = lam + lu.solve(A @ (v - A.T @ lam) - r)          # one step of iterative refinement
    return v - A.T @ lam

S = np.load(args.support)
mask = S.astype(bool) if S.dtype == bool or S.size == nf else np.isin(np.arange(nf), S)
ub = np.where(mask, args.t5 - args.margin, args.t3 - args.margin)
log(f"support |S| = {int(mask.sum())} (budget K = {K}); bounds ±{args.t5 - args.margin} on S, ±{args.t3 - args.margin} elsewhere")

x = proj(np.zeros(nf))                                     # unconstrained minimum-norm divergence-free fit
log(f"unconstrained min mean(c²) on div B = 0: {np.mean(x ** 2):.6f}, max |c| {np.max(np.abs(x)):.4f}")
z = np.clip(x, -ub, ub); u = x - z; rho = args.rho
tm = time.time(); hist = []
for it in range(1, args.max_iter + 1):
    x = proj(rho / (1 + rho) * (z - u))
    xh = args.alpha * x + (1 - args.alpha) * z            # over-relaxation (OSQP default α = 1.6)
    zp = z; z = np.clip(xh + u, -ub, ub)
    u += xh - z
    pr = np.max(np.abs(x - z)); dr = rho * np.max(np.abs(z - zp))
    if it % 50 == 0 or it == 1:
        obj = np.mean(x ** 2); hist.append((it, obj, pr, dr))
        log(f"it {it:6d} mean(c²) {obj:.8f}  primal {pr:.2e}  dual {dr:.2e}  ρ {rho:.3g}  max|x| {np.max(np.abs(x) / ub):.6f}·ub  {time.time() - tm:.0f} s")
    if pr <= args.tol and dr <= args.tol:
        log(f"converged at it {it}: primal {pr:.2e}, dual {dr:.2e}"); break
    if it % 50 == 0:                                       # residual balancing (scaled dual u rescales with ρ)
        if pr > 10 * dr:
            rho *= 2; u /= 2
        elif dr > 10 * pr:
            rho /= 2; u *= 2
else:
    log(f"max_iter reached: primal {pr:.2e}, dual {dr:.2e}")

c = x; Bf = fobs + fsig * c
d = div_of(Bf); cc = (centre_of(Bf) - cobs) / csig
bscale = np.median(np.abs(fobs)) / np.median(dxc)
over = np.abs(c) - ub
o = dict(mean_c2=float(np.mean(c ** 2)), face_max=float(np.max(np.abs(c))), face_max_off=float(np.max(np.abs(c[~mask]))),
         face_frac3=float(np.mean(np.abs(c) > 3)), face_n3=int(np.sum(np.abs(c) > 3)), face_n5=int(np.sum(np.abs(c) > 5)),
         n_at_bound=int(np.sum(over > -1e-4)), max_over_ub=float(np.max(over)),
         div_max=float(np.max(np.abs(d))), div_rel=float(np.max(np.abs(d)) / bscale),
         centre_max=float(np.max(np.abs(cc))), centre_frac3=float(np.mean(np.abs(cc) > 3)), centre_mean_c2=float(np.mean(cc ** 2)))
log(f"VERIFIED from the physical field: max|div B| {o['div_max']:.3e} T/m (relative {o['div_rel']:.1e}); mean(c²) {o['mean_c2']:.6f}; "
    f"FACE max {o['face_max']:.6f}σ (off S {o['face_max_off']:.6f}σ), >3σ {o['face_n3']} = {o['face_frac3']:.4%} (K = {K}), #>5σ {o['face_n5']}; "
    f"max(|c| − bound) {o['max_over_ub']:+.2e}; {o['n_at_bound']} faces within 1e-4 of their bound; "
    f"CENTRE max {o['centre_max']:.4f}σ, >3σ {o['centre_frac3']:.4%}, mean c² {o['centre_mean_c2']:.4f}")
ok = o["face_max"] <= args.t5 and o["face_n3"] <= K and o["face_max_off"] <= args.t3
log("BOTH face criteria met" if ok else "face criteria NOT met")
np.savez(args.out, B_face=Bf, c_face=c, c_centre=cc, support=mask, ub=ub, hist=np.array(hist), margin=args.margin,
         **{k: v for k, v in o.items()})
log(f"saved {args.out}")
