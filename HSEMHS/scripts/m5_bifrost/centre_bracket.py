#!/usr/bin/env python3
"""
Centre-model maximum feasibility: an anytime, independently verifiable bracket on
    t* = min max_i |c_i|,   c = (H B − y_c)/σ_c,   over divergence-free face fields B
(Codex M5e review §3: bounded run, profiling, valid lower-bound evidence; an LP backend independent of HiGHS).

Uses Codex's reduced centre formulation (codex_m5e_centre_feasibility.py, uniform periodic x/y): with C = H B,
L C = 0 (vertical pairing of q = Hy Dx Cx + Hx Dy Cy with the Cz difference, plus the horizontal-Nyquist rows) is
necessary and sufficient for C to come from a divergence-free B. In c-variables: A c = r, A = L diag(σ_c), r = −L y_c.

  LOWER bound, for ANY multiplier λ (weak duality):  t* ≥ λᵀr / ‖Aᵀλ‖₁.  Checked exactly at every report.
  UPPER bound: reconstruct B from C = y + σ c (Fourier inversion of Hx, Hy + divergence recurrence; exactly
               divergence-free by construction), recompute H B and report the TRUE max |c| of that physical field.
So t* ∈ [best LB, best UB] at all times; LB > 5 would show centre infeasibility at t = 5 (numerically, relying on
the necessity of L C = 0, which the random-curl check verifies), UB ≤ 5 gives a witness (saved).

Solver: PDHG (Chambolle–Pock) on min ‖c‖_∞ s.t. Ac = r, with Pock–Chambolle diagonal preconditioning and
restarts to the running average every `--restart` iterations; prox of τ‖·‖_∞ via projection onto the ℓ1 ball.
usage (from HSEMHS/): python3 centre_bracket.py --obs out/m5d/observations.h5 --out out/m5f/centre_bracket.npz
                      [--wall 3000] [--report 200] [--restart 1000]
"""
import argparse, time, os, sys
import numpy as np, h5py, scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--wall", type=float, default=3000.0)
ap.add_argument("--report", type=int, default=200)
ap.add_argument("--restart", type=int, default=1000)
ap.add_argument("--max-iter", type=int, default=10 ** 9)
args = ap.parse_args()
T0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f} s]", *a, flush=True)

with h5py.File(args.obs) as f:
    obs = [f[f"centre_obs{c}"][()].T for c in (1, 2, 3)]
    sig = [f[f"centre_sigma{c}"][()].T for c in (1, 2, 3)]
    dx, dy, hz = [f[k][:] for k in ("dxc", "dyc", "hz")]
nx, ny, nz = obs[0].shape
assert np.all(dx == dx[0]) and np.all(dy == dy[0]), "requires uniform x/y"
n = nx * ny * nz; ids = np.arange(n).reshape(nx, ny, nz); nr = nx * ny * (nz - 1); row = np.arange(nr)
rows, cols, vals = [], [], []
def add(index, coefficient):
    rows.append(row); cols.append(index.ravel()); vals.append(np.broadcast_to(coefficient, (nx, ny, nz - 1)).ravel())
for zoff in (0, 1):                                   # (identical to codex_m5e_centre_feasibility.py)
    zsel = slice(zoff, nz - 1 + zoff)
    for xoff in (0, 1):
        for yoff in (0, 1):
            index = np.roll(np.roll(ids, xoff, 0), yoff, 1)[:, :, zsel]
            add(index, (-1) ** xoff * .5 * hz[zsel] / dx[0])
            add(index + n, (-1) ** yoff * .5 * hz[zsel] / dy[0])
            add(index + 2 * n, (-1) ** zoff * .5)
L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nr, 3 * n)).tocsr()
nyq = []
if nx % 2 == 0:
    rr = np.broadcast_to(np.arange(ny * nz).reshape(1, ny, nz), (nx, ny, nz)); vv = np.broadcast_to((-1.) ** np.arange(nx)[:, None, None], (nx, ny, nz))
    nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), ids.ravel())), shape=(ny * nz, 3 * n)).tocsr())
if ny % 2 == 0:
    rr = np.broadcast_to(np.arange(nx * nz).reshape(nx, 1, nz), (nx, ny, nz)); vv = np.broadcast_to((-1.) ** np.arange(ny)[None, :, None], (nx, ny, nz))
    nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), (ids + n).ravel())), shape=(nx * nz, 3 * n)).tocsr())
L = sp.vstack([L] + nyq, format="csr")
y = np.concatenate([v.ravel() for v in obs]); s = np.concatenate([v.ravel() for v in sig])
A = (L @ sp.diags(s)).tocsr(); r = -(L @ y)
m, N = A.shape
log(f"reduced centre system {m} × {N}, nnz {A.nnz}")

def centre(B):
    return [.5 * (B[0] + np.roll(B[0], 1, 0)), .5 * (B[1] + np.roll(B[1], 1, 1)), .5 * (B[2][:, :, :-1] + B[2][:, :, 1:])]
def divergence(B):
    return ((B[0] - np.roll(B[0], 1, 0)) / dx[:, None, None] + (B[1] - np.roll(B[1], 1, 1)) / dy[None, :, None]
            + (B[2][:, :, :-1] - B[2][:, :, 1:]) / hz)
def reconstruct(C):                                   # (identical to codex_m5e_centre_feasibility.py)
    cf = [np.fft.fftn(v, axes=(0, 1)) for v in C]
    ex = np.exp(-2j * np.pi * np.fft.fftfreq(nx))[:, None, None]; ey = np.exp(-2j * np.pi * np.fft.fftfreq(ny))[None, :, None]
    hx, hy = .5 * (1 + ex), .5 * (1 + ey); dxsym, dysym = (1 - ex) / dx[0], (1 - ey) / dy[0]
    zx, zy = abs(hx) < 1e-12, abs(hy) < 1e-12
    bx = np.divide(cf[0], hx, out=np.zeros_like(cf[0]), where=~zx); by = np.divide(cf[1], hy, out=np.zeros_like(cf[1]), where=~zy)
    regular = np.broadcast_to(~(zx | zy), (nx, ny, 1))[:, :, 0]
    d = dxsym * bx + dysym * by
    bz = np.zeros((nx, ny, nz + 1), dtype=complex)
    bz[:, :, 0] = cf[2][:, :, 0] - .5 * hz[0] * d[:, :, 0]; bz[:, :, 1:] = bz[:, :, :1] + np.cumsum(d * hz, axis=2)
    bz[~regular, 0] = 0
    for k in range(nz):
        bz[~regular, k + 1] = 2 * cf[2][~regular, k] - bz[~regular, k]
    target_d = (bz[:, :, 1:] - bz[:, :, :-1]) / hz
    new_bx = np.divide(target_d - dysym * by, dxsym, out=np.zeros_like(bx), where=abs(dxsym) > 1e-12 / dx[0]); bx = np.where(zx, new_bx, bx)
    new_by = np.divide(target_d - dxsym * bx, dysym, out=np.zeros_like(by), where=abs(dysym) > 1e-12 / dy[0]); by = np.where(zy & ~zx, new_by, by)
    return [np.fft.ifftn(v, axes=(0, 1)).real for v in (bx, by, bz)]
def true_max(c):
    C = [(y + s * c)[q * n:(q + 1) * n].reshape(nx, ny, nz) for q in range(3)]
    B = reconstruct(C); pred = centre(B)
    cor = np.concatenate([((v - o) / sg).ravel() for v, o, sg in zip(pred, obs, sig)])
    return np.max(np.abs(cor)), cor, B
def lower(lam):
    den = np.sum(np.abs(A.T @ lam))
    return (lam @ r) / den if den > 0 else -np.inf

# Pock–Chambolle diagonal preconditioning (α = 1): τ_j = 1/Σ_i |A_ij|, σ_i = 1/Σ_j |A_ij|
absA = abs(A)
tau = 1 / np.maximum(np.asarray(absA.sum(axis=0)).ravel(), 1e-300)
sgm = 1 / np.maximum(np.asarray(absA.sum(axis=1)).ravel(), 1e-300)
def prox_inf(v, tw):
    # prox of ‖·‖_∞ in the diag(1/tw) metric: argmin_c ‖c‖_∞ + Σ (c_j − v_j)²/(2 tw_j)
    # = clip(v, −t, t) with the scalar t solving Σ tw_j⁻¹ ... : t ≥ 0 minimizes t + Σ max(|v_j| − t, 0)²/(2 tw_j)
    a = np.abs(v); order = np.argsort(-a); a_s = a[order]; w_s = 1 / tw[order]
    # derivative: 1 − Σ_{a_j > t} w_j (a_j − t) = 0  →  t = (Σ w a − 1)/Σ w over the active prefix
    cw = np.cumsum(w_s); cwa = np.cumsum(w_s * a_s)
    tcand = (cwa - 1) / cw
    nxt = np.append(a_s[1:], 0.0)
    k = np.nonzero((tcand >= nxt) & (tcand <= a_s))[0]
    t = max(tcand[k[0]], 0.0) if k.size else 0.0
    return np.clip(v, -t, t)

c = np.zeros(N); lam = np.zeros(m); c_avg = np.zeros(N); lam_avg = np.zeros(m); navg = 0
bestLB, bestUB = -np.inf, np.inf; best_lam = lam.copy(); best_c = c.copy()
log(f"necessary local bound (λ = unit vectors): {np.max(np.abs(r) / np.asarray(absA.sum(axis=1)).ravel()):.6f}")
it = 0
while it < args.max_iter and time.time() - T0 < args.wall:
    it += 1
    c_new = prox_inf(c - tau * (A.T @ lam), tau)
    lam = lam + sgm * (A @ (2 * c_new - c) - r)
    c = c_new
    c_avg += c; lam_avg += lam; navg += 1
    if it % args.report == 0:
        for name, cc, ll in (("last", c, lam), ("avg", c_avg / navg, lam_avg / navg)):
            lb = lower(ll); ub, _, _ = true_max(cc)
            if lb > bestLB: bestLB, best_lam = lb, ll.copy()
            if ub < bestUB: bestUB, best_c = ub, cc.copy()
        log(f"it {it:7d}  ‖c‖∞ {np.max(np.abs(c)):.5f}  ‖Ac − r‖∞ {np.max(np.abs(A @ c - r)):.2e}  |  bracket t* ∈ [{bestLB:.5f}, {bestUB:.5f}]")
        np.savez(args.out, best_lam=best_lam, best_c=best_c, LB=bestLB, UB=bestUB, it=it)
    if it % args.restart == 0:                           # restart to the average
        c, lam = c_avg / navg, lam_avg / navg; c_avg[:] = 0; lam_avg[:] = 0; navg = 0
log(f"stopped at it {it} (wall {args.wall:.0f} s or max-iter)")
# independent final verification of both ends
lb = lower(best_lam); ub, cor, B = true_max(best_c)
log(f"FINAL (re-verified): t* ∈ [{lb:.6f}, {ub:.6f}]; UB field max |div B| {np.max(np.abs(divergence(B))):.2e} T/m, "
    f">3σ {np.mean(np.abs(cor) > 3):.4%};  t = 5 is " + ("INFEASIBLE (LB > 5)" if lb > 5 else "FEASIBLE (UB ≤ 5)" if ub <= 5 else "UNRESOLVED (LB ≤ 5 < UB)"))
np.savez(args.out, best_lam=best_lam, best_c=best_c, LB=lb, UB=ub, it=it, B_face=np.concatenate([v.ravel() for v in B]), c_centre=cor)
log(f"saved {args.out}")
