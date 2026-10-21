#!/usr/bin/env python3
"""
Centre-model tail feasibility on ONE fixed support (Codex M5f review §3, deliverable 5): with the K = ⌊0.01 N⌋ largest
|c| of the M5f centre witness (3.9456σ) as S, find a divergence-free field with

    |c_i| ≤ t5 = 4.999 on S,   |c_i| ≤ t3 = 2.999 off S,   A c = r  (L(y + σ c) = 0, reduced centre formulation)

preferring a gentle one (small ‖c‖²). Acceptance is decided on the reconstructed PHYSICAL field against the true criteria
(max ≤ 5σ, #>3σ ≤ K); the 1e-3 margins absorb the final projection. A failure or a timeout says nothing about other
supports.

Solver: PDHG (Chambolle–Pock, Pock–Chambolle diagonal preconditioning, restarts to the running average) on
    min_c  (γ/2)‖c‖² + ι_box(c)   s.t.  A c = r,
the iterative method already used by centre_bracket.py: only products with A and Aᵀ; the prox is separable,
c ← clip(v/(1 + τγ), −ub, ub). (A sparse LU of A Aᵀ for an ADMM projection was tried first: 23 min and 1.3e9 fill,
then ~160 s per projection — not usable.)
Infeasibility evidence (weak duality, valid for ANY λ): for c in the box with A c = r,
|λᵀr| = |(Aᵀλ)ᵀc| ≤ Σ_j ub_j |(Aᵀλ)_j|, so a ratio |λᵀr| / Σ ub|Aᵀλ| > 1 proves this support infeasible. Checked on the
PDHG multipliers (which grow along such a direction on infeasible problems) and on the local rows.
Stops: ‖Ac − r‖∞ ≤ tol·‖r‖∞ followed by a passing physical check (feasible), a ratio > 1 (infeasible), or the wall.

usage (from HSEMHS/): python3 centre_tail.py --obs out/m5d/observations.h5 --witness out/m5f/centre_bracket.npz
                      --out out/m5g/centre_tail.npz [--K 4709] [--t5 4.999] [--t3 2.999] [--gamma 1e-2] [--wall 7200]
"""
import argparse, os, sys, time
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centre_bracket import CentreSystem

ap = argparse.ArgumentParser()
ap.add_argument("--obs", required=True)
ap.add_argument("--witness", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--K", type=int, default=-1)
ap.add_argument("--t5", type=float, default=4.999)
ap.add_argument("--t3", type=float, default=2.999)
ap.add_argument("--gamma", type=float, default=1e-2)
ap.add_argument("--tol", type=float, default=1e-7)
ap.add_argument("--report", type=int, default=1000)
ap.add_argument("--restart", type=int, default=2000)
ap.add_argument("--check-every", type=int, default=10000, help="physical check of the averaged iterate every so many iterations")
ap.add_argument("--wall", type=float, default=7200.0)
args = ap.parse_args()
T0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f} s]", *a, flush=True)

with h5py.File(args.obs) as f:
    obs = [f[f"centre_obs{c}"][()].T for c in (1, 2, 3)]
    sig = [f[f"centre_sigma{c}"][()].T for c in (1, 2, 3)]
    dx, dy, hz = [f[k][:] for k in ("dxc", "dyc", "hz")]
S = CentreSystem(obs, sig, dx, dy, hz)
A, r = S.A, S.r; m, N = A.shape
K = args.K if args.K > 0 else int(np.floor(0.01 * N))
cw = np.load(args.witness)["c_centre"]; assert cw.size == N
supp = np.zeros(N, bool); supp[np.argsort(-np.abs(cw))[:K]] = True
ub = np.where(supp, args.t5, args.t3)
rinf = float(np.max(np.abs(r)))
log(f"reduced centre system {m} × {N}, nnz {A.nnz}; witness max {np.max(np.abs(cw)):.4f}σ, >3σ {np.sum(np.abs(cw) > 3)}; "
    f"S = its K = {K} largest |c| (min |c| on S {np.min(np.abs(cw[supp])):.4f}); bounds {args.t5} on S, {args.t3} off S; "
    f"witness violates them at {np.sum(np.abs(cw) > ub)} entries")

def cert_ratio(lam):
    den = ub @ np.abs(A.T @ lam)
    return abs(lam @ r) / den if den > 0 else 0.0
loc = np.abs(r) / np.maximum(S.absA @ ub, 1e-300)
best_cert, cert_src = float(np.max(loc)), f"unit vector row {int(np.argmax(loc))}"
log(f"local weighted ratio (unit vectors): {best_cert:.6f}  (> 1 would prove infeasibility)")

tau = 1 / np.maximum(np.asarray(S.absA.sum(axis=0)).ravel(), 1e-300)
sgm = 1 / np.maximum(np.asarray(S.absA.sum(axis=1)).ravel(), 1e-300)
g = args.gamma
c = np.clip(cw, -ub, ub); lam = np.zeros(m)                  # start: the witness clipped into the box
c_avg = np.zeros(N); lam_avg = np.zeros(m); navg = 0
status = "inconclusive (wall)"; it = 0; best = None
def physical_check(cc, tag):
    ph = S.physical(cc)
    cor = ph["cor"]; fmax = float(np.max(np.abs(cor))); n3 = int(np.sum(np.abs(cor) > 3))
    ok = ph["valid"] and fmax <= 5 and n3 <= K
    log(f"  physical check [{tag}]: CG info {ph['cg_info']}, ‖Ac − r‖∞ {ph['eq_res']:.1e}, mismatch {ph['rec_mismatch']:.1e}σ, "
        f"div_rel {ph['div_rel']:.1e} | CENTRE max {fmax:.5f}σ, >3σ {n3} ({n3 / N:.4%}; K {K}; {int(np.sum((np.abs(cor) > 3) & ~supp))} off S), "
        f"#>5σ {int(np.sum(np.abs(cor) > 5))}, mean c² {np.mean(cor ** 2):.5f} → {'BOTH criteria met' if ok else 'not met'}")
    return ok, ph, fmax, n3
while time.time() - T0 < args.wall:
    it += 1
    v = c - tau * (A.T @ lam)
    c_new = np.clip(v / (1 + tau * g), -ub, ub)
    lam = lam + sgm * (A @ (2 * c_new - c) - r)
    c = c_new
    c_avg += c; lam_avg += lam; navg += 1
    if it % args.report == 0:
        ca = c_avg / navg
        res_l = np.max(np.abs(A @ c - r)) / rinf; res_a = np.max(np.abs(A @ ca - r)) / rinf
        for ll, nm in ((lam, "last"), (lam_avg / navg, "avg")):
            cr = cert_ratio(ll)
            if cr > best_cert:
                best_cert, cert_src = cr, f"PDHG {nm} multiplier at it {it}"
        log(f"it {it:7d}  ‖Ac − r‖∞/‖r‖∞ last {res_l:.2e} avg {res_a:.2e}  ‖c‖² mean {np.mean(c ** 2):.5f}  ‖λ‖∞ {np.max(np.abs(lam)):.2e}  "
            f"infeasibility ratio best {best_cert:.5f} ({cert_src})")
        if best_cert > 1:
            status = "INFEASIBLE for this support (weak-duality ratio > 1)"; break
        if it % args.check_every == 0 or min(res_l, res_a) <= args.tol:
            cc = ca if res_a <= res_l else c
            ok, ph, fmax, n3 = physical_check(cc, f"it {it}")
            if best is None or (fmax, n3) < (best[2], best[3]):
                best = (cc.copy(), ph, fmax, n3)
                np.savez(args.out.replace(".npz", "") + "_incumbent.npz", B_face=np.concatenate([v.ravel() for v in ph["B"]]),
                         c_centre=ph["cor"], centre_max=fmax, centre_n3=n3, it=it)
            if ok:
                status = "FEASIBLE for this support (physical field meets both centre criteria)"; break
    if it % args.restart == 0:
        c, lam = c_avg / navg, lam_avg / navg; c_avg[:] = 0; lam_avg[:] = 0; navg = 0
log(f"PDHG stopped at it {it}: {status}")
if best is None or not status.startswith("FEASIBLE"):
    ok, ph, fmax, n3 = physical_check(c, "final iterate")
    if best is None or (fmax, n3) < (best[2], best[3]):
        best = (c.copy(), ph, fmax, n3)
cc, ph, fmax, n3 = best
log(f"RESULT: {status}; best physical field: CENTRE max {fmax:.6f}σ, >3σ {n3} (K {K}), div_rel {ph['div_rel']:.2e}, "
    f"valid {ph['valid']}; best infeasibility ratio {best_cert:.5f} ({cert_src})")
np.savez(args.out, B_face=np.concatenate([v.ravel() for v in ph["B"]]), c_centre=ph["cor"], c_iterate=cc, support_centre=supp, ub_centre=ub,
         status=status, it=it, best_cert=best_cert, cert_src=cert_src, div_rel=ph["div_rel"], eq_res=ph["eq_res"],
         rec_mismatch=ph["rec_mismatch"], cg_info=ph["cg_info"], centre_max=fmax, centre_n3=n3)
log(f"saved {args.out}")
