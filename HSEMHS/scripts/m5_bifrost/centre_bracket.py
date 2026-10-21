#!/usr/bin/env python3
"""
Centre-model maximum feasibility: an anytime, numerically verified bracket on
    t* = min max_i |c_i|,   c = (H B − y_c)/σ_c,   over divergence-free face fields B
(Codex M5e review §3; corrected after the M5f review, findings 3 and 6).

Uses Codex's reduced centre formulation (codex_m5e_centre_feasibility.py, uniform periodic x/y): with C = H B,
L C = 0 (vertical pairing of q = Hy Dx Cx + Hx Dy Cy with the Cz difference, plus the horizontal-Nyquist rows) is
necessary and sufficient for C to come from a divergence-free B. In c-variables: A c = r, A = L diag(σ_c), r = −L y_c.

  LOWER bound, for ANY multiplier λ (weak duality): for every feasible c, |λᵀr| = |(Aᵀλ)ᵀc| ≤ ‖Aᵀλ‖₁ ‖c‖∞, so
        t* ≥ |λᵀr| / ‖Aᵀλ‖₁.
  The saddle function of the PDHG below is ‖c‖∞ + λᵀ(Ac − r), whose dual objective is −λᵀr: the useful sign of the
  iterate is −λ; the absolute value takes both signs (M5f review finding 3: the M5f script used +λᵀr and reported a
  negative, uninformative bound). The bound is seeded with the necessary local bound max_i |r_i|/‖A_i‖₁ (λ = unit
  vector), whose row index is saved as its certificate.
  UPPER bound: project c onto {A c = r} (CG on A Aᵀ), reconstruct B from C = y + σ c (Fourier inversion of Hx, Hy and the
  divergence recurrence), recompute H B and take the TRUE max |c| of that physical field. Every evaluation records CG's
  exit code, the equality residual ‖Ac − r‖∞, the reconstruction mismatch max|H B − C|/σ, finiteness and the physical
  max |div B|; a bound is accepted as an upper bound only if the field is finite and its divergence passes the
  tolerance (relative to the field scale) — an imperfect projection does not invalidate a physical field that is
  independently divergence-free. Every improved upper bound saves its physical field (B_face) at once.
So t* ∈ [best LB, best UB] at all times, numerically (no rigorous rounding bounds).

Solver: PDHG (Chambolle–Pock) on min ‖c‖_∞ s.t. Ac = r, with Pock–Chambolle diagonal preconditioning and
restarts to the running average every `--restart` iterations; prox of τ‖·‖_∞ via projection onto the ℓ1 ball.
usage (from HSEMHS/): python3 centre_bracket.py --obs out/m5d/observations.h5 --out out/m5g/centre_bracket.npz
                      [--wall 3000] [--report 200] [--restart 1000] [--reeval OLD.npz] [--warm OLD.npz]
Run under an external wall guard too: a single projection can outlast the loop's wall check.
"""
import argparse, time, sys
import numpy as np, h5py, scipy.sparse as sp, scipy.sparse.linalg as spla

DIV_TOL = 1e-9          # max |div B| relative to median|y| / median(dx)


class CentreSystem:
    """Reduced centre system for centre observations y (nx, ny, nz per component), σ, uniform dx, dy and hz[nz]."""

    def __init__(self, obs, sig, dx, dy, hz):
        self.obs, self.sig = obs, sig
        nx, ny, nz = obs[0].shape
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.hz = np.asarray(dx, float), np.asarray(dy, float), np.asarray(hz, float)
        assert np.all(self.dx == self.dx[0]) and np.all(self.dy == self.dy[0]), "requires uniform x/y"
        n = nx * ny * nz; self.n = n
        ids = np.arange(n).reshape(nx, ny, nz); nr = nx * ny * (nz - 1); row = np.arange(nr)
        rows, cols, vals = [], [], []
        def add(index, coefficient):
            rows.append(row); cols.append(index.ravel()); vals.append(np.broadcast_to(coefficient, (nx, ny, nz - 1)).ravel())
        hz_ = self.hz
        for zoff in (0, 1):                           # (identical to codex_m5e_centre_feasibility.py)
            zsel = slice(zoff, nz - 1 + zoff)
            for xoff in (0, 1):
                for yoff in (0, 1):
                    index = np.roll(np.roll(ids, xoff, 0), yoff, 1)[:, :, zsel]
                    add(index, (-1) ** xoff * .5 * hz_[zsel] / self.dx[0])
                    add(index + n, (-1) ** yoff * .5 * hz_[zsel] / self.dy[0])
                    add(index + 2 * n, (-1) ** zoff * .5)
        L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nr, 3 * n)).tocsr()
        nyq = []
        if nx % 2 == 0:
            rr = np.broadcast_to(np.arange(ny * nz).reshape(1, ny, nz), (nx, ny, nz)); vv = np.broadcast_to((-1.) ** np.arange(nx)[:, None, None], (nx, ny, nz))
            nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), ids.ravel())), shape=(ny * nz, 3 * n)).tocsr())
        if ny % 2 == 0:
            rr = np.broadcast_to(np.arange(nx * nz).reshape(nx, 1, nz), (nx, ny, nz)); vv = np.broadcast_to((-1.) ** np.arange(ny)[None, :, None], (nx, ny, nz))
            nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), (ids + n).ravel())), shape=(nx * nz, 3 * n)).tocsr())
        self.L = sp.vstack([L] + nyq, format="csr")
        self.y = np.concatenate([v.ravel() for v in obs]); self.s = np.concatenate([v.ravel() for v in sig])
        self.A = (self.L @ sp.diags(self.s)).tocsr(); self.r = -(self.L @ self.y)
        self.AAt = spla.LinearOperator((self.A.shape[0],) * 2, matvec=lambda v: self.A @ (self.A.T @ v), dtype=float)
        self.absA = abs(self.A)
        self.bscale = max(np.median(np.abs(self.y)), 1e-300) / self.dx[0]

    # --- physical field ---------------------------------------------------------------------------------------------
    def centre(self, B):
        return [.5 * (B[0] + np.roll(B[0], 1, 0)), .5 * (B[1] + np.roll(B[1], 1, 1)), .5 * (B[2][:, :, :-1] + B[2][:, :, 1:])]

    def divergence(self, B):
        return ((B[0] - np.roll(B[0], 1, 0)) / self.dx[:, None, None] + (B[1] - np.roll(B[1], 1, 1)) / self.dy[None, :, None]
                + (B[2][:, :, :-1] - B[2][:, :, 1:]) / self.hz)

    def reconstruct(self, C):                         # (identical to codex_m5e_centre_feasibility.py)
        nx, ny, nz, dx, dy, hz = self.nx, self.ny, self.nz, self.dx, self.dy, self.hz
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

    # --- bounds -------------------------------------------------------------------------------------------------------
    def lower(self, lam):
        """Weak-duality lower bound |λᵀr| / ‖Aᵀλ‖₁, valid for any λ (both signs)."""
        den = np.sum(np.abs(self.A.T @ lam))
        return abs(lam @ self.r) / den if den > 0 else 0.0

    def local_bound(self):
        """max_i |r_i| / ‖A_i‖₁ (λ = ± unit vector) and its row index (the certificate)."""
        q = np.abs(self.r) / np.maximum(np.asarray(self.absA.sum(axis=1)).ravel(), 1e-300)
        i = int(np.argmax(q)); return float(q[i]), i

    def project(self, c, maxiter=20000, rtol=1e-13):
        """Minimum-norm correction onto {A c = r}: c + Aᵀ (A Aᵀ)⁻¹ (r − A c) by CG. Returns (c, CG info, ‖Ac − r‖∞).
        A residual already at rounding level (relative to ‖A‖∞ max(‖c‖∞, 1)) is not iterated on: the relative CG
        tolerance is unattainable there."""
        rhs = self.r - self.A @ c
        scale = float(np.max(np.asarray(self.absA.sum(axis=1)).ravel())) * max(float(np.max(np.abs(c))), 1.0)
        if np.max(np.abs(rhs)) <= 1e-14 * scale:
            return c.copy(), 0, float(np.max(np.abs(rhs)))
        z, info = spla.cg(self.AAt, rhs, rtol=rtol, atol=1e-14 * scale, maxiter=maxiter)
        cp = c + self.A.T @ z
        return cp, int(info), float(np.max(np.abs(self.A @ cp - self.r)))

    def physical(self, c, proj=True, maxiter=20000):
        """Physical field from c and its checks: dict(ub, cor, B, cg_info, eq_res, rec_mismatch, div_rel, finite, valid)."""
        info, eq = 0, float(np.max(np.abs(self.A @ c - self.r)))
        if proj:
            c, info, eq = self.project(c, maxiter=maxiter)
        n, nx, ny, nz = self.n, self.nx, self.ny, self.nz
        C = [(self.y + self.s * c)[q * n:(q + 1) * n].reshape(nx, ny, nz) for q in range(3)]
        B = self.reconstruct(C); pred = self.centre(B)
        cor = np.concatenate([((v - o) / sg).ravel() for v, o, sg in zip(pred, self.obs, self.sig)])
        rec = max(float(np.max(np.abs((p - t) / sg))) for p, t, sg in zip(pred, C, self.sig))
        finite = bool(all(np.all(np.isfinite(v)) for v in B) and np.all(np.isfinite(cor)))
        div_rel = float(np.max(np.abs(self.divergence(B)))) / self.bscale if finite else np.inf
        return dict(ub=float(np.max(np.abs(cor))) if finite else np.inf, cor=cor, B=B, cg_info=info, eq_res=eq,
                    rec_mismatch=rec, div_rel=div_rel, finite=finite, valid=finite and div_rel <= DIV_TOL)


def prox_inf(v, tw):
    """prox of ‖·‖_∞ in the diag(1/tw) metric: argmin_c ‖c‖_∞ + Σ (c_j − v_j)²/(2 tw_j) = clip(v, −t, t)."""
    a = np.abs(v); order = np.argsort(-a); a_s = a[order]; w_s = 1 / tw[order]
    cw = np.cumsum(w_s); cwa = np.cumsum(w_s * a_s)
    tcand = (cwa - 1) / cw
    nxt = np.append(a_s[1:], 0.0)
    k = np.nonzero((tcand >= nxt) & (tcand <= a_s))[0]
    t = max(tcand[k[0]], 0.0) if k.size else 0.0
    return np.clip(v, -t, t)


def pdhg_step(A, r, c, lam, tau, sgm):
    """One Chambolle–Pock step on min ‖c‖∞ + λᵀ(Ac − r) (dual ascent in λ)."""
    c_new = prox_inf(c - tau * (A.T @ lam), tau)
    lam = lam + sgm * (A @ (2 * c_new - c) - r)
    return c_new, lam


def label(LB, ub_valid, UB, t=5.0):
    if LB > t:
        return "INFEASIBLE at t = %g (LB > t)" % t
    if ub_valid and UB <= t:
        return "FEASIBLE at t = %g (verified UB ≤ t)" % t
    return "UNRESOLVED at t = %g" % t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall", type=float, default=3000.0)
    ap.add_argument("--report", type=int, default=200)
    ap.add_argument("--restart", type=int, default=1000)
    ap.add_argument("--max-iter", type=int, default=10 ** 9)
    ap.add_argument("--reeval", default="", help="re-evaluate the multipliers / field saved by an earlier run, then continue")
    ap.add_argument("--warm", default="", help="start PDHG from an earlier run's best_c / best_lam")
    args = ap.parse_args()
    T0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:7.0f} s]", *a, flush=True)

    with h5py.File(args.obs) as f:
        obs = [f[f"centre_obs{c}"][()].T for c in (1, 2, 3)]
        sig = [f[f"centre_sigma{c}"][()].T for c in (1, 2, 3)]
        dx, dy, hz = [f[k][:] for k in ("dxc", "dyc", "hz")]
    S = CentreSystem(obs, sig, dx, dy, hz)
    A, r = S.A, S.r; m, N = A.shape
    log(f"reduced centre system {m} × {N}, nnz {A.nnz}")
    loc, iloc = S.local_bound()
    bestLB, best_lam, lb_cert = loc, np.zeros(m), f"unit vector row {iloc}"
    best_lam[iloc] = np.sign(r[iloc]) or 1.0
    log(f"necessary local bound (λ = unit vector, row {iloc}): {loc:.6f}")
    bestUB, best_c, best_rec = np.inf, np.zeros(N), None
    inc_path = args.out.replace(".npz", "") + "_incumbent.npz"

    def offer_ub(cc, tag):
        nonlocal bestUB, best_c, best_rec
        ph = S.physical(cc)
        if ph["cg_info"] != 0:
            log(f"  [{tag}] CG did not converge (info {ph['cg_info']}), ‖Ac − r‖∞ {ph['eq_res']:.2e}")
        if ph["valid"] and ph["ub"] < bestUB:
            bestUB, best_c = ph["ub"], cc.copy()
            best_rec = {k: ph[k] for k in ("cg_info", "eq_res", "rec_mismatch", "div_rel", "finite")}
            np.savez(inc_path, B_face=np.concatenate([v.ravel() for v in ph["B"]]), c_centre=ph["cor"], UB=bestUB, **best_rec)
        elif not ph["valid"]:
            log(f"  [{tag}] field rejected: finite {ph['finite']}, div_rel {ph['div_rel']:.2e} (tol {DIV_TOL:.0e})")
        return ph

    def offer_lb(ll, tag):
        nonlocal bestLB, best_lam, lb_cert
        lb = S.lower(ll)
        if lb > bestLB:
            bestLB, best_lam, lb_cert = lb, ll.copy(), tag
        return lb

    c = np.zeros(N); lam = np.zeros(m)
    if args.reeval:
        old = np.load(args.reeval)
        if "best_lam" in old:
            ol = old["best_lam"]
            log(f"re-evaluated saved multiplier of {args.reeval}: old (+λᵀr) bound {old['LB']:.6f} → |λᵀr|/‖Aᵀλ‖₁ = {S.lower(ol):.6f} "
                f"(λᵀr {ol @ r:+.4e})")
            offer_lb(ol, f"saved multiplier of {args.reeval}")
        if "best_c" in old:
            ph = offer_ub(old["best_c"], "saved best_c")
            log(f"re-verified saved best_c: UB {ph['ub']:.6f}, CG info {ph['cg_info']}, ‖Ac − r‖∞ {ph['eq_res']:.2e}, "
                f"reconstruction mismatch {ph['rec_mismatch']:.2e}σ, div_rel {ph['div_rel']:.2e}, valid {ph['valid']}")
    if args.warm:
        old = np.load(args.warm); c = old["best_c"].copy(); lam = old["best_lam"].copy()
        log(f"warm start from {args.warm}")

    tau = 1 / np.maximum(np.asarray(S.absA.sum(axis=0)).ravel(), 1e-300)
    sgm = 1 / np.maximum(np.asarray(S.absA.sum(axis=1)).ravel(), 1e-300)
    c_avg = np.zeros(N); lam_avg = np.zeros(m); navg = 0
    it = 0
    while it < args.max_iter and time.time() - T0 < args.wall:
        it += 1
        c, lam = pdhg_step(A, r, c, lam, tau, sgm)
        c_avg += c; lam_avg += lam; navg += 1
        if it % args.report == 0:
            for name, cc, ll in (("last", c, lam), ("avg", c_avg / navg, lam_avg / navg)):
                offer_lb(ll, f"PDHG {name} multiplier, it {it}")
                if time.time() - T0 < args.wall:
                    offer_ub(cc, f"{name} it {it}")
            log(f"it {it:7d}  ‖c‖∞ {np.max(np.abs(c)):.5f}  ‖Ac − r‖∞ {np.max(np.abs(A @ c - r)):.2e}  λᵀr {lam @ r:+.3e}  |  "
                f"bracket t* ∈ [{bestLB:.5f}, {bestUB:.5f}]")
            np.savez(args.out, best_lam=best_lam, best_c=best_c, LB=bestLB, UB=bestUB, it=it, lb_cert=lb_cert)
        if it % args.restart == 0:                       # restart to the average
            c, lam = c_avg / navg, lam_avg / navg; c_avg[:] = 0; lam_avg[:] = 0; navg = 0
    log(f"stopped at it {it} (wall {args.wall:.0f} s or max-iter)")
    lb = S.lower(best_lam) if not lb_cert.startswith("unit") else S.local_bound()[0]
    ph = S.physical(best_c) if np.isfinite(bestUB) else None
    ub_ok = ph is not None and ph["valid"]
    ub = ph["ub"] if ph is not None else np.inf
    log(f"FINAL (re-verified, numerical): t* ∈ [{lb:.6f}, {ub:.6f}]; LB certificate: {lb_cert}; "
        + (f"UB field: CG info {ph['cg_info']}, ‖Ac − r‖∞ {ph['eq_res']:.2e}, reconstruction mismatch {ph['rec_mismatch']:.2e}σ, "
           f"div_rel {ph['div_rel']:.2e}, finite {ph['finite']}, >3σ {np.mean(np.abs(ph['cor']) > 3):.4%}; " if ph is not None else "no UB field; ")
        + label(lb, ub_ok, ub))
    extra = dict(B_face=np.concatenate([v.ravel() for v in ph["B"]]), c_centre=ph["cor"], div_rel=ph["div_rel"],
                 eq_res=ph["eq_res"], rec_mismatch=ph["rec_mismatch"], cg_info=ph["cg_info"]) if ph is not None else {}
    np.savez(args.out, best_lam=best_lam, best_c=best_c, LB=lb, UB=ub, UB_valid=ub_ok, it=it, lb_cert=lb_cert, **extra)
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
