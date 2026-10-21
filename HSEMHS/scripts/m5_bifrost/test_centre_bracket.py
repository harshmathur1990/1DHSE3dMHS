#!/usr/bin/env python3
"""Checks of centre_bracket.py (Codex M5f review findings 3 and 6): the dual-bound sign on problems with known optima,
an end-to-end known divergence-free field, an invalid field and a failed projection.
usage (from HSEMHS/): python3 -I scripts/m5_bifrost/test_centre_bracket.py"""
import os, sys
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import centre_bracket as cb

fails = 0
def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what); fails += 0 if cond else 1

def tiny(A, r):
    S = object.__new__(cb.CentreSystem); S.A = sp.csr_matrix(np.asarray(A, float)); S.r = np.asarray(r, float); S.absA = abs(S.A)
    return S

def run_pdhg(S, iters):
    tau = 1 / np.asarray(S.absA.sum(axis=0)).ravel(); sgm = 1 / np.asarray(S.absA.sum(axis=1)).ravel()
    c = np.zeros(S.A.shape[1]); lam = np.zeros(S.A.shape[0]); best = 0.0
    for _ in range(iters):
        c, lam = cb.pdhg_step(S.A, S.r, c, lam, tau, sgm); best = max(best, S.lower(lam))
    return c, lam, best

# 1. scalar A = 1, r = 1 (optimum 1): the PDHG multiplier is −1; the corrected bound is 1, the M5f (+λᵀr) bound was −1
S = tiny([[1.0]], [1.0])
c, lam, best = run_pdhg(S, 50)
check(abs(c[0] - 1) < 1e-12 and abs(lam[0] + 1) < 1e-12, f"scalar: c {c[0]:.3g}, λ {lam[0]:.3g}")
check(abs(S.lower(lam) - 1) < 1e-12, f"scalar: corrected bound {S.lower(lam):.6f} = 1")
check((lam @ S.r) / np.sum(np.abs(S.A.T @ lam)) < 0, "scalar: the M5f sign gives a negative bound")
check(abs(S.local_bound()[0] - 1) < 1e-15, "scalar: local bound 1")

# 2. small LP: c1 + c2 = 2, c2 + c3 = 0 (optimum 1 at c = (1, 1, −1)); local bound 1 as well
S = tiny([[1, 1, 0], [0, 1, 1]], [2, 0])
c, lam, best = run_pdhg(S, 4000)
check(best <= 1 + 1e-12 and best > 1 - 1e-3, f"LP: best lower bound {best:.6f} ∈ (0.999, 1]")
check(abs(np.max(np.abs(c)) - 1) < 1e-3 and np.max(np.abs(S.A @ c - S.r)) < 1e-3, f"LP: ‖c‖∞ {np.max(np.abs(c)):.6f}")
# a problem whose optimum exceeds the local bound: c1 + c2 = 2, c1 − c2 = 2 → c = (2, 0), t* = 2; local bound 1
S = tiny([[1, 1], [1, -1]], [2, 2])
c, lam, best = run_pdhg(S, 4000)
check(abs(S.local_bound()[0] - 1) < 1e-15 and best > 2 - 1e-3 and best <= 2 + 1e-12, f"LP2: lower bound {best:.6f} → 2 (local bound 1)")

# 3. end-to-end known divergence-free field on a 4 × 4 × 3 periodic grid
rng = np.random.default_rng(1)
nx, ny, nz = 4, 4, 3; dx = np.full(nx, 2.0); dy = np.full(ny, 2.0); hz = np.array([1.0, 1.5, 1.2])
psi = rng.standard_normal((nx, ny, nz)); Ay = rng.standard_normal((nx, ny, nz + 1))
Bx = (psi - np.roll(psi, 1, 1)) / dy[0] - (Ay[:, :, :-1] - Ay[:, :, 1:]) / hz          # curl of (0, Ay(z-edges), ψ)
By = -(psi - np.roll(psi, 1, 0)) / dx[0]
Bz = (Ay - np.roll(Ay, 1, 0)) / dx[0] + 0.7
Bx = Bx + 0.3
tmp = cb.CentreSystem.__new__(cb.CentreSystem); tmp.dx, tmp.dy, tmp.hz = dx, dy, hz
check(np.max(np.abs(cb.CentreSystem.divergence(tmp, [Bx, By, Bz]))) < 1e-12, "known field is divergence-free")
C = cb.CentreSystem.centre(tmp, [Bx, By, Bz]); sig = [np.full((nx, ny, nz), 0.01) for _ in range(3)]
S = cb.CentreSystem(C, sig, dx, dy, hz)
check(np.max(np.abs(S.r)) < 1e-12, f"known field satisfies L C = 0 (‖r‖∞ {np.max(np.abs(S.r)):.1e})")
ph = S.physical(np.zeros(3 * nx * ny * nz))
check(ph["valid"] and ph["ub"] < 1e-9 and ph["div_rel"] < 1e-12 and ph["rec_mismatch"] < 1e-9 and ph["cg_info"] == 0,
      f"known field: UB {ph['ub']:.1e}, div_rel {ph['div_rel']:.1e}, mismatch {ph['rec_mismatch']:.1e}, CG {ph['cg_info']}")
check(cb.label(0.0, ph["valid"], ph["ub"]).startswith("FEASIBLE"), "known field labelled feasible")

# 4. invalid field: a reconstruction that is not divergence-free is rejected, whatever its residual
orig = S.reconstruct
S.reconstruct = lambda CC: [orig(CC)[0] + 1e-3 * rng.standard_normal((nx, ny, nz)), orig(CC)[1], orig(CC)[2]]
ph = S.physical(np.zeros(3 * nx * ny * nz))
check(not ph["valid"] and ph["div_rel"] > cb.DIV_TOL, f"invalid field rejected (div_rel {ph['div_rel']:.1e})")
check(cb.label(0.0, ph["valid"], ph["ub"]).startswith("UNRESOLVED"), "invalid field not labelled feasible")
nanf = lambda CC: [np.full((nx, ny, nz), np.nan), orig(CC)[1], orig(CC)[2]]
S.reconstruct = nanf
ph = S.physical(np.zeros(3 * nx * ny * nz))
check(not ph["finite"] and not ph["valid"] and ph["ub"] == np.inf, "non-finite field rejected")
S.reconstruct = orig

# 5. failed projection: an infeasible start with maxiter = 1 is recorded (CG info ≠ 0, equality residual reported)
Cn = [v + 0.05 * rng.standard_normal(v.shape) for v in C]
S = cb.CentreSystem(Cn, sig, dx, dy, hz)
ph = S.physical(np.zeros(3 * nx * ny * nz), maxiter=1)
check(ph["cg_info"] != 0 and ph["eq_res"] > 1e-8, f"failed projection recorded: CG info {ph['cg_info']}, ‖Ac − r‖∞ {ph['eq_res']:.1e}")
ph = S.physical(np.zeros(3 * nx * ny * nz))
check(ph["cg_info"] == 0 and ph["eq_res"] < 1e-10 and ph["valid"], f"converged projection: ‖Ac − r‖∞ {ph['eq_res']:.1e}, valid {ph['valid']}")

print(f"{fails} failure(s)")
sys.exit(1 if fails else 0)
