#!/usr/bin/env python3
"""
Exact data-only magnetic feasibility LP for the coarsened Bifrost cube (server version).

Question (CODEX_REVIEW_M5c_2026-10-18.md §4; CODEX_BRIEF_M5d_2026-10-18_1430.md Q2):
    is there a field B on the staggered faces with  div B = 0  (every cell)  and  |B_f − I·y_f| ≤ t σ_f ?
Variables are the normalized face corrections c_f = (B_f − I·y_f)/σ_f, so the constraints are
    sum_f a_cf σ_f c_f = −div(I·y)_c     (one equality per cell, scaled by its row norm)
    |c_f| ≤ t.
This is the formulation of Codex's codex_m5c_face_feasibility.py, with the SAME preprocessing as the Julia
pipeline (scripts/m5_bifrost/bif_setup.jl): 12×12 horizontal block means, every 2nd level, periodic x/y, face
interpolation by two-point means, the top and bottom z-faces taking the nearest centre value, σ per component
sqrt((0.05|B|)² + (5e-4 T)²), face σ = geometric mean of the two centre σ.

Modes
    feas   --t T    feasibility at t = T (default 5). Status "optimal" = a witness field exists (saved);
                    "infeasible" = no divergence-free field meets the bound (HiGHS reports the dual ray when it
                    can, which certifies infeasibility).
    mint            minimize t exactly (adds t and 2·N_f inequality rows).
    bisect --lo --hi --tol   bisection on t with feasibility solves (each solve as in `feas`).
    minl1  --t T    minimize Σ|c| subject to |c| ≤ T: the convex proxy for a small tail (the spec's data_admissible
                    also needs ≤ 1 % of faces above 3σ, which a bare feasibility vertex does not try to meet).
    minhinge --t T --knee K   minimize Σ max(|c| − K, 0) subject to |c| ≤ T: the direct convex surrogate for
                    the number of faces above K σ.
Every solution found is verified independently from the saved field: the maximum scaled equality residual,
max |c|, the fraction above 3σ and the count above 5σ (the spec's data_admissible needs max ≤ 5 and ≤ 1 % > 3).

Requirements: python3, numpy, scipy, h5py; highspy recommended (pip install highspy) for threads, the interior-
point method and dual rays. Without highspy, scipy.optimize.linprog (HiGHS, no thread control) is used.

Examples (on the server; IO/input-mhs-hse.nc must be copied there):
    python3 data_feasibility_lp.py --input input-mhs-hse.nc --mode feas --t 5 --threads 64 --solver ipm --out feas_t5.npz
    python3 data_feasibility_lp.py --input input-mhs-hse.nc --mode mint --threads 64 --solver ipm --out mint.npz
    python3 data_feasibility_lp.py --input input-mhs-hse.nc --mode feas --t 5 --write-mps feas_t5.mps   # for Gurobi/MOSEK
"""
import argparse, sys, time
import numpy as np
import h5py
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--mode", choices=("feas", "mint", "bisect", "minl1", "minhinge"), default="feas")
ap.add_argument("--knee", type=float, default=3.0, help="hinge knee for --mode minhinge")
ap.add_argument("--t", type=float, default=5.0)
ap.add_argument("--lo", type=float, default=1.92)
ap.add_argument("--hi", type=float, default=5.2)
ap.add_argument("--tol", type=float, default=0.02)
ap.add_argument("--threads", type=int, default=0, help="HiGHS threads (0: HiGHS default)")
ap.add_argument("--solver", choices=("ipm", "simplex", "choose"), default="ipm")
ap.add_argument("--time-limit", type=float, default=float("inf"))
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--zstride", type=int, default=2)
ap.add_argument("--write-mps", default="")
ap.add_argument("--out", default="")
args = ap.parse_args()

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------------------------------------------------------- preprocessing (as bif_setup.jl)
c = args.block
with h5py.File(args.input, "r") as f:
    zall = f["z"][:].astype(float)                       # top first
    nzall = len(zall)
    ks = list(range(0, nzall, args.zstride))
    if ks[-1] != nzall - 1:
        ks.append(nzall - 1)                             # keep the bottom (anchor) plane, as bif_setup.jl
    ks = np.array(ks)
    z = zall[ks]
    x = f["x"][:].astype(float); y = f["y"][:].astype(float)
    NX = len(x) // c; NY = len(y) // c
    x = x[:NX * c].reshape(NX, c).mean(axis=1); y = y[:NY * c].reshape(NY, c).mean(axis=1)
    B = []
    for name in ("B_x", "B_y", "B_z"):
        a = f[name][:NX * c, :NY * c, :][:, :, ks]       # Python order (x, y, z)
        B.append(a.reshape(NX, c, NY, c, len(ks)).mean(axis=(1, 3), dtype=np.float64))
nx, ny, nz = B[0].shape
log(f"coarse grid {nx}×{ny}×{nz}")
sig = np.sqrt((0.05 * np.sqrt(B[0]**2 + B[1]**2 + B[2]**2))**2 + (5e-4)**2)
obs = [0.5 * (B[0] + np.roll(B[0], -1, axis=0)),
       0.5 * (B[1] + np.roll(B[1], -1, axis=1)),
       np.concatenate([B[2][:, :, :1], 0.5 * (B[2][:, :, :-1] + B[2][:, :, 1:]), B[2][:, :, -1:]], axis=2)]
sf = [np.sqrt(sig * np.roll(sig, -1, axis=0)),
      np.sqrt(sig * np.roll(sig, -1, axis=1)),
      np.concatenate([sig[:, :, :1], np.sqrt(sig[:, :, :-1] * sig[:, :, 1:]), sig[:, :, -1:]], axis=2)]
dx = float(np.mean(np.diff(x))); dy = float(np.mean(np.diff(y)))
zf = np.concatenate([[z[0] + 0.5 * (z[0] - z[1])], 0.5 * (z[:-1] + z[1:]), [z[-1] - 0.5 * (z[-2] - z[-1])]])
hz = (zf[:-1] - zf[1:])[None, None, :]
ids = np.arange(nx * ny * nz).reshape(nx, ny, nz); n = ids.size
idz = np.arange(nx * ny * (nz + 1)).reshape(nx, ny, nz + 1) + 2 * n
nf = 3 * n + nx * ny
cols = [ids, np.roll(ids, 1, axis=0), ids + n, np.roll(ids + n, 1, axis=1), idz[:, :, :-1], idz[:, :, 1:]]
coeff = [sf[0] / dx, -np.roll(sf[0], 1, axis=0) / dx, sf[1] / dy, -np.roll(sf[1], 1, axis=1) / dy,
         sf[2][:, :, :-1] / hz, -sf[2][:, :, 1:] / hz]
den = sum(np.abs(q) for q in coeff)
div = (obs[0] - np.roll(obs[0], 1, axis=0)) / dx + (obs[1] - np.roll(obs[1], 1, axis=1)) / dy + \
      (obs[2][:, :, :-1] - obs[2][:, :, 1:]) / hz
Aeq = sp.coo_matrix((np.concatenate([(q / den).ravel() for q in coeff]),
                     (np.tile(ids.ravel(), 6), np.concatenate([q.ravel() for q in cols]))), shape=(n, nf)).tocsr()
rhs = (-div / den).ravel()
log(f"LP: {nf} face variables, {n} equalities, {Aeq.nnz} nonzeros; necessary bound t ≥ {np.max(np.abs(rhs)):.6f}")

def verify(cv, tag):
    res = np.max(np.abs(Aeq @ cv - rhs))
    a = np.abs(cv)
    k = int(np.argmax(a))
    comp = 0 if k < n else (1 if k < 2 * n else 2)
    log(f"[{tag}] verified: max |scaled div residual| {res:.3e}; max |c| {a.max():.6f} (component {'xyz'[comp]}, flat index {k}); "
        f"fraction > 3σ {np.mean(a > 3):.4e}; count > 5σ {int(np.sum(a > 5 + 1e-9))}; mean c² {np.mean(cv**2):.4f}")
    return res, a.max()

# ---------------------------------------------------------------- solvers
try:
    import highspy
    HAVE_HIGHSPY = True
except ImportError:
    HAVE_HIGHSPY = False
    log("highspy not available: using scipy.optimize.linprog (no thread control, no dual ray)")

def solve(t, minimize_t=False, mps=""):
    """Returns (status_string, c or None, t_opt or None)."""
    if HAVE_HIGHSPY:
        h = highspy.Highs()
        h.setOptionValue("output_flag", True)
        if args.threads > 0:
            h.setOptionValue("threads", args.threads)
        h.setOptionValue("solver", args.solver)
        if np.isfinite(args.time_limit):
            h.setOptionValue("time_limit", args.time_limit)
        inf = highspy.kHighsInf
        lp = highspy.HighsLp()
        if minimize_t:
            nv = nf + 1
            cost = np.zeros(nv); cost[-1] = 1.0
            lo = np.full(nv, -inf); hi = np.full(nv, inf); lo[-1] = 0.0
            I = sp.identity(nf, format="csr"); tcol = sp.csr_matrix(np.ones((nf, 1)))
            Amat = sp.vstack([sp.hstack([Aeq, sp.csr_matrix((n, 1))]), sp.hstack([I, -tcol]), sp.hstack([-I, -tcol])]).tocsc()
            rlo = np.concatenate([rhs, np.full(2 * nf, -inf)]); rhi = np.concatenate([rhs, np.zeros(2 * nf)])
        else:
            nv = nf
            cost = np.zeros(nv); lo = np.full(nv, -t); hi = np.full(nv, t)
            Amat = Aeq.tocsc(); rlo = rhs; rhi = rhs
        lp.num_col_ = nv; lp.num_row_ = Amat.shape[0]
        lp.col_cost_ = cost; lp.col_lower_ = lo; lp.col_upper_ = hi
        lp.row_lower_ = rlo; lp.row_upper_ = rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = Amat.indptr; lp.a_matrix_.index_ = Amat.indices; lp.a_matrix_.value_ = Amat.data
        h.passModel(lp)
        if mps:
            h.writeModel(mps); log(f"wrote {mps}")
            return "written", None, None
        tm = time.time(); h.run()
        st = h.modelStatusToString(h.getModelStatus())
        log(f"HiGHS status: {st} ({time.time() - tm:.0f} s)")
        if st == "Optimal":
            sol = np.array(h.getSolution().col_value)
            return "optimal", sol[:nf], (sol[-1] if minimize_t else None)
        if st == "Infeasible":
            try:
                has, ray = h.getDualRay()
                if has:
                    ray = np.array(ray)
                    log(f"dual ray available (certificate of infeasibility), ‖ray‖∞ {np.max(np.abs(ray)):.3e}")
                    if args.out:
                        np.savez(args.out.replace(".npz", "_dualray.npz"), ray=ray, t=t)
            except Exception as e:
                log(f"dual ray not retrieved: {e}")
            return "infeasible", None, None
        return st, None, None
    # scipy fallback
    from scipy.optimize import linprog
    opts = {"primal_feasibility_tolerance": 1e-9}
    if np.isfinite(args.time_limit):
        opts["time_limit"] = args.time_limit
    meth = {"ipm": "highs-ipm", "simplex": "highs-ds", "choose": "highs"}[args.solver]
    tm = time.time()
    if minimize_t:
        cost = np.zeros(nf + 1); cost[-1] = 1.0
        I = sp.identity(nf, format="csr"); tcol = sp.csr_matrix(np.ones((nf, 1)))
        Aub = sp.vstack([sp.hstack([I, -tcol]), sp.hstack([-I, -tcol])]).tocsr()
        Ae = sp.hstack([Aeq, sp.csr_matrix((n, 1))]).tocsr()
        res = linprog(cost, A_ub=Aub, b_ub=np.zeros(2 * nf), A_eq=Ae, b_eq=rhs,
                      bounds=[(None, None)] * nf + [(0, None)], method=meth, options=opts)
    else:
        res = linprog(np.zeros(nf), A_eq=Aeq, b_eq=rhs, bounds=(-t, t), method=meth, options=opts)
    log(f"linprog status {res.status}: {res.message} ({time.time() - tm:.0f} s)")
    if res.status == 0:
        return "optimal", res.x[:nf], (res.x[-1] if minimize_t else None)
    if res.status == 2:
        return "infeasible", None, None
    return f"status {res.status}", None, None

def solve_minl1(t):
    """min Σ(c⁺ + c⁻) s.t. Aeq(c⁺ − c⁻) = rhs, 0 ≤ c± ≤ t (scipy/HiGHS)."""
    from scipy.optimize import linprog
    meth = {"ipm": "highs-ipm", "simplex": "highs-ds", "choose": "highs"}[args.solver]
    opts = {"primal_feasibility_tolerance": 1e-9}
    if np.isfinite(args.time_limit):
        opts["time_limit"] = args.time_limit
    tm = time.time()
    res = linprog(np.ones(2 * nf), A_eq=sp.hstack([Aeq, -Aeq]).tocsr(), b_eq=rhs, bounds=(0, t), method=meth, options=opts)
    log(f"min-L1 linprog status {res.status}: {res.message} ({time.time() - tm:.0f} s)")
    return (res.x[:nf] - res.x[nf:]) if res.status == 0 else None

def solve_minhinge(t, knee):
    """min Σ s s.t. Aeq c = rhs, |c| ≤ t, s ≥ c − knee, s ≥ −c − knee, s ≥ 0 (scipy/HiGHS)."""
    from scipy.optimize import linprog
    meth = {"ipm": "highs-ipm", "simplex": "highs-ds", "choose": "highs"}[args.solver]
    opts = {"primal_feasibility_tolerance": 1e-9}
    if np.isfinite(args.time_limit):
        opts["time_limit"] = args.time_limit
    I = sp.identity(nf, format="csr")
    Aub = sp.vstack([sp.hstack([I, -I]), sp.hstack([-I, -I])]).tocsr()
    bub = np.full(2 * nf, knee)
    Ae = sp.hstack([Aeq, sp.csr_matrix((n, nf))]).tocsr()
    cost = np.concatenate([np.zeros(nf), np.ones(nf)])
    tm = time.time()
    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Ae, b_eq=rhs, bounds=[(-t, t)] * nf + [(0, None)] * nf, method=meth, options=opts)
    log(f"min-hinge linprog status {res.status}: {res.message} ({time.time() - tm:.0f} s); objective Σ max(|c|−{knee},0) = {res.fun if res.status == 0 else float('nan')}")
    return res.x[:nf] if res.status == 0 else None

# ---------------------------------------------------------------- run
if args.mode == "minhinge":
    cv = solve_minhinge(args.t, args.knee)
    if cv is not None:
        verify(cv, f"min hinge at {args.knee}, |c| ≤ {args.t}")
        args.out and np.savez(args.out, c=cv, t=args.t, nx=nx, ny=ny, nz=nz)
    sys.exit(0)
if args.mode == "minl1":
    cv = solve_minl1(args.t)
    if cv is not None:
        verify(cv, f"min L1, |c| ≤ {args.t}")
        args.out and np.savez(args.out, c=cv, t=args.t, nx=nx, ny=ny, nz=nz)
    sys.exit(0)

if args.write_mps:
    if not HAVE_HIGHSPY:
        sys.exit("--write-mps needs highspy")
    solve(args.t, minimize_t=(args.mode == "mint"), mps=args.write_mps); sys.exit(0)

if args.mode == "feas":
    st, cv, _ = solve(args.t)
    log(f"feasibility at t = {args.t}: {st}")
    if cv is not None:
        verify(cv, f"t={args.t}")
        args.out and np.savez(args.out, c=cv, t=args.t, nx=nx, ny=ny, nz=nz)
elif args.mode == "mint":
    st, cv, topt = solve(None, minimize_t=True)
    log(f"min t: {st}, t* = {topt}")
    if cv is not None:
        verify(cv, "min t")
        args.out and np.savez(args.out, c=cv, t=topt, nx=nx, ny=ny, nz=nz)
else:
    lo, hi = args.lo, args.hi
    best = None
    while hi - lo > args.tol:
        mid = 0.5 * (lo + hi)
        st, cv, _ = solve(mid)
        log(f"bisect t = {mid:.4f}: {st}")
        if st == "optimal":
            hi = mid; best = (mid, cv); verify(cv, f"t={mid:.4f}")
        elif st == "infeasible":
            lo = mid
        else:
            log("solver stopped without a verdict; bisection ended"); break
    log(f"bisection bracket: infeasible ≤ {lo:.4f} < t* ≤ {hi:.4f} (feasible)")
    if best is not None and args.out:
        np.savez(args.out, c=best[1], t=best[0], nx=nx, ny=ny, nz=nz)
