# Independent, small reproductions for CODEX_REVIEW_M5.md. No real-data solve.
# Run from HSEMHS: julia --project=. -t 2 scripts/m5_bifrost/codex_review_checks.jl
using HSEMHS, MPI, LinearAlgebra, Printf, Test
const H = HSEMHS
H.ensure_mpi()

function review_checks()
    par = H.init_parallel(4, 4, 9)
    @assert MPI.Comm_size(par.comm) == 1 "These diagnostic examples use one MPI rank."
    pb, x, us, uc, grid = H.sc1_problem(par, 4, 4, 9; lambda_Q = 1e-3)
    w = H.SMHSWork(par)
    H.objective!(pb, w, x)
    b = par.block
    d = H.newfield(par)
    lo = H.newfield(par)
    H.owned_map!((i, j, k) -> k == pb.ka ? (isodd(i) ? -2.0 : 2.0) : 0.0, d, par)
    H.owned_map!((i, j, k) -> w.u[i, j, k] - ((k == pb.ka && isodd(i)) ? 0.0 : 3.0), lo, par)
    Delta = 0.5
    for _ in 1:2
        m = H.anchor_mean(pb, d)
        H.owned_map!((i, j, k) -> clamp(d[i, j, k] - m, max(-Delta, lo[i, j, k] - w.u[i, j, k]), Delta) + m, d, par)
    end
    m = H.anchor_mean(pb, d)
    margin = minimum(w.u[i,j,k] + d[i,j,k] - m - lo[i,j,k] for i in 1:b.nx, j in 1:b.ny, k in 1:b.nz)
    @test margin < -0.12
    @printf("Alternation clipping: minimum actual bound margin = %.6f (negative violates bound)\n", margin)

    # Same issue in the L-BFGS projection: clipping changes the anchor mean.
    H.owned_map!((i, j, k) -> k == pb.ka ? (isodd(i) ? -2.0 : 2.0) : 0.0, d, par)
    H.project_lower!(d, w.u, lo, par; tol = 5e-4)
    du = H.newfield(par)
    m = H.anchor_mean(pb, d)
    H.owned_map!((i, j, k) -> d[i,j,k] - m, du, par)
    acap = min(Delta / H.maxabs_all(par, d), H.step_to_lower(du, w.u, lo, par; tol = 5e-4))
    margin = minimum(w.u[i,j,k] + acap * du[i,j,k] - lo[i,j,k] for i in 1:b.nx, j in 1:b.ny, k in 1:b.nz)
    @test margin < -0.24
    @printf("L-BFGS projection: alpha cap = %.6f, actual bound margin = %.6f\n", acap, margin)

    # gn_diag_u! is the diagonal before the anchor parameterization, whereas
    # gn_hvp! applies G^T H_u G. They agree off the anchor, but not generally on it.
    H.owned_map!((i,j,k) -> x[1][i,j,k] + 0.8 * sin(2pi * (i-1) / b.nx), x[1], par)
    H.objective!(pb, w, x)
    D = H.newfield(par)
    H.gn_diag_u!(D, pb, w)
    z = H.similar_state(x); Hz = H.similar_state(x); wl = H.SMHSWork(par)
    for k in (pb.ka - 1, pb.ka)
        fill!(parent(z[1]), 0.0)
        z[1][2,2,k] = 1.0
        H.gn_hvp!(Hz, pb, w, wl, z)
        rel = abs(D[2,2,k] - Hz[1][2,2,k]) / abs(Hz[1][2,2,k])
        k == pb.ka ? (@test rel > 1e-4) : (@test rel < 1e-12)
        @printf("u diagonal at k=%d: supplied %.9e, basis-product %.9e, relative difference %.6e\n", k, D[2,2,k], Hz[1][2,2,k], rel)
    end

    # No block work: measured_change is zero even at this non-equilibrium state.
    f0 = H.objective!(pb, w, x)[1]
    st, why, n, hist = H.alternating_solve!(pb, w, x; max_outer = 1, a_iter = 0, u_steps = 0)
    @test st == H.converged
    @test f0 > 1e-4
    @printf("Zero-work outer cycle: status=%s, change=%.1e, objective=%.6e\n", string(st), hist[end].change, f0)

    # Propagated errors of periodic centre-to-face interpolation, sigma = 1.
    Icf = zeros(8,8)
    for i in 1:8
        Icf[i,i] = 0.5
        Icf[i,mod1(i+1,8)] = 0.5
    end
    C = Icf * Icf'
    @test C[1,1] == 0.5 && C[1,2] == 0.25 && rank(C) == 7
    @printf("Interpolated independent unit noise: variance=%.2f, neighbour covariance=%.2f, rank=%d/8\n", C[1,1], C[1,2], rank(C))

    # An asymmetric midpoint stencil can preserve polynomial accuracy and
    # retain a Nyquist response; it still has an exact algebraic transpose.
    a = [3/8, 3/4, -1/8]; s = [-0.5, 0.5, 1.5]
    @test sum(a) == 1 && dot(a,s) == 0 && dot(a,s.^2) == 0
    nyquist = abs(sum(a .* exp.(im * pi .* s)))
    @test nyquist ≈ 0.5
    @printf("Asymmetric quadratic midpoint: Nyquist response=%.2f, exact on degrees 0,1,2\n", nyquist)
end

review_checks()
