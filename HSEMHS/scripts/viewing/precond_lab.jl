#=
Preconditioner lab for mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md §6.4, D3b): CG on the Gauss–Newton system
of a late-solve state with several preconditioners, to see which one removes the stall.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/precond_lab.jl INPUT.nc MU AZIMUTH_DEG GEOMETRY LOGTAU_CUT_LOS BALANCE STEPS
STEPS Gauss–Newton steps (D2 settings) from the column HSE give the state; the system (JᵀWJ + ereg) δ = −JᵀWR is then
solved to a relative residual of 1e-3 (at most 600 iterations) with
  fourier : the layer-median Fourier preconditioner of the solver;
  block   : exact per-column blocks of the operator (probed with 9 colourings × n₁ levels), Cholesky (block Jacobi);
  add     : fourier + block (additive);
  twolevel: symmetric multiplicative, block → fourier → block (two operator products per application).
Prints the relative residual at iterations 25, 50, 100, 200, 400, the iterations to 1e-1, 3e-2, 1e-2, 1e-3, and the
time of one operator product and one application of each preconditioner.
=#
using HSEMHS
using MPI
using LinearAlgebra
using Printf

const H = HSEMHS
const PT = H.PoissonTau

function cg_hist(Aop!, Minv!, par, pr, b; maxit = 600, tol = 1e-3)
    x = zero(b); r = copy(b); zv = similar(b); p = similar(b); Ap = similar(b)
    Minv!(pr, zv, r); p .= zv; rz = H.gdot(par, r, zv); nb = H.gnorm(par, b); hist = Float64[]
    for it in 1:maxit
        Aop!(pr, Ap, p)
        α = rz / H.gdot(par, p, Ap)
        x .+= α .* p; r .-= α .* Ap
        push!(hist, H.gnorm(par, r) / nb)
        hist[end] < tol && break
        Minv!(pr, zv, r); rz2 = H.gdot(par, r, zv); β = rz2 / rz; rz = rz2
        p .= zv .+ β .* p
    end
    return hist
end

"""Exact per-column blocks of the operator by probing (columns 3 apart in x and y do not couple), Cholesky factors."""
function column_blocks(pr)
    n1, ny, nx = pr.n1, pr.ny, pr.nx; i0 = pr.par.block.i0
    F = zeros(n1, n1, ny, nx)
    v = zeros(n1, ny, nx); out = similar(v)
    for cx in 0:2, cy in 0:2, m in 1:n1
        fill!(v, 0.0)
        for i in 1:nx, j in 1:ny
            (i0 + i - 2) % 3 == cx && (j - 1) % 3 == cy && (v[m, j, i] = 1.0)
        end
        PT.Aop!(pr, out, v)
        for i in 1:nx, j in 1:ny
            (i0 + i - 2) % 3 == cx && (j - 1) % 3 == cy && (@views F[:, m, j, i] .= out[:, j, i])
        end
    end
    asym = zeros(nx); nfail = zeros(Int, nx)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            B = @view F[:, :, j, i]
            asym[i] = max(asym[i], maximum(abs, B .- B') / maximum(abs, B))
            S = Symmetric((B .+ B') ./ 2)
            C = cholesky(S; check = false)
            issuccess(C) || (nfail[i] += 1; C = cholesky(S + 1e-8 * tr(S) / n1 * I))
            B .= C.L
        end
    end
    return F, H.gmax(pr.par, maximum(asym)), H.gsum(pr.par, sum(nfail))
end

function block_solve!(out, F, r)
    n1 = size(F, 1)
    Threads.@threads for i in axes(r, 3)
        for j in axes(r, 2)
            L = LowerTriangular(@view F[:, :, j, i])
            x = r[:, j, i]
            ldiv!(L, x); ldiv!(L', x)
            @views out[:, j, i] .= x
        end
    end
    return out
end

function main(path, mu_s, az_s, geom, cut_s, bal, steps_s)
    H.ensure_mpi()
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = Symbol(geom))
    cut = parse(Float64, cut_s); balance = Symbol(bal); nsteps = parse(Int, steps_s)
    t0 = time()
    NX, NY, nt = PT.tau_input_size(path)
    par = H.slab_par(NX, NY, nt)
    logf(s) = H.rprint(par, @sprintf("[%7.1fs] %s", time() - t0, s))
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    lT = log.(T); g = 274.0; dx = x[2] - x[1]; dy = y[2] - y[1]
    BLAS.set_num_threads(1)
    lnP, z0, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed")
    u, z, Φ0, Φ, s, _, _ = PT.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, dx, dy; logtau_cut = cut,
                                        max_steps = nsteps, mu = v.mu, slant = H.slant(v), stop = :phi, rtol = 0.0,
                                        balance = balance, lm = :gain, forcing = :ew, stationarity = false)
    logf(@sprintf("state after %d steps: Φ/Φ₀ %.3e; μ %.3f φ %.0f %s, cut %.2f, balance %s", s, Φ / Φ0, v.mu,
                  v.azimuth_deg, v.geometry, cut, balance))
    pr, c = PT.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z, dx, dy; logtau_cut = cut, mu = v.mu, slant = H.slant(v),
                        balance = balance)
    PT.evaluate!(pr, u, c)
    PT.tp_weights!(pr, g, dx)
    PT.evaluate!(pr, u, c)
    PT.setprec!(pr)
    b = zeros(pr.n1, pr.ny, pr.nx)
    PT.JTop!(pr, b, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); b .*= -1
    tmp = similar(b)
    ta = @elapsed for _ in 1:5; PT.Aop!(pr, tmp, b); end
    tf = @elapsed for _ in 1:5; PT.Minv!(pr, tmp, b); end
    tb = @elapsed (F, asym, nfail) = column_blocks(pr)
    logf(@sprintf("operator product %.3f s, Fourier preconditioner %.3f s; column blocks probed in %.0f s (asymmetry %.1e, %.0f not SPD)",
                  ta / 5, tf / 5, tb, asym, nfail))
    tbs = @elapsed for _ in 1:5; block_solve!(tmp, F, b); end
    logf(@sprintf("block solve %.3f s", tbs / 5))

    fourier!(pr, out, r) = PT.Minv!(pr, out, r)
    block!(pr, out, r) = block_solve!(out, F, r)
    w1 = similar(b); w2 = similar(b)
    add!(pr, out, r) = (PT.Minv!(pr, out, r); block_solve!(w1, F, r); out .+= w1)
    function twolevel!(pr, out, r)
        block_solve!(out, F, r)                                     # x1
        PT.Aop!(pr, w1, out); w1 .= r .- w1                         # r − A x1
        PT.Minv!(pr, w2, w1); out .+= w2                            # x2
        PT.Aop!(pr, w1, out); w1 .= r .- w1                         # r − A x2
        block_solve!(w2, F, w1); out .+= w2
        return out
    end
    for (name, M) in (("fourier", fourier!), ("block", block!), ("add", add!), ("twolevel", twolevel!))
        tt = @elapsed hist = cg_hist(PT.Aop!, M, par, pr, b)
        at(n) = n <= length(hist) ? @sprintf("%.1e", hist[n]) : "  –   "
        to(t) = (k = findfirst(<(t), hist); k === nothing ? ">$(length(hist))" : string(k))
        logf(@sprintf("%-9s rel @25 %s @50 %s @100 %s @200 %s @400 %s | its to 1e-1 %s, 3e-2 %s, 1e-2 %s, 1e-3 %s | %.0f s",
                      name, at(25), at(50), at(100), at(200), at(400), to(1e-1), to(3e-2), to(1e-2), to(1e-3), tt))
    end
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
