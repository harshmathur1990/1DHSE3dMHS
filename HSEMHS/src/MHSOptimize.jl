# L-BFGS with a strong-Wolfe line search on distributed 4-field states
# (implementation doc §13.7–13.8; E5 stop-reason mapping, E12).

"""Deterministic global dot product over all fields of two states (one Allreduce)."""
function dotN(par::ParCtx, a::MHSState, c::MHSState)
    b = par.block
    p = rowpartials(par, 1)
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j, k = rowjk(jk, b.ny)
        s = 0.0
        @inbounds for f in eachindex(a), i in 1:b.nx
            s += a[f][i, j, k] * c[f][i, j, k]
        end
        p[1, j, k] = s
    end
    return reduce_rows(par, p)[1]
end

"""y ← y + α x (owned nodes)."""
function axpy!(y::MHSState, α::Float64, x::MHSState, par::ParCtx)
    b = par.block
    for f in eachindex(y)
        yf = y[f]; xf = x[f]
        Threads.@threads :static for k in 1:b.nz
            @inbounds for j in 1:b.ny, i in 1:b.nx
                yf[i, j, k] += α * xf[i, j, k]
            end
        end
    end
    return y
end

"""y ← a x + b z (owned nodes)."""
function lincomb!(y::MHSState, a::Float64, x::MHSState, bb::Float64, z::MHSState, par::ParCtx)
    b = par.block
    for f in eachindex(y)
        yf = y[f]; xf = x[f]; zf = z[f]
        Threads.@threads :static for k in 1:b.nz
            @inbounds for j in 1:b.ny, i in 1:b.nx
                yf[i, j, k] = a * xf[i, j, k] + bb * zf[i, j, k]
            end
        end
    end
    return y
end

copystate!(y::MHSState, x::MHSState, par) = lincomb!(y, 1.0, x, 0.0, x, par)
scale!(y::MHSState, a::Float64, par) = lincomb!(y, a, y, 0.0, y, par)

"""Zero the gradient of frozen fields (used for B-first continuation)."""
function freeze!(g::MHSState, frozen, par)
    for f in frozen
        fill_owned!((i, j, k) -> 0.0, g[f], par)
    end
    return g
end

"""y ← x ./ D (owned nodes)."""
function divdiag!(y::MHSState, x::MHSState, D::MHSState, par::ParCtx)
    b = par.block
    for f in eachindex(y)
        yf = y[f]; xf = x[f]; Df = D[f]
        Threads.@threads :static for k in 1:b.nz
            @inbounds for j in 1:b.ny, i in 1:b.nx
                yf[i, j, k] = xf[i, j, k] / Df[i, j, k]
            end
        end
    end
    return y
end

Base.@kwdef struct LBFGSOptions
    m::Int = 10
    max_iter::Int = 5000
    gtol_rel::Float64 = 1e-6
    ftol_rel::Float64 = 1e-10
    ftol_window::Int = 20
    c1::Float64 = 1e-4
    c2::Float64 = 0.9
    ls_max_evals::Int = 20
    alpha_max::Float64 = 1e6
    verbose::Bool = false
    precondition::Bool = true        # Jacobi initial inverse Hessian from the Gauss–Newton diagonal
    max_du::Float64 = 0.5            # largest |Δ ln P| per iteration (step cap)
    max_db::Float64 = 5.0            # largest |Δ b̃| per iteration, in σ units
    refresh::Int = 0                 # recompute the diagonal every `refresh` iterations (0: never)
    frozen::Vector{Int} = Int[]      # state fields held fixed (1 = v, 2:4 = b̃): their gradient is zeroed
end

struct LBFGSResult
    status::SolverStatus
    reason::String
    iterations::Int
    evaluations::Int
    f::Float64
    terms::NTuple{5,Float64}
    gnorm0::Float64
    gnorm::Float64
    history::Vector{NTuple{4,Float64}}       # (f, ‖g‖, α, line-search evaluations)
end

"""
    lbfgs!(pb, w, x; opts) -> LBFGSResult

Minimizes Φ over x in place (owned nodes). Stops on ‖∇Φ‖ ≤ gtol_rel‖∇Φ₀‖ (converged),
relative Φ change ≤ ftol_rel over ftol_window iterations or repeated line-search failure
(stalled), or max_iter.
"""
function lbfgs!(pb::MHSProblem, w::MHSWork, x::MHSState; opts::LBFGSOptions = LBFGSOptions())
    par = pb.par
    m = opts.m
    S = [newstate(par) for _ in 1:m]; Y = [newstate(par) for _ in 1:m]
    rhoh = zeros(m); al = zeros(m)
    nh = 0; head = 0                                 # number stored, index of newest
    g = newstate(par); d = newstate(par); q = newstate(par)
    xt = newstate(par); gt = newstate(par)
    f, terms, oob = fg!(g, pb, w, x)
    freeze!(g, opts.frozen, par)
    oob && return LBFGSResult(eos_failure, "initial state outside the EOS envelope", 0, 1, f, terms, NaN, NaN, NTuple{4,Float64}[])
    nev = 1
    g0 = sqrt(dotN(par, g, g)); gn = g0
    Dg = opts.precondition ? gn_diagonal!(newstate(par), pb, w) : nothing     # w holds the state at x
    tmp = newstate(par)
    hist = NTuple{4,Float64}[(f, gn, 0.0, 1.0)]
    fwin = Float64[f]
    nfail = 0
    it = 0
    status = max_iter; reason = "max_iter"
    while true
        if gn <= opts.gtol_rel * g0 || gn == 0
            status = converged; reason = "gradient norm ≤ gtol_rel·‖∇Φ₀‖"; break
        end
        it >= opts.max_iter && (status = max_iter; reason = "max_iter"; break)
        # two-loop recursion
        copystate!(q, g, par)
        for t in 0:nh-1
            ii = mod1(head - t, m)
            al[ii] = rhoh[ii] * dotN(par, S[ii], q)
            axpy!(q, -al[ii], Y[ii], par)
        end
        if Dg === nothing
            γ = nh == 0 ? 1.0 / gn : dotN(par, S[head], Y[head]) / dotN(par, Y[head], Y[head])
            scale!(q, γ, par)
        else
            # H0 = γ D⁻¹, γ = sᵀy / (yᵀ D⁻¹ y) (1 before any curvature pair)
            γ = 1.0
            if nh > 0
                divdiag!(tmp, Y[head], Dg, par)
                γ = dotN(par, S[head], Y[head]) / dotN(par, Y[head], tmp)
            end
            divdiag!(q, q, Dg, par); scale!(q, γ, par)
        end
        for t in nh-1:-1:0
            ii = mod1(head - t, m)
            β = rhoh[ii] * dotN(par, Y[ii], q)
            axpy!(q, al[ii] - β, S[ii], par)
        end
        lincomb!(d, -1.0, q, 0.0, q, par)
        dg = dotN(par, d, g)
        if !(dg < -1e-12 * gn * sqrt(dotN(par, d, d)))
            nh = 0
            if Dg === nothing
                lincomb!(d, -1.0 / gn, g, 0.0, g, par)
            else
                divdiag!(d, g, Dg, par); scale!(d, -1.0, par)
            end
            dg = dotN(par, d, g)
        end
        α0 = (nh == 0 && Dg === nothing) ? min(1.0, 1.0 / gn) : 1.0
        # step cap: u = ln P is a log variable (a huge step can collapse P, where ∂Φ/∂u ∝ P
        # vanishes), and b̃ is in σ units
        dmu = 1 in opts.frozen ? 0.0 : maxabs(par, d[1])
        dmb = maximum((c in opts.frozen ? 0.0 : maxabs(par, d[c])) for c in 2:4)
        αcap = min(opts.alpha_max, dmu > 0 ? opts.max_du / dmu : Inf, dmb > 0 ? opts.max_db / dmb : Inf)
        α0 = min(α0, αcap)
        α, ft, termt, ok, ne = strong_wolfe!(pb, w, x, f, dg, d, xt, gt, α0, opts; alpha_max = αcap)
        nev += ne
        if !ok
            nfail += 1
            if nh == 0 && nfail >= 2
                status = stalled; reason = "line search failed twice from steepest descent"; break
            end
            nh = 0
            continue
        end
        nfail = 0
        if opts.refresh > 0 && Dg !== nothing && it > 0 && it % opts.refresh == 0
            gn_diagonal!(Dg, pb, w); nh = 0                                       # w holds the state at xt
        end
        # s = xt − x, y = gt − g ; store into the ring
        head = mod1(head + 1, m)
        lincomb!(S[head], 1.0, xt, -1.0, x, par)
        lincomb!(Y[head], 1.0, gt, -1.0, g, par)
        sy = dotN(par, S[head], Y[head])
        if sy > 1e-10 * sqrt(dotN(par, S[head], S[head]) * dotN(par, Y[head], Y[head]))
            rhoh[head] = 1 / sy
            nh = min(nh + 1, m)
        else
            head = mod1(head - 1, m)                    # discard the pair
        end
        copystate!(x, xt, par); copystate!(g, gt, par)
        f, terms = ft, termt
        gn = sqrt(dotN(par, g, g))
        it += 1
        push!(hist, (f, gn, α, Float64(ne)))
        push!(fwin, f)
        length(fwin) > opts.ftol_window + 1 && popfirst!(fwin)
        if opts.verbose && rank0(par) && (it % 50 == 0)
            @printf("  lbfgs it %5d  Φ = %.6e  ‖g‖/‖g0‖ = %.3e  α = %.3e\n", it, f, gn / g0, α)
        end
        if length(fwin) == opts.ftol_window + 1 && abs(fwin[1] - f) <= opts.ftol_rel * abs(f)
            status = stalled; reason = "relative Φ change ≤ ftol_rel over ftol_window iterations"; break
        end
    end
    # leave w consistent with the final x
    objective!(pb, w, x)
    return LBFGSResult(status, reason, it, nev, f, terms, g0, gn, hist)
end

"""
Strong-Wolfe line search (Nocedal & Wright Alg. 3.5/3.6). Trial points go to xt, gt.
Returns (α, f(α), terms, ok, evaluations); on success xt, gt hold the accepted point.
"""
function strong_wolfe!(pb, w, x, f0, dg0, d, xt, gt, α0, opts; alpha_max = opts.alpha_max)
    par = pb.par
    ne = 0
    function phi(α)
        lincomb!(xt, 1.0, x, α, d, par)
        f, terms, oob = fg!(gt, pb, w, xt)
        freeze!(gt, opts.frozen, par)
        ne += 1
        oob && return Inf, NaN, terms
        return f, dotN(par, gt, d), terms
    end
    αp = 0.0; fp = f0; dp = dg0
    α = α0
    best = (0.0, Inf, ntuple(_ -> NaN, 5))
    function zoom(lo, flo, dlo, hi, fhi, dhi)
        while ne < opts.ls_max_evals
            αj = if isfinite(fhi)
                d1 = dlo + dhi - 3 * (flo - fhi) / (lo - hi)
                disc = d1^2 - dlo * dhi
                if disc >= 0
                    d2 = sign(hi - lo) * sqrt(disc)
                    hi - (hi - lo) * (dhi + d2 - d1) / (dhi - dlo + 2d2)
                else
                    0.5 * (lo + hi)
                end
            else
                lo + 0.1 * (hi - lo)
            end
            a, bnd = minmax(lo, hi)
            δ = 0.1 * (bnd - a)
            (isfinite(αj) && a + δ <= αj <= bnd - δ) || (αj = 0.5 * (lo + hi))
            fj, dj, tj = phi(αj)
            if !isfinite(fj) || fj > f0 + opts.c1 * αj * dg0 || fj >= flo
                hi, fhi, dhi = αj, fj, dj
            else
                abs(dj) <= -opts.c2 * dg0 && return αj, fj, tj, true
                dj * (hi - lo) >= 0 && (hi, fhi, dhi = lo, flo, dlo)
                lo, flo, dlo = αj, fj, dj
            end
            abs(hi - lo) < 1e-14 * max(1.0, abs(lo)) && break
        end
        return 0.0, f0, ntuple(_ -> NaN, 5), false
    end
    i = 1
    while ne < opts.ls_max_evals
        fa, da, ta = phi(α)
        if !isfinite(fa) || fa > f0 + opts.c1 * α * dg0 || (i > 1 && fa >= fp)
            r = zoom(αp, fp, dp, α, fa, da)
            r[4] && return r..., ne
            return 0.0, f0, ntuple(_ -> NaN, 5), false, ne
        end
        abs(da) <= -opts.c2 * dg0 && return α, fa, ta, true, ne
        if da >= 0
            r = zoom(α, fa, da, αp, fp, dp)
            r[4] && return r..., ne
            return 0.0, f0, ntuple(_ -> NaN, 5), false, ne
        end
        αp, fp, dp = α, fa, da
        α >= alpha_max && return α, fa, ta, true, ne     # at the cap with sufficient decrease
        α = min(2α, alpha_max)
        i += 1
    end
    return 0.0, f0, ntuple(_ -> NaN, 5), false, ne
end

"""
    lbfgs_continuation!(pb, w, x; b_first_iter, opts) -> (phase1, phase2)

B-first continuation: relax b̃ with v frozen (up to `b_first_iter` iterations), then solve
jointly from there.
"""
function lbfgs_continuation!(pb::MHSProblem, w::MHSWork, x::MHSState; b_first_iter::Int = 500,
                             opts::LBFGSOptions = LBFGSOptions())
    o1 = LBFGSOptions(; (k => getfield(opts, k) for k in fieldnames(LBFGSOptions))..., frozen = [1], max_iter = b_first_iter)
    r1 = lbfgs!(pb, w, x; opts = o1)
    r2 = lbfgs!(pb, w, x; opts = opts)
    return r1, r2
end
