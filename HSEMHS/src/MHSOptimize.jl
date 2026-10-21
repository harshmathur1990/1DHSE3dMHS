# L-BFGS with a strong-Wolfe line search on distributed 4-field states
# (implementation doc §13.7–13.8; E5 stop-reason mapping, E12).

"""Local Σ a·c over owned i, j and the z axis of `a` (fixed row order, no communication)."""
function sdot_local(par::ParCtx, a::Field, c::Field)
    b = par.block
    kv = collect(axes(a, 3))
    p = zeros(b.ny, length(kv))
    Threads.@threads :static for jk in 1:(b.ny*length(kv))
        j = (jk - 1) % b.ny + 1; q = (jk - 1) ÷ b.ny + 1; k = kv[q]
        s = 0.0
        @inbounds for i in 1:b.nx
            s += a[i, j, k] * c[i, j, k]
        end
        p[j, q] = s
    end
    t = 0.0
    for q in axes(p, 2), j in axes(p, 1)
        t += p[j, q]
    end
    return t
end

"""Deterministic global dot product over all fields of two states (one Allreduce)."""
function dotN(par::ParCtx, a::MHSState, c::MHSState)
    t = 0.0
    for f in eachindex(a)
        t += sdot_local(par, a[f], c[f])
    end
    return MPI.Allreduce(t, +, par.comm)
end

"""Zero state with the same layout (z ranges) as x."""
similar_state(x::MHSState) = Field[(y = similar(f); fill!(parent(y), 0.0); y) for f in x]

"""Apply y[i,j,k] = fun(i,j,k) on owned i, j and the whole z axis of y."""
function owned_map!(fun::F, y::Field, par::ParCtx) where {F}
    b = par.block
    kv = collect(axes(y, 3)); nk = length(kv)
    # threaded over (j, k) rows, not k alone: with NZ ≈ 177 levels on 128 threads a k-only split leaves ~30 % idle (M5h)
    Threads.@threads :static for jk in 1:(b.ny*nk)
        j = (jk - 1) % b.ny + 1; k = kv[(jk-1)÷b.ny+1]
        @inbounds for i in 1:b.nx
            y[i, j, k] = fun(i, j, k)
        end
    end
    return y
end

"""Global max |f| over owned i, j and the whole z axis."""
maxabs_all(par::ParCtx, f::Field) =
    global_max(par, maximum(abs(f[i, j, k]) for k in axes(f, 3), j in 1:par.block.ny, i in 1:par.block.nx))

"""y ← y + α x (owned nodes)."""
function axpy!(y::MHSState, α::Float64, x::MHSState, par::ParCtx)
    for f in eachindex(y)
        yf = y[f]; xf = x[f]
        owned_map!((i, j, k) -> yf[i, j, k] + α * xf[i, j, k], yf, par)
    end
    return y
end

"""y ← a x + b z (owned nodes)."""
function lincomb!(y::MHSState, a::Float64, x::MHSState, bb::Float64, z::MHSState, par::ParCtx)
    for f in eachindex(y)
        yf = y[f]; xf = x[f]; zf = z[f]
        owned_map!((i, j, k) -> a * xf[i, j, k] + bb * zf[i, j, k], yf, par)
    end
    return y
end

copystate!(y::MHSState, x::MHSState, par) = lincomb!(y, 1.0, x, 0.0, x, par)
scale!(y::MHSState, a::Float64, par) = lincomb!(y, a, y, 0.0, y, par)

"""Zero the gradient of frozen fields (used for B-first continuation)."""
function freeze!(g::MHSState, frozen, par)
    for f in frozen
        owned_map!((i, j, k) -> 0.0, g[f], par)
    end
    return g
end

"""y ← x ./ D (owned nodes)."""
function divdiag!(y::MHSState, x::MHSState, D::MHSState, par::ParCtx)
    for f in eachindex(y)
        yf = y[f]; xf = x[f]; Df = D[f]
        owned_map!((i, j, k) -> xf[i, j, k] / Df[i, j, k], yf, par)
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
    xtol_rel::Float64 = 0.0          # converged when the measured quantities change ≤ xtol_rel over xtol_window iterations (0: off; M5)
    xtol_window::Int = 100
    collapse_du::Float64 = 3.0       # pressure-collapse safeguard: u = ln P stays ≥ its start value − collapse_du (M6d; Inf: off)
    refresh::Int = 0                 # recompute the diagonal every `refresh` iterations (0: never)
    frozen::Vector{Int} = Int[]      # state fields held fixed (1 = v, 2:4 = b̃): their gradient is zeroed
end

"""
    anchor_box_project!(δ, t, lo, hi, β, pb) -> anchor residual

δ = the point with lo ≤ δ ≤ hi (per node) and Σ_anchor α δ = 0: off the anchor plane δ = clamp(t, lo, hi);
on it δ = clamp(t − c β, lo, hi) with the scalar c found by bisection (Σ α clamp(t − cβ) is
monotone in c for β > 0). Requires lo ≤ 0 ≤ hi. With β = 1 this is the Euclidean projection for
equal anchor weights; with β = α/D it is the minimizer of gᵀδ + ½δᵀDδ for t = −g/D. `lo`, `hi`, `β`
are callables (i, j, k) -> value. Returns |Σ α δ| (Codex review M5, finding 1).
"""
function anchor_box_project!(δ::Field, t::Field, lo, hi, β, pb)
    par = pb.par; b = par.block; ka = pb.ka; α = pb.alpha
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        k == ka || (δ[i, j, k] = clamp(t[i, j, k], lo(i, j, k), hi(i, j, k)))
    end
    F(c) = MPI.Allreduce(sum(α[i, j] * clamp(t[i, j, ka] - c * β(i, j, ka), lo(i, j, ka), hi(i, j, ka)) for j in 1:b.ny, i in 1:b.nx), +, par.comm)
    # no shift needed when the clamped point already satisfies the anchor constraint (keeps inactive
    # projections exact instead of a round-off-sized bisection shift)
    sc = MPI.Allreduce(sum(α[i, j] * abs(t[i, j, ka]) for j in 1:b.ny, i in 1:b.nx), +, par.comm)
    if abs(F(0.0)) <= 1e-14 * max(sc, 1e-300)
        for j in 1:b.ny, i in 1:b.nx
            δ[i, j, ka] = clamp(t[i, j, ka], lo(i, j, ka), hi(i, j, ka))
        end
        return abs(F(0.0))
    end
    a = -1.0; z = 1.0
    for _ in 1:200
        F(a) >= 0 && break
        a *= 2
    end
    for _ in 1:200
        F(z) <= 0 && break
        z *= 2
    end
    for _ in 1:200
        m = 0.5 * (a + z)
        (m == a || m == z) && break
        F(m) >= 0 ? (a = m) : (z = m)
    end
    c = abs(F(a)) <= abs(F(z)) ? a : z
    for j in 1:b.ny, i in 1:b.nx
        δ[i, j, ka] = clamp(t[i, j, ka] - c * β(i, j, ka), lo(i, j, ka), hi(i, j, ka))
    end
    return abs(F(c))
end

"""Constant offset between the pressure variable x[1] and ln P: u = x[1] − gauge_shift(pb, x[1]) + const."""
gauge_shift(pb, f::Field) = anchor_mean(pb, f)

"""
Zeroes d where x is within `tol` of the lower bound and d < 0; returns the global count. With
tol = 1e-3·max_du every remaining node's breakpoint is ≥ 1e-3 of the |Δu| step cap, so nodes just
short of the bound cannot jam the step at α ≈ 0.
"""
function project_lower!(d::Field, x::Field, lo::Field, par::ParCtx; tol::Float64)
    b = par.block; n = 0
    for k in axes(d, 3), j in 1:b.ny, i in 1:b.nx
        if x[i, j, k] - lo[i, j, k] <= tol && d[i, j, k] < 0
            d[i, j, k] = 0.0; n += 1
        end
    end
    return MPI.Allreduce(n, +, par.comm)
end

"""
Largest α ≥ 0 with x + αd ≥ lo on every owned node farther than `tol` from the bound (nodes within
`tol` were projected; the gauge re-centring can leave them a round-off-sized d < 0).
"""
function step_to_lower(d::Field, x::Field, lo::Field, par::ParCtx; tol::Float64)
    b = par.block; a = Inf
    for k in axes(d, 3), j in 1:b.ny, i in 1:b.nx
        x[i, j, k] - lo[i, j, k] > tol && d[i, j, k] < 0 && (a = min(a, (x[i, j, k] - lo[i, j, k]) / -d[i, j, k]))
    end
    return -global_max(par, -a)
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
function lbfgs!(pb, w, x::MHSState; opts::LBFGSOptions = LBFGSOptions())
    par = pb.par
    m = opts.m
    S = [similar_state(x) for _ in 1:m]; Y = [similar_state(x) for _ in 1:m]
    rhoh = zeros(m); al = zeros(m)
    nh = 0; head = 0                                 # number stored, index of newest
    g = similar_state(x); d = similar_state(x); q = similar_state(x)
    xt = similar_state(x); gt = similar_state(x)
    # pressure-collapse safeguard on u = x[1] − gauge_shift(x[1]) (the bound must be on u, not on
    # the gauge variable: a rising anchor-plane mean of v lowers u everywhere else)
    ufix = isfinite(opts.collapse_du) && !(1 in opts.frozen)
    ucur = similar_state(x)[1]; ulo = similar_state(x)[1]; du = similar_state(x)[1]
    if ufix
        gs = gauge_shift(pb, x[1])
        owned_map!((i, j, k) -> x[1][i, j, k] - gs - opts.collapse_du, ulo, par)
    end
    f, terms, oob = fg!(g, pb, w, x)
    freeze!(g, opts.frozen, par)
    oob && return LBFGSResult(eos_failure, "initial state outside the EOS envelope", 0, 1, f, terms, NaN, NaN, NTuple{4,Float64}[])
    nev = 1
    g0 = sqrt(dotN(par, g, g)); gn = g0
    Dg = opts.precondition ? gn_diagonal!(similar_state(x), pb, w, x) : nothing     # w holds the state at x
    tmp = similar_state(x)
    hist = NTuple{4,Float64}[(f, gn, 0.0, 1.0)]
    fwin = Float64[f]
    nfail = 0
    it = 0
    snap = opts.xtol_rel > 0 ? measured_snapshot(pb, w) : nothing
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
        # pressure-collapse safeguard: project out the downward u-direction on nodes at the
        # bound, then cap α at the bound (u → vacuum is a stationary trap, ∂R/∂u ∝ P)
        if ufix
            # drop the gauge component of d[1] (changes neither u nor Φ), then fix bound nodes
            c = gauge_shift(pb, d[1]); d1 = d[1]
            owned_map!((i, j, k) -> d1[i, j, k] - c, d1, par)
            gs = gauge_shift(pb, x[1])
            owned_map!((i, j, k) -> x[1][i, j, k] - gs, ucur, par)
            # nodes at the bound may not move down; the anchor constraint is kept exactly, so the
            # projected d[1] is the actual Δu per unit α
            tolb = 1e-3 * opts.max_du
            lob(i, j, k) = ucur[i, j, k] - ulo[i, j, k] <= tolb ? 0.0 : -Inf
            proj!() = (copystate!(tmp, d, par); anchor_box_project!(d1, tmp[1], lob, (i, j, k) -> Inf, (i, j, k) -> 1.0, pb))
            proj!()
            dg = dotN(par, d, g)
            if !(dg < -1e-12 * gn * sqrt(dotN(par, d, d)))
                # fall back to projected (preconditioned) steepest descent and drop the memory
                nh = 0
                Dg === nothing ? lincomb!(d, -1.0 / gn, g, 0.0, g, par) : (divdiag!(d, g, Dg, par); scale!(d, -1.0, par))
                c = gauge_shift(pb, d1)
                owned_map!((i, j, k) -> d1[i, j, k] - c, d1, par)
                proj!()
                dg = dotN(par, d, g)
                if !(dg < -1e-12 * gn * sqrt(dotN(par, d, d)))
                    status = stalled; reason = "pressure-collapse safeguard: no descent above the u bound"; break
                end
            end
            copyto!(parent(du), parent(d1))                          # Δu per unit α (anchor mean 0)
        end
        α0 = (nh == 0 && Dg === nothing) ? min(1.0, 1.0 / gn) : 1.0
        # step cap: u = ln P is a log variable (a huge step can collapse P, where ∂Φ/∂u ∝ P
        # vanishes), and b̃ is in σ units
        dmu = 1 in opts.frozen ? 0.0 : maxabs_all(par, d[1])
        dmb = maximum((c in opts.frozen ? 0.0 : maxabs_all(par, d[c])) for c in 2:length(d))
        αcap = min(opts.alpha_max, dmu > 0 ? opts.max_du / dmu : Inf, dmb > 0 ? opts.max_db / dmb : Inf)
        ufix && (αcap = min(αcap, step_to_lower(du, ucur, ulo, par; tol = 1e-3 * opts.max_du)))
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
            gn_diagonal!(Dg, pb, w, xt); nh = 0                                   # w holds the state at xt
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
        if opts.xtol_rel > 0 && it % opts.xtol_window == 0
            objective!(pb, w, x)                         # w at the accepted iterate
            if snap !== nothing && measured_change(pb, w, snap) <= opts.xtol_rel
                status = converged; reason = "measured quantities changed ≤ xtol_rel over xtol_window iterations"; break
            end
            snap = measured_snapshot(pb, w)
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
function lbfgs_continuation!(pb, w, x::MHSState; b_first_iter::Int = 500,
                             opts::LBFGSOptions = LBFGSOptions())
    o1 = LBFGSOptions(; (k => getfield(opts, k) for k in fieldnames(LBFGSOptions))..., frozen = [1], max_iter = b_first_iter)
    r1 = lbfgs!(pb, w, x; opts = o1)
    r2 = lbfgs!(pb, w, x; opts = opts)
    return r1, r2
end
