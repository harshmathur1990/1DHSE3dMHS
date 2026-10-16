# Bicubic Hermite EOS table (implementation doc §7.6).
#
# Fields F1 = ln ρ [kg m-3], F2 = ln nₑ [m-3], F3 = ln χ₅₀₀ [m-1] on X = ln T, Y = ln P.
# Each node stores (f, f_X, f_Y, f_XY); derivatives from central differences (δ = 1e-3) of
# the EOS model (direct / blend / hot branch). Axes are nonuniform after refinement; an
# O(1) bucket lookup finds the cell. Evaluation is allocation-free and thread-safe.

abstract type AbstractEOS end

struct Bucket
    lo::Float64
    h::Float64
    first::Vector{Int32}
end

function Bucket(ax::Vector{Float64})
    # bin width = smallest regular spacing; the breakpoint node pairs (1e-9 apart) are
    # skipped by the forward scan in findcell, not resolved by the bins
    h = minimum(d for d in diff(ax) if d > 1e-6)
    nb = ceil(Int, (ax[end] - ax[1]) / h) + 2
    first = Vector{Int32}(undef, nb)
    n = length(ax)
    for b in 1:nb
        x = ax[1] + (b - 1) * h
        first[b] = clamp(searchsortedlast(ax, x), 1, n - 1)
    end
    return Bucket(ax[1], h, first)
end

@inline function findcell(bk::Bucket, ax::Vector{Float64}, x::Float64)
    b = clamp(unsafe_trunc(Int, (x - bk.lo) / bk.h) + 1, 1, length(bk.first))
    p = Int(bk.first[b])
    n = length(ax)
    @inbounds while p < n - 1 && ax[p+1] <= x
        p += 1
    end
    return p
end

mutable struct EOSTable <: AbstractEOS
    X::Vector{Float64}
    Y::Vector{Float64}
    V::Array{Float64,4}              # (4, 3, nX, nY): f, fX, fY, fXY for each field
    flags::Matrix{UInt8}             # node flags (EOS_ROBUST_FALLBACK, EOS_DIRECT_FAILED, …)
    xb::Bucket
    yb::Bucket
    T_sw::Float64
    hot_branch::String
    mass_per_H::Float64
    meta::Dict{String,Any}
end

function EOSTable(X, Y, V, flags, T_sw, hot_branch, mass_per_H, meta)
    return EOSTable(X, Y, V, flags, Bucket(X), Bucket(Y), T_sw, hot_branch, mass_per_H, meta)
end

mass_per_H(t::EOSTable) = t.mass_per_H

# ---- Hermite basis ----------------------------------------------------------------------
@inline hbasis(t) = (2t^3 - 3t^2 + 1, -2t^3 + 3t^2, t^3 - 2t^2 + t, t^3 - t^2)      # H0, H1, K0, K1
@inline hderiv(t) = (6t^2 - 6t, -6t^2 + 6t, 3t^2 - 4t + 1, 3t^2 - 2t)

@inline function in_table(tab::EOSTable, X, Y)
    @inbounds return (tab.X[1] <= X <= tab.X[end]) && (tab.Y[1] <= Y <= tab.Y[end])
end

@inline function xflags(tab::EOSTable, X, p, q)
    T = exp(X)
    f = T >= 3.0e4 ? OPACITY_INCOMPLETE : EOS_OK
    if T >= 1.1 * tab.T_sw
        f |= EOS_HOT_BRANCH
    elseif T >= tab.T_sw
        f |= EOS_BLEND
    end
    @inbounds f |= (tab.flags[p, q] | tab.flags[p+1, q] | tab.flags[p, q+1] | tab.flags[p+1, q+1]) &
                   (EOS_ROBUST_FALLBACK | EOS_DIRECT_FAILED)
    return f
end

"""Value (and ∂/∂Y) of field `k` in cell (p, q) at local coordinates (t, s)."""
@inline function hermite(tab::EOSTable, k::Int, p::Int, q::Int, t, s, hX, hY, wantdY::Bool)
    Ht0, Ht1, Kt0, Kt1 = hbasis(t)
    Hs0, Hs1, Ks0, Ks1 = hbasis(s)
    V = tab.V
    val = 0.0; dY = 0.0
    @inbounds for (a, Ha, Ka) in ((0, Ht0, Kt0), (1, Ht1, Kt1))
        for (b, Hb, Kb) in ((0, Hs0, Ks0), (1, Hs1, Ks1))
            i = p + a; j = q + b
            f = V[1, k, i, j]; fX = V[2, k, i, j]; fY = V[3, k, i, j]; fXY = V[4, k, i, j]
            val += f * Ha * Hb + hX * fX * Ka * Hb + hY * fY * Ha * Kb + hX * hY * fXY * Ka * Kb
        end
    end
    if wantdY
        dHs0, dHs1, dKs0, dKs1 = hderiv(s)
        @inbounds for (a, Ha, Ka) in ((0, Ht0, Kt0), (1, Ht1, Kt1))
            for (b, dHb, dKb) in ((0, dHs0, dKs0), (1, dHs1, dKs1))
                i = p + a; j = q + b
                f = V[1, k, i, j]; fX = V[2, k, i, j]; fY = V[3, k, i, j]; fXY = V[4, k, i, j]
                dY += f * Ha * dHb + hX * fX * Ka * dHb + hY * fY * Ha * dKb + hX * hY * fXY * Ka * dKb
            end
        end
        dY /= hY
    end
    return val, dY
end

@inline function locate(tab::EOSTable, X, Y)
    p = findcell(tab.xb, tab.X, X)
    q = findcell(tab.yb, tab.Y, Y)
    @inbounds hX = tab.X[p+1] - tab.X[p]
    @inbounds hY = tab.Y[q+1] - tab.Y[q]
    @inbounds t = (X - tab.X[p]) / hX
    @inbounds s = (Y - tab.Y[q]) / hY
    return p, q, t, s, hX, hY
end

"""ln ρ at (X = ln T, Y = ln P). Returns (value, status)."""
@inline function lnrho(tab::EOSTable, X::Float64, Y::Float64)
    in_table(tab, X, Y) || return NaN, EOS_OUT_OF_ENVELOPE
    p, q, t, s, hX, hY = locate(tab, X, Y)
    v, _ = hermite(tab, 1, p, q, t, s, hX, hY, false)
    return v, xflags(tab, X, p, q)
end

"""(ln ρ, ∂ln ρ/∂ln P at fixed T, status)."""
@inline function lnrho_dY(tab::EOSTable, X::Float64, Y::Float64)
    in_table(tab, X, Y) || return NaN, NaN, EOS_OUT_OF_ENVELOPE
    p, q, t, s, hX, hY = locate(tab, X, Y)
    v, d = hermite(tab, 1, p, q, t, s, hX, hY, true)
    return v, d, xflags(tab, X, p, q)
end

"""(ln ρ, ln nₑ, ln χ₅₀₀, status)."""
@inline function state(tab::EOSTable, X::Float64, Y::Float64)
    in_table(tab, X, Y) || return NaN, NaN, NaN, EOS_OUT_OF_ENVELOPE
    p, q, t, s, hX, hY = locate(tab, X, Y)
    r, _ = hermite(tab, 1, p, q, t, s, hX, hY, false)
    n, _ = hermite(tab, 2, p, q, t, s, hX, hY, false)
    c, _ = hermite(tab, 3, p, q, t, s, hX, hY, false)
    return r, n, c, xflags(tab, X, p, q)
end

# ---------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------

const TABLE_DELTA = 1e-3
const BREAK_EPS = 1e-9          # width (in ln T) of the node pair placed at each breakpoint

"""Uniform axis from a to b with step close to h (endpoints included)."""
function uniform_axis(a, b, h)
    n = ceil(Int, (b - a) / h)
    return collect(range(a, b; length = n + 1))
end

"""
Temperatures [K] where the EOS model is not smooth in T (implementation doc §20):
  * Wittmann partition-function grid (U(T) is linearly interpolated: slope jumps in ρ, nₑ, χ);
  * opacity jumps: H⁻ treatment at 7730 K, COOLOP below 12 kK, LUKEOP below 30 kK;
  * opacity kinks: Mg I / Si I table switches at 5–11 kK (1 kK steps), Si II at 12–18 kK
    (2 kK steps), and the integer switches of COULFF's bilinear Gaunt-factor table.
"""
function model_breaks(m::EOSModel)
    T = Float64[]
    append!(T, witt_tpf(m.ctxs[1]))
    append!(T, (7730.0, 12000.0, 30000.0))
    append!(T, 5000.0:1000.0:11000.0)
    append!(T, 12000.0:2000.0:18000.0)
    freqlg = log(2.997925e18 / WL500)
    for n in 1:11                      # IHVKT = int(HVKTLG + 9) switches
        push!(T, exp(freqlg - 1.15129 * (n + 11.63764)))
    end
    for z4 in (Z4LOG[1], Z4LOG[2]), n in 1:10    # IGAM = int(GAMLOG + 7) switches, nz = 1, 2
        push!(T, exp(1.15129 * (17.39638 + z4 - n)))
    end
    return sort(unique(T))
end

"""Smallest X with exp(X) >= T (so that the node lies on the right side of a breakpoint)."""
function xright(T)
    X = log(T)
    while exp(X) < T
        X = nextfloat(X)
    end
    while exp(prevfloat(X)) >= T
        X = prevfloat(X)
    end
    return X
end

"""Uniform X axis plus a node pair (b−ε, b) at every breakpoint strictly inside the range."""
function axis_with_breaks(Xu::Vector{Float64}, breaksX::Vector{Float64})
    h = Xu[2] - Xu[1]
    inside = filter(b -> Xu[1] + BREAK_EPS < b < Xu[end] - BREAK_EPS, breaksX)
    keep = filter(x -> all(abs(x - b) > 0.25h for b in inside) || x == Xu[1] || x == Xu[end], Xu)
    return sort(unique(vcat(keep, inside, inside .- BREAK_EPS)))
end

"""
X-derivative stencil for a node: offsets (in units of δ') and coefficients, chosen so that
the stencil stays inside the smooth interval [b_k, b_{k+1}) containing X.
"""
function xstencil(X, breaksX::Vector{Float64})
    k = searchsortedlast(breaksX, X)                  # breaksX[k] <= X < breaksX[k+1]
    L = k >= 1 ? X - breaksX[k] : Inf
    R = k < length(breaksX) ? breaksX[k+1] - X : Inf
    δ = min(TABLE_DELTA, max(L, R) / 2.5)
    if L >= δ && R > δ
        return (-1, 0, 1), (-0.5, 0.0, 0.5), δ
    elseif R >= L
        δ = min(TABLE_DELTA, R / 2.5)
        return (0, 1, 2), (-1.5, 2.0, -0.5), δ
    else
        δ = min(TABLE_DELTA, L / 2.5)
        return (-2, -1, 0), (0.5, -2.0, 1.5), δ
    end
end

"""Evaluate one node from 9 model evaluations → (4, 3) block; returns flags."""
function node_values!(out::AbstractArray{Float64,2}, m::EOSModel, X, Y, st, pfs, n17)
    offs, coef, δx = st
    δ = TABLE_DELTA
    F = Array{Float64}(undef, 3, 3, 3)          # (field, x-slot, y-slot)
    fl = EOS_OK
    ok = true
    for (ia, a) in enumerate(offs), (ib, bb) in enumerate((-1, 0, 1))
        r, n, c, f, good = model_eval(m, X + a * δx, Y + bb * δ; n17 = n17, pf = pfs[ia])
        F[1, ia, ib] = r; F[2, ia, ib] = n; F[3, ia, ib] = c
        (a == 0 && bb == 0) && (fl = f)
        ok &= good
    end
    i0 = findfirst(==(0), offs)
    for k in 1:3
        out[1, k] = F[k, i0, 2]
        out[2, k] = sum(coef[ia] * F[k, ia, 2] for ia in 1:3) / δx
        out[3, k] = (F[k, i0, 3] - F[k, i0, 1]) / (2δ)
        out[4, k] = sum(coef[ia] * (F[k, ia, 3] - F[k, ia, 1]) for ia in 1:3) / (2δ * δx)
    end
    ok || (fl |= EOS_DIRECT_FAILED)
    return fl & (EOS_ROBUST_FALLBACK | EOS_DIRECT_FAILED)
end

"""Fill nodes (i, j) with need[i, j] (threaded over X columns, partition functions cached)."""
function compute_nodes!(V, flags, m::EOSModel, X, Y, need::BitMatrix, breaksX)
    cols = [i for i in eachindex(X) if any(@view need[i, :])]
    Threads.@threads :dynamic for i in cols
        tid = Threads.threadid()
        h = m.hots[tid]
        st = xstencil(X[i], breaksX)
        pfs = [hot_pf(h, exp(X[i] + a * st[3])) for a in st[1]]
        n17 = zeros(17)
        out = zeros(4, 3)
        for j in eachindex(Y)
            need[i, j] || continue
            flags[i, j] = node_values!(out, m, X[i], Y[j], st, pfs, n17)
            @views V[:, :, i, j] .= out
        end
    end
end

"""
    check_cells(tab, m; tol) -> (xmark, ymark, maxerr)

Compare the interpolant with the model at X-edge, Y-edge and centre midpoints of every cell.
"""
function check_cells(tab::EOSTable, m::EOSModel; tol = (5e-6, 5e-5, 5e-5))
    nX, nY = length(tab.X), length(tab.Y)
    xmark = falses(nX - 1); ymark = falses(nY - 1)
    maxerr = zeros(3, Threads.maxthreadid())
    xm_t = [falses(nX - 1) for _ in 1:Threads.maxthreadid()]
    ym_t = [falses(nY - 1) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :dynamic for p in 1:nX-1
        tab.X[p+1] - tab.X[p] < 1e-6 && continue          # breakpoint pair
        tid = Threads.threadid()
        n17 = zeros(17)
        Xs = (0.5 * (tab.X[p] + tab.X[p+1]), tab.X[p])
        pfm = hot_pf(m.hots[tid], exp(Xs[1])); pf0 = hot_pf(m.hots[tid], exp(Xs[2]))
        for q in 1:nY-1
            Ym = 0.5 * (tab.Y[q] + tab.Y[q+1])
            for (X, Y, kind) in ((Xs[1], tab.Y[q], 1), (tab.X[p], Ym, 2), (Xs[1], Ym, 3))
                r, n, c, _, good = model_eval(m, X, Y; n17 = n17, pf = kind == 2 ? pf0 : pfm)
                good || continue
                ri, ni, ci, _ = state(tab, X, Y)
                e = (abs(ri - r), abs(ni - n), abs(ci - c))
                for k in 1:3
                    maxerr[k, tid] = max(maxerr[k, tid], e[k])
                end
                if any(e[k] > tol[k] for k in 1:3)
                    kind != 2 && (xm_t[tid][p] = true)
                    kind != 1 && (ym_t[tid][q] = true)
                end
            end
        end
    end
    for t in eachindex(xm_t)
        xmark .|= xm_t[t]; ymark .|= ym_t[t]
    end
    return xmark, ymark, vec(maximum(maxerr; dims = 2))
end

"""
    build_table(m, env; dlog10T, dlog10P, refine_max_levels, log=stdout) -> EOSTable
"""
function build_table(m::EOSModel, env::Envelope; dlog10T = 0.002, dlog10P = 0.02,
                     refine_max_levels = 5, log::IO = stdout)
    t0 = time()
    breaksX = [xright(T) for T in model_breaks(m)]
    X = axis_with_breaks(uniform_axis(Base.log(env.T_lo), Base.log(env.T_hi), dlog10T * Base.log(10.0)), breaksX)
    Y = uniform_axis(env.log10P_lo * Base.log(10.0), env.log10P_hi * Base.log(10.0), dlog10P * Base.log(10.0))
    V = fill(NaN, 4, 3, length(X), length(Y))
    flags = zeros(UInt8, length(X), length(Y))
    compute_nodes!(V, flags, m, X, Y, trues(length(X), length(Y)), breaksX)
    @printf(log, "[%7.1fs] table: %d × %d nodes computed\n", time() - t0, length(X), length(Y)); flush(log)
    meta = Dict{String,Any}("refinement" => String[], "breaks_T" => exp.(breaksX))
    tab = EOSTable(X, Y, V, flags, m.T_sw, m.hot_branch, m.mass_per_H, meta)
    for level in 1:refine_max_levels
        xm, ym, maxerr = check_cells(tab, m)
        msg = @sprintf("level %d: %d × %d nodes, midpoint max |Δ| = %.2e / %.2e / %.2e (ln ρ / ln nₑ / ln χ); marked %d X and %d Y intervals",
                       level - 1, length(tab.X), length(tab.Y), maxerr..., count(xm), count(ym))
        push!(meta["refinement"], msg)
        @printf(log, "[%7.1fs] table: %s\n", time() - t0, msg); flush(log)
        (count(xm) == 0 && count(ym) == 0) && break
        level == refine_max_levels && break
        tab = refine(tab, m, xm, ym, breaksX)
    end
    return tab
end

"""Insert midpoints of marked intervals; reuse old nodes, compute only new ones."""
function refine(tab::EOSTable, m::EOSModel, xm::BitVector, ym::BitVector, breaksX)
    X = sort(vcat(tab.X, [0.5 * (tab.X[p] + tab.X[p+1]) for p in findall(xm)]))
    Y = sort(vcat(tab.Y, [0.5 * (tab.Y[q] + tab.Y[q+1]) for q in findall(ym)]))
    ix = [searchsortedfirst(X, x) for x in tab.X]
    iy = [searchsortedfirst(Y, y) for y in tab.Y]
    V = fill(NaN, 4, 3, length(X), length(Y))
    flags = zeros(UInt8, length(X), length(Y))
    need = trues(length(X), length(Y))
    for (jo, jn) in enumerate(iy), (io, inn) in enumerate(ix)
        @views V[:, :, inn, jn] .= tab.V[:, :, io, jo]
        flags[inn, jn] = tab.flags[io, jo]
        need[inn, jn] = false
    end
    compute_nodes!(V, flags, m, X, Y, need, breaksX)
    return EOSTable(X, Y, V, flags, tab.T_sw, tab.hot_branch, tab.mass_per_H, tab.meta)
end

# ---------------------------------------------------------------------------------------
# Validation (Gate 1)
# ---------------------------------------------------------------------------------------

"""
    validate_table(tab, m; n=10^6, seed=20261016, nbins=(60, 50))

Withheld uniform random points: max and p99.9 of |Δ| per field, plus a coarse (log T, log P)
map of the max error, and the count of model failures.
"""
function validate_table(tab::EOSTable, m::EOSModel; n = 10^6, seed = 20261016, nbins = (60, 50))
    rng = Random.Xoshiro(seed)
    Xs = tab.X[1] .+ (tab.X[end] - tab.X[1]) .* rand(rng, n)
    Ys = tab.Y[1] .+ (tab.Y[end] - tab.Y[1]) .* rand(rng, n)
    err = fill(NaN, 3, n)
    nfail = zeros(Int, Threads.maxthreadid())
    Threads.@threads :dynamic for i in 1:n
        n17 = zeros(17)
        r, ne, c, _, good = model_eval(m, Xs[i], Ys[i]; n17 = n17)
        if !good
            nfail[Threads.threadid()] += 1
            continue
        end
        ri, ni, ci, _ = state(tab, Xs[i], Ys[i])
        err[1, i] = abs(ri - r); err[2, i] = abs(ni - ne); err[3, i] = abs(ci - c)
    end
    ok = .!isnan.(err[1, :])
    stats = Dict{String,Float64}()
    for (k, nm) in enumerate(LN_FIELDS)
        e = err[k, ok]
        stats[nm*"_max"] = maximum(e)
        stats[nm*"_p999"] = quantile(e, 0.999)
    end
    bx = clamp.(floor.(Int, (Xs .- tab.X[1]) ./ (tab.X[end] - tab.X[1]) .* nbins[1]) .+ 1, 1, nbins[1])
    by = clamp.(floor.(Int, (Ys .- tab.Y[1]) ./ (tab.Y[end] - tab.Y[1]) .* nbins[2]) .+ 1, 1, nbins[2])
    maps = zeros(3, nbins...)
    for i in 1:n
        ok[i] || continue
        for k in 1:3
            maps[k, bx[i], by[i]] = max(maps[k, bx[i], by[i]], err[k, i])
        end
    end
    worst = Dict{String,Tuple{Float64,Float64}}()
    for (k, nm) in enumerate(LN_FIELDS)
        e = copy(err[k, :]); e[.!ok] .= -1
        i = argmax(e)
        worst[nm] = (exp(Xs[i]), exp(Ys[i]))
    end
    return (stats = stats, maps = maps, worst = worst, nfail = sum(nfail), n = n)
end

"""
    derivative_check(tab, m; n=10^4) -> max |∂lnρ/∂lnP (table) − FD of the model|

For each point, central differences of the model with steps 1e-2 … 1e-4; the reference is
the step whose value differs least from the next-smaller step (§7.9).
"""
function derivative_check(tab::EOSTable, m::EOSModel; n = 10^4, seed = 7)
    rng = Random.Xoshiro(seed)
    steps = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
    errs = fill(NaN, n)
    where_ = Vector{Tuple{Float64,Float64}}(undef, n)
    Xlo, Xhi = tab.X[1] + 0.02, tab.X[end] - 0.02
    Ylo, Yhi = tab.Y[1] + 0.02, tab.Y[end] - 0.02
    Xs = Xlo .+ (Xhi - Xlo) .* rand(rng, n)
    Ys = Ylo .+ (Yhi - Ylo) .* rand(rng, n)
    Threads.@threads :dynamic for i in 1:n
        n17 = zeros(17)
        X, Y = Xs[i], Ys[i]
        d = map(steps) do h
            rp = model_eval(m, X, Y + h; n17 = n17)
            rm = model_eval(m, X, Y - h; n17 = n17)
            (rp[5] && rm[5]) ? (rp[1] - rm[1]) / (2h) : NaN
        end
        any(isnan, d) && continue
        diffs = [abs(d[k] - d[k+1]) for k in 1:length(d)-1]
        ref = d[argmin(diffs)]
        _, dt, _ = lnrho_dY(tab, X, Y)
        errs[i] = abs(dt - ref)
        where_[i] = (exp(X), exp(Y))
    end
    ok = .!isnan.(errs)
    i = argmax(replace(errs, NaN => -1.0))
    return (max = maximum(errs[ok]), p999 = quantile(errs[ok], 0.999), n = count(ok), worst = where_[i])
end

# ---------------------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------------------

function table_sha256(tab::EOSTable)
    ctx = SHA.SHA2_256_CTX()
    for a in (tab.X, tab.Y, vec(tab.V), vec(tab.flags))
        SHA.update!(ctx, reinterpret(UInt8, a))
    end
    return bytes2hex(SHA.digest!(ctx))
end

"""Write the table and its provenance; `meta` values must be HDF5-attribute compatible."""
function save_table(path::AbstractString, tab::EOSTable, meta::AbstractDict)
    mkpath(dirname(abspath(path)))
    h5open(path, "w") do f
        f["X"] = tab.X
        f["Y"] = tab.Y
        for (k, nm) in enumerate(LN_FIELDS), (c, comp) in enumerate(("f", "fX", "fY", "fXY"))
            f["$nm/$comp"] = tab.V[c, k, :, :]
        end
        f["flags"] = tab.flags
        g = create_group(f, "meta")
        _write_meta(g, meta)
        write_attribute(f, "T_sw", tab.T_sw)
        write_attribute(f, "hot_branch", tab.hot_branch)
        write_attribute(f, "mass_per_H_kg", tab.mass_per_H)
        write_attribute(f, "version", "1")
        write_attribute(f, "table_sha256", table_sha256(tab))
    end
end

"""Load a table; verifies its SHA-256 and (optionally) the partition-function file hash."""
function load_table(path::AbstractString; pf_file = nothing)
    isfile(path) || invalid("EOS table $path does not exist (build it with scripts/build_eos_table.jl)")
    h5open(path, "r") do f
        X = read(f["X"]); Y = read(f["Y"])
        V = Array{Float64}(undef, 4, 3, length(X), length(Y))
        for (k, nm) in enumerate(LN_FIELDS), (c, comp) in enumerate(("f", "fX", "fY", "fXY"))
            V[c, k, :, :] = read(f["$nm/$comp"])
        end
        flags = read(f["flags"])
        meta = Dict{String,Any}()
        for k in keys(attributes(f["meta"]))
            meta[k] = read_attribute(f["meta"], k)
        end
        tab = EOSTable(X, Y, V, flags, read_attribute(f, "T_sw"), read_attribute(f, "hot_branch"),
                       read_attribute(f, "mass_per_H_kg"), meta)
        table_sha256(tab) == read_attribute(f, "table_sha256") || invalid("EOS table $path: SHA-256 mismatch")
        if pf_file !== nothing && haskey(meta, "pf_sha256")
            file_sha256(pf_file) == meta["pf_sha256"] ||
                invalid("EOS table $path was built with a different partition-function file")
        end
        return tab
    end
end
