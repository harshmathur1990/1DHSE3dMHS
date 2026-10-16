# Grid metrics and stencil coefficients (implementation doc §6).
#
# All coefficients come from signed coordinates, so decreasing z and nonuniform spacing
# need no special cases. Periodic axes wrap with period L (no duplicated endpoint).

struct Axis
    name::String
    s::Vector{Float64}
    periodic::Bool
    period::Float64            # NaN for non-periodic axes
    idx::Matrix{Int}           # N×3 node indices used by row n (wrapped)
    c1::Matrix{Float64}        # N×3 first-derivative coefficients
    c2::Matrix{Float64}        # N×3 second-derivative coefficients (same nodes)
    h::Vector{Float64}         # dual-cell width (> 0)
end

Base.length(a::Axis) = length(a.s)

# coefficients of the derivative at s0 of the quadratic through (s0, s1, s2),
# returned in node order (s0, s1, s2); d1 = s1 - s0, d2 = s2 - s0
function onesided_c1(d1, d2)
    return (-(d1 + d2) / (d1 * d2), d2 / (d1 * (d2 - d1)), -d1 / (d2 * (d2 - d1)))
end

# second derivative of the quadratic through (sa, sb, sc), coefficients in that order
function quad_c2(sa, sb, sc)
    return (2 / ((sa - sb) * (sa - sc)), 2 / ((sb - sa) * (sb - sc)), 2 / ((sc - sa) * (sc - sb)))
end

# centred row through (s_{n-1}, s_n, s_{n+1}); h1 = s_n - s_{n-1}, h2 = s_{n+1} - s_n
function centred_c1(h1, h2)
    return (-h2 / (h1 * (h1 + h2)), (h2 - h1) / (h1 * h2), h1 / (h2 * (h1 + h2)))
end

"""
    build_axis(name, s; periodic=false, period=NaN) -> Axis

For a periodic axis, `period` NaN means `N·Δ` and requires uniform spacing to 1e-9.
Rejects a duplicated endpoint.
"""
function build_axis(name::AbstractString, s::AbstractVector{<:Real}; periodic::Bool = false,
                    period::Real = NaN)
    s = Float64.(collect(s))
    N = length(s)
    N >= 3 || invalid("axis $name has $N nodes; at least 3 are required")
    ds = diff(s)
    (all(>(0), ds) || all(<(0), ds)) || invalid("axis $name is not strictly monotone")
    L = NaN
    if periodic
        Δ = ds[1]
        if isnan(period)
            maximum(abs.(ds .- Δ)) <= 1e-9 * abs(Δ) ||
                invalid("periodic axis $name is not uniform to 1e-9 and no period was given")
            L = N * Δ
        else
            L = Float64(period)
        end
        # duplicated endpoint: last node coincides with first + L
        abs(abs(s[N] - s[1]) - L) < 1e-6 * abs(Δ) &&
            invalid("periodic axis $name has a duplicated endpoint (s[N] - s[1] == period)")
        abs(abs(s[N] + Δ - s[1]) - L) <= 1e-6 * abs(Δ) * N ||
            invalid("periodic axis $name: period $L inconsistent with coordinates")
    end
    idx = zeros(Int, N, 3)
    c1 = zeros(N, 3)
    c2 = zeros(N, 3)
    h = zeros(N)
    sgn = sign(ds[1])
    for n in 1:N
        if periodic || (2 <= n <= N - 1)
            nm = n == 1 ? N : n - 1
            np = n == N ? 1 : n + 1
            sm = n == 1 ? s[N] - sgn * L : s[nm]
            sp = n == N ? s[1] + sgn * L : s[np]
            idx[n, :] .= (nm, n, np)
            c1[n, :] .= centred_c1(s[n] - sm, sp - s[n])
            c2[n, :] .= quad_c2(sm, s[n], sp)
            h[n] = abs(sp - sm) / 2
        elseif n == 1
            idx[n, :] .= (1, 2, 3)
            c1[n, :] .= onesided_c1(s[2] - s[1], s[3] - s[1])
            c2[n, :] .= quad_c2(s[1], s[2], s[3])
            h[n] = abs(s[2] - s[1]) / 2
        else # n == N
            idx[n, :] .= (N, N - 1, N - 2)
            c1[n, :] .= onesided_c1(s[N-1] - s[N], s[N-2] - s[N])
            c2[n, :] .= quad_c2(s[N], s[N-1], s[N-2])
            h[n] = abs(s[N] - s[N-1]) / 2
        end
    end
    return Axis(String(name), s, periodic, L, idx, c1, c2, h)
end

struct Grid
    x::Axis
    y::Axis
    z::Axis
    NX::Int
    NY::Int
    NZ::Int
    wmean::Float64             # global mean cell volume
end

function Grid(x::Axis, y::Axis, z::Axis)
    x.periodic && y.periodic || invalid("non-periodic lateral boundaries are not implemented in v1")
    z.periodic && invalid("z cannot be periodic")
    issorted(x.s) && issorted(y.s) || invalid("x and y must be strictly increasing")
    issorted(z.s; rev = true) || invalid("z must be strictly decreasing (top first)")
    NX, NY, NZ = length(x), length(y), length(z)
    wmean = sum(x.h) * sum(y.h) * sum(z.h) / (NX * NY * NZ)
    return Grid(x, y, z, NX, NY, NZ, wmean)
end

# ---------------------------------------------------------------------------------------
# Per-rank stencil data
# ---------------------------------------------------------------------------------------

"""Horizontal axis restricted to a rank: rows for local i in 0:n+1 (halo rows included)."""
struct LocalHAxis
    c1::OffsetArray{Float64,2,Matrix{Float64}}   # (0:n+1, 3)
    c2::OffsetArray{Float64,2,Matrix{Float64}}
    h::OffsetVector{Float64,Vector{Float64}}     # (0:n+1)
    gidx::OffsetVector{Int,Vector{Int}}          # global index of local i (wrapped)
end

function LocalHAxis(a::Axis, i0::Int, n::Int)
    N = length(a)
    rng = 0:n+1
    g = [mod1(i0 + i - 1, N) for i in rng]
    c1 = OffsetArray(a.c1[g, :], rng, 1:3)
    c2 = OffsetArray(a.c2[g, :], rng, 1:3)
    h = OffsetArray(a.h[g], rng)
    return LocalHAxis(c1, c2, h, OffsetArray(g, rng))
end

"""Vertical axis (never decomposed) with transposed gather stencils."""
struct LocalZAxis
    idx::Matrix{Int}           # NZ×3
    c1::Matrix{Float64}
    c2::Matrix{Float64}
    h::Vector{Float64}
    tidx::Matrix{Int}          # NZ×W: rows referencing node m (unused slots: m, coef 0)
    tc1::Matrix{Float64}       # NZ×W: D[row, m]
    tc2::Matrix{Float64}       # NZ×W: D2[row, m]
end

function LocalZAxis(a::Axis)
    N = length(a)
    lists = [Tuple{Int,Float64,Float64}[] for _ in 1:N]
    for n in 1:N, q in 1:3
        push!(lists[a.idx[n, q]], (n, a.c1[n, q], a.c2[n, q]))
    end
    W = maximum(length, lists)          # 4 for NZ >= 6, 5 for NZ = 5, 3 for NZ = 3
    tidx = zeros(Int, N, W)
    tc1 = zeros(N, W)
    tc2 = zeros(N, W)
    for m in 1:N
        for (q, (n, v1, v2)) in enumerate(lists[m])
            tidx[m, q] = n; tc1[m, q] = v1; tc2[m, q] = v2
        end
        for q in length(lists[m])+1:W
            tidx[m, q] = m
        end
    end
    return LocalZAxis(a.idx, a.c1, a.c2, a.h, tidx, tc1, tc2)
end

"""Everything a rank needs to apply operators: local x, y axes and the z axis."""
struct Ops
    X::LocalHAxis
    Y::LocalHAxis
    Z::LocalZAxis
    wmean::Float64
end

Ops(grid::Grid, par::ParCtx) = Ops(LocalHAxis(grid.x, par.block.i0, par.block.nx),
                                   LocalHAxis(grid.y, par.block.j0, par.block.ny),
                                   LocalZAxis(grid.z), grid.wmean)

"""Volume weight w = V/mean(V) at local (i, j, k)."""
@inline weight(o::Ops, i, j, k) = @inbounds o.X.h[i] * o.Y.h[j] * o.Z.h[k] / o.wmean

"""Local mean spacing (E9)."""
@inline hmean(o::Ops, i, j, k) = @inbounds (o.X.h[i] + o.Y.h[j] + o.Z.h[k]) / 3
