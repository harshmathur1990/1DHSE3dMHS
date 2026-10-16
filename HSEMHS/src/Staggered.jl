# Staggered (Yee-type) mesh operators for the D9 scheme (HSE_MHS_D9_STAGGERED_ADDENDUM.md §2–4).
#
# Storage: index i of a staggered array means position i + ½ in that direction.
#   centres  (i, j, k)        k = 1:NZ   u, P, ρ
#   x-faces  (i+½, j, k)      k = 1:NZ   Bx, Rx
#   y-faces  (i, j+½, k)      k = 1:NZ   By, Ry
#   z-faces  (i, j, k+½)      k = 0:NZ   Bz, Rz   (k = 0 top boundary face, NZ bottom)
#   x-edges  (i, j+½, k+½)    k = 0:NZ   Ax, Jx
#   y-edges  (i+½, j, k+½)    k = 0:NZ   Ay, Jy
#   z-edges  (i+½, j+½, k)    k = 1:NZ   Az, Jz
# z is top first: "up" is k → k − 1, so ∂f/∂z at level k is (f_{k−½} − f_{k+½}) / h_z,k.
# All stencils use offsets in {−1, 0, +1}; one-cell x/y halos suffice.

const Field0 = Field          # same type; z range 0:NZ

"""Allocate a z-staggered field (k = 0:NZ)."""
function newfield0(par::ParCtx; fill::Float64 = 0.0)
    b = par.block
    a = Array{Float64,3}(undef, b.nx + 2HALO, b.ny + 2HALO, b.nz + 1)
    Threads.@threads :static for k in 1:b.nz+1
        @inbounds for j in axes(a, 2), i in axes(a, 1)
            a[i, j, k] = fill
        end
    end
    return OffsetArray(a, 0:b.nx+1, 0:b.ny+1, 0:b.nz)
end

struct StagOps
    dxc::OffsetVector{Float64,Vector{Float64}}   # cell width Δx_i            (0:nx+1)
    dxf::OffsetVector{Float64,Vector{Float64}}   # centre spacing Δx_{i+½}     (0:nx+1)
    dyc::OffsetVector{Float64,Vector{Float64}}
    dyf::OffsetVector{Float64,Vector{Float64}}
    zc::Vector{Float64}                          # centre heights (1:NZ)
    zf::OffsetVector{Float64,Vector{Float64}}    # face heights z_{k+½}        (0:NZ)
    hz::Vector{Float64}                          # cell heights h_z,k = z_{k−½} − z_{k+½} (1:NZ)
    dzf::OffsetVector{Float64,Vector{Float64}}   # Δz_{k+½} = z_k − z_{k+1}    (0:NZ; boundary entries = h of the adjacent cell, unused)
    nz::Int
end

function StagOps(grid::Grid, par::ParCtx)
    b = par.block
    function hor(a::Axis, i0, n)
        N = length(a); L = a.period
        rng = 0:n+1
        gi = [mod1(i0 + i - 1, N) for i in rng]
        c = OffsetArray(a.h[gi], rng)                                   # cell widths (dual)
        f = OffsetArray([gi[q] == N ? a.s[1] + L - a.s[N] : a.s[gi[q]+1] - a.s[gi[q]] for q in eachindex(gi)], rng)
        return c, f
    end
    dxc, dxf = hor(grid.x, b.i0, b.nx)
    dyc, dyf = hor(grid.y, b.j0, b.ny)
    z = grid.z.s; NZ = length(z)
    zf = OffsetArray(zeros(NZ + 1), 0:NZ)
    for k in 1:NZ-1
        zf[k] = 0.5 * (z[k] + z[k+1])
    end
    zf[0] = z[1] + 0.5 * (z[1] - z[2])
    zf[NZ] = z[NZ] - 0.5 * (z[NZ-1] - z[NZ])
    hz = [zf[k-1] - zf[k] for k in 1:NZ]
    dzf = OffsetArray(zeros(NZ + 1), 0:NZ)
    for k in 1:NZ-1
        dzf[k] = z[k] - z[k+1]
    end
    dzf[0] = hz[1]; dzf[NZ] = hz[NZ]
    return StagOps(dxc, dxf, dyc, dyf, copy(z), zf, hz, dzf, NZ)
end

"""Apply fun(i, j, k) at every owned (i, j) and every k in kr, threaded over (j, k) rows."""
function forrows(fun::F, kr, par::ParCtx) where {F}
    b = par.block
    kv = collect(kr)
    Threads.@threads :static for jk in 1:(b.ny*length(kv))
        j = (jk - 1) % b.ny + 1
        k = kv[(jk-1)÷b.ny+1]
        @inbounds for i in 1:b.nx
            fun(i, j, k)
        end
    end
    return nothing
end

"""`@rows kr expr`: shorthand for `forrows((i, j, k) -> expr, kr, par)` (fully escaped)."""
macro rows(kr, body)
    return esc(:(forrows((i, j, k) -> $body, $kr, par)))
end

# ---------------------------------------------------------------------------------------
# Curl C: edges → faces, B = curl A + B̄z ẑ
# ---------------------------------------------------------------------------------------
@inline curlA_x(Ay, Az, s::StagOps, i, j, k) =
    @inbounds (Az[i, j, k] - Az[i, j-1, k]) / s.dyc[j] - (Ay[i, j, k-1] - Ay[i, j, k]) / s.hz[k]
@inline curlA_y(Ax, Az, s::StagOps, i, j, k) =
    @inbounds (Ax[i, j, k-1] - Ax[i, j, k]) / s.hz[k] - (Az[i, j, k] - Az[i-1, j, k]) / s.dxc[i]
@inline curlA_z(Ax, Ay, s::StagOps, i, j, k) =
    @inbounds (Ay[i, j, k] - Ay[i-1, j, k]) / s.dxc[i] - (Ax[i, j, k] - Ax[i, j-1, k]) / s.dyc[j]

"""B = curl A + B̄z ẑ on owned faces (A halos needed on the −1 side)."""
function curlA!(B::NTuple{3,Field}, A::NTuple{3,Field}, Bbar::Float64, s::StagOps, par::ParCtx)
    Ax, Ay, Az = A
    @rows 1:s.nz B[1][i, j, k] = curlA_x(Ay, Az, s, i, j, k)
    @rows 1:s.nz B[2][i, j, k] = curlA_y(Ax, Az, s, i, j, k)
    @rows 0:s.nz B[3][i, j, k] = curlA_z(Ax, Ay, s, i, j, k) + Bbar
    return B
end

"""Cᵀ: gA from face gradients gB (gB halos needed on the +1 side). Returns ∂/∂B̄z = Σ gBz."""
function curlA_T!(gA::NTuple{3,Field}, gB::NTuple{3,Field}, s::StagOps, par::ParCtx)
    gBx, gBy, gBz = gB
    NZ = s.nz
    # Ax (i, j+½, k+½), k = 0:NZ: in By(k) (−1/h_k), By(k+1) (+1/h_{k+1}), Bz(j) (−1/Δy_j), Bz(j+1) (+1/Δy_{j+1})
    @rows 0:NZ gA[1][i, j, k] = (k >= 1 ? -gBy[i, j, k] / s.hz[k] : 0.0) + (k + 1 <= NZ ? gBy[i, j, k+1] / s.hz[k+1] : 0.0) -
                                gBz[i, j, k] / s.dyc[j] + gBz[i, j+1, k] / s.dyc[j+1]
    # Ay (i+½, j, k+½): in Bx(k) (+1/h_k), Bx(k+1) (−1/h_{k+1}), Bz(i) (+1/Δx_i), Bz(i+1) (−1/Δx_{i+1})
    @rows 0:NZ gA[2][i, j, k] = (k >= 1 ? gBx[i, j, k] / s.hz[k] : 0.0) - (k + 1 <= NZ ? gBx[i, j, k+1] / s.hz[k+1] : 0.0) +
                                gBz[i, j, k] / s.dxc[i] - gBz[i+1, j, k] / s.dxc[i+1]
    # Az (i+½, j+½, k): in Bx(j) (+1/Δy_j), Bx(j+1) (−1/Δy_{j+1}), By(i) (−1/Δx_i), By(i+1) (+1/Δx_{i+1})
    @rows 1:NZ gA[3][i, j, k] = gBx[i, j, k] / s.dyc[j] - gBx[i, j+1, k] / s.dyc[j+1] -
                                gBy[i, j, k] / s.dxc[i] + gBy[i+1, j, k] / s.dxc[i+1]
    return sdot_ones(par, gBz)
end

"""Divergence of face fields at a centre (needs halos on the −1 side)."""
@inline div_face(B::NTuple{3,Field}, s::StagOps, i, j, k) =
    @inbounds (B[1][i, j, k] - B[1][i-1, j, k]) / s.dxc[i] + (B[2][i, j, k] - B[2][i, j-1, k]) / s.dyc[j] +
              (B[3][i, j, k-1] - B[3][i, j, k]) / s.hz[k]

# ---------------------------------------------------------------------------------------
# Dual curl C*: faces → edges (J·μ0). Boundary edges (k = 0, NZ) carry no z-difference.
# ---------------------------------------------------------------------------------------
@inline function curlB_x(By, Bz, s::StagOps, i, j, k)
    @inbounds t = (Bz[i, j+1, k] - Bz[i, j, k]) / s.dyf[j]
    (1 <= k <= s.nz - 1) && (@inbounds t -= (By[i, j, k] - By[i, j, k+1]) / s.dzf[k])
    return t
end
@inline function curlB_y(Bx, Bz, s::StagOps, i, j, k)
    @inbounds t = -(Bz[i+1, j, k] - Bz[i, j, k]) / s.dxf[i]
    (1 <= k <= s.nz - 1) && (@inbounds t += (Bx[i, j, k] - Bx[i, j, k+1]) / s.dzf[k])
    return t
end
@inline curlB_z(Bx, By, s::StagOps, i, j, k) =
    @inbounds (By[i+1, j, k] - By[i, j, k]) / s.dxf[i] - (Bx[i, j+1, k] - Bx[i, j, k]) / s.dyf[j]

"""C B (= μ0 J) on owned edges (B halos needed on the +1 side)."""
function curlB!(J::NTuple{3,Field}, B::NTuple{3,Field}, s::StagOps, par::ParCtx)
    Bx, By, Bz = B
    @rows 0:s.nz J[1][i, j, k] = curlB_x(By, Bz, s, i, j, k)
    @rows 0:s.nz J[2][i, j, k] = curlB_y(Bx, Bz, s, i, j, k)
    @rows 1:s.nz J[3][i, j, k] = curlB_z(Bx, By, s, i, j, k)
    return J
end

"""C*ᵀ: gB from edge gradients gJ (gJ halos needed on the −1 side)."""
function curlB_T!(gB::NTuple{3,Field}, gJ::NTuple{3,Field}, s::StagOps, par::ParCtx)
    gJx, gJy, gJz = gJ
    NZ = s.nz
    # Bx (i+½, j, k): Jy(k) +1/Δz_{k+½} [k ≤ NZ−1], Jy(k−1) −1/Δz_{k−½} [k ≥ 2], Jz(j) +1/Δy_{j+½}, Jz(j−1) −1/Δy_{j−½}
    @rows 1:NZ gB[1][i, j, k] = (k <= NZ - 1 ? gJy[i, j, k] / s.dzf[k] : 0.0) - (k >= 2 ? gJy[i, j, k-1] / s.dzf[k-1] : 0.0) +
                                gJz[i, j, k] / s.dyf[j] - gJz[i, j-1, k] / s.dyf[j-1]
    # By (i, j+½, k): Jx(k) −1/Δz_{k+½}, Jx(k−1) +1/Δz_{k−½}, Jz(i) −1/Δx_{i+½}, Jz(i−1) +1/Δx_{i−½}
    @rows 1:NZ gB[2][i, j, k] = -(k <= NZ - 1 ? gJx[i, j, k] / s.dzf[k] : 0.0) + (k >= 2 ? gJx[i, j, k-1] / s.dzf[k-1] : 0.0) -
                                gJz[i, j, k] / s.dxf[i] + gJz[i-1, j, k] / s.dxf[i-1]
    # Bz (i, j, k+½): Jx(j) −1/Δy_{j+½}, Jx(j−1) +1/Δy_{j−½}, Jy(i) +1/Δx_{i+½}, Jy(i−1) −1/Δx_{i−½}
    @rows 0:NZ gB[3][i, j, k] = -gJx[i, j, k] / s.dyf[j] + gJx[i, j-1, k] / s.dyf[j-1] +
                                gJy[i, j, k] / s.dxf[i] - gJy[i-1, j, k] / s.dxf[i-1]
    return gB
end

# ---------------------------------------------------------------------------------------
# H: faces → centres (data operator); G: centres → faces (compact gradient)
# ---------------------------------------------------------------------------------------
"""HB at owned centres (B halos needed on the −1 side)."""
function H!(HB::NTuple{3,Field}, B::NTuple{3,Field}, s::StagOps, par::ParCtx)
    @rows 1:s.nz HB[1][i, j, k] = 0.5 * (B[1][i-1, j, k] + B[1][i, j, k])
    @rows 1:s.nz HB[2][i, j, k] = 0.5 * (B[2][i, j-1, k] + B[2][i, j, k])
    @rows 1:s.nz HB[3][i, j, k] = 0.5 * (B[3][i, j, k-1] + B[3][i, j, k])
    return HB
end

"""Hᵀ: gB on owned faces from centre gradients gH (gH halos needed on the +1 side)."""
function H_T!(gB::NTuple{3,Field}, gH::NTuple{3,Field}, s::StagOps, par::ParCtx)
    NZ = s.nz
    @rows 1:NZ gB[1][i, j, k] = 0.5 * (gH[1][i, j, k] + gH[1][i+1, j, k])
    @rows 1:NZ gB[2][i, j, k] = 0.5 * (gH[2][i, j, k] + gH[2][i, j+1, k])
    @rows 0:NZ gB[3][i, j, k] = 0.5 * ((k >= 1 ? gH[3][i, j, k] : 0.0) + (k + 1 <= NZ ? gH[3][i, j, k+1] : 0.0))
    return gB
end

"""Compact gradient of centre field P on owned faces; G_z only on interior z-faces (1:NZ−1)."""
function G!(GP::NTuple{3,Field}, P::Field, s::StagOps, par::ParCtx)
    NZ = s.nz
    @rows 1:NZ GP[1][i, j, k] = (P[i+1, j, k] - P[i, j, k]) / s.dxf[i]
    @rows 1:NZ GP[2][i, j, k] = (P[i, j+1, k] - P[i, j, k]) / s.dyf[j]
    @rows 0:NZ GP[3][i, j, k] = (1 <= k <= NZ - 1) ? (P[i, j, k] - P[i, j, k+1]) / s.dzf[k] : 0.0
    return GP
end

"""Gᵀ: gP at owned centres from face gradients gR (gR halos needed on the −1 side)."""
function G_T!(gP::Field, gR::NTuple{3,Field}, s::StagOps, par::ParCtx)
    NZ = s.nz
    @rows 1:NZ gP[i, j, k] = gR[1][i-1, j, k] / s.dxf[i-1] - gR[1][i, j, k] / s.dxf[i] +
                             gR[2][i, j-1, k] / s.dyf[j-1] - gR[2][i, j, k] / s.dyf[j] +
                             (k <= NZ - 1 ? gR[3][i, j, k] / s.dzf[k] : 0.0) - (k >= 2 ? gR[3][i, j, k-1] / s.dzf[k-1] : 0.0)
    return gP
end

# ---------------------------------------------------------------------------------------
# Reductions over arbitrary z ranges
# ---------------------------------------------------------------------------------------
"""Deterministic global Σ a·c over owned i, j and the z axis of `a` (fixed row order)."""
function sdot(par::ParCtx, a::Field, c::Field)
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
    return MPI.Allreduce(t, +, par.comm)
end

"""Deterministic global Σ a over owned i, j and the z axis of `a`."""
function sdot_ones(par::ParCtx, a::Field)
    b = par.block
    kv = collect(axes(a, 3))
    p = zeros(b.ny, length(kv))
    Threads.@threads :static for jk in 1:(b.ny*length(kv))
        j = (jk - 1) % b.ny + 1; q = (jk - 1) ÷ b.ny + 1; k = kv[q]
        s = 0.0
        @inbounds for i in 1:b.nx
            s += a[i, j, k]
        end
        p[j, q] = s
    end
    t = 0.0
    for q in axes(p, 2), j in axes(p, 1)
        t += p[j, q]
    end
    return MPI.Allreduce(t, +, par.comm)
end
