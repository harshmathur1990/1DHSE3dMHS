# τ₅₀₀ along inclined rays on the height grid (design/tau/VIEWING_ANGLE_DESIGN.md §5). A ray reaching the observer is
# x = ξ + t_x z, y = η + t_y z (t = tan θ (cos φ, sin φ)) through its footprint (ξ, η), which lies on the same regular
# grid as (x, y). Shear: χ on every level k is shifted to the footprint grid, χ̃(ξ, η, z_k) = χ(ξ + t_x z_k, η + t_y z_k),
# by periodic bilinear interpolation (positive, so τ stays monotonic); each footprint column is integrated as a vertical
# column with path length dz/μ (`tau_column!`, same top closure), which gives z(τ_LOS = 1) on the footprints; τ̃ is then
# shifted back to the grid points, τ(x, y, z_k) = τ̃(x − t_x z_k, y − t_y z_k). Each level's plane is gathered on every
# rank (any 2-D block decomposition, x slabs included), one plane at a time.

"""Global (NY, NX) plane from every rank's owned (ny, nx) block (collective; any 2-D block decomposition)."""
function allgather_plane(par::ParCtx, loc::AbstractMatrix{Float64})
    par.nranks == 1 && return Matrix{Float64}(loc)
    b = par.block
    meta = MPI.Allgather(Int[b.j0, b.ny, b.i0, b.nx], par.comm)
    counts = [meta[4r+2] * meta[4r+4] for r in 0:par.nranks-1]
    buf = zeros(sum(counts))
    MPI.Allgatherv!(vec(Matrix{Float64}(loc)), MPI.VBuffer(buf, counts), par.comm)
    G = zeros(par.NY, par.NX)
    off = 0
    for r in 0:par.nranks-1
        j0, ny, i0, nx = meta[4r+1], meta[4r+2], meta[4r+3], meta[4r+4]
        G[j0:j0+ny-1, i0:i0+nx-1] .= reshape(view(buf, off+1:off+ny*nx), ny, nx)
        off += ny * nx
    end
    return G
end

"""Periodic bilinear value of the plane G (NY, NX) at 0-based fractional indices (fj, fi)."""
@inline function pbilin(G::AbstractMatrix{Float64}, fj::Float64, fi::Float64)
    NY, NX = size(G)
    j0 = floor(Int, fj); i0 = floor(Int, fi)
    wj = fj - j0; wi = fi - i0
    ja = mod(j0, NY) + 1; jb = mod(j0 + 1, NY) + 1; ia = mod(i0, NX) + 1; ib = mod(i0 + 1, NX) + 1
    @inbounds return (1 - wj) * ((1 - wi) * G[ja, ia] + wi * G[ja, ib]) + wj * ((1 - wi) * G[jb, ia] + wi * G[jb, ib])
end

"""Shift every level of the local (nz, ny, nx) block A by s·(t_x z_k / dx, t_y z_k / dy) grid cells (periodic)."""
function shear_levels(par::ParCtx, A::AbstractArray{Float64,3}, z, dx, dy, t, s)
    nz, ny, nx = size(A); b = par.block
    out = similar(A, Float64)
    for k in 1:nz
        G = allgather_plane(par, view(A, k, :, :))
        fx = s * t[1] * z[k] / dx; fy = s * t[2] * z[k] / dy
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny
                out[k, j, i] = pbilin(G, b.j0 + j - 2 + fy, b.i0 + i - 2 + fx)
            end
        end
    end
    return out
end

"""
    inclined_tau!(tau, zt1, ttm, par, chi, z, dx, dy, mu, t; top=:exp_tail, H_max=5.0e6)

Collective. `chi` [m⁻¹] and `tau` are this rank's owned (nz, ny, nx) block, `z` [m] top first, `t` = (t_x, t_y).
Fills `tau` with τ_LOS at the grid points and `zt1`, `ttm` (ny, nx) with z(τ_LOS = 1) and the top-closure method of
the ray through each owned footprint.
"""
function inclined_tau!(tau, zt1, ttm, par::ParCtx, chi, z, dx, dy, mu, t; top::Symbol = :exp_tail, H_max::Float64 = 5.0e6)
    nz, ny, nx = size(chi)
    sh = shear_levels(par, chi, z, dx, dy, t, 1.0)
    Threads.@threads for i in 1:nx
        tc = zeros(nz)
        @inbounds for j in 1:ny
            ttm[j, i] = tau_column!(tc, @view(sh[:, j, i]), z; top = top, H_max = H_max, mu = mu)
            zt1[j, i] = z_tau1(tc, z)
            sh[:, j, i] .= tc
        end
    end
    tau .= shear_levels(par, sh, z, dx, dy, t, -1.0)
    return tau
end
