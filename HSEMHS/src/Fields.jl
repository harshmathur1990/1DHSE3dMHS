# Haloed local arrays (implementation doc §4.2).
# Owned indices: i in 1:nx, j in 1:ny, k in 1:NZ. Halo: i in {0, nx+1}, j in {0, ny+1}.

const HALO = 1
const Field = OffsetArray{Float64,3,Array{Float64,3}}
const Mask3 = OffsetArray{UInt8,3,Array{UInt8,3}}

"""
    newfield(par; fill=0.0) -> Field

Allocate a haloed local field. Pages are first-touched inside a threaded loop with the
same (j, k) row partition the kernels use, so they land on the NUMA node that uses them.
"""
function newfield(par::ParCtx; fill::Float64 = 0.0)
    b = par.block
    a = Array{Float64,3}(undef, b.nx + 2HALO, b.ny + 2HALO, b.nz)
    Threads.@threads :static for k in 1:b.nz
        @inbounds for j in axes(a, 2), i in axes(a, 1)
            a[i, j, k] = fill
        end
    end
    return OffsetArray(a, 0:b.nx+1, 0:b.ny+1, 1:b.nz)
end

function newmask(par::ParCtx)
    b = par.block
    OffsetArray(zeros(UInt8, b.nx + 2HALO, b.ny + 2HALO, b.nz), 0:b.nx+1, 0:b.ny+1, 1:b.nz)
end

"""Copy a plain (nx, ny, nz) array of owned values into a field's owned region."""
function setowned!(f::Field, a::AbstractArray{<:Real,3})
    nx, ny, nz = size(a)
    Threads.@threads :static for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            f[i, j, k] = a[i, j, k]
        end
    end
    return f
end

"""Return the owned region as a plain Array (copy)."""
owned(f::OffsetArray{T,3}, par::ParCtx) where {T} =
    collect(f[1:par.block.nx, 1:par.block.ny, 1:par.block.nz])

"""Apply `g(i, j, k)` -> value to every owned node of `f` (threaded over rows)."""
function fill_owned!(g, f::AbstractArray{<:Real,3}, par::ParCtx)
    b = par.block
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            f[i, j, k] = g(i, j, k)
        end
    end
    return f
end
