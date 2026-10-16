# Per-layer statistics (implementation doc §5.6).
#
# Quantiles are exact and decomposition-independent: each layer is gathered to rank 0,
# sorted, and evaluated with Statistics.quantile (type 7, same as numpy's default).

"""
    layer_values(par, val, k; stride=1) -> Vector{Float64}

Owned values `val(i, j, k)` of layer k on global nodes with gi ≡ 1, gj ≡ 1 (mod stride),
non-finite values dropped. `val` may be an array or a callable.
"""
function layer_values(par::ParCtx, val, k::Int; stride::Int = 1)
    b = par.block
    v = Float64[]
    sizehint!(v, b.nx * b.ny)
    @inbounds for j in 1:b.ny
        gj = b.j0 + j - 1
        (gj - 1) % stride == 0 || continue
        for i in 1:b.nx
            gi = b.i0 + i - 1
            (gi - 1) % stride == 0 || continue
            x = Float64(_q(val, i, j, k))
            isfinite(x) && push!(v, x)
        end
    end
    return v
end

"""Gather layer k's values on rank 0 (other ranks get an empty vector)."""
function gather_layer(par::ParCtx, v::Vector{Float64})
    assert_main(par)
    counts = MPI.Gather(Int32(length(v)), par.comm; root = 0)
    if rank0(par)
        buf = Vector{Float64}(undef, sum(counts))
        MPI.Gatherv!(v, MPI.VBuffer(buf, counts), par.comm; root = 0)
        return buf
    else
        MPI.Gatherv!(v, nothing, par.comm; root = 0)
        return Float64[]
    end
end

"""
    layer_quantiles(par, val, ps; stride=1, ks=1:NZ) -> Matrix (length(ks) × length(ps))

Exact per-layer quantiles of `val` (array or callable). Layers without finite values give
NaN. The result is identical on all ranks.
"""
function layer_quantiles(par::ParCtx, val, ps::AbstractVector; stride::Int = 1,
                         ks = 1:par.block.nz)
    out = fill(NaN, length(ks), length(ps))
    for (r, k) in enumerate(ks)
        vals = gather_layer(par, layer_values(par, val, k; stride = stride))
        if rank0(par) && !isempty(vals)
            sort!(vals)
            out[r, :] .= quantile(vals, ps; sorted = true)
        end
    end
    MPI.Bcast!(out, par.comm; root = 0)
    return out
end

"""Per-layer fraction of nodes where `pred(i, j, k)` holds (over subsampled owned nodes)."""
function layer_fraction(par::ParCtx, pred, ks = 1:par.block.nz; stride::Int = 1)
    b = par.block
    cnt = zeros(Int, 2, length(ks))
    for (r, k) in enumerate(ks)
        @inbounds for j in 1:b.ny
            ((b.j0 + j - 2) % stride == 0) || continue
            for i in 1:b.nx
                ((b.i0 + i - 2) % stride == 0) || continue
                cnt[2, r] += 1
                pred(i, j, k) && (cnt[1, r] += 1)
            end
        end
    end
    tot = MPI.Allreduce(cnt, +, par.comm)
    return tot[1, :] ./ tot[2, :]
end

"""Nearest level index to each target height (z top-first)."""
nearest_levels(z::AbstractVector, targets) = [argmin(abs.(z .- t)) for t in targets]
