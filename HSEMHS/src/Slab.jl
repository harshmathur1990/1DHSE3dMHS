# Slab decomposition for the Poisson-type modes (mhs-poisson-z, mhs-poisson-ltau) and the τ-grid column HSE.
#
# Arrays are (k, j, i) = (vertical, y, x) with the vertical index first (contiguous columns); the ranks own slabs in x
# (process grid px = ranks, py = 1, from `init_parallel`), y is complete on every rank. All MPI calls are on the main
# task, outside threaded loops (Parallel.jl rules). Reductions are exact in the sense that matters for reproducibility:
# quantiles are the true order statistics of the global data (no gathering, no per-rank approximation), so a run gives
# the same answer on any number of ranks up to the rounding of sums.
module Slab

using MPI, FFTW, LinearAlgebra, HDF5
import ..HSEMHS: ParCtx, init_parallel, ensure_mpi, rank0

export slab_par, owned_x, sddx!, sddy!, gsum, gdot, gnorm, gmax, gmin, gcount, gquantiles, gmedians, SlabFFT,
       fwd!, bwd!, coop_write, rprint

"""Slab context for an NX × NY grid: every rank owns a contiguous range of x columns and all of y."""
function slab_par(NX::Int, NY::Int, NZ::Int)
    ensure_mpi()
    nr = MPI.Comm_size(MPI.COMM_WORLD)
    NX ÷ nr >= 2 || error("$NX x columns cannot be split over $nr ranks (at least 2 per rank)")
    return init_parallel(NX, NY, max(NZ, 3); px = nr, py = 1)
end

"""Global x indices owned by this rank."""
owned_x(par::ParCtx) = par.block.i0:par.block.i0+par.block.nx-1

"""Println on rank 0 only (flushed)."""
rprint(par::ParCtx, s) = (rank0(par) && (println(s); flush(stdout)); nothing)

# ---- reductions ----
gsum(par::ParCtx, x::Real) = par.nranks == 1 ? Float64(x) : MPI.Allreduce(Float64(x), +, par.comm)
gsum(par::ParCtx, x::AbstractVector{<:Real}) = par.nranks == 1 ? Float64.(x) : MPI.Allreduce(Float64.(x), +, par.comm)
gdot(par::ParCtx, a, b) = gsum(par, dot(a, b))
gnorm(par::ParCtx, a) = sqrt(gsum(par, dot(a, a)))
gmax(par::ParCtx, x::Real) = par.nranks == 1 ? Float64(x) : MPI.Allreduce(Float64(x), max, par.comm)
gmin(par::ParCtx, x::Real) = par.nranks == 1 ? Float64(x) : MPI.Allreduce(Float64(x), min, par.comm)
gcount(par::ParCtx, n::Integer) = par.nranks == 1 ? Int(n) : MPI.Allreduce(Int(n), +, par.comm)

# order-preserving Int64 key of a Float64 (IEEE 754: flip the magnitude bits of negatives)
@inline okey(x::Float64) = (b = reinterpret(Int64, x); b >= 0 ? b : b ⊻ typemax(Int64))
@inline unkey(b::Int64) = reinterpret(Float64, b >= 0 ? b : b ⊻ typemax(Int64))

"""
    gkth(par, sets, ks) -> Vector{Float64}

The ks[s]-th smallest (1-based) of the global union over ranks of the finite values in sets[s], for every set at once:
bisection on the order-preserving integer keys (at most 64 rounds, one Allreduce of the per-set counts each), exact.
"""
function gkth(par::ParCtx, sets::Vector{Vector{Float64}}, ks::Vector{Int})
    ns = length(sets)
    keys = [sort!(Int64[okey(v) for v in s if isfinite(v)]) for s in sets]
    lo = [isempty(k) ? typemax(Int64) : k[1] for k in keys]; hi = [isempty(k) ? typemin(Int64) : k[end] for k in keys]
    if par.nranks > 1
        lo = MPI.Allreduce(lo, min, par.comm); hi = MPI.Allreduce(hi, max, par.comm)
    end
    for _ in 1:66
        all(lo .>= hi) && break
        mid = [lo[s] < hi[s] ? (lo[s] >> 1) + (hi[s] >> 1) + (lo[s] & hi[s] & 1) : lo[s] for s in 1:ns]  # floor mean, no overflow
        c = [searchsortedlast(keys[s], mid[s]) for s in 1:ns]
        c = par.nranks > 1 ? MPI.Allreduce(c, +, par.comm) : c
        for s in 1:ns
            lo[s] < hi[s] || continue
            c[s] >= ks[s] ? (hi[s] = mid[s]) : (lo[s] = mid[s] + 1)
        end
    end
    return unkey.(lo)
end

"""Global quantiles (type 7, as Statistics.quantile) of the finite values in each set, for each p."""
function gquantiles(par::ParCtx, sets::Vector{Vector{Float64}}, p::Real)
    N = gsum(par, Float64[count(isfinite, s) for s in sets])
    h = (N .- 1) .* p .+ 1
    k1 = clamp.(floor.(Int, h), 1, max.(1, Int.(N)))
    k2 = min.(k1 .+ 1, max.(1, Int.(N)))
    a = gkth(par, sets, k1); b = gkth(par, sets, k2)
    return [N[s] == 0 ? NaN : a[s] + (h[s] - k1[s]) * (b[s] - a[s]) for s in eachindex(sets)]
end
gquantiles(par::ParCtx, v::AbstractArray, ps::AbstractVector) = [gquantiles(par, [vec(Float64.(v))], p)[1] for p in ps]

"""Median over (j, i) of every vertical layer k of a local (k, j, i) array (or of f(k, j, i) for k in ks)."""
gmedians(par::ParCtx, A::AbstractArray{<:Real,3}) = gquantiles(par, [vec(Float64.(A[k, :, :])) for k in axes(A, 1)], 0.5)

# ---- x halos and derivatives (centred, periodic) ----
"""The planes just outside the owned x range: (left = global i0 − 1, right = global i1 + 1), each (nk, ny)."""
function xhalo(par::ParCtx, f::AbstractArray{T,3}) where {T}
    nk, ny, nx = size(f)
    if par.nranks == 1
        return f[:, :, nx], f[:, :, 1]
    end
    sendl = f[:, :, 1]; sendr = f[:, :, nx]; left = similar(sendl); right = similar(sendr)
    # send my first plane to the left neighbour (it is their right halo), receive my right halo from the right
    MPI.Sendrecv!(sendl, right, par.comm; dest = par.nbr_xm, sendtag = 1, source = par.nbr_xp, recvtag = 1)
    MPI.Sendrecv!(sendr, left, par.comm; dest = par.nbr_xp, sendtag = 2, source = par.nbr_xm, recvtag = 2)
    return left, right
end

function sddx!(o, f, d, par::ParCtx)
    nk, ny, nx = size(f)
    left, right = xhalo(par, f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, k in 1:nk
            fp = i == nx ? right[k, j] : f[k, j, i+1]; fm = i == 1 ? left[k, j] : f[k, j, i-1]
            o[k, j, i] = (fp - fm) / (2d)
        end
    end
    return o
end

function sddy!(o, f, d)
    nk, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            jp = j == ny ? 1 : j + 1; jm = j == 1 ? ny : j - 1
            for k in 1:nk
                o[k, j, i] = (f[k, jp, i] - f[k, jm, i]) / (2d)
            end
        end
    end
    return o
end

# ---- distributed 2-D FFT over (y, x) of (k, y, x) slabs ----
"""
Real-to-complex 2-D FFT of the horizontal planes of a slab array, distributed: rfft along y on each rank, an all-to-all
transpose to slabs of the y wavenumbers, then the FFT along x. The transformed data T has layout (k, x wavenumber,
local y wavenumber), so each mode's vertical vector T[:, kx, jl] is contiguous. `bwd!` inverts `fwd!` exactly.
"""
mutable struct SlabFFT
    nk::Int; ny::Int; nyh::Int; nx::Int; nxl::Int
    jr::Vector{UnitRange{Int}}       # y-wavenumber range owned by each rank
    ir::Vector{UnitRange{Int}}       # x range owned by each rank
    A::Array{ComplexF64,3}           # (nk, nyh, nxl) after the y transform
    T::Array{ComplexF64,3}           # (nk, nx, nyhl)
    py::Any; ipy::Any; px::Any; ipx::Any
    sbuf::Vector{ComplexF64}; rbuf::Vector{ComplexF64}
    scount::Vector{Int}; rcount::Vector{Int}
end

function SlabFFT(par::ParCtx, nk::Int, ny::Int)
    nx = par.NX; nxl = par.block.nx; nyh = ny ÷ 2 + 1; nr = par.nranks
    br(N, c) = (s = HSEMHS_block_range(N, nr, c); s[1]:s[1]+s[2]-1)
    jr = [br(nyh, c) for c in 0:nr-1]; ir = [br(nx, c) for c in 0:nr-1]
    me = par.rank + 1
    nyhl = length(jr[me])
    fl = FFTW.ESTIMATE | FFTW.UNALIGNED
    sc = [nk * length(jr[s]) * nxl for s in 1:nr]; rc = [nk * nyhl * length(ir[s]) for s in 1:nr]
    return SlabFFT(nk, ny, nyh, nx, nxl, jr, ir, zeros(ComplexF64, nk, nyh, nxl), zeros(ComplexF64, nk, nx, nyhl),
                   plan_rfft(zeros(nk, ny), 2; flags = fl), plan_irfft(zeros(ComplexF64, nk, nyh), ny, 2; flags = fl),
                   plan_fft(zeros(ComplexF64, nk, nx), 2; flags = fl), plan_ifft(zeros(ComplexF64, nk, nx), 2; flags = fl),
                   zeros(ComplexF64, sum(sc)), zeros(ComplexF64, sum(rc)), sc, rc)
end
HSEMHS_block_range(N, p, c) = (b = divrem(N, p); n = b[1] + (c < b[2] ? 1 : 0); (c * b[1] + min(c, b[2]) + 1, n))

"""Global y wavenumber index (1-based) of local T column jl."""
kyglobal(S::SlabFFT, par::ParCtx, jl) = S.jr[par.rank+1][1] + jl - 1

function fwd!(S::SlabFFT, par::ParCtx, r::AbstractArray{Float64,3})
    nk, nxl = S.nk, S.nxl
    chunks = collect(Iterators.partition(1:nxl, cld(nxl, Threads.nthreads())))
    Threads.@threads for ch in chunks
        a = zeros(nk, S.ny); b = zeros(ComplexF64, nk, S.nyh)
        for i in ch
            a .= @view r[:, :, i]; mul!(b, S.py, a); @views S.A[:, :, i] .= b
        end
    end
    transpose_fwd!(S, par)
    nyhl = size(S.T, 3)
    cj = collect(Iterators.partition(1:max(nyhl, 1), cld(max(nyhl, 1), Threads.nthreads())))
    Threads.@threads for ch in cj
        a = zeros(ComplexF64, nk, S.nx); b = similar(a)
        for jl in ch
            jl <= nyhl || continue
            a .= @view S.T[:, :, jl]; mul!(b, S.px, a); @views S.T[:, :, jl] .= b
        end
    end
    return S.T
end

function bwd!(S::SlabFFT, par::ParCtx, out::AbstractArray{Float64,3})
    nk, nxl = S.nk, S.nxl
    nyhl = size(S.T, 3)
    cj = collect(Iterators.partition(1:max(nyhl, 1), cld(max(nyhl, 1), Threads.nthreads())))
    Threads.@threads for ch in cj
        a = zeros(ComplexF64, nk, S.nx); b = similar(a)
        for jl in ch
            jl <= nyhl || continue
            a .= @view S.T[:, :, jl]; mul!(b, S.ipx, a); @views S.T[:, :, jl] .= b
        end
    end
    transpose_bwd!(S, par)
    chunks = collect(Iterators.partition(1:nxl, cld(nxl, Threads.nthreads())))
    Threads.@threads for ch in chunks
        a = zeros(ComplexF64, nk, S.nyh); b = zeros(nk, S.ny)
        for i in ch
            a .= @view S.A[:, :, i]; mul!(b, S.ipy, a); @views out[:, :, i] .= b    # c2r overwrites a (scratch)
        end
    end
    return out
end

function transpose_fwd!(S::SlabFFT, par::ParCtx)
    nr = par.nranks; me = par.rank + 1
    if nr == 1
        Threads.@threads for jl in 1:S.nyh
            @inbounds for i in 1:S.nx, k in 1:S.nk
                S.T[k, i, jl] = S.A[k, jl, i]
            end
        end
        return
    end
    off = 0
    for s in 1:nr                                    # pack: block for rank s = A[:, jr[s], :]
        for i in 1:S.nxl, j in S.jr[s], k in 1:S.nk
            off += 1; S.sbuf[off] = S.A[k, j, i]
        end
    end
    MPI.Alltoallv!(MPI.VBuffer(S.sbuf, S.scount), MPI.VBuffer(S.rbuf, S.rcount), par.comm)
    off = 0; nyhl = length(S.jr[me])
    for s in 1:nr                                    # unpack: from rank s, (k, jl, il) → T[k, ir[s][il], jl]
        for il in 1:length(S.ir[s]), jl in 1:nyhl, k in 1:S.nk
            off += 1; S.T[k, S.ir[s][il], jl] = S.rbuf[off]
        end
    end
end

function transpose_bwd!(S::SlabFFT, par::ParCtx)
    nr = par.nranks; me = par.rank + 1
    if nr == 1
        Threads.@threads for i in 1:S.nx
            @inbounds for jl in 1:S.nyh, k in 1:S.nk
                S.A[k, jl, i] = S.T[k, i, jl]
            end
        end
        return
    end
    off = 0; nyhl = length(S.jr[me])
    for s in 1:nr
        for il in 1:length(S.ir[s]), jl in 1:nyhl, k in 1:S.nk
            off += 1; S.rbuf[off] = S.T[k, S.ir[s][il], jl]
        end
    end
    MPI.Alltoallv!(MPI.VBuffer(S.rbuf, S.rcount), MPI.VBuffer(S.sbuf, S.scount), par.comm)
    off = 0
    for s in 1:nr
        for i in 1:S.nxl, j in S.jr[s], k in 1:S.nk
            off += 1; S.A[k, j, i] = S.sbuf[off]
        end
    end
end

# ---- cooperative HDF5 output (no parallel HDF5) ----
"""
    coop_write(par, path, slabs, small, attrs)

Rank 0 creates `path` with a global dataset for every `slabs` entry (name => local array whose FIRST dimension is the
owned x range: Julia (x, y, …), h5py (…, y, x), the MHS checkpoint layout), writes `small` entries whole and the
attributes; then the ranks in turn write their x hyperslabs.
"""
function coop_write(par::ParCtx, path, slabs, small, attrs::Dict)
    ir = owned_x(par)
    if rank0(par)
        h5open(path, "w") do f
            for (nm, a) in slabs
                create_dataset(f, nm, eltype(a), (par.NX, size(a)[2:end]...))
            end
            for (nm, v) in small
                f[nm] = v
            end
            for (k, v) in attrs
                attributes(f)[k] = v
            end
        end
    end
    for r in 0:par.nranks-1
        par.nranks > 1 && MPI.Barrier(par.comm)
        par.rank == r || continue
        h5open(path, "r+") do f
            for (nm, a) in slabs
                nd = ndims(a)
                nd == 1 ? (f[nm][ir] = a) : nd == 2 ? (f[nm][ir, :] = a) : (f[nm][ir, :, :] = a)
            end
        end
    end
    par.nranks > 1 && MPI.Barrier(par.comm)
end

end # module Slab
