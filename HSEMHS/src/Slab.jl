# Slab decomposition for the Poisson-type modes (mhs-poisson-z, -ltau, -logm) and the τ- and mass-grid column HSE.
#
# Arrays are (k, j, i) = (vertical, y, x) with the vertical index first (contiguous columns); the ranks own slabs in x
# (process grid px = ranks, py = 1, from `init_parallel`), y is complete on every rank. All MPI calls are on the main
# task, outside threaded loops (Parallel.jl rules). Reductions are exact in the sense that matters for reproducibility:
# quantiles are the true order statistics of the global data (no gathering, no per-rank approximation), so a run gives
# the same answer on any number of ranks up to the rounding of sums.
module Slab

using MPI, FFTW, LinearAlgebra, HDF5
import ..HSEMHS as H
import ..HSEMHS: ParCtx, init_parallel, ensure_mpi, rank0

export slab_par, slab_check, owned_x, sddx!, sddy!, gsum, gdot, gnorm, gmax, gmin, gcount, gquantiles, gmedians, SlabFFT,
       fwd!, bwd!, coop_write, rprint, @cols, tmap!, tdot, tsum

"""
Message if an NX-column grid cannot be split into x slabs over `nr` ranks, else `nothing`. The smallest `block_range`
slab is floor(NX/nr), so this is the exact condition for at least 2 columns per rank.
"""
slab_check(NX::Int, nr::Int) = NX ÷ nr >= 2 ? nothing : "$NX x columns cannot be split over $nr ranks (at least 2 per rank)"

"""
Slab context for an NX × NY grid on the parent communicator `comm`: every rank owns a contiguous range of x columns and
all of y. The context owns a duplicate of `comm` (see `init_parallel`); release it with `free_parallel!`.
"""
function slab_par(comm::MPI.Comm, NX::Int, NY::Int, NZ::Int)
    msg = slab_check(NX, MPI.Comm_size(comm))       # identical on every rank: no rank throws alone
    msg === nothing || H.invalid(msg)
    return init_parallel(comm, NX, NY, max(NZ, 3); px = MPI.Comm_size(comm), py = 1)
end

"""Standalone entry (file drivers): initialise MPI if needed and use `COMM_WORLD`."""
function slab_par(NX::Int, NY::Int, NZ::Int)
    ensure_mpi()
    return slab_par(MPI.COMM_WORLD, NX, NY, NZ)
end

"""Global x indices owned by this rank."""
owned_x(par::ParCtx) = par.block.i0:par.block.i0+par.block.nx-1

# ---- threading over the columns of a slab ----
"""
    @cols i j nx ny body
    @cols i j nx ny setup body

Threaded loop over every (j, i) column of a (·, ny, nx) slab, `i` in 1:nx and `j` in 1:ny, j fastest (memory order).
The ny·nx columns are split into contiguous chunks (8 per thread), so a rank with few x columns still uses all its
threads; `setup` (assignments of per-chunk scratch arrays) runs once per chunk before its columns. The body is a closure
called through a function barrier (`foreach_col`), so captured variables stay concretely typed; it must touch only its
own column (or read others) and must not use `return`, `continue` or `break` outside its own inner loops.
"""
macro cols(i, j, nx, ny, setup, body = nothing)
    body === nothing && ((setup, body) = (nothing, setup))
    if setup === nothing
        return :(foreach_col(($(esc(i)), $(esc(j))) -> $(esc(body)), $(esc(nx)), $(esc(ny))))
    end
    names = Symbol[]
    collect_names!(ex) = ex isa Expr && ex.head == :block ? foreach(collect_names!, ex.args) :
                         ex isa Expr && ex.head == :(=) ? (ex.args[1] isa Symbol ? push!(names, ex.args[1]) :
                                                           ex.args[1] isa Expr && ex.args[1].head == :tuple ?
                                                           append!(names, ex.args[1].args) : nothing) : nothing
    collect_names!(setup)
    tup = Expr(:tuple, map(esc, names)...)
    quote
        foreach_col(() -> begin $(esc(setup)); $tup end,
                    ($(esc(i)), $(esc(j)), $tup) -> $(esc(body)), $(esc(nx)), $(esc(ny)))
    end
end

"""Chunks of the nx·ny columns for `@cols`: (number of chunks, chunk length)."""
@inline colchunks(nc) = (nch = max(1, min(nc, 8 * Threads.nthreads())); (nch, cld(nc, nch)))

"""f(i, j) for every column (see `@cols`)."""
function foreach_col(f::F, nx::Integer, ny::Integer) where {F}
    nc = nx * ny; nch, len = colchunks(nc)
    Threads.@threads :dynamic for ch in 1:nch
        for c in (ch-1)*len+1:min(ch * len, nc)
            f((c - 1) ÷ ny + 1, (c - 1) % ny + 1)
        end
    end
    return nothing
end

"""f(i, j, s) for every column, s = setup() once per chunk (see `@cols`)."""
function foreach_col(setup::S, f::F, nx::Integer, ny::Integer) where {S,F}
    nc = nx * ny; nch, len = colchunks(nc)
    Threads.@threads :dynamic for ch in 1:nch
        sc = setup()
        for c in (ch-1)*len+1:min(ch * len, nc)
            f((c - 1) ÷ ny + 1, (c - 1) % ny + 1, sc)
        end
    end
    return nothing
end

const TCHUNK = 1 << 14                     # fixed chunk of the threaded element-wise kernels: results independent of threads

"""Threaded element-wise `out[I] = f(a[I], b[I], …)` over arrays of one shape (plain broadcasts run on one thread)."""
function tmap!(f::F, out::AbstractArray, ins::AbstractArray...) where {F}
    N = length(out); nch = cld(N, TCHUNK)
    Threads.@threads :dynamic for ch in 1:nch
        @inbounds @simd for I in (ch-1)*TCHUNK+1:min(ch * TCHUNK, N)
            out[I] = f(map(a -> a[I], ins)...)
        end
    end
    return out
end

"""Threaded Σ f(a[I], b[I], …) over arrays of one shape, summed in fixed chunks (the same value on any thread count)."""
function tsum(f::F, ins::AbstractArray...) where {F}
    N = length(ins[1]); nch = cld(N, TCHUNK); part = zeros(nch)
    Threads.@threads :dynamic for ch in 1:nch
        acc = 0.0
        @inbounds @simd for I in (ch-1)*TCHUNK+1:min(ch * TCHUNK, N)
            acc += f(map(a -> a[I], ins)...)
        end
        part[ch] = acc
    end
    return sum(part)
end

"""Threaded aᵀb (local), the same value on any thread count."""
tdot(a::AbstractArray, b::AbstractArray) = tsum(*, a, b)

"""Println on rank 0 only (flushed)."""
rprint(par::ParCtx, s) = (rank0(par) && (println(s); flush(stdout)); nothing)

# ---- reductions ----
gsum(par::ParCtx, x::Real) = par.nranks == 1 ? Float64(x) : MPI.Allreduce(Float64(x), +, par.comm)
gsum(par::ParCtx, x::AbstractVector{<:Real}) = par.nranks == 1 ? Float64.(x) : MPI.Allreduce(Float64.(x), +, par.comm)
gdot(par::ParCtx, a, b) = gsum(par, tdot(a, b))
gnorm(par::ParCtx, a) = sqrt(gsum(par, tdot(a, a)))
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
    keys = Vector{Vector{Int64}}(undef, ns)
    Threads.@threads :dynamic for s in 1:ns                  # one set per task: the sorts dominate on one rank
        keys[s] = sort!(Int64[okey(v) for v in sets[s] if isfinite(v)])
    end
    lo = [isempty(k) ? typemax(Int64) : k[1] for k in keys]; hi = [isempty(k) ? typemin(Int64) : k[end] for k in keys]
    if par.nranks > 1
        lo = MPI.Allreduce(lo, min, par.comm); hi = MPI.Allreduce(hi, max, par.comm)
    end
    for _ in 1:66
        all(lo .>= hi) && break
        mid = [lo[s] < hi[s] ? (lo[s] >> 1) + (hi[s] >> 1) + (lo[s] & hi[s] & 1) : lo[s] for s in 1:ns]  # floor mean, no overflow
        c = [searchsortedlast(keys[s], mid[s]) for s in 1:ns]     # serial: a task launch per round costs more
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
function gmedians(par::ParCtx, A::AbstractArray{<:Real,3})
    nk, ny, nx = size(A)
    sets = [Vector{Float64}(undef, ny * nx) for _ in 1:nk]
    @cols i j nx ny begin                                  # one threaded pass (a layer at a time would be strided)
        c = (i - 1) * ny + j
        @inbounds for k in 1:nk
            sets[k][c] = A[k, j, i]
        end
    end
    return gquantiles(par, sets, 0.5)
end

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
    @cols i j nx ny begin
        @inbounds for k in 1:nk
            fp = i == nx ? right[k, j] : f[k, j, i+1]; fm = i == 1 ? left[k, j] : f[k, j, i-1]
            o[k, j, i] = (fp - fm) / (2d)
        end
    end
    return o
end

function sddy!(o, f, d)
    nk, ny, nx = size(f)
    @cols i j nx ny begin
        jp = j == ny ? 1 : j + 1; jm = j == 1 ? ny : j - 1
        @inbounds for k in 1:nk
            o[k, j, i] = (f[k, jp, i] - f[k, jm, i]) / (2d)
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
    kr::Vector{UnitRange{Int}}       # row (vertical) chunks of the threaded transforms, FFT_ROWS each
    py::Any; ipy::Any; px::Any; ipx::Any   # plans per chunk length: Dict{Int, plan}
    sbuf::Vector{ComplexF64}; rbuf::Vector{ComplexF64}
    scount::Vector{Int}; rcount::Vector{Int}
end

"""Rows per threaded transform: the work items are (plane, row chunk), so a rank with few planes still uses its threads;
fixed (not from the thread count), so the transforms and their rounding are the same on any thread count."""
const FFT_ROWS = 4

function SlabFFT(par::ParCtx, nk::Int, ny::Int)
    nx = par.NX; nxl = par.block.nx; nyh = ny ÷ 2 + 1; nr = par.nranks
    br(N, c) = (s = HSEMHS_block_range(N, nr, c); s[1]:s[1]+s[2]-1)
    jr = [br(nyh, c) for c in 0:nr-1]; ir = [br(nx, c) for c in 0:nr-1]
    me = par.rank + 1
    nyhl = length(jr[me])
    fl = FFTW.ESTIMATE | FFTW.UNALIGNED
    sc = [nk * length(jr[s]) * nxl for s in 1:nr]; rc = [nk * nyhl * length(ir[s]) for s in 1:nr]
    kr = [k:min(k + FFT_ROWS - 1, nk) for k in 1:FFT_ROWS:nk]
    ls = unique(length.(kr))                                   # at most two chunk lengths; plans made here (FFTW
    py = Dict(m => plan_rfft(zeros(m, ny), 2; flags = fl) for m in ls)                     # planning is not threadsafe)
    ipy = Dict(m => plan_irfft(zeros(ComplexF64, m, nyh), ny, 2; flags = fl) for m in ls)
    px = Dict(m => plan_fft(zeros(ComplexF64, m, nx), 2; flags = fl) for m in ls)
    ipx = Dict(m => plan_ifft(zeros(ComplexF64, m, nx), 2; flags = fl) for m in ls)
    return SlabFFT(nk, ny, nyh, nx, nxl, jr, ir, zeros(ComplexF64, nk, nyh, nxl), zeros(ComplexF64, nk, nx, nyhl), kr,
                   py, ipy, px, ipx, zeros(ComplexF64, sum(sc)), zeros(ComplexF64, sum(rc)), sc, rc)
end
HSEMHS_block_range(N, p, c) = (b = divrem(N, p); n = b[1] + (c < b[2] ? 1 : 0); (c * b[1] + min(c, b[2]) + 1, n))

"""Global y wavenumber index (1-based) of local T column jl."""
kyglobal(S::SlabFFT, par::ParCtx, jl) = S.jr[par.rank+1][1] + jl - 1

"""
Threaded transform along dimension 2 of every (plane, row chunk) block of the `np` planes of `src` (3-D, planes along
dimension 3) into `dst`, with the plans `plans[chunk length]`; Ti, To the element types, nin, nout the input and output
lengths. The work items are split into chunks, each with its own scratch per chunk length (FFTW plans need the planned
layout), so the transforms neither allocate per item nor depend on the thread count.
"""
function rowffts!(dst, src, np::Integer, S::SlabFFT, plans, ::Type{Ti}, ::Type{To}, nin::Int, nout::Int) where {Ti,To}
    nc = length(S.kr); nw = np * nc
    nw == 0 && return nothing
    nch = max(1, min(nw, 8 * Threads.nthreads())); len = cld(nw, nch)
    ls = sort!(unique(length.(S.kr)))
    Threads.@threads :dynamic for ch in 1:nch
        bufs = Dict(m => (Matrix{Ti}(undef, m, nin), Matrix{To}(undef, m, nout)) for m in ls)
        for w in (ch-1)*len+1:min(ch * len, nw)
            p = (w - 1) ÷ nc + 1; kr = S.kr[(w-1)%nc+1]
            a, b = bufs[length(kr)]
            @inbounds for q in 1:nin, (r, k) in enumerate(kr)
                a[r, q] = src[k, q, p]
            end
            mul!(b, plans[length(kr)], a)
            @inbounds for q in 1:nout, (r, k) in enumerate(kr)
                dst[k, q, p] = b[r, q]
            end
        end
    end
    return nothing
end

function fwd!(S::SlabFFT, par::ParCtx, r::AbstractArray{Float64,3})
    rowffts!(S.A, r, S.nxl, S, S.py, Float64, ComplexF64, S.ny, S.nyh)              # y: r2c per (x plane, rows)
    transpose_fwd!(S, par)
    rowffts!(S.T, S.T, size(S.T, 3), S, S.px, ComplexF64, ComplexF64, S.nx, S.nx)   # x: c2c per (ky plane, rows)
    return S.T
end

function bwd!(S::SlabFFT, par::ParCtx, out::AbstractArray{Float64,3})
    rowffts!(S.T, S.T, size(S.T, 3), S, S.ipx, ComplexF64, ComplexF64, S.nx, S.nx)
    transpose_bwd!(S, par)
    rowffts!(out, S.A, S.nxl, S, S.ipy, ComplexF64, Float64, S.nyh, S.ny)          # c2r overwrites a (scratch)
    return out
end

function transpose_fwd!(S::SlabFFT, par::ParCtx)
    nr = par.nranks; me = par.rank + 1; nk = S.nk
    if nr == 1
        Threads.@threads for jl in 1:S.nyh
            @inbounds for i in 1:S.nx, k in 1:nk
                S.T[k, i, jl] = S.A[k, jl, i]
            end
        end
        return
    end
    sb = cumsum([0; S.scount]); nyhl = length(S.jr[me])
    Threads.@threads :dynamic for w in 1:nr*S.nxl                  # pack: block for rank s = A[:, jr[s], :], (k, j, i)
        s = (w - 1) ÷ S.nxl + 1; i = (w - 1) % S.nxl + 1; js = S.jr[s]
        off = sb[s] + (i - 1) * length(js) * nk
        @inbounds for j in js, k in 1:nk
            off += 1; S.sbuf[off] = S.A[k, j, i]
        end
    end
    MPI.Alltoallv!(MPI.VBuffer(S.sbuf, S.scount), MPI.VBuffer(S.rbuf, S.rcount), par.comm)
    rb = cumsum([0; S.rcount])
    Threads.@threads :dynamic for s in 1:nr                        # unpack: from rank s, (k, jl, il) → T[k, ir[s][il], jl]
        off = rb[s]
        @inbounds for il in 1:length(S.ir[s]), jl in 1:nyhl, k in 1:nk
            off += 1; S.T[k, S.ir[s][il], jl] = S.rbuf[off]
        end
    end
end

function transpose_bwd!(S::SlabFFT, par::ParCtx)
    nr = par.nranks; me = par.rank + 1; nk = S.nk
    if nr == 1
        Threads.@threads for i in 1:S.nx
            @inbounds for jl in 1:S.nyh, k in 1:nk
                S.A[k, jl, i] = S.T[k, i, jl]
            end
        end
        return
    end
    rb = cumsum([0; S.rcount]); nyhl = length(S.jr[me])
    Threads.@threads :dynamic for s in 1:nr
        off = rb[s]
        @inbounds for il in 1:length(S.ir[s]), jl in 1:nyhl, k in 1:nk
            off += 1; S.rbuf[off] = S.T[k, S.ir[s][il], jl]
        end
    end
    MPI.Alltoallv!(MPI.VBuffer(S.rbuf, S.rcount), MPI.VBuffer(S.sbuf, S.scount), par.comm)
    sb = cumsum([0; S.scount])
    Threads.@threads :dynamic for w in 1:nr*S.nxl
        s = (w - 1) ÷ S.nxl + 1; i = (w - 1) % S.nxl + 1; js = S.jr[s]
        off = sb[s] + (i - 1) * length(js) * nk
        @inbounds for j in js, k in 1:nk
            off += 1; S.A[k, j, i] = S.sbuf[off]
        end
    end
end

# ---- cooperative HDF5 output (parallel HDF5 optional: `io = :parallel`, H.hdf5_io) ----
"""
    coop_write(par, path, slabs, small, attrs; io=:cooperative)

Rank 0 creates `path` with a global dataset for every `slabs` entry (name => local array whose FIRST dimension is the
owned x range: Julia (x, y, …), h5py (…, y, x), the MHS checkpoint layout), writes `small` entries whole and the
attributes; then the ranks write their x hyperslabs: in turn (`io = :cooperative`) or all at once with collective
MPI-IO (`io = :parallel`). Both give the same file.
"""
function coop_write(par::ParCtx, path, slabs, small, attrs::Dict; io::Symbol = :cooperative)
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
    function owned!(f; pv...)
        for (nm, a) in slabs
            ds = open_dataset(f, nm; pv...)
            nd = ndims(a)
            nd == 1 ? (ds[ir] = a) : nd == 2 ? (ds[ir, :] = a) : (ds[ir, :, :] = a)
            close(ds)
        end
    end
    if io === :parallel
        MPI.Barrier(par.comm)
        h5open(f -> owned!(f; dxpl_mpio = :collective), path, "r+", par.comm)   # every rank, every dataset, same order
    else
        for r in 0:par.nranks-1
            par.nranks > 1 && MPI.Barrier(par.comm)
            par.rank == r || continue
            h5open(f -> owned!(f), path, "r+")
        end
    end
    par.nranks > 1 && MPI.Barrier(par.comm)
end

end # module Slab
