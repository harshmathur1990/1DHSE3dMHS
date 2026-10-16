# MPI + threads infrastructure (implementation doc §5).
#
# Rules: all MPI calls happen on the main task, outside `Threads.@threads` regions.
# The x and y axes are periodic in v1; z is never decomposed.

struct Block
    nx::Int
    ny::Int
    nz::Int
    i0::Int          # global index of owned local i = 1
    j0::Int          # global index of owned local j = 1
end

mutable struct ParCtx
    comm::MPI.Comm
    rank::Int
    nranks::Int
    px::Int
    py::Int
    cx::Int          # 0-based process coordinates
    cy::Int
    nbr_xm::Int
    nbr_xp::Int
    nbr_ym::Int
    nbr_yp::Int
    block::Block
    NX::Int
    NY::Int
    NZ::Int
    nthreads::Int
    debug::Bool
    bufcache::Dict{Tuple{Symbol,Int},Vector{Float64}}
end

"""Owned range of process coordinate `c` (0-based) when `N` points are split over `p`."""
function block_range(N::Int, p::Int, c::Int)
    base, rem = divrem(N, p)
    n = base + (c < rem ? 1 : 0)
    i0 = c * base + min(c, rem) + 1
    return i0, n
end

function ensure_mpi()
    if !MPI.Initialized()
        provided = MPI.Init(threadlevel = :funneled)
        provided < MPI.THREAD_FUNNELED && error("MPI does not provide THREAD_FUNNELED (got $provided)")
    end
    return nothing
end

"""
    init_parallel(NX, NY, NZ; px=0, py=0, debug=false) -> ParCtx

Create the periodic 2D Cartesian communicator and this rank's block. `px = py = 0` lets
`MPI.Dims_create` choose. Rejects (collectively) any owned size below 2.
"""
function init_parallel(NX::Int, NY::Int, NZ::Int; px::Int = 0, py::Int = 0, debug::Bool = false)
    ensure_mpi()
    world = MPI.COMM_WORLD
    nr = MPI.Comm_size(world)
    dims = Int.(MPI.Dims_create(nr, [px, py]))
    prod(dims) == nr || invalid("process grid $(dims) does not match $nr ranks")
    comm = MPI.Cart_create(world, dims; periodic = [true, true], reorder = false)
    rank = MPI.Comm_rank(comm)
    coords = MPI.Cart_coords(comm)
    cx, cy = Int(coords[1]), Int(coords[2])
    xm, xp = MPI.Cart_shift(comm, 0, 1)
    ym, yp = MPI.Cart_shift(comm, 1, 1)
    i0, nx = block_range(NX, dims[1], cx)
    j0, ny = block_range(NY, dims[2], cy)
    ok = (nx >= 2 && ny >= 2) ? 1 : 0
    if MPI.Allreduce(ok, min, comm) == 0
        invalid("process grid $(dims[1])×$(dims[2]) leaves fewer than 2 owned columns per axis on some rank for a $(NX)×$(NY) grid")
    end
    NZ >= 3 || invalid("NZ = $NZ; at least 3 vertical levels are required")
    return ParCtx(comm, rank, nr, dims[1], dims[2], cx, cy, xm, xp, ym, yp,
                  Block(nx, ny, NZ, i0, j0), NX, NY, NZ, Threads.nthreads(), debug,
                  Dict{Tuple{Symbol,Int},Vector{Float64}}())
end

assert_main(par::ParCtx) = (par.debug && !MPI.Is_thread_main()) && error("MPI call off the main thread")

function getbuf(par::ParCtx, tag::Symbol, n::Int)
    key = (tag, n)
    get!(() -> Vector{Float64}(undef, n), par.bufcache, key)
end

# ---------------------------------------------------------------------------------------
# Halo exchange (§5.3). Corners are never needed (all stencils are axis-aligned).
# ---------------------------------------------------------------------------------------

"""
    exchange_halos!(par, fields)

Fill the one-cell x and y halos of every field in `fields` (OffsetArrays indexed
0:nx+1, 0:ny+1 in x, y and any range in z, e.g. 1:NZ for centres or 0:NZ for z-staggered
arrays) with the periodically wrapped neighbour values.
"""
function exchange_halos!(par::ParCtx, fields::AbstractVector)
    isempty(fields) && return nothing
    assert_main(par)
    b = par.block
    nx, ny = b.nx, b.ny
    krs = [axes(f, 3) for f in fields]
    # ---- x direction: faces i = 1 and i = nx, j in 1:ny ----
    if par.nbr_xm == par.rank && par.nbr_xp == par.rank
        for (f, kr) in zip(fields, krs)
            Threads.@threads :static for k in collect(kr)
                @inbounds for j in 1:ny
                    f[0, j, k] = f[nx, j, k]
                    f[nx+1, j, k] = f[1, j, k]
                end
            end
        end
    else
        offs = cumsum([0; [ny * length(kr) for kr in krs]])
        n = offs[end]
        sL = getbuf(par, :sxl, n); sR = getbuf(par, :sxr, n)
        rL = getbuf(par, :rxl, n); rR = getbuf(par, :rxr, n)
        for (q, f) in enumerate(fields)
            off = offs[q]; k0 = first(krs[q])
            Threads.@threads :static for k in collect(krs[q])
                @inbounds for j in 1:ny
                    p = off + (k - k0) * ny + j
                    sL[p] = f[1, j, k]
                    sR[p] = f[nx, j, k]
                end
            end
        end
        reqs = MPI.Request[
            MPI.Irecv!(rL, par.comm; source = par.nbr_xm, tag = 1),
            MPI.Irecv!(rR, par.comm; source = par.nbr_xp, tag = 2),
            MPI.Isend(sR, par.comm; dest = par.nbr_xp, tag = 1),
            MPI.Isend(sL, par.comm; dest = par.nbr_xm, tag = 2)]
        MPI.Waitall(reqs)
        for (q, f) in enumerate(fields)
            off = offs[q]; k0 = first(krs[q])
            Threads.@threads :static for k in collect(krs[q])
                @inbounds for j in 1:ny
                    p = off + (k - k0) * ny + j
                    f[0, j, k] = rL[p]
                    f[nx+1, j, k] = rR[p]
                end
            end
        end
    end
    # ---- y direction: faces j = 1 and j = ny, i in 0:nx+1 (after x: carries the corners) ----
    if par.nbr_ym == par.rank && par.nbr_yp == par.rank
        for (f, kr) in zip(fields, krs)
            Threads.@threads :static for k in collect(kr)
                @inbounds for i in 0:nx+1
                    f[i, 0, k] = f[i, ny, k]
                    f[i, ny+1, k] = f[i, 1, k]
                end
            end
        end
    else
        offs = cumsum([0; [(nx + 2) * length(kr) for kr in krs]])
        n = offs[end]
        sL = getbuf(par, :syl, n); sR = getbuf(par, :syr, n)
        rL = getbuf(par, :ryl, n); rR = getbuf(par, :ryr, n)
        for (q, f) in enumerate(fields)
            off = offs[q]; k0 = first(krs[q])
            Threads.@threads :static for k in collect(krs[q])
                @inbounds for i in 0:nx+1
                    p = off + (k - k0) * (nx + 2) + i + 1
                    sL[p] = f[i, 1, k]
                    sR[p] = f[i, ny, k]
                end
            end
        end
        reqs = MPI.Request[
            MPI.Irecv!(rL, par.comm; source = par.nbr_ym, tag = 3),
            MPI.Irecv!(rR, par.comm; source = par.nbr_yp, tag = 4),
            MPI.Isend(sR, par.comm; dest = par.nbr_yp, tag = 3),
            MPI.Isend(sL, par.comm; dest = par.nbr_ym, tag = 4)]
        MPI.Waitall(reqs)
        for (q, f) in enumerate(fields)
            off = offs[q]; k0 = first(krs[q])
            Threads.@threads :static for k in collect(krs[q])
                @inbounds for i in 0:nx+1
                    p = off + (k - k0) * (nx + 2) + i + 1
                    f[i, 0, k] = rL[p]
                    f[i, ny+1, k] = rR[p]
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Deterministic reductions (§5.5)
# ---------------------------------------------------------------------------------------

"""Row-partial buffer: one slot per (term, j, k) row of the local block."""
rowpartials(par::ParCtx, nterms::Int) = zeros(nterms, par.block.ny, par.block.nz)

"""
    reduce_rows(par, p) -> Vector{Float64}

Sum a (nterms, ny, nz) row-partial array in fixed k-major, j-minor order on the main task,
then Allreduce over ranks. Independent of the thread count.
"""
function reduce_rows(par::ParCtx, p::Array{Float64,3})
    nt = size(p, 1)
    s = zeros(nt)
    @inbounds for k in axes(p, 3), j in axes(p, 2), t in 1:nt
        s[t] += p[t, j, k]
    end
    assert_main(par)
    return MPI.Allreduce(s, +, par.comm)
end

"""Deterministic global sum of `f` over owned nodes (threads over rows, fixed order)."""
function global_sum(par::ParCtx, f::AbstractArray{<:Real,3})
    b = par.block
    p = rowpartials(par, 1)
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        s = 0.0
        @inbounds for i in 1:b.nx
            s += f[i, j, k]
        end
        p[1, j, k] = s
    end
    return reduce_rows(par, p)[1]
end

"""Deterministic global dot product over owned nodes."""
function global_dot(par::ParCtx, a::AbstractArray{<:Real,3}, c::AbstractArray{<:Real,3})
    b = par.block
    p = rowpartials(par, 1)
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        s = 0.0
        @inbounds for i in 1:b.nx
            s += a[i, j, k] * c[i, j, k]
        end
        p[1, j, k] = s
    end
    return reduce_rows(par, p)[1]
end

global_max(par::ParCtx, x::Real) = MPI.Allreduce(Float64(x), max, par.comm)
global_min(par::ParCtx, x::Real) = MPI.Allreduce(Float64(x), min, par.comm)
global_count(par::ParCtx, n::Integer) = MPI.Allreduce(Int(n), +, par.comm)

# ---------------------------------------------------------------------------------------
# Collective error propagation (§5.7)
# ---------------------------------------------------------------------------------------

"""
    collective_status(par, local_status, msg="") -> (status, messages)

Reduce the status with `max` (severity order) and gather non-empty messages to all ranks.
"""
function collective_status(par::ParCtx, st::SolverStatus, msg::AbstractString = "")
    assert_main(par)
    g = SolverStatus(MPI.Allreduce(UInt8(st), max, par.comm))
    bytes = Vector{UInt8}(codeunits(String(msg)))
    counts = MPI.Allgather(Int32(length(bytes)), par.comm)
    all = Vector{UInt8}(undef, sum(counts))
    MPI.Allgatherv!(bytes, MPI.VBuffer(all, counts), par.comm)
    msgs = String[]
    off = 0
    for c in counts
        c > 0 && push!(msgs, String(all[off+1:off+c]))
        off += c
    end
    return g, msgs
end

"""
    run_collective(f, par) -> (status, messages, value)

Run `f()` on every rank; an `InvalidInput` becomes `invalid_input`, any other exception
`eos_failure`-free generic failure is reported as `invalid_input` with the message. All
ranks return the same status, so nobody deadlocks in a later collective.
"""
function run_collective(f, par::ParCtx)
    st = converged
    msg = ""
    val = nothing
    try
        val = f()
    catch e
        st = invalid_input
        msg = "rank $(par.rank): " * sprint(showerror, e)
    end
    g, msgs = collective_status(par, st, msg)
    return g, msgs, val
end

# ---------------------------------------------------------------------------------------
# Token ring for serial-HDF5 writes (§11.1)
# ---------------------------------------------------------------------------------------

"""Run `f()` on each rank in rank order (rank 0 first), then barrier."""
function token_ring(f, par::ParCtx)
    assert_main(par)
    tok = Int32[0]
    if par.rank > 0
        MPI.Recv!(tok, par.comm; source = par.rank - 1, tag = 77)
    end
    f()
    if par.rank < par.nranks - 1
        MPI.Send(tok, par.comm; dest = par.rank + 1, tag = 77)
    end
    MPI.Barrier(par.comm)
    return nothing
end

rank0(par::ParCtx) = par.rank == 0
