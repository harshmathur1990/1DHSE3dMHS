# Output writer (implementation doc §11).
#
# No parallel HDF5 needed: rank 0 creates every dataset at its global shape, then ranks
# write their owned hyperslabs in turn (token ring). With output.parallel_hdf5 on a parallel
# file system (`hdf5_io`) the ranks write the same hyperslabs at once with collective MPI-IO
# instead; the file layout is the same. On disk, 3D datasets follow the C convention (x, y, z),
# which Julia sees as (z, y, x).

"""A distributed 3D dataset to write: owned region of `data` goes to `path` in the file."""
struct FieldOut
    path::String                      # e.g. "/output/Pgas"
    data::OffsetArray{<:Real,3}
    units::String
    long_name::String
end

"""A distributed per-column (x, y) dataset."""
struct ColumnOut
    path::String
    data::AbstractMatrix              # (nx, ny) owned
    units::String
    long_name::String
end

function set_attrs!(obj, d::AbstractDict)
    for (k, v) in d
        write_attribute(obj, String(k), meta_value(v))
    end
end

meta_value(v::Symbol) = String(v)
meta_value(v::Nothing) = "none"
meta_value(v::AbstractVector{<:AbstractString}) = String.(v)
meta_value(v::Bool) = Int8(v)
meta_value(v) = v

"""Create a new output file with the standard groups (rank 0), honouring `overwrite`."""
function create_output(par::ParCtx, path::AbstractString; overwrite::Bool = false)
    st, msgs, _ = run_collective(par) do
        if rank0(par)
            if isfile(path)
                overwrite || invalid("output $path exists and run.overwrite = false")
                rm(path)
            end
            mkpath(dirname(abspath(path)))
            h5open(path, "w") do f
                for g in ("input", "output", "diagnostics", "meta")
                    create_group(f, g)
                end
                write_attribute(f, "vertical_order", "top_first")
                write_attribute(f, "array_order", "(x, y, z) C order; [:, :, 0] is the top; Julia HDF5.jl sees (z, y, x)")
            end
        end
    end
    st == converged || invalid(join(msgs, "; "))
    return nothing
end

"""
    fs_type(path) -> String

File-system type of the directory holding `path` (its nearest existing ancestor), from the longest matching mount
point in /proc/self/mounts; "" when unknown (not Linux).
"""
function fs_type(path::AbstractString)
    (Sys.islinux() && isfile("/proc/self/mounts")) || return ""
    d = dirname(abspath(path))
    while !isdir(d) && dirname(d) != d
        d = dirname(d)
    end
    p = realpath(d)
    best, typ = "", ""
    for ln in eachline("/proc/self/mounts")
        f = split(ln)
        length(f) >= 3 || continue
        mp = replace(f[2], "\\040" => " ")
        inside = mp == "/" || p == mp || startswith(p, mp * "/")
        inside && length(mp) > length(best) && ((best, typ) = (mp, String(f[3])))
    end
    return typ
end

"""
    hdf5_io(par, output, path) -> (mode, reason)

`:parallel` when `output.parallel_hdf5` is set, the run has more than one rank, the HDF5 library has MPI-IO
(`HDF5.has_parallel()`) and `path` is on a file system in `output.parallel_fs`; `:cooperative` otherwise, with the
reason. Decided on rank 0 and broadcast, so all ranks take the same path.
"""
function hdf5_io(par::ParCtx, output, path::AbstractString)
    why = ""
    if rank0(par)
        fs = fs_type(path)
        why = !output.parallel_hdf5 ? "output.parallel_hdf5 = false" :
              par.nranks == 1 ? "1 rank" :
              !HDF5.has_parallel() ? "the HDF5 library has no MPI-IO" :
              fs in output.parallel_fs ? "parallel: file system $fs" :
              "file system $(isempty(fs) ? "unknown" : fs) is not in output.parallel_fs"
    end
    why = MPI.bcast(why, par.comm; root = 0)
    return (startswith(why, "parallel: ") ? :parallel : :cooperative), why
end

"""Log line for the output path chosen by `hdf5_io`."""
hdf5_io_line(io, why) = io === :parallel ? "output: parallel HDF5 (collective MPI-IO, $(why[11:end]))" :
                                           "output: cooperative HDF5 ($why)"

function ensure_parent(f, path)
    parts = split(strip(path, '/'), '/')
    g = f
    for p in parts[1:end-1]
        g = haskey(g, p) ? g[p] : create_group(g, p)
    end
    return g, parts[end]
end

"""
    write_fields(par, file, fields; cols=ColumnOut[], compression=4, io=:cooperative)

Create all datasets on rank 0, then write every rank's owned part via the token ring (`io = :cooperative`) or all at
once with collective MPI-IO (`io = :parallel`, from `hdf5_io`).
"""
function write_fields(par::ParCtx, file::AbstractString, fields::Vector{FieldOut};
                      cols::Vector{ColumnOut} = ColumnOut[], compression::Int = 4, io::Symbol = :cooperative)
    NX, NY, NZ = par.NX, par.NY, par.NZ
    b = par.block
    if rank0(par)
        h5open(file, "r+") do f
            for fo in fields
                g, nm = ensure_parent(f, fo.path)
                T = eltype(fo.data)
                ds = create_dataset(g, nm, datatype(T), dataspace((NZ, NY, NX));
                                    chunk = (min(16, NZ), min(64, NY), min(64, NX)),
                                    shuffle = true, deflate = compression)
                set_attrs!(ds, Dict("units" => fo.units, "long_name" => fo.long_name,
                                    "dims" => ["x", "y", "z"]))
            end
            for co in cols
                g, nm = ensure_parent(f, co.path)
                ds = create_dataset(g, nm, datatype(eltype(co.data)), dataspace((NY, NX));
                                    chunk = (min(64, NY), min(64, NX)), deflate = compression)
                set_attrs!(ds, Dict("units" => co.units, "long_name" => co.long_name,
                                    "dims" => ["x", "y"]))
            end
        end
    end
    MPI.Barrier(par.comm)
    jr = b.j0:b.j0+b.ny-1
    ir = b.i0:b.i0+b.nx-1
    function owned!(f; pv...)
        for fo in fields
            ds = open_dataset(f, fo.path; pv...)
            ds[1:NZ, jr, ir] = permutedims(collect(fo.data[1:b.nx, 1:b.ny, 1:b.nz]), (3, 2, 1))
            close(ds)
        end
        for co in cols
            ds = open_dataset(f, co.path; pv...)
            ds[jr, ir] = permutedims(collect(co.data), (2, 1))
            close(ds)
        end
    end
    if io === :parallel
        h5open(f -> owned!(f; dxpl_mpio = :collective), file, "r+", par.comm)   # every rank, every dataset, same order
        MPI.Barrier(par.comm)
    else
        token_ring(() -> h5open(f -> owned!(f), file, "r+"), par)
    end
    return nothing
end

"""Write a per-layer table (rows = levels) on rank 0; Python sees shape (NZ, ncol)."""
function write_layer_table(par::ParCtx, file::AbstractString, path::AbstractString,
                           M::AbstractMatrix, columns::Vector{String}; units::String = "1",
                           long_name::String = "")
    if rank0(par)
        h5open(file, "r+") do f
            g, nm = ensure_parent(f, path)
            g[nm] = permutedims(Matrix{Float64}(M), (2, 1))
            set_attrs!(g[nm], Dict("columns" => columns, "units" => units,
                                   "long_name" => long_name, "dims" => ["z", "column"]))
        end
    end
    MPI.Barrier(par.comm)
end

"""Write nested metadata (Dicts become groups, everything else attributes) on rank 0."""
function write_meta(par::ParCtx, file::AbstractString, meta::AbstractDict; group = "/meta")
    if rank0(par)
        h5open(file, "r+") do f
            g, nm = ensure_parent(f, group * "/x")   # make sure the group exists
            _write_meta(g, meta)
        end
    end
    MPI.Barrier(par.comm)
end

function _write_meta(g, meta::AbstractDict)
    for (k, v) in meta
        k = String(k)
        if v isa AbstractDict
            sub = haskey(g, k) ? g[k] : create_group(g, k)
            _write_meta(sub, v)
        elseif v isa AbstractMatrix
            g[k] = permutedims(Matrix(v), (2, 1))
        elseif v isa AbstractVector && !(eltype(v) <: AbstractString) && length(v) > 64
            g[k] = collect(v)
        else
            write_attribute(g, k, meta_value(v))
        end
    end
end

"""Copy input datasets untouched (original dtype and attributes) into /input (rank 0)."""
function copy_inputs(par::ParCtx, file::AbstractString, src::AbstractString, names)
    if rank0(par)
        h5open(src, "r") do s
            h5open(file, "r+") do f
                for n in names
                    haskey(s, n) || continue
                    HDF5.copy_object(s[n], f["input"], n)
                end
            end
        end
    end
    MPI.Barrier(par.comm)
end

"""Provenance metadata common to every output."""
function provenance_meta(par::ParCtx, cfg::Config)
    git = try
        readchomp(pipeline(`git rev-parse HEAD`; stderr = devnull))
    catch
        "unavailable"
    end
    return Dict{String,Any}(
        "julia_version" => string(VERSION),
        "nthreads" => Threads.nthreads(),
        "mpi" => Dict{String,Any}("nranks" => par.nranks, "px" => par.px, "py" => par.py,
                                  "library" => string(MPI.identify_implementation())),
        "git_hash" => git,
        "config_toml" => cfg.toml_text,
        "config_sha256" => config_hash(cfg),
        "created" => string(Dates.now()),
        "constants" => constants_meta(),
        "units_system" => "SI",
        "vertical_order" => "top_first")
end
