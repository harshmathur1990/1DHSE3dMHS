# Input reader (implementation doc §8.1).
#
# NetCDF4 files are HDF5 files. Julia's HDF5.jl sees C-order dims reversed; dimension
# names come from DIMENSION_LIST (C order). Each rank reads only its own hyperslab.

const UNIT_FACTORS = Dict(
    :length => Dict("m" => 1.0, "km" => 1e3, "cm" => 1e-2, "Mm" => 1e6),
    :temperature => Dict("K" => 1.0),
    :field => Dict("T" => 1.0, "G" => 1e-4, "gauss" => 1e-4),
    :pressure => Dict("Pa" => 1.0, "dyn/cm2" => 0.1, "dyn cm-2" => 0.1, "dyn / cm2" => 0.1),
    :density => Dict("kg / m3" => 1.0, "kg/m3" => 1.0, "kg m-3" => 1.0, "g/cm3" => 1e3, "g cm-3" => 1e3),
    :ndensity => Dict("1 / m3" => 1.0, "m-3" => 1.0, "1/m3" => 1.0, "cm-3" => 1e6, "1 / cm3" => 1e6),
    :velocity => Dict("m / s" => 1.0, "m/s" => 1.0, "cm/s" => 1e-2, "km/s" => 1e3),
)

function attr_string(x)
    x isa AbstractVector && length(x) == 1 && (x = x[1])
    x isa AbstractVector{UInt8} && return strip(String(copy(x)))
    return strip(string(x))
end

"""SI factor for dataset `ds` of physical `kind`; `invalid_input` if the unit is unknown."""
function unit_factor(ds, kind::Symbol, name::AbstractString)
    a = attributes(ds)
    haskey(a, "units") || invalid("dataset $name has no units attribute")
    u = attr_string(read(a["units"]))
    tab = UNIT_FACTORS[kind]
    haskey(tab, u) || invalid("dataset $name: unit \"$u\" is not an accepted $(kind) unit ($(join(keys(tab), ", ")))")
    return tab[u]
end

"""Dimension names of a dataset in Julia order (reverse of DIMENSION_LIST)."""
function julia_dim_names(f::HDF5.File, ds::HDF5.Dataset, name::AbstractString)
    a = attributes(ds)
    haskey(a, "DIMENSION_LIST") || invalid("dataset $name has no DIMENSION_LIST; cannot identify axes")
    refs = read(a["DIMENSION_LIST"])
    cnames = [split(HDF5.name(f[r[1]]), "/")[end] for r in refs]
    return reverse(cnames)
end

canonical_axis(n) = n == "x" ? :x : n == "y" ? :y : (n == "z" || n == "depth") ? :z : :other

"""Global attributes of an HDF5 file as a Dict (scalars unwrapped)."""
function global_attrs(f::HDF5.File)
    d = Dict{String,Any}()
    for k in keys(attributes(f))
        v = read(attributes(f)[k])
        d[k] = (v isa AbstractVector && length(v) == 1) ? v[1] : v
    end
    return d
end

"""Value of a FITS card such as PERIOD_X from a `fits_header` attribute, or `nothing`."""
function fits_card(hdr::AbstractString, key::AbstractString)
    m = match(Regex("$(rpad(key, 8))=\\s*'?([^/']*)'?"), hdr)
    return m === nothing ? nothing : strip(m.captures[1])
end

"""Description of an opened input: coordinates (cropped), crop range, periodicity, attrs."""
struct InputSpec
    path::String
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}           # cropped, top-first
    krange::UnitRange{Int}       # source z indices kept
    periodic::Tuple{Bool,Bool}
    period::Tuple{Float64,Float64}
    attrs::Dict{String,Any}
    tname::String                # name of the temperature dataset (for FITS header lookup)
end

"""
    open_input(path; crop_z_m=[], tname="temperature", xname="x", yname="y", zname="z")

Read and validate coordinates; decide periodicity from global attributes
(`periodic_x/_y`, `period_x_m/_y_m`) or from the FITS header cards PERIOD_X/Y.
"""
function open_input(path::AbstractString; crop_z_m = Float64[], tname = "temperature",
                    xname = "x", yname = "y", zname = "z")
    isfile(path) || invalid("input file $path does not exist")
    h5open(path, "r") do f
        for n in (xname, yname, zname, tname)
            haskey(f, n) || invalid("input file has no dataset \"$n\"")
        end
        x = read(f[xname]) .* unit_factor(f[xname], :length, xname)
        y = read(f[yname]) .* unit_factor(f[yname], :length, yname)
        zraw = read(f[zname])
        zf = unit_factor(f[zname], :length, zname)
        if ndims(zraw) == 1
            z = Float64.(zraw) .* zf
        else
            # Julia sees reversed dims; the z axis is the longest one
            zd = argmax(size(zraw))
            z0 = selectdim(zraw, 3 - zd, 1)
            for c in axes(zraw, 3 - zd)
                zc = selectdim(zraw, 3 - zd, c)
                maximum(abs.(zc .- z0) ./ max.(abs.(z0), 1.0)) <= 1e-6 ||
                    invalid("multidimensional $zname differs between columns; not a common height grid")
            end
            z = Float64.(collect(z0)) .* zf
        end
        x = Float64.(x); y = Float64.(y)
        all(diff(x) .> 0) || invalid("x is not strictly increasing")
        all(diff(y) .> 0) || invalid("y is not strictly increasing")
        all(diff(z) .< 0) || invalid("z is not strictly decreasing (top-first storage required)")
        ga = global_attrs(f)
        haskey(ga, "vertical_order") && attr_string(ga["vertical_order"]) != "top_first" &&
            invalid("vertical_order attribute is \"$(ga["vertical_order"])\", expected top_first")
        kr = 1:length(z)
        if !isempty(crop_z_m)
            length(crop_z_m) == 2 || invalid("crop_z_m must be [zmin, zmax]")
            keep = findall(zz -> crop_z_m[1] - 1.0 <= zz <= crop_z_m[2] + 1.0, z)
            isempty(keep) && invalid("crop_z_m $(crop_z_m) keeps no levels")
            kr = first(keep):last(keep)
        end
        # periodicity
        px = py = nothing
        if haskey(ga, "periodic_x") && haskey(ga, "periodic_y")
            px = Int(ga["periodic_x"]) == 1; py = Int(ga["periodic_y"]) == 1
        else
            ta = attributes(f[tname])
            if haskey(ta, "fits_header")
                hdr = attr_string(read(ta["fits_header"]))
                cx = fits_card(hdr, "PERIOD_X"); cy = fits_card(hdr, "PERIOD_Y")
                if cx !== nothing && cy !== nothing
                    px = parse(Int, cx) == 1; py = parse(Int, cy) == 1
                end
            end
        end
        (px === nothing || py === nothing) &&
            invalid("cannot determine lateral periodicity (no periodic_x/_y attributes or PERIOD_X/Y FITS cards)")
        perx = haskey(ga, "period_x_m") ? Float64(ga["period_x_m"]) : NaN
        pery = haskey(ga, "period_y_m") ? Float64(ga["period_y_m"]) : NaN
        return InputSpec(String(path), x, y, z[kr], kr, (px, py), (perx, pery), ga, String(tname))
    end
end

"""Build the global Grid from an InputSpec."""
function build_grid(spec::InputSpec)
    gx = build_axis("x", spec.x; periodic = spec.periodic[1], period = spec.period[1])
    gy = build_axis("y", spec.y; periodic = spec.periodic[2], period = spec.period[2])
    gz = build_axis("z", spec.z)
    return Grid(gx, gy, gz)
end

"""
    read_cube!(dest::Field, spec, par, name, kind)

Read this rank's owned hyperslab of dataset `name` (any dim order, optional length-1 time
dimension), convert to Float64 SI, store in the owned region of `dest`.
"""
function read_cube!(dest::Field, spec::InputSpec, par::ParCtx, name::AbstractString, kind::Symbol)
    b = par.block
    h5open(spec.path, "r") do f
        haskey(f, name) || invalid("input file has no dataset \"$name\"")
        ds = f[name]
        fac = unit_factor(ds, kind, name)
        dn = julia_dim_names(f, ds, name)
        sz = size(ds)
        idx = Any[]
        order = Symbol[]
        for (d, n) in enumerate(dn)
            a = canonical_axis(n)
            if a == :x
                sz[d] == length(spec.x) || invalid("$name: x dimension has $(sz[d]) points, expected $(length(spec.x))")
                push!(idx, b.i0:b.i0+b.nx-1); push!(order, :x)
            elseif a == :y
                sz[d] == length(spec.y) || invalid("$name: y dimension has $(sz[d]) points, expected $(length(spec.y))")
                push!(idx, b.j0:b.j0+b.ny-1); push!(order, :y)
            elseif a == :z
                push!(idx, spec.krange); push!(order, :z)
            else
                sz[d] == 1 || invalid("$name: extra dimension \"$n\" has length $(sz[d]) ≠ 1")
                push!(idx, 1)
            end
        end
        sort(order) == [:x, :y, :z] || invalid("$name: could not identify x, y, z dimensions from $(dn)")
        a = ds[idx...]
        perm = [findfirst(==(s), order) for s in (:x, :y, :z)]
        a = permutedims(a, perm)
        Threads.@threads :static for k in 1:b.nz
            @inbounds for j in 1:b.ny, i in 1:b.nx
                dest[i, j, k] = Float64(a[i, j, k]) * fac
            end
        end
    end
    return dest
end

"""Read a cube into a new field and check it is finite (and positive if `positive`)."""
function read_field(spec::InputSpec, par::ParCtx, name::AbstractString, kind::Symbol;
                    positive::Bool = false)
    f = newfield(par)
    read_cube!(f, spec, par, name, kind)
    b = par.block
    bad = 0
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        v = f[i, j, k]
        (isfinite(v) && (!positive || v > 0)) || (bad += 1)
    end
    nbad = global_count(par, bad)
    nbad == 0 || invalid("dataset $name has $nbad non-finite$(positive ? " or non-positive" : "") values")
    return f
end
