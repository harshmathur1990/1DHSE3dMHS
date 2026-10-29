# Shared helpers for the embedded-API tests (test_embedded.jl, mpi/mpi_embedded.jl).

const EFIX = joinpath(@__DIR__, "fixtures", "embedded")
const EREF = joinpath(@__DIR__, "fixtures", "embedded_ref")
const ETABLE = joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5")

"""Height-grid fixture: coordinates, P_ref, g and this slab's (z, y, x) ln T and B for global x range `ir`."""
function load_z_fixture(tag, ir)
    spec = H.open_input(joinpath(EFIX, "z_$tag.nc"))
    rd(nm) = h5open(spec.path, "r") do f
        a = Float64.(read(f[nm]))                                          # Julia (z, y, x)
        a[:, :, ir] .* H.unit_factor(f[nm], nm == "temperature" ? :temperature : :field, nm)
    end
    return (spec = spec, z = spec.z, x = spec.x, y = spec.y, P_ref = Float64(spec.attrs["P_ref_Pa"]),
            g = Float64(spec.attrs["gravity_m_s2"]), lnT = log.(rd("temperature")), Bx = rd("B_x"), By = rd("B_y"),
            Bz = rd("B_z"))
end

"""τ-grid fixture: axis, coordinates and this slab's (τ, y, x) ln T and B for global x range `ir`."""
function load_tau_fixture(tag, ir)
    h5open(joinpath(EFIX, "ltau_$tag.h5"), "r") do f
        rd(nm) = Float64.(f[nm][:, :, ir])                                  # Julia (τ, y, x)
        (lt = read(f["log10tau500"]), x = read(f["x"]), y = read(f["y"]), lnT = log.(rd("temperature")),
         Bx = rd("B_x"), By = rd("B_y"), Bz = rd("B_z"))
    end
end

"""
Standalone result file for (mode, tag): the stored one (test/fixtures/embedded_ref, made by scripts/run.jl with
configs/<mode>_<tag>.toml), or, for a Poisson mode whose stored file was made with another thread count (the iterative
solves depend on it through BLAS), the standalone driver run now in-process with the same config at this thread count.
"""
function ref_file(mode::Symbol, tag)
    n = "$(ref_name(mode))_$tag"; p = joinpath(EREF, "$n.h5")
    H.poisson_mode(mode) || return p
    th = h5open(f -> haskey(attributes(f), "threads") ? read(attributes(f)["threads"]) : Threads.nthreads(), p)   # no record: use it
    th == Threads.nthreads() && return p
    out = tempname() * ".h5"; cfg = tempname() * ".toml"
    write(cfg, replace(read(joinpath(EREF, "configs", "$n.toml"), String), r"output = \".*\"" => "output = \"$out\""))
    b0 = BLAS.get_num_threads()
    try
        cd(() -> H.run_config(cfg), joinpath(@__DIR__, ".."))
    finally
        BLAS.set_num_threads(b0)
    end
    @info "standalone reference made now (stored one: $th threads, this run: $(Threads.nthreads()))" n
    return out
end

"""Standalone result (`ref_file`) on global x range `ir`, as (vertical, y, x) arrays."""
function load_ref(mode, tag, ir; path = joinpath(EREF, "$(mode)_$tag.h5"))
    p = path
    h5open(p, "r") do f
        if mode == "column-hse-z"
            g(nm) = read(f["output/$nm"])[:, :, ir]
            return (Pgas = g("Pgas"), rho = g("rho"), ne = g("ne"), chi500 = g("chi500"), tau500 = g("tau500"),
                    z_tau1 = read(f["diagnostics/z_tau500_1"])[:, ir])
        elseif mode == "mhs-poisson-z"
            a(nm) = permutedims(read(f["atmosphere/$nm"])[ir, :, :], (3, 2, 1))
            return (Pgas = a("Pgas"), rho = a("rho"), tau500 = a("tau500"), Pgas_start = a("Pgas_column_hse"),
                    z_tau1 = permutedims(read(f["atmosphere/z_tau1_m"])[ir, :], (2, 1)))
        else
            t(nm) = permutedims(read(f["tau_grid/$nm"])[ir, :, :], (3, 2, 1))
            return (Pgas = t("Pgas"), rho = t("rho"), z = t("z_m"), z_tau1 = permutedims(read(f["tau_grid/z_tau1_m"])[ir, :], (2, 1)))
        end
    end
end

"""Embedded options matching the reference configs."""
function ref_options(mode::Symbol, fx)
    if mode in (:column_hse_z, :mhs_poisson_z)
        col = H.ColumnZOptions(anchor = :bottom, P_ref = fx.P_ref)
        poi = mode == :mhs_poisson_z ? H.PoissonZOptions(z_cut_m = 1.0e6, max_steps = 30, rtol = 1e-4) : nothing
        return (axis = fx.z, x = fx.x, y = fx.y, g = fx.g, column = col, poisson = poi)
    end
    col = H.ColumnTauOptions(align = "ptot")
    poi = mode == :mhs_poisson_ltau ? H.PoissonTauOptions(logtau_cut = -4.0, max_steps = 30, rtol = 1e-4, stop = :phi) : nothing
    return (axis = fx.lt, x = fx.x, y = fx.y, g = 274.0, column = col, poisson = poi)
end

load_fixture(mode::Symbol, tag, ir) = mode in (:column_hse_z, :mhs_poisson_z) ? load_z_fixture(tag, ir) : load_tau_fixture(tag, ir)
ref_name(mode::Symbol) = replace(String(mode), "_" => "-")

"""Max relative difference over entries finite in both (and a count of finite/non-finite mismatches)."""
function relerr(a, b)
    size(a) == size(b) || return Inf
    fa = isfinite.(a); fb = isfinite.(b)
    fa == fb || return Inf
    any(fa) || return 0.0
    return maximum(abs.(a[fa] .- b[fa]) ./ max.(abs.(a[fa]), 1e-300))
end

"""Max relative difference between an embedded result and the stored standalone result."""
function ref_error(res, ref)
    e = 0.0
    for nm in propertynames(ref)
        nm == :ne && !hasproperty(res, :ne) && continue
        e = max(e, relerr(getproperty(res, nm), getproperty(ref, nm)))
    end
    return e
end
