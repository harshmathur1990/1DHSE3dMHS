# TOML configuration (implementation doc §16.2).
#
# Every section and key is declared in CONFIG_SCHEMA with its default. `REQUIRED` marks
# keys without a default; whether they are required depends on run.mode (see
# `required_keys`). Unknown sections or keys are rejected, which catches typos.

const REQUIRED = :REQUIRED

const CONFIG_SCHEMA = [
    "run" => [
        "mode" => REQUIRED,          # column-hse-z | column-hse-ltau | mhs-poisson-z | mhs-poisson-ltau | full-mhs-z (+ gate0, eos_table, screening; column_hse = column-hse-z)
        "output" => REQUIRED,
        "overwrite" => false,
        "debug" => false,
    ],
    "input" => [
        "file" => REQUIRED,
        "temperature" => "temperature",
        "bx" => "B_x", "by" => "B_y", "bz" => "B_z",
        "x" => "x", "y" => "y", "z" => "z",
        "sigma_x" => "", "sigma_y" => "", "sigma_z" => "",
        "crop_z_m" => Float64[],
        "resample" => "none",
        "resample_dz_m" => 0.0,
    ],
    "physics" => [
        "g" => "from_input",
    ],
    "eos" => [
        "lib" => "deps/witt/libwitt",
        "pf_file" => "../wittman-eos/pf_Kurucz.input",
        "table_file" => "cache/eos_table_fullz.h5",
        "build_if_missing" => false,
        "prec" => 1e-10,
        "maxit" => 5000,
        "abundances" => "wittmann_default",
        "dlog10T" => 0.002,
        "dlog10P" => 0.02,
        "logP_range" => Float64[],
        "envelope_from" => "reports/gate0_bifrost385.h5",
        "validate_points" => 1000000,
        "refine_max_levels" => 5,
    ],
    "column_hse" => [
        "anchor" => "bottom",
        "P_ref" => REQUIRED,
        "z_ref_m" => 0.0,
        "n_sub" => 4,
        "adapt" => true,
        "rk_tol" => 1e-6,
        "n_sub_max" => 64,
    ],
    "tau" => [
        "top" => "exp_tail",
        "H_max_m" => 5.0e6,
        "mu" => 1.0,
    ],
    "mhs" => [
        "anchor_plane" => "bottom",
        "P_ref" => REQUIRED,
        "data_kind" => "real",
        "sigma_B" => "synthetic",
        "sigma_rel" => 0.05,
        "sigma_floor_T" => 5.0e-4,
        "lambda_D" => 1.0,
        "lambda_S" => 1.0e-3,
        "lambda_U" => 1.0e-6,
        "lambda_Uz" => 0.0,            # vertical grid-scale term in Φ_U (staggered; D9b §2.3; M6d: off by default)
        "lambda_Q" => 1.0,             # force-weighted current-Nyquist term Φ_Q (staggered; D9b §2.2)
        "data_on_faces" => true,       # staggered data term on faces (D9b §2.1); B_out stays HB
        "collapse_du" => 3.0,          # L-BFGS pressure-collapse safeguard: u ≥ u_start − collapse_du
        "xtol_rel" => 1.0e-3,          # converged when P and B change by ≤ this (relative) over xtol_window iterations (M5)
        "xtol_window" => 100,
        "lorentz_bfirst" => 1.0,       # B-first continuation when some layer's median |L|/S exceeds this (M5)
        "b_first_iter" => 2000,        # iterations of the v-frozen stage
        "lambda_F" => "discrepancy",
        "lambda_F_grid" => [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4],
        "bisect_ratio" => 1.5,
        "lbfgs_m" => 10,
        "max_iter" => 5000,
        "gtol_rel" => 1.0e-6,
        "ftol_rel" => 1.0e-10,
        "ftol_window" => 20,
        "ls_c1" => 1.0e-4,
        "ls_c2" => 0.9,
        "ls_max_evals" => 20,
        "tol_rms" => "auto",
        "tol_p99" => "auto",
        "init" => "column_hse",
        "checkpoint_every" => 200,
        "restart_from" => "",
    ],
    "poisson" => [                   # mhs-poisson-z and mhs-poisson-ltau
        "z_cut_m" => 1.0e6,            # z: solved region z ≤ z_cut, column HSE above
        "logtau_cut" => -4.0,          # ltau: solved levels log10 τ ≥ logtau_cut, τ-hydrostatic above
        "max_steps" => 100,            # Gauss–Newton steps
        "rtol" => 1.0e-4,              # stop when ΔΦ/Φ < rtol on n_consec consecutive steps
        "n_consec" => 3,
        "cg_tol" => 1.0e-3,
        "cg_maxit" => 80,
        "anchor_w" => 1.0,             # z: weight of the mean-ln P anchor on the bottom plane
        "dc_cap_m" => 3.0e5,           # ltau: largest change of a column's height per step (whole-step scaling)
        "lm_mu0" => 0.0,               # ltau: initial Levenberg–Marquardt damping
    ],
    "tau_grid" => [                  # column-hse-ltau and mhs-poisson-ltau
        "cap_per_dex" => 50.0,         # largest rise of ln P per dex in τ (hot-over-cool runaway tail)
        "align" => "ptot",             # ptot: P + B²/2μ₀ equal at one height; none: z = 0 at log10 τ = 0 per column
    ],
    "full_mhs" => [                  # full-mhs-z (the M5 joint GN driver, scripts/m5_bifrost/m5h_fullres.jl); M5i defaults
        "run" => "run",                # run | smoke
        "wall_hours" => 24.0,
        "ckpt_hours" => 1.0,
        "init_iter" => 3000,
        "restart" => "",
        "sigma_rel" => 0.0,            # σ_B = sigma_rel |B| ⊕ 5 G (M5i: 0, uniform 5 G)
        "trust_region" => false,       # M5i: GN + backtracking line search
        "stop" => "phi",               # phi (|ΔΦ|/Φ < phi_rtol on 3 consecutive accepted) | dz (z(τ=1) change)
        "phi_rtol" => 1.0e-6,
    ],
    "screening" => [
        "colhse_file" => "",          # column_hse output providing Pgas and rho
        "ell_factor" => 2.0,
        "write_maps" => false,
    ],
    "gate0" => [
        "crop_z_m" => [-0.5e6, 3.0e6],
        "stride" => 4,
        "target_heights_m" => [3.0e6, 2.0e6, 1.5e6, 1.0e6, 0.5e6, 0.0, -0.5e6],
        "P_top" => 0.1,
        "sigma_rel" => 0.05,
        "sigma_floor_T" => 5.0e-4,
        "ell_factor" => 2.0,
        "pressure" => "gas_pressure",
        "density" => "density",
        "vx" => "velocity_x", "vy" => "velocity_y", "vz" => "velocity_z",
    ],
    "parallel" => [
        "px" => 0,
        "py" => 0,
    ],
    "output" => [
        "parallel_hdf5" => false,
        "compression" => 4,
    ],
]

const MODES = ("column-hse-z", "column-hse-ltau", "mhs-poisson-z", "mhs-poisson-ltau", "full-mhs-z",
               "column_hse", "mhs", "gate0", "eos_table", "screening")

"""Parsed, validated configuration. Sections are NamedTuples: `cfg.parallel.px`."""
struct Config
    sections::Dict{String,NamedTuple}
    toml_text::String
    path::String
end

Base.getproperty(c::Config, s::Symbol) =
    s in (:sections, :toml_text, :path) ? getfield(c, s) : getfield(c, :sections)[String(s)]

function required_keys(mode::String)
    req = Set(["run.mode", "run.output", "input.file"])
    mode in ("column_hse", "column-hse-z", "mhs-poisson-z") && push!(req, "column_hse.P_ref")
    mode == "mhs" && push!(req, "mhs.P_ref")
    return req
end

# A default that is a Vector{Float64} accepts any numeric TOML array.
function coerce(default, value, key)
    if default isa Vector{Float64}
        value isa AbstractVector || invalid("config key $key must be an array of numbers")
        all(v -> v isa Real, value) || invalid("config key $key must be an array of numbers")
        return Float64.(value)
    elseif default isa Float64
        value isa Real && return Float64(value)
        value isa String && return value          # e.g. "from_input"
        invalid("config key $key must be a number")
    elseif default isa Bool
        value isa Bool || invalid("config key $key must be true or false")
        return value
    elseif default isa Int
        value isa Integer || invalid("config key $key must be an integer")
        return Int(value)
    elseif default isa String
        (value isa String || value isa Real) || invalid("config key $key must be a string or number")
        return value
    end
    return value
end

"""
    parse_config(text; path="") -> Config

Parse TOML text, fill defaults, reject unknown keys and check mode-dependent required keys.
"""
function parse_config(text::AbstractString; path::AbstractString = "")
    raw = try
        TOML.parse(text)
    catch e
        invalid("cannot parse TOML $(path): $(sprint(showerror, e))")
    end
    known = Dict(first(p) => Dict(last(p)) for p in CONFIG_SCHEMA)
    for (sec, v) in raw
        haskey(known, sec) || invalid("unknown config section [$sec]")
        v isa AbstractDict || invalid("config entry $sec must be a section")
        for k in keys(v)
            haskey(known[sec], k) || invalid("unknown config key $sec.$k")
        end
    end
    mode = get(get(raw, "run", Dict()), "mode", nothing)
    mode isa String || invalid("config key run.mode is required")
    mode in MODES || invalid("run.mode must be one of $(join(MODES, ", ")); got \"$mode\"")
    req = required_keys(mode)
    sections = Dict{String,NamedTuple}()
    for (sec, entries) in CONFIG_SCHEMA
        given = get(raw, sec, Dict{String,Any}())
        names = Symbol[]
        vals = Any[]
        for (k, default) in entries
            full = "$sec.$k"
            if haskey(given, k)
                val = default === REQUIRED ? given[k] : coerce(default, given[k], full)
            elseif default === REQUIRED
                full in req && invalid("config key $full is required for run.mode = \"$mode\"")
                val = nothing
            else
                val = default isa Vector ? copy(default) : default
            end
            push!(names, Symbol(k))
            push!(vals, val)
        end
        sections[sec] = NamedTuple{Tuple(names)}(Tuple(vals))
    end
    return Config(sections, String(text), String(path))
end

load_config(path::AbstractString) = parse_config(read(path, String); path = path)

"""SHA-256 of the canonical parsed config, excluding keys irrelevant to restart identity."""
function config_hash(cfg::Config)
    excl = Set(["run.output", "run.overwrite", "mhs.restart_from"])
    io = IOBuffer()
    for (sec, entries) in CONFIG_SCHEMA
        nt = cfg.sections[sec]
        for (k, _) in entries
            "$sec.$k" in excl && continue
            print(io, sec, ".", k, "=", repr(getfield(nt, Symbol(k))), "\n")
        end
    end
    return bytes2hex(sha256(take!(io)))
end

"""Resolve a value that may be the string "from_input" using an input-file attribute."""
function resolve_from_input(val, attrs::AbstractDict, attrname::String, key::String)
    if val isa String
        val == "from_input" || invalid("config key $key: expected a number or \"from_input\", got \"$val\"")
        haskey(attrs, attrname) || invalid("config key $key = \"from_input\" but the input file has no attribute $attrname")
        return Float64(attrs[attrname])
    end
    val === nothing && invalid("config key $key is required")
    return Float64(val)
end
