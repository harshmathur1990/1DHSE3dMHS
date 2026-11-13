# Drivers of run.mode = "column-hse-ltau", "column-hse-logm", "mhs-poisson-z", "mhs-poisson-ltau" and "mhs-poisson-logm" (MPI slabs in x + threads; Slab.jl).
# "column-hse-z" is run_column_hse (RunColumnHSE.jl); "full-mhs-z" runs through scripts/run.jl (M5 driver).

using .Slab
import .PoissonZ, .PoissonTau

zyx_of(f::Field, b) = [f[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx]
xyz_of(A::AbstractArray{<:Real,3}) = permutedims(A, (3, 2, 1))

"""physics.g for a τ-grid input (no file attributes in the IO/input-mhs-ltau.nc layout)."""
function tau_gravity(cfg, path)
    v = cfg.physics.g
    v isa Real && return Float64(v)
    a = h5open(f -> attrs_dict(f), path)
    haskey(a, "gravity_m_s2") && return Float64(a["gravity_m_s2"])
    invalid("physics.g = \"from_input\" but $path has no gravity_m_s2 attribute; set physics.g (e.g. 274.0)")
end
attrs_dict(f) = Dict(k => read(attributes(f)[k]) for k in keys(attributes(f)))

"""
τ-grid or column-mass input through the embedded API: coordinates from the file, the slab workspace from `prepare_embedded` on
COMM_WORLD (so the file drivers run exactly the embedded numerics), then this rank's slab of T and B.
"""
function tau_setup(cfg, tag, mode::Symbol)
    t0 = time()
    path = cfg.input.file
    isfile(path) || invalid("input file $path does not exist")
    ensure_mpi()
    NX, NY, nt = PoissonTau.tau_input_size(path)
    mass = mass_mode(mode); axname = mass ? "log10 m" : "log10 τ500"
    lt, x, y = h5open(f -> tau_coords(f, path, cfg.input.x, cfg.input.y; mass = mass), path)
    dx, dy = x[2] - x[1], y[2] - y[1]
    (dx > 0 && dy > 0) || invalid("$path: x and y must increase")
    issorted(lt) || invalid("$axname must increase (top level first) in $path")
    g = tau_gravity(cfg, path)
    tab = load_eos(cfg)
    G = cfg.tau_grid; P = cfg.poisson
    column = ColumnTauOptions(align = G.align)
    gn = !gn_mode(mode) ? nothing :
              PoissonTauOptions(logtau_cut = mass ? P.logm_cut : P.logtau_cut, cut_frame = Symbol(P.cut_frame), max_steps = P.max_steps == 0 ? 25 : P.max_steps, rtol = P.rtol,
                                n_consec = P.n_consec, stop = Symbol(P.stop), gn_tol = P.gn_tol, gn_consec = P.gn_consec,
                                cg_tol = P.cg_tol, cg_maxit = P.cg_maxit, dc_cap_m = P.dc_cap_m, lm_mu0 = P.lm_mu0,
                                lm = Symbol(P.lm), forcing = Symbol(P.forcing), cg_cap = P.cg_cap, eta_min = P.eta_min,
                                eta_max = P.eta_max, precond = Symbol(P.precond), klow = P.klow, damping = Symbol(P.damping),
                                qual_maxit = P.qual_maxit)
    poisson = mass && gn !== nothing ? PoissonMassOptions(gn) : gn
    logm0(s) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 && (println(@sprintf("[%7.1fs] %s: %s", time() - t0, tag, s)); flush(stdout))
    vw = config_view(cfg)
    cfg.tau.mu == 1.0 || invalid("tau.mu applies to the height modes only; τ modes take view.mu")
    ws = prepare_embedded(MPI.COMM_WORLD, mode; axis = lt, x = x, y = y, eos = tab, g = g, column = column,
                          poisson = poisson, logf = logm0, view = vw)
    par = ws.par
    logm(s) = rprint(par, @sprintf("[%7.1fs] %s: %s", time() - t0, tag, s))
    _, _, _, T, Bx, By, Bz = PoissonTau.read_tau_input(par, path; xname = cfg.input.x, yname = cfg.input.y)
    logm(@sprintf("input %s: %d levels %s %.2f … %.2f, %d × %d columns, dx %.2f km, g %.2f; ranks %d (x slabs), threads %d",
                  path, nt, axname, lt[1], lt[end], NX, NY, dx / 1e3, g, par.nranks, Threads.nthreads()))
    vw.mu == 1.0 || logm(@sprintf("view: μ %.4f (θ %.2f°), azimuth %.1f°, geometry %s; %s is along the line of sight",
                                    vw.mu, acosd(vw.mu), vw.azimuth_deg, vw.geometry, mass ? "m" : "τ"))
    return (; ws, par, logm, mass, lt, x, y, T, Bx, By, Bz, dx, dy, g, tab, t0, path, table_file = eos_table_path(cfg), view = vw)
end

"""log10 τ500 (`mass`: log10 m [kg m⁻²]), x and y [m] of a τ-grid or column-mass input (`PoissonTau.read_tau_input`)."""
function tau_coords(f, path, xname, yname; mass = false)
    mass ? haskey(f, "log10m") || invalid("$path has no dataset \"log10m\", which the column-mass (logm) modes read") :
           haskey(f, "log10tau500") || haskey(f, "logtau500") ||
           invalid("$path has no log10 τ500 axis" * (haskey(f, "log10m") ? " (it has log10m: use a logm mode)" : ""))
    if mass || haskey(f, "log10tau500")
        for nm in (xname, yname)
            haskey(f, nm) || invalid("$path has no dataset \"$nm\" (τ-grid inputs carry x and y like the height-grid input)")
        end
        ax(nm) = (a = attributes(f[nm]); fac = haskey(a, "units") ? UNIT_FACTORS[:length][attr_string(read(a["units"]))] : 1.0;
                  Float64.(read(f[nm])) .* fac)
        return mass ? PoissonTau.read_log10m(f) : read(f["log10tau500"]), ax(xname), ax(yname)
    end
    return read(f["logtau500"]), Float64.(read(f["x_m"])), Float64.(read(f["y_m"]))
end

"""
Output of the τ and column-mass modes: /tau_grid (axis logtau500), or /mass_grid (axis log10m [kg m⁻²], with logtau500 a
field: `logtau`), and /coords. `zt1m` (ny, nx): z(τ = 1) of a mass axis, on which τ = 1 is not a level.
"""
function write_tau_grid(s, out, lt, z, lnP, attrs; logtau = nothing, zt1m = nothing)
    grp = s.mass ? "mass_grid" : "tau_grid"
    rho = similar(lnP)
    Threads.@threads for idx in eachindex(lnP)
        r, _, _ = lnrho_dY(s.tab, log(s.T[idx]), lnP[idx]); rho[idx] = exp(r)
    end
    zt1 = s.mass ? permutedims(zt1m, (2, 1)) :
          [PoissonTau.interp_lt(@view(z[:, j, i]), lt, 0.0) for i in 1:size(z, 3), j in 1:size(z, 2)]   # (x, y)
    slabs = ["$grp/z_m" => xyz_of(z), "$grp/Pgas" => xyz_of(exp.(lnP)), "$grp/rho" => xyz_of(rho),
             "$grp/T" => xyz_of(s.T), "$grp/Bx" => xyz_of(s.Bx), "$grp/By" => xyz_of(s.By),
             "$grp/Bz" => xyz_of(s.Bz), "$grp/z_tau1_m" => zt1]
    s.mass && push!(slabs, "$grp/logtau500" => xyz_of(logtau))
    small = ["$grp/$(s.mass ? "log10m" : "logtau500")" => lt, "coords/x_m" => s.x, "coords/y_m" => s.y]
    a = Dict{String,Any}("input" => abspath(s.path), "eos_table" => abspath(s.table_file), "eos_model" => eos_model(s.tab),
                         "g" => s.g, "mpi_ranks" => s.par.nranks, "threads" => Threads.nthreads(),
                         "units" => s.mass ? "SI: log10m of m in kg m-2, z m, Pgas Pa, rho kg m-3, T K, B T; arrays Julia (x, y, m), h5py (m, y, x)" :
                                    "SI: z m, Pgas Pa, rho kg m-3, T K, B T; arrays Julia (x, y, τ), h5py (τ, y, x)")
    merge!(a, attrs)
    merge!(a, view_attrs(s.view))
    coop_write(s.par, out, slabs, small, a)
    zq = gquantiles(s.par, s.mass ? zt1m : z[argmin(abs.(lt)), :, :], [0.01, 0.5, 0.99]) ./ 1e3
    s.logm(@sprintf("z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km; wrote %s", zq..., out))
end

function check_output(par, cfg)
    out = cfg.run.output
    ok = !isfile(out) || cfg.run.overwrite
    gcount(par, ok ? 0 : 1) == 0 || invalid("output $out exists (set run.overwrite = true)")
    rank0(par) && mkpath(dirname(abspath(out)))
    return out
end

"""
An invalid result (EOS failure, non-finite, non-positive or folded fields): logged and not written; the driver returns
its status (exit 3). Every rank takes the same branch (the status is global). The output path is left as it was: a file
already there (an earlier run's, with run.overwrite = true) is not deleted, but the log says it was not replaced, and
rank 0 writes the failure record `<output>.failed.txt` (status, reason, messages, config, time) beside it. A later
valid run at that path removes the record (`clear_failure_record`).
"""
function invalid_result(logm, st, par, out, cfg)
    st.valid && return false
    old = gcount(par, rank0(par) && isfile(out) ? 1 : 0) > 0
    logm("invalid result ($(st.reason)): " * join(st.messages, "; ") * "; no output written" *
         (old ? "; the existing $out is from an earlier run and was NOT replaced" : ""))
    if rank0(par)
        open(failure_record(out), "w") do io
            println(io, "status = ", st.status, "\nreason = ", st.reason, "\nconfig = ", abspath(cfg.path),
                    "\ntime = ", Dates.now(), "\noutput_not_written = ", abspath(out),
                    "\nexisting_file_from_an_earlier_run = ", old)
            foreach(m -> println(io, "message = ", m), st.messages)
        end
    end
    return true
end

failure_record(out) = out * ".failed.txt"
"""Remove a failure record left at this output path by an earlier invalid run (rank 0, after a valid write)."""
clear_failure_record(par, out) = (rank0(par) && isfile(failure_record(out)) && rm(failure_record(out)); nothing)

"""run.mode = "column-hse-ltau", "column-hse-logm": hydrostatic columns in τ or m (P, z, ρ on the axis), aligned on P + B²/2μ₀."""
function run_column_hse_ltau(cfg::Config)
    status = invalid_input                              # until the solve has returned
    name = cfg.run.mode
    s = tau_setup(cfg, name, Symbol(replace(name, "-" => "_")))
    try
        out = check_output(s.par, cfg)
        G = cfg.tau_grid
        res = solve_embedded!(s.ws, log.(s.T), s.Bx, s.By, s.Bz)
        st = res.status; status = st.status
        invalid_result(s.logm, st, s.par, out, cfg) && return status
        s.logm(@sprintf("columns: aligned on P + B²/2μ₀ = %.4e Pa (%d columns extrapolated)",
                        st.stats["align_total_pressure_Pa"], st.stats["align_extrapolated_columns"]))
        write_tau_grid(s, out, s.lt, res.z, res.lnP, Dict{String,Any}("mode" => name, "align" => G.align);
                       logtau = res.logtau500, zt1m = res.z_tau1)
        clear_failure_record(s.par, out)
    finally
        free_embedded!(s.ws)
    end
    return status
end

"""run.mode = "mhs-poisson-ltau", "mhs-poisson-logm": column HSE in τ or m, then the direct Poisson solve on that grid."""
function run_mhs_poisson_ltau(cfg::Config)
    status = invalid_input                              # until the solve has returned
    name = cfg.run.mode
    s = tau_setup(cfg, name, Symbol(replace(name, "-" => "_")))
    try
        out = check_output(s.par, cfg)
        G = cfg.tau_grid; P = cfg.poisson
        BLAS.set_num_threads(min(32, Threads.nthreads()))   # standalone driver only (the embedded API leaves BLAS alone)
        res = try
            solve_embedded!(s.ws, log.(s.T), s.Bx, s.By, s.Bz)
        finally
            BLAS.set_num_threads(1)
        end
        st = res.status; status = st.status
        invalid_result(s.logm, st, s.par, out, cfg) && return status
        s.logm(@sprintf("solve done in %d steps, Φ/Φ₀ %.3e; gauge shift %.1f km", st.steps, st.phi / st.phi0,
                        st.stats["gauge_shift_m"] / 1e3))
        write_tau_grid(s, out, s.lt, res.z, res.lnP, Dict{String,Any}("mode" => name,
                       (s.mass ? "logm_cut" : "logtau_cut") => s.lt[st.stats["cut_level"]],
                       (s.mass ? "logm_cut_frame" : "logtau_cut_frame") => String(s.ws.poisson.cut_frame),
                       "phi_start" => st.phi0, "phi_end" => st.phi,
                       "steps" => st.steps, "align" => G.align,
                       "stop_reason" => String(st.reason), "gn_decrement" => st.gn_decrement, "status" => string(st.status),
                       "gn_decrement_source" => st.stats["gn_decrement_source"],
                       "roundoff" => st.stats["roundoff"], "phi_floor" => st.stats["phi_floor"],
                       "qual_outcome" => haskey(st.stats, "stop_check") ? st.stats["stop_check"]["outcome"] : "none",
                       "qual_lower" => haskey(st.stats, "stop_check") ? st.stats["stop_check"]["lower"] : NaN,
                       "qual_upper" => haskey(st.stats, "stop_check") ? st.stats["stop_check"]["upper"] : NaN,
                       ("work_" * k => Float64(v) for (k, v) in st.stats["work"])...,
                       "solve_seconds" => st.seconds, "ranks" => s.par.nranks, "threads" => Threads.nthreads());
                       logtau = res.logtau500, zt1m = res.z_tau1)
        clear_failure_record(s.par, out)
    finally
        free_embedded!(s.ws)
    end
    return status
end

"""run.mode = "mhs-poisson-z": column HSE on the height grid (as column-hse-z), then the Poisson step below z_cut."""
function run_mhs_poisson_z(cfg::Config)
    status = invalid_input                              # until the solve has returned
    t0 = time()
    I = cfg.input; P = cfg.poisson
    spec = open_input(I.file; crop_z_m = I.crop_z_m, tname = I.temperature, xname = I.x, yname = I.y, zname = I.z)
    NX, NY, NZ = length(spec.x), length(spec.y), length(spec.z)
    ensure_mpi()
    g = input_gravity(cfg.physics.g, spec)
    o, Psrc = hse_opts_from_config(cfg, spec, g)
    eos = load_eos(cfg)
    column = ColumnZOptions(anchor = o.anchor, P_ref = o.P_ref, z_ref = o.z_ref, n_sub = o.n_sub, adapt = o.adapt,
                            rk_tol = o.rk_tol, n_sub_max = o.n_sub_max, tau_top = o.tau_top, H_max = o.H_max, mu = o.mu)
    poisson = PoissonZOptions(z_cut_m = P.z_cut_m, max_steps = P.max_steps == 0 ? 100 : P.max_steps, rtol = P.rtol, n_consec = P.n_consec,
                              anchor_w = P.anchor_w, cg_tol = P.cg_tol, cg_maxit = P.cg_maxit)
    logm0(s) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 && (println(@sprintf("[%7.1fs] mhs-poisson-z: %s", time() - t0, s)); flush(stdout))
    vw = config_view(cfg)
    ws = prepare_embedded(MPI.COMM_WORLD, :mhs_poisson_z; axis = spec.z, x = spec.x, y = spec.y, eos = eos, g = g,
                          column = column, poisson = poisson, logf = logm0, view = vw)
    par = ws.par; b = par.block
    logm(s) = rprint(par, @sprintf("[%7.1fs] mhs-poisson-z: %s", time() - t0, s))
    try
        out = check_output(par, cfg)
        logm(@sprintf("grid %d×%d×%d, z_cut %.3f Mm, anchor %s P_ref = %.6g Pa (%s), g = %.3f, ranks %d (x slabs), threads %d",
                      NX, NY, NZ, P.z_cut_m / 1e6, o.anchor, o.P_ref, Psrc, g, par.nranks, Threads.nthreads()))
        Tf = read_field(spec, par, I.temperature, :temperature; positive = true)
        T = zyx_of(Tf, b)
        Bx, By, Bz = (zyx_of(read_field(spec, par, nm, :field), b) for nm in (I.bx, I.by, I.bz))
        res = solve_embedded!(ws, log.(T), Bx, By, Bz)
        st = res.status; status = st.status
        invalid_result(logm, st, par, out, cfg) && return status
        logm("column HSE: " * join(["$k = $v" for (k, v) in sort(collect(st.stats); by = first)], ", "))
        z = spec.z
        zt1 = res.z_tau1
        zq = gquantiles(par, zt1, [0.01, 0.5, 0.99]) ./ 1e3
        kc = st.stats["cut_level"]
        dq = gquantiles(par, res.lnP[kc:end, :, :] .- log.(res.Pgas_start[kc:end, :, :]), [0.01, 0.5, 0.99])
        logm(@sprintf("done: ln(P/P_colhse) below z_cut p1/p50/p99 %.3f / %.3f / %.3f; z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km", dq..., zq...))
        slabs = ["atmosphere/Pgas" => xyz_of(res.Pgas), "atmosphere/rho" => xyz_of(res.rho), "atmosphere/T" => xyz_of(T),
                 "atmosphere/tau500" => xyz_of(res.tau500), "atmosphere/Bx_centre" => xyz_of(Bx), "atmosphere/By_centre" => xyz_of(By),
                 "atmosphere/Bz_centre" => xyz_of(Bz), "atmosphere/Pgas_column_hse" => xyz_of(res.Pgas_start),
                 "atmosphere/z_tau1_m" => permutedims(zt1, (2, 1))]
        small = ["coords/x_m" => spec.x, "coords/y_m" => spec.y, "coords/z_m" => z]
        a = Dict{String,Any}("kind" => "poisson_step", "mode" => "mhs-poisson-z", "accepted" => 0, "input" => abspath(I.file),
                             "z_cut_m" => P.z_cut_m, "eos_table" => abspath(eos_table_path(cfg)), "eos_model" => eos_model(eos), "phi_start" => st.phi0,
                             "phi_end" => st.phi, "steps" => st.steps, "P_ref_Pa" => o.P_ref, "anchor" => String(o.anchor),
                             "g" => g, "mpi_ranks" => par.nranks, "threads" => Threads.nthreads(),
                             "stop_reason" => String(st.reason), "gn_decrement" => st.gn_decrement,
                             "units" => "SI: Pgas Pa, rho kg m-3, T K, B T, z m (top first); arrays Julia (x, y, z), h5py (z, y, x)")
        merge!(a, view_attrs(vw))
        coop_write(par, out, slabs, small, a)
        clear_failure_record(par, out)
        logm("wrote $out")
    finally
        free_embedded!(ws)
    end
    return status
end
