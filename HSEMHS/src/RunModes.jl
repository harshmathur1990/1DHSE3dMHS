# Drivers of run.mode = "column-hse-ltau", "mhs-poisson-z" and "mhs-poisson-ltau" (MPI slabs in x + threads; Slab.jl).
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

"""τ-grid input on this rank's slab (x, y from the file, as for the height-grid input)."""
function tau_setup(cfg, tag)
    t0 = time()
    path = cfg.input.file
    isfile(path) || invalid("input file $path does not exist")
    NX, NY, nt = PoissonTau.tau_input_size(path)
    par = slab_par(NX, NY, nt)
    logm(s) = rprint(par, @sprintf("[%7.1fs] %s: %s", time() - t0, tag, s))
    lt, x, y, T, Bx, By, Bz = PoissonTau.read_tau_input(par, path; xname = cfg.input.x, yname = cfg.input.y)
    dx, dy = x[2] - x[1], y[2] - y[1]
    (dx > 0 && dy > 0) || invalid("$path: x and y must increase")
    issorted(lt) || invalid("log10 τ500 must increase (top level first) in $path")
    g = tau_gravity(cfg, path)
    tab = load_table(cfg.eos.table_file; pf_file = cfg.eos.pf_file)
    logm(@sprintf("input %s: %d levels log10 τ500 %.2f … %.2f, %d × %d columns, dx %.2f km, g %.2f; ranks %d (x slabs), threads %d",
                  path, nt, lt[1], lt[end], NX, NY, dx / 1e3, g, par.nranks, Threads.nthreads()))
    return (; par, logm, lt, x, y, T, Bx, By, Bz, dx, dy, g, tab, t0, path, table_file = cfg.eos.table_file)
end

function write_tau_grid(s, out, lt, z, lnP, attrs)
    rho = similar(lnP)
    Threads.@threads for idx in eachindex(lnP)
        r, _, _ = lnrho_dY(s.tab, log(s.T[idx]), lnP[idx]); rho[idx] = exp(r)
    end
    k0 = argmin(abs.(lt))
    zt1 = [PoissonTau.interp_lt(@view(z[:, j, i]), lt, 0.0) for i in 1:size(z, 3), j in 1:size(z, 2)]   # (x, y)
    slabs = ["tau_grid/z_m" => xyz_of(z), "tau_grid/Pgas" => xyz_of(exp.(lnP)), "tau_grid/rho" => xyz_of(rho),
             "tau_grid/T" => xyz_of(s.T), "tau_grid/Bx" => xyz_of(s.Bx), "tau_grid/By" => xyz_of(s.By),
             "tau_grid/Bz" => xyz_of(s.Bz), "tau_grid/z_tau1_m" => zt1]
    small = ["tau_grid/logtau500" => lt, "coords/x_m" => s.x, "coords/y_m" => s.y]
    a = Dict{String,Any}("input" => abspath(s.path), "eos_table" => abspath(s.table_file),
                         "g" => s.g, "mpi_ranks" => s.par.nranks, "threads" => Threads.nthreads(),
                         "units" => "SI: z m, Pgas Pa, rho kg m-3, T K, B T; arrays Julia (x, y, τ), h5py (τ, y, x)")
    merge!(a, attrs)
    coop_write(s.par, out, slabs, small, a)
    zq = gquantiles(s.par, z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3
    s.logm(@sprintf("z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km; wrote %s", zq..., out))
end

function check_output(par, cfg)
    out = cfg.run.output
    ok = !isfile(out) || cfg.run.overwrite
    gcount(par, ok ? 0 : 1) == 0 || invalid("output $out exists (set run.overwrite = true)")
    rank0(par) && mkpath(dirname(abspath(out)))
    return out
end

"""run.mode = "column-hse-ltau": hydrostatic columns in τ (P(τ), z(τ), ρ(τ)), aligned on P + B²/2μ₀."""
function run_column_hse_ltau(cfg::Config)
    s = tau_setup(cfg, "column-hse-ltau")
    out = check_output(s.par, cfg)
    G = cfg.tau_grid
    lnP, z, ncap, Pt, nb = PoissonTau.column_hse_tau(s.par, s.lt, log.(s.T), s.Bx, s.By, s.Bz, s.tab, s.g;
                                                     cap_per_dex = G.cap_per_dex, align = G.align)
    s.logm(@sprintf("columns: ln P capped in %d level intervals (> %.0f per dex); aligned on P + B²/2μ₀ = %.4e Pa (%d columns extrapolated)",
                    ncap, G.cap_per_dex, Pt, nb))
    write_tau_grid(s, out, s.lt, z, lnP, Dict{String,Any}("mode" => "column-hse-ltau", "align" => G.align,
                                                         "cap_per_dex" => G.cap_per_dex, "capped_intervals" => ncap))
    return converged
end

"""run.mode = "mhs-poisson-ltau": column HSE in τ, then the direct Poisson solve on the τ grid."""
function run_mhs_poisson_ltau(cfg::Config)
    s = tau_setup(cfg, "mhs-poisson-ltau")
    out = check_output(s.par, cfg)
    G = cfg.tau_grid; P = cfg.poisson
    lT = log.(s.T)
    lnP, z0, ncap, Pt, nb = PoissonTau.column_hse_tau(s.par, s.lt, lT, s.Bx, s.By, s.Bz, s.tab, s.g;
                                                      cap_per_dex = G.cap_per_dex, align = G.align)
    s.logm(@sprintf("column HSE in τ: ln P capped in %d intervals; aligned on P + B²/2μ₀ = %.4e Pa", ncap, Pt))
    BLAS.set_num_threads(min(32, Threads.nthreads()))
    u, z, Φ0, Φ, steps, kc = PoissonTau.solve_tau(s.par, s.tab, s.g, s.lt, lT, s.Bx, s.By, s.Bz, lnP, z0, s.dx, s.dy;
                                                  logtau_cut = P.logtau_cut, max_steps = P.max_steps, rtol = P.rtol,
                                                  n_consec = P.n_consec,
                                                  cg_maxit = P.cg_maxit, cg_tol = P.cg_tol, dc_cap = P.dc_cap_m,
                                                  lm_mu0 = P.lm_mu0, logf = s.logm)
    BLAS.set_num_threads(1)
    sg = PoissonTau.gauge!(s.par, z, s.lt)
    s.logm(@sprintf("solve done in %d steps, Φ/Φ₀ %.3e; gauge shift %.1f km", steps, Φ / Φ0, sg / 1e3))
    write_tau_grid(s, out, s.lt, z, u, Dict{String,Any}("mode" => "mhs-poisson-ltau", "logtau_cut" => s.lt[kc],
                   "phi_start" => Φ0, "phi_end" => Φ, "steps" => steps, "align" => G.align, "cap_per_dex" => G.cap_per_dex))
    return converged
end

"""run.mode = "mhs-poisson-z": column HSE on the height grid (as column-hse-z), then the Poisson step below z_cut."""
function run_mhs_poisson_z(cfg::Config)
    t0 = time()
    I = cfg.input; P = cfg.poisson
    spec = open_input(I.file; crop_z_m = I.crop_z_m, tname = I.temperature, xname = I.x, yname = I.y, zname = I.z)
    NX, NY, NZ = length(spec.x), length(spec.y), length(spec.z)
    par = slab_par(NX, NY, NZ); b = par.block
    logm(s) = rprint(par, @sprintf("[%7.1fs] mhs-poisson-z: %s", time() - t0, s))
    out = check_output(par, cfg)
    grid = build_grid(spec)
    g = input_gravity(cfg.physics.g, spec)
    o, Psrc = hse_opts_from_config(cfg, spec, g)
    eos = load_table(cfg.eos.table_file; pf_file = cfg.eos.pf_file)
    logm(@sprintf("grid %d×%d×%d, z_cut %.3f Mm, anchor %s P_ref = %.6g Pa (%s), g = %.3f, ranks %d (x slabs), threads %d",
                  NX, NY, NZ, P.z_cut_m / 1e6, o.anchor, o.P_ref, Psrc, g, par.nranks, Threads.nthreads()))
    Tf = read_field(spec, par, I.temperature, :temperature; positive = true)
    res = column_hse(par, grid, Tf, eos, o)
    logm("column HSE: " * join(["$k = $v" for (k, v) in sort(collect(res.stats))], ", "))
    T = zyx_of(Tf, b); P0 = zyx_of(res.fields["Pgas"], b)
    Bx, By, Bz = (zyx_of(read_field(spec, par, nm, :field), b) for nm in (I.bx, I.by, I.bz))
    z = spec.z; dx = spec.x[2] - spec.x[1]; dy = spec.y[2] - spec.y[1]
    lnT = log.(T)
    Lx, Ly, Lz = PoissonZ.lorentz(par, Bx, By, Bz, z, dx, dy)
    sol = PoissonZ.solve_pressure(par, eos, g, lnT, Lx, Ly, Lz, P0, z, dx, dy; zcut = P.z_cut_m, maxouter = P.max_steps,
                                  rtol = P.rtol, anchor_w = P.anchor_w, cg_tol = P.cg_tol, cg_maxit = P.cg_maxit,
                                  n_consec = P.n_consec, logf = logm)
    T_ = cfg.tau
    rho, tau, zt1 = PoissonZ.tau_columns(eos, lnT, sol.P, z; top = Symbol(T_.top), H_max = Float64(T_.H_max_m), mu = Float64(T_.mu))
    zq = gquantiles(par, zt1, [0.01, 0.5, 0.99]) ./ 1e3
    dq = gquantiles(par, sol.u .- sol.u0, [0.01, 0.5, 0.99])
    logm(@sprintf("done: ln(P/P_colhse) below z_cut p1/p50/p99 %.3f / %.3f / %.3f; z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km", dq..., zq...))
    slabs = ["atmosphere/Pgas" => xyz_of(sol.P), "atmosphere/rho" => xyz_of(rho), "atmosphere/T" => xyz_of(T),
             "atmosphere/tau500" => xyz_of(tau), "atmosphere/Bx_centre" => xyz_of(Bx), "atmosphere/By_centre" => xyz_of(By),
             "atmosphere/Bz_centre" => xyz_of(Bz), "atmosphere/Pgas_column_hse" => xyz_of(P0),
             "atmosphere/z_tau1_m" => permutedims(zt1, (2, 1))]
    small = ["coords/x_m" => spec.x, "coords/y_m" => spec.y, "coords/z_m" => z]
    a = Dict{String,Any}("kind" => "poisson_step", "mode" => "mhs-poisson-z", "accepted" => 0, "input" => abspath(I.file),
                         "z_cut_m" => P.z_cut_m, "eos_table" => abspath(cfg.eos.table_file), "phi_start" => sol.Φ0,
                         "phi_end" => sol.Φ, "steps" => sol.steps, "P_ref_Pa" => o.P_ref, "anchor" => String(o.anchor),
                         "g" => g, "mpi_ranks" => par.nranks, "threads" => Threads.nthreads(),
                         "units" => "SI: Pgas Pa, rho kg m-3, T K, B T, z m (top first); arrays Julia (x, y, z), h5py (z, y, x)")
    coop_write(par, out, slabs, small, a)
    logm("wrote $out")
    return converged
end
