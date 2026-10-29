# run.mode = "column_hse": read → column_hse → diagnostics → HDF5 (implementation doc §9, §11).

"""Global quantiles of a per-column (nx, ny) array."""
function column_quantiles(par::ParCtx, a::AbstractMatrix, ps)
    v = Float64[x for x in a if isfinite(x)]
    vals = gather_layer(par, v)
    out = fill(NaN, length(ps))
    if rank0(par) && !isempty(vals)
        sort!(vals)
        out .= quantile(vals, ps; sorted = true)
    end
    MPI.Bcast!(out, par.comm; root = 0)
    return out
end

function hse_opts_from_config(cfg::Config, spec::InputSpec, g)
    C = cfg.column_hse
    anchor = Symbol(C.anchor)
    anchor in (:bottom, :top, :height) || invalid("column_hse.anchor must be bottom, top or height")
    P_ref, src = if C.P_ref === nothing
        anchor == :top || invalid("column_hse.P_ref is required for anchor = $(C.anchor)")
        0.1, "default for anchor = top (spec D2)"
    elseif C.P_ref == "from_input"
        resolve_from_input(C.P_ref, spec.attrs, "P_ref_Pa", "column_hse.P_ref"), "input attribute P_ref_Pa"
    else
        Float64(C.P_ref), "config"
    end
    T = cfg.tau
    v = column_with_view(ColumnZOptions(P_ref = P_ref, mu = Float64(T.mu)), config_view(cfg))
    o = HSEOpts(anchor, P_ref, Float64(C.z_ref_m), g, C.n_sub, C.adapt, C.rk_tol, C.n_sub_max,
                Symbol(T.top), Float64(T.H_max_m), v.mu)
    return o, src
end

"""
    run_column_hse(cfg) -> SolverStatus
"""
function run_column_hse(cfg::Config)
    t0 = time()
    I = cfg.input
    spec = open_input(I.file; crop_z_m = I.crop_z_m, tname = I.temperature, xname = I.x, yname = I.y, zname = I.z)
    NX, NY, NZ = length(spec.x), length(spec.y), length(spec.z)
    par = init_parallel(NX, NY, NZ; px = cfg.parallel.px, py = cfg.parallel.py, debug = cfg.run.debug)
    logm(msg) = rank0(par) && (@printf("[%7.1fs] column_hse: %s\n", time() - t0, msg); flush(stdout))
    grid = build_grid(spec)
    ops = Ops(grid, par)
    g = input_gravity(cfg.physics.g, spec)
    o, Psrc = hse_opts_from_config(cfg, spec, g)
    logm(@sprintf("grid %d×%d×%d, anchor %s, P_ref = %.6g Pa (%s), g = %.3f, ranks %d (%d×%d), threads %d",
                  NX, NY, NZ, o.anchor, o.P_ref, Psrc, g, par.nranks, par.px, par.py, Threads.nthreads()))
    eos = load_table(cfg.eos.table_file; pf_file = cfg.eos.pf_file)
    T = read_field(spec, par, I.temperature, :temperature; positive = true)
    logm("read temperature; EOS table $(cfg.eos.table_file) (T_sw = $(eos.T_sw) K)")
    res = column_hse(par, grid, T, eos, o)
    vw = config_view(cfg); tsl = slant(vw)
    if tsl != (0.0, 0.0) && res.status == converged          # τ along inclined rays (ViewTau.jl); z(τ=1) on footprints
        b = par.block; F0 = res.fields
        chi = [F0["chi500"][i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx]
        tau = similar(chi); zt = zeros(b.ny, b.nx); tm = zeros(UInt8, b.ny, b.nx)
        inclined_tau!(tau, zt, tm, par, chi, grid.z.s, spec.x[2] - spec.x[1], spec.y[2] - spec.y[1], o.mu, tsl;
                      top = o.tau_top, H_max = o.H_max)
        for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
            F0["tau500"][i, j, k] = tau[k, j, i]; F0["logtau500"][i, j, k] = log10(tau[k, j, i])
        end
        res.z_tau1 .= permutedims(zt); res.tau_top_method .= permutedims(tm)
        logm(@sprintf("τ along inclined rays: μ %.4f, azimuth %.1f°, t = (%.4f, %.4f)", o.mu, vw.azimuth_deg, tsl...))
    end
    logm("integrated: " * join(["$k = $v" for (k, v) in sort(collect(res.stats); by = first)], ", "))
    F = res.fields
    Rx, Ry, Rz = column_residual(par, ops, F["Pgas"], F["rho"], g)
    logm("residual done")

    # ---- per-layer tables ----
    ps3 = [0.01, 0.5, 0.99]
    lq(val, ps) = layer_quantiles(par, val, ps)
    P = F["Pgas"]; rho = F["rho"]
    tab_P = hcat(lq((i, j, k) -> log10(P[i, j, k]), ps3), lq((i, j, k) -> F["logtau500"][i, j, k], ps3),
                 lq((i, j, k) -> log10(rho[i, j, k]), ps3))
    cols_P = ["log10P_p1", "log10P_p50", "log10P_p99", "logtau_p1", "logtau_p50", "logtau_p99",
              "log10rho_p1", "log10rho_p50", "log10rho_p99"]
    rg(i, j, k) = rho[i, j, k] * g
    tab_R = hcat(lq((i, j, k) -> sqrt(Rx[i, j, k]^2 + Ry[i, j, k]^2) / rg(i, j, k), [0.5, 0.99]),
                 lq((i, j, k) -> abs(Rz[i, j, k]) / rg(i, j, k), [0.5, 0.99]))
    cols_R = ["Rh_over_rhog_p50", "Rh_over_rhog_p99", "Rz_over_rhog_p50", "Rz_over_rhog_p99"]
    zt = column_quantiles(par, res.z_tau1, [0.01, 0.5, 0.99])
    logm(@sprintf("z(τ500 = 1) p1/p50/p99 = %.1f / %.1f / %.1f km", (zt ./ 1e3)...))

    # ---- write ----
    out = cfg.run.output
    create_output(par, out; overwrite = cfg.run.overwrite)
    copy_inputs(par, out, spec.path, [I.temperature, I.bx, I.by, I.bz, I.x, I.y, I.z])
    fo = FieldOut[FieldOut("/output/$nm", F[nm], u, ln) for (nm, u, ln) in COLHSE_FIELDS]
    push!(fo, FieldOut("/diagnostics/R_x", Rx, "N m-3", "x force residual, B = 0 (vertical_hse_only)"))
    push!(fo, FieldOut("/diagnostics/R_y", Ry, "N m-3", "y force residual, B = 0 (vertical_hse_only)"))
    push!(fo, FieldOut("/diagnostics/R_z", Rz, "N m-3", "z force residual dP/dz + rho g (discrete operator)"))
    push!(fo, FieldOut("/diagnostics/eos_status", res.eos_status, "1", "EOS status bit mask (implementation doc §4.5)"))
    co = ColumnOut[ColumnOut("/diagnostics/column_status", res.column_status, "1", "column_hse bit mask"),
                   ColumnOut("/diagnostics/tau_top_method", res.tau_top_method, "1", "0 exp_tail, 1 half_cell, 2 zero"),
                   ColumnOut("/diagnostics/rk_max_substeps", res.rk_max_substeps, "1", "largest RK4 substep count used"),
                   ColumnOut("/diagnostics/z_tau500_1", res.z_tau1, "m", "height of tau500 = 1")]
    write_fields(par, out, fo; cols = co, compression = cfg.output.compression)
    write_layer_table(par, out, "/diagnostics/layer_pressure", tab_P, cols_P)
    write_layer_table(par, out, "/diagnostics/layer_residual", tab_R, cols_R)
    meta = provenance_meta(par, cfg)
    merge!(meta, Dict{String,Any}(
        "mode" => "column_hse", "label" => "vertical_hse_only", "g" => g,
        "anchor" => Dict{String,Any}("type" => String(o.anchor), "value_Pa" => o.P_ref, "source" => Psrc,
                                     "z_ref_m" => o.z_ref, "plane_z_m" => o.anchor == :bottom ? spec.z[end] :
                                                                         o.anchor == :top ? spec.z[1] : o.z_ref),
        "integrator" => Dict{String,Any}("scheme" => "RK4, ln T linear in z", "n_sub" => o.n_sub,
                                         "adapt" => o.adapt, "rk_tol" => o.rk_tol, "n_sub_max" => o.n_sub_max),
        "tau" => Dict{String,Any}("convention" => o.mu == 1.0 ? "vertical" : tsl != (0.0, 0.0) ? "inclined_rays" : "per_column",
                                  "top_method" => String(o.tau_top), "H_max_m" => o.H_max, "mu" => o.mu,
                                  "azimuth_deg" => vw.azimuth_deg, "geometry" => String(vw.geometry),
                                  "z_tau1" => tsl != (0.0, 0.0) ? "on ray footprints (x, y at z = 0)" : "per column",
                                  "wavelength_A" => WL500, "extinction" => "absorption + scattering (counted once)"),
        "eos" => Dict{String,Any}("table_file" => abspath(cfg.eos.table_file), "table_sha256" => table_sha256(eos),
                                  "T_sw" => eos.T_sw, "hot_branch" => eos.hot_branch,
                                  "fix_gasc_h2" => get(eos.meta, "fix_gasc_h2", -1),
                                  "pf_sha256" => get(eos.meta, "pf_sha256", "")),
        "stats" => Dict{String,Any}(res.stats..., "z_tau1_p1_p50_p99_m" => zt),
        "flags" => Dict{String,Any}("solver_status" => string(res.status), "messages" => join(res.messages, "; ")),
        "preprocessing" => Dict{String,Any}("crop_z_m" => isempty(I.crop_z_m) ? "none" : I.crop_z_m,
                                            "resample" => I.resample)))
    write_meta(par, out, meta)
    if rank0(par)
        h5open(out, "r+") do f
            f["diagnostics/z"] = spec.z
        end
    end
    logm("wrote $out; status $(res.status)")
    return res.status
end
