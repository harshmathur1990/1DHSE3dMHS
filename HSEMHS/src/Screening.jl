# M3: force-budget screening and the column-HSE horizontal-residual report
# (spec §3 item 5, implementation doc §12). Labelled "screening", never "feasibility".

"""
    read_own_cube!(dest, path, dsname, par)

Read this rank's hyperslab of a 3D dataset written by `write_fields` (Julia dims (z, y, x)).
"""
function read_own_cube!(dest::Field, path::AbstractString, dsname::AbstractString, par::ParCtx)
    b = par.block
    h5open(path, "r") do f
        haskey(f, dsname) || invalid("$path has no dataset $dsname")
        ds = f[dsname]
        size(ds) == (b.nz, par.NY, par.NX) || invalid("$path:$dsname has shape $(size(ds)), expected $((b.nz, par.NY, par.NX))")
        a = ds[1:b.nz, b.j0:b.j0+b.ny-1, b.i0:b.i0+b.nx-1]
        Threads.@threads :static for k in 1:b.nz
            @inbounds for j in 1:b.ny, i in 1:b.nx
                dest[i, j, k] = a[k, j, i]
            end
        end
    end
    return dest
end

"""Synthetic σ per component: sqrt((σ_rel |B|)² + σ_floor²) (spec §6.3)."""
@inline sigma_synth(Bm, srel, sfloor) = sqrt((srel * Bm)^2 + sfloor^2)

"""
    screening_fields(par, ops, P, Bx, By, Bz; sigma_rel, sigma_floor, ell) -> (gradhP, Lh, M, dL)

Per node (halos of P and B are exchanged here):
|∇ₕP|, |Lₕ(B)|, the mismatch M = |∇ₕP − Lₕ|, and the correction scale
|δL| = |B| σ_B/(μ₀ ℓ) with σ_B = sqrt(σx² + σy² + σz²) (E10).
"""
function screening_fields(par::ParCtx, ops::Ops, P::Field, Bx::Field, By::Field, Bz::Field;
                          sigma_rel = 0.05, sigma_floor = 5e-4, ell)
    exchange_halos!(par, [P, Bx, By, Bz])
    G = newfield(par); L = newfield(par); M = newfield(par); D = newfield(par)
    b = par.block
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            Lx, Ly, _ = lorentz(Bx, By, Bz, ops, i, j, k)
            gx = dx(P, ops.X, i, j, k); gy = dy(P, ops.Y, i, j, k)
            Bm = sqrt(Bx[i, j, k]^2 + By[i, j, k]^2 + Bz[i, j, k]^2)
            sB = sqrt(3.0) * sigma_synth(Bm, sigma_rel, sigma_floor)
            G[i, j, k] = sqrt(gx^2 + gy^2)
            L[i, j, k] = sqrt(Lx^2 + Ly^2)
            M[i, j, k] = sqrt((gx - Lx)^2 + (gy - Ly)^2)
            D[i, j, k] = Bm * sB / (MU0 * ell)
        end
    end
    return G, L, M, D
end

"""
    run_screening(cfg) -> SolverStatus

run.mode = "screening": needs `input.file` (for B) and `screening.colhse_file` (Pgas, rho,
R_x/R_y/R_z from a column_hse run on the same grid). Writes `run.output` (.h5) and a .md report.
"""
function run_screening(cfg::Config)
    t0 = time()
    I = cfg.input; S = cfg.screening
    isempty(S.colhse_file) && invalid("screening.colhse_file is required for run.mode = \"screening\"")
    spec = open_input(I.file; crop_z_m = I.crop_z_m, tname = I.temperature, xname = I.x, yname = I.y, zname = I.z)
    NX, NY, NZ = length(spec.x), length(spec.y), length(spec.z)
    par = init_parallel(NX, NY, NZ; px = cfg.parallel.px, py = cfg.parallel.py, debug = cfg.run.debug)
    logm(msg) = rank0(par) && (@printf("[%7.1fs] screening: %s\n", time() - t0, msg); flush(stdout))
    zc, anchor = h5open(S.colhse_file, "r") do f
        read(f["diagnostics/z"]), read_attribute(f["meta/anchor"], "type")
    end
    maximum(abs.(zc .- spec.z)) <= 1e-3 || invalid("height grid of $(S.colhse_file) differs from $(I.file)")
    grid = build_grid(spec); ops = Ops(grid, par)
    g = input_gravity(cfg.physics.g, spec)
    Bx = read_field(spec, par, I.bx, :field); By = read_field(spec, par, I.by, :field); Bz = read_field(spec, par, I.bz, :field)
    P = newfield(par); rho = newfield(par)
    read_own_cube!(P, S.colhse_file, "output/Pgas", par)
    read_own_cube!(rho, S.colhse_file, "output/rho", par)
    Rx = newfield(par); Ry = newfield(par); Rz = newfield(par)
    read_own_cube!(Rx, S.colhse_file, "diagnostics/R_x", par)
    read_own_cube!(Ry, S.colhse_file, "diagnostics/R_y", par)
    read_own_cube!(Rz, S.colhse_file, "diagnostics/R_z", par)
    logm("read B, Pgas, rho, R from $(S.colhse_file) (anchor $anchor)")
    ell = S.ell_factor * (spec.x[2] - spec.x[1])
    G, L, M, D = screening_fields(par, ops, P, Bx, By, Bz; sigma_rel = cfg.mhs.sigma_rel,
                                  sigma_floor = Float64(cfg.mhs.sigma_floor_T), ell = ell)
    lq(v, ps) = layer_quantiles(par, v, ps)
    p3 = [0.5, 0.9, 0.99]
    beta(i, j, k) = @inbounds 2MU0 * P[i, j, k] / max(Bx[i, j, k]^2 + By[i, j, k]^2 + Bz[i, j, k]^2, ONE_GAUSS^2)
    tab = hcat(lq(G, [0.5]), lq(L, [0.5]), lq(M, [0.5]), lq(D, [0.5]),
               layer_fraction(par, (i, j, k) -> D[i, j, k] >= M[i, j, k]),
               lq((i, j, k) -> M[i, j, k] / D[i, j, k], p3), lq(beta, [0.5]))
    cols = ["gradhP_p50", "Lh_p50", "mismatch_p50", "dL_p50", "frac_dL_ge_mismatch",
            "mismatch_over_dL_p50", "mismatch_over_dL_p90", "mismatch_over_dL_p99", "beta_p50"]
    # column-HSE horizontal residual: |R_h| and |R_z| over ρg and over the local force scale
    sc(i, j, k) = @inbounds sqrt(Rx[i, j, k]^2 + Ry[i, j, k]^2) / (rho[i, j, k] * g)
    sz(i, j, k) = @inbounds abs(Rz[i, j, k]) / (rho[i, j, k] * g)
    sf(i, j, k) = @inbounds sqrt(Rx[i, j, k]^2 + Ry[i, j, k]^2) / (G[i, j, k] + rho[i, j, k] * g)
    rtab = hcat(lq(sc, p3), lq(sz, p3), lq(sf, p3))
    rcols = ["Rh_over_rhog_p50", "Rh_over_rhog_p90", "Rh_over_rhog_p99", "Rz_over_rhog_p50", "Rz_over_rhog_p90",
             "Rz_over_rhog_p99", "Rh_over_scale_p50", "Rh_over_scale_p90", "Rh_over_scale_p99"]
    logm("layer statistics done")
    out = cfg.run.output
    create_output(par, out; overwrite = cfg.run.overwrite)
    write_layer_table(par, out, "/diagnostics/layer_screening", tab, cols; long_name = "force-budget screening (not a feasibility certificate)")
    write_layer_table(par, out, "/diagnostics/layer_colhse_residual", rtab, rcols; long_name = "column-HSE residual with B = 0 (vertical_hse_only)")
    if S.write_maps
        ok = newmask(par)
        fill_owned!((i, j, k) -> UInt8(D[i, j, k] >= M[i, j, k]), ok, par)
        write_fields(par, out, [FieldOut("/diagnostics/screening_ok", ok, "1", "|δL| ≥ |∇ₕP − Lₕ| (screening)"),
                                FieldOut("/diagnostics/mismatch", M, "N m-3", "|∇ₕP − Lₕ(B_obs)|"),
                                FieldOut("/diagnostics/dL", D, "N m-3", "|B| σ_B/(μ₀ ℓ)")]; compression = cfg.output.compression)
    end
    meta = provenance_meta(par, cfg)
    merge!(meta, Dict{String,Any}("mode" => "screening", "label" => "screening", "colhse_file" => abspath(S.colhse_file),
        "colhse_anchor" => anchor, "ell_m" => ell, "sigma_model" => "per component sqrt((sigma_rel |B|)^2 + floor^2), sigma_B = sqrt(3) sigma",
        "sigma_rel" => cfg.mhs.sigma_rel, "sigma_floor_T" => cfg.mhs.sigma_floor_T, "g" => g))
    write_meta(par, out, meta)
    if rank0(par)
        h5open(out, "r+") do f
            f["diagnostics/z"] = spec.z
        end
        write(splitext(out)[1] * ".md", screening_markdown(spec.z, tab, cols, rtab, rcols, S.colhse_file, anchor, ell, cfg))
    end
    logm("wrote $out")
    return converged
end

function screening_markdown(z, tab, cols, rtab, rcols, colhse, anchor, ell, cfg)
    col(M, c, n) = M[:, findfirst(==(n), c)]
    ks = nearest_levels(z, [3.0e6, 2.5e6, 2.0e6, 1.5e6, 1.0e6, 0.5e6, 0.0, -0.5e6])
    io = IOBuffer()
    println(io, "# M3: force-budget screening and column-HSE residual\n")
    println(io, "- Pressure: `$colhse` (column_hse, anchor $anchor, Wittmann table). B: `$(cfg.input.file)`.")
    @printf(io, "- ℓ = %.1f km; σ per component = sqrt((%.3g |B|)² + (%.3g T)²); σ_B = √3 σ. **Screening only, not a feasibility certificate.**\n\n", ell / 1e3, cfg.mhs.sigma_rel, cfg.mhs.sigma_floor_T)
    println(io, "## Force budget per layer (medians, N m⁻³)\n")
    println(io, "| z [Mm] | \\|∇ₕP\\| | \\|Lₕ(B)\\| | mismatch M | \\|δL\\| | fraction \\|δL\\| ≥ M | M/\\|δL\\| p50 / p90 / p99 | median β |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for k in ks
        @printf(io, "| %.2f | %.3g | %.3g | %.3g | %.3g | %.3f | %.3g / %.3g / %.3g | %.3g |\n", z[k] / 1e6,
                col(tab, cols, "gradhP_p50")[k], col(tab, cols, "Lh_p50")[k], col(tab, cols, "mismatch_p50")[k],
                col(tab, cols, "dL_p50")[k], col(tab, cols, "frac_dL_ge_mismatch")[k], col(tab, cols, "mismatch_over_dL_p50")[k],
                col(tab, cols, "mismatch_over_dL_p90")[k], col(tab, cols, "mismatch_over_dL_p99")[k], col(tab, cols, "beta_p50")[k])
    end
    println(io, "\n## Column-HSE residual with B = 0 (label `vertical_hse_only`)\n")
    println(io, "| z [Mm] | \\|R_h\\|/ρg p50 / p90 / p99 | \\|R_z\\|/ρg p50 / p90 / p99 | \\|R_h\\|/(\\|∇ₕP\\|+ρg) p50 / p99 |")
    println(io, "|---|---|---|---|")
    for k in ks
        @printf(io, "| %.2f | %.3g / %.3g / %.3g | %.2e / %.2e / %.2e | %.3g / %.3g |\n", z[k] / 1e6,
                rtab[k, 1], rtab[k, 2], rtab[k, 3], rtab[k, 4], rtab[k, 5], rtab[k, 6], rtab[k, 7], rtab[k, 9])
    end
    println(io, "\nR_z is the discrete vertical residual of the RK4 solution (truncation level); R_h is the unbalanced horizontal pressure gradient that MHS must supply through Lₕ.")
    println(io, "\nFull per-level tables: `/diagnostics/layer_screening`, `/diagnostics/layer_colhse_residual`.")
    return String(take!(io))
end
