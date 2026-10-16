# Gate 0 input audit (spec §3, implementation doc §8.2).
#
# Proxy only (no EOS): column HSE with rho/P frozen from the snapshot. Derivatives are
# computed at full resolution; statistics use the configured horizontal subsampling.

"""Gravity [m s^-2]: number, or "from_input" (attribute gravity_m_s2, else Bifrost FITS cards)."""
function input_gravity(cfgval, spec::InputSpec)
    cfgval isa Real && return Float64(cfgval)
    cfgval == "from_input" || invalid("physics.g must be a number or \"from_input\"")
    haskey(spec.attrs, "gravity_m_s2") && return Float64(spec.attrs["gravity_m_s2"])
    h5open(spec.path, "r") do f
        ta = attributes(f[spec.tname])
        if haskey(ta, "fits_header")
            hdr = attr_string(read(ta["fits_header"]))
            gr, ul, ut = fits_card(hdr, "GRAV"), fits_card(hdr, "U_L"), fits_card(hdr, "U_T")
            if gr !== nothing && ul !== nothing && ut !== nothing
                return -parse(Float64, gr) * parse(Float64, ul) / parse(Float64, ut)^2 / 100.0
            end
        end
        invalid("physics.g = \"from_input\" but the input has no gravity_m_s2 attribute or GRAV/U_L/U_T cards")
    end
end

"""Integrate d ln P/dz = -g rho/P (frozen ratio) by the trapezoid rule, per owned column."""
function proxy_column_hse!(utop::Field, ubot::Field, P::Field, rho::Field, z, g, Ptop, par)
    b = par.block
    NZ = b.nz
    Threads.@threads :dynamic for c in 1:(b.nx*b.ny)
        i = (c - 1) % b.nx + 1
        j = (c - 1) ÷ b.nx + 1
        @inbounds begin
            utop[i, j, 1] = log(Ptop)
            for k in 1:NZ-1
                r0 = rho[i, j, k] / P[i, j, k]; r1 = rho[i, j, k+1] / P[i, j, k+1]
                utop[i, j, k+1] = utop[i, j, k] + 0.5 * g * (r0 + r1) * (z[k] - z[k+1])
            end
            ubot[i, j, NZ] = log(P[i, j, NZ])
            for k in NZ-1:-1:1
                r0 = rho[i, j, k] / P[i, j, k]; r1 = rho[i, j, k+1] / P[i, j, k+1]
                ubot[i, j, k] = ubot[i, j, k+1] - 0.5 * g * (r0 + r1) * (z[k] - z[k+1])
            end
        end
    end
end

const GATE0_P3 = [0.01, 0.5, 0.99]
const GATE0_P_RES = [0.5, 0.9, 0.99]

"""
    gate0_audit(cfg) -> Dict with all per-layer tables

Writes `run.output` (HDF5) and a markdown summary next to it (same stem, `.md`).
"""
function gate0_audit(cfg::Config)
    G = cfg.gate0
    I = cfg.input
    spec = open_input(I.file; crop_z_m = G.crop_z_m, tname = I.temperature, xname = I.x,
                      yname = I.y, zname = I.z)
    NX, NY, NZ = length(spec.x), length(spec.y), length(spec.z)
    par = init_parallel(NX, NY, NZ; px = cfg.parallel.px, py = cfg.parallel.py, debug = cfg.run.debug)
    t0 = time()
    logm(msg) = rank0(par) && (@printf("[%7.1fs] gate0: %s\n", time() - t0, msg); flush(stdout))
    grid = build_grid(spec)
    ops = Ops(grid, par)
    g = input_gravity(cfg.physics.g, spec)
    z = spec.z
    stride = G.stride
    logm("grid $(NX)×$(NY)×$(NZ), z = $(z[1]/1e6) .. $(z[end]/1e6) Mm, g = $g, ranks $(par.nranks) ($(par.px)×$(par.py)), threads $(Threads.nthreads())")

    T = read_field(spec, par, I.temperature, :temperature; positive = true)
    P = read_field(spec, par, G.pressure, :pressure; positive = true)
    rho = read_field(spec, par, G.density, :density; positive = true)
    Bx = read_field(spec, par, I.bx, :field)
    By = read_field(spec, par, I.by, :field)
    Bz = read_field(spec, par, I.bz, :field)
    vx = read_field(spec, par, G.vx, :velocity)
    vy = read_field(spec, par, G.vy, :velocity)
    vz = read_field(spec, par, G.vz, :velocity)
    exchange_halos!(par, [P, Bx, By, Bz])
    logm("read inputs")

    absB(i, j, k) = @inbounds sqrt(Bx[i, j, k]^2 + By[i, j, k]^2 + Bz[i, j, k]^2)
    beta_of(Pf) = (i, j, k) -> @inbounds 2MU0 * Pf(i, j, k) / max(absB(i, j, k), ONE_GAUSS)^2
    q(val, ps) = layer_quantiles(par, val, ps; stride = stride)
    frac(pred) = layer_fraction(par, pred; stride = stride)

    # ---- item 3: proxy anchors ----
    utop = newfield(par); ubot = newfield(par)
    proxy_column_hse!(utop, ubot, P, rho, z, g, G.P_top, par)
    logm("proxy column HSE done")

    tables = Dict{String,Tuple{Matrix{Float64},Vector{String}}}()

    # ---- item 2: thermodynamics per layer ----
    Ptopf(i, j, k) = @inbounds exp(utop[i, j, k])
    Pbotf(i, j, k) = @inbounds exp(ubot[i, j, k])
    Pnat(i, j, k) = @inbounds P[i, j, k]
    th = hcat(q(T, [0.0, 0.5, 1.0]), q(absB, [0.0, 0.5, 1.0]),
              q((i, j, k) -> log10(P[i, j, k]), GATE0_P3),
              q(beta_of(Pnat), [0.0, 0.5, 1.0]),
              q(beta_of(Ptopf), [0.5]), q(beta_of(Pbotf), [0.5]),
              frac((i, j, k) -> T[i, j, k] > 3.0e4),
              q((i, j, k) -> rho[i, j, k] * (vx[i, j, k]^2 + vy[i, j, k]^2 + vz[i, j, k]^2) / P[i, j, k], [0.5]))
    thcols = ["T_min", "T_p50", "T_max", "absB_min", "absB_p50", "absB_max",
              "log10P_p1", "log10P_p50", "log10P_p99", "beta_native_min", "beta_native_p50",
              "beta_native_max", "beta_topanchor_p50", "beta_botanchor_p50", "frac_T_gt_3e4",
              "rhov2_over_P_p50"]
    tables["thermo"] = (th, thcols)
    logm("item 2 done")

    an = hcat(q((i, j, k) -> (utop[i, j, k] - log(P[i, j, k])) / log(10), GATE0_P3),
              q((i, j, k) -> (ubot[i, j, k] - log(P[i, j, k])) / log(10), GATE0_P3),
              q((i, j, k) -> utop[i, j, k] / log(10), GATE0_P3),
              q((i, j, k) -> ubot[i, j, k] / log(10), GATE0_P3))
    ancols = ["dex_top_p1", "dex_top_p50", "dex_top_p99", "dex_bot_p1", "dex_bot_p50", "dex_bot_p99",
              "log10P_toprun_p1", "log10P_toprun_p50", "log10P_toprun_p99",
              "log10P_botrun_p1", "log10P_botrun_p50", "log10P_botrun_p99"]
    tables["anchor"] = (an, ancols)
    logm("item 3 done")

    # ---- item 4: native residual ----
    Rx = newfield(par); Ry = newfield(par); Rz = newfield(par); den = newfield(par)
    b = par.block
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            Lx, Ly, Lz = lorentz(Bx, By, Bz, ops, i, j, k)
            gx = dx(P, ops.X, i, j, k); gy = dy(P, ops.Y, i, j, k); gz = dz(P, ops.Z, i, j, k)
            rg = rho[i, j, k] * g
            Rx[i, j, k] = gx - Lx
            Ry[i, j, k] = gy - Ly
            Rz[i, j, k] = gz + rg - Lz
            den[i, j, k] = sqrt(gx^2 + gy^2 + gz^2) + rg + sqrt(Lx^2 + Ly^2 + Lz^2)
        end
    end
    vres(i, j, k) = begin
        lnP(kk) = log(P[i, j, kk])
        d = ops.Z.c1[k, 1] * lnP(ops.Z.idx[k, 1]) + ops.Z.c1[k, 2] * lnP(ops.Z.idx[k, 2]) +
            ops.Z.c1[k, 3] * lnP(ops.Z.idx[k, 3])
        s = rho[i, j, k] * g / P[i, j, k]
        abs(d + s) / s
    end
    rr = hcat(q((i, j, k) -> abs(Rx[i, j, k]) / (rho[i, j, k] * g), GATE0_P_RES),
              q((i, j, k) -> abs(Ry[i, j, k]) / (rho[i, j, k] * g), GATE0_P_RES),
              q((i, j, k) -> abs(Rz[i, j, k]) / (rho[i, j, k] * g), GATE0_P_RES),
              q((i, j, k) -> sqrt(Rx[i, j, k]^2 + Ry[i, j, k]^2 + Rz[i, j, k]^2) / den[i, j, k], GATE0_P_RES),
              q(vres, GATE0_P_RES))
    rrcols = ["Rx_rhog_p50", "Rx_rhog_p90", "Rx_rhog_p99", "Ry_rhog_p50", "Ry_rhog_p90", "Ry_rhog_p99",
              "Rz_rhog_p50", "Rz_rhog_p90", "Rz_rhog_p99", "R_scale_p50", "R_scale_p90", "R_scale_p99",
              "vertical_only_p50", "vertical_only_p90", "vertical_only_p99"]
    tables["native_residual"] = (rr, rrcols)
    logm("item 4 done")

    # ---- item 5: force-budget screening (proxy, bottom-anchored P) ----
    utop_saved = copy(utop)
    Pb = utop   # reuse storage: Pb = exp(ubot) with halos
    fill_owned!((i, j, k) -> exp(ubot[i, j, k]), Pb, par)
    exchange_halos!(par, [Pb])
    ell = G.ell_factor * (spec.x[2] - spec.x[1])
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            Lx, Ly, _ = lorentz(Bx, By, Bz, ops, i, j, k)
            gx = dx(Pb, ops.X, i, j, k); gy = dy(Pb, ops.Y, i, j, k)
            Bm = sqrt(Bx[i, j, k]^2 + By[i, j, k]^2 + Bz[i, j, k]^2)
            sig = sqrt((G.sigma_rel * Bm)^2 + G.sigma_floor_T^2)       # per component
            sigB = sqrt(3.0) * sig                                       # E10
            Rx[i, j, k] = sqrt(gx^2 + gy^2)
            Ry[i, j, k] = sqrt(Lx^2 + Ly^2)
            Rz[i, j, k] = sqrt((gx - Lx)^2 + (gy - Ly)^2)
            den[i, j, k] = Bm * sigB / (MU0 * ell)
        end
    end
    sc = hcat(q(Rx, [0.5]), q(Ry, [0.5]), q(Rz, [0.5]), q(den, [0.5]),
              frac((i, j, k) -> den[i, j, k] >= Rz[i, j, k]))
    sccols = ["gradhP_p50", "Lh_p50", "mismatch_p50", "dL_p50", "frac_dL_ge_mismatch"]
    tables["screening"] = (sc, sccols)
    logm("item 5 done")

    # ---- write ----
    out = cfg.run.output
    create_output(par, out; overwrite = cfg.run.overwrite)
    for (name, (M, cols)) in tables
        write_layer_table(par, out, "/diagnostics/layer_$(name)", M, cols)
    end
    meta = provenance_meta(par, cfg)
    meta["mode"] = "gate0"
    meta["label"] = "baseline / screening (proxy, frozen rho/P)"
    meta["g"] = g
    meta["stride"] = stride
    # full-resolution extrema for the EOS envelope (implementation doc §7.3)
    fmin(f) = global_min(par, minimum(f[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
    fmax(f) = global_max(par, maximum(f[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
    meta["envelope_inputs"] = Dict{String,Any}(
        "T_min" => fmin(T), "T_max" => fmax(T),
        "log10P_toprun_min" => fmin(utop_saved) / log(10), "log10P_toprun_max" => fmax(utop_saved) / log(10),
        "log10P_botrun_min" => fmin(ubot) / log(10), "log10P_botrun_max" => fmax(ubot) / log(10),
        "log10P_native_min" => log10(fmin(P)), "log10P_native_max" => log10(fmax(P)),
        "note" => "full resolution (no subsampling); proxy runs use frozen rho/P")
    meta["input"] = Dict{String,Any}("file" => spec.path, "NX" => NX, "NY" => NY, "NZ" => NZ,
        "dx" => spec.x[2] - spec.x[1], "dz_min" => minimum(abs.(diff(z))),
        "dz_max" => maximum(abs.(diff(z))), "periodic_x" => spec.periodic[1],
        "periodic_y" => spec.periodic[2], "crop_source_levels" => [first(spec.krange), last(spec.krange)],
        "units" => "converted to SI from dataset units attributes", "frame" => "local Cartesian, z up")
    write_meta(par, out, meta)
    if rank0(par)
        h5open(out, "r+") do f
            f["diagnostics/z"] = z
            write_attribute(f["diagnostics/z"], "units", "m")
        end
        write(splitext(out)[1] * ".md", gate0_markdown(spec, tables, G, g, stride, par))
    end
    logm("wrote $out")
    return tables
end

function gate0_markdown(spec, tables, G, g, stride, par)
    z = spec.z
    ks = nearest_levels(z, G.target_heights_m)
    th, thc = tables["thermo"]; an, anc = tables["anchor"]
    rr, rrc = tables["native_residual"]; sc, scc = tables["screening"]
    col(M, cols, name) = M[:, findfirst(==(name), cols)]
    io = IOBuffer()
    println(io, "# Gate 0 audit\n")
    println(io, "- Input: `$(spec.path)`")
    println(io, "- Grid: $(length(spec.x))×$(length(spec.y))×$(length(z)); z = $(round(z[1]/1e6, digits=4)) … $(round(z[end]/1e6, digits=4)) Mm (top first); Δx = $(round((spec.x[2]-spec.x[1])/1e3, digits=2)) km; Δz = $(round(minimum(abs.(diff(z)))/1e3, digits=2))–$(round(maximum(abs.(diff(z)))/1e3, digits=2)) km")
    println(io, "- Periodic x, y: $(spec.periodic); g = $(round(g, digits=3)) m s⁻²; statistics on every $(stride)th column in x and y")
    println(io, "- MPI ranks $(par.nranks) ($(par.px)×$(par.py)), threads $(Threads.nthreads())\n")
    println(io, "## Native thermodynamics\n")
    println(io, "| z [Mm] | log₁₀P p1 / p50 / p99 [Pa] | spread [dex] | median β | median ρv²/P | fraction T > 3×10⁴ K |")
    println(io, "|---|---|---|---|---|---|")
    for k in ks
        p1, p50, p99 = col(th, thc, "log10P_p1")[k], col(th, thc, "log10P_p50")[k], col(th, thc, "log10P_p99")[k]
        @printf(io, "| %.2f | %.2f / %.2f / %.2f | %.2f | %.3g | %.2f | %.3f |\n", z[k] / 1e6, p1, p50, p99,
                p99 - p1, col(th, thc, "beta_native_p50")[k], col(th, thc, "rhov2_over_P_p50")[k],
                col(th, thc, "frac_T_gt_3e4")[k])
    end
    println(io, "\n## Anchor experiment (proxy; log₁₀ P_hse/P_native, p1 / p50 / p99)\n")
    println(io, "| z [Mm] | top anchor $(G.P_top) Pa | bottom anchor (native P per column) |")
    println(io, "|---|---|---|")
    for k in ks
        @printf(io, "| %.2f | %+.2f / %+.2f / %+.2f | %+.2f / %+.2f / %+.2f |\n", z[k] / 1e6,
                an[k, 1], an[k, 2], an[k, 3], an[k, 4], an[k, 5], an[k, 6])
    end
    println(io, "\n## Native residual (baseline, not a floor)\n")
    println(io, "| z [Mm] | vertical-only p50 / p90 | \\|R\\|/(\\|∇P\\|+ρg+\\|L\\|) p50 | \\|R_x\\|/ρg p50 | \\|R_z\\|/ρg p50 |")
    println(io, "|---|---|---|---|---|")
    for k in ks
        @printf(io, "| %.2f | %.3g / %.3g | %.3g | %.3g | %.3g |\n", z[k] / 1e6,
                col(rr, rrc, "vertical_only_p50")[k], col(rr, rrc, "vertical_only_p90")[k],
                col(rr, rrc, "R_scale_p50")[k], col(rr, rrc, "Rx_rhog_p50")[k], col(rr, rrc, "Rz_rhog_p50")[k])
    end
    println(io, "\n## Force-budget screening (proxy P, bottom anchor; screening only)\n")
    println(io, "| z [Mm] | \\|∇ₕP\\| p50 | \\|Lₕ\\| p50 | mismatch p50 | \\|δL\\| p50 | fraction \\|δL\\| ≥ mismatch |")
    println(io, "|---|---|---|---|---|---|")
    for k in ks
        @printf(io, "| %.2f | %.3g | %.3g | %.3g | %.3g | %.3f |\n", z[k] / 1e6, sc[k, 1], sc[k, 2], sc[k, 3], sc[k, 4], sc[k, 5])
    end
    println(io, "\nτ = 1 height maps: pending M2 (need the EOS/opacity table).")
    println(io, "\nFull per-level tables: `/diagnostics/layer_*` in the HDF5 report.")
    return String(take!(io))
end
