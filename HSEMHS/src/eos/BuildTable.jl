# Gate 1 driver: coverage scan, T_sw, table build, validation, save, report (doc §7).

function read_log_abund(spec::AbstractString)
    spec == "wittmann_default" && return nothing
    v = parse.(Float64, split(read(spec, String)))
    length(v) == 99 || invalid("abundance file $spec must contain 99 log10 fractions")
    return v
end

"""
    build_eos_table(cfg) -> EOSTable

run.mode = "eos_table": builds `eos.table_file` and writes the Gate 1 table report to
`run.output` (.h5) and the same stem with `.md`.
"""
function build_eos_table(cfg::Config)
    E = cfg.eos
    t0 = time()
    logm(msg) = (@printf("[%7.1fs] eos_table: %s\n", time() - t0, msg); flush(stdout))
    la = read_log_abund(E.abundances)
    m0 = EOSModel(E.pf_file; prec = E.prec, maxit = E.maxit, log_abund = la)
    witt_fix_gasc_h2() || @warn "libwitt built without FIX_GASC_H2: gasc and pe_pg disagree where H2 matters"
    env = envelope_from_gate0(E.envelope_from; logP_range = E.logP_range)
    logm(@sprintf("envelope T = %.4g .. %.4g K, log10 P = %.2f .. %.2f (%s)", env.T_lo, env.T_hi,
                  env.log10P_lo, env.log10P_hi, env.source))
    scan = coverage_scan(m0, env)
    ncf = count(.!scan.conv)
    logm("coverage scan: $(ncf) of $(length(scan.conv)) direct calls not converged; fully-ionized test $(scan.full_ionization_ok)")
    T_sw, hb, rep = select_T_sw(m0, scan)
    logm(rep)
    hb == "rejected" && invalid("no admissible T_sw: " * rep)
    # cold end: report non-convergence below T_sw (handled by the robust fallback, flagged)
    cold = [(10.0^scan.lT[a], 10.0^scan.lP[b]) for a in eachindex(scan.lT), b in eachindex(scan.lP)
            if !scan.conv[a, b] && 10.0^scan.lT[a] < T_sw]
    m = with_switch(m0, T_sw, hb)
    tab = build_table(m, env; dlog10T = E.dlog10T, dlog10P = E.dlog10P, refine_max_levels = E.refine_max_levels)
    nfailed = count(f -> f & EOS_DIRECT_FAILED != 0, tab.flags)
    nfallback = count(f -> f & EOS_ROBUST_FALLBACK != 0, tab.flags)
    logm("table $(length(tab.X)) × $(length(tab.Y)) nodes; failed nodes $nfailed; robust-fallback nodes $nfallback")
    val = validate_table(tab, m; n = E.validate_points)
    logm("withheld validation: " * join(["$k = $(@sprintf("%.2e", v))" for (k, v) in sort(collect(val.stats))], ", "))
    der = derivative_check(tab, m; n = 10^4)
    logm(@sprintf("∂lnρ/∂lnP check: max %.2e, p99.9 %.2e over %d points", der.max, der.p999, der.n))
    meta = Dict{String,Any}(
        "prec" => E.prec, "maxit" => E.maxit, "abundances" => E.abundances,
        "pf_file" => abspath(E.pf_file), "pf_sha256" => file_sha256(E.pf_file),
        "shim_sha256" => file_sha256(joinpath(pkgdir(@__MODULE__), "deps", "witt", "witt_shim.cpp")),
        "source_sha256" => file_sha256(joinpath(pkgdir(@__MODULE__), "deps", "witt", "witt_eos_cpp.cpp")),
        "compiler" => read(joinpath(pkgdir(@__MODULE__), "deps", "witt", "compiler.txt"), String),
        "fix_gasc_h2" => witt_fix_gasc_h2(), "T_sw" => T_sw, "hot_branch" => hb,
        "envelope_T" => [env.T_lo, env.T_hi], "envelope_log10P" => [env.log10P_lo, env.log10P_hi],
        "envelope_source" => env.source, "dlog10T" => E.dlog10T, "dlog10P" => E.dlog10P,
        "refinement" => join(tab.meta["refinement"], "\n"), "breaks_T" => tab.meta["breaks_T"],
        "created" => string(Dates.now()))
    save_table(E.table_file, tab, meta)
    logm("saved $(E.table_file) (sha256 $(table_sha256(tab)))")
    # ---- report ----
    out = cfg.run.output
    mkpath(dirname(abspath(out)))
    h5open(out, "w") do f
        f["scan/log10T"] = scan.lT; f["scan/log10P"] = scan.lP
        f["scan/direct_converged"] = UInt8.(scan.conv); f["scan/closure_residual"] = scan.closure
        f["validation/maxerr_maps"] = val.maps
        write_attribute(f["validation/maxerr_maps"], "fields", collect(LN_FIELDS))
        for (k, v) in val.stats
            write_attribute(f["validation"], k, v)
        end
    end
    pass = (val.stats["lnrho_max"] <= 1e-5, val.stats["lnne_max"] <= 1e-4, val.stats["lnchi_max"] <= 1e-4,
            der.max <= 1e-3, nfailed == 0, val.nfail == 0)
    io = IOBuffer()
    println(io, "# Gate 1: EOS table report\n")
    println(io, "- Table: `$(E.table_file)`, $(length(tab.X)) × $(length(tab.Y)) nodes, sha256 `$(table_sha256(tab))`")
    @printf(io, "- Envelope: T = %.4g – %.4g K, log₁₀P = %.2f – %.2f Pa (%s)\n", env.T_lo, env.T_hi, env.log10P_lo, env.log10P_hi, env.source)
    println(io, "- Direct solver: prec = $(E.prec), maxit = $(E.maxit), FIX_GASC_H2 = $(witt_fix_gasc_h2())")
    println(io, "- Coverage scan (400 × 200): $(ncf) direct calls did not converge; fully-ionized test: $(scan.full_ionization_ok)")
    println(io, "- Hot branch: `$hb`; $rep")
    println(io, "- Below T_sw, $(length(cold)) scan points needed the robust fallback; $(nfallback) table nodes are flagged EOS_ROBUST_FALLBACK")
    println(io, "- Breakpoints inserted: $(length(tab.meta["breaks_T"])) temperatures (partition-function grid and opacity switches)\n")
    println(io, "## Refinement\n")
    for r in tab.meta["refinement"]
        println(io, "- ", r)
    end
    println(io, "\n## Gate 1 table rows\n")
    println(io, "| Check | Value | Threshold | Pass |\n|---|---|---|---|")
    @printf(io, "| max \\|Δ ln ρ\\| (%d withheld points) | %.2e | 1e-5 | %s |\n", val.n, val.stats["lnrho_max"], pass[1])
    @printf(io, "| max \\|Δ ln nₑ\\| | %.2e | 1e-4 | %s |\n", val.stats["lnne_max"], pass[2])
    @printf(io, "| max \\|Δ ln χ₅₀₀\\| | %.2e | 1e-4 | %s |\n", val.stats["lnchi_max"], pass[3])
    @printf(io, "| p99.9 \\|Δ\\| (ρ / nₑ / χ) | %.1e / %.1e / %.1e | — | — |\n", val.stats["lnrho_p999"], val.stats["lnne_p999"], val.stats["lnchi_p999"])
    @printf(io, "| ∂ln ρ/∂ln P vs FD sweep (%d points) | max %.2e | 1e-3 abs | %s |\n", der.n, der.max, pass[4])
    @printf(io, "| table nodes from failed model calls | %d | 0 | %s |\n", nfailed, pass[5])
    @printf(io, "| model failures at withheld points | %d | 0 | %s |\n", val.nfail, pass[6])
    println(io, "\nWorst points (T [K], P [Pa]): ", join(["$k: $(round.(v; sigdigits=4))" for (k, v) in val.worst], "; "))
    write(splitext(out)[1] * ".md", String(take!(io)))
    logm("report written to $(splitext(out)[1]).md; all table rows pass = $(all(pass))")
    return tab
end
