#=
Solver-time measurements for mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md §3): the standalone solve of
scripts/run.jl (column HSE in τ, aligned, then PoissonTau.solve_tau), with the state of the solved levels saved after
every step up to step 40 and every 10th step after that, for accuracy against the step count.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/measure_steps.jl INPUT.nc OUTDIR MU AZIMUTH_DEG GEOMETRY [KEY=VALUE ...]
KEY: max_steps, rtol, n_consec, cg_maxit, cg_tol, logtau_cut, stop (phi | gn), gn_tol, gn_consec, balance (vertical | ray),
lm (alpha | gain | fixed), forcing (fixed | ew), cg_cap, eta_min, eta_max, precond, damping, klow, cap_frac, mlm_min,
lm_mu0, qual_tol, qual_maxit, feas (as the solve_tau keywords), start (truth:TRUTH.h5: the truth ln P below the cut,
shifted per column to the fixed column-HSE cut pressure, or unshifted with truth_raw=1, and the upper structure moved to
the truth cut height; else the directory of an earlier run: start from its final
state, Float64 when saved; the run's ranks, slab, grid, cut, view, balance and input must match), common_weights (1: the weights of the column-HSE start, for any start; defaults: the earlier rule,
stop = phi, max_steps 100), perturb_lnp, perturb_c_km (a smooth |k| = 1 perturbation of the start: ln P ×
exp(−log τ²/2), cut height in km). logtau_cut is on the line-of-sight axis.
OUTDIR/snap_r<rank>.h5: s<step>/{lnP, z} (and final/ at the returned state) (Float32, solved levels k_c … nt, this rank's x slab, Julia (τ, y, x)), with
step attributes phi, pred, cg_its, cg_rel, alpha, seconds (and the work per step: cg_step, solves, evals, cap_scale); s0
holds the evaluated start state (its heights from the start ln P and cut heights); final64/{lnP, z} the returned state in
Float64 (the restart checkpoint); weights/{wc, wf, ereg} the solve's objective; root attributes i0 (first global x), kc,
NX, NY, nt, logtau_cut, balance, input (path, size, Σ log τ), view, work_* (info.work) and qual_* (the last stop check).
=#
using HSEMHS
using HDF5
using MPI
using LinearAlgebra
using Printf

const H = HSEMHS

function main(path, outdir, mu_s, az_s, geom, kv...)
    H.ensure_mpi()
    o = Dict{String,Float64}("max_steps" => 100, "rtol" => 1e-4, "n_consec" => 3, "cg_maxit" => 80, "cg_tol" => 1e-3,
                             "logtau_cut" => -4.0, "gn_tol" => 1e-3, "gn_consec" => 2, "cg_cap" => 200, "klow" => 8, "cap_frac" => 0.5, "mlm_min" => 0.01,
                             "eta_min" => 0.005, "eta_max" => 0.1, "lm_mu0" => 0.0, "qual_tol" => 1e-2,
                             "qual_maxit" => 400, "truth_raw" => 0, "perturb_lnp" => 0.0, "perturb_c_km" => 0.0,
                             "feas" => 1)
    sy = Dict(:stop => :phi, :balance => :vertical, :lm => :alpha, :forcing => :fixed, :precond => :fourier, :damping => :reg)
    start = ""; o["common_weights"] = 0.0
    for a in kv
        k, v = split(a, "=")
        k == "start" && (start = String(v); continue)
        haskey(sy, Symbol(k)) && (sy[Symbol(k)] = Symbol(v); continue)
        haskey(o, k) || error("unknown key $k"); o[k] = parse(Float64, v)
    end
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = Symbol(geom))
    t0 = time()
    NX, NY, nt = H.PoissonTau.tau_input_size(path)
    par = H.slab_par(NX, NY, nt)
    logf(s) = H.rprint(par, @sprintf("[%7.1fs] %s", time() - t0, s))
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = H.PoissonTau.read_tau_input(par, path)
    lT = log.(T); g = 274.0
    lnP, z0, nf, _, _, nbad = H.PoissonTau.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed in $nf columns")
    logf(@sprintf("column HSE done; view μ %.3f φ %.1f %s; %s; options %s%s", v.mu, v.azimuth_deg, v.geometry, sy, o,
                  start == "" ? "" : "; start " * start))
    mkpath(outdir)
    kc = argmin(abs.(lt .- o["logtau_cut"]))
    dx = x[2] - x[1]; dy = y[2] - y[1]
    wts = nothing
    if o["common_weights"] != 0                     # the objective of the column-HSE start (as a cold solve uses)
        _, _, _, _, _, _, i0 = H.PoissonTau.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, copy(lnP), copy(z0), dx, dy;
                                                       logtau_cut = o["logtau_cut"], max_steps = 0, mu = v.mu,
                                                       slant = H.slant(v), balance = sy[:balance], stationarity = false)
        wts = i0.weights
    end
    meta = Dict{String,Any}("i0" => par.block.i0, "kc" => kc, "NX" => NX, "NY" => NY, "nt" => nt, "ranks" => par.nranks,
                            "logtau_cut" => o["logtau_cut"], "balance" => String(sy[:balance]), "mu" => v.mu,
                            "azimuth_deg" => v.azimuth_deg, "geometry" => String(v.geometry),
                            "input" => abspath(path), "input_bytes" => filesize(path), "input_lt_sum" => sum(lt))
    if startswith(start, "truth:")                  # the truth-derived state (design/CODEX_REVIEW_solver_time_round3)
        tf = h5open(start[7:end], "r")
        isapprox(read(tf["logtau500"]), lt; rtol = 1e-6) || error("start: the truth's τ axis differs from the input's")
        gi = H.owned_x(par)
        tu = permutedims(read(tf["truth/lnP"]), (3, 2, 1))[:, :, gi]; tz = permutedims(read(tf["truth/z_m"]), (3, 2, 1))[:, :, gi]
        close(tf)
        all(isfinite, tu[kc:end, :, :]) && all(isfinite, tz[kc, :, :]) || error("start: non-finite truth below the cut")
        du = tu[kc, :, :] .- lnP[kc, :, :]; dc = tz[kc, :, :] .- z0[kc, :, :]
        logf(@sprintf("start: truth %s; at the cut (fixed: column HSE) ln P_truth − ln P_col rms %.4f max %.4f, z_truth − z_col rms %.1f km",
                      start[7:end], sqrt(H.gsum(par, sum(abs2, du)) / (NX * NY)), H.gmax(par, maximum(abs, du)),
                      sqrt(H.gsum(par, sum(abs2, dc)) / (NX * NY)) / 1e3))
        if o["truth_raw"] != 0                      # truth ln P below the cut as it is (a pressure jump at the cut)
            lnP[kc+1:end, :, :] .= tu[kc+1:end, :, :]
            H.PoissonTau.warm_start!(copy(lnP), z0, kc, lnP[kc:end, :, :], tz[kc, :, :])
            logf("start: truth ln P below the cut unshifted; upper structure moved to the truth cut height")
        else                                        # truth columns shifted to the fixed cut pressure (admissible)
            H.PoissonTau.warm_start!(lnP, z0, kc, tu[kc:end, :, :], tz[kc, :, :])
            logf("start: truth ln P below the cut shifted by the cut mismatch per column; upper structure moved to the truth cut height")
        end
    elseif start != ""                              # the final state of an earlier run (same ranks, cut and input)
        fn = joinpath(start, "snap_r$(par.rank).h5")
        msg = ""; su = sz = nothing
        if !isfile(fn)
            msg = "no $fn"
        else
            sf = h5open(fn, "r"); sa = attributes(sf)
            for (k, val) in meta
                if !haskey(sa, k)
                    k in ("i0", "kc") && (msg *= " $k missing;")             # older runs: only i0 and kc were saved
                    continue
                end
                rv = read(sa[k])
                ok = val isa AbstractFloat ? isapprox(rv, val; rtol = 1e-12) : (k == "input" ? true : rv == val)
                ok || (msg *= " $k $(rv) ≠ $(val);")
            end
            g64 = haskey(sf, "final64") ? "final64" : "final"
            su = Float64.(read(sf["$g64/lnP"])); sz = Float64.(read(sf["$g64/z"])); close(sf)
            size(su) == (nt - kc + 1, size(lnP, 2), size(lnP, 3)) && size(sz) == size(su) ||
                (msg *= " state size $(size(su)) ≠ $((nt - kc + 1, size(lnP, 2), size(lnP, 3)));")
            g64 == "final" && logf("start: no Float64 checkpoint in $start, using the Float32 final state")
        end
        H.gcount(par, msg == "" ? 0 : 1) == 0 || error("start: $start does not match this run (rank $(par.rank):$msg)")
        H.PoissonTau.warm_start!(lnP, z0, kc, su, sz[1, :, :])
        logf("start: final state of " * start)
    end
    if o["perturb_lnp"] != 0 || o["perturb_c_km"] != 0  # smooth perturbation: |k| = 1 in x, Gaussian in log τ around 0
        gi = H.owned_x(par); ph = reshape(cos.(2π .* (gi .- 1) ./ NX), 1, :)
        for k in kc+1:nt
            lnP[k, :, :] .+= o["perturb_lnp"] .* exp(-lt[k]^2 / 2) .* ph
        end
        z0[1:kc, :, :] .+= 1e3 * o["perturb_c_km"] .* reshape(ph, 1, 1, :)
        logf(@sprintf("start: perturbed by %.3g in ln P (cos 2πx/L × exp(−log τ²/2)) and %.3g km in the cut height (cos 2πx/L)",
                      o["perturb_lnp"], o["perturb_c_km"]))
    end
    # the evaluated start state: heights below the cut follow from the start ln P (s0 snapshot)
    pr0, c0 = H.PoissonTau.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z0, dx, dy; logtau_cut = o["logtau_cut"], mu = v.mu,
                                    slant = H.slant(v), balance = sy[:balance])
    _, bad0 = H.PoissonTau.evaluate!(pr0, lnP, c0); zs0 = copy(pr0.z)
    bad0 == 0 || logf("start state: $bad0 cells outside the EOS table (s0 heights not evaluated there)")
    if wts === nothing
        H.PoissonTau.tp_weights!(pr0, g, dx)
    else
        pr0.wc = copy(wts.wc); pr0.wf = copy(wts.wf); pr0.ereg = copy(wts.ereg)
    end
    φh0, φv0 = bad0 == 0 ? H.PoissonTau.phi_parts(pr0) : (NaN, NaN); pr0 = nothing
    f = h5open(joinpath(outdir, "snap_r$(par.rank).h5"), "w")
    a = attributes(f); a["i0"] = par.block.i0; a["kc"] = kc; a["mu"] = v.mu; a["azimuth_deg"] = v.azimuth_deg
    a["geometry"] = String(v.geometry); a["ranks"] = par.nranks; a["threads"] = Threads.nthreads()
    for (k, val) in meta; haskey(a, k) || (a[k] = val); end
    start != "" && (a["start"] = abspath(start))
    function save(s, u, z, info; name = "s$s")
        (s <= 40 || s % 10 == 0 || name == "final") || return
        f["$name/lnP"] = Float32.(u[kc:end, :, :]); f["$name/z"] = Float32.(z[kc:end, :, :])
        b = attributes(f[name])
        for (k, val) in pairs(info)
            b[String(k)] = Float64(val)
        end
    end
    save(0, lnP, zs0, (phi = φh0 + φv0, pred = NaN, cg_its = 0, cg_rel = NaN, alpha = NaN, seconds = 0.0, phi_h = φh0,
                       phi_v = φv0))
    BLAS.set_num_threads(min(32, Threads.nthreads()))
    ts = time()
    u, z, Φ0, Φ, steps, _, info = H.PoissonTau.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, dx, dy; weights = wts,
                                     logtau_cut = o["logtau_cut"], max_steps = Int(o["max_steps"]), rtol = o["rtol"],
                                     n_consec = Int(o["n_consec"]), cg_maxit = Int(o["cg_maxit"]), cg_tol = o["cg_tol"],
                                     logf = logf, mu = v.mu, slant = H.slant(v), on_step = save, stop = sy[:stop],
                                     gn_tol = o["gn_tol"], gn_consec = Int(o["gn_consec"]), balance = sy[:balance],
                                     lm = sy[:lm], forcing = sy[:forcing], precond = sy[:precond], damping = sy[:damping], klow = Int(o["klow"]), cap_frac = o["cap_frac"],
                               mlm_min = o["mlm_min"], cg_cap = Int(o["cg_cap"]),
                                     eta_min = o["eta_min"], eta_max = o["eta_max"], lm_mu0 = o["lm_mu0"],
                                     qual_tol = o["qual_tol"], qual_maxit = Int(o["qual_maxit"]), feas = o["feas"] != 0)
    tsolve = time() - ts
    BLAS.set_num_threads(1)
    save(steps, u, z, (phi = Φ, pred = info.gn_decrement, cg_its = info.cg_its, cg_rel = info.cg_rel, alpha = NaN,
                       seconds = tsolve); name = "final")
    f["final64/lnP"] = u[kc:end, :, :]; f["final64/z"] = z[kc:end, :, :]
    W = info.weights
    if W !== nothing
        f["weights/wc"] = W.wc; f["weights/wf"] = W.wf; f["weights/ereg"] = W.ereg
    end
    a["steps"] = steps; a["reason"] = String(info.reason); a["solve_seconds"] = tsolve; a["phi0"] = Φ0
    for (k, val) in pairs(info.work); a["work_$k"] = val; end
    if info.stop_check !== nothing
        q = info.stop_check
        a["qual_outcome"] = String(q.outcome); a["qual_step"] = q.step; a["qual_lower"] = q.lower; a["qual_upper"] = q.upper
    end
    close(f)
    logf(@sprintf("solve: %d steps, %s, %.1f s, Φ/Φ₀ %.3e, GN decrement %.2e", steps, info.reason, tsolve, Φ / Φ0, info.gn_decrement))
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
