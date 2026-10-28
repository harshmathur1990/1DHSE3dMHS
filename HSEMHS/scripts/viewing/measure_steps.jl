#=
Solver-time measurements for mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md §3): the standalone solve of
scripts/run.jl (column HSE in τ, aligned, then PoissonTau.solve_tau), with the state of the solved levels saved after
every step up to step 40 and every 10th step after that, for accuracy against the step count.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/measure_steps.jl INPUT.nc OUTDIR MU AZIMUTH_DEG GEOMETRY [KEY=VALUE ...]
KEY: max_steps, rtol, n_consec, cg_maxit, cg_tol, logtau_cut, stop (phi | gn), gn_tol, gn_consec, balance (vertical | ray),
lm (alpha | gain), forcing (fixed | ew), cg_cap, eta_min, eta_max (as the solve_tau keywords; defaults: the earlier rule,
stop = phi, max_steps 100). logtau_cut is on the line-of-sight axis.
OUTDIR/snap_r<rank>.h5: s<step>/{lnP, z} (and final/ at the returned state) (Float32, solved levels k_c … nt, this rank's x slab, Julia (τ, y, x)), with
step attributes phi, pred, cg_its, cg_rel, alpha, seconds; root attributes i0 (first global x), kc, view.
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
                             "logtau_cut" => -4.0, "gn_tol" => 1e-3, "gn_consec" => 2, "cg_cap" => 200,
                             "eta_min" => 0.005, "eta_max" => 0.1)
    sy = Dict(:stop => :phi, :balance => :vertical, :lm => :alpha, :forcing => :fixed, :precond => :fourier)
    for a in kv
        k, v = split(a, "=")
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
    logf(@sprintf("column HSE done; view μ %.3f φ %.1f %s; %s; options %s", v.mu, v.azimuth_deg, v.geometry, sy, o))
    mkpath(outdir)
    kc = argmin(abs.(lt .- o["logtau_cut"]))
    f = h5open(joinpath(outdir, "snap_r$(par.rank).h5"), "w")
    a = attributes(f); a["i0"] = par.block.i0; a["kc"] = kc; a["mu"] = v.mu; a["azimuth_deg"] = v.azimuth_deg
    a["geometry"] = String(v.geometry); a["ranks"] = par.nranks; a["threads"] = Threads.nthreads()
    function save(s, u, z, info; name = "s$s")
        (s <= 40 || s % 10 == 0 || name == "final") || return
        f["$name/lnP"] = Float32.(u[kc:end, :, :]); f["$name/z"] = Float32.(z[kc:end, :, :])
        b = attributes(f[name])
        for (k, val) in pairs(info)
            b[String(k)] = Float64(val)
        end
    end
    save(0, lnP, z0, (phi = NaN, pred = NaN, cg_its = 0, cg_rel = NaN, alpha = NaN, seconds = 0.0))
    BLAS.set_num_threads(min(32, Threads.nthreads()))
    ts = time()
    u, z, Φ0, Φ, steps, _, info = H.PoissonTau.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, x[2] - x[1], y[2] - y[1];
                                     logtau_cut = o["logtau_cut"], max_steps = Int(o["max_steps"]), rtol = o["rtol"],
                                     n_consec = Int(o["n_consec"]), cg_maxit = Int(o["cg_maxit"]), cg_tol = o["cg_tol"],
                                     logf = logf, mu = v.mu, slant = H.slant(v), on_step = save, stop = sy[:stop],
                                     gn_tol = o["gn_tol"], gn_consec = Int(o["gn_consec"]), balance = sy[:balance],
                                     lm = sy[:lm], forcing = sy[:forcing], precond = sy[:precond], cg_cap = Int(o["cg_cap"]),
                                     eta_min = o["eta_min"], eta_max = o["eta_max"])
    tsolve = time() - ts
    BLAS.set_num_threads(1)
    save(steps, u, z, (phi = Φ, pred = info.gn_decrement, cg_its = info.cg_its, cg_rel = info.cg_rel, alpha = NaN,
                       seconds = tsolve); name = "final")
    a["steps"] = steps; a["reason"] = String(info.reason); a["solve_seconds"] = tsolve; a["phi0"] = Φ0
    close(f)
    logf(@sprintf("solve: %d steps, %s, %.1f s, Φ/Φ₀ %.3e, GN decrement %.2e", steps, info.reason, tsolve, Φ / Φ0, info.gn_decrement))
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
