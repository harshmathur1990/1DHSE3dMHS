#=
Warm start of mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md D4) against a cold start, after an inversion-sized
change of the temperature.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/measure_warm.jl INPUT.nc OUTDIR MU AZIMUTH_DEG GEOMETRY EPS [KEY=VALUE ...]
1. Solve INPUT cold (`first_steps`, default 25).
2. Perturb ln T by a smooth random field of rms EPS (30 modes, horizontal wavelengths 4–32 cells, vertical 1–4 dex in
   log τ; B unchanged).
3. Solve the perturbed cube cold for `ref_steps` (default 40): the reference, and the cold trajectory.
4. Solve it warm from the solution of 1 (PoissonTau.warm_start!) for `max_steps` (default 25).
For every step of 3 and 4: rms and max distance of z(τ_LOS = 1) (mean removed) and rms ln P distance on the solved
levels from the reference's final state, the predicted decrease, CG iterations and cumulative seconds → stdout and
OUTDIR/warm.csv. KEY: as measure_steps.jl (max_steps, logtau_cut (line of sight), stop, gn_tol, gn_consec, balance, lm,
forcing, cg_cap, eta_min, eta_max, cg_maxit, cg_tol) plus first_steps, ref_steps, seed.
=#
using HSEMHS
using MPI
using LinearAlgebra
using Printf
using Random

const H = HSEMHS

function perturbation(par, lt, NX, NY, nx, eps, seed)
    rng = MersenneTwister(seed)
    modes = NTuple{6,Float64}[]
    while length(modes) < 30
        mx, my = rand(rng, -126:126), rand(rng, -126:126)
        λ = 1 / sqrt((mx / NX)^2 + (my / NY)^2)            # cells
        4 <= λ <= 32 || continue
        push!(modes, (mx, my, 2π * rand(rng), 1 + 3 * rand(rng), 2π * rand(rng), 0.0))
    end
    d = zeros(length(lt), NY, nx); i0 = par.block.i0
    Threads.@threads for i in 1:nx
        for j in 1:NY, k in eachindex(lt), (mx, my, φ, λv, ψ, _) in modes
            d[k, j, i] += cos(2π * (mx * (i0 + i - 2) / NX + my * (j - 1) / NY) + φ) * cos(2π * lt[k] / λv + ψ)
        end
    end
    return d .* (eps / sqrt(length(modes) / 4))
end

function main(path, outdir, mu_s, az_s, geom, eps_s, kv...)
    H.ensure_mpi()
    o = Dict{String,Float64}("max_steps" => 25, "first_steps" => 25, "ref_steps" => 40, "logtau_cut" => -4.0,
                             "gn_tol" => 1e-3, "gn_consec" => 2, "cg_cap" => 200, "eta_min" => 0.005, "eta_max" => 0.1,
                             "cg_maxit" => 80, "cg_tol" => 1e-3, "seed" => 1)
    sy = Dict(:stop => :gn, :balance => :ray, :lm => :gain, :forcing => :ew)
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
    quiet(s) = nothing
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = H.PoissonTau.read_tau_input(par, path)
    lT = log.(T); g = 274.0; dx = x[2] - x[1]; dy = y[2] - y[1]
    kc = argmin(abs.(lt .- o["logtau_cut"])); k0 = argmin(abs.(lt)); N = NX * NY
    solve(lTs, u, z0, nsteps, on_step) =
        H.PoissonTau.solve_tau(par, tab, g, lt, lTs, Bx, By, Bz, u, z0, dx, dy; logtau_cut = o["logtau_cut"],
                               max_steps = Int(nsteps), cg_maxit = Int(o["cg_maxit"]), cg_tol = o["cg_tol"], logf = quiet,
                               mu = v.mu, slant = H.slant(v), on_step = on_step, stop = sy[:stop], gn_tol = o["gn_tol"],
                               gn_consec = Int(o["gn_consec"]), balance = sy[:balance], lm = sy[:lm],
                               forcing = sy[:forcing], cg_cap = Int(o["cg_cap"]), eta_min = o["eta_min"],
                               eta_max = o["eta_max"], stationarity = false)
    column(lTs) = (r = H.PoissonTau.column_hse_tau(par, lt, lTs, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu);
                   r[6] == 0 || error("column HSE failed"); (r[1], r[2]))
    z1(z) = (a = z[k0, :, :]; a .- H.gsum(par, sum(a)) / N)

    BLAS.set_num_threads(min(32, Threads.nthreads()))
    u0, z0 = column(lT)
    ts = time()
    ua, za, _, _, sa, _, ia = solve(lT, u0, z0, o["first_steps"], nothing)
    H.PoissonTau.gauge!(par, za, lt)
    logf(@sprintf("first solve: %d steps, %s, %.0f s", sa, ia.reason, time() - ts))

    lT2 = lT .+ perturbation(par, lt, NX, NY, size(lT, 3), parse(Float64, eps_s), Int(o["seed"]))
    u1, z1c = column(lT2)
    rec(tr) = (s, u, z, info) -> push!(tr, (s, z1(z), Float32.(u[kc:end, :, :]), info.pred, info.cg_its, info.seconds))
    cold = Any[(0, z1(z1c), Float32.(u1[kc:end, :, :]), NaN, 0, 0.0)]
    ts = time()
    ur, zr, _, _, sr, _, ir = solve(lT2, copy(u1), copy(z1c), o["ref_steps"], rec(cold))
    logf(@sprintf("reference (cold) solve of the perturbed cube: %d steps, %s, %.0f s", sr, ir.reason, time() - ts))
    uw, zw = copy(u1), copy(z1c)
    H.PoissonTau.warm_start!(uw, zw, kc, ua[kc:end, :, :], za[kc, :, :])
    warm = Any[(0, z1(zw), Float32.(uw[kc:end, :, :]), NaN, 0, 0.0)]
    ts = time()
    _, _, _, _, sw, _, iw = solve(lT2, uw, zw, o["max_steps"], rec(warm))
    logf(@sprintf("warm solve of the perturbed cube: %d steps, %s, %.0f s", sw, iw.reason, time() - ts))
    BLAS.set_num_threads(1)

    zref = z1(zr); uref = ur[kc:end, :, :]
    lines = String["start,step,cum_s,z_rms_km,z_max_km,lnP_rms,pred,cg_its"]
    for (name, tr) in (("cold", cold), ("warm", warm))
        cum = 0.0
        for (s, zz, uu, pred, its, sec) in tr
            cum += sec
            dz = zz .- zref
            zr_ = sqrt(H.gsum(par, sum(abs2, dz)) / N) / 1e3; zm = H.gmax(par, maximum(abs, dz)) / 1e3
            du = sqrt(H.gsum(par, sum(abs2, Float64.(uu) .- uref)) / (N * size(uref, 1)))
            push!(lines, @sprintf("%s,%d,%.1f,%.3f,%.2f,%.5f,%.3e,%d", name, s, cum, zr_, zm, du, pred, its))
            logf(@sprintf("%s step %2d: %7.0f s | z(τ=1) from the reference %6.2f km rms, %6.1f km max | ln P %.5f | pred %.2e | CG %d",
                          name, s, cum, zr_, zm, du, pred, its))
        end
    end
    if par.rank == 0
        mkpath(outdir)
        write(joinpath(outdir, "warm.csv"), join(lines, "\n") * "\n")
    end
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
