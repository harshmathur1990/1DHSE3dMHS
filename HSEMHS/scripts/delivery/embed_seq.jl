#=
Delivery check of the embedding API (design/CODEX_REVIEW_solver_time_round6_2026-10-29.md, step 5): one cold/warm
sequence of mhs-poisson-ltau with the selected options, through prepare_embedded / solve_embedded!.

usage: mpiexecjl -n R julia --project=. -t T scripts/delivery/embed_seq.jl INPUT.nc MU AZIMUTH_DEG EPS [KEY=VALUE ...]
1. cold solve of INPUT (start = :column_hse);
2. ln T perturbed by a smooth random field of rms EPS (scripts/viewing/measure_warm.jl `perturbation`, seed 1; B
   unchanged), solved warm (start = :previous: the solution of 1 on top of the new column HSE);
3. the same perturbed cube solved cold.
Per solve: status, reason, valid, steps, Φ/Φ₀, qualification outcome and bounds, roundoff, work (products, CG, abnormal
exits), seconds; and the rms distance of z(τ_LOS = 1) (mean removed) of 2 from 3. KEY: PoissonTauOptions fields
(numbers, or symbols as names: lm=gain).
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
end   # as scripts/viewing/measure_warm.jl

function main(path, mu_s, az_s, eps_s, kv...)
    H.ensure_mpi()
    t0 = time()
    NX, NY, nt = H.PoissonTau.tau_input_size(path)
    mu = parse(Float64, mu_s)
    v = H.ViewOptions(mu = mu, azimuth_deg = parse(Float64, az_s), geometry = mu == 1 ? :plane_parallel : :inclined)
    d = Dict{Symbol,Any}(:max_steps => 25, :stop => :gn, :gn_tol => 1e-3, :lm => :gain, :forcing => :ew, :cg_cap => 200)
    for a in kv
        k, s = split(a, "="); T = fieldtype(H.PoissonTauOptions, Symbol(k))
        d[Symbol(k)] = T == Symbol ? Symbol(s) : parse(T, s)
    end
    opts = H.PoissonTauOptions(; d...)
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y = H.HDF5.h5open(f -> H.tau_coords(f, path, "x", "y"), path)
    logf(s) = MPI.Comm_rank(MPI.COMM_WORLD) == 0 && (println(@sprintf("[%7.1fs] %s", time() - t0, s)); flush(stdout))
    ws = H.prepare_embedded(MPI.COMM_WORLD, :mhs_poisson_ltau; axis = lt, x = x, y = y, eos = tab, g = 274.0,
                            column = H.ColumnTauOptions(align = "ptot"), poisson = opts, view = v)
    _, _, _, T, Bx, By, Bz = H.PoissonTau.read_tau_input(ws.par, path)
    par = ws.par
    logf(@sprintf("%s: %d × %d × %d, μ %.3f φ %g, ranks %d, threads %d; %s", path, NX, NY, nt, mu, v.azimuth_deg,
                  par.nranks, Threads.nthreads(), opts))
    k0 = argmin(abs.(lt)); N = NX * NY
    z1(z) = (a = z[k0, :, :]; a .- H.gsum(par, sum(a)) / N)
    function report(tag, res)
        st = res.status; w = st.stats["work"]; q = get(st.stats, "stop_check", nothing)
        logf(@sprintf("%-14s status %-10s reason %-14s valid %s steps %2d Φ/Φ₀ %.3e  qual %s [%.2e, %.2e]  roundoff %s  A %d M %d CG %d abnormal %d  %.1f s",
                      tag, st.status, st.reason, st.valid, st.steps, st.phi / st.phi0,
                      q === nothing ? "none" : q["outcome"], q === nothing ? NaN : q["lower"], q === nothing ? NaN : q["upper"],
                      st.stats["roundoff"], w["aprod"] + w["qual_aprod"] + w["final_aprod"], w["minv"] + w["qual_minv"] + w["final_minv"],
                      w["cg"], w["cg_abnormal"], st.seconds))
        for m in st.messages; logf("    " * m); end
    end
    lT = log.(T)
    BLAS.set_num_threads(min(32, Threads.nthreads()))
    r1 = H.solve_embedded!(ws, lT, Bx, By, Bz); report("1 cold", r1)
    lTp = lT .+ perturbation(par, lt, NX, NY, size(lT, 3), parse(Float64, eps_s), 1)
    r2 = H.solve_embedded!(ws, lTp, Bx, By, Bz; start = :previous); report("2 warm", r2)
    zw = copy(z1(r2.z))                                     # the result buffers are reused by the next solve
    r3 = H.solve_embedded!(ws, lTp, Bx, By, Bz); report("3 cold", r3)
    dz = sqrt(H.gsum(par, sum(abs2, zw .- z1(r3.z))) / N)
    logf(@sprintf("z(τ_LOS=1) rms distance warm − cold (perturbed cube): %.3f km", dz / 1e3))
    H.free_embedded!(ws)
end

main(ARGS...)
