# Strong scaling of the Poisson modes at 252² (the adjoint benchmark's cases, adjoint_benchmark.jl): wall time of a solve
# and its adjoint, the solver's work counters, and optionally a profile of rank 0.
#   mpiexec -n R julia --project=. -t T test/scripts/poisson_scaling.jl      (from HSEMHS/)
# ENV: MODE = ltau | logm | z (default ltau), STEPS (Gauss–Newton steps, default 4), POLISH (default 0), ADJ = 1 (also
# time the adjoint), PROFILE = path (write a flat profile of rank 0's solve there).
using HSEMHS, MPI, HDF5, Random, Statistics, Printf, Profile
const H = HSEMHS; H.ensure_mpi()
include(joinpath(@__DIR__, "..", "adjoint_helpers.jl"))
tab = H.load_table(ETABLE); comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); R = MPI.Comm_size(comm)
mode = get(ENV, "MODE", "ltau"); steps = parse(Int, get(ENV, "STEPS", "4")); polish = parse(Int, get(ENV, "POLISH", "0"))
N = 252
tile(a) = repeat(a, 1, cld(N, size(a, 2)), cld(N, size(a, 3)))[:, 1:N, 1:N]
xs = collect(0:N-1) .* 95238.1
if mode == "z"
    f = load_z_fixture("11x12", Colon()); axis = f.z
    ws = H.prepare_embedded(comm, :mhs_poisson_z; axis = axis, x = xs, y = xs, eos = tab, g = 274.0,
                            column = H.ColumnZOptions(anchor = :bottom, P_ref = f.P_ref),
                            poisson = H.PoissonZOptions(z_cut_m = 1.0e6, max_steps = steps, polish = polish))
else
    f = load_tau_fixture("11x12", Colon()); mass = mode == "logm"
    axis = mass ? f.lt .+ 1.0 : f.lt
    opts = H.PoissonTauOptions(max_steps = steps, polish = polish)
    ws = H.prepare_embedded(comm, mass ? :mhs_poisson_logm : :mhs_poisson_ltau; axis = axis, x = xs, y = xs, eos = tab,
                            g = 274.0, column = H.ColumnTauOptions(align = "ptot"),
                            poisson = mass ? H.PoissonMassOptions(; logm_cut = -1.5, max_steps = steps, polish = polish) : opts)
end
b = ws.par.block; ir = b.i0:b.i0+b.nx-1
lnT, Bx, By, Bz = (tile(a)[:, :, ir] for a in (f.lnT, f.Bx, f.By, f.Bz))
say(s) = rank == 0 && (println(s); flush(stdout))
timed(g) = (MPI.Barrier(comm); t = @elapsed g(); MPI.Allreduce(t, max, comm))
say("mode $mode, ranks $R × threads $(Threads.nthreads()), x slab $(b.nx), steps $steps, polish $polish")
H.solve_embedded!(ws, lnT, Bx, By, Bz)                               # compile
pf = get(ENV, "PROFILE", "")
if !isempty(pf)
    Profile.init(n = 10^8, delay = 0.005); Profile.clear()
    ts = timed(() -> (Profile.@profile H.solve_embedded!(ws, lnT, Bx, By, Bz)))
    rank == 0 && open(io -> Profile.print(IOContext(io, :displaysize => (100000, 400)); format = :flat, sortedby = :count,
                                          mincount = 20), pf, "w")
    rank == 0 && open(io -> Profile.print(IOContext(io, :displaysize => (100000, 300)); threads = 1, format = :tree,
                                          mincount = 25, maxdepth = 40, noisefloor = 1), pf * ".main", "w")
else
    st = Ref{Any}(nothing)
    ts = timed(() -> (st[] = @timed H.solve_embedded!(ws, lnT, Bx, By, Bz)))
    say(@sprintf("allocated %.2f GB, GC %.2f s", st[].bytes / 1e9, st[].gctime))
end
wk = get(ws.res.status.stats, "work", Dict{String,Any}())
say(@sprintf("solve %.2f s  status %s  work %s", ts, ws.res.status.status,
             join(("$k=$(v isa Float64 ? round(v; digits = 2) : v)" for (k, v) in sort(collect(wk))), " ")))
if get(ENV, "ADJ", "0") == "1"
    H.solve_embedded!(ws, lnT, Bx, By, Bz; record = true)
    bar = H.EquilibriumCotangent(ws); res = ws.tape.res
    bar.rho .= 1 ./ res.rho; bar.ne .= 1 ./ res.ne; bar.Pgas .= 1 ./ res.Pgas; bar.z .= 1e-5
    H.adjoint_embedded!(ws, bar)                                     # compile
    ta = timed(() -> H.adjoint_embedded!(ws, bar)); say(@sprintf("adjoint %.2f s", ta))
end
H.free_embedded!(ws)
