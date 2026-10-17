# Staggered Φ, ⟨∇Φ, p⟩ and ‖∇Φ‖² for the M6b test problem on any rank/thread layout.
# usage: [mpiexec -n R] julia --project -t T sgrad_ranks.jl OUT.txt
using Test, HSEMHS, HDF5, MPI, OffsetArrays, Random, Statistics, Printf
const H = HSEMHS
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "helpers.jl"))
rank0(par) = par.rank == 0
src = read(joinpath(@__DIR__, "..", "test_stag_gradient.jl"), String)
include_string(Main, split(src, "sgradient_checks(H.IdealGasMu")[1])
# the M6b problem, then the D9b production terms (face data, Φ_Q, λ_Uz)
vals = Float64[]
for kw in ((;), (; dfaces = true, lq = 1.0, luz = 1e-4))
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); kw...)
    w = H.SMHSWork(par); g = H.similar_state(xs)
    f, terms, _ = H.fg!(g, pb, w, xs)
    append!(vals, (f, H.dotN(par, g, srand_dir(par, xs, 1:5)), H.dotN(par, g, g)))
end
MPI.Comm_rank(MPI.COMM_WORLD) == 0 && open(ARGS[1], "w") do io
    println(io, join(repr.(vals), " "))
end
MPI.Finalize()
