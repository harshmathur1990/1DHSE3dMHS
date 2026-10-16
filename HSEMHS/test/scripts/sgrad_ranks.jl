# Staggered Φ, ⟨∇Φ, p⟩ and ‖∇Φ‖² for the M6b test problem on any rank/thread layout.
# usage: [mpiexec -n R] julia --project -t T sgrad_ranks.jl OUT.txt
using Test, HSEMHS, HDF5, MPI, OffsetArrays, Random, Statistics, Printf
const H = HSEMHS
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "helpers.jl"))
rank0(par) = par.rank == 0
src = read(joinpath(@__DIR__, "..", "test_stag_gradient.jl"), String)
include_string(Main, split(src, "sgradient_checks(H.IdealGasMu")[1])
pb, xs, par = sgrad_problem(H.IdealGasMu(1.3))
w = H.SMHSWork(par); g = H.similar_state(xs)
f, terms, _ = H.fg!(g, pb, w, xs)
gp = H.dotN(par, g, srand_dir(par, xs, 1:5))
gg = H.dotN(par, g, g)
par.rank == 0 && open(ARGS[1], "w") do io
    @printf(io, "%s %s %s\n", repr(f), repr(gp), repr(gg))
end
MPI.Finalize()
