# Multi-rank checks of the equilibrium adjoint; run with `mpiexec -n R julia --project -t 2 test/mpi/mpi_adjoint.jl`
# (R = 2, 3). Every case of adjoint_cases() passes its dot-product check on R ranks, and its adjoint pairing equals the
# one computed by rank 0 alone (COMM_SELF, the whole domain) to round-off of the global sums.
using Test
using HSEMHS
using MPI
using HDF5
using LinearAlgebra
using Statistics

const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "adjoint_helpers.jl"))

const comm = MPI.COMM_WORLD
const me = MPI.Comm_rank(comm)

@testset "equilibrium adjoint on $(MPI.Comm_size(comm)) ranks" begin
    tab = H.load_table(ETABLE)
    for case in adjoint_cases()
        inp = adjoint_inputs(case.grid)
        r = adjoint_dot_error(comm, case, inp, tab)
        me == 0 && @info "adjoint check" case = case.name ranks = MPI.Comm_size(comm) err = r.err
        @test r.err < case.tol
        one = me == 0 ? adjoint_dot_error(MPI.COMM_SELF, case, inp, tab; hs = (1e-5,)) : nothing
        rhs1 = MPI.bcast(me == 0 ? one.rhs : 0.0, 0, comm)
        tolr = H.poisson_mode(case.mode) ? 1e-6 : 1e-10                  # Poisson: iterative solves on other rank counts
        @test abs(r.rhs - rhs1) <= tolr * abs(rhs1)
    end
end
