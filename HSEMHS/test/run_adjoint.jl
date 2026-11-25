# Run only the equilibrium-adjoint tests: julia --project=. -t 4 test/run_adjoint.jl  (from HSEMHS/)
using Test, HSEMHS, MPI, HDF5, LinearAlgebra, Random, Statistics
const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include("test_adjoint.jl")
