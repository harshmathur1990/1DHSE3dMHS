# Run only the embedded-API tests: julia --project=. -t 4 test/run_embedded.jl  (from HSEMHS/)
using Test, HSEMHS, MPI, HDF5, LinearAlgebra
const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include("test_embedded.jl")
