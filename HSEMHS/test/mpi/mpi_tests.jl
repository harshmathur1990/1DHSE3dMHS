# Multi-rank checks; run with `mpiexec -n R julia --project -t 2 test/mpi/mpi_tests.jl`.
# Every rank runs the same code; global results are reduced so that all ranks agree.
using Test
using HSEMHS
using MPI
using HDF5
using OffsetArrays
using Random
using Statistics

const H = HSEMHS
const FIX = joinpath(@__DIR__, "..", "fixtures")
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "helpers.jl"))

const comm = MPI.COMM_WORLD
const R = MPI.Comm_size(comm)
quiet = MPI.Comm_rank(comm) != 0

ts = @testset "MPI n=$R" begin
    # several process-grid shapes for this rank count
    shapes = unique([(0, 0), (R, 1), (1, R)])
    for (px, py) in shapes
        par, grid, ops, x, y, z = test_setup(13, 12, 7; px = px, py = py)
        @test halo_errors(par) == 0
        @test all(transpose_errors(par, ops, x, y, z) .<= 1e-13)
        # layer quantiles identical to the single-rank (gathered) definition
        f = H.newfield(par)
        fill_global!(f, par, x, y, z, (xx, yy, zz, gi, gj, k) -> grand(gi, gj, k))
        Q = H.layer_quantiles(par, f, [0.0, 0.5, 0.99]; stride = 2)
        ref = [quantile([grand(gi, gj, k) for gj in 1:2:12 for gi in 1:2:13], [0.0, 0.5, 0.99]) for k in 1:7]
        @test all(Q[k, :] == ref[k] for k in 1:7)
        s = H.global_sum(par, f)
        sref = sum(grand(gi, gj, k) for k in 1:7, gj in 1:12, gi in 1:13)
        @test abs(s - sref) <= 1e-12 * sum(abs, (grand(gi, gj, k) for k in 1:7, gj in 1:12, gi in 1:13))
        # collective error: one rank fails, all ranks see invalid_input
        st, msgs, _ = H.run_collective(par) do
            par.rank == R - 1 && H.invalid("rank-local failure")
            1
        end
        @test st == H.invalid_input && any(occursin("rank-local failure", m) for m in msgs)
        # staggered operators (M6a): div∘curl ≡ 0 and transposes on this decomposition
        divrel, errs = staggered_checks(par, grid)
        @test divrel <= 1e-14
        @test all(errs[nm] <= 1e-13 for nm in ("C", "Cstar", "H", "G"))
    end
    # a grid too small for the process grid is rejected on every rank (no deadlock)
    @test_throws H.InvalidInput H.init_parallel(R + 1, 4, 5; px = R, py = 1)
    # reader on fixtures (10×8×9) with this rank count
    par = H.init_parallel(10, 8, 9)
    @test check_fixture_read(joinpath(FIX, "good.nc"), par) == 0
    @test check_fixture_read(joinpath(FIX, "good4d.nc"), par) == 0
    # token-ring writer
    dir = MPI.bcast(MPI.Comm_rank(comm) == 0 ? mktempdir() : "", comm; root = 0)
    path = joinpath(dir, "out_mpi.h5")
    write_pattern(par, path)
    ok = MPI.Comm_rank(comm) == 0 ? check_pattern(path, 10, 8, 9) : true
    @test MPI.bcast(ok, comm; root = 0)
end

nfail = 0   # reaching here means the top-level testset passed (it throws otherwise)
nfail = MPI.Allreduce(nfail, max, comm)
MPI.Finalize()
exit(nfail)
