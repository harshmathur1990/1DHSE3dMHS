using Test
using HSEMHS
using MPI
using HDF5
using OffsetArrays
using Random
using Statistics
using LinearAlgebra

BLAS.set_num_threads(1)     # HSEMHS no longer sets this on load (the caller owns threading)

const H = HSEMHS
const FIX = joinpath(@__DIR__, "fixtures")
const PY = get(ENV, "HSEMHS_PYTHON", "/Users/harshm/env/bin/python3")

# fixtures are produced by the Python oracle script (needs numpy + h5py)
if !isfile(joinpath(FIX, "good.nc"))
    run(`$PY -I $(joinpath(@__DIR__, "..", "oracles", "python", "make_fixture.py")) $FIX`)
end

H.ensure_mpi()

include("test_units_config.jl")
include("test_operators.jl")
include("test_parallel.jl")
include("test_reader.jl")
include("test_output.jl")
include("test_gate0.jl")
include("test_gate1.jl")
include("test_gate2.jl")
include("test_screening.jl")
include("test_gradient.jl")
include("test_gateC.jl")
include("test_staggered.jl")
include("test_stag_gradient.jl")
include("test_gateC_staggered.jl")
include("test_face_bounds.jl")
include("test_tau_height.jl")
include("test_embedded.jl")
include("test_view.jl")
include("test_block.jl")
include("test_warm.jl")

# multi-rank tests: the same checks on 2, 4 and 6 ranks
@testset "MPI ranks" begin
    if get(ENV, "HSEMHS_SKIP_MPI", "0") == "1"
        @info "skipping multi-rank tests (HSEMHS_SKIP_MPI=1)"
    else
        script = joinpath(@__DIR__, "mpi", "mpi_tests.jl")
        proj = dirname(@__DIR__)
        for n in (2, 4, 6)
            ok = MPI.mpiexec() do exe
                success(pipeline(`$exe -n $n $(Base.julia_cmd()) --project=$proj -t 2 $script`;
                                 stdout = stdout, stderr = stderr))
            end
            @test ok
        end
        escript = joinpath(@__DIR__, "mpi", "mpi_embedded.jl")
        vscript = joinpath(@__DIR__, "mpi", "mpi_view.jl")
        for n in (2, 3, 4)
            ok = MPI.mpiexec() do exe
                success(pipeline(`$exe -n $n $(Base.julia_cmd()) --project=$proj -t 2 $vscript`;
                                 stdout = stdout, stderr = stderr))
            end
            @test ok
        end
        for n in (2, 3, 4)
            ok = MPI.mpiexec() do exe
                success(pipeline(`$exe -n $n $(Base.julia_cmd()) --project=$proj -t 2 $escript`;
                                 stdout = stdout, stderr = stderr))
            end
            @test ok
        end
    end
end
