"""
    HSEMHS

Hydrostatic (column) and magnetohydrostatic equilibria on a geometric-height grid.
Implements `HSE_MHS_FINAL_SPEC.md` following `HSE_MHS_IMPLEMENTATION.md`.

Parallel model: MPI (2D horizontal process grid, full columns local, THREAD_FUNNELED)
plus Julia threads inside each rank. Every run, including tests, goes through the MPI
layer (E11); a single rank needs no `mpiexec`.
"""
module HSEMHS

using MPI
using HDF5
using OffsetArrays
using TOML
using SHA
using Statistics
using Printf
using Dates
using Random
using LinearAlgebra
using Libdl

include("Constants.jl")
include("Status.jl")
include("Config.jl")
include("Parallel.jl")
include("Fields.jl")
include("Geometry.jl")
include("Operators.jl")
include("Diagnostics.jl")
include("eos/WittmannDirect.jl")
include("eos/SahaHot.jl")
include("eos/Opacity500.jl")
include("eos/Envelope.jl")
include("eos/Table.jl")
include("eos/TestEOS.jl")
include("eos/BuildTable.jl")
include("IOInput.jl")
include("IOOutput.jl")
include("Gate0.jl")
include("Tau.jl")
include("ColumnHSE.jl")
include("RunColumnHSE.jl")
include("Screening.jl")
include("MHSResidual.jl")
include("MHSOptimize.jl")
include("Synthetic.jl")
include("Staggered.jl")
include("Driver.jl")

function __init__()
    LinearAlgebra.BLAS.set_num_threads(1)
end

end # module
