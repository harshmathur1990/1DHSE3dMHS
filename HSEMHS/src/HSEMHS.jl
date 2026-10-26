"""
    HSEMHS

Hydrostatic (column) and magnetohydrostatic equilibria on a geometric-height grid.
Implements `HSE_MHS_FINAL_SPEC.md` following `HSE_MHS_IMPLEMENTATION.md`.

Parallel model: MPI (2D horizontal process grid, full columns local, THREAD_FUNNELED)
plus Julia threads inside each rank. Every run, including tests, goes through the MPI
layer (E11); a single rank needs no `mpiexec`. Embedding callers pass their own
communicator to the in-memory API (`prepare_embedded`, Embedded.jl).
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
using ProgressMeter: Progress, next!, finish!

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
include("Slab.jl")
include("PoissonZ.jl")
include("PoissonTau.jl")
include("Embedded.jl")
include("RunModes.jl")
include("Screening.jl")
include("MHSResidual.jl")
include("MHSOptimize.jl")
include("Synthetic.jl")
include("Staggered.jl")
include("MHSStaggered.jl")
include("MHSGaussNewton.jl")
include("MHSAlternate.jl")
include("MHSAudit.jl")
include("MHSJointGN.jl")
include("TauHeight.jl")
include("Driver.jl")

# The embedded API (Embedded.jl) for callers that own MPI and threading.
export prepare_embedded, solve_embedded!, free_embedded!, snapshot, agree,
       prepare_column_hse_z, prepare_column_hse_ltau, prepare_mhs_poisson_z, prepare_mhs_poisson_ltau,
       ColumnZOptions, ColumnTauOptions, PoissonZOptions, PoissonTauOptions, option_errors,
       EquilibriumStatus, EquilibriumResult, EmbeddedWorkspace, EMBEDDED_MODES, slab_range, init_parallel, free_parallel!

# No __init__: loading HSEMHS changes no global state (BLAS and FFTW threads belong to the caller; the standalone entry
# point scripts/run.jl sets BLAS to one thread itself).

end # module
