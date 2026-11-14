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

# The EOS (Wittmann, the Bifrost hion NLTE hydrogen, χ₅₀₀, tables) is WittBifrostEOS's; every name it defines is imported, so the
# solvers use them as before and can extend them (TestEOS).
import WittBifrostEOS
import WittBifrostEOS: A0, A0_DATA, AbstractEOS, BK_CGS, BRANCH_FAIL, BREAK_EPS, BranchOut, Bucket, COULX_A1, COULX_B1, COULX_C1, EOSModel, EOSTable, EOS_BLEND, EOS_DIRECT_FAILED, EOS_HOT_BRANCH, EOS_NLTE, EOS_OK, EOS_OUT_OF_ENVELOPE, EOS_ROBUST_FALLBACK, EV_CGS, Envelope, ExtendedEOS, FE_E1, FE_G1, FE_WNO1, FIELDS_FAIL, FLOG0, FLOG1, FLOG2, FREQMG, FREQSI1, FREQSI2, HEFREQ0, HE_CHI0, HE_G0, HH_CGS, HotEOS, InvalidInput, K_B_WITT, LN_FIELDS, LN_FIELDS_H, ME_CGS, M_U_WITT, NCONTR, NLTECorrection, OPACITY_INCOMPLETE, PEACH0, PEACH1, PEACH2, SAHA_C, SAHA_FAC, TABLE_DELTA, TLG0, TLG1, TLG2, WITT_FN, WITT_LIB, WITT_LOCK, WITT_READY, WITT_SYMS, WL500, WittCtx, WittState, Z4LOG, _write_meta, al1op, axis_with_breaks, brent, build_libwitt, build_table, c1op, ca2op, catmull_rom, check_cells, chi500_m, compute_nodes!, coolop, cop, coulff, coulx, coverage_scan, default_nlte_correction, default_pf_file, derivative_check, direct_branch, elecop, eos_model, extended_fields, extended_model, fe1op, field_names, fields5, fields_at, file_sha256, findcell, grid_value, h2plop, h2raop, hbasis, hderiv, he1op, he2op, hemiop, heraop, hermite, hminop, hop, hot_branch, hot_partials!, hot_pf, hot_solve, hotop, hrayop, hydrogen, hydrogen_fractions, hydrogen_ln, in_table, invalid, libwitt_handle, libwitt_path, lnrho, lnrho_dY, load_nlte_correction, load_table, locate, lukeop, mass_per_H, meta_value, mg1op, mg2op, model_breaks, model_eval, model_fields, model_nfields, molecb, n1op, nfields, nlte_departure, node_values!, o1op, py_break_index, read_log_abund, refine, saha_w, save_table, seaton, select_T_sw, si1op, si2op, state, table_model, table_sha256, uniform_axis, validate_table, with_nlte, with_switch, witt_ab_others, witt_abundances, witt_eion_ev, witt_fix_gasc_h2, witt_mass_per_H, witt_nstage, witt_partials!, witt_pf, witt_rho_from_H, witt_state, witt_tpf, witt_tpf_range, wittfn, xflags, xright, xstencil

include("Constants.jl")
include("Status.jl")
include("Config.jl")
include("Parallel.jl")
include("Fields.jl")
include("Geometry.jl")
include("Operators.jl")
include("Diagnostics.jl")
include("eos/Envelope.jl")
include("eos/TestEOS.jl")
include("eos/BuildTable.jl")
include("IOInput.jl")
include("IOOutput.jl")
include("Gate0.jl")
include("Tau.jl")
include("ColumnHSE.jl")
include("RunColumnHSE.jl")
include("Slab.jl")
include("ViewTau.jl")
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
       prepare_column_hse_logm, prepare_mhs_poisson_logm,
       ColumnZOptions, ColumnTauOptions, ColumnMassOptions, PoissonZOptions, PoissonTauOptions, PoissonMassOptions, ViewOptions, option_errors,
       EquilibriumStatus, EquilibriumResult, EmbeddedWorkspace, EMBEDDED_MODES, slab_range, init_parallel, free_parallel!

# No __init__: loading HSEMHS changes no global state (BLAS and FFTW threads belong to the caller; the standalone entry
# point scripts/run.jl sets BLAS to one thread itself).

end # module
