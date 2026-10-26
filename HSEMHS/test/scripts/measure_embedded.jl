# Timings and solver diagnostics of the embedded API for reports/embedding_api_gate.md.
# usage (from HSEMHS/): [mpiexec -n R] julia --project=. -t T test/scripts/measure_embedded.jl [tag ...]
using HSEMHS, MPI, HDF5, LinearAlgebra, Printf, Statistics
const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "embedded_helpers.jl"))
comm = MPI.COMM_WORLD; R = MPI.Comm_size(comm); me = MPI.Comm_rank(comm)
tab = H.load_table(ETABLE)
tags = isempty(ARGS) ? ["24x18"] : ARGS
for tag in tags, mode in H.EMBEDDED_MODES
    probe = load_fixture(mode, tag, 1:1)
    ir = H.slab_range(length(probe.x), R, me)
    fx = load_fixture(mode, tag, ir)
    o = ref_options(mode, fx)
    MPI.Barrier(comm); t = time()
    ws = H.prepare_embedded(comm, mode; axis = o.axis, x = o.x, y = o.y, eos = tab, g = o.g, column = o.column,
                            poisson = o.poisson)
    tprep = time() - t
    H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)                       # warm-up (compilation)
    MPI.Barrier(comm); t = time()
    res = H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
    MPI.Barrier(comm); tsolve = time() - t
    st = res.status
    drift = NaN
    if mode == :mhs_poisson_z
        # anchor cells = the bottom plane: mean ln P after the solve minus its column-HSE start value
        loc = [sum(res.lnP[end, :, :]), sum(log.(res.Pgas_start[end, :, :])), length(res.lnP[end, :, :])]
        tot = MPI.Allreduce(loc, +, comm)
        drift = (tot[1] - tot[2]) / tot[3]
    end
    me == 0 && @printf("%-6s n=%d T=%d %-17s prepare %6.2f s  solve %7.3f s  steps %3d  reason %-11s  Φ/Φ₀ %.3e  GN dec %.3e  ‖JᵀWR‖ %.3e  lm_mu %.3g  anchor mean ΔlnP %+.3e\n",
                       tag, R, Threads.nthreads(), mode, tprep, tsolve, st.steps, st.reason, st.phi / st.phi0,
                       st.gn_decrement, st.grad_norm, st.lm_mu, drift)
    H.free_embedded!(ws)
end
MPI.Finalize()
