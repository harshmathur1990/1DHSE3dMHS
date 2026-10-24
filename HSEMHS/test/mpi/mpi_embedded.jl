# Multi-rank checks of the embedded API; run with `mpiexec -n R julia --project -t 2 test/mpi/mpi_embedded.jl`
# (R = 2, 3, 4; the communicator-split test needs R = 4). Every rank runs the same code.
using Test
using HSEMHS
using MPI
using HDF5
using LinearAlgebra
using Printf

const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "embedded_helpers.jl"))

const comm = MPI.COMM_WORLD
const R = MPI.Comm_size(comm)
const me = MPI.Comm_rank(comm)

"""Global maximum of a per-rank value."""
gmaxv(x) = MPI.Allreduce(Float64(x), max, comm)
"""True when every rank holds the same string."""
same_everywhere(s, c = comm) = (h = hash(s) % UInt32; MPI.Allreduce(h, max, c) == MPI.Allreduce(h, min, c))

"""Solve `mode` on fixture `tag` over communicator `c`; returns (result snapshot, owned x range, workspace stats)."""
function solve_on(c, mode, tag, tab; blas = 1)
    nr = MPI.Comm_size(c); r = MPI.Comm_rank(c)
    probe = load_fixture(mode, tag, 1:1)
    NX = length(probe.x)
    ir = H.slab_range(NX, nr, r)
    fx = load_fixture(mode, tag, ir)
    o = ref_options(mode, fx)
    ws = H.prepare_embedded(c, mode; axis = o.axis, x = o.x, y = o.y, eos = tab, g = o.g, column = o.column,
                            poisson = o.poisson)
    @assert H.owned_columns(ws) == ir
    BLAS.set_num_threads(blas)
    res = try
        H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz))
    finally
        BLAS.set_num_threads(1)
    end
    H.free_embedded!(ws)
    return res, ir
end

# Tolerances for agreement with the single-rank result. Sums reduce in a rank-dependent order: the column-z path has
# no reduction in its values (bitwise), the τ column path only the alignment median and the gauge mean. The Poisson
# solves amplify rounding differences through their step acceptance, so they are checked on the converged 11x12 window;
# on the 24x18 window (a non-periodic crop: z stops at max_steps, τ is step-path sensitive) the differences are
# recorded, not asserted (reports/embedding_api_gate.md).
const RANK_TOL = Dict(:column_hse_z => 0.0, :column_hse_ltau => 1e-11, :mhs_poisson_z => 1e-9, :mhs_poisson_ltau => 1e-9)
asserted(mode, tag) = !(H.poisson_mode(mode) && tag == "24x18")

ts = @testset "embedded MPI n=$R" begin
    tab = H.load_table(ETABLE)

    @testset "rank-count independence ($mode, $tag)" for mode in H.EMBEDDED_MODES, tag in ("24x18", "11x12")
        res, ir = solve_on(comm, mode, tag, tab)
        err = gmaxv(ref_error(res, load_ref(ref_name(mode), tag, ir)))
        allvalid = MPI.Allreduce(res.status.valid ? 1 : 0, min, comm) == 1
        me == 0 && @printf("  n=%d %-18s %-6s max rel diff vs 1 rank %.3e  steps %d  reason %s\n", R, mode, tag, err,
                           res.status.steps, res.status.reason)
        @test allvalid
        asserted(mode, tag) && @test err <= RANK_TOL[mode]
        @test same_everywhere(string(res.status.reason, res.status.steps, res.status.phi))
    end

    @testset "one-rank-only failures are collective" begin
        fx = load_z_fixture("24x18", H.slab_range(24, R, me))
        o = ref_options(:mhs_poisson_z, fx)
        kw = (axis = o.axis, x = o.x, y = o.y, g = o.g, column = o.column, poisson = o.poisson)
        # missing EOS table on the last rank only
        msg = try
            H.prepare_embedded(comm, :mhs_poisson_z; kw..., eos = me == R - 1 ? "/nonexistent/eos.h5" : ETABLE); ""
        catch e
            e isa H.InvalidInput ? e.msg : "unexpected $(typeof(e))"
        end
        @test occursin("does not exist", msg) && same_everywhere(msg)
        # a file that is not an EOS table on rank 0 only (stands in for a corrupt table / hash mismatch)
        bogus = tempname() * ".h5"; h5open(f -> (f["junk"] = [1.0]), bogus, "w")
        msg = try
            H.prepare_embedded(comm, :mhs_poisson_z; kw..., eos = me == 0 ? bogus : ETABLE); ""
        catch e
            e isa H.InvalidInput ? e.msg : "unexpected $(typeof(e))"
        end
        rm(bogus)
        @test occursin("rank 0", msg) && same_everywhere(msg)
        MPI.Barrier(comm)
        # malformed array on rank 1 only, inside solve
        ws = H.prepare_embedded(comm, :mhs_poisson_z; kw..., eos = tab)
        lnT = me == 1 ? fx.lnT[:, 1:end-1, :] : fx.lnT
        msg = try
            H.solve_embedded!(ws, lnT, fx.Bx, fx.By, fx.Bz); ""
        catch e
            e isa H.InvalidInput ? e.msg : "unexpected $(typeof(e))"
        end
        @test occursin("rank 1", msg) && same_everywhere(msg)
        # out-of-envelope temperature on the last rank only: a status, identical everywhere, no deadlock
        hot = copy(fx.lnT); me == R - 1 && (hot[:, 2:3, 1] .= log(5.0e7))
        res = H.solve_embedded!(ws, hot, fx.Bx, fx.By, fx.Bz)
        @test !res.status.valid && res.status.reason == :eos_out_of_envelope
        @test same_everywhere(string(res.status.reason, res.status.counts))
        res = H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
        @test res.status.valid
        H.free_embedded!(ws)
        MPI.Barrier(comm)
    end

    @testset "slab size limits" begin
        x5 = collect(0:4) .* 5.0e4
        z = collect(range(2.0e6, -0.4e6; length = 20))
        f = () -> H.prepare_embedded(comm, :column_hse_z; axis = z, x = x5, y = x5, eos = tab, g = 274.0,
                                     column = H.ColumnZOptions(P_ref = 1e5))
        if 5 ÷ R < 2
            msg = try f(); "" catch e; e isa H.InvalidInput ? e.msg : "unexpected" end
            @test occursin("cannot be split", msg) && same_everywhere(msg)
        else
            ws = f(); @test H.owned_columns(ws) == H.slab_range(5, R, me); H.free_embedded!(ws)
        end
    end

    @testset "communicator lifetime: 50 prepare/solve/free cycles" begin
        ok = true
        for cyc in 1:50
            res, _ = solve_on(comm, :column_hse_ltau, "11x12", tab)
            ok &= res.status.valid
        end
        @test ok
        MPI.Barrier(comm)
    end

    if R == 4
        @testset "two sub-communicators solving different atmospheres at once" begin
            # each pair solves its own atmosphere; done one pair at a time (the other pair waits on the world) and then
            # both at once: any hidden use of COMM_WORLD or cross-talk between the pairs changes or deadlocks the result
            color = me ÷ 2
            sub = MPI.Comm_split(comm, color, me)
            tag = color == 0 ? "24x18" : "11x12"
            alone = nothing
            for turn in 0:1
                color == turn && (alone = first(solve_on(sub, :mhs_poisson_z, tag, tab)))
                MPI.Barrier(comm)
            end
            together, ir = solve_on(sub, :mhs_poisson_z, tag, tab)
            same = together.Pgas == alone.Pgas && together.rho == alone.rho && together.status.steps == alone.status.steps
            @test MPI.Allreduce(same ? 1 : 0, min, comm) == 1
            err = MPI.Allreduce(ref_error(together, load_ref("mhs-poisson-z", tag, ir)), max, comm)
            me == 0 && @printf("  split 2+2: concurrent == sequential: %s; max rel diff vs 1 rank %.3e\n", same, err)
            @test together.status.valid
            MPI.free(sub)
            MPI.Barrier(comm)                                                  # no stray messages on the world
        end
    end
end

# a failed @testset throws above (non-zero exit on that rank, which ends the mpiexec job)
MPI.Barrier(comm)
MPI.Finalize()
