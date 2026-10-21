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
    # M5g: face bounds with the limiting face on the last rank — global fraction to the boundary, violation and a
    # bounded joint GN step agree on every rank (Codex M5f review finding 7)
    let par = H.init_parallel(8, 8, 9), b = par.block
        pb, x, _, _, _ = H.sc1_problem(par, 8, 8, 9; Bobs = (0.0, 0.0, 0.005), lambda_F = 1e-4, lambda_Q = 1e-3)
        w = H.SMHSWork(par); H.objective!(pb, w, x)
        gi, gj, gk = 7, 8, 4                                           # a corner face: not on rank 0 when R > 1
        mine = b.i0 <= gi <= b.i0 + b.nx - 1 && b.j0 <= gj <= b.j0 + b.ny - 1
        hard = (H.newfield(par; fill = 50.0), H.newfield(par; fill = 50.0), H.newfield0(par; fill = 50.0))
        c0 = 0.0; isg = 0.0
        if mine
            i, j = gi - b.i0 + 1, gj - b.j0 + 1
            isg = sqrt(pb.invsf2[3][i, j, gk]); c0 = (w.B[3][i, j, gk] - pb.Bobsf[3][i, j, gk]) * isg
            hard[3][i, j, gk] = abs(c0) + 0.25
        end
        c0 = MPI.Allreduce(c0, +, comm); isg = MPI.Allreduce(isg, +, comm)
        @test MPI.Allreduce(mine ? 1 : 0, +, comm) == 1
        @test R == 1 || !(mine && par.rank == 0)
        d = H.similar_state(x); H.setscalar!(par, d[5], 1.0)          # b̄ shifts every z-face by 1 (T)
        a = H.face_fraction(pb, w, H.SMHSWork(par), d, hard)
        @test isapprox(a, 0.25 / isg; rtol = 1e-10)
        @test MPI.Allreduce(a, max, comm) == MPI.Allreduce(a, min, comm)
        @test isapprox(H.face_violation(pb, w, hard)[1], -0.25; rtol = 1e-12)
        H.set_face_hinge!(pb, hard; lambda = 0.1, kind = :barrier)
        ulo = H.newfield(par); H.owned_map!((i, j, k) -> w.u[i, j, k] - 3.0, ulo, par)
        st, why, n, res = H.joint_gn_solve!(pb, w, x, ulo; face_hard = hard, bound_tau = 0.99, max_candidates = 3)
        H.objective!(pb, w, x)
        @test H.face_violation(pb, w, hard)[1] < 0
        @test MPI.Allreduce(n, max, comm) == MPI.Allreduce(n, min, comm)
    end
    # M5h: τ = 1 height change — global maximum, its column and the validity counts agree on every rank
    let par = H.init_parallel(8, 8, 9), b = par.block
        pb, x, _, _, _ = H.sc1_problem(par, 8, 8, 9; Bobs = (0.0, 0.0, 0.005), lambda_F = 1e-4, lambda_Q = 1e-3, P_ref = 2.0e5)
        w = H.SMHSWork(par); H.objective!(pb, w, x)
        z0 = fill(NaN, b.nx, b.ny); f0 = zeros(UInt8, b.nx, b.ny); H.tau1_heights!(z0, f0, pb, w.u)
        gi, gj = 7, 8
        xp = deepcopy(x)
        if b.i0 <= gi <= b.i0 + b.nx - 1 && b.j0 <= gj <= b.j0 + b.ny - 1
            for k in 1:b.nz-1; xp[1][gi-b.i0+1, gj-b.j0+1, k] += 0.2; end
        end
        H.objective!(pb, w, xp)
        z1 = similar(z0); f1 = similar(f0); H.tau1_heights!(z1, f1, pb, w.u)
        s = H.tau1_change_stats(par, z1, f1, z0, f0)
        @test (s.worst_i, s.worst_j) == (gi, gj) && s.dz_max > 0 && s.all_valid && s.ncomparable == 64
        @test MPI.Allreduce(s.dz_max, max, comm) == MPI.Allreduce(s.dz_max, min, comm)
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
