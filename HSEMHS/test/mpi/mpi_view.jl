# Multi-rank checks of the viewing angle; run with `mpiexec -n R julia --project -t 2 test/mpi/mpi_view.jl` (R = 2, 3, 4).
# Inclined τ on the height grid with rays crossing rank boundaries (x slabs and 2-D blocks), and the inclined τ modes
# on R ranks against the same solve on one rank.
using Test
using HSEMHS
using MPI
using HDF5
using Random
using LinearAlgebra

const H = HSEMHS
BLAS.set_num_threads(1)
H.ensure_mpi()
include(joinpath(@__DIR__, "..", "embedded_helpers.jl"))
include(joinpath(@__DIR__, "..", "view_helpers.jl"))

const comm = MPI.COMM_WORLD
const R = MPI.Comm_size(comm)
const me = MPI.Comm_rank(comm)
gmaxv(x) = MPI.Allreduce(Float64(x), max, comm)

@testset "viewing angle on $R ranks" begin
    @testset "inclined τ, x slabs (φ $φ)" for φ in (0.0, 45.0, 200.0)
        par = H.slab_par(comm, 12, 10, 9)
        et, ez = shear_exact_error(par, φ)
        @test gmaxv(et) < 1e-12
        @test gmaxv(ez) < 1e-9
        H.free_parallel!(par)
    end
    if R % 2 == 0
        @testset "inclined τ, 2-D blocks (φ $φ)" for φ in (0.0, 45.0, 200.0)
            par = H.init_parallel(comm, 12, 10, 9; px = R ÷ 2, py = 2)
            et, ez = shear_exact_error(par, φ)
            @test gmaxv(et) < 1e-12
            @test gmaxv(ez) < 1e-9
            H.free_parallel!(par)
        end
    end
    @testset "inclined view, $R ranks vs one rank ($mode)" for mode in H.EMBEDDED_MODES
        tab = H.load_table(ETABLE)
        v = H.ViewOptions(mu = 0.6, azimuth_deg = 30.0, geometry = :inclined)
        full = load_fixture(mode, "11x12", Colon())
        NX = length(full.x)
        o = ref_options(mode, full)
        prep(c, f) = H.prepare_embedded(c, mode; axis = o.axis, x = o.x, y = o.y, eos = tab, g = o.g, column = o.column,
                                        poisson = o.poisson, view = v)
        ws1 = prep(MPI.COMM_SELF, full)
        r1 = H.snapshot(H.solve_embedded!(ws1, full.lnT, full.Bx, full.By, full.Bz)); H.free_embedded!(ws1)
        ir = H.slab_range(NX, R, me)
        fx = load_fixture(mode, "11x12", ir)
        ws = prep(comm, fx)
        r = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)); H.free_embedded!(ws)
        @test r.status.valid && r1.status.valid
        tol = mode == :mhs_poisson_ltau ? 1e-5 : 1e-11
        @test gmaxv(maximum(abs, r.lnP .- r1.lnP[:, :, ir])) < tol
        @test gmaxv(maximum(abs.(r.tau500 .- r1.tau500[:, :, ir]) ./ r1.tau500[:, :, ir])) < tol
        @test gmaxv(maximum(abs, r.z_tau1 .- r1.z_tau1[:, ir])) < (mode == :mhs_poisson_ltau ? 1.0 : 1e-6)
    end
end
