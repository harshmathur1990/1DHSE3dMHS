# A given electron density: P and ρ come from (T, nₑ) through the EOS at every level, no hydrostatic integration (column modes).
using Test, MPI
using HSEMHS
const HN = HSEMHS

@testset "given electron density" begin
    HN.ensure_mpi(); comm = MPI.COMM_WORLD
    eos = HN.IdealGasMu(1.3); g = 274.0; nk, ny, nx = 12, 3, 4
    x = collect(0.0:nx-1) .* 5e4; y = collect(0.0:ny-1) .* 5e4
    ne = [1e16 * exp(0.9 * k + 0.05 * j - 0.03 * i) for k in 1:nk, j in 1:ny, i in 1:nx]; lnne = log.(ne)

    @testset "the inversion" begin
        for X in (log(4000.0), log(1e5)), lnn in (log(1e14), log(1e22))
            lp = HN.lnP_from_lnne(eos, X, lnn)
            @test HN.state(eos, X, lp)[2] ≈ lnn atol=1e-9
        end
    end

    table_file = get(ENV, "HSEMHS_TEST_TABLE", "/mn/stornext/d9/data/harshm/hsemhs_env112/work/e2e/inputs/eos_table_fullz.h5")
    if isfile(table_file)
        tab = HN.load_table(table_file)
        T3 = [5000.0 + 200.0 * k + 30.0 * j + 7.0 * i for k in 1:nk, j in 1:ny, i in 1:nx]; lT3 = log.(T3)
        @testset "height grid" begin
            z = collect(range(2.0e6, 0.0; length = nk))
            ws = HN.prepare_embedded(comm, :column_hse_z; axis = z, x, y, eos = tab, g, column = HN.ColumnZOptions(anchor = :bottom, P_ref = 1e4))
            res = HN.solve_embedded!(ws, lT3; lnne)
            @test res.status.valid && res.status.stats["ne_given"] === true
            for (k, j, i) in ((1, 1, 1), (6, 2, 3), (12, 3, 4))
                r, n, c, _ = HN.state(tab, lT3[k, j, i], res.lnP[k, j, i])
                @test n ≈ lnne[k, j, i] atol=1e-9
                @test res.rho[k, j, i] ≈ exp(r) && res.chi500[k, j, i] ≈ exp(c) && res.Pgas[k, j, i] ≈ exp(res.lnP[k, j, i])
            end
            @test res.ne ≈ ne rtol=1e-14                                  # the given values
            @test all(k -> res.z[k, 1, 1] == z[k], 1:nk)
            @test all(diff(res.tau500[:, 1, 1]) .> 0)                     # τ integrated on the heights as usual
            plain = HN.solve_embedded!(ws, lT3)                           # without nₑ: hydrostatic, as before
            @test plain.status.stats["ne_given"] === false
            @test plain.Pgas[end, 1, 1] ≈ 1e4 rtol=1e-9
            HN.free_embedded!(ws)
        end

        @testset "τ grid" begin
            lt = collect(range(-6.0, 1.0; length = nk))
            ws = HN.prepare_embedded(comm, :column_hse_ltau; axis = lt, x, y, eos = tab, g, column = HN.ColumnTauOptions(align = "ptot"))
            res = HN.solve_embedded!(ws, lT3; lnne)                       # no B needed: the alignment is not used
            @test res.status.valid
            @test res.ne ≈ ne rtol=1e-14
            @test all(res.logtau500[:, 1, 1] .== lt)
            # heights: dz = -dτ/χ, trapezoid in ln τ, and mean z(τ = 1) = 0
            chi = res.chi500[:, 2, 3]; τ = exp10.(lt)
            dz = [0.5 * (τ[k] / chi[k] + τ[k+1] / chi[k+1]) * (log(τ[k+1]) - log(τ[k])) for k in 1:nk-1]
            @test -diff(res.z[:, 2, 3]) ≈ dz rtol=1e-8
            @test all(<(0), diff(res.z[:, 2, 3]))
            @test abs(sum(res.z_tau1) / length(res.z_tau1)) < 1e-6
            HN.free_embedded!(ws)
        end

        @testset "refused" begin
            z = collect(range(2.0e6, 0.0; length = nk))
            ws = HN.prepare_embedded(comm, :column_hse_z; axis = z, x, y, eos = tab, g, column = HN.ColumnZOptions(anchor = :bottom, P_ref = 1e4))
            @test_throws HN.InvalidInput HN.solve_embedded!(ws, lT3; lnne = lnne[1:end-1, :, :])
            bad = copy(lnne); bad[3, 1, 1] = NaN
            @test_throws HN.InvalidInput HN.solve_embedded!(ws, lT3; lnne = bad)
            HN.free_embedded!(ws)
            ws = HN.prepare_embedded(comm, :mhs_poisson_z; axis = z, x, y, eos = tab, g, column = HN.ColumnZOptions(anchor = :bottom, P_ref = 1e4), poisson = HN.PoissonZOptions())
            @test_throws HN.InvalidInput HN.solve_embedded!(ws, lT3, zeros(nk, ny, nx), zeros(nk, ny, nx), zeros(nk, ny, nx); lnne)
            HN.free_embedded!(ws)
        end
    end

    # the real table: nₑ from a hydrostatic solve, given back, returns the same pressure
    if isfile(table_file)
        @testset "round trip on the EOS table" begin
            tab = HN.load_table(table_file)
            z = collect(range(1.8e6, -1.0e5; length = 40))
            T2 = [12000.0 - 1600.0 * sqrt(k) / 1.0 + 6000.0 * (k / 40)^2 + 9.0 * j + 3.0 * i for k in 1:40, j in 1:ny, i in 1:nx]
            ws = HN.prepare_embedded(comm, :column_hse_z; axis = z, x, y, eos = tab, g, column = HN.ColumnZOptions(anchor = :bottom, P_ref = 1.5e4))
            hse = HN.snapshot(HN.solve_embedded!(ws, log.(T2)))
            @test hse.status.valid
            res = HN.solve_embedded!(ws, log.(T2); lnne = log.(hse.ne))
            @test res.status.valid
            @test maximum(abs.(res.Pgas ./ hse.Pgas .- 1)) < 1e-6
            @test maximum(abs.(res.rho ./ hse.rho .- 1)) < 1e-6
            @test res.ne == hse.ne
            # outside the table: the same failure as the column modes
            impossible = fill(log(1e40), size(hse.ne))
            failed = HN.solve_embedded!(ws, log.(T2); lnne = impossible)
            @test !failed.status.valid && failed.status.reason === :eos_out_of_envelope
            HN.free_embedded!(ws)
        end
    end
end
