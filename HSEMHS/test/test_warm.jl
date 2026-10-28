# Warm start of mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md D4): the start construction, the fallback to the
# column HSE when the workspace has no previous solution, and a repeated solve starting at the previous solution.

@isdefined(ETABLE) || include("embedded_helpers.jl")

@testset "warm start" begin
    comm = MPI.COMM_WORLD
    tab = H.load_table(ETABLE)
    fx = load_tau_fixture("11x12", Colon())
    mk(mode = :mhs_poisson_ltau; view = H.ViewOptions()) =
        H.prepare_embedded(comm, mode; axis = fx.lt, x = fx.x, y = fx.y, eos = tab, g = 274.0,
                           column = H.ColumnTauOptions(align = "ptot"),
                           poisson = mode == :mhs_poisson_ltau ? H.PoissonTauOptions(max_steps = 30) : nothing, view = view)

    @testset "warm_start!" begin
        nt, kc = 8, 3
        u = rand(nt, 2, 2); z0 = rand(nt, 2, 2); up = rand(nt - kc + 1, 2, 2); cp = rand(2, 2)
        u1 = copy(u); z1 = copy(z0)
        H.PoissonTau.warm_start!(u1, z1, kc, up, cp)
        @test u1[1:kc, :, :] == u[1:kc, :, :]                                  # above the cut and the fixed cut level
        for i in 1:2, j in 1:2
            @test u1[kc+1:nt, j, i] ≈ up[2:end, j, i] .+ (u[kc, j, i] - up[1, j, i])
            @test z1[kc, j, i] ≈ cp[j, i]
            @test z1[1:kc, j, i] .- z1[kc, j, i] ≈ z0[1:kc, j, i] .- z0[kc, j, i]   # heights above the cut move as a block
        end
        @test z1[kc+1:nt, :, :] == z0[kc+1:nt, :, :]
        uc = rand(nt - kc + 1, 2, 2); zc = rand(2, 2); uc[1, :, :] .= up[1, :, :]   # previous column HSE (same cut ln P)
        u2 = copy(u); z2 = copy(z0)
        H.PoissonTau.warm_start!(u2, z2, kc, up, cp, uc, zc)
        @test u2[1:kc, :, :] == u[1:kc, :, :]
        for i in 1:2, j in 1:2
            @test u2[kc+1:nt, j, i] ≈ up[2:end, j, i] .+ (u[kc+1:nt, j, i] .- uc[2:end, j, i])   # + column-HSE response
            @test z2[1:kc, j, i] ≈ z0[1:kc, j, i] .+ (cp[j, i] - zc[j, i])
        end
    end

    @testset "validation" begin
        ws = mk()
        @test_throws H.InvalidInput H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz; start = :last)
        H.free_embedded!(ws)
        ws = mk(:column_hse_ltau)
        @test_throws H.InvalidInput H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz; start = :previous)
        H.free_embedded!(ws)
    end

    @testset "fallback and repeated solve ($(v.geometry), μ $(v.mu))" for v in (H.ViewOptions(),
                                                                              H.ViewOptions(mu = 0.6, azimuth_deg = 30.0, geometry = :inclined))
        ws = mk(; view = v)
        a = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz; start = :previous))   # nothing to start from
        @test a.status.valid
        @test a.status.stats["start"] == "column_hse"
        ws2 = mk(; view = v)
        c = H.snapshot(H.solve_embedded!(ws2, fx.lnT, fx.Bx, fx.By, fx.Bz))
        H.free_embedded!(ws2)
        @test isequal(a.lnP, c.lnP) && isequal(a.z, c.z)                     # the fallback is the cold solve
        b = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz; start = :previous))  # same input again
        @test b.status.valid
        @test b.status.stats["start"] == "previous"
        @test b.status.phi0 < 2 * a.status.phi           # starts where the previous solve ended (weights from the start state)
        @test b.status.steps <= a.status.steps
        @test norm(b.lnP .- a.lnP) < 0.25 * norm(a.lnP .- log.(a.Pgas_start))  # moves much less than the cold solve did
        H.free_embedded!(ws)
    end
end
