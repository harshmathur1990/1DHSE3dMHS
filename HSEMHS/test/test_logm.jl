# Column-mass axis (run.mode = "column-hse-logm", "mhs-poisson-logm"): P = μ g m, dz = −μ m/ρ d ln m, τ computed from the
# state, the same Poisson solver as the τ axis with its Jacobian, the viewing angle, and the file input and output.

@isdefined(ETABLE) || include("embedded_helpers.jl")

@testset "column-mass axis" begin
    comm = MPI.COMM_WORLD
    tab = H.load_table(ETABLE)
    fx = load_tau_fixture("11x12", Colon())
    g = 274.0
    lm0 = fx.lt .+ 1.0                                       # the fixture's T and B read on log10 m = −7 … 2.4
    mk(mode, view = H.ViewOptions(); lm = lm0, cut = -1.5, frame = :vertical, kw...) =
        H.prepare_embedded(comm, mode; axis = lm, x = fx.x, y = fx.y, eos = tab, g = g, column = H.ColumnMassOptions(align = "ptot"),
                           poisson = mode == :mhs_poisson_logm ? H.PoissonMassOptions(; logm_cut = cut, cut_frame = frame, max_steps = 30, rtol = 1e-4, stop = :phi, kw...) : nothing,
                           view = view)
    solve(ws, lnT = fx.lnT, B = (fx.Bx, fx.By, fx.Bz)) = (r = H.snapshot(H.solve_embedded!(ws, lnT, B...)); H.free_embedded!(ws); r)
    uniform(a) = repeat(a[:, 1, 1], 1, size(a, 2), size(a, 3))

    @testset "validation" begin
        o = H.PoissonMassOptions(logm_cut = -1.0)
        @test isempty(H.option_errors(o, lm0))
        @test any(occursin("logm_cut = 9.0", e) for e in H.option_errors(H.PoissonMassOptions(logm_cut = 9.0), lm0))   # outside the axis
        @test_throws H.InvalidInput H.prepare_embedded(comm, :mhs_poisson_logm; axis = lm0, x = fx.x, y = fx.y, eos = tab, g = g,
                                                       column = H.ColumnMassOptions(), poisson = H.PoissonTauOptions())   # a τ cut
        @test_throws H.InvalidInput H.prepare_embedded(comm, :mhs_poisson_ltau; axis = fx.lt, x = fx.x, y = fx.y, eos = tab, g = g,
                                                       column = H.ColumnTauOptions(), poisson = H.PoissonMassOptions())
        @test_throws H.InvalidInput mk(:column_hse_logm; lm = reverse(lm0))
    end

    @testset "column HSE: P = μ g m (μ $μ)" for μ in (1.0, 0.6)
        r = solve(mk(:column_hse_logm, H.ViewOptions(mu = μ)))
        @test r.status.valid && r.status.status == H.converged
        for i in axes(r.lnP, 3), j in axes(r.lnP, 2)
            @test maximum(abs, r.lnP[:, j, i] .- log.(μ * g .* exp10.(lm0))) < 1e-11
        end
        @test all(diff(r.z; dims = 1) .< 0) && all(diff(r.tau500; dims = 1) .> 0)
        @test abs(mean(r.z_tau1)) < 1e-6                     # gauge: mean z(τ = 1) = 0
        @test all(r.logtau500 .≈ log10.(r.tau500))
    end

    @testset "column HSE: the mass the heights hold" begin
        r = solve(mk(:column_hse_logm))
        m = exp10.(lm0)
        dm = (r.rho[2:end, :, :] .+ r.rho[1:end-1, :, :]) ./ 2 .* (r.z[1:end-1, :, :] .- r.z[2:end, :, :])
        @test median(abs.(dm ./ diff(m) .- 1)) < 0.02        # ∫ρ dz = Δm
    end

    @testset "column HSE: isothermal scale height" begin
        lT = fill(log(6000.0), size(fx.lnT))
        r = solve(mk(:column_hse_logm), lT)
        Hs = r.Pgas ./ (r.rho .* g)
        k = findall(-2 .< lm0 .< 2)
        dz = r.z[k[1:end-1], 1, 1] .- r.z[k[2:end], 1, 1]
        @test maximum(abs.(dz ./ ((Hs[k[1:end-1], 1, 1] .+ Hs[k[2:end], 1, 1]) ./ 2 .* (log(10) .* diff(lm0[k]))) .- 1)) < 2e-3
    end

    # One atmosphere on both axes: the τ-axis column gives m = P/g on its levels; the mass axis then has to give back its
    # heights and its τ. They are different discretisations (T linear in log τ or in log m, opacity by a trapezoid), which
    # differ where one τ level holds many decades of m (hot over cool), so the difference must vanish with refinement.
    @testset "column HSE: τ axis and mass axis converge to one atmosphere" begin
        errs = map((1, 4, 16, 64)) do N
            nk = length(fx.lt)
            l = range(fx.lt[1], fx.lt[end]; length = (nk - 1) * N + 1) |> collect
            T1 = [H.PoissonTau.interp_lt(fx.lnT[:, 1, 1], fx.lt, v) for v in l]
            xs = fx.x[1:4]; ys = fx.y[1:4]
            lT = repeat(T1, 1, 4, 4); B0 = zero(lT)
            run(m, ax, col) = solve(H.prepare_embedded(MPI.COMM_SELF, m; axis = ax, x = xs, y = ys, eos = tab, g = g, column = col), lT, (B0, B0, B0))
            a = run(:column_hse_ltau, l, H.ColumnTauOptions(align = "none"))
            b = run(:column_hse_logm, log10.(a.Pgas[:, 1, 1] ./ g), H.ColumnMassOptions(align = "none"))
            @test b.status.valid
            @test maximum(abs, b.lnP .- a.lnP) < 1e-9
            k = 1:N:length(l)
            (maximum(abs, (b.z .- a.z)[k, 1, 1]), maximum(abs, b.logtau500[k, 1, 1] .- l[k]))
        end
        @info "τ vs mass axis, max |Δz| [m], max |Δ log τ| for 1, 4, 16, 64 × the levels" errs
        @test errs[4][1] < errs[1][1] / 10 && errs[4][2] < errs[1][2] / 10
        @test errs[4][1] < 1e4 && errs[4][2] < 0.01
    end

    @testset "viewing angle ($mode)" for mode in (:column_hse_logm, :mhs_poisson_logm)
        a = solve(mk(mode))
        b = solve(mk(mode, H.ViewOptions(mu = 1.0, azimuth_deg = 30.0, geometry = :inclined)))
        for nm in (:lnP, :z, :rho, :chi500, :z_tau1)
            @test isequal(getfield(a, nm), getfield(b, nm))
        end
        μ = 0.6                                              # plane-parallel μ: the vertical view on log m + log10 μ
        c = solve(mk(mode, H.ViewOptions(mu = μ)))
        d = solve(mk(mode; lm = lm0 .+ log10(μ)))
        @test c.status.valid && d.status.valid
        @test maximum(abs, c.lnP .- d.lnP) < (mode == :column_hse_logm ? 1e-10 : 1e-5)
        dz = c.z .- d.z
        @test maximum(abs, dz .- mean(dz)) < (mode == :column_hse_logm ? 1e-5 : 1.0)
    end

    @testset "Jacobian on the mass axis (μ $μ, φ $φ)" for (μ, φ) in ((1.0, 0.0), (0.6, 45.0), (0.25, 120.0))
        v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
        ws = mk(:column_hse_logm, v)
        par = ws.par; PT = H.PoissonTau
        lnP, z0, _ = PT.column_hse_tau(par, lm0, fx.lnT, fx.Bx, fx.By, fx.Bz, tab, g; mu = μ, mass = true)
        pr, c = PT.tp_setup(par, tab, g, lm0, fx.lnT, fx.Bx, fx.By, fx.Bz, z0, fx.x[2] - fx.x[1], fx.y[2] - fx.y[1];
                            logtau_cut = -1.5, mu = μ, slant = H.slant(v), mass = true)
        kc, n1, nt = pr.kc, pr.n1, pr.nt
        R(u, c) = (PT.evaluate!(pr, u, c); (copy(pr.Rx), copy(pr.Ry), copy(pr.Rz)))
        rng = MersenneTwister(7)
        dv = 0.01 .* randn(rng, size(pr.Rx)); dv[1, :, :] .= 100.0 .* randn(rng, size(c))
        R(lnP, c)
        jx = similar(dv); jy = similar(dv); jz = similar(dv)
        PT.Jop!(pr, jx, jy, jz, dv)
        wx, wy, wz = randn(rng, size(dv)), randn(rng, size(dv)), randn(rng, size(dv)); wz[n1, :, :] .= 0
        out = similar(dv)
        PT.JTop!(pr, out, wx, wy, wz)
        lhs = sum(jx .* wx) + sum(jy .* wy) + sum(jz .* wz); rhs = sum(dv .* out)
        @test abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))
        ε = 1e-4
        shift(s) = (u = copy(lnP); @views u[kc+1:nt, :, :] .+= s .* dv[2:end, :, :]; (u, c .+ s .* dv[1, :, :]))
        Rp = R(shift(ε)...); Rm = R(shift(-ε)...)
        fd = ((Rp[1] .- Rm[1]) ./ 2ε, (Rp[2] .- Rm[2]) ./ 2ε, (Rp[3] .- Rm[3]) ./ 2ε)
        num = sqrt(sum(abs2, fd[1] .- jx) + sum(abs2, fd[2] .- jy) + sum(abs2, fd[3][1:n1-1, :, :] .- jz[1:n1-1, :, :]))
        den = sqrt(sum(abs2, jx) + sum(abs2, jy) + sum(abs2, jz))
        @test num / den < 1e-5
        H.free_embedded!(ws)
    end

    @testset "Poisson solve" begin
        r = solve(mk(:mhs_poisson_logm))
        @test r.status.valid
        @test r.status.steps >= 1 && r.status.phi < r.status.phi0
        kc = H.cut_levels(H.PoissonTauOptions(logtau_cut = -1.5), lm0)[1]
        @test maximum(abs, r.lnP[1:kc, :, :] .- log.(r.Pgas_start[1:kc, :, :])) < 1e-10   # above the cut: hydrostatic, fixed
        @test abs(mean(r.z_tau1)) < 1e-6
        @test r.status.stats["start"] == "column_hse"
        # a horizontally uniform, field-free atmosphere is its own solution
        lT = uniform(fx.lnT); B0 = zero(lT)
        u = solve(mk(:mhs_poisson_logm), lT, (B0, B0, B0))
        @test u.status.status == H.converged && u.status.steps == 0
        ws = mk(:column_hse_logm)
        c = solve(ws, lT, (B0, B0, B0))
        @test maximum(abs, u.lnP .- c.lnP) < 1e-12
        # warm start from the solution of the same cube
        ws = mk(:mhs_poisson_logm)
        a = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz))
        b = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz; start = :previous))
        H.free_embedded!(ws)
        @test b.status.stats["start"] == "previous" && b.status.valid
        @test b.status.phi0 < a.status.phi0 / 10            # (the weights of Φ differ: they come from each start)
    end

    @testset "input file and output ($units)" for (units, off) in (("kg m-2", 0.0), ("g cm-2", -1.0))
        mktempdir() do d
            inp = joinpath(d, "in.h5"); out = joinpath(d, "out.h5"); cfgp = joinpath(d, "c.toml")
            h5open(inp, "w") do f
                f["log10m"] = lm0 .+ off; attributes(f["log10m"])["units"] = units
                f["x"] = fx.x; f["y"] = fx.y
                f["temperature"] = exp.(fx.lnT); f["B_x"] = fx.Bx; f["B_y"] = fx.By; f["B_z"] = fx.Bz
            end
            write(cfgp, """
                [run]
                mode = "column-hse-logm"
                output = "$out"
                [input]
                file = "$inp"
                [physics]
                g = $g
                [eos]
                table_file = "$ETABLE"
                """)
            rc() = cd(() -> H.run_config(cfgp), joinpath(@__DIR__, ".."))
            @test rc() == 0
            r = solve(mk(:column_hse_logm))
            h5open(out, "r") do f
                @test read(f["mass_grid/log10m"]) ≈ lm0
                @test permutedims(read(f["mass_grid/Pgas"]), (3, 2, 1)) ≈ r.Pgas
                @test permutedims(read(f["mass_grid/z_m"]), (3, 2, 1)) ≈ r.z
                @test permutedims(read(f["mass_grid/logtau500"]), (3, 2, 1)) ≈ r.logtau500
                @test permutedims(read(f["mass_grid/z_tau1_m"]), (2, 1)) ≈ r.z_tau1
                @test read(attributes(f)["mode"]) == "column-hse-logm"
            end
            # the wrong axis for the mode, and an unknown unit, are invalid input
            write(cfgp, replace(read(cfgp, String), "column-hse-logm" => "column-hse-ltau"))
            @test rc() == 4
            write(cfgp, replace(read(cfgp, String), "column-hse-ltau" => "column-hse-logm"))
            h5open(inp, "r+") do f
                a = attributes(f["log10m"]); delete!(a, "units"); a["units"] = "erg"
            end
            @test rc() == 4
        end
    end
end
