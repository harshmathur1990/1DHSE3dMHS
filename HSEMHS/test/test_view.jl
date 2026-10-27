# Viewing angle in the τ modes (design/tau/VIEWING_ANGLE_DESIGN.md §8): validation, μ = 1 is the vertical view
# bitwise, plane-parallel μ is the vertical view on the shifted axis log τ + log10 μ, and the Jacobian of the inclined
# geometry (adjoint and finite differences) at several angles.

@isdefined(ETABLE) || include("embedded_helpers.jl")
include("view_helpers.jl")

@testset "viewing angle" begin
    comm = MPI.COMM_WORLD
    tab = H.load_table(ETABLE)
    fx = load_tau_fixture("11x12", Colon())
    g = 274.0
    function mk(mode, view; lt = fx.lt, cut = -4.0)
        poi = mode == :mhs_poisson_ltau ? H.PoissonTauOptions(logtau_cut = cut, max_steps = 30, rtol = 1e-4) : nothing
        H.prepare_embedded(comm, mode; axis = lt, x = fx.x, y = fx.y, eos = tab, g = g,
                           column = H.ColumnTauOptions(align = "ptot"), poisson = poi, view = view)
    end
    solve(ws) = (r = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)); H.free_embedded!(ws); r)

    @testset "validation" begin
        @test isempty(H.option_errors(H.ViewOptions()))
        @test !isempty(H.option_errors(H.ViewOptions(mu = 0.0)))
        @test !isempty(H.option_errors(H.ViewOptions(mu = 1.2)))
        @test !isempty(H.option_errors(H.ViewOptions(azimuth_deg = NaN)))
        @test !isempty(H.option_errors(H.ViewOptions(geometry = :spherical)))
        @test H.slant(H.ViewOptions(mu = 0.5)) == (0.0, 0.0)                         # plane-parallel: no slant
        t = H.slant(H.ViewOptions(mu = 0.5, azimuth_deg = 90.0, geometry = :inclined))
        @test t[1] ≈ 0.0 atol = 1e-12
        @test t[2] ≈ sqrt(3.0)
        @test_throws H.InvalidInput mk(:column_hse_ltau, H.ViewOptions(mu = -0.1))
        fz = load_z_fixture("11x12", Colon())
        zprep(col, v) = H.prepare_embedded(comm, :column_hse_z; axis = fz.z, x = fz.x, y = fz.y, eos = tab, g = fz.g,
                                           column = col, view = v)
        @test_throws H.InvalidInput zprep(H.ColumnZOptions(P_ref = fz.P_ref, mu = 0.7), H.ViewOptions(mu = 0.5))
        ws = zprep(H.ColumnZOptions(P_ref = fz.P_ref), H.ViewOptions(mu = 0.5))
        @test ws.column.mu == 0.5
        H.free_embedded!(ws)
    end

    @testset "μ = 1 is the vertical view ($mode, $geom)" for mode in (:column_hse_ltau, :mhs_poisson_ltau),
                                                              geom in (:plane_parallel, :inclined)
        a = solve(mk(mode, H.ViewOptions()))
        b = solve(mk(mode, H.ViewOptions(mu = 1.0, azimuth_deg = 30.0, geometry = geom)))
        for nm in (:lnP, :z, :rho, :chi500, :z_tau1)
            @test isequal(getfield(a, nm), getfield(b, nm))
        end
    end

    @testset "plane-parallel μ = vertical view on log τ + log10 μ ($mode)" for mode in (:column_hse_ltau, :mhs_poisson_ltau)
        μ = 0.6
        a = solve(mk(mode, H.ViewOptions(mu = μ)))
        b = solve(mk(mode, H.ViewOptions(); lt = fx.lt .+ log10(μ), cut = -4.0 + log10(μ)))
        @test a.status.valid && b.status.valid
        tol = mode == :column_hse_ltau ? 1e-10 : 1e-5
        @test maximum(abs, a.lnP .- b.lnP) < tol
        dz = a.z .- b.z                                     # the gauges differ (z(τ_LOS = 1) vs z(τ_vertical = 1))
        @test maximum(abs, dz .- mean(dz)) < (mode == :column_hse_ltau ? 1e-6 : 1.0)
    end

    @testset "inclined τ on the height grid: rays through grid points (φ $φ)" for φ in (0.0, 45.0, 90.0, 180.0, 225.0)
        par = H.slab_par(comm, 12, 10, 9)
        et, ez = shear_exact_error(par, φ)
        @test et < 1e-12
        @test ez < 1e-9
        H.free_parallel!(par)
    end

    @testset "inclined τ on the height grid: horizontally uniform atmosphere ($mode)" for mode in (:column_hse_z, :mhs_poisson_z)
        fz = load_z_fixture("11x12", Colon())
        lnT = repeat(sum(fz.lnT; dims = (2, 3)) ./ (size(fz.lnT, 2) * size(fz.lnT, 3)), 1, size(fz.lnT, 2), size(fz.lnT, 3))
        B0 = zeros(size(lnT))
        run(v) = (poi = mode == :mhs_poisson_z ? H.PoissonZOptions(z_cut_m = 1.0e6, max_steps = 5) : nothing;
                  ws = H.prepare_embedded(comm, mode; axis = fz.z, x = fz.x, y = fz.y, eos = tab, g = fz.g,
                                          column = H.ColumnZOptions(P_ref = fz.P_ref), poisson = poi, view = v);
                  r = H.snapshot(H.solve_embedded!(ws, lnT, B0, B0, B0)); H.free_embedded!(ws); r)
        a = run(H.ViewOptions(mu = 0.7))
        b = run(H.ViewOptions(mu = 0.7, azimuth_deg = 30.0, geometry = :inclined))
        @test a.status.valid && b.status.valid
        @test maximum(abs.(a.tau500 .- b.tau500) ./ a.tau500) < 1e-12
        @test maximum(abs, a.z_tau1 .- b.z_tau1) < 1e-6
        @test a.Pgas == b.Pgas                               # the pressure does not depend on the view
    end

    @testset "Jacobian of the inclined geometry (μ $μ, φ $φ)" for (μ, φ) in ((1.0, 0.0), (0.8, 0.0), (0.6, 45.0), (0.5, 200.0))
        v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
        ws = mk(:column_hse_ltau, v)
        par = ws.par; lt = fx.lt
        lnP, z0, _ = H.PoissonTau.column_hse_tau(par, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, tab, g; mu = μ)
        dx = fx.x[2] - fx.x[1]; dy = fx.y[2] - fx.y[1]
        pr, c = H.PoissonTau.tp_setup(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, z0, dx, dy; logtau_cut = -4.0,
                                      mu = μ, slant = H.slant(v))
        kc, n1, nt = pr.kc, pr.n1, pr.nt
        R(u, c) = (H.PoissonTau.evaluate!(pr, u, c); (copy(pr.Rx), copy(pr.Ry), copy(pr.Rz)))
        rng = MersenneTwister(7)
        dv = 0.01 .* randn(rng, size(pr.Rx)); dv[1, :, :] .= 100.0 .* randn(rng, size(c))
        R(lnP, c)
        jx = similar(dv); jy = similar(dv); jz = similar(dv)
        H.PoissonTau.Jop!(pr, jx, jy, jz, dv)
        wx, wy, wz = randn(rng, size(dv)), randn(rng, size(dv)), randn(rng, size(dv)); wz[n1, :, :] .= 0
        out = similar(dv)
        H.PoissonTau.JTop!(pr, out, wx, wy, wz)
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
end
