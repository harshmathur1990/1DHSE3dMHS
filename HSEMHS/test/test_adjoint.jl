# Adjoint of the embedded equilibrium (design/adjoint): the building blocks against central differences, then every mode
# end to end by the dot-product check of adjoint_helpers.jl (single rank here; mpi/mpi_adjoint.jl on 2 and 3 ranks).

@isdefined(adjoint_dot_error) || include("adjoint_helpers.jl")

"""Relative error of a central-difference directional derivative against an adjoint pairing, best over `hs`."""
function fd_pair_error(f, x, dx, ybar, rhs; hs = (1e-4, 1e-5, 1e-6, 1e-7))
    best = Inf
    for h in hs
        lhs = sum(ybar .* (f(x .+ h .* dx) .- f(x .- h .* dx))) / (2h)
        best = min(best, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
    end
    return best
end

@testset "equilibrium adjoint" begin
    comm = MPI.COMM_WORLD
    tab = H.load_table(ETABLE)
    rng = MersenneTwister(7)

    @testset "EOS: state_d is the derivative of state" begin
        fx = load_tau_fixture("11x12", Colon())
        ws = H.prepare_embedded(comm, :column_hse_ltau; axis = fx.lt, x = fx.x, y = fx.y, eos = tab, g = 274.0,
                                column = H.ColumnTauOptions(align = "none"))
        r = H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
        worst = 0.0
        for _ in 1:400
            k, j, i = rand(rng, 1:size(fx.lnT, 1)), rand(rng, 1:size(fx.lnT, 2)), rand(rng, 1:size(fx.lnT, 3))
            X = fx.lnT[k, j, i]; Y = r.lnP[k, j, i]
            d = H.state_d(tab, X, Y)
            v0 = H.state(tab, X, Y)
            @test all(d[1:3] .== v0[1:3])                     # the same values as state
            for (col, dir) in ((4, (1, 0)), (5, (0, 1)), (6, (1, 0)), (7, (0, 1)), (8, (1, 0)), (9, (0, 1)))
                fld = (col - 2) ÷ 2                           # 1 ρ, 2 nₑ, 3 χ
                h = 1e-6
                fd = (H.state(tab, X + h * dir[1], Y + h * dir[2])[fld] - H.state(tab, X - h * dir[1], Y - h * dir[2])[fld]) / (2h)
                worst = max(worst, abs(fd - d[col]) / max(abs(fd), 1e-3))
            end
        end
        @test worst < 1e-6
        H.free_embedded!(ws)
    end

    @testset "τ on the height grid ($top)" for top in (:exp_tail, :half_cell, :zero)
        z = collect(range(2.0e6, -4.0e5; length = 60))
        chi0 = exp.(range(log(1e-12), log(1e-4); length = 60)) .* (1 .+ 0.3 .* sin.(1:60))
        τ = zeros(60)
        method = H.tau_column!(τ, chi0, z; top = top, mu = 0.8)
        f(c) = (t = zeros(60); H.tau_column!(t, c, z; top = top, mu = 0.8); t)
        τbar = randn(rng, 60) ./ τ
        χbar = zeros(60)
        H.tau_column_vjp!(χbar, τbar, chi0, z, method; mu = 0.8)
        dχ = chi0 .* randn(rng, 60)
        @test fd_pair_error(f, chi0, dχ, τbar, sum(χbar .* dχ)) < 1e-8
    end

    @testset "log τ on the mass axis" begin
        fx = load_tau_fixture("11x12", Colon())
        lm = fx.lt .+ 1.0
        ws = H.prepare_embedded(comm, :column_hse_logm; axis = lm, x = fx.x, y = fx.y, eos = tab, g = 274.0,
                                column = H.ColumnMassOptions(align = "none"))
        r = H.snapshot(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)); H.free_embedded!(ws)
        lbar = randn(rng, size(fx.lnT))
        Xbar = zeros(size(fx.lnT)); Ybar = zeros(size(fx.lnT))
        H.PoissonTau.logtau_mass_vjp!(Xbar, Ybar, lbar, fx.lnT, r.lnP, lm, tab)
        dX = 0.05 .* randn(rng, size(fx.lnT)); dY = 0.05 .* randn(rng, size(fx.lnT))
        fX(X) = H.PoissonTau.logtau_mass(X, r.lnP, lm, tab)
        fY(Y) = H.PoissonTau.logtau_mass(fx.lnT, Y, lm, tab)
        @test fd_pair_error(fX, fx.lnT, dX, lbar, sum(Xbar .* dX)) < 1e-7
        @test fd_pair_error(fY, r.lnP, dY, lbar, sum(Ybar .* dY)) < 1e-7
    end

    @testset "inclined τ (azimuth $φ)" for φ in (0.0, 30.0, 90.0)
        fz = load_z_fixture("11x12", Colon())
        nz, ny, nx = size(fz.lnT)
        par = H.slab_par(comm, nx, ny, nz)
        v = H.ViewOptions(mu = 0.8, azimuth_deg = φ, geometry = :inclined); t = H.slant(v)
        dx = fz.x[2] - fz.x[1]; dy = fz.y[2] - fz.y[1]
        chi0 = exp.(-(fz.z .- fz.z[end]) ./ 1.5e5 .- 9.0) .* (1 .+ 0.2 .* rand(rng, nz, ny, nx))
        f(c) = (τ = similar(c); H.inclined_tau!(τ, zeros(ny, nx), zeros(UInt8, ny, nx), par, c, fz.z, dx, dy, 0.8, t); τ)
        ttm = zeros(UInt8, ny, nx); τ = similar(chi0)
        H.inclined_tau!(τ, zeros(ny, nx), ttm, par, chi0, fz.z, dx, dy, 0.8, t)
        τbar = randn(rng, size(τ)) ./ τ
        χbar = zeros(size(chi0))
        H.inclined_tau_vjp!(χbar, τbar, par, chi0, fz.z, dx, dy, 0.8, t, ttm)
        dχ = chi0 .* randn(rng, size(chi0))
        @test fd_pair_error(f, chi0, dχ, τbar, sum(χbar .* dχ)) < 1e-8
        H.free_parallel!(par)
    end

    @testset "Heun column on the $(mass ? "mass" : "τ") axis: replay and frozen-step adjoint" for mass in (false, true)
        fx = load_tau_fixture("11x12", Colon())
        lt = mass ? fx.lt .+ 1.0 : fx.lt
        X = fx.lnT[:, 3, 4]; n = length(lt)
        lnP = zeros(n); z = zeros(n)
        tape = H.PoissonTau.HeunTape()
        @test H.PoissonTau.hse_tau!(lnP, z, X, lt, tab, 274.0; mu = 0.9, mass = mass, tape = tape)
        lnP2 = zeros(n); z2 = zeros(n)
        H.PoissonTau.hse_tau_replay!(lnP2, z2, X, lt, tab, 274.0, tape; mu = 0.9, mass = mass)
        @test lnP2 == lnP && z2 == z                          # the replay is the recorded computation
        ubar = randn(rng, n); zbar = randn(rng, n) ./ 1e5
        Xbar = zeros(n)
        H.PoissonTau.hse_tau_vjp!(Xbar, ubar, zbar, X, lt, tab, 274.0, tape; mu = 0.9, mass = mass)
        out(Xp) = (a = zeros(n); b = zeros(n); H.PoissonTau.hse_tau_replay!(a, b, Xp, lt, tab, 274.0, tape; mu = 0.9, mass = mass); vcat(a, b))
        dX = 0.05 .* randn(rng, n)
        @test fd_pair_error(out, X, dX, vcat(ubar, zbar), sum(Xbar .* dX)) < 1e-8
    end

    @testset "RK4 column (anchor $anchor, adapt $adapt)" for anchor in (:bottom, :top, :height), adapt in (true, false)
        fz = load_z_fixture("11x12", Colon())
        X = fz.lnT[:, 5, 6]; z = fz.z; n = length(z)
        o = H.HSEOpts(anchor = anchor, P_ref = fz.P_ref, z_ref = (z[1] + z[end]) / 2, g = fz.g, adapt = adapt)
        u = fill(NaN, n); m = fill(NaN, n)
        tape = H.RK4Tape()
        H.hse_column!(u, m, X, z, tab, o; tape = tape)
        u2 = fill(NaN, n)
        H.hse_column_replay!(u2, X, z, tab, o, tape)
        @test u2 == u
        ubar = randn(rng, n)
        Xbar = zeros(n)
        H.hse_column_vjp!(Xbar, ubar, X, z, tab, o, tape)
        out(Xp) = (a = fill(NaN, n); H.hse_column_replay!(a, Xp, z, tab, o, tape); a)
        dX = 0.05 .* randn(rng, n)
        @test fd_pair_error(out, X, dX, ubar, sum(Xbar .* dX)) < 1e-8
    end

    @testset "embedded: $(case.name)" for case in adjoint_cases()
        inp = adjoint_inputs(case.grid)
        r = adjoint_dot_error(comm, case, inp, tab)
        @info "adjoint check" case = case.name err = r.err lhs = r.lhs rhs = r.rhs status = r.status
        @test r.err < case.tol
        @test r.status in (:exact, :stationary)
    end

    @testset "column modes are local: a cotangent at one node reaches one column" for (mode, grid) in
            ((:column_hse_ltau, :ltau), (:column_hse_logm, :logm), (:column_hse_z, :z))
        inp = adjoint_inputs(grid)
        case = (mode = mode, view = PPV, align = "none", anchor = :bottom, adapt = true, top = :exp_tail, poisson = (;))
        ws = adjoint_workspace(comm, case, inp, tab)
        H.solve_embedded!(ws, inp.lnT, zero(inp.lnT), zero(inp.lnT), zero(inp.lnT); record = true)
        bar = H.EquilibriumCotangent(ws)
        bar.rho[20, 4, 5] = 1.0
        adj = H.adjoint_embedded!(ws, bar)
        others = copy(adj.lnT); others[:, 4, 5] .= 0.0
        @test maximum(abs, others) == 0.0
        @test maximum(abs, adj.lnT[:, 4, 5]) > 0.0
        H.free_embedded!(ws)
    end

    @testset "refusals" begin
        inp = adjoint_inputs(:ltau)
        case = (mode = :column_hse_ltau, view = PPV, align = "ptot", anchor = :bottom, adapt = true, top = :exp_tail, poisson = (;))
        ws = adjoint_workspace(comm, case, inp, tab)
        H.solve_embedded!(ws, inp.lnT, inp.Bx, inp.By, inp.Bz)
        @test_throws H.InvalidInput H.adjoint_embedded!(ws, H.EquilibriumCotangent(ws))          # not recorded
        H.solve_embedded!(ws, inp.lnT, inp.Bx, inp.By, inp.Bz; record = true)
        H.solve_embedded!(ws, inp.lnT .+ 0.01, inp.Bx, inp.By, inp.Bz)                            # a later solve
        @test_throws H.InvalidInput H.adjoint_embedded!(ws, H.EquilibriumCotangent(ws))
        ne = H.snapshot(H.solve_embedded!(ws, inp.lnT, inp.Bx, inp.By, inp.Bz)).ne
        H.solve_embedded!(ws, inp.lnT; lnne = log.(ne), record = true)                             # given nₑ: forward only
        @test_throws H.InvalidInput H.adjoint_embedded!(ws, H.EquilibriumCotangent(ws))
        bad = H.EquilibriumCotangent(ws); bad.rho = zeros(3, 3, 3)
        H.solve_embedded!(ws, inp.lnT, inp.Bx, inp.By, inp.Bz; record = true)
        @test_throws H.InvalidInput H.adjoint_embedded!(ws, bad)                                  # wrong shape
        H.free_embedded!(ws)
    end
end
