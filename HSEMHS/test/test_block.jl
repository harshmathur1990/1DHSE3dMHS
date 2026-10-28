# Column-block preconditioner of mhs-poisson-ltau (design/tau/SOLVER_TIME_DESIGN.md D3b): the analytically assembled
# blocks equal the diagonal blocks of the operator (probed with unit vectors), the block solve inverts them, and solves
# preconditioned with them (alone, and with the low-|k| Fourier modes added: D3c) converge.

@isdefined(ETABLE) || include("embedded_helpers.jl")

@testset "column-block preconditioner" begin
    tab = H.load_table(ETABLE)
    fx = load_tau_fixture("11x12", Colon())
    g = 274.0; lt = fx.lt; nt, ny, nx = size(fx.lnT)
    par = H.slab_par(MPI.COMM_WORLD, nx, ny, nt)
    dx = fx.x[2] - fx.x[1]; dy = fx.y[2] - fx.y[1]
    PT = H.PoissonTau

    @testset "exact blocks (μ $μ, φ $φ, $bal balance)" for (μ, φ, bal) in ((1.0, 0.0, :vertical), (0.6, 45.0, :vertical),
                                                                             (0.6, 45.0, :ray), (0.3, 200.0, :vertical))
        v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
        lnP, z0, _ = PT.column_hse_tau(par, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, tab, g; mu = μ)
        pr, c = PT.tp_setup(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, z0, dx, dy; logtau_cut = -4.0, mu = μ,
                            slant = H.slant(v), balance = bal, precond = :block)
        @test pr.prec == :block
        PT.evaluate!(pr, lnP, c); PT.tp_weights!(pr, g, dx); PT.evaluate!(pr, lnP, c)
        PT.setprec!(pr)
        bal == :ray && (pr.mlm = 0.3; PT.setprec!(pr))                      # Marquardt damping in the operator and blocks
        n1 = pr.n1
        e = zeros(n1, ny, nx); out = similar(e)
        for (j, i) in ((1, 1), (ny, nx), (5, 6), (ny, 1))
            L = LowerTriangular(pr.blk[:, :, j, i]); B = L * L'
            A = zeros(n1, n1)
            for m in 1:n1
                fill!(e, 0.0); e[m, j, i] = 1.0
                PT.Aop!(pr, out, e); A[:, m] = out[:, j, i]
            end
            @test maximum(abs, B .- A) <= 1e-10 * maximum(abs, A)
        end
        r = randn(MersenneTwister(3), n1, ny, nx); x = similar(r)
        PT.block_solve!(x, pr.blk, r)
        for (j, i) in ((2, 3), (ny, nx))
            L = LowerTriangular(pr.blk[:, :, j, i])
            @test L * (L' * x[:, j, i]) ≈ r[:, j, i] rtol = 1e-8                 # limited by the conditioning of the block
        end
    end

    @testset "solve with the block preconditioner" begin
        v = H.ViewOptions(mu = 0.5, azimuth_deg = 30.0, geometry = :inclined)
        lnP, z0, _ = PT.column_hse_tau(par, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, tab, g; mu = v.mu)
        run(p; d = :reg) = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, copy(lnP), copy(z0), dx, dy; logtau_cut = -4.0,
                              max_steps = 15, mu = v.mu, slant = H.slant(v), precond = p, damping = d, stop = :gn, lm = :gain,
                              forcing = :ew, stationarity = false)
        f = run(:fourier)
        # the objective of another solve: same start and weights → the same solve; another start uses the weights given
        sv(u0, zz; w = nothing) = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, u0, zz, dx, dy; logtau_cut = -4.0,
                                     max_steps = 5, mu = v.mu, slant = H.slant(v), stop = :gn, lm = :gain, forcing = :ew,
                                     stationarity = false, weights = w)
        f5 = sv(copy(lnP), copy(z0))
        r5 = sv(copy(lnP), copy(z0); w = f5[7].weights)
        @test r5[3] == f5[3] && r5[4] == f5[4] && r5[1] == f5[1]
        w5 = sv(copy(f5[1]), copy(f5[2]); w = f5[7].weights)
        @test w5[7].weights.wc == f5[7].weights.wc && w5[7].weights.ereg == f5[7].weights.ereg
        @test w5[3] <= f5[4] * (1 + 1e-10)                                # starts at the first solve's Φ (same objective)
        for (p, d) in ((:block, :reg), (:hybrid, :reg), (:hybrid, :marquardt))
            b = run(p; d = d)
            @test b[7].reason in (:converged, :max_steps)
            @test b[4] / b[3] < 0.1                                         # Φ/Φ₀
            @test b[4] <= 2 * f[4]
        end
    end
end
