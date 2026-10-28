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
        # stop qualification (qualify_stop): heavy damping makes the step's pred small though the gradient is not; the
        # decrement interval [L, U] contains bᵀA⁻¹b/Φ of the dense solve at any CG budget and narrows with it; b = 0
        # qualifies exactly; non-finite or non-positive inputs are :invalid; a straddling interval is :undecided
        pr, cc = PT.tp_setup(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, z0, dx, dy; logtau_cut = -4.0, mu = v.mu,
                             slant = H.slant(v), precond = :block)
        PT.evaluate!(pr, lnP, cc); PT.tp_weights!(pr, g, dx); Φq, _ = PT.evaluate!(pr, lnP, cc)
        e0 = copy(pr.ereg); b = zeros(pr.n1, pr.ny, pr.nx); d = similar(b)
        PT.JTop!(pr, b, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); b .*= -1
        pr.ereg = e0 .+ 1e6 .* e0 ./ 1e-2; PT.setprec!(pr)
        PT.pcg!(PT.Aop!, PT.Minv!, par, pr, d, b; tol = 1e-6, maxit = 400)
        pdamped = H.gdot(par, d, b) / Φq
        pr.ereg = copy(e0)
        par.nranks == 1 || error("the dense check needs one rank")
        n = length(b); A = zeros(n, n); ev = zero(b); ov = zero(b)
        for k in 1:n
            fill!(ev, 0.0); ev[k] = 1.0; PT.Aop!(pr, ov, ev); A[:, k] = vec(ov)
        end
        dex = dot(vec(b), cholesky(Symmetric((A .+ A') ./ 2)) \ vec(b)) / Φq          # bᵀA⁻¹b / Φ
        qs = [PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq; tol = 1e-12, maxit = m, gn_tol = 1e-3, refine = false)
              for m in (1, 5, 30, 400)]
        @test all(q -> q.lower <= dex * (1 + 1e-9) && dex <= q.upper * (1 + 1e-9), qs)
        @test qs[end].upper - qs[end].lower < 1e-3 * (qs[1].upper - qs[1].lower)
        @test pdamped < 1e-3 && qs[end].pred > 100 * pdamped && qs[end].outcome == :not_stationary
        @test pr.ereg == e0 && pr.mlm == 0 && qs[end].step == 0 && qs[end].phi == Φq
        q1 = PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq; tol = 1e-12, maxit = 1, gn_tol = dex)
        @test q1.its == 1 && q1.outcome == :undecided
        qq = PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq; tol = 1e-2, maxit = 400, gn_tol = 2dex)
        @test qq.outcome == :qualified && qq.upper < 2dex                       # refined until the interval decides
        @test PT.qualify_stop(pr, par, similar(b), similar(b), e0, NaN).outcome == :invalid
        @test PT.qualify_stop(pr, par, similar(b), similar(b), vcat(0.0, e0[2:end]), Φq).outcome == :invalid
        R0 = (copy(pr.Rx), copy(pr.Ry), copy(pr.Rz)); pr.Rx .= 0; pr.Ry .= 0; pr.Rz .= 0
        qz = PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq)
        @test qz.outcome == :qualified && qz.lower == 0 && qz.upper == 0 && qz.true_rel == 0 && qz.grad_norm == 0
        pr.Rx .= R0[1]; pr.Ry .= R0[2]; pr.Rz .= R0[3]
        u5 = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, copy(lnP), copy(z0), dx, dy; logtau_cut = -4.0,
                          max_steps = 10, mu = v.mu, slant = H.slant(v), stop = :gn, gn_tol = 1.0, gn_consec = 1,
                          lm = :gain, forcing = :ew, qual_tol = 1e-12, qual_maxit = 1)
        q5 = u5[7].stop_check
        @test q5.outcome in (:undecided, :qualified, :not_stationary)
        @test q5.step >= 1 && q5.its >= 1
        q5.outcome == :undecided && @test u5[7].reason == :unresolved && u5[5] == q5.step
        q5.step == u5[5] && q5.outcome != :invalid && @test u5[7].gn_decrement == q5.pred      # reused, not re-solved
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
        @test w5[7].weights.kc == f5[7].weights.kc && w5[7].weights.n1 == f5[7].weights.n1
        wbad = merge(f5[7].weights, (ereg = -f5[7].weights.ereg,))
        @test_throws ErrorException sv(copy(lnP), copy(z0); w = wbad)
        @test_throws ErrorException sv(copy(lnP), copy(z0); w = merge(f5[7].weights, (wf = f5[7].weights.wc,)))
        @test_throws ErrorException sv(copy(lnP), copy(z0); w = merge(f5[7].weights, (kc = f5[7].weights.kc + 1,)))
        for (p, d) in ((:block, :reg), (:hybrid, :reg), (:hybrid, :marquardt))
            b = run(p; d = d)
            @test b[7].reason in (:converged, :max_steps)
            @test b[4] / b[3] < 0.1                                         # Φ/Φ₀
            @test b[4] <= 2 * f[4]
        end
    end
end
