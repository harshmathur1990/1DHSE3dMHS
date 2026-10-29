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
        @test all(q -> q.aprod >= q.its + 1 && q.minv >= 1, qs[2:end])           # restarts and interval residuals counted
        # the qualification preconditioner is the column blocks whatever the outer one (restored), Fourier on request;
        # both intervals contain the dense decrement
        for p0 in (:fourier, :block, :hybrid)
            pr.prec = p0
            qb = PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq; tol = 1e-8, maxit = 400, gn_tol = 1e-3)
            qf = PT.qualify_stop(pr, par, similar(b), similar(b), e0, Φq; tol = 1e-8, maxit = 400, gn_tol = 1e-3, precond = :fourier)
            @test pr.prec == p0 && qb.precond == :block && qf.precond == :fourier
            @test qb.lower <= dex * (1 + 1e-9) <= qb.upper * (1 + 2e-9) && qf.lower <= dex * (1 + 1e-9) <= qf.upper * (1 + 2e-9)
        end
        pr.prec = :block; PT.setprec!(pr)
        # restarted PCG (x0): an exact or within-tol start returns at once; b = 0 gives xs = 0 whatever the start; the
        # zero start converges with one A and one M⁻¹ per iteration; a restart from a partial solve reaches the tol
        PT.setprec!(pr)
        xe = reshape(cholesky(Symmetric((A .+ A') ./ 2)) \ vec(b), size(b)); t = similar(b)
        tr(x) = (PT.Aop!(pr, t, x); norm(vec(b .- t)) / norm(vec(b)))
        x = copy(xe); cn = zeros(Int, 2)
        k, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, b; tol = 1e-6, x0 = true, cnt = cn)
        @test k == 0 && rel < 1e-6 && x == xe && cn == [1, 0]
        x = xe .* (1 + 1e-9); k, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, b; tol = 1e-6, x0 = true)
        @test k == 0 && rel ≈ tr(x)
        x = copy(xe); k, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, zero(b); tol = 1e-6, x0 = true)
        @test k == 0 && rel == 0 && all(==(0), x)
        x = similar(b); cn = zeros(Int, 2)
        k, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, b; tol = 1e-8, maxit = 400, cnt = cn)
        @test rel < 1e-8 && cn == [k, k] && tr(x) < 1e-7
        x = similar(b); PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, b; tol = 1e-12, maxit = 5)
        k, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, x, b; tol = 1e-8, maxit = 400, x0 = true)
        @test k >= 1 && rel < 1e-8 && tr(x) < 1e-7
        # outcomes and breakdown protection on toy operators: xs stays the last good (finite) iterate, the returned
        # residual is the true one, and the flag names the exit
        fl = Ref(:none); pid!(_, o, r) = (o .= r)
        bt = ones(3, 4, 5); xt = zeros(3, 4, 5)
        k, rel = PT.pcg!((_, o, v) -> (o .= 2 .* v), pid!, par, nothing, xt, bt; tol = 1e-10, flag = fl)
        @test fl[] == :converged && k == 1 && xt ≈ bt ./ 2
        dg = reshape(collect(1.0:60.0), 3, 4, 5); xt .= 0
        k, rel = PT.pcg!((_, o, v) -> (o .= dg .* v), pid!, par, nothing, xt, bt; tol = 1e-12, maxit = 2, flag = fl)
        @test fl[] == :maxit && k == 2 && rel > 1e-12
        xt .= 1; PT.pcg!((_, o, v) -> (o .= v), pid!, par, nothing, xt, zero(bt); flag = fl)
        @test fl[] == :zero_rhs && all(==(0), xt)
        xt .= 0.5; k, rel = PT.pcg!((_, o, v) -> (o .= 2 .* v), pid!, par, nothing, xt, bt; x0 = true, flag = fl)
        @test fl[] == :start_within_tol && k == 0 && rel == 0
        cn = zeros(Int, 2)
        k, rel = PT.pcg!((_, o, v) -> (o .= .-v), pid!, par, nothing, xt, bt; flag = fl, cnt = cn)    # indefinite
        @test fl[] == :breakdown_pAp && k == 0 && all(==(0), xt) && rel ≈ 1 && cn == [2, 1]
        k, rel = PT.pcg!((_, o, v) -> (o .= 1e-320 .* v), pid!, par, nothing, xt, bt; flag = fl)      # α overflows
        @test fl[] == :nonfinite_step && all(==(0), xt) && rel ≈ 1
        k, rel = PT.pcg!((_, o, v) -> (o .= 1e-300 .* v), pid!, par, nothing, xt, 1e10 .* bt; flag = fl)  # x + αp overflows
        @test fl[] == :nonfinite_update && all(==(0), xt) && rel ≈ 1
        k, rel = PT.pcg!((_, o, v) -> (o .= 1e-307 .* v), pid!, par, nothing, xt, bt; flag = fl)     # Σx overflows, x finite
        @test fl[] == :converged && k == 1 && all(isfinite, xt) && !isfinite(sum(xt)) && xt ≈ 1e307 .* bt
        k, rel = PT.pcg!((_, o, v) -> (o .= v), (_, o, r) -> (o .= -r), par, nothing, xt, bt; flag = fl)  # M⁻¹ not SPD
        @test fl[] == :breakdown_rz && all(==(0), xt)
        nA = Ref(0); sick!(_, o, v) = (nA[] += 1; o .= nA[] <= 2 ? dg .* v : NaN)                 # A fails at call 3
        k, rel = PT.pcg!(sick!, pid!, par, nothing, xt, bt; tol = 1e-12, maxit = 50, flag = fl)
        @test fl[] == :breakdown_pAp && k == 2 && all(isfinite, xt) && any(!=(0), xt) && rel == Inf
        xt .= NaN; k, rel = PT.pcg!((_, o, v) -> (o .= v), pid!, par, nothing, xt, bt; x0 = true, flag = fl)
        @test fl[] == :nonfinite_start && k == 0 && rel == Inf
        # EOS domain: the fraction to the table boundary along δ, and the cells evaluate! rejects, by kind
        dd = zero(b); dd[2:end, :, :] .= 1.0
        Yhi = tab.Y[end] - 2PT.EPS_PROBE; umax = maximum(lnP[pr.kc+1:end, :, :])
        @test PT.feas_scale(pr, lnP, dd) ≈ Yhi - umax
        @test PT.feas_scale(pr, lnP, -dd) ≈ minimum(lnP[pr.kc+1:end, :, :]) - (tab.Y[1] + 2PT.EPS_PROBE)
        @test PT.feas_scale(pr, lnP, zero(b)) == Inf
        # one moving cell against the upper edge: before the safety bound, in the strip between one and two probe
        # widths, at the safety bound, at and beyond the evaluator bound; inward from outside either edge; non-finite
        ε = PT.EPS_PROBE; Y1 = tab.Y[1]; Y2 = tab.Y[end]; kl = size(lnP, 1)
        onecell(d) = (q = zero(b); q[end, 1, 1] = d; q)
        fs(v, d) = (uu = copy(lnP); uu[kl, 1, 1] = v; PT.feas_scale(pr, uu, onecell(d)))
        @test fs(Y2 - 1.0, 1.0) ≈ 1.0 - 2ε
        @test fs(Y2 - 1.5ε, 1.0) ≈ 0.5ε
        @test fs(Y2 - 1.5ε, 2.0) ≈ 0.25ε
        @test fs(Y2 - 2ε, 1.0) ≈ ε
        @test fs(Y2 - ε, 1.0) == 0 && fs(Y2 + 0.1, 1.0) == 0 && fs(Y1 + ε, -1.0) == 0 && fs(Y1 - 0.1, -1.0) == 0
        @test fs(Y2 - 1.5ε, -1.0) ≈ (Y2 - 1.5ε) - (Y1 + 2ε)                 # moving away from the strip's edge
        @test fs(Y2 + 0.1, -1.0) ≈ (Y2 + 0.1) - (Y1 + 2ε)                   # inward from above: the far edge only
        @test fs(Y1 - 0.1, 1.0) ≈ (Y2 - 2ε) - (Y1 - 0.1)                    # inward from below
        @test fs(Y2 - 1.0, NaN) == 0 && fs(NaN, 1.0) == 0 && fs(Y2 - 1.0, Inf) == 0
        @test fs(NaN, 0.0) == Inf                                           # a cell that does not move is ignored
        @test_throws ArgumentError PT.feas_scale(pr, lnP, dd; margin = ε / 2)
        @test PT.eos_cells(pr, lnP).total == 0
        uh = copy(lnP); uh[end, 1, 1] = tab.Y[end] + 0.1; uh[end, 2, 1] = tab.Y[end] - PT.EPS_PROBE / 2
        e = PT.eos_cells(pr, uh)
        @test e.hi == 1 && e.probe_hi == 1 && e.total == 2 && e.worst.level == size(lnP, 1) && e.worst.lnP == uh[end, 1, 1]
        @test PT.evaluate!(pr, uh, cc)[2] == 2
        PT.evaluate!(pr, lnP, cc)
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
        q5.step == u5[5] && q5.outcome != :invalid && @test u5[7].work.final_cg == 0 && u5[7].work.final_aprod == 0
        @test u5[7].work.qual_aprod >= u5[7].work.qual_cg && u5[7].work.aprod >= u5[7].work.cg && u5[7].work.total_s > 0
        @test q5.precond == :block && u5[7].gn_decrement_approx == (q5.step != u5[5] || q5.outcome != :qualified)
        u6 = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, copy(lnP), copy(z0), dx, dy; logtau_cut = -4.0,
                          max_steps = 2, mu = v.mu, slant = H.slant(v), cg_maxit = 1)    # final diagnostic capped at 1 it
        @test u6[7].gn_decrement_approx && u6[7].work.final_cg == 1
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

    # round-off stationarity (Codex round 5, A): an exact discrete equilibrium (horizontally uniform, field-free) ends
    # :converged at the round-off floor, not :no_descent; restarting from it returns at once; a small perturbation
    # converges again; a non-stationary state without a usable direction (cg_maxit = 0) still ends :no_descent
    @testset "round-off stationarity" begin
        lTu = repeat(fx.lnT[:, 1:1, 1:1], 1, size(fx.lnT, 2), size(fx.lnT, 3)); B0 = zero(fx.Bx)
        lnPu, zu, _ = PT.column_hse_tau(par, lt, lTu, B0, B0, B0, tab, g)
        sv0(u0, zz; kw...) = PT.solve_tau(par, tab, g, lt, lTu, B0, B0, B0, u0, zz, dx, dy; logtau_cut = -4.0, stop = :gn,
                                          max_steps = 25, kw...)
        r1 = sv0(copy(lnPu), copy(zu))
        @test r1[7].reason == :converged
        zr = copy(r1[2])
        r2 = sv0(copy(r1[1]), copy(zr))                                 # restart from the returned state
        @test r2[7].reason == :converged && r2[5] <= 1
        up = copy(r1[1]); kc = r1[6]
        for k in kc+1:size(up, 1); up[k, :, :] .+= 1e-3 .* cos.(2π .* (0:size(up, 3)-1)' ./ size(up, 3)); end
        r3 = sv0(up, copy(zr))
        @test r3[7].reason == :converged && r3[5] >= 1 && r3[7].roundoff != :start
        for r in (r1, r2, r3); @test isfinite(r[4]) && isfinite(r[7].phi_floor) && r[7].phi_floor > 0; end
        # the same equilibrium in a translated height gauge (+1 Mm): still accepted at once; the floor moves (z enters
        # its magnitudes as the arithmetic stores it) but not materially
        r5 = sv0(copy(r1[1]), zr .+ 1e6)
        @test r5[7].reason == :converged && r5[5] <= 1 && isfinite(r5[7].phi_floor)
        @test 1 / 2 < r5[7].phi_floor / r2[7].phi_floor < 2                  # measured: +0.7 %
        lnP, z0, _ = PT.column_hse_tau(par, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, tab, g)
        r4 = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, copy(lnP), copy(z0), dx, dy; logtau_cut = -4.0,
                          max_steps = 5, cg_maxit = 0)
        @test r4[7].reason == :no_descent && r4[7].roundoff == :none && r4[4] > r4[7].phi_floor
        @test PT.ROUNDOFF_K == 64
        # precond = :switch: Fourier until a CG solve ends at its limit, column blocks after; without a limit reached it
        # is the Fourier solve exactly
        sw(pc; kw...) = PT.solve_tau(par, tab, g, lt, fx.lnT, fx.Bx, fx.By, fx.Bz, copy(lnP), copy(z0), dx, dy;
                                     logtau_cut = -4.0, max_steps = 6, stop = :gn, precond = pc, kw...)
        a = sw(:switch; cg_maxit = 400); b = sw(:fourier; cg_maxit = 400)
        @test a[7].work.switch_step == 0 && a[1] == b[1] && a[4] == b[4]
        c = sw(:switch; cg_maxit = 2)
        @test c[7].work.switch_step >= 1 && c[5] > c[7].work.switch_step && c[4] < c[3] && all(isfinite, c[1])
        @test sw(:fourier; cg_maxit = 2)[7].work.switch_step == 0
    end
end
