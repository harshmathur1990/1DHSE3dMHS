# M5g: face-bound (hinge / log-barrier) terms, the barrier-safe preconditioner and the bounded joint GN step
# (Codex M5f review, findings 1, 2, 5 and 7).

"""Small synthetic problem with every z-face at |c| = 3 − slack against hard bounds 3 (uniform B_z, b̄ shifted)."""
function nearbound_problem(slack; lambda = 0.1, kind = :barrier)
    par = H.init_parallel(4, 4, 9)
    pb, x, us, uc, grid = H.sc1_problem(par, 4, 4, 9; Bobs = (0.0, 0.0, 0.005), lambda_F = 1e-4, lambda_Q = 1e-3)
    sig = 1 / sqrt(pb.invsf2[3][1, 1, 0])
    H.setscalar!(par, x[5], 0.005 + (3.0 - slack) * sig)
    hard = (H.newfield(par; fill = 3.0), H.newfield(par; fill = 3.0), H.newfield0(par; fill = 3.0))
    H.exchange_halos!(par, collect(hard))
    kind === nothing || H.set_face_hinge!(pb, hard; lambda = lambda, kind = kind)
    return par, pb, x, hard, sig
end

randdir(par, x; seed = 1, fields = 2:5) = (Random.seed!(seed); d = H.similar_state(x);
    for f in fields; f == 5 ? H.setscalar!(par, d[5], randn()) : H.owned_map!((i, j, k) -> randn(), d[f], par); end; d)

@testset "M5g: face hinge/barrier parameters are validated" begin
    par, pb, x, hard, _ = nearbound_problem(0.5; kind = nothing)
    @test_throws H.InvalidInput H.set_face_hinge!(pb, hard; lambda = -1.0)
    @test_throws H.InvalidInput H.set_face_hinge!(pb, hard; lambda = 0.0, kind = :barrier)
    @test_throws H.InvalidInput H.set_face_hinge!(pb, hard; lambda = NaN, kind = :barrier)
    bad = deepcopy(hard); bad[2][2, 3, 4] = NaN
    @test_throws H.InvalidInput H.set_face_hinge!(pb, bad; lambda = 0.1, kind = :barrier)
    zero_ = deepcopy(hard); zero_[3][1, 1, 0] = 0.0
    @test_throws H.InvalidInput H.set_face_hinge!(pb, zero_; lambda = 0.1, kind = :barrier)
    @test H.set_face_hinge!(pb, zero_; lambda = 0.0, kind = :hinge) === pb          # s = 0 is a valid hinge threshold
    @test H.set_face_hinge!(pb, nothing).hinge === nothing
end

@testset "M5g: barrier gradient and Hessian near the bound (finite differences inside the domain)" begin
    for slack in (0.5, 1e-3)
        par, pb, x, hard, sig = nearbound_problem(slack; lambda = 10.0)   # barrier-dominated: no cancellation in the FD
        w = H.SMHSWork(par); g = H.similar_state(x); f0 = H.fg!(g, pb, w, x)[1]
        @test isfinite(f0)
        d = randdir(par, x; seed = 3)
        # x ± h d stays well inside: h is 1e-3 of the fraction to the boundary along ±d
        H.objective!(pb, w, x); wsc = H.SMHSWork(par)
        a = min(H.face_fraction(pb, w, wsc, d, hard), H.face_fraction(pb, w, wsc, H.lincomb!(H.similar_state(x), -1.0, d, 0.0, d, par), hard))
        h = 1e-3 * a                                                      # both x ± h d well inside
        gd = H.dotN(par, g, d)
        fp = H.objective!(pb, H.SMHSWork(par), H.lincomb!(H.similar_state(x), 1.0, x, h, d, par))[1]
        fm = H.objective!(pb, H.SMHSWork(par), H.lincomb!(H.similar_state(x), 1.0, x, -h, d, par))[1]
        @test isfinite(fp) && isfinite(fm)
        @test abs((fp - fm) / 2h - gd) <= 1e-5 * abs(gd)
        # isolated barrier Hessian: (H_GN with − without the face term) d against the FD of the same gradient difference
        hg = pb.hinge
        function gbar(xx)
            gw = H.similar_state(x); H.fg!(gw, pb, H.SMHSWork(par), xx)
            pb.hinge = nothing; g0 = H.similar_state(x); H.fg!(g0, pb, H.SMHSWork(par), xx); pb.hinge = hg
            return H.lincomb!(gw, 1.0, gw, -1.0, g0, par)
        end
        gp = gbar(H.lincomb!(H.similar_state(x), 1.0, x, h, d, par)); gm = gbar(H.lincomb!(H.similar_state(x), 1.0, x, -h, d, par))
        fd = H.dotN(par, d, H.lincomb!(gp, 1 / 2h, gp, -1 / 2h, gm, par))
        H.objective!(pb, w, x); wl = H.SMHSWork(par)
        Hd = H.similar_state(x); H.gn_hvp!(Hd, pb, w, wl, d)
        pb.hinge = nothing; H.objective!(pb, w, x); H0 = H.similar_state(x); H.gn_hvp!(H0, pb, w, wl, d); pb.hinge = hg
        ex = H.dotN(par, d, Hd) - H.dotN(par, d, H0)
        @test ex > 0
        @test abs(fd - ex) <= 1e-4 * ex
    end
end

@testset "M5g: barrier-safe preconditioner near the bound (Codex M5f reproducer, asserted)" begin
    par, pb, x, hard, sig = nearbound_problem(1e-6)
    w = H.SMHSWork(par); f = H.objective!(pb, w, x)[1]
    @test isfinite(f)
    v = H.similar_state(x); H.setscalar!(par, v[5], 1.0)
    hv = H.similar_state(x); H.gn_hvp!(hv, pb, w, H.SMHSWork(par), v)
    exact = H.getscalar(par, hv[5])
    H.objective!(pb, w, x); Dnew = H.gn_diagonal!(H.similar_state(x), pb, w, x)
    H.objective!(pb, w, x); Dold = H.gn_diagonal!(H.similar_state(x), pb, w, x; barrier_safe = false)
    @test exact > 0
    @test isapprox(H.getscalar(par, Dnew[5]), exact; rtol = 1e-3)           # barrier-dominated: local term exact
    @test H.getscalar(par, Dold[5]) < 0.1exact                               # the reproduced pre-M5g defect
    # Ã: the face-term part of the diagonal equals the exact face-term part of the GN diagonal, node by node
    hg = pb.hinge; pb.hinge = nothing; H.objective!(pb, w, x)
    D0 = H.gn_diagonal!(H.similar_state(x), pb, w, x); pb.hinge = hg; H.objective!(pb, w, x)
    wl = H.SMHSWork(par)
    for (fld, i, j, k) in ((2, 2, 3, 0), (3, 1, 4, 5), (4, 3, 2, 9), (2, 4, 1, 9))
        e = H.similar_state(x); e[fld][i, j, k] = 1.0
        H.gn_hvp!(hv, pb, w, wl, e); a = hv[fld][i, j, k]
        pb.hinge = nothing; H.gn_hvp!(hv, pb, w, wl, e); a0 = hv[fld][i, j, k]; pb.hinge = hg
        @test isapprox(Dnew[fld][i, j, k] - D0[fld][i, j, k], a - a0; rtol = 1e-8, atol = 1e-12 * abs(a))
    end
    # no probe is evaluated with the face term: pb.hinge restored, all entries finite and positive
    @test pb.hinge === hg
    @test all(isfinite(Dnew[f][i, j, k]) && Dnew[f][i, j, k] > 0 for f in 1:4 for k in axes(Dnew[f], 3) for j in 1:4 for i in 1:4)
end

@testset "M5g: fraction to the boundary from the exact linear face map" begin
    @test H.fraction_to_boundary(0.0, 2.0, 1.0) == 0.5
    @test H.fraction_to_boundary(0.5, -3.0, 1.0) == 0.5
    @test H.fraction_to_boundary(0.9, 0.05, 1.0) == 1.0
    @test H.fraction_to_boundary(0.3, 0.0, 1.0) == 1.0
    par, pb, x, hard, sig = nearbound_problem(1e-3)
    w = H.SMHSWork(par); ws = H.SMHSWork(par); H.objective!(pb, w, x)
    for seed in 1:3
        d = randdir(par, x; seed = seed); H.lincomb!(d, 1e-3, d, 0.0, d, par)
        a = H.face_fraction(pb, w, ws, d, hard)
        @test 0 <= a <= 1
        wt = H.SMHSWork(par)
        H.objective!(pb, wt, H.lincomb!(H.similar_state(x), 1.0, x, a, d, par))
        @test H.face_violation(pb, wt, hard)[1] <= 1e-10
        if a < 1
            H.objective!(pb, wt, H.lincomb!(H.similar_state(x), 1.0, x, min(1.0, 1.01a), d, par))
            @test H.face_violation(pb, wt, hard)[1] > 0
        end
    end
end

@testset "M5g: bounded joint GN near an active face — validation, feasibility, invariants" begin
    par, pb, x, hard, sig = nearbound_problem(1e-6)
    b = par.block
    # perturb the pressure so the force term is not at its minimum
    H.owned_map!((i, j, k) -> x[1][i, j, k] + 0.2 * sin(1.37i + 2.21j + 0.7k), x[1], par)
    w = H.SMHSWork(par); f0 = H.objective!(pb, w, x)[1]
    ulo = H.newfield(par); H.owned_map!((i, j, k) -> w.u[i, j, k] - 3.0, ulo, par)
    @test_throws H.InvalidInput H.joint_gn_solve!(pb, w, deepcopy(x), ulo; face_hard = hard, bound_tau = 1.0, max_candidates = 1)
    @test_throws H.InvalidInput H.joint_gn_solve!(pb, w, deepcopy(x), ulo; face_hard = hard, bound_tau = 0.0, max_candidates = 1)
    tight = (H.newfield(par; fill = 2.0), H.newfield(par; fill = 2.0), H.newfield0(par; fill = 2.0))
    @test_throws H.InvalidInput H.joint_gn_solve!(pb, w, deepcopy(x), ulo; face_hard = tight, bound_tau = 0.99, max_candidates = 1)
    xs = deepcopy(x); ws = H.SMHSWork(par); H.objective!(pb, ws, xs)
    hist = Float64[]
    st, why, n, res = H.joint_gn_solve!(pb, ws, xs, ulo; face_hard = hard, bound_tau = 0.99, max_candidates = 12, diagnostics = true,
                                        callback = (a, row) -> (push!(hist, row.f); nothing))
    @test n >= 1
    f1 = H.objective!(pb, ws, xs)[1]
    @test f1 < f0
    @test H.face_violation(pb, ws, hard)[1] < 0                                  # strictly inside after every accepted step
    @test all(c.bound_viol <= 1e-9 for c in res.candidates if c.accepted)
    @test all(c.pred > 0 for c in res.candidates)                               # model decrease of the FINAL (truncated) step
    @test haskey(res, :μ) && isfinite(res.μ) && isfinite(res.Δ)
    # the selected step's predicted decrease is never below that of the truncated GN step (Cauchy only replaces it when better)
    @test all(c.pred >= c.pred_gn * (1 - 1e-12) for c in res.candidates)
    # divergence-free and anchor-preserving
    dmax = maximum(abs(H.div_face(ws.B, pb.s, i, j, k)) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
    Bs = maximum(abs, ws.B[3][i, j, k] for k in axes(ws.B[3], 3), j in 1:b.ny, i in 1:b.nx) / minimum(pb.s.dxc)
    @test dmax <= 1e-12 * Bs
    @test abs(H.anchor_mean(pb, ws.u) - pb.lnPref) <= 1e-12 * abs(pb.lnPref)
    # callback :stop ends the solve with the declared reason
    xc = deepcopy(x); wc = H.SMHSWork(par); H.objective!(pb, wc, xc)
    st, why, n, _ = H.joint_gn_solve!(pb, wc, xc, ulo; face_hard = hard, bound_tau = 0.99, max_candidates = 12, callback = (a, row) -> :stop)
    @test st == H.stalled && why == "callback stop" && n == 1
end

@testset "M5g: pressure and magnetic changes are measured separately" begin
    par, pb, x, hard, sig = nearbound_problem(0.5; kind = nothing)
    w = H.SMHSWork(par); H.objective!(pb, w, x); snap = H.measured_snapshot(pb, w)
    xp = deepcopy(x); H.owned_map!((i, j, k) -> x[1][i, j, k] + (i == 2 && j == 3 && k == 4 ? 0.1 : 0.0), xp[1], par)
    wp = H.SMHSWork(par); H.objective!(pb, wp, xp)
    @test isapprox(H.measured_change_P(pb, wp, snap), expm1(0.1); rtol = 1e-12)
    @test H.measured_change_B(pb, wp, snap) == 0
    xb = deepcopy(x); xb[3][2, 2, 3] += 0.5
    wb = H.SMHSWork(par); H.objective!(pb, wb, xb)
    @test H.measured_change_P(pb, wb, snap) == 0
    @test H.measured_change_B(pb, wb, snap) > 0
    @test H.measured_change(pb, wb, snap) == H.measured_change_B(pb, wb, snap)
end
