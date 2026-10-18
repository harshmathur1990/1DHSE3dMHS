# M5: joint damped trust-region Gauss–Newton on the full state (v, Ã, b̄) — the challenger of the solver
# comparison proposed in CODEX_REVIEW_M5_FOLLOWUP.md (Q3) and CODEX_REVIEW_M5_SECOND_FOLLOWUP.md (Q2).
#
# One model per trust-region iteration, at the accepted base state (trial points in a separate workspace):
#   1. Steihaug–Toint PCG on (H_GN + μD) y = −g, D the full GN diagonal (exact v-diagonal; per-layer Hutchinson
#      for Ã; exact b̄ curvature), ε-active pressure nodes fixed, D-norm trust ball, preconditioned residual
#      stop, exit reason recorded;
#   2. candidate: the pressure part re-centred and projected exactly onto bounds ∩ {Σ_anchor α δu = 0}; the Ã
#      and b̄ parts unconstrained; the WHOLE step then scaled toward 0 until it fits the D-norm ball (feasible:
#      the box contains 0, the anchor constraint is linear);
#   3. projected-Cauchy safeguard on the full state (≥ 0.1 of the Cauchy model decrease, else the Cauchy step);
#   4. accept on the actual/predicted ratio; radius updated from the final step.
# The gauge null directions of Ã (curl-free changes of A) carry no curvature; the Levenberg damping μD keeps
# the system positive definite, and gauge components of the step change neither B nor Φ. They are not
# removed by a quotient solve; all comparisons are made through physical quantities (P, B) and Φ.

"""
    joint_gn_solve!(pb, w, x, ulo; work_budget, cg_iter, cg_rtol, du_cap, μ0, callback, verbose)
        -> (status, reason, accepted, history)

Joint trust-region GN until `work_budget` equivalent GN products (`work_equivalent`) are spent, measured from
the call. `callback(acc, row)` runs after every accepted step with the work counters saved and restored.
`history` rows: accepted count, Φ, terms, ρ, step norms, trust radius, CG exit, Cauchy use, margin, work.
"""
function joint_gn_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState, ulo::Field; work_budget::Float64 = 3000.0,
                         cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, du_cap::Float64 = 5.0, μ0::Float64 = 1e-6,
                         callback = nothing, verbose::Bool = false)
    par = pb.par; b = par.block
    w0 = work_equivalent()
    wl = SMHSWork(par); wt = SMHSWork(par)
    g = similar_state(x); D = similar_state(x); Dinv = similar_state(x)
    r = similar_state(x); zr = similar_state(x); p = similar_state(x); q = similar_state(x); y = similar_state(x)
    δ = similar_state(x); δc = similar_state(x); cdir = similar_state(x); xt = similar_state(x); act = newfield(par)
    t1 = newfield(par); s1 = newfield(par)
    f = 0.0
    function rebuild!()
        f = fg!(g, pb, w, x)[1]
        gn_diagonal!(D, pb, w, x)                        # leaves w at x
        for fld in eachindex(D)
            Df = D[fld]; Di = Dinv[fld]
            owned_map!((i, j, k) -> 1 / Df[i, j, k], Di, par)
        end
        fill_owned!((i, j, k) -> ((w.u[i, j, k] - ulo[i, j, k] <= 1e-6 && g[1][i, j, k] > 0) ? 1.0 : 0.0), act, par)
        objective!(pb, w, x)
        return f
    end
    rebuild!()
    free!(v) = (vf = v[1]; owned_map!((i, j, k) -> act[i, j, k] == 1.0 ? 0.0 : vf[i, j, k], vf, par); v)
    Dnorm2(v) = (divdiag!(zr, v, Dinv, par); dotN(par, v, zr))       # vᵀ D v   (zr is scratch here)
    Hv!(out, v) = gn_hvp!(out, pb, w, wl, v)
    function predicted(dlt)
        Hv!(q, dlt)
        return -(dotN(par, g, dlt) + 0.5 * dotN(par, dlt, q))
    end
    # actual step from a direction: project the pressure part, keep Ã/b̄, fit the whole step into the ball
    ntrust = 0
    function candidate!(dlt, dir, τ, Δ)
        lincomb!(dlt, τ, dir, 0.0, dir, par)
        owned_map!((i, j, k) -> dlt[1][i, j, k], t1, par)
        m = anchor_mean(pb, t1); owned_map!((i, j, k) -> t1[i, j, k] - m, t1, par)
        ae = anchor_box_project!(s1, t1, (i, j, k) -> min(max(ulo[i, j, k] - w.u[i, j, k], -du_cap), 0.0), (i, j, k) -> du_cap,
                                 (i, j, k) -> 1.0, pb)
        copyto!(parent(dlt[1]), parent(s1))
        nrm = sqrt(max(Dnorm2(dlt), 0.0))
        if nrm > Δ * (1 + 1e-12)
            ntrust += 1; lincomb!(dlt, Δ / nrm, dlt, 0.0, dlt, par)
        end
        return ae
    end
    Δ = 0.5 * sqrt(_sum1(par, (i, j, k) -> D[1][i, j, k]))   # as in the pressure block: all nodes moving 0.5 in ln P
    Δstart = Δ; μ = μ0
    nacc = 0; ncand = 0; ncauchy = 0; nexit = Dict{Symbol,Int}(); hist = NamedTuple[]
    status = max_iter; reason = "work budget"
    while work_equivalent() - w0 < work_budget
        # ---- 1. Steihaug–Toint PCG on the full state ----
        lincomb!(r, -1.0, g, 0.0, g, par); free!(r)
        divdiag!(zr, r, D, par); lincomb!(zr, 1 / (1 + μ), zr, 0.0, zr, par); free!(zr)
        lincomb!(p, 1.0, zr, 0.0, zr, par); lincomb!(y, 0.0, y, 0.0, y, par)
        rz = dotN(par, r, zr); rz0 = rz; cg_rel = 1.0; cg_exit = :budget
        rz0 <= 0 && (status = converged; reason = "zero preconditioned gradient"; break)
        for _ in 1:cg_iter
            Hv!(q, p); free!(q)
            divdiag!(zr, p, Dinv, par)                    # D p
            lincomb!(q, 1.0, q, μ, zr, par)               # (H + μD) p
            pq = dotN(par, p, q)
            yDy = Dnorm2(y); pDp = Dnorm2(p)
            divdiag!(zr, p, Dinv, par); yDp = dotN(par, y, zr)
            τt = pDp > 0 ? max((-yDp + sqrt(max(yDp^2 - pDp * (yDy - Δ^2), 0.0))) / pDp, 0.0) : Inf
            α = pq > 0 ? rz / pq : Inf
            if α >= τt
                lincomb!(y, 1.0, y, τt, p, par); cg_exit = pq > 0 ? :boundary : :negcurv
                break
            end
            lincomb!(y, 1.0, y, α, p, par)
            lincomb!(r, 1.0, r, -α, q, par)
            divdiag!(zr, r, D, par); lincomb!(zr, 1 / (1 + μ), zr, 0.0, zr, par); free!(zr)
            rzn = dotN(par, r, zr)
            cg_rel = sqrt(max(rzn, 0.0) / rz0)
            cg_rel <= cg_rtol && (cg_exit = :residual; break)
            lincomb!(p, 1.0, zr, rzn / rz, p, par); free!(p)
            rz = rzn
        end
        nexit[cg_exit] = get(nexit, cg_exit, 0) + 1
        # ---- 2. candidate ----
        ncand += 1
        aerr = candidate!(δ, y, 1.0, Δ)
        pred = predicted(δ)
        # ---- 3. projected-Cauchy safeguard ----
        divdiag!(cdir, g, D, par); lincomb!(cdir, -1.0, cdir, 0.0, cdir, par); free!(cdir)
        cn = sqrt(max(Dnorm2(cdir), 0.0))
        predc = -Inf; τc = cn > 0 ? Δ / cn : 0.0
        for _ in 1:20
            candidate!(xt, cdir, τc, Δ); pc = predicted(xt)
            if pc > predc
                predc = pc; copystate!(δc, xt, par)
            elseif predc > 0
                break
            end
            τc /= 2
        end
        usedc = false
        if predc > 0 && pred < 0.1 * predc
            ncauchy += 1; copystate!(δ, δc, par); pred = predc; usedc = true
        end
        nfin = sqrt(max(Dnorm2(δ), 0.0))
        # ---- 4. trial in a separate workspace ----
        lincomb!(xt, 1.0, x, 1.0, δ, par)
        ft, tt, oob = objective!(pb, wt, xt)
        margin = -global_max(par, -minimum(wt.u[i, j, k] - ulo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
        ρ = (oob || pred <= 0) ? -Inf : (f - ft) / pred
        Δused = Δ
        if ρ > 1e-4 && ft < f && margin >= -1e-12
            copystate!(x, xt, par)
            rebuild!()
            nacc += 1
            ρ > 0.75 && nfin >= 0.99Δ && (Δ *= 2)
            ρ < 0.25 && (Δ = nfin / 4)
            μ = ρ > 0.75 ? max(μ / 3, 1e-12) : (ρ < 0.25 ? min(4μ, 1e2) : μ)
            row = (accepted = nacc, f = f, terms = tt, rho = ρ, pred = pred, predc = predc, cauchy = usedc, ynorm = nfin,
                   Delta_used = Δused, Delta = Δ, cg_exit = cg_exit, cg_rel = cg_rel, margin = margin, anchor_err = aerr,
                   work = work_equivalent() - w0)
            push!(hist, row)
            if verbose && rank0(par) && nacc % 10 == 0
                @printf("  jgn %4d: Φ %.6e  Φ_F %.3e  ρ %.2f  ‖δ‖_D/Δ %.2f  CG %s  Cauchy %d/%d  work %.0f\n", nacc, f, tt[1], ρ, nfin / Δused,
                        string(cg_exit), ncauchy, ncand, row.work)
                flush(stdout)
            end
            if callback !== nothing
                saved = copy(WORK_COUNTS); callback(nacc, row); copyto!(WORK_COUNTS, saved)
            end
        else
            Δ = nfin / 4; μ = min(4μ, 1e2)
            if Δ < 1e-14 * Δstart
                status = stalled; reason = "trust radius collapsed"; break
            end
        end
    end
    return status, reason, nacc, (history = hist, counts = (cand = ncand, cauchy = ncauchy, trust = ntrust, cg_exit = nexit))
end
