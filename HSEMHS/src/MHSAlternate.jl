# M5: block alternation for real data (user decision 17 Oct 2026; reports/M5a_realdata.md),
# revised after the Codex review (CODEX_REVIEW_M5.md, CLAUDE_RESPONSE_CODEX_M5.md).
#
#   Ã block: L-BFGS on (Ã, b̄) with v frozen.
#   u block: trust-region Gauss–Newton on the pressure with (Ã, b̄) fixed (TRON-like). Each step solves
#            the bound-constrained trust-region model by Steihaug–Toint PCG on the free set (nodes at a
#            bound with an outward gradient fixed; CG truncated where it reaches a bound or the D-norm
#            trust boundary), in ln P (`coord = :lnP`) or in p = P/P_ref (`coord = :P`); maps the step to the
#            actual δu, projects it onto the bounds ∩ {Σ_anchor α δu = 0} exactly (`anchor_box_project!`),
#            and accepts on the ratio of actual to predicted reduction. Clipping a full CG step per node
#            (the first version) is not a descent step and collapsed the trust radius on Bifrost.
# The outer loop is converged only if P and B change by ≤ xtol over one cycle AND both blocks succeeded.

_dot1(par, a::Field, c::Field) =
    MPI.Allreduce(sum(a[i, j, k] * c[i, j, k] for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx), +, par.comm)
_maxabs1(par, a::Field) = global_max(par, maximum(abs(a[i, j, k]) for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx))

"""
    pressure_optimality(pb, w, g1, D1, ulo, δs) -> ‖δu*‖∞

Scaled constrained optimality of the pressure block: δu* minimizes gᵀδ + ½δᵀDδ subject to
u + δ ≥ ulo and Σ_anchor α δ = 0, with g the v-gradient and D the v-diagonal. It is invariant under
a common scaling of g and D, so λ_F drops out (Codex review M5, experiment step 3). `δs` is scratch.
"""
function pressure_optimality(pb::SMHSProblem, w::SMHSWork, g1::Field, D1::Field, ulo::Field, δs::Field, ts::Field)
    par = pb.par
    owned_map!((i, j, k) -> -g1[i, j, k] / D1[i, j, k], ts, par)
    anchor_box_project!(δs, ts, (i, j, k) -> min(ulo[i, j, k] - w.u[i, j, k], 0.0), (i, j, k) -> Inf,
                        (i, j, k) -> pb.alpha[i, j] / D1[i, j, k], pb)
    return _maxabs1(par, δs)
end

"""
    u_gauss_newton!(pb, w, x, ulo; steps, cg_iter, cg_rtol, du_max, coord, opt_tol, record)
        -> NamedTuple(status, accepted, tries, opt, f, cg_rel, min_margin, anchor_err, history)

Trust-region Gauss–Newton on the pressure with (Ã, b̄) fixed. `w` must hold the state at `x` on entry
and holds it on exit. `ulo` is the lower bound on u = ln P (owned centres). `status` is `:converged`
(optimality ≤ opt_tol), `:budget` (`steps` accepted updates done) or `:failed` (trust radius
collapsed or tries exhausted). With `record`, `history` has one row per accepted update.
"""
function u_gauss_newton!(pb::SMHSProblem, w::SMHSWork, x::MHSState, ulo::Field;
                         steps::Int = 5, cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, du_max::Float64 = 0.5,
                         coord::Symbol = :lnP, opt_tol::Float64 = 1e-6, record::Bool = false, max_tries::Int = 3steps + 10)
    coord in (:lnP, :P) || invalid("u_gauss_newton!: coord must be :lnP or :P")
    par = pb.par; b = par.block
    wl = SMHSWork(par); g = similar_state(x); z = similar_state(x); Hz = similar_state(x); xt = similar_state(x)
    D1 = newfield(par); r = newfield(par); p = newfield(par); q = newfield(par); d = newfield(par); zr = newfield(par)
    sc = newfield(par); t = newfield(par); δ = newfield(par); δs = newfield(par); ts = newfield(par); hd = newfield(par)
    for c in 2:5
        fill!(parent(z[c]), 0.0)
    end
    f = fg!(g, pb, w, x)[1]
    gn_diag_v!(D1, pb, w, x)
    opt = pressure_optimality(pb, w, g[1], D1, ulo, δs, ts)
    μ = 1e-8; nacc = 0; tries = 0; status = :budget; cg_rel = NaN
    hist = NamedTuple[]
    opt <= opt_tol && return (status = :converged, accepted = 0, tries = 0, opt = opt, f = f, cg_rel = NaN,
                              min_margin = NaN, anchor_err = 0.0, history = hist)
    # H_v δ for a v-coordinate δ (GN, no damping)
    Hv!(out, v) = begin
        copyto!(parent(z[1]), parent(v)); exchange_halos!(par, [z[1]]); gn_hvp!(Hz, pb, w, wl, z)
        owned_map!((i, j, k) -> Hz[1][i, j, k], out, par)
    end
    margin = NaN; aerr = 0.0
    act = newfield(par); ylo = newfield(par); yhi = newfield(par)
    Δ = NaN                                              # trust radius in the D-scaled 2-norm (set below)
    while nacc < steps
        tries += 1
        tries > max_tries && (status = :failed; break)
        # coordinate scaling: δv = sc ⊙ y (sc = 1 for ln P; sc = 1/p for p = P/P_ref, so δu = log1p(sc y))
        fill_owned!((i, j, k) -> coord == :P ? exp(-(w.u[i, j, k] - pb.lnPref)) : 1.0, sc, par)
        # per-node bounds on y from lo ≤ δu ≤ hi: the collapse bound and a cap of du_max·10 in ln P
        cap = 10du_max
        fill_owned!((i, j, k) -> (l = min(max(ulo[i, j, k] - w.u[i, j, k], -cap), 0.0); coord == :P ? expm1(l) / sc[i, j, k] : l), ylo, par)
        fill_owned!((i, j, k) -> coord == :P ? expm1(cap) / sc[i, j, k] : cap, yhi, par)
        Dy(i, j, k) = D1[i, j, k] * sc[i, j, k]^2
        gy(i, j, k) = sc[i, j, k] * g[1][i, j, k]
        isnan(Δ) && (Δ = du_max * sqrt(MPI.Allreduce(sum(D1[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx), +, par.comm)))
        # ε-active set: within εb of the lower bound with the gradient pushing outward
        εb = 1e-6
        fill_owned!((i, j, k) -> ((ylo[i, j, k] >= -εb * sc[i, j, k]^-1 * (coord == :P ? 1.0 : 1.0) && gy(i, j, k) > 0) ? 1.0 : 0.0), act, par)
        fill_owned!((i, j, k) -> act[i, j, k] == 1.0 ? 0.0 : ylo[i, j, k], ylo, par)
        free(i, j, k) = act[i, j, k] == 0.0
        # Steihaug–Toint PCG on the free set: min gᵀy + ½ yᵀ(H + μD)y, ‖y‖_D ≤ Δ, ylo ≤ y ≤ yhi
        fill!(parent(d), 0.0)
        owned_map!((i, j, k) -> free(i, j, k) ? -gy(i, j, k) : 0.0, r, par)
        owned_map!((i, j, k) -> r[i, j, k] / ((1 + μ) * Dy(i, j, k)), zr, par)
        copyto!(parent(p), parent(zr))
        rz = _dot1(par, r, zr); r0 = sqrt(_dot1(par, r, r)); cg_rel = 0.0
        r0 == 0 && (status = :converged; break)
        dnorm2() = MPI.Allreduce(sum(Dy(i, j, k) * d[i, j, k]^2 for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx), +, par.comm)
        function to_boundary(pp)                         # (τ_box, τ_trust): largest steps keeping d + τ pp in the box / D-ball
            τb = Inf
            for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
                pv = pp[i, j, k]
                pv > 0 && (τb = min(τb, (yhi[i, j, k] - d[i, j, k]) / pv))
                pv < 0 && (τb = min(τb, (ylo[i, j, k] - d[i, j, k]) / pv))
            end
            τb = max(-global_max(par, -τb), 0.0)
            dd = MPI.Allreduce(sum(Dy(i, j, k) * d[i, j, k] * pp[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx), +, par.comm)
            pD = MPI.Allreduce(sum(Dy(i, j, k) * pp[i, j, k]^2 for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx), +, par.comm)
            c0 = dnorm2() - Δ^2
            τt = pD > 0 ? max((-dd + sqrt(max(dd^2 - pD * c0, 0.0))) / pD, 0.0) : Inf
            return τb, τt
        end
        # residual of the model on the free set at the current d: r = −g − (H + μD) d
        function reset_residual!()
            owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * d[i, j, k] : 0.0, hd, par)
            Hv!(q, hd)
            owned_map!((i, j, k) -> free(i, j, k) ? -gy(i, j, k) - (sc[i, j, k] * q[i, j, k] + μ * Dy(i, j, k) * d[i, j, k]) : 0.0, r, par)
            owned_map!((i, j, k) -> r[i, j, k] / ((1 + μ) * Dy(i, j, k)), zr, par)
            copyto!(parent(p), parent(zr))
            return _dot1(par, r, zr)
        end
        nrestart = 0
        for _ in 1:cg_iter
            owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * p[i, j, k] : 0.0, hd, par)
            Hv!(q, hd)
            owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * q[i, j, k] + μ * Dy(i, j, k) * p[i, j, k] : 0.0, q, par)
            pq = _dot1(par, p, q)
            τb, τt = to_boundary(p)
            α = pq > 0 ? rz / pq : Inf
            if α >= τt                                   # trust boundary: stop on it (Steihaug)
                owned_map!((i, j, k) -> d[i, j, k] + τt * p[i, j, k], d, par); break
            end
            # bounds are handled afterwards by the projected arc search, not by truncating CG
            owned_map!((i, j, k) -> d[i, j, k] + α * p[i, j, k], d, par)
            owned_map!((i, j, k) -> r[i, j, k] - α * q[i, j, k], r, par)
            cg_rel = sqrt(_dot1(par, r, r)) / r0
            cg_rel <= cg_rtol && break
            owned_map!((i, j, k) -> r[i, j, k] / ((1 + μ) * Dy(i, j, k)), zr, par)
            rzn = _dot1(par, r, zr)
            owned_map!((i, j, k) -> free(i, j, k) ? zr[i, j, k] + (rzn / rz) * p[i, j, k] : 0.0, p, par)
            rz = rzn
        end
        # projected arc search: δ(t) = Proj_{bounds ∩ anchor}(map(t y)), t = 1, ½, …; the first t with a good
        # actual/predicted ratio is taken (as t → 0 only nodes strictly inside move, so it is a descent arc)
        ρ = -Inf; ft = Inf; oob = true; pred = 0.0; tarc = 1.0
        for _ in 1:12
            owned_map!((i, j, k) -> coord == :P ? log1p(max(tarc * sc[i, j, k] * d[i, j, k], -0.999999)) : tarc * d[i, j, k], t, par)
            m = anchor_mean(pb, t); owned_map!((i, j, k) -> t[i, j, k] - m, t, par)
            aerr = anchor_box_project!(δ, t, (i, j, k) -> min(max(ulo[i, j, k] - w.u[i, j, k], -cap), 0.0), (i, j, k) -> cap,
                                       (i, j, k) -> 1.0, pb)
            if coord == :P
                owned_map!((i, j, k) -> expm1(δ[i, j, k]), hd, par)      # δp/p, the δv-equivalent in p
            else
                copyto!(parent(hd), parent(δ))
            end
            Hv!(q, hd)
            pred = -(_dot1(par, g[1], hd) + 0.5 * _dot1(par, hd, q))
            copystate!(xt, x, par)
            owned_map!((i, j, k) -> x[1][i, j, k] + δ[i, j, k], xt[1], par)
            ft, _, oob = objective!(pb, w, xt)
            margin = -global_max(par, -minimum(w.u[i, j, k] - ulo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
            ρ = (oob || pred <= 0) ? -Inf : (f - ft) / pred
            ρ > 1e-4 && ft < f && margin >= -1e-12 && break
            tarc /= 2
        end
        steplen = _maxabs1(par, δ)
        ynorm = tarc * sqrt(dnorm2())
        if ρ > 1e-4 && ft < f && margin >= -1e-12
            copystate!(x, xt, par)
            f = fg!(g, pb, w, x)[1]
            gn_diag_v!(D1, pb, w, x)
            nacc += 1
            ρ > 0.75 && ynorm >= 0.99Δ && (Δ *= 2)
            ρ < 0.25 && (Δ = ynorm / 4)
            μ = ρ > 0.75 ? max(μ / 3, 1e-12) : μ
            opt = pressure_optimality(pb, w, g[1], D1, ulo, δs, ts)
            record && push!(hist, (accepted = nacc, f = f, rho = ρ, cg_rel = cg_rel, opt = opt, margin = margin,
                                anchor_err = aerr, steplen = steplen, Delta = Δ))
            opt <= opt_tol && (status = :converged; break)
        else
            objective!(pb, w, x)                         # restore w at x
            Δ = min(Δ, ynorm) / 4
            Δ < 1e-14 * max(ynorm, 1.0) && (status = :failed; break)
        end
    end
    return (status = status, accepted = nacc, tries = tries, opt = opt, f = f, cg_rel = cg_rel,
            min_margin = margin, anchor_err = aerr, history = hist)
end

"""
    alternating_solve!(pb, w, x; max_outer, a_iter, u_steps, xtol, collapse_du, du_max, cg_iter, cg_rtol,
                       coord, opts, verbose) -> (status, reason, outer, history)

Block alternation (Ã block by L-BFGS with v frozen, u block by `u_gauss_newton!`). Converged only when
P and B change by ≤ `xtol` over one cycle, the Ã block did work (or converged) and the u block met its
constrained optimality tolerance; a budget-limited or failed block cannot satisfy the test (Codex review
M5, finding 2).
`history` rows: (outer, Φ, terms, change, Ã status and iterations, u status, accepted steps, optimality).
"""
function alternating_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState; max_outer::Int = 100, a_iter::Int = 500,
                            u_steps::Int = 5, xtol::Float64 = 1e-3, collapse_du::Float64 = 3.0, du_max::Float64 = 0.5,
                            cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, coord::Symbol = :lnP,
                            opts::LBFGSOptions = LBFGSOptions(), verbose::Bool = false)
    par = pb.par
    objective!(pb, w, x)
    ulo = newfield(par); owned_map!((i, j, k) -> w.u[i, j, k] - collapse_du, ulo, par)
    oA = LBFGSOptions(; (k => getfield(opts, k) for k in fieldnames(LBFGSOptions))..., frozen = [1], max_iter = a_iter, xtol_rel = 0.0)
    hist = NamedTuple[]
    for outer in 1:max_outer
        snap = measured_snapshot(pb, w)
        ra = lbfgs!(pb, w, x; opts = oA)                 # leaves w at x
        ru = u_gauss_newton!(pb, w, x, ulo; steps = u_steps, cg_iter = cg_iter, cg_rtol = cg_rtol, du_max = du_max, coord = coord)
        f, t, _ = objective!(pb, w, x)
        ch = measured_change(pb, w, snap)
        a_ok = ra.status == converged || (ra.iterations > 0 && ra.status != eos_failure)   # did work, no failure
        u_ok = ru.status == :converged                                                      # constrained optimality met
        push!(hist, (outer = outer, f = f, terms = t, change = ch, a_status = ra.status, a_iter = ra.iterations,
                     u_status = ru.status, usteps = ru.accepted, uopt = ru.opt))
        if verbose && rank0(par)
            @printf("  cycle %3d: Φ %.4e  Φ_F %.3e  Φ_Q %.3e  χ² %.4f  change %.2e  Ã %s/%d  u %s/%d opt %.1e\n", outer, f, t[1], t[2], t[3], ch,
                    string(ra.status), ra.iterations, string(ru.status), ru.accepted, ru.opt)
            flush(stdout)
        end
        if ch <= xtol && a_ok && u_ok
            return converged, "P and B changed ≤ xtol over one cycle; Ã block did work, u block optimal", outer, hist
        end
    end
    return max_iter, "max_outer", max_outer, hist
end
