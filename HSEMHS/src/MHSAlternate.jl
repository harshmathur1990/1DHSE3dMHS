# M5: block alternation for real data (user decision 17 Oct 2026; reports/M5a_realdata.md), revised after
# the Codex reviews (CODEX_REVIEW_M5.md, CODEX_REVIEW_M5_FOLLOWUP.md; CLAUDE_RESPONSE_CODEX_M5.md).
#
#   Ã block: L-BFGS on (Ã, b̄) with v frozen.
#   u block: trust-region Gauss–Newton on the pressure with (Ã, b̄) fixed. One candidate per trust-region
#            iteration, all built from ONE model at the accepted base state (gradient, diagonal, curvature,
#            bounds, scaling); trial points are evaluated in a separate workspace:
#              1. Steihaug–Toint PCG on the free set (ε-active set fixed) in the D-scaled norm;
#              2. map to the actual δu and project exactly onto bounds ∩ {Σ_anchor α δu = 0};
#              3. scale δu linearly toward 0 until it lies in the trust ball in the model coordinates
#                 (feasible: the box contains 0 and the anchor constraint is linear);
#              4. projected-Cauchy safeguard: the candidate must give ≥ 0.1 of the Cauchy model decrease,
#                 otherwise the Cauchy step is used;
#              5. accept on the actual/predicted ratio; the radius is updated from the FINAL step's norm.
#            Coordinates: ln P (`coord = :lnP`) or p = P/P_ref (`coord = :P`, a scalar reference scale).
# Outer convergence: P and B change ≤ xtol over one cycle, no block failed, and fresh optimality audits of
# both blocks at the same final state pass; if only the change test passes, the blocks are tightened.

_dot1(par, a::Field, c::Field) =
    MPI.Allreduce(sum(a[i, j, k] * c[i, j, k] for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx), +, par.comm)
_maxabs1(par, a::Field) = global_max(par, maximum(abs(a[i, j, k]) for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx))
_sum1(par, f) = MPI.Allreduce(sum(f(i, j, k) for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx), +, par.comm)

"""
    pressure_optimality(pb, w, g1, D1, ulo, δs, ts) -> ‖δu*‖∞

Scaled constrained optimality of the pressure block: δu* minimizes gᵀδ + ½δᵀDδ subject to
u + δ ≥ ulo and Σ_anchor α δ = 0, with g the v-gradient and D the v-diagonal at the state held by `w`.
Invariant under a common scaling of g and D, so λ_F drops out. `δs`, `ts` are scratch.
"""
function pressure_optimality(pb::SMHSProblem, w::SMHSWork, g1::Field, D1::Field, ulo::Field, δs::Field, ts::Field)
    par = pb.par
    owned_map!((i, j, k) -> -g1[i, j, k] / D1[i, j, k], ts, par)
    anchor_box_project!(δs, ts, (i, j, k) -> min(ulo[i, j, k] - w.u[i, j, k], 0.0), (i, j, k) -> Inf,
                        (i, j, k) -> pb.alpha[i, j] / D1[i, j, k], pb)
    return _maxabs1(par, δs)
end

"""
    u_gauss_newton!(pb, w, x, ulo; steps, cg_iter, cg_rtol, du_max, coord, opt_tol, opt_target, Δ0, μ0, record)
        -> NamedTuple(status, accepted, tries, opt, opt0, f, Δ, μ, cg_rel, min_margin, anchor_err, counts, history)

Trust-region Gauss–Newton on the pressure with (Ã, b̄) fixed. `w` holds the state at `x` on entry and on
exit (it is only overwritten on acceptance). `ulo` bounds u = ln P from below.
Stops at optimality ≤ `opt_tol` (`:converged`), at optimality ≤ `opt_target` or after `steps` accepted
updates (`:progress` if any were accepted), or when the radius collapses / tries run out (`:failed` if none
were accepted, `:progress` otherwise). Pass the returned `Δ`, `μ` back as `Δ0`, `μ0` to keep the
trust-region state across calls. `counts`: candidates, Cauchy fallbacks, trust scalings, cap hits.
"""
function u_gauss_newton!(pb::SMHSProblem, w::SMHSWork, x::MHSState, ulo::Field;
                         steps::Int = 5, cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, du_max::Float64 = 0.5,
                         coord::Symbol = :lnP, opt_tol::Float64 = 1e-6, opt_target::Float64 = 0.0, opt_factor::Float64 = 0.0,
                         Δ0::Float64 = NaN, μ0::Float64 = 1e-8, record::Bool = false, max_tries::Int = 3steps + 10)
    coord in (:lnP, :P) || invalid("u_gauss_newton!: coord must be :lnP or :P")
    par = pb.par; b = par.block
    wl = SMHSWork(par); wt = SMHSWork(par)                     # GN scratch; trial-point workspace
    g = similar_state(x); z = similar_state(x); Hz = similar_state(x); xt = similar_state(x)
    D1 = newfield(par); r = newfield(par); p = newfield(par); q = newfield(par); y = newfield(par); zr = newfield(par)
    sc = newfield(par); t = newfield(par); δ = newfield(par); δc = newfield(par); δs = newfield(par); ts = newfield(par)
    hd = newfield(par); act = newfield(par); ylo = newfield(par); yhi = newfield(par)
    for c in 2:5
        fill!(parent(z[c]), 0.0)
    end
    cap = 10du_max
    # ---- the base model (fixed until a step is accepted) ----
    f = 0.0
    function rebuild_base!()
        f = fg!(g, pb, w, x)[1]
        gn_diag_v!(D1, pb, w, x)
        fill_owned!((i, j, k) -> coord == :P ? exp(-(w.u[i, j, k] - pb.lnPref)) : 1.0, sc, par)
        fill_owned!((i, j, k) -> (l = min(max(ulo[i, j, k] - w.u[i, j, k], -cap), 0.0); coord == :P ? expm1(l) / sc[i, j, k] : l), ylo, par)
        fill_owned!((i, j, k) -> coord == :P ? expm1(cap) / sc[i, j, k] : cap, yhi, par)
        return f
    end
    rebuild_base!()
    Dy(i, j, k) = D1[i, j, k] * sc[i, j, k]^2
    gy(i, j, k) = sc[i, j, k] * g[1][i, j, k]
    Hv!(out, v) = begin                                        # GN product at the base state (w)
        copyto!(parent(z[1]), parent(v)); exchange_halos!(par, [z[1]]); gn_hvp!(Hz, pb, w, wl, z)
        owned_map!((i, j, k) -> Hz[1][i, j, k], out, par)
    end
    # model-coordinate step (δv-equivalent) of an actual δu, its D-norm, and the predicted decrease
    model_step!(hd, δ) = coord == :P ? owned_map!((i, j, k) -> expm1(δ[i, j, k]), hd, par) : copyto!(parent(hd), parent(δ))
    ynorm(δ) = sqrt(_sum1(par, (i, j, k) -> (coord == :P ? D1[i, j, k] * expm1(δ[i, j, k])^2 : D1[i, j, k] * δ[i, j, k]^2)))
    function predicted(δ)
        model_step!(hd, δ); Hv!(q, hd)
        return -(_dot1(par, g[1], hd) + 0.5 * _dot1(par, hd, q))
    end
    # actual δu from a y-direction: map, re-centre, project onto bounds ∩ anchor, then fit into the trust ball
    ntrust = 0; ncap = 0
    function candidate!(δ, ydir, τ, Δ)
        owned_map!((i, j, k) -> coord == :P ? log1p(max(τ * sc[i, j, k] * ydir[i, j, k], -0.999999)) : τ * ydir[i, j, k], t, par)
        m = anchor_mean(pb, t); owned_map!((i, j, k) -> t[i, j, k] - m, t, par)
        ae = anchor_box_project!(δ, t, (i, j, k) -> min(max(ulo[i, j, k] - w.u[i, j, k], -cap), 0.0), (i, j, k) -> cap,
                                 (i, j, k) -> 1.0, pb)
        nrm = ynorm(δ)
        if nrm > Δ * (1 + 1e-12)                               # linear scaling in u keeps bounds and anchor
            ntrust += 1
            if coord == :lnP
                s = Δ / nrm
            else
                lo_s = 0.0; hi_s = 1.0
                for _ in 1:60
                    ms = 0.5 * (lo_s + hi_s)
                    sqrt(_sum1(par, (i, j, k) -> D1[i, j, k] * expm1(ms * δ[i, j, k])^2)) <= Δ ? (lo_s = ms) : (hi_s = ms)
                end
                s = lo_s
            end
            owned_map!((i, j, k) -> s * δ[i, j, k], δ, par)
        end
        return ae
    end
    opt = pressure_optimality(pb, w, g[1], D1, ulo, δs, ts); opt0 = opt
    opt_target = max(opt_target, opt_factor * opt0)            # inexact rule: optimality reduced by opt_factor
    μ = μ0; nacc = 0; tries = 0; ncand = 0; ncauchy = 0; cg_rel = NaN; margin = NaN; aerr = 0.0
    nexit = Dict{Symbol,Int}(); cg_exit = :none
    Δ = isnan(Δ0) ? du_max * sqrt(_sum1(par, (i, j, k) -> D1[i, j, k])) : Δ0
    Δstart = Δ
    hist = NamedTuple[]
    status = :progress
    opt <= opt_tol && return (status = :converged, accepted = 0, tries = 0, opt = opt, opt0 = opt0, f = f, Δ = Δ, μ = μ,
                              cg_rel = NaN, min_margin = NaN, anchor_err = 0.0, counts = (cand = 0, cauchy = 0, trust = 0, cap = 0, cg_exit = Dict{Symbol,Int}()),
                              history = hist)
    while nacc < steps
        tries += 1
        tries > max_tries && break
        # ε-active set at the base state: within 1e-6 of the lower bound with an outward gradient
        fill_owned!((i, j, k) -> ((ylo[i, j, k] >= -1e-6 / sc[i, j, k] && gy(i, j, k) > 0) ? 1.0 : 0.0), act, par)
        free(i, j, k) = act[i, j, k] == 0.0
        # ---- 1. Steihaug–Toint PCG on the free set ----
        fill!(parent(y), 0.0)
        owned_map!((i, j, k) -> free(i, j, k) ? -gy(i, j, k) : 0.0, r, par)
        owned_map!((i, j, k) -> r[i, j, k] / ((1 + μ) * Dy(i, j, k)), zr, par)
        copyto!(parent(p), parent(zr))
        rz = _dot1(par, r, zr); rz0 = rz; cg_rel = 1.0; cg_exit = :budget
        rz0 <= 0 && (status = :converged; break)
        ynorm2() = _sum1(par, (i, j, k) -> Dy(i, j, k) * y[i, j, k]^2)
        for _ in 1:cg_iter
            owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * p[i, j, k] : 0.0, hd, par)
            Hv!(q, hd)
            owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * q[i, j, k] + μ * Dy(i, j, k) * p[i, j, k] : 0.0, q, par)
            pq = _dot1(par, p, q)
            dd = _sum1(par, (i, j, k) -> Dy(i, j, k) * y[i, j, k] * p[i, j, k]); pD = _sum1(par, (i, j, k) -> Dy(i, j, k) * p[i, j, k]^2)
            τt = pD > 0 ? max((-dd + sqrt(max(dd^2 - pD * (ynorm2() - Δ^2), 0.0))) / pD, 0.0) : Inf
            α = pq > 0 ? rz / pq : Inf
            if α >= τt                                         # trust boundary (or negative curvature): stop on it
                owned_map!((i, j, k) -> y[i, j, k] + τt * p[i, j, k], y, par)
                cg_exit = pq > 0 ? :boundary : :negcurv
                # residual of the RETURNED direction: r = −g − (H + μD) y on the free set
                owned_map!((i, j, k) -> free(i, j, k) ? sc[i, j, k] * y[i, j, k] : 0.0, hd, par); Hv!(q, hd)
                owned_map!((i, j, k) -> free(i, j, k) ? -gy(i, j, k) - (sc[i, j, k] * q[i, j, k] + μ * Dy(i, j, k) * y[i, j, k]) : 0.0, r, par)
                cg_rel = sqrt(max(_sum1(par, (i, j, k) -> r[i, j, k]^2 / ((1 + μ) * Dy(i, j, k))), 0.0) / rz0)
                break
            end
            owned_map!((i, j, k) -> y[i, j, k] + α * p[i, j, k], y, par)
            owned_map!((i, j, k) -> r[i, j, k] - α * q[i, j, k], r, par)
            owned_map!((i, j, k) -> r[i, j, k] / ((1 + μ) * Dy(i, j, k)), zr, par)
            rzn = _dot1(par, r, zr)
            cg_rel = sqrt(max(rzn, 0.0) / rz0)                 # preconditioned residual (coordinate-invariant)
            cg_rel <= cg_rtol && (cg_exit = :residual; break)
            owned_map!((i, j, k) -> free(i, j, k) ? zr[i, j, k] + (rzn / rz) * p[i, j, k] : 0.0, p, par)
            rz = rzn
        end
        # ---- 2–3. candidate: actual δu, exactly feasible and inside the trust ball ----
        ncand += 1
        nexit[cg_exit] = get(nexit, cg_exit, 0) + 1
        aerr = candidate!(δ, y, 1.0, Δ)
        pred = predicted(δ)
        # ---- 4. projected-Cauchy safeguard: steepest descent −D⁻¹g on the free set, best of a few τ ----
        owned_map!((i, j, k) -> free(i, j, k) ? -gy(i, j, k) / Dy(i, j, k) : 0.0, ts, par)
        tn = sqrt(_sum1(par, (i, j, k) -> Dy(i, j, k) * ts[i, j, k]^2))
        predc = -Inf; τc = tn > 0 ? Δ / tn : 0.0
        for _ in 1:20
            candidate!(δs, ts, τc, Δ); pc = predicted(δs)
            if pc > predc
                predc = pc; copyto!(parent(δc), parent(δs))
            elseif predc > 0
                break
            end
            τc /= 2
        end
        if predc > 0 && pred < 0.1 * predc                       # insufficient model decrease: use the Cauchy step
            ncauchy += 1; copyto!(parent(δ), parent(δc)); pred = predc
        end
        steplen = _maxabs1(par, δ); nfin = ynorm(δ)
        ncap += steplen >= cap * (1 - 1e-12) ? 1 : 0
        # ---- 5. actual objective in the trial workspace; accept on the ratio ----
        copystate!(xt, x, par)
        owned_map!((i, j, k) -> x[1][i, j, k] + δ[i, j, k], xt[1], par)
        ft, _, oob = objective!(pb, wt, xt)
        margin = -global_max(par, -minimum(wt.u[i, j, k] - ulo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
        ρ = (oob || pred <= 0) ? -Inf : (f - ft) / pred
        Δused = Δ
        if ρ > 1e-4 && ft < f && margin >= -1e-12
            copystate!(x, xt, par)
            rebuild_base!()                                   # w now holds the new base state
            nacc += 1
            ρ > 0.75 && nfin >= 0.99Δ && (Δ *= 2)
            ρ < 0.25 && (Δ = nfin / 4)
            μ = ρ > 0.75 ? max(μ / 3, 1e-12) : μ
            opt = pressure_optimality(pb, w, g[1], D1, ulo, δs, ts)
            record && push!(hist, (accepted = nacc, f = f, rho = ρ, pred = pred, predc = predc, cg_rel = cg_rel, cg_exit = cg_exit, opt = opt,
                                   margin = margin, anchor_err = aerr, steplen = steplen, ynorm = nfin, Delta_used = Δused, Delta = Δ,
                                   nactive = Int(_sum1(par, (i, j, k) -> act[i, j, k])), cauchy = ncauchy, trust = ntrust, cap = ncap))
            opt <= opt_tol && (status = :converged; break)
            opt <= opt_target && break
        else
            Δ = nfin / 4                                       # w still holds the base state
            Δ < 1e-14 * Δstart && break
        end
    end
    status != :converged && (status = nacc > 0 ? :progress : :failed)
    return (status = status, accepted = nacc, tries = tries, opt = opt, opt0 = opt0, f = f, Δ = Δ, μ = μ, cg_rel = cg_rel,
            min_margin = margin, anchor_err = aerr, counts = (cand = ncand, cauchy = ncauchy, trust = ntrust, cap = ncap, cg_exit = nexit), history = hist)
end

"""
    atilde_audit(pb, w, x, g) -> (combined, gA, gb_sigma)

Scaled magnetic-block gradient at the state of `x` (`w` holds it on exit): ‖∇_Ã Φ‖₂ for the dimensionless
Ã, and |∂Φ/∂(b̄/σ_B)| = σ_B |∂Φ/∂b̄| for the mean flux measured in units of the field uncertainty (Codex
second follow-up: b̄ in tesla must not be mixed unscaled with Ã). `combined` = √(gA² + gb_sigma²).
"""
function atilde_audit(pb::SMHSProblem, w::SMHSWork, x::MHSState, g::MHSState)
    fg!(g, pb, w, x)
    par = pb.par; b = par.block
    s = 0.0
    for f in 2:4, k in axes(g[f], 3), j in 1:b.ny, i in 1:b.nx
        s += g[f][i, j, k]^2
    end
    gA = sqrt(MPI.Allreduce(s, +, par.comm))
    σB = sqrt(_sum1(par, (i, j, k) -> pb.sig[3][i, j, k]^2) / pb.N)
    gb = σB * abs(getscalar(par, g[5]))
    return sqrt(gA^2 + gb^2), gA, gb
end

"""
    alternating_solve!(pb, w, x; max_outer, a_iter, u_steps, u_opt_factor, xtol, final_uopt, final_arel, final_aabs,
                       floor_check, Dref, res_rel, work_budget, callback, collapse_du, du_max, cg_iter, cg_rtol,
                       coord, opts, verbose) -> (status, reason, outer, history)

Block alternation with inexact blocks: the Ã block (L-BFGS, v frozen, `a_iter` iterations) and the pressure
block (`u_gauss_newton!`: up to `u_steps` accepted steps or optimality reduced by `u_opt_factor`; the
trust-region state persists across cycles). A block FAILS when L-BFGS stops on a line-search failure or an EOS
failure, or the pressure block accepts no step; budget-limited progress is not failure.

Whenever P and B change by ≤ `xtol` over a cycle, fresh audits are computed at the same final state, also when
a block failed (for diagnosis): pressure optimality, the scaled magnetic gradient (`atilde_audit`) and, when a
reference metric is available, the fixed-metric full-state audit (`full_audit`).
- `converged`: no block failed, pressure optimality ≤ `final_uopt`, and the scaled magnetic gradient
  ≤ `final_arel` × its start value + `final_aabs`.
- `stalled` with reason "stalled_at_floor" (user decision 18 Oct 2026, Codex's four conditions): the change and
  feasibility tests hold; the audits were computed and are reported; the caller's independent floor evidence
  `floor_check(pb, w, x)` holds (for synthetic gates: the validated reference floor and the gate's accuracy
  tests; production passes none, so it cannot obtain this label); and the fixed-metric diagnostic correction
  predicts a decrease ≤ `res_rel` × Φ. This label is not `converged` (`success_mhs` is unchanged).
- otherwise the blocks are tightened (×4 budgets) and the loop continues; `work_budget` (equivalent GN products,
  `work_equivalent`) or `max_outer` end it with `max_iter`.
`callback(outer, row)` runs after every cycle with the work counters saved and restored, so diagnostics it
computes are not charged to the solver.
"""
function alternating_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState; max_outer::Int = 100, a_iter::Int = 500,
                            u_steps::Int = 5, u_opt_factor::Float64 = 0.5, xtol::Float64 = 1e-3,
                            final_uopt::Float64 = 1e-6, final_arel::Float64 = 1e-6, final_aabs::Float64 = 0.0,
                            floor_check = nothing, Dref = nothing, res_rel::Float64 = 1e-10,
                            work_budget::Float64 = Inf, callback = nothing, collapse_du::Float64 = 3.0,
                            du_max::Float64 = 0.5, cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, coord::Symbol = :lnP,
                            opts::LBFGSOptions = LBFGSOptions(), verbose::Bool = false, ulo = nothing)
    par = pb.par; b = par.block
    w0 = work_equivalent()
    objective!(pb, w, x)
    if ulo === nothing
        ulo = newfield(par); owned_map!((i, j, k) -> w.u[i, j, k] - collapse_du, ulo, par)
    end
    gs = similar_state(x); ga0 = atilde_audit(pb, w, x, gs)[1]
    D1 = newfield(par); δs = newfield(par); ts = newfield(par)
    hist = NamedTuple[]
    Δu = NaN; μu = 1e-8; tight = 1
    for outer in 1:max_outer
        # cap the blocks by the remaining work budget (an L-BFGS iteration costs ≈ 1–2 equivalent products)
        rem = work_budget - (work_equivalent() - w0)
        rem <= 0 && return max_iter, "work budget", outer - 1, hist
        aI = isfinite(rem) ? clamp(floor(Int, rem / 2), 1, tight * a_iter) : tight * a_iter
        oA = LBFGSOptions(; (k => getfield(opts, k) for k in fieldnames(LBFGSOptions))..., frozen = [1], max_iter = aI, xtol_rel = 0.0)
        snap = measured_snapshot(pb, w)
        ra = lbfgs!(pb, w, x; opts = oA)                 # leaves w at x
        ru = u_gauss_newton!(pb, w, x, ulo; steps = tight * u_steps, cg_iter = cg_iter, cg_rtol = cg_rtol, du_max = du_max,
                             coord = coord, Δ0 = Δu, μ0 = μu, opt_factor = tight == 1 ? u_opt_factor : 0.0)
        Δu, μu = ru.Δ, ru.μ
        f, t, _ = objective!(pb, w, x)
        ch = measured_change(pb, w, snap)
        a_fail = ra.status == eos_failure || (ra.status == stalled && occursin("line search", ra.reason))
        u_fail = ru.status == :failed
        audit = (uopt = NaN, arel = NaN, gA = NaN, gb_sigma = NaN, m = NaN, margin = NaN)
        status_now = :continue; floor_ok = false
        if ch <= xtol
            # fresh audits at the same final state, computed even when a block failed
            fg!(gs, pb, w, x)
            gn_diag_v!(D1, pb, w, x)
            uopt = pressure_optimality(pb, w, gs[1], D1, ulo, δs, ts)
            ac, gA, gbs = atilde_audit(pb, w, x, gs)
            m = Dref === nothing ? NaN : full_audit(pb, w, x, Dref, ulo).m
            margin = -global_max(par, -minimum(w.u[i, j, k] - ulo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
            objective!(pb, w, x)
            audit = (uopt = uopt, arel = ac / max(ga0, 1e-300), gA = gA, gb_sigma = gbs, m = m, margin = margin)
            if !a_fail && !u_fail && uopt <= final_uopt && ac <= final_arel * ga0 + final_aabs
                status_now = :converged
            else
                floor_ok = margin >= -1e-12 && floor_check !== nothing && floor_check(pb, w, x) && !isnan(m) && m <= res_rel * max(f, 1e-300)
                status_now = floor_ok ? :floor : :tighten
            end
        end
        row = (outer = outer, f = f, terms = t, change = ch, a_status = ra.status, a_reason = ra.reason, a_iter = ra.iterations,
               u_status = ru.status, usteps = ru.accepted, uopt = ru.opt, ucounts = ru.counts, a_fail = a_fail, u_fail = u_fail,
               audit = audit, floor_ok = floor_ok, tight = tight, work = work_equivalent() - w0)
        push!(hist, row)
        if verbose && rank0(par)
            @printf("  cycle %3d: Φ %.4e  Φ_F %.3e  Φ_Q %.3e  χ² %.4f  change %.2e  Ã %s/%d%s  u %s/%d opt %.1e→%.1e%s  work %.0f\n", outer, f, t[1], t[2], t[3], ch,
                    string(ra.status), ra.iterations, a_fail ? " FAIL" : "", string(ru.status), ru.accepted, ru.opt0, ru.opt,
                    isnan(audit.uopt) ? "" : @sprintf("  audit uopt %.1e arel %.1e (gA %.1e, gb·σ %.1e) m %.1e", audit.uopt, audit.arel, audit.gA, audit.gb_sigma, audit.m),
                    row.work)
            flush(stdout)
        end
        if callback !== nothing
            saved = copy(WORK_COUNTS); callback(outer, row); copyto!(WORK_COUNTS, saved)
        end
        status_now == :converged && return converged, "P and B changed ≤ xtol; no block failed; final audits passed", outer, hist
        status_now == :floor && return stalled, "stalled_at_floor: change ≤ xtol, feasible, floor evidence holds, diagnostic decrease ≤ res_rel·Φ", outer, hist
        status_now == :tighten && (tight = min(4tight, 16))
        work_equivalent() - w0 >= work_budget && return max_iter, "work budget", outer, hist
    end
    return max_iter, "max_outer", max_outer, hist
end
