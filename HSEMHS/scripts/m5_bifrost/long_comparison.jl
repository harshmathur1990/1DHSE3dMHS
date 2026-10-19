# Longer matched solver comparison from one starting state (Codex M5d review, step 4–5).
# Arms, each with the same work budget (equivalent GN products, actual work reported):
#   alt         corrected alternation (L-BFGS Ã block, ln P pressure block)
#   magtr-alt   alternation with a magnetic trust-region GN block (joint_gn_solve!, magnetic_only) and periodic
#               pressure blocks (u_gauss_newton!), trust states carried across cycles
#   joint-x100  joint GN, pressure ×100 in the step geometry
#   joint-cap   joint GN, direct per-node log-pressure step limit |δu| ≤ 0.5 (du_cap), default geometry
#   joint-at    joint GN, anchor-tangent PCG, default geometry
#   joint-x100-at  joint GN, ×100 and anchor-tangent
# All joint arms start from the same explicit radius Δ0 (the default rule at this state).
# usage (from HSEMHS/): julia --project=. -t 4 scripts/m5_bifrost/long_comparison.jl START.h5 TAG [budget] [arms,comma,separated]
include(joinpath(@__DIR__, "bif_setup.jl"))
start = ARGS[1]; tag = ARGS[2]; budget = parse(Float64, get(ARGS, 3, "3000"))
arms = split(get(ARGS, 4, "alt,magtr-alt,joint-x100,joint-cap,joint-at,joint-x100-at"), ",")
ck = normpath(joinpath(@__DIR__, "..", "..", "out", "m5c", "ck.h5"))            # u_init (collapse bound) lives here
pb, x0, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
uinit = H.newfield(par)
h5open(start) do f
    for q in 1:5; copyto!(parent(x0[q]), read(f["x$q"])); end
end
h5open(ck) do f; copyto!(parent(uinit), read(f["u_init"])); end
H.exchange_halos!(par, collect(x0))
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
w = H.SMHSWork(par); H.objective!(pb, w, x0)
Dref = H.reference_metric(pb, w, x0)                                    # fixed metric of this starting state
xr = deepcopy(x0); wr = H.SMHSWork(par); H.objective!(pb, wr, xr)
_, _, _, rr = H.joint_gn_solve!(pb, wr, xr, ulo; max_candidates = 0); Δc = rr.Δ
u0 = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]; B0 = H.measured_snapshot(pb, w)
function bdev(w)
    m = 0.0
    for q in 1:3, k in axes(w.B[q], 3), j in 1:b.ny, i in 1:b.nx
        m = max(m, abs(w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]))
    end
    return m
end
function summary(name, x, work, wall, extra)
    wx = H.SMHSWork(par); f, t, _ = H.objective!(pb, wx, x); rms, _ = H.layer_residual_stats(pb, wx)
    a = H.full_audit(pb, wx, x, Dref, ulo)
    g = H.similar_state(x); H.fg!(g, pb, wx, x); D1 = H.newfield(par); H.gn_diag_v!(D1, pb, wx, x)
    uopt = H.pressure_optimality(pb, wx, g[1], D1, ulo, H.newfield(par), H.newfield(par))
    ac, gA, gb = H.atilde_audit(pb, wx, x, g)
    H.objective!(pb, wx, x)
    dP = maximum(abs(expm1(wx.u[i, j, k] - u0[i, j, k])) for i in 1:b.nx, j in 1:b.ny, k in 1:NZ)
    dB = H.measured_change(pb, wx, (copy(parent(wx.u)), B0[2]))
    margin = minimum(wx.u[i, j, k] - ulo[i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx)
    @printf("%s %-14s work %6.0f %5.0f s | Φ %.6f raw force %.4e max RMS %.3e data %.5f max|δB|/σ %.4f | fresh u-opt %.2e | gÃ %.2e gb·σ %.1e | fixed-metric m %.3e | total ΔP/P %.2e ΔB %.2e | margin %.1e | %s\n",
            tag, name, work, wall, f, t[1] / pb.lamF, maximum(rms), t[3], bdev(wx), uopt, gA, gb, a.m, dP, dB, margin, extra)
    flush(stdout)
    h5open(joinpath(dirname(start), "long_$(tag)_$(name).h5"), "w") do fo
        for q in 1:5; fo["x$q"] = parent(x[q]); end
    end
end
hist_str(h) = join((@sprintf("%.0f:%.4f", q[1], q[2]) for q in h[1:max(1, length(h) ÷ 6):end]), " ")
@printf("%s start: Φ %.6f; common Δ0 = %.4f; budget %.0f\n", tag, H.objective!(pb, w, x0)[1], Δc, budget)
for arm in arms
    x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
    H.reset_work!(); tm = time(); hist = Tuple{Float64,Float64}[]
    if arm == "alt"
        cb(o, row) = push!(hist, (row.work, row.f))
        st, why, n, hh = H.alternating_solve!(pb, wx, x; max_outer = 10_000, a_iter = 100, u_steps = 5, ulo = ulo, work_budget = budget, callback = cb)
        extra = @sprintf("cycles %d; Ã fails %d; u fails %d | %s", n, count(q -> q.a_fail, hh), count(q -> q.u_fail, hh), hist_str(hist))
    elseif arm == "magtr-alt"
        Δm = Δc; Δu = NaN; μu = 1e-8; cyc = 0
        while H.work_equivalent() < budget
            cyc += 1
            st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; magnetic_only = true, max_accepted = 5, Δ0 = Δm,
                                                work_budget = budget - H.work_equivalent())
            Δm = res.Δ
            H.work_equivalent() >= budget && break
            ru = H.u_gauss_newton!(pb, wx, x, ulo; steps = 5, cg_iter = 100, cg_rtol = 1e-2, Δ0 = Δu, μ0 = μu, opt_factor = 0.5)
            Δu, μu = ru.Δ, ru.μ
            push!(hist, (H.work_equivalent(), H.objective!(pb, wx, x)[1]))
        end
        extra = @sprintf("cycles %d | %s", cyc, hist_str(hist))
    else
        kw = arm == "joint-x100" ? (; block_scale = (100.0, 1.0, 1.0)) :
             arm == "joint-cap" ? (; du_cap = 0.5) :
             arm == "joint-at" ? (; anchor_tangent = true) :
             arm == "joint-x100-at" ? (; block_scale = (100.0, 1.0, 1.0), anchor_tangent = true) : error("unknown arm $arm")
        cbj(acc, row) = push!(hist, (row.work, row.f))
        st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; work_budget = budget, Δ0 = Δc, callback = cbj, kw...)
        extra = @sprintf("accepted %d / %d; Cauchy %d; %s | %s", n, res.counts.cand, res.counts.cauchy, why, hist_str(hist))
    end
    summary(arm, x, H.work_equivalent(), time() - tm, extra)
end
