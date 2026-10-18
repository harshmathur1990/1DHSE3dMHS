# Frozen-model step-geometry experiment and diagonal comparison (CODEX_REVIEW_M5c_2026-10-18.md §2–3), at the
# common checkpoint. Part 1: the default Ã layer diagonal (full Hessian, differences of ∇Φ) against the same
# probes with GN products. Part 2: ONE trust-region candidate from the same radius for several step geometries,
# with predicted/actual reduction, block D-norm shares and the model split. Part 3: short matched continuations.
# v2 (Codex M5d review step 4): every joint arm uses the SAME explicit initial radius Δ0; adds anchor-tangent arms.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/geometry_experiment_v2.jl out/m5c/ck.h5 [budget]
include(joinpath(@__DIR__, "bif_setup.jl"))
ck = ARGS[1]; budget = parse(Float64, get(ARGS, 2, "1000"))
function load(; kw...)
    pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3, kw...)
    uinit = H.newfield(par)
    h5open(ck) do f
        for q in 1:5; copyto!(parent(x[q]), read(f["x$q"])); end
        copyto!(parent(uinit), read(f["u_init"]))
    end
    H.exchange_halos!(par, collect(x))
    ma = H.anchor_mean(pb, uinit)
    ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
    return pb, x, par, b, NZ, ulo
end
pb, x0, par, b, NZ, ulo = load()
w = H.SMHSWork(par); H.objective!(pb, w, x0)
xr = deepcopy(x0); wr = H.SMHSWork(par); H.objective!(pb, wr, xr)
_, _, _, rr = H.joint_gn_solve!(pb, wr, xr, ulo; max_candidates = 0)
Δc = rr.Δ
configs = [("joint (default)", (;)), ("magnetic only", (; magnetic_only = true)),
           ("pressure stiffer ×10", (; block_scale = (10.0, 1.0, 1.0))), ("pressure stiffer ×100", (; block_scale = (100.0, 1.0, 1.0))),
           ("joint + anchor-tangent", (; anchor_tangent = true)), ("×100 + anchor-tangent", (; block_scale = (100.0, 1.0, 1.0), anchor_tangent = true))]
println("v2 Part 2: one candidate, common radius Δ0 = $(Δc) for every configuration")
for (name, kw) in configs
    x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
    H.reset_work!()
    st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; max_candidates = 1, diagnostics = true, Δ0 = Δc, kw...)
    c = res.candidates[1]; d = c.diag
    @printf("  %-24s pred %.4e actual %.4e ρ %.3f | shares y (u %.2f A %.2f) → final (u %.2f A %.2f) | max ΔP/P %.2e ΔB %.2e | model p/m/x %.3e/%.3e/%.3e | CG %s | work %.0f\n",
            name, c.pred, c.actual, c.rho, d.share_y.u, d.share_y.A, d.share_final.u, d.share_final.A, d.dPmax, d.dBmax, d.pred_u, d.pred_A, d.pred_cross, string(c.cg_exit), H.work_equivalent())
    flush(stdout)
end
println("v2 Part 3: continuations, budget $(budget), common Δ0 = $(Δc) for every joint arm; actual work reported")
for (name, kw) in configs
    x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
    H.reset_work!(); tm = time()
    st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; work_budget = budget, Δ0 = Δc, kw...)
    f, t, _ = H.objective!(pb, wx, x); rms, _ = H.layer_residual_stats(pb, wx)
    hist = join((@sprintf("%.0f:%.4f", q.work, q.f) for q in res.history[1:max(1, length(res.history) ÷ 4):end]), " ")
    @printf("  %-24s Φ %.6f raw force %.4e max RMS %.3e | accepted %d / %d | work %.0f, %.0f s | history work:Φ %s\n",
            name, f, t[1] / pb.lamF, maximum(rms), n, res.counts.cand, H.work_equivalent(), time() - tm, hist)
    flush(stdout)
end
