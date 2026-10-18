# Frozen-model step-geometry experiment and diagonal comparison (CODEX_REVIEW_M5c_2026-10-18.md §2–3), at the
# common checkpoint. Part 1: the default Ã layer diagonal (full Hessian, differences of ∇Φ) against the same
# probes with GN products. Part 2: ONE trust-region candidate from the same radius for several step geometries,
# with predicted/actual reduction, block D-norm shares and the model split. Part 3: short matched continuations.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/geometry_experiment.jl out/m5c/ck.h5 [budget]
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
# ---- Part 1: diagonal estimates ----
D1 = H.gn_diagonal!(H.similar_state(x0), pb, w, x0)
pb.pc_atilde = :layer_gn; H.objective!(pb, w, x0)
H.reset_work!(); D2 = H.gn_diagonal!(H.similar_state(x0), pb, w, x0); wgn = H.work_equivalent()
pb.pc_atilde = :layer; H.objective!(pb, w, x0)
H.reset_work!(); H.gn_diagonal!(H.similar_state(x0), pb, w, x0); wfd = H.work_equivalent()
println("Part 1: per-layer Ã diagonal, full-Hessian (differences of ∇Φ) vs GN products; build cost $(wfd) vs $(wgn) equivalent products")
for f in 2:4
    ks = collect(axes(x0[f], 3))
    r = [D2[f][1, 1, k] / D1[f][1, 1, k] for k in ks]
    @printf("  Ã%s: ratio GN/full per layer: min %.3e median %.3e max %.3e; layers with ratio < 0.5 or > 2: %d of %d\n",
            "xyz"[f-1], minimum(r), median(r), maximum(r), count(q -> q < 0.5 || q > 2, r), length(r))
end
flush(stdout)
# ---- Part 2: one candidate per geometry, same radius ----
configs = [("joint (default)", (;), :layer), ("magnetic only", (; magnetic_only = true), :layer),
           ("pressure stiffer ×10", (; block_scale = (10.0, 1.0, 1.0)), :layer), ("pressure stiffer ×100", (; block_scale = (100.0, 1.0, 1.0)), :layer),
           ("Δ from full D-norm", (; Δ_init = :full), :layer), ("joint, GN layer diagonal", (;), :layer_gn)]
# the common radius: the default joint rule at the checkpoint
xr = deepcopy(x0); wr = H.SMHSWork(par); H.objective!(pb, wr, xr)
_, _, _, rr = H.joint_gn_solve!(pb, wr, xr, ulo; max_candidates = 0)
Δc = rr.Δ
println("Part 2: one candidate from the checkpoint, common radius Δ = $(Δc) (default joint rule); Δ_init = :full uses its own rule")
for (name, kw, pc) in configs
    pb.pc_atilde = pc
    x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
    H.reset_work!()
    kwd = haskey(kw, :Δ_init) ? kw : merge(kw, (; Δ0 = Δc))
    st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; max_candidates = 1, diagnostics = true, kwd...)
    c = res.candidates[1]; d = c.diag
    @printf("  %-26s pred %.4e actual %.4e ρ %.3f | ‖δ‖_D/Δ %.2f | shares y (u %.2f, A %.2f, b %.1e) → final (u %.2f, A %.2f, b %.1e) | max ΔP/P %.2e, ΔB %.2e | model: pressure %.3e, magnetic %.3e, cross %.3e | CG %s rel %.2e | Cauchy %s | work %.0f\n",
            name, c.pred, c.actual, c.rho, c.ynorm / c.Delta, d.share_y.u, d.share_y.A, d.share_y.b, d.share_final.u, d.share_final.A, d.share_final.b,
            d.dPmax, d.dBmax, d.pred_u, d.pred_A, d.pred_cross, string(c.cg_exit), c.cg_rel, c.cauchy, H.work_equivalent())
    flush(stdout)
end
pb.pc_atilde = :layer
# ---- Part 3: short continuations at matched work ----
println("Part 3: continuations, $(budget) equivalent products each")
for (name, kw, pc) in configs[[1, 2, 4, 6]]
    pb.pc_atilde = pc
    x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
    H.reset_work!(); tm = time()
    st, why, n, res = H.joint_gn_solve!(pb, wx, x, ulo; work_budget = budget, kw...)
    f, t, _ = H.objective!(pb, wx, x); rms, _ = H.layer_residual_stats(pb, wx)
    @printf("  %-26s Φ %.5e raw force %.4e max RMS %.3e | accepted %d, candidates %d, Cauchy %d | work %.0f, %.0f s\n",
            name, f, t[1] / pb.lamF, maximum(rms), n, res.counts.cand, res.counts.cauchy, H.work_equivalent(), time() - tm)
    flush(stdout)
end
pb.pc_atilde = :layer
# the alternation baseline at the same budget, for reference
x = deepcopy(x0); wx = H.SMHSWork(par); H.objective!(pb, wx, x)
H.reset_work!(); tm = time()
st, why, n, hist = H.alternating_solve!(pb, wx, x; max_outer = 10_000, a_iter = 100, u_steps = 5, ulo = ulo, work_budget = budget)
f, t, _ = H.objective!(pb, wx, x); rms, _ = H.layer_residual_stats(pb, wx)
@printf("  %-26s Φ %.5e raw force %.4e max RMS %.3e | cycles %d | work %.0f, %.0f s\n", "alternation (ln P)", f, t[1] / pb.lamF, maximum(rms), n, H.work_equivalent(), time() - tm)
