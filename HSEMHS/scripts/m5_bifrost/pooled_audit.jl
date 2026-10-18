# Post-hoc sensitivity of the fixed-metric audit (CODEX_REVIEW_M5c_2026-10-18.md §1): evaluate the checkpoint and
# both saved endpoints with (a) each state's own reference metric and (b) a symmetric pooled metric — the
# elementwise maximum of the three diagonals, floored as in `reference_metric`. One ruler per row, the same for
# every state. The original checkpoint-metric verdict is retained; this is a declared post-hoc analysis.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/pooled_audit.jl out/m5c
include(joinpath(@__DIR__, "bif_setup.jl"))
root = ARGS[1]
pb, xinit, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
states = Dict{String,Any}(); uinit = H.newfield(par)
for (tag, file) in (("checkpoint", "ck.h5"), ("alt", "end_alt.h5"), ("joint", "end_joint.h5"))
    x = deepcopy(xinit)
    h5open(joinpath(root, file)) do f
        for q in 1:5; copyto!(parent(x[q]), read(f["x$q"])); end
        tag == "checkpoint" && copyto!(parent(uinit), read(f["u_init"]))
    end
    H.exchange_halos!(par, collect(x)); states[tag] = x
end
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
w = H.SMHSWork(par)
H.objective!(pb, w, states["checkpoint"])
dPu = H.dP_unobs!(H.newfield(par), pb, w)
ident = [dPu[i, j, k] <= 0.1 * w.P[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
metrics = Dict{String,Any}()
for tag in ("checkpoint", "alt", "joint")
    H.objective!(pb, w, states[tag]); metrics[tag] = H.reference_metric(pb, w, states[tag])
end
pooled = H.similar_state(xinit)
for f in 1:5
    pf = pooled[f]; d1 = metrics["checkpoint"][f]; d2 = metrics["alt"][f]; d3 = metrics["joint"][f]
    H.owned_map!((i, j, k) -> max(d1[i, j, k], d2[i, j, k], d3[i, j, k]), pf, par)
end
metrics["pooled (max)"] = pooled
println("| metric | state | m | m_u | m_A | u* max (ident / rest) | u* p99 |")
println("|---|---|---|---|---|---|---|")
for met in ("checkpoint", "alt", "joint", "pooled (max)"), tag in ("checkpoint", "alt", "joint")
    H.objective!(pb, w, states[tag])
    a = H.full_audit(pb, w, states[tag], metrics[met], ulo; ident = ident)
    @printf("| %s | %s | %.4e | %.3e | %.3e | %.3e / %.3e | %.3e |\n", met, tag, a.m, a.m_u, a.m_A, a.u_max_ident, a.u_max_rest, a.u_p99)
    flush(stdout)
end
