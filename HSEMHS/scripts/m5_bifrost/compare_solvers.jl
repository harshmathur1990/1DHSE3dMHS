# M5c solver comparison (CODEX_REVIEW_M5_FOLLOWUP.md Q3; CODEX_REVIEW_M5_SECOND_FOLLOWUP.md Q2–Q3):
# corrected inexact alternation in ln P (`alt`) vs joint damped trust-region GN (`joint`), from the same
# frozen-B checkpoint, with the same collapse bound, fixed reference metric and identifiability mask, and a
# matched budget of equivalent GN products (`work_equivalent`: objective+gradient = 1, GN product = 1,
# bare objective = ½; Cauchy trials and diagonal builds included). Diagnostics are not charged.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/compare_solvers.jl alt|joint CHECKPOINT.h5 [budget] [out.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
arm = ARGS[1]; ck = ARGS[2]; budget = parse(Float64, get(ARGS, 3, "3000")); out = get(ARGS, 4, replace(ck, ".h5" => "_cmp_$(arm).h5"))
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
w = H.SMHSWork(par)
uinit = H.newfield(par)
h5open(ck) do f
    for q in 1:5
        copyto!(parent(x[q]), read(f["x$q"]))
    end
    copyto!(parent(uinit), read(f["u_init"]))
end
H.exchange_halos!(par, collect(x))
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
H.objective!(pb, w, x)
dPu = H.dP_unobs!(H.newfield(par), pb, w)
ident = [dPu[i, j, k] <= 0.1 * w.P[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
Dref = H.reference_metric(pb, w, x)                    # fixed metric at the common checkpoint
lF = pb.lamF
function bdev(w)                                       # max |B − Bobs|/σ on faces (magnetic outliers)
    m = 0.0
    for q in 1:3, k in axes(w.B[q], 3), j in 1:b.ny, i in 1:b.nx
        m = max(m, abs(w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]))
    end
    return m
end
prevu = Ref([w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]); prevB = Ref(H.measured_snapshot(pb, w))
function log_state(tag, work, extra = "")
    a = H.full_audit(pb, w, x, Dref, ulo; ident = ident)
    f, t, _ = H.objective!(pb, w, x)
    rms, p99 = H.layer_residual_stats(pb, w)
    u = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    dP = abs.(expm1.(u .- prevu[]))
    chB = H.measured_change(pb, w, (copy(parent(w.u)), prevB[][2]))  # field part (pressure part is 0 by construction)
    margin = minimum(w.u[i, j, k] - ulo[i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx)
    aerr = abs(H.anchor_mean(pb, w.u) - pb.lnPref)
    @printf("%s %-8s work %6.0f | Φ %.5e raw force %.4e data %.4f Q %.3e max|δB|/σ %.2f | audit m %.3e (u %.2e, Ã %.2e, b̄ %.1e) u* max %.2e p99 %.2e ident %.2e rest %.2e | gA %.2e gb·σ %.1e | ΔP/P max %.2e (ident %.2e) ΔB %.2e | max RMS|R|/S %.2e | margin %.1e anchor %.1e %s\n",
            arm, tag, work, f, t[1] / lF, t[3], t[2], bdev(w), a.m, a.m_u, a.m_A, a.m_b, a.u_max, a.u_p99, a.u_max_ident, a.u_max_rest,
            a.gA, a.gb_sigma, maximum(dP), maximum(dP[ident]), chB, maximum(rms), margin, aerr, extra)
    flush(stdout)
    prevu[] = u; prevB[] = H.measured_snapshot(pb, w)
    return a
end
saved = copy(H.WORK_COUNTS); a0 = log_state("start", 0.0); copyto!(H.WORK_COUNTS, saved)
H.reset_work!(); tm = time()
if arm == "alt"
    cb(outer, row) = log_state(@sprintf("cyc %d", outer), row.work,
                               @sprintf("| Ã %s/%d%s u %s/%d", string(row.a_status), row.a_iter, row.a_fail ? " FAIL" : "", string(row.u_status), row.usteps))
    st, why, n, hist = H.alternating_solve!(pb, w, x; max_outer = 10_000, a_iter = 100, u_steps = 5, coord = :lnP, Dref = Dref, ulo = ulo,
                                            work_budget = budget, callback = cb)
    summary = @sprintf("cycles %d; Ã block failures %d; u block failures %d", n, count(h -> h.a_fail, hist), count(h -> h.u_fail, hist))
else
    cbj(acc, row) = acc % 5 == 0 && log_state(@sprintf("acc %d", acc), row.work, @sprintf("| ρ %.2f CG %s Cauchy %s", row.rho, string(row.cg_exit), row.cauchy))
    st, why, n, res = H.joint_gn_solve!(pb, w, x, ulo; work_budget = budget, cg_iter = 100, cg_rtol = 1e-2, callback = cbj)
    summary = @sprintf("accepted %d; counts %s", n, string(res.counts))
end
wall = time() - tm; work = H.work_equivalent(); cnt = copy(H.WORK_COUNTS)
a1 = log_state("final", work)
@printf("%s RESULT: %s (%s) | work %.0f (objective %d, adjoint %d, GN products %d) in %.0f s | audit m %.3e → %.3e (÷%.2e) | %s\n",
        arm, string(st), why, work, cnt[1], cnt[2], cnt[3], wall, a0.m, a1.m, a0.m / a1.m, summary)
println("worst pressure-audit nodes at the end: ", a1.worst)
h5open(out, "w") do f
    for q in 1:5
        f["x$q"] = parent(x[q])
    end
end
