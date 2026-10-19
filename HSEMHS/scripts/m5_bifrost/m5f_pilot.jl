# M5f: short data-feasible MHS pilot (Codex M5e review §4, deliverable 3).
# Joint GN ×100 (fixed reference configuration, no anchor-tangent) from START.h5, with the face-model data
# limits preserved as HARD bounds on the fixed support of FIELD.h5 (support1..3: |c| ≤ t5 − m on S, t3 − m off S),
# enforced by fraction-to-the-boundary truncation, plus a hinge penalty from the soft thresholds (bounds − band)
# so the GN model sees them. mode = "hard" (that) or "free" (same start and solver, no bounds or hinge: the drift
# diagnostic of option (a)), or "barrier" (hard bounds + log barrier −λ log(1 − (c/u)²) on the hard bounds u
# instead of the hinge; band unused).
# Logs at fixed work intervals: objective components, raw force, all-layer RMS and p99 with the worst location,
# max and tail for both observation operators, pressure stationarity, anchor and bound errors, accepted/rejected
# steps and work, and the max |ΔP/P|, |ΔB|/max(|B|, σ) over the last interval (windowed changes). A checkpoint
# (x1..x5) is saved at every log point.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/m5f_pilot.jl START.h5 FIELD.h5 OUTDIR mode budget
#                        [interval=500] [lambda_hinge=100] [band=0.05] [margin=1e-3] [bound_tau=0.999]
#                        julia ... m5f_pilot.jl START.h5 FIELD.h5 OUTDIR check [hinge|barrier]  (gradient / GN checks)
include(joinpath(@__DIR__, "bif_setup.jl"))
start, fieldf, outdir, mode = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
budget = something(tryparse(Float64, get(ARGS, 5, "3000")), 3000.0); interval = parse(Float64, get(ARGS, 6, "500"))
lamH = parse(Float64, get(ARGS, 7, "100")); band = parse(Float64, get(ARGS, 8, "0.05")); margin = parse(Float64, get(ARGS, 9, "1e-3"))
btau = parse(Float64, get(ARGS, 10, "0.999"))
mkpath(outdir)
ck = normpath(joinpath(@__DIR__, "..", "..", "out", "m5c", "ck.h5"))
pb, x0, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
h5open(start) do f; for q in 1:5; copyto!(parent(x0[q]), read(f["x$q"])); end; end
uinit = H.newfield(par); h5open(ck) do f; copyto!(parent(uinit), read(f["u_init"])); end
H.exchange_halos!(par, collect(x0))
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
# support → hard and soft bounds in σ units, Bobsf layout
sup = h5open(fieldf) do f; [read(f["support$q"]) for q in 1:3]; end
hard = (H.newfield(par), H.newfield(par), H.newfield0(par)); soft = (H.newfield(par), H.newfield(par), H.newfield0(par))
for q in 1:3
    k0 = first(axes(hard[q], 3))
    H.owned_map!((i, j, k) -> (sup[q][i, j, k-k0+1] > 0.5 ? 5.0 : 3.0) - margin, hard[q], par)
    H.owned_map!((i, j, k) -> hard[q][i, j, k] - band, soft[q], par)
end
nS = sum(count(>(0.5), s) for s in sup)

w = H.SMHSWork(par)
cvals(w) = [(w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]) for q in 1:3 for k in axes(w.B[q], 3) for j in 1:b.ny for i in 1:b.nx]
hvals() = [hard[q][i, j, k] for q in 1:3 for k in axes(hard[q], 3) for j in 1:b.ny for i in 1:b.nx]
const HV = hvals()

if mode == "check"
    # hinge with thresholds at half the current |c| (many faces active): directional derivative and GN symmetry
    H.objective!(pb, w, x0)
    sh = (H.newfield(par), H.newfield(par), H.newfield0(par))
    for q in 1:3
        H.owned_map!((i, j, k) -> 0.5abs((w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k])), sh[q], par)
    end
    if get(ARGS, 5, "hinge") == "barrier"           # barrier at 2|c| + 1 (all faces well inside)
        for q in 1:3; shq = sh[q]; H.owned_map!((i, j, k) -> 4shq[i, j, k] + 1.0, shq, par); end
        H.set_face_hinge!(pb, sh; lambda = lamH, kind = :barrier)
    else
        H.set_face_hinge!(pb, sh; lambda = lamH)
    end
    g = H.similar_state(x0); f0 = H.fg!(g, pb, w, x0)[1]
    d = H.similar_state(x0); for q in 2:4; dq = d[q]; H.owned_map!((i, j, k) -> randn(), dq, par); end
    gd = H.dotN(par, g, d)
    for h in (1e-3, 1e-4, 1e-5)
        xp = deepcopy(x0); H.lincomb!(xp, 1.0, x0, h, d, par); fp = H.objective!(pb, H.SMHSWork(par), xp)[1]
        xm = deepcopy(x0); H.lincomb!(xm, 1.0, x0, -h, d, par); fm = H.objective!(pb, H.SMHSWork(par), xm)[1]
        @printf("check: h %.0e  central FD %.10e  gradient %.10e  rel err %.2e  (Φ_H %.4e)\n", h, (fp - fm) / 2h, gd, abs((fp - fm) / 2h - gd) / abs(gd), pb.hinge.val[])
    end
    H.objective!(pb, w, x0); wl = H.SMHSWork(par)
    e = H.similar_state(x0); for q in 2:4; eq = e[q]; H.owned_map!((i, j, k) -> randn(), eq, par); end
    Hd = H.similar_state(x0); He = H.similar_state(x0)
    H.gn_hvp!(Hd, pb, w, wl, d); H.gn_hvp!(He, pb, w, wl, e)
    @printf("check: GN symmetry eᵀHd %.10e  dᵀHe %.10e  dᵀHd %.4e (> 0)\n", H.dotN(par, e, Hd), H.dotN(par, d, He), H.dotN(par, d, Hd))
    exit(0)
end

hardmode = mode in ("hard", "barrier")
mode == "hard" && H.set_face_hinge!(pb, soft; lambda = lamH)
mode == "barrier" && H.set_face_hinge!(pb, hard; lambda = lamH, kind = :barrier)
f0, t0, _ = H.objective!(pb, w, x0)
c = cvals(w); viol = maximum(abs.(c) .- HV)
@printf("start %s: Φ %.6f | |S| = %d | hard bounds 5/3 − %.0e, soft band %.3f, λ_H %g, τ %g | max(|c| − hard) %.3e%s\n",
        start, f0, nS, margin, band, lamH, btau, viol, hardmode && viol > 0 ? "  START VIOLATES THE HARD BOUNDS" : "")
hardmode && viol > 0 && error("start outside the hard bounds")
# default-rule radius at this state (as in long_comparison.jl)
xr = deepcopy(x0); wr = H.SMHSWork(par); H.objective!(pb, wr, xr)
_, _, _, rr = H.joint_gn_solve!(pb, wr, xr, ulo; max_candidates = 0); Δc = rr.Δ
Dref = H.reference_metric(pb, w, x0)
Nf = length(c); Kf = floor(Int, 0.01Nf)

function worst_force(pb, w)
    cc = pb.NF / (2pb.lamF); best = (0.0, 0, 0, 0, 'x')
    for k in 1:NZ, j in 1:b.ny, i in 1:b.nx
        rx = abs(w.aR[1][i, j, k] * cc / (H.wxf(pb, i, j, k) * sqrt(pb.invS2f[1][i, j, k])))
        ry = abs(w.aR[2][i, j, k] * cc / (H.wyf(pb, i, j, k) * sqrt(pb.invS2f[2][i, j, k])))
        rz = k <= NZ - 1 ? abs(w.aR[3][i, j, k] * cc / (H.wzf(pb, i, j, k) * sqrt(pb.invS2f[3][i, j, k]))) : 0.0
        for (r, a) in ((rx, 'x'), (ry, 'y'), (rz, 'z'))
            r > best[1] && (best = (r, i, j, k, a))
        end
    end
    return best
end
snap_prev = Ref(H.measured_snapshot(pb, w)); acc_rows = NamedTuple[]
nextlog = Ref(interval); nlog = Ref(0)
function logpoint(x, work, tag)
    wx = H.SMHSWork(par); f, t, _ = H.objective!(pb, wx, x)
    rms, p99 = H.layer_residual_stats(pb, wx); wf = worst_force(pb, wx)
    kr = argmax(rms); kp = argmax(p99)
    cf = cvals(wx)
    cc = [wx.dB[q][i, j, k] / pb.sig[q][i, j, k] for q in 1:3 for k in 1:NZ for j in 1:b.ny for i in 1:b.nx]
    g = H.similar_state(x); H.fg!(g, pb, wx, x); D1 = H.newfield(par); H.gn_diag_v!(D1, pb, wx, x)
    uopt = H.pressure_optimality(pb, wx, g[1], D1, ulo, H.newfield(par), H.newfield(par))
    a = H.full_audit(pb, wx, x, Dref, ulo)
    H.objective!(pb, wx, x)
    dPw = maximum(abs(expm1(wx.u[i, j, k] - snap_prev[][1][i - wx.u.offsets[1], j - wx.u.offsets[2], k - wx.u.offsets[3]])) for i in 1:b.nx, j in 1:b.ny, k in 1:NZ)
    dBw = H.measured_change(pb, wx, snap_prev[])
    snap_prev[] = H.measured_snapshot(pb, wx)
    margin_u = minimum(wx.u[i, j, k] - ulo[i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx)
    nacc = length(acc_rows); aerr = nacc > 0 ? acc_rows[end].anchor_err : NaN
    hval = pb.hinge === nothing ? 0.0 : pb.hinge.val[]
    @printf("%s work %6.0f | Φ %.6f F %.4e NQ+H %.3e (Φ_H %.2e) B %.6f S %.4e | raw force %.4e | layer RMS max %.3e (k %d) p99 max %.3e (k %d) worst %.3e at %c-face (%d,%d,%d) | FACE max %.5fσ >3σ %d (%.4f%%, K %d) #>5σ %d max(|c|−hard) %+.2e | CENTRE max %.4fσ >3σ %.4f%% | u-opt %.2e m_u %.2e gÃ %.2e | anchor err %.1e u-margin %.1e | accepted %d | window ΔP/P %.2e ΔB %.2e\n",
            tag, work, f, t[1], t[2], hval, t[3], t[4], t[1] / pb.lamF, maximum(rms), kr, maximum(p99), kp, wf[1], wf[5], wf[2], wf[3], wf[4],
            maximum(abs, cf), count(>(3), abs.(cf)), 100count(>(3), abs.(cf)) / Nf, Kf, count(>(5), abs.(cf)), maximum(abs.(cf) .- HV),
            maximum(abs, cc), 100count(>(3), abs.(cc)) / length(cc), uopt, a.m_u, a.gA, aerr, margin_u, nacc, dPw, dBw)
    flush(stdout)
    nlog[] += 1
    h5open(joinpath(outdir, @sprintf("%s_%02d.h5", mode, nlog[])), "w") do fo
        for q in 1:5; fo["x$q"] = parent(x[q]); end
        attributes(fo)["work"] = work
        nacc > 0 && (attributes(fo)["Delta"] = acc_rows[end].Delta)
    end
end
x = deepcopy(x0)
H.reset_work!()
logpoint(x, 0.0, mode)
function cb(acc, row)
    push!(acc_rows, row)
    if row.work >= nextlog[]
        logpoint(x, row.work, mode)
        while nextlog[] <= row.work; nextlog[] += interval; end
    end
end
tm = time()
st, why, n, res = H.joint_gn_solve!(pb, w, x, ulo; work_budget = budget, Δ0 = Δc, block_scale = (100.0, 1.0, 1.0),
                                     face_hard = hardmode ? hard : nothing, bound_tau = btau, callback = cb)
logpoint(x, H.work_equivalent(), mode * "-end")
nb = count(cd -> cd.bound_frac < 1, res.candidates)
@printf("%s done: %s (%s); accepted %d / %d candidates; Cauchy %d; bound-truncated %d (median fraction %.3g); wall %.0f s; common Δ0 %.4f\n",
        mode, string(st), why, n, res.counts.cand, res.counts.cauchy, nb,
        nb > 0 ? sort([cd.bound_frac for cd in res.candidates if cd.bound_frac < 1])[cld(nb, 2)] : 1.0, time() - tm, Δc)
