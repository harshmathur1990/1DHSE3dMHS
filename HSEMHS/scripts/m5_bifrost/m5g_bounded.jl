# M5g: short validation of the repaired bounded joint GN (Codex M5f review, findings 1–2 and §4–5).
# Joint GN ×100 from START.h5 (the M5f pilot start, QP m = 0.05) with the face limits of FIELD.h5 as hard bounds
# (|c| ≤ 5 − m on S, 3 − m off S), a log barrier λ on them, fraction to the boundary τ, and the M5g repairs:
# barrier-safe preconditioner (no probe leaves the barrier domain; local barrier diagonal), face-feasible candidates
# compared after truncation, final bound check.
#
# PREDECLARED engineering stops (not convergence, not infeasibility):
#   - work cap `budget` (default 5000 equivalent GN products) and wall limit `wall` seconds (default 7200);
#   - radius collapse (solver);
#   - stagnation: at windows of `window` products (default 1000), "progress" = a ≥ 1 % decrease, relative to the
#     previous window, of the raw force OR the max layer RMS OR the magnetic stationarity ‖g_mag‖_{D⁻¹}; stop after
#     two consecutive windows without progress;
#   - truncation stall: in a window, ≥ 50 % of candidates truncated to a bound fraction < 1e-3.
# Logs per window: Φ, barrier-free Φ₀ = Φ − Φ_H, Φ_H, raw force, layer RMS / p99, face and centre max / tail, min face
# slack and #faces within 1e-3 of the bound, stationarity (pressure optimality; ‖g‖_{D⁻¹} of the magnetic block),
# windowed max |ΔP/P| and |ΔB|/max(|B|, σ) SEPARATELY, and the window's candidate statistics (accepted, Cauchy,
# truncated, median fraction, CG exits). Checkpoints carry the full restart state: x, work, Δ, μ, λ, τ, margin,
# start/field paths, git revision.
# usage (from HSEMHS/): julia --project=. -t 16 scripts/m5_bifrost/m5g_bounded.jl START.h5 FIELD.h5 OUTDIR tau
#                        [budget=5000] [window=1000] [lambda=0.1] [margin=1e-3] [wall=7200] [RESTART.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
start, fieldf, outdir = ARGS[1], ARGS[2], ARGS[3]
btau = parse(Float64, ARGS[4])
budget = parse(Float64, get(ARGS, 5, "5000")); window = parse(Float64, get(ARGS, 6, "1000"))
lamH = parse(Float64, get(ARGS, 7, "0.1")); margin = parse(Float64, get(ARGS, 8, "1e-3")); wall = parse(Float64, get(ARGS, 9, "7200"))
restart = get(ARGS, 10, "")
mkpath(outdir)
gitrev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "unknown" end
ck = normpath(joinpath(@__DIR__, "..", "..", "out", "m5c", "ck.h5"))
pb, x0, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
src = isempty(restart) ? start : restart
h5open(src) do f; for q in 1:5; copyto!(parent(x0[q]), read(f["x$q"])); end; end
Δrs, μrs, wrs = NaN, 1e-6, 0.0
if !isempty(restart)
    h5open(restart) do f
        a = attributes(f)
        Δrs = read(a["Delta"]); μrs = read(a["mu"]); wrs = read(a["work"])
        (read(a["lambda"]) == lamH && read(a["tau"]) == btau && read(a["margin"]) == margin) ||
            error("restart configuration differs from the arguments: a restart with changed settings is a new experiment")
    end
end
uinit = H.newfield(par); h5open(ck) do f; copyto!(parent(uinit), read(f["u_init"])); end
H.exchange_halos!(par, collect(x0))
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
sup = h5open(fieldf) do f; [read(f["support$q"]) for q in 1:3]; end
hard = (H.newfield(par), H.newfield(par), H.newfield0(par))
for q in 1:3
    k0 = first(axes(hard[q], 3))
    H.owned_map!((i, j, k) -> (sup[q][i, j, k-k0+1] > 0.5 ? 5.0 : 3.0) - margin, hard[q], par)
end
H.exchange_halos!(par, collect(hard))
nS = sum(count(>(0.5), s) for s in sup)

w = H.SMHSWork(par)
cvals(w) = [(w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]) for q in 1:3 for k in axes(w.B[q], 3) for j in 1:b.ny for i in 1:b.nx]
const HV = [hard[q][i, j, k] for q in 1:3 for k in axes(hard[q], 3) for j in 1:b.ny for i in 1:b.nx]
# audit metric without the face term (comparable across barrier treatments)
H.objective!(pb, w, x0); Dref = H.reference_metric(pb, w, x0)
H.set_face_hinge!(pb, hard; lambda = lamH, kind = :barrier)
f0, t0, _ = H.objective!(pb, w, x0)
viol = H.face_violation(pb, w, hard)[1]
@printf("M5g bounded validation (git %s): start %s%s | |S| = %d | bounds 5/3 − %.0e | barrier λ %g | τ %g | budget %.0f, window %.0f, wall %.0f s | Φ %.6f | max(|c| − u) %.3e\n",
        gitrev, start, isempty(restart) ? "" : " (restart from $restart, work $wrs)", nS, margin, lamH, btau, budget, window, wall, f0, viol)
viol < 0 || error("start not strictly inside the bounds")
# common default-rule radius, as in m5f_pilot.jl (without the face term, so it matches M5f's Δ0 at this start)
hg = pb.hinge; pb.hinge = nothing
xr = deepcopy(x0); wr = H.SMHSWork(par); H.objective!(pb, wr, xr)
_, _, _, rr = H.joint_gn_solve!(pb, wr, xr, ulo; max_candidates = 0); Δc = rr.Δ
pb.hinge = hg
Δstart = isnan(Δrs) ? Δc : Δrs
Nf = length(HV); Kf = floor(Int, 0.01Nf)

function magstat(pb, w, x)
    g = H.similar_state(x); H.fg!(g, pb, w, x)
    D = H.gn_diagonal!(H.similar_state(x), pb, w, x)
    s = 0.0
    for f in 2:5, k in axes(g[f], 3), j in 1:b.ny, i in 1:b.nx     # g[5] is zero off rank 0's [1, 1, 1]
        s += g[f][i, j, k]^2 / D[f][i, j, k]
    end
    s = MPI.Allreduce(s, +, par.comm)
    D1 = H.newfield(par); H.gn_diag_v!(D1, pb, w, x)
    uopt = H.pressure_optimality(pb, w, g[1], D1, ulo, H.newfield(par), H.newfield(par))
    return sqrt(s), uopt
end

snap_prev = Ref(H.measured_snapshot(pb, w)); acc_rows = NamedTuple[]; res_ref = Ref{Any}(nothing)
nextlog = Ref(window); nlog = Ref(0); prev = Ref((F = Inf, rms = Inf, gm = Inf)); noprog = Ref(0)
stopwhy = Ref(""); tm0 = time(); lastcand = Ref(0)
const res_cands = Ref{Vector{NamedTuple}}(NamedTuple[])   # the solver's candidate list (cand_sink), read by the callback
function logpoint(x, work, tag, cands)
    wx = H.SMHSWork(par); f, t, _ = H.objective!(pb, wx, x)
    rms, p99 = H.layer_residual_stats(pb, wx)
    cf = cvals(wx); slack = HV .- abs.(cf)
    cc = [wx.dB[q][i, j, k] / pb.sig[q][i, j, k] for q in 1:3 for k in 1:NZ for j in 1:b.ny for i in 1:b.nx]
    gm, uopt = magstat(pb, wx, x)
    a = H.full_audit(pb, wx, x, Dref, ulo)
    H.objective!(pb, wx, x)
    dPw = H.measured_change_P(pb, wx, snap_prev[]); dBw = H.measured_change_B(pb, wx, snap_prev[])
    snap_prev[] = H.measured_snapshot(pb, wx)
    hval = pb.hinge.val[]
    nacc = length(acc_rows); aerr = nacc > 0 ? acc_rows[end].anchor_err : NaN
    nc = length(cands); ntr = count(c -> c.bound_frac < 1, cands); nsmall = count(c -> c.bound_frac < 1e-3, cands)
    fr = sort([c.bound_frac for c in cands if c.bound_frac < 1]); medfr = isempty(fr) ? 1.0 : fr[cld(length(fr), 2)]
    ex = Dict{Symbol,Int}(); for c in cands; ex[c.cg_exit] = get(ex, c.cg_exit, 0) + 1; end
    Δn = nacc > 0 ? acc_rows[end].Delta : Δstart; μn = nacc > 0 ? acc_rows[end].mu : μrs
    @printf("%s work %6.0f | Φ %.6f Φ₀ %.6f Φ_H %.3e F %.4e B %.6f S %.4e | raw force %.4e | layer RMS max %.3e (k %d) p99 max %.3e | FACE max %.5fσ >3σ %.4f%% (K %d) #>5σ %d min slack %.2e #slack<1e-3 %d | CENTRE max %.4fσ >3σ %.4f%% | u-opt %.3e ‖g_mag‖_D⁻¹ %.3e gÃ(ref) %.2e | anchor err %.1e | Δ %.3e μ %.2e | window: ΔP/P %.2e ΔB %.2e; cand %d acc %d Cauchy %d truncated %d (<1e-3: %d, median %.2e) CG %s | wall %.0f s\n",
            tag, work, f, f - hval, hval, t[1], t[3], t[4], t[1] / pb.lamF, maximum(rms), argmax(rms), maximum(p99),
            maximum(abs, cf), 100count(>(3), abs.(cf)) / Nf, Kf, count(>(5), abs.(cf)), minimum(slack), count(<(1e-3), slack),
            maximum(abs, cc), 100count(>(3), abs.(cc)) / length(cc), uopt, gm, a.gA, aerr, Δn, μn, dPw, dBw,
            nc, count(c -> c.accepted, cands), count(c -> c.cauchy, cands), ntr, nsmall, medfr, string(ex), time() - tm0)
    flush(stdout)
    nlog[] += 1
    h5open(joinpath(outdir, @sprintf("ck_%02d.h5", nlog[])), "w") do fo
        for q in 1:5; fo["x$q"] = parent(x[q]); end
        at = attributes(fo)
        at["work"] = work + wrs; at["Delta"] = Δn; at["mu"] = μn; at["lambda"] = lamH; at["tau"] = btau; at["margin"] = margin
        at["start"] = start; at["field"] = fieldf; at["git"] = gitrev; at["tag"] = tag
    end
    return (F = t[1] / pb.lamF, rms = maximum(rms), gm = gm, nc = nc, nsmall = nsmall)
end
x = deepcopy(x0)
H.reset_work!()
logpoint(x, 0.0, "start", NamedTuple[])
function cb(acc, row)
    push!(acc_rows, row)
    if time() - tm0 > wall
        stopwhy[] = "wall limit"; return :stop
    end
    if row.work >= nextlog[]
        cands = res_cands[][lastcand[]+1:end]; lastcand[] = length(res_cands[])
        s = logpoint(x, row.work, "window", cands)
        prog = s.F < 0.99prev[].F || s.rms < 0.99prev[].rms || s.gm < 0.99prev[].gm
        noprog[] = prog ? 0 : noprog[] + 1
        prev[] = (F = s.F, rms = s.rms, gm = s.gm)
        while nextlog[] <= row.work; nextlog[] += window; end
        if noprog[] >= 2
            stopwhy[] = "stagnation: no ≥1 % progress in force, RMS or ‖g_mag‖ over two windows"; return :stop
        end
        if s.nc > 0 && s.nsmall >= 0.5s.nc
            stopwhy[] = "truncation stall: ≥ 50 % of the window's candidates truncated below 1e-3"; return :stop
        end
    end
    return nothing
end
st, why, n, res = H.joint_gn_solve!(pb, w, x, ulo; work_budget = budget, Δ0 = Δstart, μ0 = μrs, block_scale = (100.0, 1.0, 1.0),
                                     face_hard = hard, bound_tau = btau, callback = cb, cand_sink = res_cands)
cands = res.candidates[lastcand[]+1:end]
logpoint(x, H.work_equivalent(), "end", cands)
nb = count(cd -> cd.bound_frac < 1, res.candidates)
@printf("done: %s (%s%s); accepted %d / %d candidates; Cauchy %d; truncated %d (median fraction %.3g); wall %.0f s; Δ0 %.4f\n",
        string(st), why, isempty(stopwhy[]) ? "" : ": " * stopwhy[], n, res.counts.cand, res.counts.cauchy, nb,
        nb > 0 ? sort([cd.bound_frac for cd in res.candidates if cd.bound_frac < 1])[cld(nb, 2)] : 1.0, time() - tm0, Δstart)
h5open(joinpath(outdir, "candidates.h5"), "w") do fo
    for key in (:rho, :pred, :pred_gn, :predc, :actual, :Delta, :ynorm, :bound_frac, :bound_frac_gn, :bound_frac_cauchy, :bound_viol, :mu, :cg_rel)
        fo[string(key)] = Float64[getfield(c, key) for c in res.candidates]
    end
    fo["accepted"] = Int[c.accepted for c in res.candidates]; fo["cauchy"] = Int[c.cauchy for c in res.candidates]
end
