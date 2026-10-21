# M5h: full-input MHS progress test (REVIEW_M5g_2026-10-21.md; user request 21 Oct 2026).
# The complete IO/input-mhs-hse.nc: 504 × 504 × 177 cells, every column and level (bif_problem(1, 1): no averaging, no
# subsampling, no crop). Current code, current scientific objective (staggered MHS, D9b defaults, λ_F = 1e-4, λ_Q = 1e-3,
# σ_B synthetic 5 % ⊕ 5 G, data on faces, bottom anchor P_ref from the file, EOS table cache/eos_table.h5), solver joint
# trust-region GN with the ×100 pressure block scaling (the M5e/M5f reference configuration) and the M5g preconditioner.
# NO face hard bounds or barrier: no native-grid admissible field or support exists, so this is a current-code MHS
# progress run, not a run shown to satisfy hard data limits.
#
# Convergence variable (the user's criterion): after every ACCEPTED iteration n, z_τ1[n, x, y] = height of log₁₀ τ₅₀₀ = 0
# (`tau1_heights!`: χ₅₀₀ from the solver's EOS table at the actual P and the input T, `tau_column!` top = :exp_tail,
# H_max = 5e6 m, μ = 1, `z_tau1`), and dz_max[n] = max |z_τ1[n] − z_τ1[n−1]| over all columns [m]. Threshold < 1e-3 m with
# a valid crossing in every column. The initial state has no previous-state difference.
#
# DECLARED stops: (i) user criterion met on 3 consecutive accepted iterations (a safeguard against a single small step;
# the threshold itself is unchanged); (ii) wall cap; (iii) solver radius collapse; (iv) an external wall guard.
# Per accepted iteration a row goes to OUTDIR/history.csv (height statistics, work, time, force, objective terms, data
# maxima/tails, step/trust/rejection information) and the z_τ1 map to OUTDIR/ztau1/. Full checkpoints (atmosphere +
# restart state) every `ckpt_hours` and at the end: OUTDIR/atmosphere_A.h5 / _B.h5 (alternating) and atmosphere_final.h5.
#
# usage (from HSEMHS/): julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR probe [init_iter]
#                        julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR run [wall_hours=24] [ckpt_hours=2] [init_iter=3000] [RESTART.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
using Dates
outdir, mode = ARGS[1], ARGS[2]
mkpath(outdir); mkpath(joinpath(outdir, "ztau1"))
gitrev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "unknown" end
const TAU_TOP = :exp_tail; const TAU_HMAX = 5.0e6; const TAU_MU = 1.0; const DZ_THRESH = 1e-3; const NCONSEC = 3
wall_h = mode == "run" ? parse(Float64, get(ARGS, 3, "24")) : 0.0
ckpt_h = mode == "run" ? parse(Float64, get(ARGS, 4, "2")) : 0.0
init_iter = parse(Int, get(ARGS, mode == "run" ? 5 : 3, "3000"))
restart = mode == "run" ? get(ARGS, 6, "") : ""
T0 = time()
logf(s) = (println(Dates.format(now(), "HH:MM:SS"), " [", round(Int, time() - T0), " s] ", s); flush(stdout))
logf("M5h full-resolution MHS (git $gitrev), mode $mode, $(Threads.nthreads()) threads, input $BIF")
pb, x, par, b, NZ = bif_problem(1, 1; init_iter = init_iter, lambda_F = 1e-4, lambda_Q = 1e-3)
MPI.Comm_size(par.comm) == 1 || error("m5h_fullres.jl writes rank-local arrays: run it on one rank (threads only)")
(par.NX, par.NY, NZ) == (504, 504, 177) || error("expected the full 504 × 504 × 177 grid, got $((par.NX, par.NY, NZ))")
logf(@sprintf("setup done: %d × %d × %d, P_ref %.6g Pa (bottom anchor), g %.3f, maxrss %.1f GB", par.NX, par.NY, NZ, exp(pb.lnPref), pb.g, Sys.maxrss() / 2^30))
w = H.SMHSWork(par)
Δ0, μ0, acc_off, work_off = NaN, 1e-6, 0, 0.0
if !isempty(restart)                       # a declared continuation segment: x, ulo, Δ, μ, counters and the last z(τ = 1)
    h5open(restart) do f
        for q in 1:5; copyto!(parent(x[q]), read(f["solver/x$q"])); end
        a = attributes(f); global Δ0 = read(a["Delta"]); global μ0 = read(a["mu"]); global acc_off = read(a["accepted"]); global work_off = read(a["work"])
    end
    H.exchange_halos!(par, collect(x))
end
f0, t0, oob0 = H.objective!(pb, w, x)
oob0 && error("initial state outside the EOS envelope")
ulo = H.newfield(par)
if !isempty(restart)
    h5open(restart) do f; copyto!(parent(ulo), read(f["solver/ulo"])); end
else
    H.owned_map!((i, j, k) -> w.u[i, j, k] - 3.0, ulo, par)      # pressure floor: initial ln P − 3 (as in the M5 runs)
end
zprev = fill(NaN, b.nx, b.ny); fprev = zeros(UInt8, b.nx, b.ny); zcur = similar(zprev); fcur = similar(fprev)
taud = H.newfield(par)
td = @elapsed H.tau1_heights!(zprev, fprev, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
isempty(restart) || h5open(restart) do f       # continuity check: the restart's saved map must equal the recomputed one
    zr = read(f["atmosphere/z_tau1_m"]); fr = read(f["atmosphere/z_tau1_flag"])
    (isequal(zr, zprev) && fr == fprev) || error("restart: recomputed z(τ = 1) differs from the saved map")
end
s0 = H.tau1_change_stats(par, zprev, fprev, nothing, nothing); z0sum = H.tau1_summary(par, zprev, fprev)
logf(@sprintf("initial: Φ %.6e terms %s | z(τ=1) min/median/max %.1f / %.1f / %.1f m, valid %d, missing %d, nonfinite %d, EOS %d | diagnostic %.2f s",
              f0, string(round.(t0; sigdigits = 5)), z0sum.min, z0sum.median, z0sum.max, z0sum.nvalid, s0.nmissing, s0.nnonfinite, s0.neos, td))

if mode == "probe"
    g = H.similar_state(x); wl = H.SMHSWork(par)
    tf = @elapsed H.fg!(g, pb, w, x)
    tg = @elapsed H.gn_hvp!(g, pb, w, wl, x)
    H.objective!(pb, w, x)
    tD = @elapsed H.gn_diagonal!(H.similar_state(x), pb, w, x)
    logf(@sprintf("probe: objective+gradient %.2f s, GN product %.2f s, gn_diagonal! %.2f s, τ diagnostic %.2f s, maxrss %.1f GB",
                  tf, tg, tD, td, Sys.maxrss() / 2^30))
    exit(0)
end

# ---------------------------------------------------------------------------------------------------------------
cvals_face() = (m = 0.0; n3 = 0; n = 0; for q in 1:3, k in axes(w.B[q], 3), j in 1:b.ny, i in 1:b.nx
                    c = abs((w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k])); m = max(m, c); n3 += c > 3; n += 1
                end; (m, n3 / n))
cvals_centre() = (m = 0.0; n3 = 0; n = 0; for q in 1:3, k in 1:NZ, j in 1:b.ny, i in 1:b.nx
                    c = abs(w.dB[q][i, j, k] / pb.sig[q][i, j, k]); m = max(m, c); n3 += c > 3; n += 1
                  end; (m, n3 / n))
hist = open(joinpath(outdir, "history.csv"), isempty(restart) ? "w" : "a")
cols = "accepted,work,elapsed_s,diag_s,dz_max_m,dz_median_m,dz_p99_m,worst_i,worst_j,ncomparable,nmissing,nnonfinite,neos,nprev_invalid,all_valid,below_thresh,consec," *
       "ztau1_min_m,ztau1_median_m,ztau1_max_m,Phi,Phi_F,Phi_NQ,Phi_B,Phi_S,Phi_U,raw_force,rms_max,p99_max,face_max,face_frac3,centre_max,centre_frac3," *
       "dPmax,dBmax,rho,pred,Delta_used,Delta,mu,step_over_Delta,cg_exit,cg_rel,cauchy,rejected_since_last,margin_u,anchor_err"
isempty(restart) && (println(hist, cols); flush(hist))
snap = Ref(H.measured_snapshot(pb, w))
isempty(restart) && h5open(joinpath(outdir, "ztau1", "ztau1_00000.h5"), "w") do fo; fo["z_tau1_m"] = zprev; fo["flag"] = fprev; end
cands = Ref{Vector{NamedTuple}}(NamedTuple[]); lastcand = Ref(0); consec = Ref(0); stopwhy = Ref("")
lastck = Ref(time()); ckslot = Ref(0); nacc_tot = Ref(0); work0 = Ref(0.0); Δlast = Ref(NaN); μlast = Ref(NaN)
if isempty(restart)
    (fm, f3) = cvals_face(); (cm, c3) = cvals_centre(); (rms, p99) = H.layer_residual_stats(pb, w)
    @printf(hist, "0,0,%.1f,%.2f,NaN,NaN,NaN,0,0,0,%d,%d,%d,0,false,false,0,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,,NaN,false,0,NaN,NaN\n",
            time() - T0, td, s0.nmissing, s0.nnonfinite, s0.neos, z0sum.min, z0sum.median, z0sum.max, f0, t0..., t0[1] / pb.lamF, maximum(rms), maximum(p99), fm, f3, cm, c3)
    flush(hist)
end

function save_atmosphere(path, tag)
    tsave = time()
    H.tau1_heights!(zcur, fcur, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU, tau = taud)
    s = pb.s
    h5open(path * ".tmp", "w") do fo
        own(f) = f[1:b.nx, 1:b.ny, axes(f, 3)]
        fo["coords/x_m"] = [pb.grid.x.s[i] for i in 1:b.nx]
        fo["coords/y_m"] = [pb.grid.y.s[j] for j in 1:b.ny]
        fo["coords/z_m"] = collect(pb.grid.z.s); fo["coords/zf_m"] = collect(s.zf)
        fo["atmosphere/Pgas"] = own(w.P); fo["atmosphere/T"] = exp.(own(pb.X)); fo["atmosphere/rho"] = own(w.rho)
        fo["atmosphere/Bx_xface"] = own(w.B[1]); fo["atmosphere/By_yface"] = own(w.B[2]); fo["atmosphere/Bz_zface"] = own(w.B[3])
        for q in 1:3; fo["atmosphere/B$("xyz"[q])_centre"] = own(w.HB[q]); end
        fo["atmosphere/tau500"] = own(taud); fo["atmosphere/z_tau1_m"] = zcur; fo["atmosphere/z_tau1_flag"] = fcur
        for q in 1:5; fo["solver/x$q"] = parent(x[q]); end
        fo["solver/ulo"] = parent(ulo)
        a = attributes(fo)
        a["tag"] = tag; a["git"] = gitrev; a["input"] = BIF; a["eos_table"] = TAB; a["grid"] = [par.NX, par.NY, NZ]
        a["accepted"] = nacc_tot[]; a["work"] = work0[]; a["Delta"] = Δlast[]; a["mu"] = μlast[]
        a["lambda_F"] = pb.lamF; a["lambda_Q"] = pb.lamQ; a["lambda_S"] = pb.lamS; a["block_scale"] = [100.0, 1.0, 1.0]
        a["P_ref_Pa"] = exp(pb.lnPref); a["anchor"] = "bottom"; a["g"] = pb.g; a["face_bounds"] = "none"
        a["tau_top"] = string(TAU_TOP); a["tau_H_max_m"] = TAU_HMAX; a["tau_mu"] = TAU_MU
        a["units"] = "Pgas Pa, T K, rho kg m-3, B T (Bx on x-faces, By on y-faces, Bz on z-faces 0:NZ top-first; *_centre = H B), z m (top-first), tau500 1"
    end
    mv(path * ".tmp", path; force = true)
    logf(@sprintf("saved %s (%s) in %.0f s", path, tag, time() - tsave))
end

function cb(acc0, row)
    acc = acc0 + acc_off
    nacc_tot[] = acc; work0[] = row.work + work_off; Δlast[] = row.Delta; μlast[] = row.mu
    td = @elapsed H.tau1_heights!(zcur, fcur, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
    td += @elapsed (s = H.tau1_change_stats(par, zcur, fcur, zprev, fprev); zs = H.tau1_summary(par, zcur, fcur))
    below = s.all_valid && s.dz_max < DZ_THRESH
    consec[] = below ? consec[] + 1 : 0
    rms, p99 = H.layer_residual_stats(pb, w)
    (fm, f3) = cvals_face(); (cm, c3) = cvals_centre()
    dP = H.measured_change_P(pb, w, snap[]); dB = H.measured_change_B(pb, w, snap[]); snap[] = H.measured_snapshot(pb, w)
    cl = cands[]; nrej = count(c -> !c.accepted, cl[lastcand[]+1:end]); lastcand[] = length(cl)
    t = row.terms
    @printf(hist, "%d,%.1f,%.1f,%.2f,%.6e,%.6e,%.6e,%d,%d,%d,%d,%d,%d,%d,%s,%s,%d,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,%.6e,%.6e,%.6f,%.6e,%.6e,%.6e,%.3e,%.4f,%s,%.3e,%s,%d,%.3e,%.3e\n",
            acc, row.work + work_off, time() - T0, td, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, s.nnonfinite, s.neos,
            s.nprev_invalid, s.all_valid, below, consec[], zs.min, zs.median, zs.max, row.f, t..., t[1] / pb.lamF, maximum(rms), maximum(p99),
            fm, f3, cm, c3, dP, dB, row.rho, row.pred, row.Delta_used, row.Delta, row.mu, row.ynorm / row.Delta_used, string(row.cg_exit), row.cg_rel,
            row.cauchy, nrej, row.margin, row.anchor_err)
    flush(hist)
    h5open(joinpath(outdir, "ztau1", @sprintf("ztau1_%05d.h5", acc)), "w") do fo; fo["z_tau1_m"] = zcur; fo["flag"] = fcur; end
    logf(@sprintf("acc %4d work %8.0f | dz_max %.4e m (median %.3e, p99 %.3e) at (%d,%d), comparable %d, missing %d | Φ %.6e F %.3e raw force %.4e RMS %.3e p99 %.3e | face %.3fσ centre %.3fσ | ΔP/P %.2e ΔB %.2e | ρ %.2f ‖δ‖/Δ %.2f %s rej %d | consec %d",
                  acc, row.work + work_off, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, row.f, t[1], t[1] / pb.lamF,
                  maximum(rms), maximum(p99), fm, cm, dP, dB, row.rho, row.ynorm / row.Delta_used, string(row.cg_exit), nrej, consec[]))
    zprev .= zcur; fprev .= fcur
    if time() - lastck[] > 3600ckpt_h
        save_atmosphere(joinpath(outdir, ckslot[] == 0 ? "atmosphere_A.h5" : "atmosphere_B.h5"), "checkpoint at accepted $acc"); ckslot[] = 1 - ckslot[]
        lastck[] = time()
    end
    if consec[] >= NCONSEC
        stopwhy[] = "user criterion met: dz_max < $(DZ_THRESH) m with all columns valid on $NCONSEC consecutive accepted iterations"; return :stop
    end
    if time() - T0 > 3600wall_h
        stopwhy[] = "wall cap $(wall_h) h"; return :stop
    end
    return nothing
end

logf(@sprintf("run: wall cap %.1f h, checkpoints every %.1f h, threshold dz_max < %.0e m on %d consecutive accepted iterations, block_scale (100, 1, 1), no face bounds",
              wall_h, ckpt_h, DZ_THRESH, NCONSEC))
H.reset_work!()
st, why, n, res = H.joint_gn_solve!(pb, w, x, ulo; work_budget = 1e12, block_scale = (100.0, 1.0, 1.0), callback = cb, cand_sink = cands,
                                     Δ0 = Δ0, μ0 = μ0, verbose = false)
H.objective!(pb, w, x)
logf("solver returned: $(st) ($(why))$(isempty(stopwhy[]) ? "" : ": " * stopwhy[]); accepted $n / $(res.counts.cand) candidates, Cauchy $(res.counts.cauchy), work $(round(H.work_equivalent()))")
Δlast[] = res.Δ; μlast[] = res.μ; work0[] = H.work_equivalent() + work_off
save_atmosphere(joinpath(outdir, "atmosphere_final.h5"), "final: $(st) ($(why)) $(stopwhy[])")
close(hist)
logf("done")
