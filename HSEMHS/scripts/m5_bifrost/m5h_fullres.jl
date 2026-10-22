# M5h: full-input MHS progress test (REVIEW_M5g_2026-10-21.md; user request 21 Oct 2026).
# The complete IO/input-mhs-hse.nc: 504 × 504 × 177 cells, every column and level (bif_problem(1, 1): no averaging, no
# subsampling, no crop). Current code, current scientific objective (staggered MHS, D9b defaults, λ_F = 1e-4, λ_Q = 1e-3,
# σ_B synthetic 5 % ⊕ 5 G, data on faces, bottom anchor P_ref from the file, EOS table bif_setup.TAB, which is cache/eos_table_fullz.h5 from 22 Oct 2026; run 2 used cache/eos_table.h5), solver joint
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
# the threshold itself is unchanged); (ii) wall cap; (iii) solver radius collapse; (iv) OUTDIR/STOP exists (checked after
# every candidate: save, then stop); (v) an external wall guard.
# Robustness (after the first attempt crashed in its first checkpoint, 21 Oct 2026):
#   - the initialized state is saved (OUTDIR/init.h5) before any iteration; `run … RESTART.h5` resumes from any saved file
#     (init.h5, a checkpoint or the final file) without repeating the initialization;
#   - checkpoints every `ckpt_hours`, also after rejected candidates, alternating atmosphere_A/B.h5 (written to .tmp, then
#     renamed); any error during the solve writes atmosphere_crash.h5 before re-raising;
#   - every candidate (accepted or rejected) is logged to OUTDIR/candidates.csv; init_A! prints its residual every 100 iterations;
#   - `smoke` mode runs every path on the coarse 42 × 42 × 89 grid in minutes (setup, init save, iterations, a forced
#     checkpoint, the STOP file, a restart with the z(τ = 1) continuity check).
#
# usage (from HSEMHS/): julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR run [wall_hours=24] [ckpt_hours=1] [init_iter=3000] [RESTART.h5]
#                        julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR smoke [wall_hours] [ckpt_hours] [init_iter] [RESTART.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
using Dates

const TAU_TOP = :exp_tail; const TAU_HMAX = 5.0e6; const TAU_MU = 1.0; const DZ_THRESH = 1e-3; const NCONSEC = 3
const T0 = time()
logf(s) = (println(Dates.format(now(), "HH:MM:SS"), " [", round(Int, time() - T0), " s] ", s); flush(stdout))

"""Owned part of a Field as a plain Array (halo width 1; every z level of the field)."""
own(f, b) = parent(f)[2:b.nx+1, 2:b.ny+1, :]

mutable struct RunState
    acc::Int; work::Float64; Δ::Float64; μ::Float64
    consec::Int; lastck::Float64; ckslot::Int; stopwhy::String; ncand_logged::Int
end

function save_atmosphere(path, tag, pb, w, x, ulo, rs, zbuf, fbuf, taud, gitrev, wall_h, ckpt_h)
    tsave = time(); par = pb.par; b = par.block; NZ = b.nz
    H.tau1_heights!(zbuf, fbuf, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU, tau = taud)
    h5open(path * ".tmp", "w") do fo
        fo["coords/x_m"] = [pb.grid.x.s[i] for i in 1:b.nx]
        fo["coords/y_m"] = [pb.grid.y.s[j] for j in 1:b.ny]
        fo["coords/z_m"] = collect(pb.grid.z.s); fo["coords/zf_m"] = collect(pb.s.zf)
        fo["atmosphere/Pgas"] = own(w.P, b); fo["atmosphere/T"] = exp.(own(pb.X, b)); fo["atmosphere/rho"] = own(w.rho, b)
        fo["atmosphere/Bx_xface"] = own(w.B[1], b); fo["atmosphere/By_yface"] = own(w.B[2], b); fo["atmosphere/Bz_zface"] = own(w.B[3], b)
        for q in 1:3; fo["atmosphere/B$("xyz"[q])_centre"] = own(w.HB[q], b); end
        fo["atmosphere/tau500"] = own(taud, b); fo["atmosphere/z_tau1_m"] = copy(zbuf); fo["atmosphere/z_tau1_flag"] = copy(fbuf)
        for q in 1:5; fo["solver/x$q"] = copy(parent(x[q])); end
        fo["solver/ulo"] = copy(parent(ulo))
        a = attributes(fo)
        a["tag"] = tag; a["git"] = gitrev; a["input"] = BIF; a["eos_table"] = TAB; a["grid"] = [par.NX, par.NY, NZ]
        a["accepted"] = rs.acc; a["work"] = rs.work; a["Delta"] = rs.Δ; a["mu"] = rs.μ
        a["lambda_F"] = pb.lamF; a["lambda_Q"] = pb.lamQ; a["lambda_S"] = pb.lamS; a["block_scale"] = [100.0, 1.0, 1.0]
        a["P_ref_Pa"] = exp(pb.lnPref); a["anchor"] = "bottom"; a["g"] = pb.g; a["face_bounds"] = "none"
        a["tau_top"] = string(TAU_TOP); a["tau_H_max_m"] = TAU_HMAX; a["tau_mu"] = TAU_MU; a["wall_hours"] = wall_h; a["ckpt_hours"] = ckpt_h
        a["units"] = "Pgas Pa, T K, rho kg m-3, B T (Bx on x-faces, By on y-faces, Bz on z-faces 0:NZ top-first; *_centre = H B), z m (top-first), tau500 1"
    end
    mv(path * ".tmp", path; force = true)
    logf(@sprintf("saved %s (%s) in %.0f s", path, tag, time() - tsave))
end

function main()
    outdir, mode = ARGS[1], ARGS[2]
    mode in ("run", "smoke") || error("mode must be run or smoke")
    mkpath(outdir); mkpath(joinpath(outdir, "ztau1"))
    gitrev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "unknown" end
    wall_h = parse(Float64, get(ARGS, 3, "24")); ckpt_h = parse(Float64, get(ARGS, 4, "1"))
    init_iter = parse(Int, get(ARGS, 5, "3000")); restart = get(ARGS, 6, "")
    logf("M5h MHS (git $gitrev), mode $mode, $(Threads.nthreads()) threads, input $BIF, wall $(wall_h) h, checkpoints every $(ckpt_h) h" *
         (isempty(restart) ? "" : ", restart from $restart"))
    c, sz = mode == "run" ? (1, 1) : (12, 2)
    pb, x, par, b, NZ = bif_problem(c, sz; init_iter = isempty(restart) ? init_iter : 0, init_progress = 100, lambda_F = 1e-4, lambda_Q = 1e-3)
    MPI.Comm_size(par.comm) == 1 || error("m5h_fullres.jl writes rank-local arrays: run it on one rank (threads only)")
    mode == "run" && ((par.NX, par.NY, NZ) == (504, 504, 177) || error("expected the full 504 × 504 × 177 grid, got $((par.NX, par.NY, NZ))"))
    rs = RunState(0, 0.0, NaN, 1e-6, 0, time(), 0, "", 0)
    ulo = H.newfield(par)
    if !isempty(restart)
        h5open(restart) do f
            for q in 1:5; copyto!(parent(x[q]), read(f["solver/x$q"])); end
            copyto!(parent(ulo), read(f["solver/ulo"]))
            a = attributes(f); rs.acc = read(a["accepted"]); rs.work = read(a["work"]); rs.Δ = read(a["Delta"]); rs.μ = read(a["mu"])
            read(a["grid"]) == [par.NX, par.NY, NZ] || error("restart grid $(read(a["grid"])) differs from $((par.NX, par.NY, NZ))")
        end
        H.exchange_halos!(par, collect(x)); H.exchange_halos!(par, [ulo])
    end
    logf(@sprintf("setup done: %d × %d × %d, P_ref %.6g Pa (bottom anchor), g %.3f, maxrss %.1f GB", par.NX, par.NY, NZ, exp(pb.lnPref), pb.g, Sys.maxrss() / 2^30))
    w = H.SMHSWork(par)
    f0, t0, oob0 = H.objective!(pb, w, x)
    oob0 && error("initial state outside the EOS envelope")
    isempty(restart) && H.owned_map!((i, j, k) -> w.u[i, j, k] - 3.0, ulo, par)   # pressure floor: initial ln P − 3 (as in the M5 runs)
    zprev = fill(NaN, b.nx, b.ny); fprev = zeros(UInt8, b.nx, b.ny); zcur = similar(zprev); fcur = similar(fprev); taud = H.newfield(par)
    td = @elapsed H.tau1_heights!(zprev, fprev, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
    if !isempty(restart)
        h5open(restart) do f
            (isequal(read(f["atmosphere/z_tau1_m"]), zprev) && read(f["atmosphere/z_tau1_flag"]) == fprev) ||
                error("restart: recomputed z(τ = 1) differs from the saved map")
        end
        logf("restart: state, ulo, Δ = $(rs.Δ), μ = $(rs.μ), accepted $(rs.acc), work $(rs.work); z(τ = 1) continuity check passed")
    end
    s0 = H.tau1_change_stats(par, zprev, fprev, nothing, nothing); zs0 = H.tau1_summary(par, zprev, fprev)
    tdd = @elapsed (dstats = H.data_residual_stats(pb, w); rstats = H.layer_residual_stats(pb, w))
    fm, f3, cm, c3 = dstats; rms, p99 = rstats
    logf(@sprintf("initial: Φ %.6e terms %s | z(τ=1) min/median/max %.1f / %.1f / %.1f m, valid %d, missing %d, nonfinite %d, EOS %d | face %.3fσ centre %.3fσ | RMS %.3e | τ diagnostic %.2f s, data/force diagnostics %.2f s",
                  f0, string(round.(t0; sigdigits = 5)), zs0.min, zs0.median, zs0.max, zs0.nvalid, s0.nmissing, s0.nnonfinite, s0.neos, fm, cm, maximum(rms), td, tdd))
    isempty(restart) && save_atmosphere(joinpath(outdir, "init.h5"), "initialized state (no iteration)", pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h)
    hpath = joinpath(outdir, "history.csv"); cpath = joinpath(outdir, "candidates.csv")
    newh = !isfile(hpath)
    hist = open(hpath, "a"); candf = open(cpath, "a")
    if newh
        println(hist, "accepted,work,elapsed_s,diag_s,dz_max_m,dz_median_m,dz_p99_m,worst_i,worst_j,ncomparable,nmissing,nnonfinite,neos,nprev_invalid,all_valid,below_thresh,consec," *
                      "ztau1_min_m,ztau1_median_m,ztau1_max_m,Phi,Phi_F,Phi_NQ,Phi_B,Phi_S,Phi_U,raw_force,rms_max,p99_max,face_max,face_frac3,centre_max,centre_frac3," *
                      "dPmax,dBmax,rho,pred,Delta_used,Delta,mu,step_over_Delta,cg_exit,cg_rel,cauchy,margin_u,anchor_err")
        @printf(hist, "%d,%.1f,%.1f,%.2f,NaN,NaN,NaN,0,0,0,%d,%d,%d,0,false,false,0,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,,NaN,false,NaN,NaN\n",
                rs.acc, rs.work, time() - T0, td, s0.nmissing, s0.nnonfinite, s0.neos, zs0.min, zs0.median, zs0.max, f0, t0..., t0[1] / pb.lamF, maximum(rms), maximum(p99), fm, f3, cm, c3)
        println(candf, "candidate,accepted_total,accepted,elapsed_s,work,rho,pred,actual,Delta,ynorm,cg_exit,cg_rel,cauchy")
        h5open(joinpath(outdir, "ztau1", @sprintf("ztau1_%05d.h5", rs.acc)), "w") do fo; fo["z_tau1_m"] = zprev; fo["flag"] = fprev; end
    end
    flush(hist); flush(candf)
    snap = Ref(H.measured_snapshot(pb, w)); cands = Ref{Vector{NamedTuple}}(NamedTuple[])
    acc0 = rs.acc; work0 = rs.work
    stopfile = joinpath(outdir, "STOP")
    function log_new_candidates()
        cl = cands[]
        for q in rs.ncand_logged+1:length(cl)
            cd = cl[q]
            @printf(candf, "%d,%d,%s,%.1f,%.1f,%.6e,%.6e,%.6e,%.6e,%.6e,%s,%.3e,%s\n", q, rs.acc, cd.accepted, time() - T0, H.work_equivalent() + work0,
                    cd.rho, cd.pred, cd.actual, cd.Delta, cd.ynorm, string(cd.cg_exit), cd.cg_rel, cd.cauchy)
        end
        rs.ncand_logged = length(cl); flush(candf)
    end
    ckname() = joinpath(outdir, rs.ckslot == 0 ? "atmosphere_A.h5" : "atmosphere_B.h5")
    function maybe_checkpoint(tag; force = false)
        if force || time() - rs.lastck > 3600ckpt_h
            save_atmosphere(ckname(), tag, pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h); rs.ckslot = 1 - rs.ckslot; rs.lastck = time()
        end
    end
    function common_stops()
        if isfile(stopfile)
            rs.stopwhy = "STOP file"; maybe_checkpoint("STOP file at accepted $(rs.acc)"; force = true); return :stop
        end
        if time() - T0 > 3600wall_h
            rs.stopwhy = "wall cap $(wall_h) h"; return :stop
        end
        return nothing
    end
    function cb(a, row)
        rs.acc = acc0 + a; rs.work = row.work + work0; rs.Δ = row.Delta; rs.μ = row.mu
        log_new_candidates()
        td = @elapsed H.tau1_heights!(zcur, fcur, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
        td += @elapsed (s = H.tau1_change_stats(par, zcur, fcur, zprev, fprev); zs = H.tau1_summary(par, zcur, fcur))
        below = s.all_valid && s.dz_max < DZ_THRESH
        rs.consec = below ? rs.consec + 1 : 0
        tdd = @elapsed begin
            rms, p99 = H.layer_residual_stats(pb, w); fm, f3, cm, c3 = H.data_residual_stats(pb, w)
            dP = H.measured_change_P(pb, w, snap[]); dB = H.measured_change_B(pb, w, snap[]); snap[] = H.measured_snapshot(pb, w)
        end
        t = row.terms
        @printf(hist, "%d,%.1f,%.1f,%.2f,%.6e,%.6e,%.6e,%d,%d,%d,%d,%d,%d,%d,%s,%s,%d,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,%.6e,%.6e,%.6f,%.6e,%.6e,%.6e,%.3e,%.4f,%s,%.3e,%s,%.3e,%.3e\n",
                rs.acc, rs.work, time() - T0, td, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, s.nnonfinite, s.neos,
                s.nprev_invalid, s.all_valid, below, rs.consec, zs.min, zs.median, zs.max, row.f, t..., t[1] / pb.lamF, maximum(rms), maximum(p99),
                fm, f3, cm, c3, dP, dB, row.rho, row.pred, row.Delta_used, row.Delta, row.mu, row.ynorm / row.Delta_used, string(row.cg_exit), row.cg_rel,
                row.cauchy, row.margin, row.anchor_err)
        flush(hist)
        h5open(joinpath(outdir, "ztau1", @sprintf("ztau1_%05d.h5", rs.acc)), "w") do fo; fo["z_tau1_m"] = zcur; fo["flag"] = fcur; end
        logf(@sprintf("acc %4d work %8.0f | dz_max %.4e m (median %.3e, p99 %.3e) at (%d,%d), comparable %d, missing %d | Φ %.6e raw force %.4e RMS %.3e p99 %.3e | face %.3fσ centre %.3fσ | ΔP/P %.2e ΔB %.2e | ρ %.2f ‖δ‖/Δ %.2f %s | consec %d | diag τ %.1f s, other %.1f s",
                      rs.acc, rs.work, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, row.f, t[1] / pb.lamF,
                      maximum(rms), maximum(p99), fm, cm, dP, dB, row.rho, row.ynorm / row.Delta_used, string(row.cg_exit), rs.consec, td, tdd))
        zprev .= zcur; fprev .= fcur
        maybe_checkpoint("checkpoint at accepted $(rs.acc)"; force = mode == "smoke" && rs.acc == acc0 + 1)
        if rs.consec >= NCONSEC
            rs.stopwhy = "user criterion met: dz_max < $(DZ_THRESH) m with all columns valid on $NCONSEC consecutive accepted iterations"; return :stop
        end
        mode == "smoke" && rs.acc >= acc0 + 3 && (rs.stopwhy = "smoke: 3 accepted iterations"; return :stop)
        return common_stops()
    end
    function ccb(rec)
        log_new_candidates()
        logf(@sprintf("candidate %d REJECTED: ρ %.3g pred %.3e actual %.3e Δ %.3e ‖δ‖ %.3e CG %s", rs.ncand_logged, rec.rho, rec.pred, rec.actual, rec.Delta, rec.ynorm, string(rec.cg_exit)))
        maybe_checkpoint("checkpoint at accepted $(rs.acc) (after a rejected candidate)")
        return common_stops()
    end
    logf(@sprintf("run: threshold dz_max < %.0e m on %d consecutive accepted iterations, block_scale (100, 1, 1), no face bounds, Δ0 %s, μ0 %.3g",
                  DZ_THRESH, NCONSEC, isnan(rs.Δ) ? "default rule" : string(rs.Δ), rs.μ))
    H.reset_work!()
    st, why, n, res = try
        H.joint_gn_solve!(pb, w, x, ulo; work_budget = 1e12, block_scale = (100.0, 1.0, 1.0), callback = cb, cand_sink = cands,
                          cand_callback = ccb, Δ0 = rs.Δ, μ0 = rs.μ, verbose = false)
    catch e
        logf("ERROR during the solve: " * sprint(showerror, e))
        try
            H.objective!(pb, w, x)
            save_atmosphere(joinpath(outdir, "atmosphere_crash.h5"), "after an error at accepted $(rs.acc)", pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h)
        catch e2
            logf("emergency save failed: " * sprint(showerror, e2))
        end
        rethrow()
    end
    log_new_candidates()
    H.objective!(pb, w, x)
    rs.Δ = res.Δ; rs.μ = res.μ; rs.work = H.work_equivalent() + work0
    logf("solver returned: $(st) ($(why))$(isempty(rs.stopwhy) ? "" : ": " * rs.stopwhy); accepted $(rs.acc) in total ($n this segment) / $(res.counts.cand) candidates, Cauchy $(res.counts.cauchy), work $(round(rs.work))")
    save_atmosphere(joinpath(outdir, "atmosphere_final.h5"), "final: $(st) ($(why)) $(rs.stopwhy)", pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h)
    close(hist); close(candf)
    logf("done")
end

main()
