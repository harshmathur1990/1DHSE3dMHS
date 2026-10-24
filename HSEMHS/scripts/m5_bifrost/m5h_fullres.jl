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
# input: bif_setup.BIF (ENV["MHS_INPUT"], default IO/input-mhs-hse.nc); every level of the chosen file is used.
# MPI (22 Oct 2026): any number of ranks (x-y decomposition, threads within a rank). Output is cooperative (no parallel
# HDF5): rank 0 writes each file's skeleton with the global datasets, then the ranks in turn write their owned (x, y)
# hyperslabs. Saved solver arrays are global owned values (no halos), so a restart may use a different rank count; files
# from the single-rank run 2 (haloed solver arrays) still restart on one rank. Stop / checkpoint decisions are made on
# rank 0 and broadcast; logs, history.csv and candidates.csv come from rank 0.
# usage (from HSEMHS/): [mpiexec -n R] julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR run [wall_hours=24] [ckpt_hours=1] [init_iter=3000] [RESTART.h5]
#                        julia --project=. -t T scripts/m5_bifrost/m5h_fullres.jl OUTDIR smoke [wall_hours] [ckpt_hours] [init_iter] [RESTART.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
using Dates

const TAU_TOP = :exp_tail; const TAU_HMAX = 5.0e6; const TAU_MU = 1.0; const DZ_THRESH = 1e-3; const NCONSEC = 3
# M5i experiment switches (ENV; the defaults reproduce M5h). MHS_SIGMA_REL: σ_B = (MHS_SIGMA_REL |Bobs|) ⊕ 5 G, 0 = uniform
# 5 G. MHS_TRUST = on | off: joint_gn_solve!(trust_region = …) (off: no trust radius, backtracking line search).
# MHS_STOP = dz | phi: the convergence variable, dz_max < DZ_THRESH (M5h) or |Φ[n−1] − Φ[n]|/Φ[n] < MHS_PHI_RTOL, on
# NCONSEC consecutive accepted iterations; z(τ = 1) is computed and logged in both cases.
const SIGMA_REL = parse(Float64, get(ENV, "MHS_SIGMA_REL", "0.05"))
const TRUST = get(ENV, "MHS_TRUST", "on") == "on"
const STOPVAR = get(ENV, "MHS_STOP", "dz")
STOPVAR in ("dz", "phi") || error("MHS_STOP must be dz or phi")
const PHI_RTOL = parse(Float64, get(ENV, "MHS_PHI_RTOL", "1e-6"))
const T0 = time()
const R0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0
stamp(s) = string(Dates.format(now(), "HH:MM:SS"), " [", round(Int, time() - T0), " s] ", s)
logf(s) = R0 && (println(stamp(s)); flush(stdout))

"""Owned part of a Field as a plain Array (halo width 1; every z level of the field)."""
own(f, b) = parent(f)[2:b.nx+1, 2:b.ny+1, :]
owned_ranges(b) = (b.i0:b.i0+b.nx-1, b.j0:b.j0+b.ny-1)
"""Rank 0's value of `v` on every rank (stop / checkpoint decisions must agree across ranks)."""
decide(par, v::Bool) = MPI.bcast(v, par.comm; root = 0)

"""
    coop_write(par, path, dist, small, setattrs!)

Cooperative HDF5 output without parallel HDF5. Rank 0 creates `path` with a global (NX, NY, …) dataset for every
`dist` entry (name => local owned array, nx × ny or nx × ny × n), writes the `small` entries whole and the attributes
(`setattrs!(attributes(file))`), and closes it; then each rank in turn reopens it and writes its owned hyperslab.
"""
function coop_write(par, path, dist, small, setattrs!)
    ir, jr = owned_ranges(par.block)
    if H.rank0(par)
        h5open(path, "w") do fo
            for (nm, a) in dist
                create_dataset(fo, nm, eltype(a), (par.NX, par.NY, size(a)[3:end]...))
            end
            for (nm, v) in small; fo[nm] = v; end
            setattrs!(attributes(fo))
        end
    end
    for r in 0:par.nranks-1
        MPI.Barrier(par.comm)
        par.rank == r || continue
        h5open(path, "r+") do fo
            for (nm, a) in dist
                ndims(a) == 2 ? (fo[nm][ir, jr] = a) : (fo[nm][ir, jr, :] = a)
            end
        end
    end
    MPI.Barrier(par.comm)
end

"""Owned hyperslab of a global saved array into a field's owned region; a single-rank run-2 file stores haloed arrays."""
function read_owned!(f, ds, par)
    b = par.block; ir, jr = owned_ranges(b)
    if size(ds, 1) == par.NX + 2
        par.nranks == 1 || error("restart file has haloed (single-rank) solver arrays: restart it on one rank")
        copyto!(parent(f), read(ds))
    else
        parent(f)[2:b.nx+1, 2:b.ny+1, :] .= ds[ir, jr, :]
    end
    return f
end

mutable struct RunState
    acc::Int; work::Float64; Δ::Float64; μ::Float64
    consec::Int; lastck::Float64; ckslot::Int; stopwhy::String; ncand_logged::Int
end

function save_atmosphere(path, tag, pb, w, x, ulo, rs, zbuf, fbuf, taud, gitrev, wall_h, ckpt_h)
    tsave = time(); par = pb.par; b = par.block; NZ = b.nz
    H.tau1_heights!(zbuf, fbuf, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU, tau = taud)
    dist = Pair{String,Any}["atmosphere/Pgas" => own(w.P, b), "atmosphere/T" => exp.(own(pb.X, b)), "atmosphere/rho" => own(w.rho, b),
                            "atmosphere/Bx_xface" => own(w.B[1], b), "atmosphere/By_yface" => own(w.B[2], b), "atmosphere/Bz_zface" => own(w.B[3], b)]
    for q in 1:3; push!(dist, "atmosphere/B$("xyz"[q])_centre" => own(w.HB[q], b)); end
    push!(dist, "atmosphere/tau500" => own(taud, b), "atmosphere/z_tau1_m" => copy(zbuf), "atmosphere/z_tau1_flag" => copy(fbuf))
    for q in 1:5; push!(dist, "solver/x$q" => own(x[q], b)); end
    push!(dist, "solver/ulo" => own(ulo, b))
    small = Pair{String,Any}["coords/x_m" => [pb.grid.x.s[i] for i in 1:par.NX], "coords/y_m" => [pb.grid.y.s[j] for j in 1:par.NY],
                             "coords/z_m" => collect(pb.grid.z.s), "coords/zf_m" => collect(pb.s.zf)]
    coop_write(par, path * ".tmp", dist, small, a -> begin
        a["tag"] = tag; a["git"] = gitrev; a["input"] = BIF; a["eos_table"] = TAB; a["grid"] = [par.NX, par.NY, NZ]
        a["accepted"] = rs.acc; a["work"] = rs.work; a["Delta"] = rs.Δ; a["mu"] = rs.μ
        a["lambda_F"] = pb.lamF; a["lambda_Q"] = pb.lamQ; a["lambda_S"] = pb.lamS; a["block_scale"] = [100.0, 1.0, 1.0]
        a["P_ref_Pa"] = exp(pb.lnPref); a["anchor"] = "bottom"; a["g"] = pb.g; a["face_bounds"] = "none"
        a["sigma_rel"] = SIGMA_REL; a["sigma_floor_T"] = 5e-4; a["trust_region"] = TRUST; a["stop_variable"] = STOPVAR; a["phi_rtol"] = PHI_RTOL
        a["tau_top"] = string(TAU_TOP); a["tau_H_max_m"] = TAU_HMAX; a["tau_mu"] = TAU_MU; a["wall_hours"] = wall_h; a["ckpt_hours"] = ckpt_h
        a["units"] = "Pgas Pa, T K, rho kg m-3, B T (Bx on x-faces, By on y-faces, Bz on z-faces 0:NZ top-first; *_centre = H B), z m (top-first), tau500 1"
        a["mpi_ranks"] = par.nranks; a["mpi_px_py"] = [par.px, par.py]; a["solver_arrays"] = "global owned values, no halos"
    end)
    H.rank0(par) && mv(path * ".tmp", path; force = true)
    MPI.Barrier(par.comm)
    logf(@sprintf("saved %s (%s) in %.0f s", path, tag, time() - tsave))
end

function main(args = ARGS)
    outdir, mode = args[1], args[2]
    mode in ("run", "smoke") || error("mode must be run or smoke")
    R0 && (mkpath(outdir); mkpath(joinpath(outdir, "ztau1")))
    MPI.Barrier(MPI.COMM_WORLD)
    gitrev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "unknown" end
    wall_h = parse(Float64, get(args, 3, "24")); ckpt_h = parse(Float64, get(args, 4, "1"))
    init_iter = parse(Int, get(args, 5, "3000")); restart = get(args, 6, "")
    logf("M5h MHS (git $gitrev), mode $mode, $(MPI.Comm_size(MPI.COMM_WORLD)) ranks × $(Threads.nthreads()) threads, input $BIF, wall $(wall_h) h, checkpoints every $(ckpt_h) h" *
         (isempty(restart) ? "" : ", restart from $restart"))
    c, sz = mode == "run" ? (1, 1) : (12, 2)
    pb, x, par, b, NZ = bif_problem(c, sz; init_iter = isempty(restart) ? init_iter : 0, init_progress = 100, lambda_F = 1e-4, lambda_Q = 1e-3, sigma_rel = SIGMA_REL)
    logf("decomposition $(par.px) × $(par.py), rank 0 block $(b.nx) × $(b.ny) × $(b.nz)")
    nzin = h5open(f -> length(read(f["z"])), BIF)
    mode == "run" && ((par.NX, par.NY, NZ) == (504, 504, nzin) || error("expected the full 504 × 504 × $nzin grid of $BIF, got $((par.NX, par.NY, NZ))"))
    rs = RunState(0, 0.0, NaN, 1e-6, 0, time(), 0, "", 0)
    ulo = H.newfield(par)
    if !isempty(restart)
        h5open(restart) do f
            for q in 1:5; read_owned!(x[q], f["solver/x$q"], par); end
            read_owned!(ulo, f["solver/ulo"], par)
            a = attributes(f); rs.acc = read(a["accepted"]); rs.work = read(a["work"]); rs.Δ = read(a["Delta"]); rs.μ = read(a["mu"])
            read(a["grid"]) == [par.NX, par.NY, NZ] || error("restart grid $(read(a["grid"])) differs from $((par.NX, par.NY, NZ))")
        end
        H.exchange_halos!(par, collect(x)); H.exchange_halos!(par, [ulo])
    end
    logf(@sprintf("setup done: %d × %d × %d, P_ref %.6g Pa (bottom anchor), g %.3f, maxrss %.1f GB", par.NX, par.NY, NZ, exp(pb.lnPref), pb.g, Sys.maxrss() / 2^30))
    w = H.SMHSWork(par)
    f0, t0, oob0 = H.objective!(pb, w, x)
    oob0 && error("initial state outside the EOS envelope")
    phiprev = Ref(f0)
    isempty(restart) && H.owned_map!((i, j, k) -> w.u[i, j, k] - 3.0, ulo, par)   # pressure floor: initial ln P − 3 (as in the M5 runs)
    zprev = fill(NaN, b.nx, b.ny); fprev = zeros(UInt8, b.nx, b.ny); zcur = similar(zprev); fcur = similar(fprev); taud = H.newfield(par)
    td = @elapsed H.tau1_heights!(zprev, fprev, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
    if !isempty(restart)
        # with a different rank count the reductions run in another order, so the recomputed map may differ by rounding:
        # flags must agree exactly and heights within 1e-6 m (1000 × below the convergence threshold)
        ir, jr = owned_ranges(b)
        zs, fs = h5open(f -> (f["atmosphere/z_tau1_m"][ir, jr], f["atmosphere/z_tau1_flag"][ir, jr]), restart)
        nflag = MPI.Allreduce(count(fs .!= fprev), +, par.comm)
        ok = fprev .== H.TAU1_OK
        dz = H.global_max(par, any(ok) ? maximum(abs.(zs[ok] .- zprev[ok])) : 0.0)
        (nflag == 0 && dz <= 1e-6) || error(@sprintf("restart: recomputed z(τ = 1) differs from the saved map (%d flags, max |Δz| %.3e m)", nflag, dz))
        logf(@sprintf("restart: state, ulo, Δ = %s, μ = %s, accepted %d, work %s; z(τ = 1) continuity check passed (max |Δz| %.3e m)",
                      rs.Δ, rs.μ, rs.acc, rs.work, dz))
    end
    s0 = H.tau1_change_stats(par, zprev, fprev, nothing, nothing); zs0 = H.tau1_summary(par, zprev, fprev)
    tdd = @elapsed (dstats = H.data_residual_stats(pb, w); rstats = H.layer_residual_stats(pb, w))
    fm, f3, cm, c3 = dstats; rms, p99 = rstats
    logf(@sprintf("initial: Φ %.6e terms %s | z(τ=1) min/median/max %.1f / %.1f / %.1f m, valid %d, missing %d, nonfinite %d, EOS %d | face %.3fσ centre %.3fσ | RMS %.3e | τ diagnostic %.2f s, data/force diagnostics %.2f s",
                  f0, string(round.(t0; sigdigits = 5)), zs0.min, zs0.median, zs0.max, zs0.nvalid, s0.nmissing, s0.nnonfinite, s0.neos, fm, cm, maximum(rms), td, tdd))
    isempty(restart) && save_atmosphere(joinpath(outdir, "init.h5"), "initialized state (no iteration)", pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h)
    hpath = joinpath(outdir, "history.csv"); cpath = joinpath(outdir, "candidates.csv")
    save_ztau1(n, z, fl) = coop_write(par, joinpath(outdir, "ztau1", @sprintf("ztau1_%05d.h5", n)),
                                      Pair{String,Any}["z_tau1_m" => z, "flag" => fl], Pair{String,Any}[], a -> nothing)
    newh = decide(par, !isfile(hpath))
    hist = R0 ? open(hpath, "a") : devnull; candf = R0 ? open(cpath, "a") : devnull
    if newh
        println(hist, "accepted,work,elapsed_s,diag_s,dz_max_m,dz_median_m,dz_p99_m,worst_i,worst_j,ncomparable,nmissing,nnonfinite,neos,nprev_invalid,all_valid,below_thresh,consec," *
                      "ztau1_min_m,ztau1_median_m,ztau1_max_m,Phi,Phi_F,Phi_NQ,Phi_B,Phi_S,Phi_U,raw_force,rms_max,p99_max,face_max,face_frac3,centre_max,centre_frac3," *
                      "dPmax,dBmax,rho,pred,Delta_used,Delta,mu,step_over_Delta,cg_exit,cg_rel,cauchy,margin_u,anchor_err,dPhi_rel,halvings")
        @printf(hist, "%d,%.1f,%.1f,%.2f,NaN,NaN,NaN,0,0,0,%d,%d,%d,0,false,false,0,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,,NaN,false,NaN,NaN,NaN,0\n",
                rs.acc, rs.work, time() - T0, td, s0.nmissing, s0.nnonfinite, s0.neos, zs0.min, zs0.median, zs0.max, f0, t0..., t0[1] / pb.lamF, maximum(rms), maximum(p99), fm, f3, cm, c3)
        println(candf, "candidate,accepted_total,accepted,elapsed_s,work,rho,pred,actual,Delta,ynorm,cg_exit,cg_rel,cauchy")
        save_ztau1(rs.acc, zprev, fprev)
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
        if force || decide(par, time() - rs.lastck > 3600ckpt_h)
            save_atmosphere(ckname(), tag, pb, w, x, ulo, rs, zcur, fcur, taud, gitrev, wall_h, ckpt_h); rs.ckslot = 1 - rs.ckslot; rs.lastck = time()
        end
    end
    function common_stops()
        if decide(par, R0 && isfile(stopfile))
            rs.stopwhy = "STOP file"; maybe_checkpoint("STOP file at accepted $(rs.acc)"; force = true); return :stop
        end
        if decide(par, time() - T0 > 3600wall_h)
            rs.stopwhy = "wall cap $(wall_h) h"; return :stop
        end
        return nothing
    end
    function cb(a, row)
        rs.acc = acc0 + a; rs.work = row.work + work0; rs.Δ = row.Delta; rs.μ = row.mu
        log_new_candidates()
        td = @elapsed H.tau1_heights!(zcur, fcur, pb, w.u; top = TAU_TOP, H_max = TAU_HMAX, mu = TAU_MU)
        td += @elapsed (s = H.tau1_change_stats(par, zcur, fcur, zprev, fprev); zs = H.tau1_summary(par, zcur, fcur))
        dphi = (phiprev[] - row.f) / row.f; phiprev[] = row.f
        below = STOPVAR == "dz" ? (s.all_valid && s.dz_max < DZ_THRESH) : abs(dphi) < PHI_RTOL
        rs.consec = below ? rs.consec + 1 : 0
        tdd = @elapsed begin
            rms, p99 = H.layer_residual_stats(pb, w); fm, f3, cm, c3 = H.data_residual_stats(pb, w)
            dP = H.measured_change_P(pb, w, snap[]); dB = H.measured_change_B(pb, w, snap[]); snap[] = H.measured_snapshot(pb, w)
        end
        t = row.terms
        @printf(hist, "%d,%.1f,%.1f,%.2f,%.6e,%.6e,%.6e,%d,%d,%d,%d,%d,%d,%d,%s,%s,%d,%.3f,%.3f,%.3f,%.8e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6f,%.6e,%.6f,%.6e,%.6e,%.6e,%.6f,%.6e,%.6e,%.6e,%.3e,%.4f,%s,%.3e,%s,%.3e,%.3e,%.6e,%d\n",
                rs.acc, rs.work, time() - T0, td, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, s.nnonfinite, s.neos,
                s.nprev_invalid, s.all_valid, below, rs.consec, zs.min, zs.median, zs.max, row.f, t..., t[1] / pb.lamF, maximum(rms), maximum(p99),
                fm, f3, cm, c3, dP, dB, row.rho, row.pred, row.Delta_used, row.Delta, row.mu, row.ynorm / row.Delta_used, string(row.cg_exit), row.cg_rel,
                row.cauchy, row.margin, row.anchor_err, dphi, row.halvings)
        flush(hist)
        save_ztau1(rs.acc, zcur, fcur)
        logf(@sprintf("acc %4d work %8.0f | dz_max %.4e m (median %.3e, p99 %.3e) at (%d,%d), comparable %d, missing %d | Φ %.6e raw force %.4e RMS %.3e p99 %.3e | face %.3fσ centre %.3fσ | ΔP/P %.2e ΔB %.2e | ΔΦ/Φ %.3e | ρ %.2f ‖δ‖/Δ %.2f %s halvings %d | consec %d | diag τ %.1f s, other %.1f s",
                      rs.acc, rs.work, s.dz_max, s.dz_median, s.dz_p99, s.worst_i, s.worst_j, s.ncomparable, s.nmissing, row.f, t[1] / pb.lamF,
                      maximum(rms), maximum(p99), fm, cm, dP, dB, dphi, row.rho, row.ynorm / row.Delta_used, string(row.cg_exit), row.halvings, rs.consec, td, tdd))
        zprev .= zcur; fprev .= fcur
        maybe_checkpoint("checkpoint at accepted $(rs.acc)"; force = mode == "smoke" && rs.acc == acc0 + 1)
        if rs.consec >= NCONSEC
            rs.stopwhy = STOPVAR == "dz" ? "user criterion met: dz_max < $(DZ_THRESH) m with all columns valid on $NCONSEC consecutive accepted iterations" :
                                           "user criterion met: |ΔΦ|/Φ < $(PHI_RTOL) on $NCONSEC consecutive accepted iterations"
            return :stop
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
    logf(@sprintf("run: stop on %s on %d consecutive accepted iterations, σ_B = %.3g |B| ⊕ 5 G, trust region %s, block_scale (100, 1, 1), no face bounds, Δ0 %s, μ0 %.3g",
                  STOPVAR == "dz" ? @sprintf("dz_max < %.0e m", DZ_THRESH) : @sprintf("|ΔΦ|/Φ < %.0e", PHI_RTOL), NCONSEC, SIGMA_REL,
                  TRUST ? "on" : "off (line search)", isnan(rs.Δ) ? "default rule" : string(rs.Δ), rs.μ))
    H.reset_work!()
    st, why, n, res = try
        H.joint_gn_solve!(pb, w, x, ulo; work_budget = 1e12, block_scale = (100.0, 1.0, 1.0), callback = cb, cand_sink = cands,
                          cand_callback = ccb, Δ0 = rs.Δ, μ0 = rs.μ, verbose = false, trust_region = TRUST)
    catch e
        println(stamp("ERROR on rank $(par.rank) during the solve: " * sprint(showerror, e))); flush(stdout)
        # the emergency save is collective: with several ranks the others may not be in the same place, so abort instead
        par.nranks > 1 && MPI.Abort(par.comm, 1)
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
    R0 && (close(hist); close(candf))
    logf("done")
end

# run directly, or through scripts/run.jl (run.mode = "full-mhs-z"), which calls main with the config's arguments
abspath(PROGRAM_FILE) == @__FILE__ && main()
