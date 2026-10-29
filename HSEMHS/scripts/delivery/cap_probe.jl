#=
Cap probe (design/CODEX_REVIEW_solver_time_round7_2026-10-29.md, the one remaining experiment): at a saved late state,
does the whole-step scaling to the cut-height / ln P caps (dc_cap 300 km, du_cap 2) waste progress?

usage: mpiexecjl -n R julia --project=. -t T scripts/delivery/cap_probe.jl INPUT.nc STATE_DIR MU AZIMUTH_DEG LOGTAU_CUT_LOS
                                                                            [mu_lm=0,1] [precond=block]
STATE_DIR: a measure_steps.jl output directory (its final64 state; same ranks, cut, input). The objective is that of the
column-HSE start (common weights, as the cold solve and measure_steps common_weights=1 use). At that state, for each
Levenberg–Marquardt damping μ of `mu_lm` (ereg × (1 + 100 μ), as solve_tau) and each inner accuracy (loose: η 0.1,
cap 200, as the switch tail; tight: 1e-3, cap 3000): one Gauss–Newton direction δ, its cap scale sc = min(1,
du_cap/max|δu|, dc_cap/max|δc|) and which cap binds, the EOS-feasible fraction, and at α ∈ {sc, 2sc, 4sc, 1} (feasible
ones only) the actual decrease, the model decrease 2α pred − α² ‖Jδ‖²/Φ and the gain ratio. A few residual evaluations;
nothing is stepped.
=#
using HSEMHS
using HDF5
using MPI
using LinearAlgebra
using Printf

const H = HSEMHS
const PT = H.PoissonTau

function main(path, sdir, mu_s, az_s, cut_s, kv...)
    H.ensure_mpi()
    o = Dict("mu_lm" => "0,1", "precond" => "block")
    for a in kv; k, s = split(a, "="); haskey(o, k) || error("unknown key $k"); o[k] = String(s); end
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = :inclined)
    cut = parse(Float64, cut_s)
    t0 = time()
    NX, NY, nt = PT.tau_input_size(path)
    par = H.slab_par(NX, NY, nt)
    logf(s) = H.rprint(par, @sprintf("[%7.1fs] %s", time() - t0, s))
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    lT = log.(T); g = 274.0; dx = x[2] - x[1]; dy = y[2] - y[1]
    lnP, z0, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed")
    kc = argmin(abs.(lt .- cut))
    _, _, _, _, _, _, i0 = PT.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, copy(lnP), copy(z0), dx, dy; logtau_cut = cut,
                                        max_steps = 0, mu = v.mu, slant = H.slant(v), stationarity = false)
    W = i0.weights
    sf = h5open(joinpath(sdir, "snap_r$(par.rank).h5"), "r")
    read(attributes(sf)["kc"]) == kc || error("state cut level differs")
    su = read(sf["final64/lnP"]); sz = read(sf["final64/z"]); close(sf)
    PT.warm_start!(lnP, z0, kc, su, sz[1, :, :])
    pr, c = PT.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z0, dx, dy; logtau_cut = cut, mu = v.mu, slant = H.slant(v),
                        balance = :vertical, precond = Symbol(o["precond"]))
    pr.wc = copy(W.wc); pr.wf = copy(W.wf); ereg0 = copy(W.ereg)
    u = copy(lnP)
    Φ, bad = PT.evaluate!(pr, u, c); bad == 0 || error("state outside the EOS table")
    n1 = pr.n1; Z() = zeros(n1, size(u, 2), size(u, 3))
    rhs = Z(); PT.JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
    logf(@sprintf("state %s (cut level %d, %d levels), Φ %.6e, precond %s", sdir, kc, n1, Φ, pr.prec))
    jx, jy, jz = Z(), Z(), Z(); ut = similar(u); ct = similar(c)
    for μs in split(o["mu_lm"], ","), (tag, tol, mit) in (("loose", 0.1, 200), ("tight", 1e-3, 3000))
        μ = parse(Float64, μs)
        pr.ereg = ereg0 .* (1 + 100μ)
        PT.evaluate!(pr, u, c); PT.setprec!(pr)
        δ = Z(); fl = Ref(:none)
        its, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, δ, rhs; tol = tol, maxit = mit, flag = fl)
        pred = H.gdot(par, δ, rhs) / Φ
        PT.Jop!(pr, jx, jy, jz, δ); jj = PT.wnorm2(pr, jx, jy, jz)
        mu_ = H.gmax(par, maximum(abs, @view δ[2:end, :, :])); mc = H.gmax(par, maximum(abs, @view δ[1, :, :]))
        rc = 3e5 / mc; ru = 2.0 / mu_; sc = min(1.0, rc, ru)
        αf = PT.feas_scale(pr, u, δ)
        rmsc = sqrt(H.gsum(par, sum(abs2, @view δ[1, :, :])) / (NX * NY))
        logf(@sprintf("μ %g, %s: CG %d its (%s, rel %.1e), pred %.3e, ‖Jδ‖²/Φ %.3e; max|δc| %.1f km (rms %.1f km), max|δu| %.3f; cap scale %.3f (%s), EOS-feasible %.3f",
                      μ, tag, its, fl[], rel, pred, jj / Φ, mc / 1e3, rmsc / 1e3, mu_, sc, sc == 1 ? "none" : rc < ru ? "height cap" : "ln P cap", αf))
        for (lab, α) in (("sc", sc), ("2sc", 2sc), ("4sc", 4sc), ("full", 1.0), ("0.5 feasible", 0.5αf))
            (α <= 1.0 && α < 0.99αf) || (logf(@sprintf("    α %-12s %.3f: skipped (beyond the full step or EOS-infeasible)", lab, α)); continue)
            ut .= u; @views ut[kc+1:end, :, :] .+= α .* δ[2:end, :, :]
            @views ct .= c .+ α .* δ[1, :, :]
            Φt, b = PT.evaluate!(pr, ut, ct)
            model = 2α * pred - α^2 * jj / Φ
            logf(b > 0 ? @sprintf("    α %-12s %.3f: %d cells outside the EOS table", lab, α, b) :
                         @sprintf("    α %-12s %.3f: ΔΦ/Φ %+.4e, model %+.4e, gain %.3f", lab, α, (Φ - Φt) / Φ, model, (Φ - Φt) / Φ / model))
        end
        PT.evaluate!(pr, u, c)
    end
end

main(ARGS...)
