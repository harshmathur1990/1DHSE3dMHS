# M5h: τ₅₀₀ = 1 height diagnostic (REVIEW_M5g_2026-10-21.md §2): known profiles, missing crossings, pressure-driven and
# magnetic-only changes.

@testset "M5h: z(τ = 1) on a known exponential extinction profile" begin
    # χ = χ1 exp((z1 − z)/H) (increasing downward). The exp-tail closure is exact (τ_top = χ1 H); each trapezoid
    # overestimates its interval by r = (Δz/2H) coth(Δz/2H), so on this uniform grid τ(z_k) = χ1 H + r (χ(z_k) H − χ1 H)
    # exactly at the nodes. The crossing follows from χ(z) H = (1 − χ1 H)/r + χ1 H, z = z1 − H ln(χ(z)/χ1)
    # (≈ 1.085 m above the continuous τ = χH crossing here); between nodes log τ is interpolated linearly.
    Hs = 1.2e5; χ1 = 1e-9; z = collect(range(2.0e6, -5.0e5; length = 2001))
    chi = χ1 .* exp.((z[1] .- z) ./ Hs)
    tau = zeros(length(z))
    m = H.tau_column!(tau, chi, z)
    @test m == H.TAU_EXP_TAIL
    dz = z[1] - z[2]; r = (dz / (2Hs)) * coth(dz / (2Hs))
    zpred(t1) = z[1] - Hs * log(((t1 - χ1 * Hs) / r + χ1 * Hs) / (χ1 * Hs))      # τ_trap(z) = t1 crossing
    @test all(isapprox(tau[k], χ1 * Hs + r * (chi[k] * Hs - χ1 * Hs); rtol = 1e-10) for k in (1, 500, 1500, 2001))
    @test abs(H.z_tau1(tau, z) - zpred(1.0)) < 0.05
    @test abs(zpred(1.0) - (z[1] - Hs * log(1 / (χ1 * Hs)))) > 1.0           # the discretization shift is resolved
    # μ = 0.5 doubles τ: crossing where τ_trap = 0.5
    H.tau_column!(tau, chi, z; mu = 0.5)
    @test abs(H.z_tau1(tau, z) - zpred(0.5)) < 0.05
    # no crossing: τ < 1 everywhere
    H.tau_column!(tau, chi .* 1e-6, z)
    @test isnan(H.z_tau1(tau, z))
end

"""sc1 problem with P_ref high enough for τ = 1 inside the domain (IdealGasMu: χ = κ₀ ρ)."""
function tau_problem()
    par = H.init_parallel(4, 4, 9)
    pb, x, us, uc, grid = H.sc1_problem(par, 4, 4, 9; Bobs = (20e-4, 0.0, 50e-4), lambda_Q = 1e-3, P_ref = 2.0e5)
    return par, pb, x
end

@testset "M5h: tau1_heights! and change statistics — pressure-driven and magnetic-only changes, missing crossings" begin
    par, pb, x = tau_problem(); b = par.block
    w = H.SMHSWork(par); H.objective!(pb, w, x)
    z0 = fill(NaN, b.nx, b.ny); f0 = zeros(UInt8, b.nx, b.ny)
    tau = H.newfield(par)
    H.tau1_heights!(z0, f0, pb, w.u; tau = tau)
    @test all(==(H.TAU1_OK), f0)
    # matches a direct column evaluation
    i, j = 2, 3; NZ = b.nz; z = pb.grid.z.s
    chi = [exp(H.state(pb.eos, pb.X[i, j, k], w.u[i, j, k])[3]) for k in 1:NZ]
    tc = zeros(NZ); H.tau_column!(tc, chi, z)
    @test z0[i, j] == H.z_tau1(tc, z)
    @test all(tau[i, j, k] == tc[k] for k in 1:NZ)
    s0 = H.tau1_change_stats(par, z0, f0, nothing, nothing)
    @test isnan(s0.dz_max) && !s0.all_valid && s0.nmissing == 0
    # magnetic-only change (Ã, b̄): thermodynamics unchanged, heights bitwise unchanged
    xm = deepcopy(x); xm[3][2, 2, 3] += 0.5; H.setscalar!(par, xm[5], H.getscalar(par, x[5]) + 1e-3)
    wm = H.SMHSWork(par); H.objective!(pb, wm, xm)
    zm = similar(z0); fm = similar(f0); H.tau1_heights!(zm, fm, pb, wm.u)
    sm = H.tau1_change_stats(par, zm, fm, z0, f0)
    @test sm.dz_max == 0.0 && sm.all_valid && sm.ncomparable == par.NX * par.NY
    # pressure-driven change: raise ln P in one column off the anchor plane → χ and τ grow → z(τ = 1) moves up there only
    xp = deepcopy(x); for k in 1:NZ-1; xp[1][3, 1, k] += 0.2; end
    wp = H.SMHSWork(par); H.objective!(pb, wp, xp)
    zp = similar(z0); fp = similar(f0); H.tau1_heights!(zp, fp, pb, wp.u)
    sp = H.tau1_change_stats(par, zp, fp, z0, f0)
    @test sp.dz_max > 0 && (sp.worst_i, sp.worst_j) == (3, 1) && zp[3, 1] > z0[3, 1]
    @test all(zp[ii, jj] == z0[ii, jj] for jj in 1:b.ny for ii in 1:b.nx if (ii, jj) != (3, 1))
    @test sp.dz_max == abs(zp[3, 1] - z0[3, 1]) && sp.dz_median == 0.0
    # missing crossing in one column: excluded from the change, counted, and the domain does not pass
    xq = deepcopy(x); for k in 1:NZ; xq[1][1, 4, k] -= 6.0; end             # P down by ~e⁶ (anchor re-centring aside): τ < 1
    wq = H.SMHSWork(par); H.objective!(pb, wq, xq)
    zq = similar(z0); fq = similar(f0); H.tau1_heights!(zq, fq, pb, wq.u)
    sq = H.tau1_change_stats(par, zq, fq, z0, f0)
    @test fq[1, 4] == H.TAU1_NO_CROSSING && isnan(zq[1, 4])
    @test sq.nmissing == 1 && sq.ncomparable == par.NX * par.NY - 1 && !sq.all_valid
    # a non-finite pressure is reported as such, not as a missing crossing
    uq = deepcopy(w.u); uq[2, 2, 5] = NaN
    zn = similar(z0); fn = similar(f0); H.tau1_heights!(zn, fn, pb, uq)
    @test fn[2, 2] & (H.TAU1_NONFINITE | H.TAU1_EOS) != 0 && isnan(zn[2, 2])
    sm2 = H.tau1_summary(par, z0, f0)
    @test sm2.nvalid == par.NX * par.NY && sm2.min <= sm2.median <= sm2.max
end

@testset "tau-grid column HSE: a hot-over-cool level is resolved, not averaged" begin
    # 8400 K over 3000 K within one 0.1 dex level: d ln P/d ln τ rises ~1000x across it. A step sized from the slope at
    # the top alone averaged the bottom slope over the whole level (P × 1e9 in one column of a 504² MURaM run). The
    # coarse-grid result must match the same column on a 10x finer grid (T linear in ln T in between, as integrated).
    tab = H.load_table(joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5"))
    lt = collect(-6.0:0.1:1.0); nt = length(lt)
    # 6000 K above (low κ, so P at the hot level is high enough that its small slope takes the level in one step)
    lT = log.([l <= -5.05 ? 6000.0 : l <= -4.95 ? 8400.0 : 2700.0 + 3000.0 * (l + 4.9) / 5.9 for l in lt])
    fine = collect(-6.0:0.01:1.0); lTf = [H.PoissonTau.interp_lt(lT, lt, l) for l in fine]
    lnP = zeros(nt); z = zeros(nt); lnPf = zeros(length(fine)); zf = zeros(length(fine))
    @test H.PoissonTau.hse_tau!(lnP, z, lT, lt, tab, 274.0)
    @test H.PoissonTau.hse_tau!(lnPf, zf, lTf, fine, tab, 274.0)
    @test maximum(abs.(lnP .- lnPf[1:10:end])) < 0.05
    @test exp(lnP[end]) < 1e5                                      # photospheric, not ~1e9 Pa
end
