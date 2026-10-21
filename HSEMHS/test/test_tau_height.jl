# M5h: τ₅₀₀ = 1 height diagnostic (REVIEW_M5g_2026-10-21.md §2): known profiles, missing crossings, pressure-driven and
# magnetic-only changes.

@testset "M5h: z(τ = 1) on a known exponential extinction profile" begin
    # χ = χ1 exp(−(z1 − z)/H) (increasing downward): exact τ(z) = χ(z) H with the exp-tail top closure, so τ = 1 at
    # z* = z1 − H ln(1/(χ1 H)). The trapezoid adds O(Δz²/H²).
    Hs = 1.2e5; χ1 = 1e-9; z = collect(range(2.0e6, -5.0e5; length = 2001))
    chi = χ1 .* exp.((z[1] .- z) ./ Hs)
    tau = zeros(length(z))
    m = H.tau_column!(tau, chi, z)
    @test m == H.TAU_EXP_TAIL
    zstar = z[1] - Hs * log(1 / (χ1 * Hs))
    @test abs(H.z_tau1(tau, z) - zstar) < 1.0                     # m (Δz = 1.25 km, H = 120 km)
    # μ = 0.5 doubles τ: the crossing moves up by H ln 2
    H.tau_column!(tau, chi, z; mu = 0.5)
    @test abs(H.z_tau1(tau, z) - (zstar + Hs * log(2))) < 1.0
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
