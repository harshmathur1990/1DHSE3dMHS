# Gate C1–C3 (spec §6 Gate C, implementation doc §14; E3 floor-relative tolerance).

"""Solve a C1-type problem; return (status, RMS/floor, p99/floor, max|u−uex|, max|b̃|, w, x, pb)."""
function c1_solve(NZ; Bobs = (0.0, 0.0, 0.0), NX = 8, NY = 8, max_iter = 20000)
    par = H.init_parallel(NX, NY, NZ)
    pb, x, uex, grid = H.c1_problem(par, NX, NY, NZ; Bobs = Bobs)
    fr, fp = H.residual_floor(pb, uex)
    w = H.MHSWork(par)
    r = H.lbfgs!(pb, w, x; opts = H.LBFGSOptions(max_iter = max_iter))
    rms, p99 = H.layer_residual_stats(pb, w)
    du = H.newfield(par); H.fill_owned!((i, j, k) -> w.u[i, j, k] - uex[i, j, k], du, par)
    tolr = max.(1e-6, 1.5 .* fr); tolp = max.(1e-4, 1.5 .* fp)
    return (r = r, okrms = all(rms .<= tolr), okp99 = all(p99 .<= tolp), du = H.maxabs(par, du),
            bt = maximum(H.maxabs(par, x[c]) for c in 2:4), w = w, x = x, pb = pb, par = par, floor = maximum(fr))
end

@testset "Gate C1a: uniform T(z), B = 0 reproduces column_hse" begin
    res = [c1_solve(nz) for nz in (33, 65, 129)]
    for s in res
        @test s.r.status in (H.converged, H.stalled)
        @test s.okrms && s.okp99                                   # E3: ≤ max(1e-6, 1.5 floor)
        @test s.bt <= 1e-3                                          # B_out within 1e-3 σ of 0
    end
    orders = [log2(res[i].du / res[i+1].du) for i in 1:2]
    @info "C1a" du = [s.du for s in res] floor = [s.floor for s in res] orders
    @test orders[end] >= 1.8
    # floor itself converges at second order
    @test log2(res[2].floor / res[3].floor) >= 1.9
end

@testset "Gate C1b: uniform B, L = 0 to rounding" begin
    Bo = (20e-4, 0.0, 50e-4)
    s = c1_solve(65; Bobs = Bo, max_iter = 3000)    # B adds slow modes; the floor rule is met well before
    @test s.okrms && s.okp99
    # discrete Lorentz force of a uniform field is zero to rounding
    par = s.par; o = s.pb.ops; b = par.block
    B = ntuple(c -> H.newfield(par; fill = Bo[c]), 3)
    H.exchange_halos!(par, collect(B))
    Lmax = maximum(maximum(abs.(H.lorentz(B..., o, i, j, k))) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
    @test Lmax <= 1e-12 * sum(abs2, Bo) / H.MU0 / 1e3
    @info "C1b" status = s.r.status maxbt = s.bt du = s.du
end

@testset "Gate C2: potential field gives L = 0 and div B = 0 to truncation order" begin
    eL = Float64[]; eD = Float64[]
    # e^{k z} varies by e^7 over the box: 16–64 are pre-asymptotic (orders 1.2–1.6), so the
    # order is measured on 64 → 128 → 256 and gated on the finest pair
    for N in (64, 128, 256)
        par = H.init_parallel(N, N, N + 1)
        grid, x, y, z = H.synthetic_grid(par, N, N, N + 1)
        o = H.Ops(grid, par); b = par.block
        B = ntuple(_ -> H.newfield(par), 3)
        kk = H.potential_field!(B, par, x, y, z)
        B0 = 0.01
        push!(eL, maximum(maximum(abs.(H.lorentz(B..., o, i, j, k))) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx) / (B0^2 * kk / H.MU0))
        push!(eD, maximum(abs(H.divergence(B..., o, i, j, k)) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx) / (B0 * kk))
    end
    @info "C2" eL eD
    @test log2(eL[end-1] / eL[end]) >= 1.85
    @test log2(eD[end-1] / eD[end]) >= 1.9
    @test all(eL[1:end-1] .> eL[2:end]) && all(eD[1:end-1] .> eD[2:end])
end

@testset "Gate C3: checkerboard modes are penalized and decay" begin
    Bo = (20e-4, 0.0, 50e-4)
    s = c1_solve(33; Bobs = Bo, NX = 16, NY = 16, max_iter = 3000)
    pb, par, w = s.pb, s.par, s.w
    b = par.block
    xeq = H.newstate(par); H.copystate!(xeq, s.x, par)
    ueq = H.newfield(par); H.fill_owned!((i, j, k) -> w.u[i, j, k], ueq, par)
    f_eq, t_eq, _ = H.objective!(pb, w, xeq)
    ε = 1e-3
    xp = H.newstate(par); H.copystate!(xp, xeq, par)
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        sgn = iseven(b.i0 + i + b.j0 + j + k) ? 1.0 : -1.0
        xp[1][i, j, k] += ε * sgn
        for c in 2:4
            xp[c][i, j, k] += ε * sgn                             # δB_c = ε σ_c (−1)^{i+j+k}
        end
    end
    f_p, t_p, _ = H.objective!(pb, w, xp)
    @test f_p > f_eq                                              # (a) the objective sees the mode
    @test t_p[4] + t_p[5] > t_eq[4] + t_eq[5]                     #     ... through the regularizers
    a_u0 = abs(H.checker_amp(par, w.u, ueq)); a_b0 = abs(H.checker_amp(par, xp[4], xeq[4]))
    # B-first continuation (reports/M4a_c3_remedies.md): without it the joint solve drifts into a
    # collapsed-pressure trap; with it the pressure mode decays, but the B checkerboard (a null
    # mode of the collocated curl) does not, so (b) still fails for b̃
    r1, r = H.lbfgs_continuation!(pb, w, xp; b_first_iter = 500, opts = H.LBFGSOptions(max_iter = 1500))
    a_u = abs(H.checker_amp(par, w.u, ueq)); a_b = abs(H.checker_amp(par, xp[4], xeq[4]))
    @info "C3" a_u0 a_u a_b0 a_b status = r.status iters = r.iterations
    @test a_u <= 1e-3 * a_u0                                      # (b) decay by ≥ 10³
    @test a_b <= 1e-3 * a_b0
end
