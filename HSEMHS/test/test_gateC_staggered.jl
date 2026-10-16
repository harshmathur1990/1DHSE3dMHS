# M6c: Gate C1–C3 on the staggered scheme (D9 addendum §10; E3 withdrawn: RMS ≤ 1e-6, p99 ≤ 1e-4).

"""Staggered C1 solve (plain joint L-BFGS from column_hse u and A₀)."""
function sc1_solve(NZ; Bobs = (0.0, 0.0, 0.0), NX = 8, NY = 8, max_iter = 5000)
    par = H.init_parallel(NX, NY, NZ)
    pb, x, us, uc, grid = H.sc1_problem(par, NX, NY, NZ; Bobs = Bobs)
    w = H.SMHSWork(par)
    r = H.lbfgs!(pb, w, x; opts = H.LBFGSOptions(max_iter = max_iter))
    H.objective!(pb, w, x)
    rms, p99 = H.layer_residual_stats(pb, w)
    du_s = H.newfield(par); H.fill_owned!((i, j, k) -> w.u[i, j, k] - us[i, j, k], du_s, par)
    du_c = H.newfield(par); H.fill_owned!((i, j, k) -> w.u[i, j, k] - uc[i, j, k], du_c, par)
    Bex = (H.newfield(par; fill = Bobs[1]), H.newfield(par; fill = Bobs[2]), H.newfield0(par; fill = Bobs[3]))
    dB = maximum(H.rms_diff(par, w.B[c], Bex[c]) for c in 1:3) / pb.sig[1][1, 1, 1]
    return (r = r, rms = maximum(rms), p99 = maximum(p99), dus = H.maxabs(par, du_s), duc = H.maxabs(par, du_c),
            dB = dB, pb = pb, w = w, x = x, us = us, Bex = Bex, par = par)
end

@testset "Gate C1a (staggered): uniform T(z), B = 0, no floor" begin
    res = [sc1_solve(nz) for nz in (33, 65, 129)]
    for s in res
        @test s.r.status == H.converged
        @test s.rms <= 1e-6 && s.p99 <= 1e-4
        @test s.dus <= 1e-6                                         # reaches the exact discrete equilibrium
    end
    orders = [log2(res[i].duc / res[i+1].duc) for i in 1:2]
    @info "C1a staggered" rms = [s.rms for s in res] p99 = [s.p99 for s in res] dus = [s.dus for s in res] duc = [s.duc for s in res] orders iters = [s.r.iterations for s in res]
    @test orders[end] >= 1.8                                        # u − u_colhse = O(Δz²)
end

@testset "Gate C1b (staggered): uniform B" begin
    s = sc1_solve(65; Bobs = (20e-4, 0.0, 50e-4))
    @info "C1b staggered" status = s.r.status iters = s.r.iterations s.rms s.p99 s.dus s.dB
    @test s.r.status == H.converged
    @test s.rms <= 1e-6 && s.p99 <= 1e-4
    @test s.dB <= 1e-3                                              # B within 1e-3 σ of Bobs
end

@testset "Gate C2 (staggered): potential field from an analytic A" begin
    eI = Float64[]; eE = Float64[]; dv = Float64[]
    for N in (32, 64, 128)
        par = H.init_parallel(N, N, N + 1)
        grid, x, y, z = H.synthetic_grid(par, N, N, N + 1)
        s = H.StagOps(grid, par); b = par.block; NZ = b.nz
        A = (H.newfield0(par), H.newfield0(par), H.newfield(par))
        B = (H.newfield(par), H.newfield(par), H.newfield0(par))
        J = (H.newfield0(par), H.newfield0(par), H.newfield(par))
        kk = H.potential_A!(A, par, s, x, y)
        H.curlA!(B, A, 0.0, s, par); H.exchange_halos!(par, collect(B))
        H.curlB!(J, B, s, par)
        for c in 1:3
            Jc = J[c]; H.owned_map!((i, j, k) -> Jc[i, j, k] / H.MU0, Jc, par)
        end
        H.exchange_halos!(par, collect(J))
        sc = 0.01^2 * kk / H.MU0
        li = 0.0; le = 0.0
        for k in 1:NZ, j in 1:b.ny, i in 1:b.nx
            l = max(abs(H.lor_x(J, B, i, j, k)[1]), abs(H.lor_y(J, B, i, j, k)[1]), k <= NZ - 1 ? abs(H.lor_z(J, B, i, j, k)[1]) : 0.0)
            (k == 1 || k == NZ) ? (le = max(le, l)) : (li = max(li, l))
        end
        push!(eI, H.global_max(par, li) / sc); push!(eE, H.global_max(par, le) / sc)
        push!(dv, H.global_max(par, maximum(abs(H.div_face(B, s, i, j, k)) for k in 1:NZ, j in 1:b.ny, i in 1:b.nx)) / (0.01 * kk))
    end
    oI = [log2(eI[i] / eI[i+1]) for i in 1:2]
    @info "C2 staggered" eI eE dv oI
    @test all(dv .<= 1e-13)                                         # ∇·B = 0 exactly
    @test all(eI[1:end-1] .> eI[2:end])
    # interior layers: order 1.36 → 1.69 at 128; the spec gate (≥ 1.9) is not reached here
    @test_broken oI[end] >= 1.9
    # top/bottom layers: horizontal faces use the one-sided boundary-edge J (addendum §4 line
    # "boundary edges ... only enter boundary faces" does not hold): L does not converge
    @test_broken eE[end] < 0.5 * eE[1]
end

@testset "Gate C3 (staggered): checkerboard modes" begin
    s = sc1_solve(33; Bobs = (20e-4, 0.0, 50e-4), NX = 16, NY = 16)
    pb, par, w = s.pb, s.par, s.w; b = par.block
    r = H.lbfgs!(pb, w, s.x; opts = H.LBFGSOptions(max_iter = 5000, gtol_rel = 1e-10))   # tighten the reference
    xeq = deepcopy(s.x)
    dist() = (H.rms_diff(par, w.u, s.us), maximum(H.rms_diff(par, w.B[c], s.Bex[c]) for c in 1:3))
    function run(ε, fields)
        xp = deepcopy(xeq)
        for f in fields, k in axes(xp[f], 3), j in 1:b.ny, i in 1:b.nx
            xp[f][i, j, k] += ε * (iseven(b.i0 + i + b.j0 + j + k + f) ? 1.0 : -1.0)
        end
        fe = H.objective!(pb, w, xeq)[1]; fp = H.objective!(pb, w, xp)[1]; d0 = dist()
        rr = H.lbfgs!(pb, w, xp; opts = H.LBFGSOptions(max_iter = 5000, gtol_rel = 1e-10))
        H.objective!(pb, w, xp)
        return fp > fe, d0, dist(), rr
    end
    seen, d0, d1, rr = run(1e-3, 1:1)                               # pressure checkerboard
    @info "C3 staggered, u checkerboard" d0 d1 status = rr.status
    @test seen
    @test d1[1] <= 1e-3 * d0[1]
    seen, d0, d1, rr = run(1e-5, 2:4)                               # Ã checkerboard (linear regime)
    @info "C3 staggered, Ã checkerboard" d0 d1 status = rr.status
    @test seen
    # B decays only ÷8.5 (Hessian Rayleigh quotient 1.5e-5 along the remaining error vs 3e2 for
    # smooth Ã); see reports/M6c_gate.md
    @test_broken d1[2] <= 1e-3 * d0[2]
end
