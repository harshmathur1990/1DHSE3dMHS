# M6b: staggered objective/adjoint verification (D9 addendum §8).

"""Staggered MHS test problem with structured T, smooth Bobs and a perturbed state."""
function sgrad_problem(eos; NX = 16, NY = 14, NZ = 15, lam = (1.0, 1e-3, 1e-6, 1e-2), Bscale = 1.0, apert = 0.3, seed = 3, px = 0, py = 0, dfaces = false, lq = 0.0, luz = 0.0)
    par, grid, ops, x, y, z = test_setup(NX, NY, NZ; px = px, py = py)
    b = par.block
    Lx, Ly = 4.0e6, 3.0e6
    T = H.newfield(par)
    fill_global!(T, par, x, y, z, (xx, yy, zz, gi, gj, k) ->
        5500 + 1500 * tanh((zz - 1.0e6) / 4e5) + 600 * sin(2π * xx / Lx) * cos(2π * yy / Ly) * exp(-(zz / 1e6)^2))
    Bo = ntuple(_ -> H.newfield(par), 3)
    fB = ((xx, yy, zz) -> 0.004 + 0.003 * sin(2π * xx / Lx + 0.3) * cos(2π * yy / Ly) * exp(-zz / 2e6),
          (xx, yy, zz) -> 0.002 * cos(4π * xx / Lx) * sin(2π * yy / Ly + 1.0),
          (xx, yy, zz) -> 0.01 + 0.004 * cos(2π * xx / Lx) * cos(2π * yy / Ly) * exp(-zz / 1e6))
    for c in 1:3
        fill_global!(Bo[c], par, x, y, z, (xx, yy, zz, a...) -> Bscale * fB[c](xx, yy, zz))
    end
    sig = ntuple(_ -> H.newfield(par), 3)
    for c in 1:3
        H.fill_owned!((i, j, k) -> H.sigma_synth(sqrt(sum(Bo[d][i, j, k]^2 for d in 1:3)), 0.05, 5e-4), sig[c], par)
    end
    pb = H.SMHSProblem(par, grid, T, Bo, sig, eos; P_ref = 1.0e4, lambda_F = lam[1], lambda_S = lam[2], lambda_U = lam[3], lambda_N = lam[4], data_on_faces = dfaces, lambda_Q = lq, lambda_Uz = luz)
    xs = H.init_state(pb)
    H.set_scales!(pb, xs[1])
    for f in 1:4, k in axes(xs[f], 3), j in 1:b.ny, i in 1:b.nx
        gi, gj = b.i0 + i - 1, b.j0 + j - 1
        xs[f][i, j, k] += (f == 1 ? 0.05 : apert) * grand(gi, gj, k; seed = seed + f)
    end
    return pb, xs, par
end

"""Random direction with unit global RMS per field in `fields` (field 5 = b̄: unit step)."""
function srand_dir(par, xs, fields; seed = 11)
    b = par.block
    p = H.similar_state(xs)
    for f in fields
        if f == 5
            H.setscalar!(par, p[5], 1e-3)
            continue
        end
        for k in axes(p[f], 3), j in 1:b.ny, i in 1:b.nx
            p[f][i, j, k] = grand(b.i0 + i - 1, b.j0 + j - 1, k; seed = seed + 17f)
        end
        n = par.NX * par.NY * length(axes(p[f], 3))
        r = sqrt(H.sdot(par, p[f], p[f]) / n)
        H.owned_map!((i, j, k) -> p[f][i, j, k] / r, p[f], par)
    end
    return p
end

function sfd_error(pb, xs, p)
    par = pb.par
    w = H.SMHSWork(par)
    g = H.similar_state(xs)
    H.fg!(g, pb, w, xs)
    gp = H.dotN(par, g, p)
    xt = H.similar_state(xs)
    best = Inf
    for ε in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        H.lincomb!(xt, 1.0, xs, ε, p, par); fp = H.objective!(pb, w, xt)[1]
        H.lincomb!(xt, 1.0, xs, -ε, p, par); fm = H.objective!(pb, w, xt)[1]
        best = min(best, abs((fp - fm) / (2ε) - gp) / abs(gp))
    end
    return best
end

function sgradient_checks(eos, label)
    @testset "staggered gradient: $label" begin
        pb, xs, par = sgrad_problem(eos)
        for (fields, nm) in ((1:5, "all"), (1:1, "v"), (2:2, "Ãx"), (3:3, "Ãy"), (4:4, "Ãz"), (5:5, "b̄"))
            e = sfd_error(pb, xs, srand_dir(par, xs, fields))
            @test e <= 1e-6
            rank0(par) && @info "staggered FD $label, fields $nm" e
        end
        for (lam, nm) in (((1.0, 0.0, 0.0, 0.0), "force"), ((0.0, 1e-3, 0.0, 0.0), "reg δB"),
                          ((0.0, 0.0, 1e-6, 0.0), "reg u"), ((0.0, 0.0, 0.0, 1e-2), "reg normal Nyquist"),
                          ((0.0, 0.0, 0.0, 0.0), "data"))
            pb2, xs2, par2 = sgrad_problem(eos; lam = lam)
            e = sfd_error(pb2, xs2, srand_dir(par2, xs2, 1:5))
            @test e <= 1e-6
            rank0(par2) && @info "staggered FD $label, term $nm" e
        end
        pb3, xs3, par3 = sgrad_problem(eos; dfaces = true)
        e = sfd_error(pb3, xs3, srand_dir(par3, xs3, 1:5))
        @test e <= 1e-6
        rank0(par3) && @info "staggered FD $label, data on faces" e
        # Φ_Q is scaled by λ_F: λ_F = 1e-8, λ_Q = 1e8 isolates it (the force term is then negligible)
        for (lam, lq, luz, nm) in (((1e-8, 0.0, 0.0, 0.0), 1e8, 0.0, "Φ_Q alone"), ((0.0, 0.0, 0.0, 0.0), 0.0, 1e-2, "λ_Uz alone"),
                                   ((1.0, 1e-3, 1e-6, 1e-2), 1.0, 1e-4, "all terms + Φ_Q + λ_Uz"))
            pb4, xs4, par4 = sgrad_problem(eos; lam = lam, dfaces = true, lq = lq, luz = luz)
            for (fs, fn) in ((1:5, "all fields"), (2:4, "Ã"))
                e = sfd_error(pb4, xs4, srand_dir(par4, xs4, fs))
                @test e <= 1e-6
                rank0(par4) && @info "staggered FD $label, $nm, $fn" e
            end
        end
    end
end

sgradient_checks(H.IdealGasMu(1.3), "ideal gas")
let tabf = joinpath(@__DIR__, "..", "cache", "eos_table.h5")
    isfile(tabf) ? sgradient_checks(H.load_table(tabf), "Wittmann table") : @info "table missing; skipped"
end

@testset "staggered: vertical initializer is exact discrete HSE" begin
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); Bscale = 0.0, apert = 0.0)
    x0 = H.init_state(pb)
    H.set_scales!(pb, x0[1])
    w = H.SMHSWork(par)
    f, terms, oob = H.objective!(pb, w, x0)
    @test !oob
    # T varies horizontally, so only the vertical (z-face) equations balance exactly
    s = pb.s; b = par.block; NZ = b.nz
    e = 0.0
    for k in 1:NZ-1, j in 1:b.ny, i in 1:b.nx
        rb = sqrt(w.rho[i, j, k] * w.rho[i, j, k+1])
        Rz = (w.P[i, j, k] - w.P[i, j, k+1]) / s.dzf[k] + pb.g * rb
        e = max(e, abs(Rz) / (pb.g * rb))
    end
    e = H.global_max(par, e)
    rank0(par) && @info "B = 0 initial state: max |Rz|/(g ρ̄)" e
    @test e < 1e-10
end

@testset "staggered: GN diagonal positive, A0 data fit" begin
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); apert = 0.0, lam = (1.0, 1e-3, 1e-6, 0.0))   # λ_N = 0: exact fit
    w = H.SMHSWork(par)
    x0 = H.init_state(pb)
    H.objective!(pb, w, x0)
    D = H.gn_diagonal!(H.similar_state(x0), pb, w, x0)
    b = par.block
    @test all(D[f][i, j, k] > 0 for f in 1:5 for k in axes(D[f], 3) for j in 1:b.ny for i in 1:b.nx)
    # generic smooth Bobs (not solenoidal at centres): CG converges to the least-squares floor
    f0 = H.objective!(pb, w, x0)[2][3]
    it, rel = H.init_A!(pb, w, x0; max_iter = 3000)
    f1 = H.objective!(pb, w, x0)[2][3]
    rank0(par) && @info "A0 CG, generic Bobs: Φ_B before/after, iterations, ‖r‖/‖r0‖" f0 f1 it rel
    @test rel <= 1e-10
    @test f1 < f0
    # representable Bobs = H(curl A* + b̄*): the fit is exact
    xt = H.similar_state(x0); copyto!(parent(xt[1]), parent(x0[1]))
    for f in 2:4
        fill_rand!(xt[f], par; seed = 40 + f)
    end
    H.setscalar!(par, xt[5], 0.007)
    H.objective!(pb, w, xt)
    for c in 1:3
        copyto!(parent(pb.Bobs[c]), parent(w.HB[c]))
    end
    H.exchange_halos!(par, collect(pb.Bobs))
    x1 = H.init_state(pb)
    f0 = H.objective!(pb, w, x1)[2][3]
    it, rel = H.init_A!(pb, w, x1; max_iter = 3000)
    f1 = H.objective!(pb, w, x1)[2][3]
    rank0(par) && @info "A0 CG, representable Bobs: Φ_B before/after, iterations" f0 f1 it rel
    @test f1 < 1e-7 * f0          # Φ_B ∝ error²; κ(Q) leaves ~1e-4 rms in δB at rtol 1e-10
end

@testset "staggered gradient: forward/adjoint allocations" begin
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3))
    w = H.SMHSWork(par); g = H.similar_state(xs)
    H.fg!(g, pb, w, xs); H.fg!(g, pb, w, xs)
    a = @allocated H.fg!(g, pb, w, xs)
    @info "staggered bytes allocated per objective+gradient" a
    @test a < 2_000_000
end

@testset "staggered gradient: thread and rank invariance" begin
    if get(ENV, "HSEMHS_SKIP_MPI", "0") == "1"
        @info "skipped (HSEMHS_SKIP_MPI=1)"
    else
        script = joinpath(@__DIR__, "scripts", "sgrad_ranks.jl"); proj = dirname(@__DIR__)
        dir = mktempdir()
        vals = Dict{Tuple{Int,Int},Vector{Float64}}()
        for (r, t) in ((1, 1), (1, 4), (2, 2), (4, 1))
            f = joinpath(dir, "g_$(r)_$(t).txt")
            ok = MPI.mpiexec() do exe
                success(pipeline(`$exe -n $r $(Base.julia_cmd()) --project=$proj -t $t $script $f`; stderr = stderr))
            end
            @test ok
            vals[(r, t)] = parse.(Float64, split(read(f, String)))
        end
        rank0_vals = vals[(1, 1)]
        @info "staggered Φ, ⟨∇Φ,p⟩, ‖∇Φ‖² by layout" vals
        @test vals[(1, 4)] == rank0_vals                                     # bitwise across threads
        for key in ((2, 2), (4, 1))
            @test maximum(abs.(vals[key] ./ rank0_vals .- 1)) <= 1e-12       # reduction order only
        end
    end
end

@testset "staggered: Gauss–Newton Hessian product" begin
    nrm(par, a) = sqrt(H.dotN(par, a, a))
    # quadratic terms only: H_GN is the exact Hessian (centre and face data)
    for (df, lF, lq, luz) in ((false, 0.0, 0.0, 0.0), (true, 0.0, 0.0, 0.0), (true, 1e-12, 1e12, 1e-4))   # Φ_Q ∝ λ_F λ_Q
        pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); lam = (lF, 1e-3, 1e-6, 1e-2), dfaces = df, lq = lq, luz = luz)
        w = H.SMHSWork(par); wl = H.SMHSWork(par)
        v = srand_dir(par, xs, 1:5); g1 = H.similar_state(xs); g2 = H.similar_state(xs); Hv = H.similar_state(xs); xt = H.similar_state(xs)
        H.fg!(g1, pb, w, xs); H.gn_hvp!(Hv, pb, w, wl, v)
        H.lincomb!(xt, 1.0, xs, 1e-3, v, par); H.fg!(g2, pb, H.SMHSWork(par), xt)
        H.lincomb!(g2, 1e3, g2, -1e3, g1, par); H.lincomb!(g2, 1.0, g2, -1.0, Hv, par)
        e = nrm(par, g2) / nrm(par, Hv)
        rank0(par) && @info "GN = Hessian on quadratic terms (faces = $df, λ_Q = $lq, λ_Uz = $luz)" e
        @test e <= 1e-9
    end
    # symmetry and positivity with the force term (production terms: face data, Φ_Q, λ_Uz)
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); dfaces = true, lq = 1.0, luz = 1e-4)
    w = H.SMHSWork(par); wl = H.SMHSWork(par); H.objective!(pb, w, xs)
    a = srand_dir(par, xs, 1:5; seed = 3); c = srand_dir(par, xs, 1:5; seed = 9)
    Ha = H.similar_state(xs); Hc = H.similar_state(xs)
    H.gn_hvp!(Ha, pb, w, wl, a); H.gn_hvp!(Hc, pb, w, wl, c)
    @test abs(H.dotN(par, c, Ha) - H.dotN(par, a, Hc)) <= 1e-12 * abs(H.dotN(par, c, Ha))
    @test H.dotN(par, a, Ha) > 0
    # linearized force seeds = FD of the nonlinear ones
    v = srand_dir(par, xs, 1:5); H.gn_hvp!(Ha, pb, w, wl, v)
    ε = 1e-6; b = par.block
    wp = H.SMHSWork(par); wm = H.SMHSWork(par); xt = H.similar_state(xs)
    H.lincomb!(xt, 1.0, xs, ε, v, par); H.objective!(pb, wp, xt)
    H.lincomb!(xt, 1.0, xs, -ε, v, par); H.objective!(pb, wm, xt)
    for q in 1:3
        num = 0.0; den = 0.0
        for k in axes(w.aR[q], 3), j in 1:b.ny, i in 1:b.nx
            fd = (wp.aR[q][i, j, k] - wm.aR[q][i, j, k]) / (2ε)
            num = max(num, abs(fd - wl.aR[q][i, j, k])); den = max(den, abs(fd))
        end
        @test num <= 1e-4 * den
    end
end

@testset "staggered: colored Ã preconditioner diagonal is the exact GN diagonal (M5)" begin
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); dfaces = true, lq = 1.0)   # 16×14: period 5 needs remainder colors
    pb.pc_atilde = :colored; pb.pc_period = 5
    w = H.SMHSWork(par); wl = H.SMHSWork(par); H.objective!(pb, w, xs)
    D = H.gn_diagonal!(H.similar_state(xs), pb, w, xs)
    b = par.block; rng = MersenneTwister(4); e = 0.0
    z = H.similar_state(xs); Hz = H.similar_state(xs)
    for t in 1:40
        f = rand(rng, 2:4); i, j, k = rand(rng, 1:b.nx), rand(rng, 1:b.ny), rand(rng, axes(xs[f], 3))
        for q in 1:5; fill!(parent(z[q]), 0.0); end
        z[f][i, j, k] = 1.0
        H.gn_hvp!(Hz, pb, w, wl, z)
        e = max(e, abs(D[f][i, j, k] - Hz[f][i, j, k]) / abs(Hz[f][i, j, k]))
    end
    rank0(par) && @info "colored GN diagonal vs single-node products (40 nodes)" e
    @test e <= 1e-12
    # u floor: no node below the fraction of its layer median
    pb.pc_ufloor = 1e-2
    D2 = H.gn_diagonal!(H.similar_state(xs), pb, w, xs)
    med = H.layer_quantiles(par, D2[1], [0.5])[:, 1]
    @test all(D2[1][i, j, k] >= 1e-2 * med[k] * (1 - 1e-12) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
end

@testset "M5: constrained pressure projection and block termination (Codex review counterexamples)" begin
    par = H.init_parallel(4, 4, 9); b = par.block
    pb, x, us, uc, grid = H.sc1_problem(par, 4, 4, 9; lambda_Q = 1e-3)
    w = H.SMHSWork(par); H.objective!(pb, w, x)
    t = H.newfield(par); δ = H.newfield(par); lo = H.newfield(par)
    H.owned_map!((i, j, k) -> k == pb.ka ? (isodd(i) ? -2.0 : 2.0) : 0.0, t, par)
    H.owned_map!((i, j, k) -> w.u[i, j, k] - ((k == pb.ka && isodd(i)) ? 0.0 : 3.0), lo, par)
    Δ = 0.5
    # alternation trust-region step: bounds and anchor hold exactly
    err = H.anchor_box_project!(δ, t, (i, j, k) -> min(max(-Δ, lo[i, j, k] - w.u[i, j, k]), 0.0), (i, j, k) -> Δ, (i, j, k) -> 1.0, pb)
    margin = minimum(w.u[i, j, k] + δ[i, j, k] - lo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
    @test err <= 1e-12 && abs(H.anchor_mean(pb, δ)) <= 1e-12
    @test margin >= -1e-12 && maximum(abs.(δ[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)) <= Δ + 1e-12
    # L-BFGS direction: nodes at the bound do not move down, anchor kept, so u + α δ ≥ lo for α ≤ cap
    lob(i, j, k) = w.u[i, j, k] - lo[i, j, k] <= 5e-4 ? 0.0 : -Inf
    H.anchor_box_project!(δ, t, lob, (i, j, k) -> Inf, (i, j, k) -> 1.0, pb)
    acap = min(1e6, Δ / max(H.maxabs_all(par, δ), 1e-300), H.step_to_lower(δ, w.u, lo, par; tol = 5e-4))   # here δ ≡ 0 is the only feasible direction
    margin = minimum(w.u[i, j, k] + acap * δ[i, j, k] - lo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
    @test margin >= -1e-12 && abs(H.anchor_mean(pb, δ)) <= 1e-12
    # v-coordinate u diagonal equals the single-node GN product, also on the anchor plane
    H.owned_map!((i, j, k) -> x[1][i, j, k] + 0.8 * sin(2pi * (i - 1) / b.nx), x[1], par)
    H.objective!(pb, w, x)
    D = H.newfield(par); H.gn_diag_v!(D, pb, w, x)
    z = H.similar_state(x); Hz = H.similar_state(x); wl = H.SMHSWork(par)
    for k in (pb.ka - 1, pb.ka)
        for q in 1:5; fill!(parent(z[q]), 0.0); end
        z[1][2, 2, k] = 1.0
        H.gn_hvp!(Hz, pb, w, wl, z)
        @test abs(D[2, 2, k] - Hz[1][2, 2, k]) <= 1e-12 * abs(Hz[1][2, 2, k])
    end
    # zero block work is not convergence
    st, why, n, hist = H.alternating_solve!(pb, w, x; max_outer = 1, a_iter = 0, u_steps = 0)
    @test st != H.converged
end
