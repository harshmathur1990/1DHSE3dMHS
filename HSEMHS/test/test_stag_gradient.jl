# M6b: staggered objective/adjoint verification (D9 addendum §8).

"""Staggered MHS test problem with structured T, smooth Bobs and a perturbed state."""
function sgrad_problem(eos; NX = 16, NY = 14, NZ = 15, lam = (1.0, 1e-3, 1e-6), Bscale = 1.0, apert = 0.3, seed = 3, px = 0, py = 0)
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
    pb = H.SMHSProblem(par, grid, T, Bo, sig, eos; P_ref = 1.0e4, lambda_F = lam[1], lambda_S = lam[2], lambda_U = lam[3])
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
        for (lam, nm) in (((1.0, 0.0, 0.0), "force"), ((0.0, 1e-3, 0.0), "reg δB"),
                          ((0.0, 0.0, 1e-6), "reg u"), ((0.0, 0.0, 0.0), "data"))
            pb2, xs2, par2 = sgrad_problem(eos; lam = lam)
            e = sfd_error(pb2, xs2, srand_dir(par2, xs2, 1:5))
            @test e <= 1e-6
            rank0(par2) && @info "staggered FD $label, term $nm" e
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
    pb, xs, par = sgrad_problem(H.IdealGasMu(1.3); apert = 0.0)
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
