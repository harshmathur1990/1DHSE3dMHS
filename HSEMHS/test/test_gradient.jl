# M4a: adjoint gradient verification (implementation doc §13.6).

"""A 16³-ish MHS test problem with structured T, smooth B and a perturbed state."""
function grad_problem(eos; NX = 16, NY = 14, NZ = 15, lam = (1.0, 1.0, 1e-3, 1e-6), Bscale = 1.0, bpert = 0.5, seed = 3)
    par, grid, ops, x, y, z = test_setup(NX, NY, NZ)
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
    # initial pressure: column HSE (bottom anchor)
    res = H.column_hse(par, grid, T, eos, H.HSEOpts(anchor = :bottom, P_ref = 1.0e4, g = 274.0))
    u0 = H.newfield(par); H.fill_owned!((i, j, k) -> log(res.fields["Pgas"][i, j, k]), u0, par)
    pb = H.MHSProblem(par, grid, T, Bo, sig, eos; P_ref = 1.0e4, lambda_F = lam[1], lambda_D = lam[2],
                      lambda_S = lam[3], lambda_U = lam[4])
    H.set_scales!(pb, u0)
    xs = H.init_state(pb, u0)
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        gi, gj = b.i0 + i - 1, b.j0 + j - 1
        xs[1][i, j, k] += 0.05 * grand(gi, gj, k; seed = seed)
        for c in 1:3
            xs[c+1][i, j, k] = bpert * grand(gi, gj, k; seed = seed + c)
        end
    end
    return pb, xs, par
end

"""Random direction with unit global RMS per field (restricted to `fields`)."""
function rand_dir(par, fields; seed = 11)
    b = par.block
    p = H.newstate(par)
    n = par.NX * par.NY * par.NZ
    for f in fields
        for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
            p[f][i, j, k] = grand(b.i0 + i - 1, b.j0 + j - 1, k; seed = seed + 17f)
        end
        r = sqrt(H.global_dot(par, p[f], p[f]) / n)
        H.fill_owned!((i, j, k) -> p[f][i, j, k] / r, p[f], par)
    end
    return p
end

"""min over ε of |FD − ⟨∇Φ, p⟩| / |⟨∇Φ, p⟩|."""
function fd_error(pb, xs, p)
    par = pb.par
    w = H.MHSWork(par)
    g = H.newstate(par)
    H.fg!(g, pb, w, xs)
    gp = H.dotN(par, g, p)
    xt = H.newstate(par)
    best = Inf
    for ε in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        H.lincomb!(xt, 1.0, xs, ε, p, par); fp = H.objective!(pb, w, xt)[1]
        H.lincomb!(xt, 1.0, xs, -ε, p, par); fm = H.objective!(pb, w, xt)[1]
        best = min(best, abs((fp - fm) / (2ε) - gp) / abs(gp))
    end
    return best
end

function gradient_checks(eos, label)
    @testset "gradient: $label" begin
        pb, xs, par = grad_problem(eos)
        for (fields, nm) in ((1:4, "all"), (2:2, "b̃x"), (3:3, "b̃y"), (4:4, "b̃z"))
            e = fd_error(pb, xs, rand_dir(par, fields))
            @test e <= 1e-6
            rank0(par) && @info "FD check $label, fields $nm" e
        end
        # v alone: on the strong-field state Φ ≈ 1e11 is dominated by Lorentz terms and the
        # v-derivative is below the FD round-off floor, so use a weak-field state
        pbw, xsw, parw = grad_problem(eos; Bscale = 0.02, bpert = 0.0)
        e = fd_error(pbw, xsw, rand_dir(parw, 1:1))
        @test e <= 1e-6
        rank0(parw) && @info "FD check $label, field v (weak field)" e
        # each term alone (data term alone: all λ = 0)
        for (lam, nm) in (((1.0, 0.0, 0.0, 0.0), "force"), ((0.0, 1.0, 0.0, 0.0), "divergence"),
                          ((0.0, 0.0, 1e-3, 0.0), "reg δB"), ((0.0, 0.0, 0.0, 1e-6), "reg u"), ((0.0, 0.0, 0.0, 0.0), "data"))
            pb2, xs2, par2 = grad_problem(eos; lam = lam)
            e = fd_error(pb2, xs2, rand_dir(par2, 1:4))
            @test e <= 1e-6
            rank0(par2) && @info "FD check $label, term $nm" e
        end
    end
end
rank0(par) = par.rank == 0

gradient_checks(H.IdealGasMu(1.3), "ideal gas")
let tabf = joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5")
    isfile(tabf) ? gradient_checks(H.load_table(tabf), "Wittmann table") : @info "table missing; skipped"
end

@testset "gradient: forward/adjoint allocations" begin
    pb, xs, par = grad_problem(H.IdealGasMu(1.3))
    w = H.MHSWork(par); g = H.newstate(par)
    H.fg!(g, pb, w, xs); H.fg!(g, pb, w, xs)
    a = @allocated H.fg!(g, pb, w, xs)
    @info "bytes allocated per objective+gradient (threads, halo buffers cached)" a
    @test a < 1_000_000
end

@testset "gradient: thread and rank invariance" begin
    if get(ENV, "HSEMHS_SKIP_MPI", "0") == "1"
        @info "skipped (HSEMHS_SKIP_MPI=1)"
    else
        script = joinpath(@__DIR__, "scripts", "grad_ranks.jl"); proj = dirname(@__DIR__)
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
        @test vals[(1, 4)] == vals[(1, 1)]                                   # bitwise across threads
        for key in ((2, 2), (4, 1))
            @test maximum(abs.(vals[key] ./ vals[(1, 1)] .- 1)) <= 1e-12     # reduction order only
        end
    end
end
