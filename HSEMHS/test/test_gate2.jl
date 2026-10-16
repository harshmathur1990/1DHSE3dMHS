# Gate 2 (spec §5, implementation doc §9.5).

"""Run column_hse on a single rank for a T(x,y,z) function with the given EOS."""
function colhse_run(NX, NY, z, Tfun, eos, o)
    par = H.init_parallel(NX, NY, length(z))
    x = collect((0:NX-1) .* 1.0e5); y = collect((0:NY-1) .* 1.0e5)
    grid = H.Grid(H.build_axis("x", x; periodic = true), H.build_axis("y", y; periodic = true), H.build_axis("z", z))
    T = H.newfield(par)
    for k in eachindex(z), j in 1:NY, i in 1:NX
        T[i, j, k] = Tfun(i, j, z[k])
    end
    return H.column_hse(par, grid, T, eos, o), grid
end

@testset "Gate 2: isothermal ideal gas" begin
    z = test_z(251; top = 2.0e6, bot = -0.5e6, jitter = 0.3, seed = 5)
    mu = 1.3; T0 = 6000.0; g = 274.0
    Hs = H.K_B * T0 / (mu * H.M_U * g)
    eos = H.IdealGasMu(mu)
    for (anchor, Pref, za) in ((:bottom, 1.0e5, z[end]), (:top, 0.1, z[1]))
        o = H.HSEOpts(anchor = anchor, P_ref = Pref, g = g, n_sub = 64, adapt = false)
        res, _ = colhse_run(4, 3, z, (i, j, zz) -> T0, eos, o)
        P = res.fields["Pgas"]
        err = maximum(abs(P[i, j, k] / (Pref * exp(-(z[k] - za) / Hs)) - 1) for i in 1:4, j in 1:3, k in eachindex(z))
        @info "isothermal" anchor err
        @test err <= 1e-12
        ak = anchor == :bottom ? length(z) : 1
        @test all(P[i, j, ak] === Pref for i in 1:4, j in 1:3)                 # bitwise anchor
        @test all(P[i, j, k+1] > P[i, j, k] for i in 1:4, j in 1:3, k in 1:length(z)-1)
        tau = res.fields["tau500"]
        @test all(tau[i, j, k+1] > tau[i, j, k] for i in 1:4, j in 1:3, k in 1:length(z)-1)
    end
    # height anchor: P(z_ref) = P_ref
    zr = 0.3e6
    o = H.HSEOpts(anchor = :height, P_ref = 1.0e3, z_ref = zr, g = g, n_sub = 64, adapt = false)
    res, _ = colhse_run(3, 3, z, (i, j, zz) -> T0, eos, o)
    P = res.fields["Pgas"]
    @test maximum(abs(P[1, 1, k] / (1.0e3 * exp(-(z[k] - zr) / Hs)) - 1) for k in eachindex(z)) <= 1e-12
end

@testset "Gate 2: ln T-linear ideal gas, analytic and 4th order" begin
    # T = Ta (Tb/Ta)^s on each interval (the solver's interpolant), so
    # ∫ dz/T = Δz (1/Ta − 1/Tb)/ln(Tb/Ta)   (erratum E13: the spec's formula is for linear T)
    mu = 1.3; g = 274.0; c = mu * H.M_U * g / H.K_B
    z = collect(range(2.0e6, -0.5e6; length = 21))
    Tz(zz) = 4000.0 + 4000.0 * (2.0e6 - zz) / 2.5e6 + 1500 * sin(zz / 3.0e5)
    eos = H.IdealGasMu(mu)
    exact = zeros(length(z))
    exact[end] = log(1.0e5)
    for k in length(z)-1:-1:1
        Ta, Tb = Tz(z[k+1]), Tz(z[k])
        I = (z[k] - z[k+1]) * (1 / Ta - 1 / Tb) / log(Tb / Ta)
        exact[k] = exact[k+1] - c * I
    end
    errs = Float64[]
    for ns in (1, 2, 4, 8)
        o = H.HSEOpts(anchor = :bottom, P_ref = 1.0e5, g = g, n_sub = ns, adapt = false)
        res, _ = colhse_run(3, 3, z, (i, j, zz) -> Tz(zz), eos, o)
        push!(errs, maximum(abs(log(res.fields["Pgas"][1, 1, k]) - exact[k]) for k in eachindex(z)))
    end
    orders = log2.(errs[1:end-1] ./ errs[2:end])
    @info "RK4 order" errs orders
    @test all(orders .>= 3.9)
    o = H.HSEOpts(anchor = :bottom, P_ref = 1.0e5, g = g, n_sub = 64, adapt = false)
    res, _ = colhse_run(3, 3, z, (i, j, zz) -> Tz(zz), eos, o)
    @test maximum(abs(log(res.fields["Pgas"][1, 1, k]) - exact[k]) for k in eachindex(z)) <= 1e-8
end

@testset "Gate 2: tau500 closure (E1) and order" begin
    Hc = 1.5e5; chi0 = 1e-3
    errs = Float64[]
    for N in (21, 41, 81, 161)
        z = collect(range(2.0e6, 0.0; length = N))
        chi = chi0 .* exp.(-(z .- z[1]) ./ Hc)
        tau = zeros(N)
        m = H.tau_column!(tau, chi, z)
        @test m == H.TAU_EXP_TAIL
        @test tau[1] ≈ chi0 * Hc rtol = 1e-12
        push!(errs, maximum(abs.(tau ./ (chi .* Hc) .- 1)))
    end
    @test all(log2.(errs[1:end-1] ./ errs[2:end]) .>= 1.95)
    # χ increasing upward → half-cell fallback, flagged
    z = collect(range(1.0e6, 0.0; length = 11)); chi = collect(range(2.0, 1.0; length = 11)); tau = zeros(11)
    @test H.tau_column!(tau, chi, z) == H.TAU_HALF_CELL
    @test tau[1] == 0.5 * chi[1] * (z[1] - z[2])
    @test H.tau_column!(tau, chi, z; top = :zero) == H.TAU_ZERO && tau[1] == 0.0
    @test H.tau_column!(tau, chi, z; mu = 0.5) == H.TAU_HALF_CELL && tau[1] == chi[1] * (z[1] - z[2])
end

@testset "Gate 2: Wittmann table on Bifrost columns" begin
    path = joinpath(@__DIR__, "..", "..", "IO", "input-mhs-hse.nc")
    tabf = joinpath(@__DIR__, "..", "cache", "eos_table.h5")
    if !(isfile(path) && isfile(tabf))
        @info "input or table missing; skipped"
    else
        Tcols, z = h5open(path, "r") do f
            permutedims(f["temperature"][:, 1:4, 1:8], (3, 2, 1)), read(f["z"])
        end
        eos = H.load_table(tabf)
        for anchor in (:bottom, :top)
            o = H.HSEOpts(anchor = anchor, P_ref = anchor == :bottom ? 103509.296875 : 0.1, g = 274.0)
            res, _ = colhse_run(8, 4, z, (i, j, zz) -> Float64(Tcols[i, j, findfirst(==(zz), z)]), eos, o)
            F = res.fields
            ok = [res.column_status[i, j] & H.COL_EOS_FAIL == 0 for i in 1:8, j in 1:4]
            P = F["Pgas"]; m = F["colmass"]
            err = maximum(abs(P[i, j, k] - P[i, j, 1] - 274.0 * m[i, j, k]) / P[i, j, k]
                          for i in 1:8, j in 1:4, k in 1:length(z) if ok[i, j])
            @info "mass identity" anchor err
            @test err <= 1e-6
            ak = anchor == :bottom ? length(z) : 1
            @test all(P[i, j, ak] === o.P_ref for i in 1:8, j in 1:4 if ok[i, j])
            @test all(P[i, j, k+1] > P[i, j, k] for i in 1:8, j in 1:4, k in 1:length(z)-1 if ok[i, j])
            @test all(isfinite(F["logtau500"][i, j, k]) for i in 1:8, j in 1:4, k in 1:length(z) if ok[i, j])
        end
    end
end

@testset "Gate 2: thread and rank invariance (bitwise)" begin
    tabf = joinpath(@__DIR__, "..", "cache", "eos_table.h5")
    if !isfile(tabf) || get(ENV, "HSEMHS_SKIP_MPI", "0") == "1"
        @info "skipped (no table or HSEMHS_SKIP_MPI=1)"
    else
        script = joinpath(@__DIR__, "scripts", "colhse_small.jl")
        proj = dirname(@__DIR__)
        dir = mktempdir()
        runs = [(1, 1), (1, 8), (4, 2), (6, 1)]                     # (ranks, threads)
        files = String[]
        for (r, t) in runs
            f = joinpath(dir, "r$(r)_t$(t).h5"); push!(files, f)
            ok = MPI.mpiexec() do exe
                success(pipeline(`$exe -n $r $(Base.julia_cmd()) --project=$proj -t $t $script $f`; stderr = stderr))
            end
            @test ok
        end
        ref = h5open(read, files[1])
        for f in files[2:end]
            d = h5open(read, f)
            @test isequal(d["output"]["Pgas"], ref["output"]["Pgas"])
            @test isequal(d["output"]["logtau500"], ref["output"]["logtau500"])
            @test isequal(d["output"]["colmass"], ref["output"]["colmass"])
        end
    end
end
