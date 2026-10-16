include("helpers.jl")

# apply a global 1D axis operator (coefficient matrix) to a vector
apply1d(a::H.Axis, c::Matrix, f) = [sum(c[n, q] * f[a.idx[n, q]] for q in 1:3) for n in 1:length(a)]

@testset "operators: 1D exactness and order" begin
    for z in (test_z(40), reverse(test_z(40)))          # decreasing and increasing
        a = H.build_axis("z", z)
        s = a.s
        rowsum = maximum(abs.(sum(a.c1; dims = 2))) / maximum(abs.(a.c1))
        @test rowsum < 1e-13
        for (f, df, d2f) in ((one.(s), zero.(s), zero.(s)), (s, one.(s), zero.(s)),
                             (s .^ 2, 2s, fill(2.0, length(s))))
            D = apply1d(a, a.c1, f)
            @test maximum(abs.(D .- df)) <= 1e-12 * max(1.0, maximum(abs.(df)))
            D2 = apply1d(a, a.c2, f)
            @test maximum(abs.(D2 .- d2f)) <= 1e-9 * max(1.0, maximum(abs.(d2f)))
        end
    end
    # second-order convergence (interior and boundary rows) on sin
    errs_i = Float64[]; errs_b = Float64[]
    for N in (21, 41, 81, 161)
        t = collect(range(1.0, -1.0; length = N))
        s = t .+ 0.15 .* sin.(π .* t)                               # smooth nonuniform map, decreasing
        a = H.build_axis("z", s)
        D = apply1d(a, a.c1, sin.(3 .* a.s))
        e = abs.(D .- 3cos.(3 .* a.s))
        push!(errs_i, maximum(e[2:end-1])); push!(errs_b, max(e[1], e[end]))
    end
    @test all(log2.(errs_i[1:end-1] ./ errs_i[2:end]) .> 1.8)
    @test all(log2.(errs_b[1:end-1] ./ errs_b[2:end]) .> 1.8)
    # periodic: wrap and order
    perr = Float64[]
    for N in (16, 32, 64, 128)
        L = 2.0
        a = H.build_axis("x", (0:N-1) .* (L / N); periodic = true)
        @test a.idx[1, 1] == N && a.idx[N, 3] == 1
        D = apply1d(a, a.c1, sin.(2π .* a.s ./ L))
        push!(perr, maximum(abs.(D .- (2π / L) .* cos.(2π .* a.s ./ L))))
    end
    @test all(log2.(perr[1:end-1] ./ perr[2:end]) .> 1.95)
    # validation
    @test_throws H.InvalidInput H.build_axis("x", [0.0, 1.0, 2.0, 3.0]; periodic = true, period = 3.0)  # duplicated endpoint
    @test_throws H.InvalidInput H.build_axis("x", [0.0, 1.0, 3.0, 4.0]; periodic = true)                 # nonuniform, no period
    @test_throws H.InvalidInput H.build_axis("z", [3.0, 2.0, 2.5, 1.0])
end

@testset "operators: numpy.gradient oracle" begin
    if isfile(PY)
        z = test_z(30)
        f = sin.(z ./ 3.0e5) .+ (z ./ 1.0e6) .^ 2
        a = H.build_axis("z", z)
        D = apply1d(a, a.c1, f)
        zs = join(string.(z), ","); fs = join(string.(f), ",")
        out = read(`$PY -I -c "import numpy as np; z=np.array([$zs]); f=np.array([$fs]); print(' '.join(repr(float(v)) for v in np.gradient(f, z, edge_order=2)))"`, String)
        g = parse.(Float64, split(strip(out)))
        @test maximum(abs.(D .- g)) <= 1e-13 * maximum(abs.(g))
    else
        @info "python oracle not available; skipped"
    end
end

@testset "operators: 3D transposes, single rank" begin
    par, grid, ops, x, y, z = test_setup(12, 10, 9)
    errs = transpose_errors(par, ops, x, y, z)
    @test all(errs .<= 1e-13)
end

@testset "operators: Lorentz and divergence of simple fields" begin
    par, grid, ops, x, y, z = test_setup(16, 12, 11)
    Bx = H.newfield(par); By = H.newfield(par); Bz = H.newfield(par)
    # uniform field: L = 0 and div B = 0 to rounding (row sums are zero)
    fill_global!(Bx, par, x, y, z, (a...) -> 2.0e-3)
    fill_global!(By, par, x, y, z, (a...) -> 0.0)
    fill_global!(Bz, par, x, y, z, (a...) -> 5.0e-3)
    H.exchange_halos!(par, [Bx, By, Bz])
    m = 0.0; md = 0.0
    for k in 1:par.NZ, j in 1:par.block.ny, i in 1:par.block.nx
        L = H.lorentz(Bx, By, Bz, ops, i, j, k)
        m = max(m, maximum(abs.(L)))
        md = max(md, abs(H.divergence(Bx, By, Bz, ops, i, j, k)))
    end
    @test m < 1e-12 * (5e-3)^2 / H.MU0 / 1e5
    @test md < 1e-15
    # B = (0, b cos(kx), 0): L_x = (b² k / 2μ0) sin(2kx), second order
    errs = Float64[]
    for N in (16, 32, 64)
        par, grid, ops, x, y, z = test_setup(N, 8, 5)
        k_ = 2π / 4.0e6; b = 1.0e-2
        Bx = H.newfield(par); By = H.newfield(par); Bz = H.newfield(par)
        fill_global!(By, par, x, y, z, (xx, a...) -> b * cos(k_ * xx))
        H.exchange_halos!(par, [Bx, By, Bz])
        e = 0.0
        for kk in 1:par.NZ, j in 1:par.block.ny, i in 1:par.block.nx
            Lx = H.lorentz(Bx, By, Bz, ops, i, j, kk)[1]
            e = max(e, abs(Lx - b^2 * k_ / (2H.MU0) * sin(2k_ * x[i])))
        end
        push!(errs, e / (b^2 * k_ / (2H.MU0)))
    end
    @test all(log2.(errs[1:end-1] ./ errs[2:end]) .> 1.9)
end
