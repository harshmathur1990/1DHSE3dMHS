# M3 hand cases (implementation doc §12).
@testset "M3: screening hand cases" begin
    # (a) uniform P, B = 0: M = 0 and |δL| = 0, so the screening test |δL| ≥ M holds
    par, grid, ops, x, y, z = test_setup(12, 10, 7)
    P = H.newfield(par); Bx = H.newfield(par); By = H.newfield(par); Bz = H.newfield(par)
    fill_global!(P, par, x, y, z, (a...) -> 1.0e4)
    G, L, M, D = H.screening_fields(par, ops, P, Bx, By, Bz; ell = 1e5)
    b = par.block
    @test maximum(abs(M[i, j, k]) for i in 1:b.nx, j in 1:b.ny, k in 1:b.nz) <= 1e-12
    @test all(D[i, j, k] >= M[i, j, k] - 1e-12 for i in 1:b.nx, j in 1:b.ny, k in 1:b.nz)
    # (b) P = P0 (1 + ε cos kx), B = 0: M = P0 ε k |sin kx|, second order
    errs = Float64[]
    for N in (16, 32, 64)
        par, grid, ops, x, y, z = test_setup(N, 8, 5)
        P = H.newfield(par); Bx = H.newfield(par); By = H.newfield(par); Bz = H.newfield(par)
        kx = 2π / 4.0e6; P0 = 1.0e3; ε = 0.1
        fill_global!(P, par, x, y, z, (xx, a...) -> P0 * (1 + ε * cos(kx * xx)))
        G, L, M, D = H.screening_fields(par, ops, P, Bx, By, Bz; ell = 1e5)
        push!(errs, maximum(abs(M[i, j, k] - P0 * ε * kx * abs(sin(kx * x[i]))) for i in 1:N, j in 1:8, k in 1:5) / (P0 * ε * kx))
    end
    @test all(log2.(errs[1:end-1] ./ errs[2:end]) .> 1.9)
    # (c) uniform P, B = (0, b cos kx, 0): |Lh| = |Lx| = (b² k/2μ0)|sin 2kx| = M, second order
    errs = Float64[]
    for N in (16, 32, 64)
        par, grid, ops, x, y, z = test_setup(N, 8, 5)
        P = H.newfield(par); Bx = H.newfield(par); By = H.newfield(par); Bz = H.newfield(par)
        kx = 2π / 4.0e6; bb = 1e-2
        fill_global!(P, par, x, y, z, (a...) -> 1.0e3)
        fill_global!(By, par, x, y, z, (xx, a...) -> bb * cos(kx * xx))
        G, L, M, D = H.screening_fields(par, ops, P, Bx, By, Bz; ell = 1e5)
        ref(i) = bb^2 * kx / (2H.MU0) * abs(sin(2kx * x[i]))
        push!(errs, maximum(max(abs(L[i, j, k] - ref(i)), abs(M[i, j, k] - ref(i))) for i in 1:N, j in 1:8, k in 1:5) / (bb^2 * kx / (2H.MU0)))
        # δL = |B| σ_B/(μ0 ℓ), σ = sqrt((0.05|B|)² + (5e-4)²) per component
        i = 3; Bm = abs(bb * cos(kx * x[i]))
        @test D[i, 1, 1] ≈ Bm * sqrt(3) * sqrt((0.05Bm)^2 + 5e-4^2) / (H.MU0 * 1e5) rtol = 1e-12
    end
    @test all(log2.(errs[1:end-1] ./ errs[2:end]) .> 1.9)
end
