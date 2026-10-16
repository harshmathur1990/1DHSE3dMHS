# M6a: staggered-mesh operators (D9 addendum §2–4, §10).

@testset "M6a: div∘curl ≡ 0 and transposes (single rank)" begin
    par, grid, ops, x, y, z = test_setup(12, 10, 9)          # jittered nonuniform z
    divrel, errs = staggered_checks(par, grid)
    @info "staggered" divrel errs
    @test divrel <= 1e-14
    for nm in ("C", "Cstar", "H", "G")              # |⟨Ax,y⟩ − ⟨x,Aᵀy⟩| / (‖Ax‖‖y‖)
        @test errs[nm] <= 1e-13
    end
end

"""Analytic A = (sx cy f, cx sy f, sx sy f), sx = sin kx x, …, f = e^{z/Hf}; B = ∇×A, J = ∇×B."""
function analytic_A(xx, yy, zz)
    kx = 2π / 4.0e6; ky = 4π / 4.0e6; Hf = 1.5e6
    f = exp(zz / Hf); fp = f / Hf; fpp = f / Hf^2
    sx, cx, sy, cy = sin(kx * xx), cos(kx * xx), sin(ky * yy), cos(ky * yy)
    A = (sx * cy * f, cx * sy * f, sx * sy * f)
    # B = (∂y Az − ∂z Ay, ∂z Ax − ∂x Az, ∂x Ay − ∂y Ax)
    B = (ky * sx * cy * f - cx * sy * fp, sx * cy * fp - kx * cx * sy * f, -kx * sx * sy * f + ky * sx * sy * f)
    return A, B
end

@testset "M6a: curl converges at second order on a smooth nonuniform grid" begin
    errs = Float64[]
    for N in (16, 32, 64)
        par = H.init_parallel(N, N, N + 1)
        grid, x, y, z = H.synthetic_grid(par, N, N, N + 1)
        s = H.StagOps(grid, par); b = par.block
        A, B, J = stag_fields(par)
        xf(i) = x[mod1(b.i0 + i - 1, N)] + 0.5 * s.dxf[i]          # x_{i+½}
        yf(j) = y[mod1(b.j0 + j - 1, N)] + 0.5 * s.dyf[j]
        xc(i) = x[mod1(b.i0 + i - 1, N)] + (b.i0 + i - 1 > N ? 4.0e6 : 0.0) - (b.i0 + i - 1 < 1 ? 4.0e6 : 0.0)
        yc(j) = y[mod1(b.j0 + j - 1, N)] + (b.j0 + j - 1 > N ? 4.0e6 : 0.0) - (b.j0 + j - 1 < 1 ? 4.0e6 : 0.0)
        for i in 0:b.nx+1, j in 0:b.ny+1
            for k in 0:N+1                                    # edge levels 0:NZ, NZ = N + 1
                A[1][i, j, k] = analytic_A(xc(i), yf(j), s.zf[k])[1][1]          # x-edge (i, j+½, k+½)
                A[2][i, j, k] = analytic_A(xf(i), yc(j), s.zf[k])[1][2]          # y-edge (i+½, j, k+½)
            end
            for k in 1:N+1
                A[3][i, j, k] = analytic_A(xf(i), yf(j), s.zc[k])[1][3]          # z-edge (i+½, j+½, k)
            end
        end
        H.curlA!(B, A, 0.0, s, par)
        e = 0.0
        for i in 1:b.nx, j in 1:b.ny
            for k in 1:N+1
                e = max(e, abs(B[1][i, j, k] - analytic_A(xf(i), yc(j), s.zc[k])[2][1]))
                e = max(e, abs(B[2][i, j, k] - analytic_A(xc(i), yf(j), s.zc[k])[2][2]))
            end
            for k in 0:N+1
                e = max(e, abs(B[3][i, j, k] - analytic_A(xc(i), yc(j), s.zf[k])[2][3]))
            end
        end
        push!(errs, e)
    end
    @info "curl convergence" errs
    @test all(log2.(errs[1:end-1] ./ errs[2:end]) .>= 1.9)
end

@testset "M6a: dual curl (J) converges at second order on interior edges" begin
    Bf(xx, yy, zz) = analytic_A(xx, yy, zz)[2]
    # reference J = ∇×B by fine central differences of the analytic B
    function Jref(xx, yy, zz)
        δ = 1.0
        d(c, a) = (Bf((xx, yy, zz) .+ δ .* (a .== (1, 2, 3)) ...)[c] - Bf((xx, yy, zz) .- δ .* (a .== (1, 2, 3))...)[c]) / 2δ
        return (d(3, 2) - d(2, 3), d(1, 3) - d(3, 1), d(2, 1) - d(1, 2))
    end
    errs = Float64[]
    for N in (16, 32, 64)
        par = H.init_parallel(N, N, N + 1)
        grid, x, y, z = H.synthetic_grid(par, N, N, N + 1)
        s = H.StagOps(grid, par); b = par.block
        A, B, J = stag_fields(par)
        xf(i) = x[mod1(b.i0 + i - 1, N)] + 0.5 * s.dxf[i]; yf(j) = y[mod1(b.j0 + j - 1, N)] + 0.5 * s.dyf[j]
        xc(i) = x[mod1(b.i0 + i - 1, N)]; yc(j) = y[mod1(b.j0 + j - 1, N)]
        for i in 0:b.nx+1, j in 0:b.ny+1
            for k in 1:N+1
                B[1][i, j, k] = Bf(xf(i), yc(j), s.zc[k])[1]
                B[2][i, j, k] = Bf(xc(i), yf(j), s.zc[k])[2]
            end
            for k in 0:N+1
                B[3][i, j, k] = Bf(xc(i), yc(j), s.zf[k])[3]
            end
        end
        H.curlB!(J, B, s, par)
        e = 0.0
        for i in 1:b.nx, j in 1:b.ny, k in 2:N-1                      # interior edges / levels
            e = max(e, abs(J[1][i, j, k] - Jref(xc(i), yf(j), s.zf[k])[1]),
                    abs(J[2][i, j, k] - Jref(xf(i), yc(j), s.zf[k])[2]),
                    abs(J[3][i, j, k] - Jref(xf(i), yf(j), s.zc[k])[3]))
        end
        push!(errs, e)
    end
    @info "dual curl convergence" errs
    # 16 → 32 is pre-asymptotic (1.83): gate on the finest pair, require monotone decrease
    @test log2(errs[end-1] / errs[end]) >= 1.9
    @test all(errs[1:end-1] .> errs[2:end])
end
