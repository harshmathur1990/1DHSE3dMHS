@testset "parallel: decomposition, halos, reductions (single rank)" begin
    # block ranges cover 1:N exactly, sizes differ by at most one
    for N in (7, 10, 504), p in (1, 2, 3, 4, 7)
        p > N ÷ 2 && continue
        rs = [H.block_range(N, p, c) for c in 0:p-1]
        @test rs[1][1] == 1
        @test sum(last.(rs)) == N
        @test all(rs[c+1][1] == rs[c][1] + rs[c][2] for c in 1:p-1)
        @test maximum(last.(rs)) - minimum(last.(rs)) <= 1
    end
    par, grid, ops, x, y, z = test_setup(9, 7, 5)
    @test halo_errors(par) == 0

    # deterministic sum equals the fixed-order serial sum bitwise
    f = H.newfield(par)
    fill_global!(f, par, x, y, z, (xx, yy, zz, gi, gj, k) -> grand(gi, gj, k) * 10.0^(k % 7))
    s = 0.0
    for k in 1:par.NZ, j in 1:par.block.ny
        r = 0.0
        for i in 1:par.block.nx
            r += f[i, j, k]
        end
        s += r
    end
    @test H.global_sum(par, f) === s

    # layer quantiles = Statistics.quantile (type 7) on the subsampled layer
    ps = [0.0, 0.01, 0.5, 0.99, 1.0]
    Q = H.layer_quantiles(par, f, ps; stride = 2)
    for k in 1:par.NZ
        v = [f[i, j, k] for j in 1:2:par.NY for i in 1:2:par.NX]
        @test Q[k, :] == quantile(v, ps)
    end
    fr = H.layer_fraction(par, (i, j, k) -> f[i, j, k] > 0)
    @test fr[1] == mean(f[i, j, 1] > 0 for i in 1:par.NX, j in 1:par.NY)

    # collective status
    st, msgs, _ = H.run_collective(() -> H.invalid("boom"), par)
    @test st == H.invalid_input && occursin("boom", msgs[1])
    st, msgs, v = H.run_collective(() -> 42, par)
    @test st == H.converged && v == 42
end
