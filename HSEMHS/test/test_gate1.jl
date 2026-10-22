# Gate 1 (spec §4, implementation doc §7.9).
using DelimitedFiles

const WITT_DIR = joinpath(@__DIR__, "..", "..", "wittman-eos")
const PF_FILE = joinpath(WITT_DIR, "pf_Kurucz.input")
const ORACLE = joinpath(@__DIR__, "..", "oracles", "python", "witt_oracle.py")
const TABLE_FILE = joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5")

"""Random (T, P) points, log-uniform, fixed seed."""
function rand_TP(n, T_lo, T_hi, lP_lo, lP_hi; seed = 1)
    rng = Xoshiro(seed)
    T = exp.(log(T_lo) .+ (log(T_hi) - log(T_lo)) .* rand(rng, n))
    P = 10.0 .^ (lP_lo .+ (lP_hi - lP_lo) .* rand(rng, n))
    return T, P
end

function run_oracle(T, P; opacity = false)
    dir = mktempdir()
    fin = joinpath(dir, "in.txt"); fout = joinpath(dir, "out.txt")
    writedlm(fin, hcat(T, P), ' ')
    args = opacity ? `--opacity` : ``
    run(`$PY -I $ORACLE $WITT_DIR $fin $fout 1e-10 5000 $args`)
    return readdlm(fout)
end

@testset "Gate 1: direct solver vs Python oracle" begin
    if !isfile(PY) || !isfile(PF_FILE)
        @info "python or pf file missing; skipped"
    else
        H.witt_fix_gasc_h2() || @warn "libwitt built without FIX_GASC_H2; oracle comparison applies it"
        c = H.WittCtx(PF_FILE)
        T, P = rand_TP(500, 2200.0, 1.0e5, -4.0, 6.0)
        ref = run_oracle(T, P)
        maxrel = 0.0; nconv = 0
        for i in eachindex(T)
            s, code = H.witt_state(c, T[i], P[i])
            code == 0 || continue
            nconv += 1
            maxrel = max(maxrel, abs(s.rho / ref[i, 1] - 1), abs(s.ne / ref[i, 2] - 1))
        end
        @info "direct vs Python" maxrel nconv
        @test nconv >= 450
        @test maxrel <= 1e-8
        # closure is exact with the gasc correction (where the direct solver converges)
        cl = maximum(H.witt_state(c, T[i], P[i])[1].closure_residual for i in eachindex(T)
                     if H.witt_state(c, T[i], P[i])[2] == 0)
        @test cl <= 1e-8
    end
end

@testset "Gate 1: opacity port and partials vs Python" begin
    if !isfile(PY) || !isfile(PF_FILE)
        @info "python or pf file missing; skipped"
    else
        c = H.WittCtx(PF_FILE)
        T, P = rand_TP(1000, 2200.0, 1.0e5, -4.0, 6.0; seed = 2)
        d = run_oracle(T, P; opacity = true)
        port = 0.0; part = 0.0
        n17 = zeros(17)
        for r in 1:size(d, 1)
            Tr, xna, xne = d[r, 1], d[r, 2], d[r, 3]
            n = d[r, 4:20]
            tk = Tr * H.BK_CGS
            op, _ = H.cop(Tr, tk / H.EV_CGS, tk, H.HH_CGS / tk, log(Tr), xna, xne, 5000.0, n)
            port = max(port, abs(op / d[r, 21] - 1))
            s, code = H.witt_state(c, T[r], P[r])
            code == 0 || continue
            H.witt_partials!(n17, c, T[r], P[r], s.pe * 10)
            part = max(part, maximum(abs(n17[k] / n[k] - 1) for k in 1:17 if n[k] > 1e-250))
        end
        @info "opacity port" port part
        @test port <= 1e-12
        @test part <= 1e-8
    end
end

@testset "Gate 1: hot branch = direct model where both converge" begin
    if isfile(PF_FILE)
        c = H.WittCtx(PF_FILE); h = H.HotEOS(c)
        T, P = rand_TP(2000, 2200.0, 1.6e6, -10.0, 14.0; seed = 3)
        worst = 0.0; nfail_hot = 0
        for i in eachindex(T)
            r, ne, pe, ok = H.hot_solve(h, T[i], P[i])
            ok || (nfail_hot += 1; continue)
            s, code = H.witt_state(c, T[i], P[i])
            code == 0 || continue
            worst = max(worst, abs(log(r / s.rho)), abs(log(ne / s.ne)))
        end
        @info "hot vs direct" worst nfail_hot
        @test nfail_hot == 0
        @test worst <= 1e-8
    end
end

@testset "Gate 1: table" begin
    if !isfile(TABLE_FILE)
        @info "cache/eos_table_fullz.h5 not found; run scripts/run.jl config/gate0_bifrost385_fullz.toml, then config/eos_table_bifrost385_fullz.toml"
    else
        tab = H.load_table(TABLE_FILE; pf_file = PF_FILE)
        @test count(f -> f & H.EOS_DIRECT_FAILED != 0, tab.flags) == 0
        m = H.with_switch(H.EOSModel(PF_FILE), tab.T_sw, tab.hot_branch)
        v = H.validate_table(tab, m; n = 50_000, seed = 99)        # fresh withheld sample
        @test v.nfail == 0
        @test v.stats["lnrho_max"] <= 1e-5
        @test v.stats["lnne_max"] <= 1e-4
        @test v.stats["lnchi_max"] <= 1e-4
        d = H.derivative_check(tab, m; n = 2000, seed = 98)
        @test d.max <= 1e-3
        # out of envelope, status, allocation-free lookup
        @test H.lnrho(tab, tab.X[1] - 0.1, tab.Y[1])[2] == H.EOS_OUT_OF_ENVELOPE
        X = 0.5 * (tab.X[1] + tab.X[end]); Y = 0.5 * (tab.Y[1] + tab.Y[end])
        H.lnrho_dY(tab, X, Y)
        @test (@allocated H.lnrho_dY(tab, X, Y)) == 0
        @test (@allocated H.state(tab, X, Y)) == 0
    end
end
