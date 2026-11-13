# NLTE hydrogen ionization (eos.model = "nlte", src/eos/NLTE.jl).

const NLTE_CORRECTION = joinpath(@__DIR__, "..", "data", "nlte", "nlte_hion_correction.h5")
@isdefined(PF_FILE) || (const PF_FILE = joinpath(@__DIR__, "..", "..", "wittman-eos", "pf_Kurucz.input"))
@isdefined(ETABLE) || include("embedded_helpers.jl")

@testset "NLTE EOS" begin
    if !isfile(PF_FILE) || !isfile(NLTE_CORRECTION)
        @info "pf file or NLTE correction missing; skipped"
    else
        corr = H.load_nlte_correction(NLTE_CORRECTION)
        @test all(((d, w),) -> w != 0 || d == 0, zip(corr.D, corr.W))
        m0 = H.with_switch(H.EOSModel(PF_FILE), 15019.647166203165, "robust_witt")
        mN = H.with_nlte(m0, corr)
        h = m0.hots[1]; eW = h.eion[1][1]
        rng = Xoshiro(11)
        pts = [(log(10^(3.45 + 1.3rand(rng))), log(10^(-2.5 + 7rand(rng)))) for _ in 1:400]

        @testset "zero correction is the LTE model" begin
            z0 = H.NLTECorrection(corr.lT0, corr.dlT, corr.lP0, corr.dlP, zero(corr.D), zero(corr.W), "", "", Dict{String,Any}())
            mZ = H.with_nlte(m0, z0)
            @test all(H.model_fields(m0, X, Y) == H.model_fields(mZ, X, Y) for (X, Y) in pts)
        end

        @testset "n_p/n_HI = Saha(T, pₑ) 10^dH at the NLTE solution" begin
            n = 0; worst = 0.0
            for (X, Y) in pts
                T = exp(X); uH, _ = H.hot_pf(h, T)
                dH = H.nlte_departure(corr, X, Y, () -> uH, eW)
                dH == 0 && continue
                v, fl, ok = H.model_fields(mN, X, Y)
                ok || continue
                @test fl & H.EOS_NLTE != 0
                pe = exp(v[2]) * 1e-6 * H.BK_CGS * T
                worst = max(worst, abs((v[5] - v[4]) / log(10) - log10(H.saha_w(5040 / T, eW, uH[1], uH[2], pe)) - dH))
                n += 1
            end
            @test n > 50
            @test worst < 1e-10
            # cool chromospheric gas is over-ionized, warm gas under-ionized (Bifrost hion)
            @test H.nlte_departure(corr, log(4000.0), log(1.0), () -> [2.0, 1.0], eW) > 0.5
            @test H.nlte_departure(corr, log(9000.0), log(0.1), () -> [2.0, 1.0], eW) < -0.5
        end

        @testset "NLTE table: five fields, hydrogen, model checks" begin
            env = H.Envelope(4000.0, 20000.0, -1.0, 2.0, "test")
            tabN = H.build_table(mN, env; dlog10T = 0.005, dlog10P = 0.05, refine_max_levels = 3, log = devnull)
            tabL = H.build_table(m0, env; dlog10T = 0.02, dlog10P = 0.2, refine_max_levels = 1, log = devnull)
            @test H.nfields(tabN) == 5 && H.nfields(tabL) == 3
            @test count(f -> f & H.EOS_NLTE != 0, tabN.flags) > 0
            for _ in 1:200
                X = log(10^(3.62 + 0.66rand(rng))); Y = log(10^(-0.9 + 2.8rand(rng)))
                v, _, ok = H.model_fields(mN, X, Y)
                ok || continue
                r, ne, c, _ = H.state(tabN, X, Y)
                hi, hp, _ = H.hydrogen(tabN, X, Y)
                @test maximum(abs.((r, ne, c, hi, hp) .- v)) < 5e-2        # 728 × 215 nodes: max 4.3e-2 (ln n_HI, 11 kK) in 20000 points
            end
            @test_throws H.InvalidInput H.hydrogen(tabL, log(6000.0), log(10.0))
            dir = mktempdir(); fN = joinpath(dir, "nlte.h5"); fL = joinpath(dir, "lte.h5")
            H.save_table(fN, tabN, Dict{String,Any}("eos_model" => "nlte"))
            H.save_table(fL, tabL, Dict{String,Any}("eos_model" => "lte"))
            back = H.load_table(fN; model = "nlte")
            @test H.eos_model(back) == "nlte" && H.table_sha256(back) == H.table_sha256(tabN)
            @test H.hydrogen(back, log(6000.0), log(10.0))[1:2] == H.hydrogen(tabN, log(6000.0), log(10.0))[1:2]
            @test H.eos_model(H.load_table(fL)) == "lte"
            @test_throws H.InvalidInput H.load_table(fN; model = "lte")
            @test_throws H.InvalidInput H.load_table(fL; model = "nlte")
        end
    end

    @testset "config: eos.model selects the table" begin
        base = "[run]\nmode=\"column-hse-ltau\"\noutput=\"a\"\n[input]\nfile=\"f\"\n"
        c = H.parse_config(base)
        @test c.eos.model == "lte" && H.eos_table_path(c) == c.eos.table_file
        n = H.parse_config(base * "[eos]\nmodel=\"nlte\"\nnlte_table_file=\"cache/t.h5\"\n")
        @test H.eos_table_path(n) == "cache/t.h5"
        @test H.config_hash(H.parse_config(base * "[eos]\nmodel=\"lte\"\n")) == H.config_hash(c)   # earlier hashes unchanged
        @test H.config_hash(n) != H.config_hash(c)
        @test_throws H.InvalidInput H.parse_config(base * "[eos]\nmodel=\"non-lte\"\n")
    end

    @testset "prepare_embedded: eos_model must match the table" begin
        if !isfile(ETABLE)
            @info "cache/eos_table_fullz.h5 not found; skipped"
        else
            tab = H.load_table(ETABLE)
            z = collect(range(1.0e6, -0.2e6; length = 12)); x = [0.0, 5.0e4]; y = [0.0, 5.0e4]
            prep(m) = H.prepare_embedded(MPI.COMM_WORLD, :column_hse_z; axis = z, x = x, y = y, eos = tab, g = 274.0,
                                         column = H.ColumnZOptions(anchor = :top, P_ref = 0.1), eos_model = m)
            ws = prep("lte"); @test ws.eos === tab; H.free_embedded!(ws)
            @test_throws H.InvalidInput prep("nlte")
            @test_throws H.InvalidInput prep("NLTE")
        end
    end
end
