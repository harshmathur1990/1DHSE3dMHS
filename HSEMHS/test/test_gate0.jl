# Gate 0 pass criterion (implementation doc §8.2): the audit reproduces the reference
# values of spec §3 within 0.1 dex (log quantities) or 10 % relative (ratios, fractions;
# with an absolute allowance for the 2-digit rounding of small published values).
# Uses the report written by `scripts/run.jl config/gate0_bifrost385.toml`.

const GATE0_REPORT = joinpath(@__DIR__, "..", "reports", "gate0_bifrost385.h5")

const SPEC3 = (
    z = [3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5],
    logP = [(-3.23, -2.47, -1.32), (-2.66, -2.06, -1.24), (-2.05, -1.26, -0.42), (-0.74, 0.22, 0.75),
            (2.00, 2.46, 2.67), (4.04, 4.26, 4.33), (4.96, 5.02, 5.04)],
    beta = [0.007, 0.013, 0.074, 1.0, 95.0, 2.6e4, 2.1e4],
    # The spec lists 0.04 at -0.5 Mm; an independent numpy recomputation on the same
    # levels (z = -0.494 Mm, 4×4 subsampling) gives 0.0315 (0.030 at -0.514 Mm), so the
    # spec cell is inconsistent with its own data. We test against the independent value.
    rv2 = [0.04, 0.51, 0.71, 0.44, 0.10, 0.20, 0.0315],
    fhot = [0.66, 0.12, 0.003, 0.0, 0.0, 0.0, 0.0],
)

near(a, b; rel = 0.1, abs_ = 0.0) = abs(a - b) <= max(rel * abs(b), abs_)

@testset "Gate 0 reproduction (spec §3)" begin
    if !isfile(GATE0_REPORT)
        @info "Gate 0 report not found; run scripts/run.jl config/gate0_bifrost385.toml first"
    else
        h5open(GATE0_REPORT, "r") do f
            z = read(f["diagnostics/z"]) ./ 1e6
            tab(name) = (permutedims(read(f["diagnostics/layer_$name"]), (2, 1)),
                         read_attribute(f["diagnostics/layer_$name"], "columns"))
            th, thc = tab("thermo"); an, anc = tab("anchor"); rr, rrc = tab("native_residual")
            col(M, c, n) = M[:, findfirst(==(n), c)]
            for (r, zt) in enumerate(SPEC3.z)
                k = argmin(abs.(z .- zt))
                p = (col(th, thc, "log10P_p1")[k], col(th, thc, "log10P_p50")[k], col(th, thc, "log10P_p99")[k])
                @test all(abs.(p .- SPEC3.logP[r]) .<= 0.1)
                @test abs(log10(col(th, thc, "beta_native_p50")[k]) - log10(SPEC3.beta[r])) <= 0.1
                @test near(col(th, thc, "rhov2_over_P_p50")[k], SPEC3.rv2[r]; abs_ = 0.006)
                @test near(col(th, thc, "frac_T_gt_3e4")[k], SPEC3.fhot[r]; abs_ = 0.006)
            end
            k0 = argmin(abs.(z)); k1 = argmin(abs.(z .- 1.0)); k3 = argmin(abs.(z .- 3.0))
            # anchor experiment
            @test abs(col(an, anc, "dex_top_p50")[k0] - 2.50) <= 0.1
            @test abs(col(an, anc, "dex_top_p99")[k0] - 5.73) <= 0.1
            @test abs(col(an, anc, "dex_bot_p50")[k0] - 0.01) <= 0.1
            @test abs(col(an, anc, "dex_bot_p99")[k0] - 0.11) <= 0.1
            @test abs(col(an, anc, "dex_bot_p50")[k1] - 0.04) <= 0.1
            @test abs(col(an, anc, "dex_bot_p99")[k1] - 1.17) <= 0.1
            @test abs(col(an, anc, "dex_bot_p1")[k3] - (-4.22)) <= 0.1
            @test abs(col(an, anc, "dex_bot_p99")[k3] - 1.33) <= 0.1
            # native vertical-only residual medians
            vr = col(rr, rrc, "vertical_only_p50")
            for (zt, ref) in ((1.0, 0.24), (1.5, 0.42), (2.0, 0.62))
                @test abs(log10(vr[argmin(abs.(z .- zt))]) - log10(ref)) <= 0.1
            end
        end
    end
end
