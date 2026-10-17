@testset "units and config" begin
    @test H.UNIT_FACTORS[:field]["G"] == 1e-4
    @test H.UNIT_FACTORS[:pressure]["dyn cm-2"] == 0.1
    @test H.UNIT_FACTORS[:length]["km"] == 1e3

    cfg = H.parse_config("""
    [run]
    mode = "gate0"
    output = "x.h5"
    [input]
    file = "in.nc"
    [parallel]
    px = 2
    """)
    @test cfg.run.mode == "gate0"
    @test cfg.parallel.px == 2
    @test cfg.parallel.py == 0
    @test cfg.mhs.lambda_D == 1.0
    @test cfg.mhs.lambda_Q == 1.0 && cfg.mhs.lambda_Uz == 0.0 && cfg.mhs.data_on_faces === true && cfg.mhs.collapse_du == 3.0
    @test cfg.mhs.bfirst_ratio == 10.0 && cfg.mhs.b_first_iter == 2000
    @test cfg.gate0.stride == 4
    @test length(H.config_hash(cfg)) == 64

    # unknown key / section, missing required, wrong type, bad mode
    @test_throws H.InvalidInput H.parse_config("[run]\nmode=\"gate0\"\noutput=\"a\"\nbogus=1\n[input]\nfile=\"f\"")
    @test_throws H.InvalidInput H.parse_config("[run]\nmode=\"gate0\"\noutput=\"a\"\n[input]\nfile=\"f\"\n[nope]\na=1")
    @test_throws H.InvalidInput H.parse_config("[run]\nmode=\"column_hse\"\noutput=\"a\"\n[input]\nfile=\"f\"")
    @test_throws H.InvalidInput H.parse_config("[run]\nmode=\"gate0\"\noutput=\"a\"\n[input]\nfile=\"f\"\n[parallel]\npx=\"two\"")
    @test_throws H.InvalidInput H.parse_config("[run]\nmode=\"hse3d\"\noutput=\"a\"\n[input]\nfile=\"f\"")
    c2 = H.parse_config("[run]\nmode=\"column_hse\"\noutput=\"a\"\n[input]\nfile=\"f\"\n[column_hse]\nP_ref=\"from_input\"")
    @test H.resolve_from_input(c2.column_hse.P_ref, Dict("P_ref_Pa" => 1.0e5), "P_ref_Pa", "column_hse.P_ref") == 1.0e5
    @test_throws H.InvalidInput H.resolve_from_input("from_input", Dict(), "P_ref_Pa", "k")
end
