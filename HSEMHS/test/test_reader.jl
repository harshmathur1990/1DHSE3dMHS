@testset "reader: fixtures" begin
    par = H.init_parallel(10, 8, 9)
    for f in ("good.nc", "good4d.nc")
        spec = H.open_input(joinpath(FIX, f))
        @test spec.z[1] == 2.0e6 && issorted(spec.z; rev = true)
        @test spec.periodic == (true, true)
        @test check_fixture_read(joinpath(FIX, f), par) == 0
        grid = H.build_grid(spec)
        @test grid.x.period ≈ 5.0e5
    end
    # crop keeps top-first order and the right levels
    spec = H.open_input(joinpath(FIX, "good.nc"); crop_z_m = [0.0, 1.5e6])
    @test all(0.0 - 1 .<= spec.z .<= 1.5e6 + 1) && spec.krange == 3:7
    # rejections
    @test_throws H.InvalidInput H.open_input(joinpath(FIX, "bad_z.nc"))
    @test_throws H.InvalidInput H.build_grid(H.open_input(joinpath(FIX, "bad_dup.nc")))
    s = H.open_input(joinpath(FIX, "bad_T.nc"))
    @test_throws H.InvalidInput H.read_field(s, par, "temperature", :temperature; positive = true)
    s = H.open_input(joinpath(FIX, "bad_units.nc"))
    @test_throws H.InvalidInput H.read_field(s, par, "B_y", :field)
    @test_throws H.InvalidInput H.open_input(joinpath(FIX, "missing.nc"))
end

@testset "reader: production input hyperslab" begin
    path = joinpath(@__DIR__, "..", "..", "IO", "input-mhs-hse.nc")
    if isfile(path)
        spec = H.open_input(path)
        @test length(spec.z) == 177 && spec.periodic == (true, true)
        grid = H.build_grid(spec)
        @test grid.x.period == 23999976.0
        par = H.init_parallel(504, 504, 177)
        T = H.read_field(spec, par, "temperature", :temperature; positive = true)
        ref = h5open(path, "r") do f
            permutedims(f["temperature"][:, 1:7, 1:9], (3, 2, 1))   # independent path: (x, y, z)
        end
        @test T[1:9, 1:7, 1:177] == Float64.(ref)
    else
        @info "IO/input-mhs-hse.nc not found; skipped"
    end
end
