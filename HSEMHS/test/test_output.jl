@testset "output: schema and token-ring write (single rank)" begin
    par = H.init_parallel(9, 7, 5)
    path = joinpath(mktempdir(), "out.h5")
    write_pattern(par, path)
    @test check_pattern(path, 9, 7, 5)
    @test_throws H.InvalidInput H.create_output(par, path; overwrite = false)
end
