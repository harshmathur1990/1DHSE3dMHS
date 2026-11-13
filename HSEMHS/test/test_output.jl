@testset "output: schema and token-ring write (single rank)" begin
    par = H.init_parallel(9, 7, 5)
    path = joinpath(mktempdir(), "out.h5")
    write_pattern(par, path)
    @test check_pattern(path, 9, 7, 5)
    @test_throws H.InvalidInput H.create_output(par, path; overwrite = false)
end

@testset "output: parallel HDF5 only when asked, > 1 rank, MPI-IO and a parallel file system" begin
    par = H.init_parallel(9, 7, 5)
    path = joinpath(mktempdir(), "sub", "out.h5")                 # not yet existing: its nearest existing ancestor
    fs = H.fs_type(path)
    @test Sys.islinux() ? !isempty(fs) : fs == ""
    base = "[run]\nmode=\"column-hse-ltau\"\noutput=\"a\"\n[input]\nfile=\"f\"\n"
    c = H.parse_config(base)
    @test c.output.parallel_hdf5 == false && "lustre" in c.output.parallel_fs && !("cvfs" in c.output.parallel_fs)
    @test H.hdf5_io(par, c.output, path) == (:cooperative, "output.parallel_hdf5 = false")
    on = H.parse_config(base * "[output]\nparallel_hdf5 = true\nparallel_fs = [\"$fs\"]\n")
    @test on.output.parallel_fs == [fs]
    @test H.hdf5_io(par, on.output, path) == (:cooperative, "1 rank")                 # this test runs on one rank
    @test H.config_hash(H.parse_config(base * "[output]\nparallel_fs = $(repr(c.output.parallel_fs))\n")) == H.config_hash(c)
    @test_throws H.InvalidInput H.parse_config(base * "[output]\nparallel_fs = [1, 2]\n")
    @test H.hdf5_io_line(:parallel, "parallel: file system lustre") == "output: parallel HDF5 (collective MPI-IO, file system lustre)"
    @test H.hdf5_io_line(:cooperative, "1 rank") == "output: cooperative HDF5 (1 rank)"
end
