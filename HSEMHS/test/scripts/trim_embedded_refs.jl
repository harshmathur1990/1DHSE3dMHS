# Keep only the datasets the embedded tests read from test/fixtures/embedded_ref (values unchanged), deflated.
using HDF5
const KEEP = ["output/Pgas", "output/rho", "output/ne", "output/chi500", "output/tau500", "diagnostics/z_tau500_1",
              "atmosphere/Pgas", "atmosphere/rho", "atmosphere/tau500", "atmosphere/Pgas_column_hse", "atmosphere/z_tau1_m",
              "tau_grid/Pgas", "tau_grid/rho", "tau_grid/z_m", "tau_grid/z_tau1_m"]
dir = joinpath(@__DIR__, "..", "fixtures", "embedded_ref")
for f in filter(endswith(".h5"), readdir(dir; join = true))
    data = h5open(fd -> Dict(k => read(fd[k]) for k in KEEP if haskey(fd, k)), f)
    tmp = f * ".tmp"
    h5open(tmp, "w") do fo
        for (k, v) in data
            write_dataset(fo, k, v; deflate = 6, shuffle = true, chunk = size(v))
        end
        attributes(fo)["note"] = "subset of the b2401a3 standalone output used by test_embedded.jl / mpi_embedded.jl"
    end
    check = h5open(fd -> all(read(fd[k]) == v for (k, v) in data), tmp)
    check || error("trim changed values in $f")
    mv(tmp, f; force = true)
    println(basename(f), " ", filesize(f))
end
