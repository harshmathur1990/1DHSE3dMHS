# Re-run the reference configs of test/fixtures/embedded_ref with the current code and compare every dataset with the
# stored pre-refactor output (b2401a3). usage (from HSEMHS/): julia --project=. -t 4 test/scripts/compare_embedded_refs.jl OUTDIR
using HSEMHS, HDF5, MPI, TOML, LinearAlgebra, Printf
BLAS.set_num_threads(1)
HSEMHS.ensure_mpi()
outdir = abspath(ARGS[1]); mkpath(outdir)
refdir = joinpath(@__DIR__, "..", "fixtures", "embedded_ref")
function walk(f, g, prefix = "")
    for k in keys(g)
        o = g[k]; p = prefix * "/" * k
        o isa HDF5.Group ? walk(f, o, p) : f(p, o)
    end
end
worst = 0.0; ok = true
for cfgpath in sort(readdir(joinpath(refdir, "configs"); join = true))
    d = TOML.parsefile(cfgpath)
    ref = joinpath(dirname(refdir), "..", "..", d["run"]["output"])        # paths in the configs are relative to HSEMHS/
    new = joinpath(outdir, basename(d["run"]["output"]))
    d["run"]["output"] = new
    tmp = joinpath(outdir, basename(cfgpath)); open(io -> TOML.print(io, d), tmp, "w")
    code = HSEMHS.run_config(tmp)
    m = 0.0; n = 0; bad = String[]
    h5open(ref, "r") do fr
        h5open(new, "r") do fn
            walk(fr) do p, ds
                haskey(fn, p[2:end]) || (push!(bad, "missing $p"); return)
                a = read(ds); b = read(fn[p[2:end]])
                if eltype(a) <: Real && !(eltype(a) <: Bool)
                    size(a) == size(b) || (push!(bad, "size $p"); return)
                    fa = Float64.(a); fb = Float64.(b)
                    both = isnan.(fa) .& isnan.(fb)
                    r = maximum(abs.(fa[.!both] .- fb[.!both]) ./ max.(abs.(fa[.!both]), 1e-300); init = 0.0)
                    m = max(m, r); n += 1
                end
            end
        end
    end
    global worst = max(worst, m)
    m == 0 && isempty(bad) || (global ok = false)
    @printf("%-32s exit %d  datasets %3d  max rel diff %.3e  %s\n", basename(cfgpath), code, n, m, join(bad, ", "))
end
println(ok ? "ALL BITWISE IDENTICAL" : "DIFFERENCES (worst $worst)")
