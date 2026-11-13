# ln nₑ of run outputs from their own EOS table at (T, Pgas): writes <out>/<run>.ne.h5 with grid/lnne (same layout).
# usage: julia --project=. add_ne.jl TABLE OUTDIR RUN.h5 [RUN.h5 ...]
using HSEMHS, HDF5
const H = HSEMHS
tab = H.load_table(ARGS[1])
for fn in ARGS[3:end]
    h5open(fn) do f
        g = haskey(f, "tau_grid") ? f["tau_grid"] : f["mass_grid"]
        T = read(g["T"]); P = read(g["Pgas"]); ne = fill(NaN, size(T))
        Threads.@threads for i in eachindex(T)
            ne[i] = H.state(tab, log(Float64(T[i])), log(Float64(P[i])))[2]
        end
        out = joinpath(ARGS[2], replace(basename(fn), ".h5" => ".ne.h5"))
        h5open(out, "w") do o
            o["grid/lnne"] = ne; attributes(o)["eos_table"] = abspath(ARGS[1]); attributes(o)["eos_model"] = H.eos_model(tab)
        end
        println("$out: ", count(!isfinite, ne), " non-finite")
    end
end
