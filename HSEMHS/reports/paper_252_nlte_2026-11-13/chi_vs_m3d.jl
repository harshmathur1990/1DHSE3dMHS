# χ500 of the LTE and NLTE tables at the cells of snapshot 385 against Multi3D's (dτ500/dz), subsample of columns.
using HSEMHS, HDF5, Statistics; const H = HSEMHS
S = "/mn/stornext/d21/RoCS/matsc/3d/run/en024048_hion"; M = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385"
nx = ny = 252; nz = 467; sh = (504, 504, 496)
function rd(fn, v, shape)
    c = prod(shape); a = Vector{Float32}(undef, c)
    open(fn) do f; seek(f, v * c * 4); read!(f, a); end
    reshape(a, shape)
end
T = rd("$S/en024048_hion.hion_385.snap", 1, sh)[1:2:end, 1:2:end, 1:nz]
P = rd("$S/en024048_hion_385.aux", 0, sh)[1:2:end, 1:2:end, 1:nz] .* 1f4          # Pa
ns = filesize("$M/H/tau500") ÷ (nx * ny * nz); println("tau500 bytes/cell ", ns)
tau = ns == 8 ? reshape(reinterpret(Float64, read("$M/H/tau500")), nx, ny, nz) : reshape(reinterpret(Float32, read("$M/H/tau500")), nx, ny, nz)
ml = readlines("$M/mesh"); zs = parse.(Float64, split(ml[6]))       # mesh: nx, x, ny, y, nz, z [cm]
@assert length(zs) == nz
tL = H.load_table("cache/eos_table_fullz.h5"); tN = H.load_table("cache/eos_table_nlte.h5")
cols = [(i, j) for i in 1:4:nx for j in 1:4:ny]
out = zeros(Float32, 5, nz - 2, length(cols))       # T, P, lnχ_m3d [per cm], lnχ_LTE, lnχ_NLTE
Threads.@threads for c in eachindex(cols)
    i, j = cols[c]
    for k in 2:nz-1
        X = log(Float64(T[i, j, k])); Y = log(Float64(P[i, j, k]))
        chim = (tau[i, j, k+1] - tau[i, j, k-1]) / abs(zs[k+1] - zs[k-1])
        out[:, k-1, c] .= (T[i, j, k], P[i, j, k], chim > 0 ? log(chim) : NaN, H.state(tL, X, Y)[3], H.state(tN, X, Y)[3])
    end
end
h5open(ARGS[1], "w") do f; f["v"] = out; f["z"] = zs[2:end-1]; f["tau"] = Float32[tau[i, j, k] for k in 2:nz-1, (i, j) in cols]; end
println("done")
