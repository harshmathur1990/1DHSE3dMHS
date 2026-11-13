# χ500 of the NLTE model (direct, no table) at snapshot-385 cells against Multi3D, every 16th column.
using HSEMHS, HDF5; const H = HSEMHS
S = "/mn/stornext/d21/RoCS/matsc/3d/run/en024048_hion"; M = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385"
nx = ny = 252; nz = 467; sh = (504, 504, 496)
rd(fn, v, shape) = (c = prod(shape); a = Vector{Float32}(undef, c); open(f -> (seek(f, v * c * 4); read!(f, a)), fn); reshape(a, shape))
T = rd("$S/en024048_hion.hion_385.snap", 1, sh)[1:2:end, 1:2:end, 1:nz]
P = rd("$S/en024048_hion_385.aux", 0, sh)[1:2:end, 1:2:end, 1:nz] .* 1f4
tau = reshape(reinterpret(Float32, read("$M/H/tau500")), nx, ny, nz)
zs = parse.(Float64, split(readlines("$M/mesh")[6]))
PF = joinpath(pkgdir(HSEMHS), "..", "wittman-eos", "pf_Kurucz.input")
m0 = H.with_switch(H.EOSModel(PF), 15019.647166203165, "robust_witt")
mN = H.with_nlte(m0, H.load_nlte_correction(joinpath(pkgdir(HSEMHS), "data", "nlte", "nlte_hion_correction.h5")))
cols = [(i, j) for i in 1:16:nx for j in 1:16:ny]
out = fill(NaN32, 4, nz - 2, length(cols))
Threads.@threads for c in eachindex(cols)
    i, j = cols[c]; n17 = zeros(17)
    for k in 2:nz-1
        X = log(Float64(T[i, j, k])); Y = log(Float64(P[i, j, k]))
        chim = (tau[i, j, k+1] - tau[i, j, k-1]) / abs(zs[k+1] - zs[k-1])
        v, _, ok = H.model_fields(mN, X, Y; n17 = n17)
        out[:, k-1, c] .= (T[i, j, k], P[i, j, k], chim > 0 ? log(chim) : NaN, ok ? v[3] : NaN)
    end
end
h5open(ARGS[1], "w") do f; f["v"] = out; end
