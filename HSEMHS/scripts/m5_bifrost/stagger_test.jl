# Is the input B face-staggered (Bifrost native) or cell-centred? Compare the discrete divergence
# of the full-resolution cropped input under both hypotheses, relative to |B|/h.
using HDF5, Statistics, Printf
f = h5open(ARGS[1])
x = read(f["x"]); y = read(f["y"]); z = read(f["z"])               # z top first
Bx = read(f["B_x"]); By = read(f["B_y"]); Bz = read(f["B_z"])      # Julia (z, y, x)
close(f)
nz, ny, nx = size(Bx); dx = x[2] - x[1]; dy = y[2] - y[1]
# Bifrost: bx at x-face i−½ (between cells i−1 and i), by at j−½, bz at k−½ in its own index (z increasing
# DOWNWARD in Bifrost's mesh, i.e. toward larger depth). Test both vertical face conventions.
res = Dict{String,Vector{Float64}}()
for (name, conv) in (("staggered, bz at upper face", :up), ("staggered, bz at lower face", :down), ("cell-centred (central diff)", :centre))
    vals = Float64[]
    for i in 2:nx-1, j in 2:ny-1, k in 3:nz-2
        Bm = sqrt(Bx[k, j, i]^2 + By[k, j, i]^2 + Bz[k, j, i]^2) + 1e-12
        if conv == :centre
            d = (Bx[k, j, i+1] - Bx[k, j, i-1]) / 2dx + (By[k, j+1, i] - By[k, j-1, i]) / 2dy +
                (Bz[k-1, j, i] - Bz[k+1, j, i]) / (z[k-1] - z[k+1])
            h = dx
        else
            # x-face i−½ ↔ Bx[.,.,i]; cell i lies between faces i−½ and i+½ ↔ Bx[i], Bx[i+1]
            dxB = (Bx[k, j, i+1] - Bx[k, j, i]) / dx
            dyB = (By[k, j+1, i] - By[k, j, i]) / dy
            if conv == :up        # Bz[k] on the face above cell k (z top first: k−1 is above)
                dzB = (Bz[k, j, i] - Bz[k+1, j, i]) / (0.5 * (z[k-1] - z[k+1]))
            else                  # Bz[k] on the face below cell k
                dzB = (Bz[k-1, j, i] - Bz[k, j, i]) / (0.5 * (z[k-1] - z[k+1]))
            end
            d = dxB + dyB + dzB; h = dx
        end
        push!(vals, abs(d) * h / Bm)
    end
    res[name] = vals
end
for (k, v) in res
    @printf("%-32s  |div B| h/|B|: p50 %.2e  p90 %.2e  p99 %.2e  max %.2e\n", k, quantile(v, 0.5), quantile(v, 0.9), quantile(v, 0.99), maximum(v))
end
