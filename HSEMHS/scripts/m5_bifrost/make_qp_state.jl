# Put a divergence-free face field into the solver representation (Codex M5e review §2): B = curl(A_ref Ã) + b̄ ẑ.
# Explicit construction in the gauge Ãz = 0 (exact, no iterative fit):
#   b̄ = area mean of Bz on the top face; top level: Ãy from the x-integral of Bz − b̄ − (its row mean), Ãx from the
#   y-integral of minus that row mean (both periodic because the means vanish);
#   downwards: Ay[k] = Ay[k−1] + h_k Bx[k],  Ax[k] = Ax[k−1] − h_k By[k]  (Bx, By exactly; Bz then by div B = 0).
# Then B is recomputed by the solver's own objective! and compared with the target face field, face by face in σ.
# The pressure part v is copied from VSTART.h5 (x1). Reports the solver's data/smoothness terms and both observation
# models' residuals for the new state, and for comparison for the unconstrained init_A! fit (Φ_B + Φ_S optimum).
# usage (from HSEMHS/): julia --project=. -t 4 scripts/m5_bifrost/make_qp_state.jl FIELD.h5 VSTART.h5 OUT.h5
include(joinpath(@__DIR__, "bif_setup.jl"))
fin, vstart, out = ARGS[1], ARGS[2], ARGS[3]
pb, x0, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
s = pb.s; nx, ny = b.nx, b.ny
Bx, By, Bz = h5open(fin) do f; read(f["Bx"]), read(f["By"]), read(f["Bz"]); end   # Bz[:, :, kk], kk = k + 1, k = 0:NZ
@assert size(Bx) == (nx, ny, NZ) && size(Bz) == (nx, ny, NZ + 1)
dx = [s.dxc[i] for i in 1:nx]; dy = [s.dyc[j] for j in 1:ny]
area = sum(dx) * sum(dy)
bbar = sum(dx[i] * dy[j] * Bz[i, j, 1] for i in 1:nx, j in 1:ny) / area
F = [Bz[i, j, 1] - bbar for i in 1:nx, j in 1:ny]
Fbar = [sum(dx[i] * F[i, j] for i in 1:nx) / sum(dx) for j in 1:ny]
Ax = zeros(nx, ny, NZ + 1); Ay = zeros(nx, ny, NZ + 1)              # [:, :, k+1] = level k + ½, k = 0:NZ
for j in 1:ny
    a = 0.0
    for i in 1:nx
        a += dx[i] * (F[i, j] - Fbar[j]); Ay[i, j, 1] = a           # Ay[nx] closes to ≈ 0 (periodic)
    end
end
ax = 0.0
for j in 1:ny
    global ax -= dy[j] * Fbar[j]; Ax[:, j, 1] .= ax
end
for k in 1:NZ
    Ay[:, :, k+1] .= Ay[:, :, k] .+ s.hz[k] .* Bx[:, :, k]
    Ax[:, :, k+1] .= Ax[:, :, k] .- s.hz[k] .* By[:, :, k]
end
x = deepcopy(x0)
h5open(vstart) do f; copyto!(parent(x[1]), read(f["x1"])); end
for k in 0:NZ, j in 1:ny, i in 1:nx
    x[2][i, j, k] = Ax[i, j, k+1] / pb.Aref; x[3][i, j, k] = Ay[i, j, k+1] / pb.Aref
end
fill!(parent(x[4]), 0.0); H.setscalar!(par, x[5], bbar)
H.exchange_halos!(par, collect(x))
w = H.SMHSWork(par)
function report(name, x)
    f, t, oob = H.objective!(pb, w, x)
    cf = Float64[]; tgt = name == "new" ? (Bx, By, Bz) : nothing; err = 0.0
    for q in 1:3, k in axes(w.B[q], 3), j in 1:ny, i in 1:nx
        isv = sqrt(pb.invsf2[q][i, j, k]); push!(cf, (w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * isv)
        tgt === nothing || (err = max(err, abs(w.B[q][i, j, k] - tgt[q][i, j, k - first(axes(w.B[q], 3)) + 1]) * isv))
    end
    cc = Float64[]
    for q in 1:3, k in 1:NZ, j in 1:ny, i in 1:nx
        push!(cc, w.dB[q][i, j, k] / pb.sig[q][i, j, k])
    end
    dv = maximum(abs(H.div_face(w.B, s, i, j, k)) for k in 1:NZ, j in 1:ny, i in 1:nx)
    rms, _ = H.layer_residual_stats(pb, w)
    Nf = length(cf); Kf = floor(Int, 0.01Nf)
    @printf("%-8s Φ %.6f | terms F %.4e NQ %.3e B %.6f S %.4e U %.1e | raw force %.4e max RMS %.3e | FACE max %.6fσ, >3σ %d (%.4f%%, K %d), #>5σ %d, mean c² %.6f | CENTRE max %.4fσ, >3σ %.4f%%, mean c² %.4f | max|div B| %.2e T/m%s\n",
            name, f, t..., t[1] / pb.lamF, maximum(rms), maximum(abs, cf), count(>(3), abs.(cf)), 100count(>(3), abs.(cf)) / Nf, Kf, count(>(5), abs.(cf)),
            sum(abs2, cf) / Nf, maximum(abs, cc), 100count(>(3), abs.(cc)) / length(cc), sum(abs2, cc) / length(cc), dv,
            tgt === nothing ? "" : @sprintf(" | reconstruction error max %.3e σ (closure of Ãy %.1e)", err, maximum(abs, Ay[nx, :, 1]) / pb.Aref))
    return err
end
xi = deepcopy(x0); copyto!(parent(xi[1]), parent(x[1]))             # init_A! fit (unconstrained data + λ_S), same v
report("init_A", xi)
err = report("new", x)
h5open(out, "w") do f
    for q in 1:5; f["x$q"] = parent(x[q]); end
    attributes(f)["source"] = "make_qp_state.jl: $(fin) (B), $(vstart) (v)"
    attributes(f)["reconstruction_error_sigma"] = err
end
println("saved $(out)")
