#=
Force balance of a τ-grid atmosphere (ln P, z on the log10 τ levels; T and B from the τ-grid input), the same
geometric formula for every run and for the Bifrost truth, independent of the solver's own residuals and weights:

  f_h = H (−∇_h ln P|_z + L_h/P),   f_z = H (−∂_z ln P − g ρ/P + L_z/P),   H = P/(ρ g),

i.e. the net force over ρg (∇_h at fixed height: ∂_x|_τ − ∂_x z|_τ ∂_z, centred differences, periodic in x and y;
∂_z centred in level index down each column; L = J×B evaluated at the given heights, as PoissonTau.lorentz_tau!).
Also writes the Lorentz force over ρg (|L|/ρg) as the scale of what the field can do.

usage: julia --project=. -t N scripts/ltaucut/force_diag.jl TAU_INPUT.nc STATE.h5 OUT.h5
STATE.h5 is a run output (tau_grid/z_m, tau_grid/Pgas) or the truth file (truth/z_m, truth/lnP).
output (h5py (τ, y, x)): fh (|f_h|), fz, lor (|L|/ρg), beta
=#
using HSEMHS
using HDF5
using MPI

const H = HSEMHS

function main(tfile, sfile, out)
    H.ensure_mpi()
    tab = H.load_table("cache/eos_table_fullz.h5"); g = 274.0
    f = h5open(tfile, "r")
    lt = read(f["log10tau500"]); x = read(f["x"]); y = read(f["y"])
    T = Float64.(read(f["temperature"])); Bx = Float64.(read(f["B_x"])); By = Float64.(read(f["B_y"])); Bz = Float64.(read(f["B_z"]))
    close(f)                                                  # Julia (τ, y, x)
    p(a) = permutedims(a, (3, 2, 1))                          # state files: Julia (x, y, τ) → (τ, y, x)
    z, u = h5open(sfile, "r") do s
        haskey(s, "truth") ? (p(read(s["truth/z_m"])), p(read(s["truth/lnP"]))) :
                             (p(read(s["tau_grid/z_m"])), log.(p(read(s["tau_grid/Pgas"]))))
    end
    nt, ny, nx = size(T); dx = x[2] - x[1]; dy = y[2] - y[1]
    par = H.slab_par(nx, ny, nt)
    D(o, a) = H.PoissonTau.dzcol!(o, a, z)
    zx = H.sddx!(similar(z), z, dx, par); zy = H.sddy!(similar(z), z, dy)
    function grad(a)                                          # (∂_x|_z, ∂_y|_z, ∂_z)
        fz = D(similar(a), a); fx = H.sddx!(similar(a), a, dx, par); fy = H.sddy!(similar(a), a, dy)
        @. fx -= zx * fz; @. fy -= zy * fz
        return fx, fy, fz
    end
    Bxx, Bxy, Bxz = grad(Bx); Byx, Byy, Byz = grad(By); Bzx, Bzy, Bzz = grad(Bz)
    ux, uy, uz = grad(u)
    fh = similar(z); fz = similar(z); lor = similar(z); beta = similar(z); nbad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(z)
        @inbounds begin
            r, _ = H.lnrho(tab, log(T[idx]), u[idx])
            isfinite(r) || (Threads.atomic_add!(nbad, 1))
            P = exp(u[idx]); ρ = exp(r); Hs = P / (ρ * g)
            Jx = (Bzy[idx] - Byz[idx]) / H.MU0; Jy = (Bxz[idx] - Bzx[idx]) / H.MU0; Jz = (Byx[idx] - Bxy[idx]) / H.MU0
            Lx = Jy * Bz[idx] - Jz * By[idx]; Ly = Jz * Bx[idx] - Jx * Bz[idx]; Lz = Jx * By[idx] - Jy * Bx[idx]
            fh[idx] = Hs * hypot(-ux[idx] + Lx / P, -uy[idx] + Ly / P)
            fz[idx] = Hs * (-uz[idx] - ρ * g / P + Lz / P)
            lor[idx] = sqrt(Lx^2 + Ly^2 + Lz^2) / (ρ * g)
            beta[idx] = 2 * H.MU0 * P / max(Bx[idx]^2 + By[idx]^2 + Bz[idx]^2, 1e-30)
        end
    end
    println("EOS outside the table at $(nbad[]) points")
    h5open(out, "w") do o
        o["fh"] = p(fh); o["fz"] = p(fz); o["lor"] = p(lor); o["beta"] = p(beta); o["logtau500"] = lt
        attributes(o)["state"] = abspath(sfile)
    end
    println("wrote $out")
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
