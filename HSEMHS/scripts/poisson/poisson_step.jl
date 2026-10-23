# Poisson-type pressure step (FIRTEZ-like, done as weighted least squares).
#
# T and B are fixed (column-HSE input); only P changes. Below z_cut the pressure minimises
#     Φ = Σ_centres w/S² (F_x² + F_y²) + Σ_z-faces h/S² F_z²,   F = −∇P + J×B − ρ(T, P) g ẑ,
# the same residual and discretisation as scripts/m5_bifrost/force_split.py (cell-centred B, centred periodic x, y
# differences, F_z on interior z-faces, S = layer median of ρ₀g of the starting state). F is almost linear in P (ρ ∝ P at
# fixed T apart from ionisation), so Gauss–Newton converges in a few steps. Each step solves the normal equations
# (a weighted Poisson equation with a buoyancy term) by PCG, preconditioned by the same operator with ∂ρ/∂P replaced by
# its layer median: FFT in x, y and a tridiagonal solve in z per mode. Above z_cut each column is integrated in
# hydrostatic equilibrium from the solved pressure at z_cut (FIRTEZ likewise drops the magnetic term above the
# lines' sensitivity). τ₅₀₀ from the EOS table with the exp_tail top closure, as m5h_fullres.jl.
#
# usage (from HSEMHS/): julia --project=. -t 64 scripts/poisson/poisson_step.jl COLHSE.h5 OUT.h5 [z_cut_Mm = 3.0]
#   ENV MHS_CROP = n : use only the first n × n columns (test runs; x, y treated as periodic regardless)
# OUT.h5 has the layout of an MHS checkpoint (/atmosphere, /coords), so force_split.py and m5h_isotau*.py read it.
using HSEMHS, HDF5, FFTW, Printf, Statistics, LinearAlgebra
const H = HSEMHS
const G0 = 274.0
const MU0 = 4e-7 * π
const TAB = normpath(joinpath(@__DIR__, "..", "..", "cache", "eos_table_fullz.h5"))

const t_start = time()
stamp() = @sprintf("[%6.1f s]", time() - t_start)
infile, outfile = ARGS[1], ARGS[2]
zcut = parse(Float64, get(ARGS, 3, "3.0")) * 1e6
crop = parse(Int, get(ENV, "MHS_CROP", "0"))
FFTW.set_num_threads(Threads.nthreads())

# ---- input: column HSE output, arrays (z, y, x), z top first ----
P0, T, Bx, By, Bz, z, x, y = h5open(infile, "r") do f
    rd(n) = Float64.(read(f[n]))
    rd("output/Pgas"), rd("input/temperature"), rd("input/B_x"), rd("input/B_y"), rd("input/B_z"),
        read(f["input/z"]), read(f["input/x"]), read(f["input/y"])
end
if crop > 0
    P0, T, Bx, By, Bz = (A[:, 1:crop, 1:crop] for A in (P0, T, Bx, By, Bz)); x = x[1:crop]; y = y[1:crop]
end
nz, ny, nx = size(P0)
@assert length(z) == nz && z[1] > z[end] "expected (z, y, x) arrays with z top first"
dx, dy = x[2] - x[1], y[2] - y[1]
tab = H.load_table(TAB)
lnT = log.(T)
println(stamp(), " input $infile: $(nz)×$(ny)×$(nx), z $(round(z[1]/1e6, digits = 3)) … $(round(z[end]/1e6, digits = 3)) Mm, ",
        "dx $(round(dx/1e3, digits = 2)) km, threads $(Threads.nthreads()), z_cut $(zcut/1e6) Mm, EOS $TAB")

# ---- derivatives: centred periodic in x (dim 3), y (dim 2); np.gradient-type in z (dim 1) ----
function ddx!(o, f, d)
    nz, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        ip = i == nx ? 1 : i + 1; im = i == 1 ? nx : i - 1
        @inbounds for j in 1:ny, k in 1:nz
            o[k, j, i] = (f[k, j, ip] - f[k, j, im]) / (2d)
        end
    end
    return o
end
function ddy!(o, f, d)
    nz, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            jp = j == ny ? 1 : j + 1; jm = j == 1 ? ny : j - 1
            for k in 1:nz
                o[k, j, i] = (f[k, jp, i] - f[k, jm, i]) / (2d)
            end
        end
    end
    return o
end
function ddz!(o, f, z)
    nz, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            o[1, j, i] = (f[2, j, i] - f[1, j, i]) / (z[2] - z[1])
            o[nz, j, i] = (f[nz, j, i] - f[nz-1, j, i]) / (z[nz] - z[nz-1])
            for k in 2:nz-1
                d1 = z[k] - z[k-1]; d2 = z[k+1] - z[k]
                o[k, j, i] = -d2 / (d1 * (d1 + d2)) * f[k-1, j, i] + (d2 - d1) / (d1 * d2) * f[k, j, i] +
                             d1 / (d2 * (d1 + d2)) * f[k+1, j, i]
            end
        end
    end
    return o
end

# ---- Lorentz force (fixed) ----
Lx, Ly, Lz = similar(P0), similar(P0), similar(P0)
let a = similar(P0), b = similar(P0), Jc = similar(P0)
    ddy!(a, Bz, dy); ddz!(b, By, z); Jx = (a .- b) ./ MU0
    ddz!(a, Bx, z); ddx!(b, Bz, dx); Jy = (a .- b) ./ MU0
    ddx!(a, By, dx); ddy!(b, Bx, dy); Jc .= (a .- b) ./ MU0
    @. Lx = Jy * Bz - Jc * By; @. Ly = Jc * Bx - Jx * Bz; @. Lz = Jx * By - Jy * Bx
end
println(stamp(), " Lorentz force computed")

# ---- solved region k = kc … nz (z ≤ z_cut) ----
kc = findfirst(<=(zcut), z)
R = kc:nz; n = length(R)
zr = z[R]
h = zr[1:end-1] .- zr[2:end]                                     # centre spacing
zf = vcat(zr[1] + h[1] / 2, (zr[1:end-1] .+ zr[2:end]) ./ 2, zr[end] - h[end] / 2)
w = zf[1:end-1] .- zf[2:end]                                     # cell widths
P = P0[R, :, :]; lT = lnT[R, :, :]; lx = Lx[R, :, :]; ly = Ly[R, :, :]; lz = Lz[R, :, :]
rho = similar(P); rhoP = similar(P)

function eos!(rho, rhoP, P, lT)
    bad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(P)
        lr, d, _ = H.lnrho_dY(tab, lT[idx], log(P[idx]))
        isfinite(lr) || Threads.atomic_add!(bad, 1)
        r = exp(lr); rho[idx] = r; rhoP[idx] = r * d / P[idx]
    end
    return bad[]
end
eos!(rho, rhoP, P, lT)
S = [median(@view rho[k, :, :]) * G0 for k in 1:n]
Sf = sqrt.(S[1:end-1] .* S[2:end])
wc = w ./ S .^ 2; wf = h ./ Sf .^ 2
Ncol = ny * nx
Wtot = (2 * sum(w) + sum(h)) * Ncol
κ = 2π / (nx * dx)
ereg = 1e-2 * κ^2 .* wc                                          # damps the modes centred differences cannot see
Aanc = 1e2 * wf[end] / h[end]^2 / Ncol                          # bottom-plane mean pressure kept fixed

Fx, Fy, Fz = similar(P), similar(P), zeros(n - 1, ny, nx)
function resid!(Fx, Fy, Fz, P, rho)
    ddx!(Fx, P, dx); ddy!(Fy, P, dy)
    @. Fx = lx - Fx; @. Fy = ly - Fy
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, f in 1:n-1
            Fz[f, j, i] = -(P[f, j, i] - P[f+1, j, i]) / h[f] - G0 * (rho[f, j, i] + rho[f+1, j, i]) / 2 +
                          (lz[f, j, i] + lz[f+1, j, i]) / 2
        end
    end
end
function phi(Fx, Fy, Fz)
    s = Threads.Atomic{Float64}(0.0)
    Threads.@threads for i in 1:nx
        a = 0.0
        @inbounds for j in 1:ny
            for k in 1:n
                a += wc[k] * (Fx[k, j, i]^2 + Fy[k, j, i]^2)
            end
            for f in 1:n-1
                a += wf[f] * Fz[f, j, i]^2
            end
        end
        Threads.atomic_add!(s, a)
    end
    return s[]
end

# Jᵀ W applied to a residual-shaped (Rx, Ry, Rz) -> P-shaped out (uses rhoP of the current linearisation point)
tmpx, tmpy = similar(P), similar(P)
function JtW!(out, Rx, Ry, Rz)
    @. tmpx = wc * Rx; @. tmpy = wc * Ry                         # wc broadcasts along dim 1 (z)
    ddx!(out, tmpx, dx); ddy!(tmpx, tmpy, dy); out .+= tmpx      # (−D)ᵀ = D for centred periodic differences
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, f in 1:n-1
            r = wf[f] * Rz[f, j, i]
            out[f, j, i] -= (1 / h[f] + G0 * rhoP[f, j, i] / 2) * r
            out[f+1, j, i] -= (-1 / h[f] + G0 * rhoP[f+1, j, i] / 2) * r
        end
    end
    return out
end
Jx_, Jy_, Jz_ = similar(P), similar(P), zeros(n - 1, ny, nx)
function Aop!(out, v)
    ddx!(Jx_, v, dx); ddy!(Jy_, v, dy); Jx_ .*= -1; Jy_ .*= -1
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, f in 1:n-1
            Jz_[f, j, i] = -((1 / h[f] + G0 * rhoP[f, j, i] / 2) * v[f, j, i] + (-1 / h[f] + G0 * rhoP[f+1, j, i] / 2) * v[f+1, j, i])
        end
    end
    JtW!(out, Jx_, Jy_, Jz_)
    @. out += ereg * v
    sb = sum(@view v[n, :, :])
    @views out[n, :, :] .+= Aanc * sb
    return out
end

# ---- preconditioner: same operator with ∂ρ/∂P → layer median c_k; rfft in (y, x), tridiagonal in z per mode ----
nyh = ny ÷ 2 + 1
sy2 = [sin(2π * (j - 1) / ny)^2 / dy^2 for j in 1:nyh]
sx2 = [sin(2π * fftfreq(nx)[i])^2 / dx^2 for i in 1:nx]
Vh = zeros(ComplexF64, n, nyh, nx)
pf = plan_rfft(P, (2, 3)); pb = plan_irfft(Vh, ny, (2, 3))
cdiag = zeros(n); coff = zeros(n - 1)                             # vertical part (set by setprec!)
function setprec!()
    c = [median(@view rhoP[k, :, :]) for k in 1:n]
    fill!(cdiag, 0.0)
    for f in 1:n-1
        a = 1 / h[f] + G0 * c[f] / 2; b = -1 / h[f] + G0 * c[f+1] / 2
        cdiag[f] += wf[f] * a^2; cdiag[f+1] += wf[f] * b^2; coff[f] = wf[f] * a * b
    end
end
function Minv!(out, r)
    mul!(Vh, pf, r)
    Threads.@threads for i in 1:nx
        cp = zeros(n); dp = zeros(ComplexF64, n)
        @inbounds for j in 1:nyh
            s2 = sx2[i] + sy2[j]
            for k in 1:n
                d = cdiag[k] + wc[k] * s2 + ereg[k]
                (i == 1 && j == 1 && k == n) && (d += Aanc * Ncol)
                if k == 1
                    cp[1] = coff[1] / d; dp[1] = Vh[1, j, i] / d
                else
                    den = d - coff[k-1] * cp[k-1]
                    cp[k] = k < n ? coff[k] / den : 0.0
                    dp[k] = (Vh[k, j, i] - coff[k-1] * dp[k-1]) / den
                end
            end
            Vh[n, j, i] = dp[n]
            for k in n-1:-1:1
                Vh[k, j, i] = dp[k] - cp[k] * Vh[k+1, j, i]
            end
        end
    end
    mul!(out, pb, Vh)
    return out
end

function pcg!(xs, b; tol = 1e-3, maxit = 60)
    fill!(xs, 0.0)
    r = copy(b); zv = similar(b); p = similar(b); Ap = similar(b)
    Minv!(zv, r); p .= zv; rz = dot(r, zv); nb = norm(b); it = 0; rel = 1.0
    for outer it in 1:maxit
        Aop!(Ap, p)
        α = rz / dot(p, Ap)
        @. xs += α * p; @. r -= α * Ap
        rel = norm(r) / nb
        rel < tol && break
        Minv!(zv, r); rz2 = dot(r, zv); β = rz2 / rz; rz = rz2
        @. p = zv + β * p
    end
    return it, rel
end

# ---- Gauss–Newton ----
resid!(Fx, Fy, Fz, P, rho)
Φ = phi(Fx, Fy, Fz); Φ0 = Φ
@printf("%s start (column HSE): Φ %.6e, RMS |F|/S %.4e over %d layers (z %.3f … %.3f Mm)\n", stamp(), Φ, sqrt(Φ / Wtot), n,
        zr[1] / 1e6, zr[end] / 1e6)
δ = similar(P); rhs = similar(P); Ptry = similar(P); rho_t = similar(P); rhoP_t = similar(P)
maxouter = parse(Int, get(ENV, "MHS_POISSON_OUTER", "20"))
for outer in 1:maxouter
    t0 = time()
    setprec!()
    JtW!(rhs, Fx, Fy, Fz); rhs .*= -1
    its, rel = pcg!(δ, rhs)
    α = 1.0; Φt = Inf; nclamp = 0; bad = 0
    for ls in 1:12
        nclamp = 0
        @inbounds for idx in eachindex(P)
            v = P[idx] + α * δ[idx]; lo = 0.1 * P[idx]
            v < lo && (v = lo; nclamp += 1)
            Ptry[idx] = v
        end
        bad = eos!(rho_t, rhoP_t, Ptry, lT)
        resid!(Fx, Fy, Fz, Ptry, rho_t)
        Φt = bad == 0 ? phi(Fx, Fy, Fz) : Inf
        Φt < Φ && break
        α /= 2
    end
    if !(Φt < Φ)
        resid!(Fx, Fy, Fz, P, rho)
        @printf("%s outer %2d: no decrease along the step (CG %d its, rel %.1e); stopping\n", stamp(), outer, its, rel)
        break
    end
    rel_dec = (Φ - Φt) / Φ
    P .= Ptry; rho .= rho_t; rhoP .= rhoP_t; Φ = Φt
    @printf("%s outer %2d: Φ %.6e (Φ/Φ₀ %.3e, ΔΦ/Φ %.2e), RMS |F|/S %.4e | CG %d its (rel %.1e), α %.3g, clamped %d, %.1f s\n",
            stamp(), outer, Φ, Φ / Φ0, rel_dec, sqrt(Φ / Wtot), its, rel, α, nclamp, time() - t0)
    rel_dec < 1e-4 && break
end
t_solve = time() - t_start

# ---- full column: solved region below z_cut, hydrostatic integration above it ----
Pfull = copy(P0); Pfull[R, :, :] .= P
Threads.@threads for i in 1:nx
    @inbounds for j in 1:ny
        for k in kc-1:-1:1
            Δz = z[k] - z[k+1]
            ub = log(Pfull[k+1, j, i]); lrb, _, _ = H.lnrho_dY(tab, lnT[k+1, j, i], ub); qb = exp(lrb - ub)
            u = ub - Δz * G0 * qb
            for _ in 1:30
                lr, d, _ = H.lnrho_dY(tab, lnT[k, j, i], u); q = exp(lr - u)
                fu = u - ub + Δz * G0 / 2 * (q + qb); du = fu / (1 + Δz * G0 / 2 * q * (d - 1))
                u -= du
                abs(du) < 1e-12 && break
            end
            Pfull[k, j, i] = exp(u)
        end
    end
end
rhofull = similar(Pfull); tau = similar(Pfull); zt1 = zeros(ny, nx)
Threads.@threads for i in 1:nx
    chi = zeros(nz); tc = zeros(nz)
    @inbounds for j in 1:ny
        for k in 1:nz
            r, _, c, _ = H.state(tab, lnT[k, j, i], log(Pfull[k, j, i]))
            rhofull[k, j, i] = exp(r); chi[k] = exp(c)
        end
        H.tau_column!(tc, chi, z; top = :exp_tail, H_max = 5.0e6, mu = 1.0)
        tau[:, j, i] .= tc; zt1[j, i] = H.z_tau1(tc, z)
    end
end
lt_top = log10.(tau[1, :, :])
zt1_in = h5open(f -> read(f["diagnostics/z_tau500_1"]), infile)
crop > 0 && (zt1_in = zt1_in[1:crop, 1:crop])
@printf("%s done: solve %.1f s; ln(P/P_colhse) below z_cut p1/p50/p99 %.3f / %.3f / %.3f; z(τ=1) median %.1f km (column HSE %.1f km); log10 τ500 at z_top min/median/max %.2f / %.2f / %.2f\n",
        stamp(), t_solve, quantile(vec(log.(P ./ P0[R, :, :])), [0.01, 0.5, 0.99])..., median(filter(isfinite, zt1)) / 1e3,
        median(filter(isfinite, zt1_in)) / 1e3, minimum(lt_top), median(lt_top), maximum(lt_top))

# ---- output in the MHS checkpoint layout: Julia (x, y, z) arrays ----
xyz(A) = permutedims(A, (3, 2, 1))
h5open(outfile, "w") do f
    a = create_group(f, "atmosphere"); c = create_group(f, "coords")
    a["Pgas"] = xyz(Pfull); a["rho"] = xyz(rhofull); a["T"] = xyz(T); a["tau500"] = xyz(tau)
    a["Bx_centre"] = xyz(Bx); a["By_centre"] = xyz(By); a["Bz_centre"] = xyz(Bz)
    a["z_tau1_m"] = permutedims(zt1, (2, 1))
    c["x_m"] = x; c["y_m"] = y; c["z_m"] = z
    attributes(f)["accepted"] = 0
    attributes(f)["kind"] = "poisson_step"
    attributes(f)["input"] = infile
    attributes(f)["z_cut_m"] = zcut
    attributes(f)["eos_table"] = TAB
    attributes(f)["phi_start"] = Φ0; attributes(f)["phi_end"] = Φ
    attributes(f)["solve_seconds"] = t_solve
end
println(stamp(), " wrote $outfile")
