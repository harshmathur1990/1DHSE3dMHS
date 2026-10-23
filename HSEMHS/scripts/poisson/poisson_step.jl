# Poisson-type pressure step (FIRTEZ-like, done as a weighted least-squares problem in u = ln P).
#
# T and B are fixed (column-HSE input); only the pressure changes. Below z_cut, u = ln P minimises
#     Φ = Σ_centres w H̄² (R_x² + R_y²) + Σ_z-faces h H̄² R_z²,     R = F/P = −∇u − ẑ/H(T, P) + J×B/P,
# with H = P/(ρ g) and H̄ the layer median of H in the starting state (fixed), so R H̄ ≈ F/(ρ g) layer by layer. In u the
# residual is measured against the local pressure: lowering P cannot reduce it (the P-space form with a fixed scale S
# rewards collapsing P where column HSE leaves |∇ₕP| ≫ ρg), and P stays positive. Discretisation as
# scripts/m5_bifrost/force_split.py: cell-centred B, centred periodic x, y differences, R_z on interior z-faces.
# Gauss–Newton; each step solves the normal equations (a weighted Poisson equation in δu plus local terms) by PCG,
# preconditioned by the Laplacian part with layer-median coefficients: FFT in x, y and a tridiagonal solve in z per mode.
# Steps are capped at |δu| ≤ 2 per node and backtracked on Φ. The bottom-plane mean of u is held fixed.
# Above z_cut each column is integrated in hydrostatic equilibrium from the solved pressure at z_cut (FIRTEZ likewise
# drops the magnetic term above the lines' sensitivity). τ₅₀₀ from the EOS table with the exp_tail top closure.
#
# usage (from HSEMHS/): julia --project=. -t 64 scripts/poisson/poisson_step.jl COLHSE.h5 OUT.h5 [z_cut_Mm = 3.0]
#   ENV MHS_CROP = n : only the first n × n columns (test runs; x, y treated as periodic regardless)
#   ENV MHS_POISSON_OUTER = maximum Gauss–Newton steps (default 30); stop when ΔΦ/Φ < 1e-4
# OUT.h5 has the layout of an MHS checkpoint (/atmosphere, /coords), so force_split.py and m5h_isotau*.py read it.
using HSEMHS, HDF5, FFTW, Printf, Statistics, LinearAlgebra
const H = HSEMHS
const G0 = 274.0
const MU0 = 4e-7 * π
const TAB = normpath(joinpath(@__DIR__, "..", "..", "cache", "eos_table_fullz.h5"))
const DU_CAP = 2.0

const t_start = time()
stamp() = @sprintf("[%6.1f s]", time() - t_start)

# ---- derivatives on (z, y, x) arrays: centred periodic in x (dim 3), y (dim 2); np.gradient-type in z (dim 1) ----
function ddx!(o, f, d)
    mz, my, mx = size(f)
    Threads.@threads for i in 1:mx
        ip = i == mx ? 1 : i + 1; im = i == 1 ? mx : i - 1
        @inbounds for j in 1:my, k in 1:mz
            o[k, j, i] = (f[k, j, ip] - f[k, j, im]) / (2d)
        end
    end
    return o
end
function ddy!(o, f, d)
    mz, my, mx = size(f)
    Threads.@threads for i in 1:mx
        @inbounds for j in 1:my
            jp = j == my ? 1 : j + 1; jm = j == 1 ? my : j - 1
            for k in 1:mz
                o[k, j, i] = (f[k, jp, i] - f[k, jm, i]) / (2d)
            end
        end
    end
    return o
end
function ddz!(o, f, z)
    mz, my, mx = size(f)
    Threads.@threads for i in 1:mx
        @inbounds for j in 1:my
            o[1, j, i] = (f[2, j, i] - f[1, j, i]) / (z[2] - z[1])
            o[mz, j, i] = (f[mz, j, i] - f[mz-1, j, i]) / (z[mz] - z[mz-1])
            for k in 2:mz-1
                d1 = z[k] - z[k-1]; d2 = z[k+1] - z[k]
                o[k, j, i] = -d2 / (d1 * (d1 + d2)) * f[k-1, j, i] + (d2 - d1) / (d1 * d2) * f[k, j, i] +
                             d1 / (d2 * (d1 + d2)) * f[k+1, j, i]
            end
        end
    end
    return o
end

# ---- problem state on the solved region (n layers) ----
struct Prob
    tab::H.EOSTable
    n::Int; ny::Int; nx::Int
    dx::Float64; dy::Float64
    h::Vector{Float64}                 # centre spacing (n − 1)
    wc::Vector{Float64}                # centre weights w H̄² (n)
    wf::Vector{Float64}                # face weights h H̄_f² (n − 1)
    ereg::Vector{Float64}              # δu damping for modes centred differences cannot see
    Aanc::Float64                      # bottom-plane mean anchor weight
    lT::Array{Float64,3}; lx::Array{Float64,3}; ly::Array{Float64,3}; lz::Array{Float64,3}
    q::Array{Float64,3}                # ρ/P at the linearisation point
    qd::Array{Float64,3}               # ∂(ρ/P)/∂u = q (∂lnρ/∂lnP − 1)
    ex::Array{Float64,3}               # e^{−u}
    tx::Array{Float64,3}; ty::Array{Float64,3}; jx::Array{Float64,3}; jy::Array{Float64,3}; jz::Array{Float64,3}
    sx2::Vector{Float64}; sy2::Vector{Float64}
    cdiag::Vector{Float64}; coff::Vector{Float64}
    Vh::Array{ComplexF64,3}
    pf::Any; pb::Any
end

"""ρ/P, its u-derivative and e^{−u} at u; returns the number of nodes outside the EOS table."""
function eos!(pr::Prob, q, qd, ex, u)
    bad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(u)
        lr, d, _ = H.lnrho_dY(pr.tab, pr.lT[idx], u[idx])
        isfinite(lr) || Threads.atomic_add!(bad, 1)
        qq = exp(lr - u[idx]); q[idx] = qq; qd[idx] = qq * (d - 1); ex[idx] = exp(-u[idx])
    end
    return bad[]
end

"""R = −∇u − ẑ g ρ/P + J×B/P (R_x, R_y at centres, R_z on interior faces)."""
function resid!(pr::Prob, Rx, Ry, Rz, u, q, ex)
    ddx!(Rx, u, pr.dx); ddy!(Ry, u, pr.dy)
    @. Rx = pr.lx * ex - Rx; @. Ry = pr.ly * ex - Ry
    h = pr.h; lz = pr.lz
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, f in 1:pr.n-1
            Rz[f, j, i] = -(u[f, j, i] - u[f+1, j, i]) / h[f] - G0 * (q[f, j, i] + q[f+1, j, i]) / 2 +
                          (lz[f, j, i] * ex[f, j, i] + lz[f+1, j, i] * ex[f+1, j, i]) / 2
        end
    end
end

function phi(pr::Prob, Rx, Ry, Rz)
    s = Threads.Atomic{Float64}(0.0)
    wc = pr.wc; wf = pr.wf
    Threads.@threads for i in 1:pr.nx
        a = 0.0
        @inbounds for j in 1:pr.ny
            for k in 1:pr.n
                a += wc[k] * (Rx[k, j, i]^2 + Ry[k, j, i]^2)
            end
            for f in 1:pr.n-1
                a += wf[f] * Rz[f, j, i]^2
            end
        end
        Threads.atomic_add!(s, a)
    end
    return s[]
end

# Linearisation: δR_x = −D_x δu − (L_x e^{−u}) δu, δR_y likewise,
# δR_z,f = −[a_f δu_k + b_f δu_k+1], a_f = 1/h + (g qd_k + L_z,k e^{−u_k})/2, b_f = −1/h + (g qd_k+1 + L_z,k+1 e^{−u_k+1})/2.
@inline coef_a(pr, f, j, i) = 1 / pr.h[f] + (G0 * pr.qd[f, j, i] + pr.lz[f, j, i] * pr.ex[f, j, i]) / 2
@inline coef_b(pr, f, j, i) = -1 / pr.h[f] + (G0 * pr.qd[f+1, j, i] + pr.lz[f+1, j, i] * pr.ex[f+1, j, i]) / 2

"""out = Jᵀ W (Rx, Ry, Rz)."""
function JtW!(pr::Prob, out, Rx, Ry, Rz)
    wc = pr.wc; wf = pr.wf
    @. pr.tx = wc * Rx; @. pr.ty = wc * Ry
    ddx!(out, pr.tx, pr.dx); ddy!(pr.jx, pr.ty, pr.dy); out .+= pr.jx            # (−D)ᵀ = D
    @. out -= pr.lx * pr.ex * pr.tx + pr.ly * pr.ex * pr.ty
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, f in 1:pr.n-1
            r = wf[f] * Rz[f, j, i]
            out[f, j, i] -= coef_a(pr, f, j, i) * r
            out[f+1, j, i] -= coef_b(pr, f, j, i) * r
        end
    end
    return out
end

"""out = (Jᵀ W J + damping + anchor) v."""
function Aop!(pr::Prob, out, v)
    ddx!(pr.jx, v, pr.dx); ddy!(pr.jy, v, pr.dy)
    @. pr.jx = -pr.jx - pr.lx * pr.ex * v; @. pr.jy = -pr.jy - pr.ly * pr.ex * v
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, f in 1:pr.n-1
            pr.jz[f, j, i] = -(coef_a(pr, f, j, i) * v[f, j, i] + coef_b(pr, f, j, i) * v[f+1, j, i])
        end
    end
    JtW!(pr, out, pr.jx, pr.jy, pr.jz)
    @. out += pr.ereg * v
    sb = sum(@view v[pr.n, :, :])
    @views out[pr.n, :, :] .+= pr.Aanc * sb
    return out
end

"""Vertical part of the preconditioner from layer medians of the current coefficients."""
function setprec!(pr::Prob)
    n = pr.n
    fill!(pr.cdiag, 0.0)
    for f in 1:n-1
        a = median(coef_a(pr, f, j, i) for j in 1:pr.ny, i in 1:pr.nx)
        b = median(coef_b(pr, f, j, i) for j in 1:pr.ny, i in 1:pr.nx)
        pr.cdiag[f] += pr.wf[f] * a^2; pr.cdiag[f+1] += pr.wf[f] * b^2; pr.coff[f] = pr.wf[f] * a * b
    end
end

function Minv!(pr::Prob, out, r)
    n = pr.n; Vh = pr.Vh; nyh = size(Vh, 2); Ncol = pr.ny * pr.nx
    mul!(Vh, pr.pf, r)
    Threads.@threads for i in 1:pr.nx
        cp = zeros(n); dp = zeros(ComplexF64, n)
        @inbounds for j in 1:nyh
            s2 = pr.sx2[i] + pr.sy2[j]
            for k in 1:n
                d = pr.cdiag[k] + pr.wc[k] * s2 + pr.ereg[k]
                (i == 1 && j == 1 && k == n) && (d += pr.Aanc * Ncol)
                if k == 1
                    cp[1] = pr.coff[1] / d; dp[1] = Vh[1, j, i] / d
                else
                    den = d - pr.coff[k-1] * cp[k-1]
                    cp[k] = k < n ? pr.coff[k] / den : 0.0
                    dp[k] = (Vh[k, j, i] - pr.coff[k-1] * dp[k-1]) / den
                end
            end
            Vh[n, j, i] = dp[n]
            for k in n-1:-1:1
                Vh[k, j, i] = dp[k] - cp[k] * Vh[k+1, j, i]
            end
        end
    end
    mul!(out, pr.pb, Vh)
    return out
end

function pcg!(pr::Prob, xs, b; tol = 1e-3, maxit = 80)
    fill!(xs, 0.0)
    r = copy(b); zv = similar(b); p = similar(b); Ap = similar(b)
    Minv!(pr, zv, r); p .= zv; rz = dot(r, zv); nb = norm(b); it = 0; rel = 1.0
    for outer it in 1:maxit
        Aop!(pr, Ap, p)
        αc = rz / dot(p, Ap)
        @. xs += αc * p; @. r -= αc * Ap
        rel = norm(r) / nb
        rel < tol && break
        Minv!(pr, zv, r); rz2 = dot(r, zv); βc = rz2 / rz; rz = rz2
        @. p = zv + βc * p
    end
    return it, rel
end

function main()
    infile, outfile = ARGS[1], ARGS[2]
    zcut = parse(Float64, get(ARGS, 3, "3.0")) * 1e6
    crop = parse(Int, get(ENV, "MHS_CROP", "0"))
    maxouter = parse(Int, get(ENV, "MHS_POISSON_OUTER", "30"))
    FFTW.set_num_threads(1)              # FFTW's threaded execution segfaults under Julia 1.13 (spawn_apply); 0.4 s per pair anyway

    # ---- input: column HSE output, arrays (z, y, x), z top first ----
    P0, T, Bx, By, Bz, z, x, y = h5open(infile, "r") do f
        c = crop > 0 ? (1:crop) : Colon()
        rd(nm) = Float64.(read(f[nm])[:, c, c])
        rd("output/Pgas"), rd("input/temperature"), rd("input/B_x"), rd("input/B_y"), rd("input/B_z"),
            read(f["input/z"]), read(f["input/x"])[c], read(f["input/y"])[c]
    end
    nz, ny, nx = size(P0)
    @assert length(z) == nz && z[1] > z[end] "expected (z, y, x) arrays with z top first"
    dx, dy = x[2] - x[1], y[2] - y[1]
    tab = H.load_table(TAB)
    lnT = log.(T)
    println(stamp(), " input $infile: $(nz)×$(ny)×$(nx), z $(round(z[1]/1e6, digits = 3)) … $(round(z[end]/1e6, digits = 3)) Mm, ",
            "dx $(round(dx/1e3, digits = 2)) km, threads $(Threads.nthreads()), z_cut $(zcut/1e6) Mm, EOS $TAB")

    # ---- Lorentz force (fixed) ----
    Lx, Ly, Lz = similar(P0), similar(P0), similar(P0)
    let a = similar(P0), b = similar(P0)
        ddy!(a, Bz, dy); ddz!(b, By, z); Jx = (a .- b) ./ MU0
        ddz!(a, Bx, z); ddx!(b, Bz, dx); Jy = (a .- b) ./ MU0
        ddx!(a, By, dx); ddy!(b, Bx, dy); Jz = (a .- b) ./ MU0
        @. Lx = Jy * Bz - Jz * By; @. Ly = Jz * Bx - Jx * Bz; @. Lz = Jx * By - Jy * Bx
    end
    if get(ENV, "MHS_NO_LORENTZ", "0") == "1"                    # diagnostic: pure hydrostatics with the same T
        fill!(Lx, 0.0); fill!(Ly, 0.0); fill!(Lz, 0.0)
    end
    println(stamp(), " Lorentz force computed")

    # ---- solved region k = kc … nz (z ≤ z_cut) ----
    kc = findfirst(<=(zcut), z)
    R = kc:nz; n = length(R)
    zr = z[R]
    h = zr[1:end-1] .- zr[2:end]
    zf = vcat(zr[1] + h[1] / 2, (zr[1:end-1] .+ zr[2:end]) ./ 2, zr[end] - h[end] / 2)
    w = zf[1:end-1] .- zf[2:end]
    u0 = log.(P0[R, :, :]); u = copy(u0)
    lT = lnT[R, :, :]
    q = similar(u); qd = similar(u); ex = similar(u)
    Hl = let qq = similar(u)
        Threads.@threads for idx in eachindex(u)
            lr, _, _ = H.lnrho_dY(tab, lT[idx], u[idx]); qq[idx] = exp(lr - u[idx])
        end
        [median(1 ./ (G0 .* @view qq[k, :, :])) for k in 1:n]          # layer median scale height H̄
    end
    Hf = sqrt.(Hl[1:end-1] .* Hl[2:end])
    wc = w .* Hl .^ 2; wf = h .* Hf .^ 2
    Ncol = ny * nx
    Wtot = (2 * sum(w) + sum(h)) * Ncol
    κ = 2π / (nx * dx)
    ereg = 1e-2 * κ^2 .* wc
    Aanc = 1e2 * wf[end] / h[end]^2 / Ncol
    nyh = ny ÷ 2 + 1
    sy2 = [sin(2π * (j - 1) / ny)^2 / dy^2 for j in 1:nyh]
    sx2 = [sin(2π * fftfreq(nx)[i])^2 / dx^2 for i in 1:nx]
    Vh = zeros(ComplexF64, n, nyh, nx)
    pr = Prob(tab, n, ny, nx, dx, dy, h, wc, wf, ereg, Aanc, lT, Lx[R, :, :], Ly[R, :, :], Lz[R, :, :], q, qd, ex,
              similar(u), similar(u), similar(u), similar(u), zeros(n - 1, ny, nx), sx2, sy2, zeros(n), zeros(n - 1), Vh,
              plan_rfft(u, (2, 3); flags = FFTW.ESTIMATE | FFTW.UNALIGNED),
              plan_irfft(Vh, ny, (2, 3); flags = FFTW.ESTIMATE | FFTW.UNALIGNED))   # applied to other arrays than planned

    Rx, Ry, Rz = similar(u), similar(u), zeros(n - 1, ny, nx)
    eos!(pr, q, qd, ex, u); resid!(pr, Rx, Ry, Rz, u, q, ex)
    Φ = phi(pr, Rx, Ry, Rz); Φ0 = Φ
    @printf("%s start (column HSE): Φ %.6e, RMS |R| H̄ %.4e over %d layers (z %.3f … %.3f Mm)\n", stamp(), Φ, sqrt(Φ / Wtot), n,
            zr[1] / 1e6, zr[end] / 1e6)
    δ = similar(u); rhs = similar(u); ut = similar(u); qt = similar(u); qdt = similar(u); ext = similar(u)
    for outer in 1:maxouter
        t0 = time()
        setprec!(pr)
        JtW!(pr, rhs, Rx, Ry, Rz); rhs .*= -1
        its, rel = pcg!(pr, δ, rhs)
        ncap = count(v -> abs(v) > DU_CAP, δ)
        @. δ = clamp(δ, -DU_CAP, DU_CAP)
        α = 1.0; Φt = Inf
        for ls in 1:12
            @. ut = u + α * δ
            bad = eos!(pr, qt, qdt, ext, ut)
            resid!(pr, Rx, Ry, Rz, ut, qt, ext)
            Φt = bad == 0 ? phi(pr, Rx, Ry, Rz) : Inf
            Φt < Φ && break
            α /= 2
        end
        if !(Φt < Φ)
            resid!(pr, Rx, Ry, Rz, u, q, ex)
            @printf("%s outer %2d: no decrease along the step (CG %d its, rel %.1e); stopping\n", stamp(), outer, its, rel)
            break
        end
        rel_dec = (Φ - Φt) / Φ
        u .= ut; q .= qt; qd .= qdt; ex .= ext; Φ = Φt
        @printf("%s outer %2d: Φ %.6e (Φ/Φ₀ %.3e, ΔΦ/Φ %.2e), RMS |R| H̄ %.4e | CG %d its (rel %.1e), α %.3g, capped %d, max|αδu| %.3f, %.1f s\n",
                stamp(), outer, Φ, Φ / Φ0, rel_dec, sqrt(Φ / Wtot), its, rel, α, ncap, α * maximum(abs, δ), time() - t0)
        rel_dec < 1e-4 && break
    end
    t_solve = time() - t_start

    # ---- full column: solved region below z_cut, hydrostatic integration above it ----
    Pfull = copy(P0); Pfull[R, :, :] .= exp.(u)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for k in kc-1:-1:1
                Δz = z[k] - z[k+1]
                ub = log(Pfull[k+1, j, i]); lrb, _, _ = H.lnrho_dY(tab, lnT[k+1, j, i], ub); qb = exp(lrb - ub)
                uk = ub - Δz * G0 * qb
                for _ in 1:30
                    lr, d, _ = H.lnrho_dY(tab, lnT[k, j, i], uk); qk = exp(lr - uk)
                    fu = uk - ub + Δz * G0 / 2 * (qk + qb); du = fu / (1 + Δz * G0 / 2 * qk * (d - 1))
                    uk -= du
                    abs(du) < 1e-12 && break
                end
                Pfull[k, j, i] = exp(uk)
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
            for k in 1:nz
                tau[k, j, i] = tc[k]
            end
            zt1[j, i] = H.z_tau1(tc, z)
        end
    end
    lt_top = log10.(tau[1, :, :])
    zt1_in = h5open(f -> read(f["diagnostics/z_tau500_1"]), infile)
    crop > 0 && (zt1_in = zt1_in[1:crop, 1:crop])
    @printf("%s done: solve %.1f s; ln(P/P_colhse) below z_cut p1/p50/p99 %.3f / %.3f / %.3f; z(τ=1) median %.1f km (column HSE %.1f km); log10 τ500 at z_top min/median/max %.2f / %.2f / %.2f\n",
            stamp(), t_solve, quantile(vec(u .- u0), [0.01, 0.5, 0.99])..., median(filter(isfinite, zt1)) / 1e3,
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
end

main()
