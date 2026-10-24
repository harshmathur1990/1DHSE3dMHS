#!/usr/bin/env julia
#=
Poisson step directly on the log10 τ500 grid: one Gauss–Newton solve, no height grid and no outer fixed point.

Unknowns per column: ln P on the levels below the cut level k_c (log10 τ ≥ MHS_LTCUT) and the height c of that level.
The heights follow from the pressure, z_k = c + Σ_{m<k} Δz_m with Δz_m = −∫ τ/χ(T, P) d ln τ (trapezoid between the
levels): a lower-triangular (cumulative) operator down each column. Above k_c the column is τ-hydrostatic (column HSE in
τ, fixed: P(τ) there is set by the mass above, P ≈ g τ/κ̄).

Residuals (relative, in ln P; R = F/P):
  vertical, per level interval:  Δ ln P − ∫ (g ρ − L_z) τ/(χ P) d ln τ                 (HSE in τ with the Lorentz force)
  horizontal, per level:         −∂_x ln P|_τ − (g ρ − L_z)/P · ∂_x z|_τ + L_x/P       (−∂_x P|_z + L_x, over P)
L = J×B from B on the τ surfaces at their current heights (curvilinear derivatives); held fixed in the Jacobian of a step,
re-evaluated in the line search. Weighted least squares (H̄² per volume, as poisson_core.jl); CG on the normal equations,
preconditioned per horizontal Fourier mode by the layer-median operator A + |k|² B with the triangular operator inside B,
diagonalised once per step (A = LLᵀ, L⁻¹BL⁻ᵀ = VΛVᵀ): two dense (n+1)² products per mode.

usage: julia --project=. -t N poisson_tau_direct.jl IN OUT
ENV: MHS_CROP, MHS_CROP_OFF ("y0,x0"), MHS_DX_M, MHS_LTCUT (-4), MHS_GN_STEPS (40), MHS_GN_RTOL (1e-4),
     MHS_CG_IT (80), MHS_ALIGN (ptot)
=#
include(joinpath(@__DIR__, "poisson_tau.jl"))     # core + τ helpers (its main() only runs as a script)

const DC_CAP = parse(Float64, get(ENV, "MHS_DC_CAP_KM", "300")) * 1e3   # largest change of the cut-level height per step [m]

mutable struct TP
    tab::H.EOSTable
    nt::Int; kc::Int; n1::Int; ny::Int; nx::Int     # n1 nodes k = kc … nt; slot 1 = c, slots 2…n1 = ln P at k > kc
    dx::Float64; dy::Float64
    lτ::Vector{Float64}; dl::Vector{Float64}        # ln τ of the levels, ln τ_{k+1} − ln τ_k
    lT::Array{Float64,3}; Bx::Array{Float64,3}; By::Array{Float64,3}; Bz::Array{Float64,3}
    zrel::Array{Float64,3}                          # z_k − z_kc of the fixed levels k ≤ kc
    z::Array{Float64,3}                             # heights of all levels (last evaluation)
    Lx::Array{Float64,3}; Ly::Array{Float64,3}; Lz::Array{Float64,3}
    # on the nodes (n1, ny, nx), from the last evaluation
    a::Array{Float64,3}; aχ::Array{Float64,3}       # τ/χ and (τ/χ) ∂ln χ/∂ln P
    q::Array{Float64,3}                             # ρ/P
    W::Array{Float64,3}; dw::Array{Float64,3}       # (gρ − L_z)/P, ∂[(gρ − L_z) τ/(χP)]/∂ln P
    lx::Array{Float64,3}; ly::Array{Float64,3}      # local (non-differential) Jacobian terms of R_x, R_y
    Rx::Array{Float64,3}; Ry::Array{Float64,3}; Rz::Array{Float64,3}
    wc::Vector{Float64}; wf::Vector{Float64}; ereg::Vector{Float64}
    Q::Matrix{ComplexF64}; Qt::Matrix{ComplexF64}; λ::Vector{Float64}; s2::Vector{Float64}
    Vh::Array{ComplexF64,3}; Yh::Matrix{ComplexF64}
    pf::Any; pb::Any
    # Lorentz linearisation in the heights: δR_x += px ζ + pXx ∂_xδz + pYx ∂_yδz (same for y), δ(e a L_z) = zζ ζ + zX ∂_xδz
    # + zY ∂_yδz enters R_z; ζ_m = (δz_{m−1} − δz_{m+1})/zden_m (δz above the cut = δc)
    px::Array{Float64,3}; pXx::Array{Float64,3}; pYx::Array{Float64,3}
    py::Array{Float64,3}; pXy::Array{Float64,3}; pYy::Array{Float64,3}
    zζ::Array{Float64,3}; zX::Array{Float64,3}; zY::Array{Float64,3}; zden::Array{Float64,3}
    dLorentz::Bool
end

node(pr, m) = pr.kc + m - 1

"""∂f/∂z down each column (centred in level index, one-sided at the ends), z top first."""
function dzcol!(o, f, z)
    nt, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            o[1, j, i] = (f[1, j, i] - f[2, j, i]) / (z[1, j, i] - z[2, j, i])
            o[nt, j, i] = (f[nt-1, j, i] - f[nt, j, i]) / (z[nt-1, j, i] - z[nt, j, i])
            for k in 2:nt-1
                o[k, j, i] = (f[k-1, j, i] - f[k+1, j, i]) / (z[k-1, j, i] - z[k+1, j, i])
            end
        end
    end
    return o
end

"""J×B on the τ levels at heights z: ∂/∂z along the column, ∂/∂x|_z = ∂/∂x|_τ − ∂_x z|_τ ∂/∂z."""
function lorentz_tau!(pr::TP)
    z = pr.z; zx = similar(z); zy = similar(z); ddx!(zx, z, pr.dx); ddy!(zy, z, pr.dy)
    d(B) = (fz = dzcol!(similar(B), B, z); fx = ddx!(similar(B), B, pr.dx); fy = ddy!(similar(B), B, pr.dy);
            @. fx -= zx * fz; @. fy -= zy * fz; (fx, fy, fz))
    Bxx, Bxy, Bxz = d(pr.Bx); Byx, Byy, Byz = d(pr.By); Bzx, Bzy, Bzz = d(pr.Bz)
    Bx, By, Bz = pr.Bx, pr.By, pr.Bz
    Threads.@threads for idx in eachindex(z)
        @inbounds begin
            Jx = (Bzy[idx] - Byz[idx]) / MU0; Jy = (Bxz[idx] - Bzx[idx]) / MU0; Jz = (Byx[idx] - Bxy[idx]) / MU0
            pr.Lx[idx] = Jy * Bz[idx] - Jz * By[idx]; pr.Ly[idx] = Jz * Bx[idx] - Jx * Bz[idx]
            pr.Lz[idx] = Jx * By[idx] - Jy * Bx[idx]
        end
    end
    return Bxz, Byz, Bzz, zx, zy
end

"""State, residuals and Jacobian coefficients at (u, c); u holds ln P of all levels (k ≤ kc fixed). Returns (Φ, bad)."""
function evaluate!(pr::TP, u, c)
    nt, kc, n1, ny, nx = pr.nt, pr.kc, pr.n1, pr.ny, pr.nx
    bad = Threads.Atomic{Int}(0); ε = 1e-4
    ρu = similar(pr.a)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1
                k = node(pr, m); lT = pr.lT[k, j, i]; lp = u[k, j, i]
                r, dr, _ = H.lnrho_dY(pr.tab, lT, lp)
                _, _, c0, _ = H.state(pr.tab, lT, lp)
                _, _, cp, _ = H.state(pr.tab, lT, lp + ε); _, _, cm, _ = H.state(pr.tab, lT, lp - ε)
                isfinite(r + c0 + cp + cm) || (Threads.atomic_add!(bad, 1); r = 0.0; dr = 1.0; c0 = cp = cm = 0.0)
                pr.q[m, j, i] = exp(r - lp); ρu[m, j, i] = dr
                a = exp(pr.lτ[k] - c0); pr.a[m, j, i] = a; pr.aχ[m, j, i] = a * (cp - cm) / (2ε)
            end
            pr.z[kc, j, i] = c[j, i]
            for k in 1:kc-1
                pr.z[k, j, i] = c[j, i] + pr.zrel[k, j, i]
            end
            for m in 1:n1-1
                k = node(pr, m)
                pr.z[k+1, j, i] = pr.z[k, j, i] - pr.dl[k] / 2 * (pr.a[m, j, i] + pr.a[m+1, j, i])
            end
        end
    end
    bad[] > 0 && return Inf, bad[]
    Bxz, Byz, Bzz, zxf, zyf = lorentz_tau!(pr)
    if pr.dLorentz
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny, m in 1:n1
                k = node(pr, m); e = exp(-u[k, j, i]); bx = pr.Bx[k, j, i]; by = pr.By[k, j, i]; bz = pr.Bz[k, j, i]
                xz = Bxz[k, j, i]; yz = Byz[k, j, i]; zz = Bzz[k, j, i]; sx = zxf[k, j, i]; sy = zyf[k, j, i]
                cr(a1, a2, a3) = (a2 * bz - a3 * by, a3 * bx - a1 * bz, a1 * by - a2 * bx)
                Lζ = cr((sy * zz + yz) / MU0, (-xz - sx * zz) / MU0, (sx * yz - sy * xz) / MU0)
                LX = cr(0.0, zz / MU0, -yz / MU0); LY = cr(-zz / MU0, 0.0, xz / MU0)
                pr.px[m, j, i] = e * (Lζ[1] + sx * Lζ[3]); pr.pXx[m, j, i] = e * (LX[1] + sx * LX[3]); pr.pYx[m, j, i] = e * (LY[1] + sx * LY[3])
                pr.py[m, j, i] = e * (Lζ[2] + sy * Lζ[3]); pr.pXy[m, j, i] = e * (LX[2] + sy * LX[3]); pr.pYy[m, j, i] = e * (LY[2] + sy * LY[3])
                ea = e * pr.a[m, j, i]
                pr.zζ[m, j, i] = ea * Lζ[3]; pr.zX[m, j, i] = ea * LX[3]; pr.zY[m, j, i] = ea * LY[3]
                pr.zden[m, j, i] = k < nt ? pr.z[k-1, j, i] - pr.z[k+1, j, i] : pr.z[nt-1, j, i] - pr.z[nt, j, i]
            end
        end
    end
    un = u[kc:nt, :, :]; zn = pr.z[kc:nt, :, :]
    ux = ddx!(similar(un), un, pr.dx); uy = ddy!(similar(un), un, pr.dy)
    zx = ddx!(similar(zn), zn, pr.dx); zy = ddy!(similar(zn), zn, pr.dy)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1
                k = node(pr, m); e = exp(-u[k, j, i]); q = pr.q[m, j, i]; a = pr.a[m, j, i]
                W = G0 * q - pr.Lz[k, j, i] * e
                dW = G0 * q * (ρu[m, j, i] - 1) + pr.Lz[k, j, i] * e
                pr.W[m, j, i] = W; pr.dw[m, j, i] = dW * a - W * pr.aχ[m, j, i]
                pr.Rx[m, j, i] = -ux[m, j, i] - W * zx[m, j, i] + pr.Lx[k, j, i] * e
                pr.Ry[m, j, i] = -uy[m, j, i] - W * zy[m, j, i] + pr.Ly[k, j, i] * e
                pr.lx[m, j, i] = -zx[m, j, i] * dW - pr.Lx[k, j, i] * e
                pr.ly[m, j, i] = -zy[m, j, i] * dW - pr.Ly[k, j, i] * e
            end
            for f in 1:n1-1
                k = node(pr, f)
                pr.Rz[f, j, i] = u[k+1, j, i] - u[k, j, i] -
                                 pr.dl[k] / 2 * (pr.W[f, j, i] * pr.a[f, j, i] + pr.W[f+1, j, i] * pr.a[f+1, j, i])
            end
        end
    end
    return phi(pr), 0
end

function phi(pr::TP)
    s = zeros(pr.nx)
    Threads.@threads for i in 1:pr.nx
        acc = 0.0
        @inbounds for j in 1:pr.ny
            for m in 1:pr.n1
                acc += pr.wc[m] * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2)
            end
            for f in 1:pr.n1-1
                acc += pr.wf[f] * pr.Rz[f, j, i]^2
            end
        end
        s[i] = acc
    end
    return sum(s)
end

"""(jx, jy, jz) = J v; v slot 1 = δc, slots 2… = δ ln P."""
function Jop!(pr::TP, jx, jy, jz, v)
    n1, ny, nx = pr.n1, pr.ny, pr.nx
    du = copy(v); du[1, :, :] .= 0.0
    dz = similar(v)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            dz[1, j, i] = v[1, j, i]
            for f in 1:n1-1
                h = pr.dl[node(pr, f)] / 2
                dz[f+1, j, i] = dz[f, j, i] + h * (pr.aχ[f, j, i] * du[f, j, i] + pr.aχ[f+1, j, i] * du[f+1, j, i])
                jz[f, j, i] = du[f+1, j, i] - du[f, j, i] - h * (pr.dw[f, j, i] * du[f, j, i] + pr.dw[f+1, j, i] * du[f+1, j, i])
            end
        end
    end
    t = similar(v)
    ddx!(jx, du, pr.dx); ddx!(t, dz, pr.dx)
    @inbounds Threads.@threads for idx in eachindex(jx)
        jx[idx] = -jx[idx] - pr.W[idx] * t[idx] + pr.lx[idx] * du[idx]
    end
    ddy!(jy, du, pr.dy); t2 = ddy!(similar(v), dz, pr.dy)
    @inbounds Threads.@threads for idx in eachindex(jy)
        jy[idx] = -jy[idx] - pr.W[idx] * t2[idx] + pr.ly[idx] * du[idx]
    end
    pr.dLorentz || return
    # Lorentz force through the heights: t = ∂_xδz, t2 = ∂_yδz, ζ from the vertical neighbours
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            Tp = 0.0
            for m in 1:n1
                zm = m == 1 ? dz[1, j, i] : dz[m-1, j, i]          # δz above the cut = δc
                ζ = m < n1 ? (zm - dz[m+1, j, i]) / pr.zden[m, j, i] : (dz[n1-1, j, i] - dz[n1, j, i]) / pr.zden[m, j, i]
                X = t[m, j, i]; Y = t2[m, j, i]
                jx[m, j, i] += pr.px[m, j, i] * ζ + pr.pXx[m, j, i] * X + pr.pYx[m, j, i] * Y
                jy[m, j, i] += pr.py[m, j, i] * ζ + pr.pXy[m, j, i] * X + pr.pYy[m, j, i] * Y
                T = pr.zζ[m, j, i] * ζ + pr.zX[m, j, i] * X + pr.zY[m, j, i] * Y
                m >= 2 && (jz[m-1, j, i] += pr.dl[node(pr, m - 1)] / 2 * (Tp + T))
                Tp = T
            end
        end
    end
end

"""out = Jᵀ (rx, ry, rz)."""
function JTop!(pr::TP, out, rx, ry, rz)
    n1, ny, nx = pr.n1, pr.ny, pr.nx
    a = similar(out); b = similar(out)
    ddx!(out, rx, pr.dx); ddy!(a, ry, pr.dy)                 # (−D)ᵀ = D
    @. out += a + pr.lx * rx + pr.ly * ry
    @. a = pr.W * rx; @. b = pr.W * ry
    t = similar(out); ddx!(t, a, pr.dx); ddy!(a, b, pr.dy); t .+= a  # gradient with respect to δz on the nodes
    if pr.dLorentz
        gX = similar(out); gY = similar(out); gζ = similar(out)
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny, m in 1:n1
                sm = (m >= 2 ? pr.dl[node(pr, m - 1)] / 2 * rz[m-1, j, i] : 0.0) + (m < n1 ? pr.dl[node(pr, m)] / 2 * rz[m, j, i] : 0.0)
                X = rx[m, j, i]; Y = ry[m, j, i]
                gζ[m, j, i] = (pr.px[m, j, i] * X + pr.py[m, j, i] * Y + pr.zζ[m, j, i] * sm) / pr.zden[m, j, i]
                gX[m, j, i] = pr.pXx[m, j, i] * X + pr.pXy[m, j, i] * Y + pr.zX[m, j, i] * sm
                gY[m, j, i] = pr.pYx[m, j, i] * X + pr.pYy[m, j, i] * Y + pr.zY[m, j, i] * sm
            end
        end
        ddx!(a, gX, pr.dx); t .-= a; ddy!(a, gY, pr.dy); t .-= a     # Dᵀ = −D
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny
                for m in 1:n1-1
                    g = gζ[m, j, i]
                    t[m == 1 ? 1 : m - 1, j, i] += g; t[m+1, j, i] -= g
                end
                g = gζ[n1, j, i]; t[n1-1, j, i] += g; t[n1, j, i] -= g
            end
        end
    end
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            G = 0.0; tc = t[1, j, i]
            for f in n1-1:-1:1
                h = pr.dl[node(pr, f)] / 2
                r = rz[f, j, i]
                out[f+1, j, i] += r * (1 - h * pr.dw[f+1, j, i])
                out[f, j, i] += r * (-1 - h * pr.dw[f, j, i])
                G += t[f+1, j, i]; tc += t[f+1, j, i]
                out[f, j, i] += h * pr.aχ[f, j, i] * G
                out[f+1, j, i] += h * pr.aχ[f+1, j, i] * G
            end
            out[1, j, i] = tc                                  # slot 1 is δc (ln P at k_c is fixed)
        end
    end
    return out
end

function Aop!(pr::TP, out, v)
    jx = similar(v); jy = similar(v); jz = similar(v)
    Jop!(pr, jx, jy, jz, v)
    jx .*= pr.wc; jy .*= pr.wc; jz .*= vcat(pr.wf, 0.0)
    JTop!(pr, out, jx, jy, jz)
    @. out += pr.ereg * v
    return out
end

"""Layer-median operator A + s² B per horizontal mode, diagonalised: M⁻¹ = Qᵀ diag(1/(1 + s²λ)) Q."""
function setprec!(pr::TP)
    n1 = pr.n1
    med(A, m) = median(@view A[m, :, :])
    aχ = [med(pr.aχ, m) for m in 1:n1]; dw = [med(pr.dw, m) for m in 1:n1]; W = [med(pr.W, m) for m in 1:n1]
    Gz = zeros(n1 - 1, n1)                                     # vertical rows on the slots (slot 1 = c: no entry)
    for f in 1:n1-1
        h = pr.dl[node(pr, f)] / 2
        Gz[f, f+1] = 1 - h * dw[f+1]
        f >= 2 && (Gz[f, f] = -1 - h * dw[f])
    end
    S = zeros(n1, n1)                                          # δz_m = δc + Σ S[m, m'] δu_m'
    for m in 1:n1
        S[m, 1] = 1.0
        for f in 1:m-1
            h = pr.dl[node(pr, f)] / 2
            f >= 2 && (S[m, f] += h * aχ[f])
            S[m, f+1] += h * aχ[f+1]
        end
    end
    Gh = -(W .* S); for m in 2:n1; Gh[m, m] -= 1; end            # horizontal symbol / (i k)
    A = Gz' * Diagonal(pr.wf) * Gz + Diagonal(pr.ereg)
    B = Gh' * Diagonal(pr.wc) * Gh
    L = cholesky(Symmetric(A)).L
    Li = inv(Matrix(L))
    E = eigen(Symmetric(Li * B * Li'))
    Q = E.vectors' * Li
    pr.Q = ComplexF64.(Q); pr.Qt = ComplexF64.(Matrix(Q')); pr.λ = max.(E.values, 0.0)
end

function Minv!(pr::TP, out, r)
    n1 = pr.n1
    fft_layers!(pr.Vh, pr.pf, r, pr.ny)
    X = reshape(pr.Vh, n1, :)
    mul!(pr.Yh, pr.Q, X)
    Threads.@threads for col in axes(pr.Yh, 2)
        s = pr.s2[col]
        @inbounds for m in 1:n1
            pr.Yh[m, col] /= 1 + s * pr.λ[m]
        end
    end
    mul!(X, pr.Qt, pr.Yh)
    fft_layers!(out, pr.pb, pr.Vh, pr.ny)
    return out
end

function tpcg!(pr::TP, xs, b; tol = 1e-3, maxit = 80)
    fill!(xs, 0.0)
    r = copy(b); zv = similar(b); p = similar(b); Ap = similar(b)
    Minv!(pr, zv, r); p .= zv; rz = dot(r, zv); nb = norm(b); it = 0; rel = 1.0
    for outer it in 1:maxit
        Aop!(pr, Ap, p)
        αc = rz / dot(p, Ap)
        let a = αc; tbcast!((x, rr, pp, ap) -> (x + a * pp, rr - a * ap), (xs, r), p, Ap); end
        rel = norm(r) / nb
        rel < tol && break
        Minv!(pr, zv, r); rz2 = dot(r, zv); βc = rz2 / rz; rz = rz2
        let b = βc; tbcast!((pp, z) -> (z + b * pp,), (p,), zv); end
    end
    return it, rel
end

"""Residual summary by log10 τ band: RMS H̄|R_h| (horizontal) and H̄ R_z/Δz̄ (vertical), ≈ |F|/ρg."""
function bands(pr::TP, lt, Hb, dzb)
    out = String[]
    for (lo, hi) in ((-9.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 9.0))
        ms = [m for m in 1:pr.n1 if lo <= lt[node(pr, m)] < hi]
        isempty(ms) && continue
        h = sqrt(mean(Hb[m]^2 * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2) for m in ms, j in 1:pr.ny, i in 1:pr.nx))
        fs = [f for f in ms if f < pr.n1]
        v = isempty(fs) ? NaN : sqrt(mean((Hb[f] * pr.Rz[f, j, i] / dzb[f])^2 for f in fs, j in 1:pr.ny, i in 1:pr.nx))
        push!(out, @sprintf("log τ [%.0f, %.0f): horiz %.3f, vert %.3f", lo, hi, h, v))
    end
    return join(out, "; ")
end

function read_input(infile, crop)
    zyx(A) = permutedims(A, (3, 2, 1))
    h5open(infile, "r") do f
        oy, ox = parse.(Int, split(get(ENV, "MHS_CROP_OFF", "0,0"), ","))
        cy = crop > 0 ? (oy+1:oy+crop) : Colon(); cx = crop > 0 ? (ox+1:ox+crop) : Colon()
        if haskey(f, "log10tau500")          # IO/input-mhs-ltau.nc layout: h5py (x, y, τ), SI, Julia sees (τ, y, x)
            rn(nm) = Float64.(read(f[nm]))[:, cy, cx]
            d = parse(Float64, get(ENV, "MHS_DX_M", "47619.0"))
            xs = collect(0:size(f["temperature"], 3)-1) .* d
            return read(f["log10tau500"]), xs[cx], xs[cy], rn("temperature"), rn("B_x"), rn("B_y"), rn("B_z")
        else                                 # make_tau_input.py layout: h5py (τ, y, x)
            rd(nm) = zyx(Float64.(read(f[nm])))[:, cy, cx]
            return read(f["logtau500"]), read(f["x_m"])[cx], read(f["y_m"])[cy], rd("T"), rd("Bx"), rd("By"), rd("Bz")
        end
    end
end

"""MHS_CHECK = 1: adjoint test ⟨Jv, r⟩ = ⟨v, Jᵀr⟩ and J v against central differences of the residual (those also
move L through the heights, which the Jacobian holds fixed)."""
function check_jacobian(pr::TP, u, c)
    n1, ny, nx, kc, nt = pr.n1, pr.ny, pr.nx, pr.kc, pr.nt
    v = randn(n1, ny, nx) .* 1e-3; v[1, :, :] .= randn(ny, nx) .* 1e3
    jx, jy, jz = similar(v), similar(v), similar(v); Jop!(pr, jx, jy, jz, v); jz[end, :, :] .= 0
    rx, ry, rz = randn(n1, ny, nx), randn(n1, ny, nx), randn(n1, ny, nx); rz[end, :, :] .= 0
    g = similar(v); JTop!(pr, g, rx, ry, rz)
    l = dot(jx, rx) + dot(jy, ry) + dot(jz, rz); r = dot(v, g)
    @printf("%s check: adjoint ⟨Jv, r⟩ %.10e, ⟨v, Jᵀr⟩ %.10e, rel %.2e\n", stamp(), l, r, abs(l - r) / abs(l))
    for ε in (1e-2, 1e-3)
        res = map((1, -1)) do sgn
            ut = copy(u); @views ut[kc+1:nt, :, :] .+= sgn * ε .* v[2:end, :, :]
            evaluate!(pr, ut, c .+ sgn * ε .* v[1, :, :]); (copy(pr.Rx), copy(pr.Ry), copy(pr.Rz))
        end
        for (nm, d, jv) in (("x", (res[1][1] .- res[2][1]) ./ 2ε, jx), ("y", (res[1][2] .- res[2][2]) ./ 2ε, jy),
                            ("z", ((res[1][3] .- res[2][3]) ./ 2ε)[1:end-1, :, :], jz[1:end-1, :, :]))
            @printf("%s check ε %.0e: R_%s finite difference vs J v: rel error %.2e (median per node %.2e)\n", stamp(), ε, nm,
                    norm(d .- jv) / norm(jv), median(abs.(d .- jv)) / median(abs.(jv)))
        end
    end
    evaluate!(pr, u, c)
end

function main_direct()
    infile, outfile = ARGS[1], ARGS[2]
    crop = parse(Int, get(ENV, "MHS_CROP", "0"))
    ltcut = parse(Float64, get(ENV, "MHS_LTCUT", "-4"))
    steps = parse(Int, get(ENV, "MHS_GN_STEPS", "40"))
    rtol = parse(Float64, get(ENV, "MHS_GN_RTOL", "1e-4"))
    cgit = parse(Int, get(ENV, "MHS_CG_IT", "80"))
    FFTW.set_num_threads(1); BLAS.set_num_threads(min(32, Threads.nthreads()))

    lt, x, y, T, Bx, By, Bz = read_input(infile, crop)
    @assert issorted(lt) "logtau500 must increase (top first)"
    if get(ENV, "MHS_NO_LORENTZ", "0") == "1"                  # diagnostic: field-free
        Bx .= 0; By .= 0; Bz .= 0
    end
    nt, ny, nx = size(T); dx, dy = x[2] - x[1], y[2] - y[1]
    tab = H.load_table(TAB); lT = log.(T)
    kc = argmin(abs.(lt .- ltcut)); n1 = nt - kc + 1
    println(stamp(), " input $infile: $nt levels, $(ny)×$(nx) columns; cut at log10 τ = $(lt[kc]) (level $kc): ",
            "$(n1 - 1) ln P levels + 1 height per column; threads $(Threads.nthreads())")

    # start: hydrostatic columns in τ, aligned on the total pressure (as poisson_tau.jl)
    u = zeros(nt, ny, nx); z0 = zeros(nt, ny, nx)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            hse_tau!(@view(u[:, j, i]), @view(z0[:, j, i]), @view(lT[:, j, i]), lt, tab)
        end
    end
    get(ENV, "MHS_ALIGN", "ptot") == "ptot" && align_ptot!(z0, u, lT, Bx, By, Bz, tab)
    gauge!(z0, lt)
    zrel = z0[1:kc, :, :] .- z0[kc:kc, :, :]
    c = z0[kc, :, :]
    k0 = argmin(abs.(lt))

    nyh = ny ÷ 2 + 1
    sy2 = [sin(2π * (j - 1) / ny)^2 / dy^2 for j in 1:nyh]
    sx2 = [sin(2π * fftfreq(nx)[i])^2 / dx^2 for i in 1:nx]
    s2 = vec([sy2[j] + sx2[i] for j in 1:nyh, i in 1:nx])
    Z() = zeros(n1, ny, nx)
    pr = TP(tab, nt, kc, n1, ny, nx, dx, dy, LN10 .* lt, diff(LN10 .* lt), lT, Bx, By, Bz, zrel, zeros(nt, ny, nx),
            zeros(nt, ny, nx), zeros(nt, ny, nx), zeros(nt, ny, nx), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(),
            ones(n1), ones(n1 - 1), zeros(n1), zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0), Float64[], s2,
            zeros(ComplexF64, n1, nyh, nx), zeros(ComplexF64, n1, nyh * nx),
            plan_rfft(zeros(ny, nx); flags = FFTW.ESTIMATE | FFTW.UNALIGNED),
            plan_irfft(zeros(ComplexF64, nyh, nx), ny; flags = FFTW.ESTIMATE | FFTW.UNALIGNED),
            Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), ones(n1, ny, nx), get(ENV, "MHS_DLORENTZ", "1") == "1")
    _, bad = evaluate!(pr, u, c)
    bad > 0 && error("start state outside the EOS table in $bad nodes")
    # weights from the start (fixed during the solve): H̄² per volume, as poisson_core.jl
    Hb = [median(1 ./ (G0 .* @view pr.q[m, :, :])) for m in 1:n1]
    dzf = [median(@view(pr.z[node(pr, f), :, :]) .- @view(pr.z[node(pr, f)+1, :, :])) for f in 1:n1-1]
    dzn = [m == 1 ? dzf[1] / 2 : m == n1 ? dzf[end] / 2 : (dzf[m-1] + dzf[m]) / 2 for m in 1:n1]
    pr.wc = dzn .* Hb .^ 2
    pr.wf = [Hb[f] * Hb[f+1] / dzf[f] for f in 1:n1-1]
    κ = 2π / (nx * dx)
    Wb = [median(@view pr.W[m, :, :]) for m in 1:n1]
    pr.ereg = 1e-2 * κ^2 .* vcat(sum(pr.wc .* Wb .^ 2), pr.wc[2:end])
    Wtot = (2 * sum(pr.wc) + sum(pr.wf)) * ny * nx
    Φ, _ = evaluate!(pr, u, c); Φ0 = Φ
    if get(ENV, "MHS_CHECK", "0") == "1"
        check_jacobian(pr, u, c); Φ, _ = evaluate!(pr, u, c)
    end
    @printf("%s start: Φ %.6e, RMS %.4e | %s | z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km\n", stamp(), Φ, sqrt(Φ / Wtot),
            bands(pr, lt, Hb, dzf), pct(pr.z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3...)
    flush(stdout)

    # Levenberg–Marquardt: damping μ D (D = the regularisation's diagonal shape, κ² H̄²-weighted volume), raised ×4 when a
    # step fails or exceeds the caps, lowered ×3 after a full step. Steps over the caps are scaled as a whole (an
    # element-wise clamp of δc turned the step into an ascent direction where it asked for > 1 Mm in a few columns).
    δ = Z(); rhs = Z(); ut = copy(u); ct = copy(c)
    ereg0 = copy(pr.ereg); Dlm = ereg0 ./ 1e-2
    μ = parse(Float64, get(ENV, "MHS_LM_MU0", "0"))
    dbg = get(ENV, "MHS_LS_DEBUG", "0") == "1"
    for s in 1:steps
        t0 = time()
        JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
        accepted = false; its = 0; rel = 0.0; α = 1.0; Φt = Inf; sc = 1.0
        for tries in 1:8
            pr.ereg = ereg0 .+ μ .* Dlm
            setprec!(pr)
            its, rel = tpcg!(pr, δ, rhs; maxit = cgit)
            mu_ = maximum(abs, @view δ[2:end, :, :]); mc = maximum(abs, @view δ[1, :, :])
            sc = min(1.0, DU_CAP / mu_, DC_CAP / mc)
            dbg && @printf("%s   μ %.3g: CG %d (rel %.1e), max|δu| %.3f, max|δc| %.1f km, scale %.3g\n", stamp(), μ, its, rel, mu_, mc / 1e3, sc)
            α = sc
            for ls in 1:4
                ut .= u; @views ut[kc+1:nt, :, :] .+= α .* δ[2:end, :, :]
                @views ct .= c .+ α .* δ[1, :, :]
                Φt, bad = evaluate!(pr, ut, ct)
                dbg && @printf("%s     ls %d: α %.3g, Φt/Φ %.6f, bad %d\n", stamp(), ls, α, Φt / Φ, bad)
                bad == 0 && Φt < Φ && (accepted = true; break)
                α /= 2
            end
            accepted && break
            μ = max(4μ, 1.0)
        end
        if !accepted
            evaluate!(pr, u, c)
            @printf("%s step %2d: no decrease up to μ = %.3g; stopping\n", stamp(), s, μ)
            break
        end
        μ = α == 1.0 ? μ / 3 : (α >= sc / 2 ? μ : 2μ)
        μ < 1e-3 && (μ = 0.0)
        dec = (Φ - Φt) / Φ
        u .= ut; c .= ct; Φ = Φt
        @printf("%s step %2d: Φ/Φ₀ %.3e (ΔΦ/Φ %.2e), RMS %.4e | CG %d its (rel %.1e), α %.3g, μ next %.3g, max|δc| %.1f km | z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km | %.1f s\n",
                stamp(), s, Φ / Φ0, dec, sqrt(Φ / Wtot), its, rel, α, μ, α * maximum(abs, @view δ[1, :, :]) / 1e3,
                pct(pr.z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3..., time() - t0)
        flush(stdout)
        dec < rtol && break
    end
    pr.ereg = ereg0
    println(stamp(), " end: ", bands(pr, lt, Hb, dzf))

    z = copy(pr.z); sg = gauge!(z, lt)
    @printf("%s gauge shift %.1f km; z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km; z(log τ = %.1f) p1/p50/p99 %.1f / %.1f / %.1f km\n",
            stamp(), sg / 1e3, pct(z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3..., lt[kc], pct(z[kc, :, :], [0.01, 0.5, 0.99]) ./ 1e3...)
    rhot = similar(u)
    Threads.@threads for idx in eachindex(u)
        r, _, _ = H.lnrho_dY(tab, lT[idx], u[idx]); rhot[idx] = exp(r)
    end
    xyz(A) = permutedims(A, (3, 2, 1))
    h5open(outfile, "w") do f
        t = create_group(f, "tau_grid")
        t["logtau500"] = lt; t["z_m"] = xyz(z); t["Pgas"] = xyz(exp.(u)); t["rho"] = xyz(rhot)
        t["T"] = xyz(T); t["Bx"] = xyz(Bx); t["By"] = xyz(By); t["Bz"] = xyz(Bz)
        attrs(f)["method"] = "poisson_tau_direct.jl: one Gauss–Newton solve on the τ grid"
        attrs(f)["logtau_cut"] = lt[kc]; attrs(f)["source"] = infile
    end
    println(stamp(), " wrote $outfile")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_direct()
end
