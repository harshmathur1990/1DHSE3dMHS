# run.mode = "mhs-poisson-z": Poisson-type pressure step on the height grid (from scripts/poisson/poisson_core.jl).
#
# T and B are fixed; only the pressure changes. Below z_cut, u = ln P minimises
#     Φ = Σ_centres w H̄² (R_x² + R_y²) + Σ_z-faces h H̄² R_z²,     R = F/P = −∇u − ẑ/H(T, P) + J×B/P,
# with H = P/(ρ g) and H̄ the layer median of H in the starting state (fixed). Cell-centred B, centred periodic x, y
# differences, R_z on interior z-faces. Gauss–Newton; each step solves the normal equations by PCG, preconditioned by
# the Laplacian part with layer-median coefficients: distributed FFT in x, y and a tridiagonal solve in z per mode.
# Steps are capped at |δu| ≤ 2 per node and backtracked on Φ. The mean of u over the anchor cells (default: the bottom
# plane) is held fixed. Above z_cut each column is integrated hydrostatically from the solved pressure at z_cut.
# Arrays are (z, y, x) slabs in x (Slab.jl); every reduction is global, so results do not depend on the rank count
# beyond the rounding of sums.
module PoissonZ

using MPI, FFTW, LinearAlgebra, Printf
import ..HSEMHS as H
using ..Slab
import ..HSEMHS: ParCtx, rank0, MU0

const DU_CAP = 2.0

struct Prob
    par::ParCtx; S::SlabFFT
    tab::H.EOSTable; g::Float64
    n::Int; ny::Int; nx::Int; NX::Int
    dx::Float64; dy::Float64
    h::Vector{Float64}; wc::Vector{Float64}; wf::Vector{Float64}; ereg::Vector{Float64}
    Aanc::Float64; anc::BitArray{3}; acnt::Vector{Float64}
    lT::Array{Float64,3}; lx::Array{Float64,3}; ly::Array{Float64,3}; lz::Array{Float64,3}
    q::Array{Float64,3}; qd::Array{Float64,3}; ex::Array{Float64,3}
    tx::Array{Float64,3}; ty::Array{Float64,3}; jx::Array{Float64,3}; jy::Array{Float64,3}; jz::Array{Float64,3}
    sx2::Vector{Float64}; sy2::Vector{Float64}
    cdiag::Vector{Float64}; coff::Vector{Float64}
end

"""np.gradient-type ∂/∂z on a (z, y, x) array, z top first (local: columns are complete on every rank)."""
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

"""ρ/P, its u-derivative and e^{−u} at u; returns the global number of nodes outside the EOS table."""
function eos!(pr::Prob, q, qd, ex, u)
    bad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(u)
        lr, d, _ = H.lnrho_dY(pr.tab, pr.lT[idx], u[idx])
        isfinite(lr) || Threads.atomic_add!(bad, 1)
        qq = exp(lr - u[idx]); q[idx] = qq; qd[idx] = qq * (d - 1); ex[idx] = exp(-u[idx])
    end
    return gcount(pr.par, bad[])
end

"""R = −∇u − ẑ g ρ/P + J×B/P (R_x, R_y at centres, R_z on interior faces)."""
function resid!(pr::Prob, Rx, Ry, Rz, u, q, ex)
    sddx!(Rx, u, pr.dx, pr.par); sddy!(Ry, u, pr.dy)
    @. Rx = pr.lx * ex - Rx; @. Ry = pr.ly * ex - Ry
    h = pr.h; lz = pr.lz; g = pr.g
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, f in 1:pr.n-1
            Rz[f, j, i] = -(u[f, j, i] - u[f+1, j, i]) / h[f] - g * (q[f, j, i] + q[f+1, j, i]) / 2 +
                          (lz[f, j, i] * ex[f, j, i] + lz[f+1, j, i] * ex[f+1, j, i]) / 2
        end
    end
end

function phi(pr::Prob, Rx, Ry, Rz)
    part = zeros(pr.nx); wc = pr.wc; wf = pr.wf
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
        part[i] = a
    end
    return gsum(pr.par, sum(part))
end

# Linearisation: δR_x = −D_x δu − (L_x e^{−u}) δu, δR_y likewise,
# δR_z,f = −[a_f δu_k + b_f δu_k+1], a_f = 1/h + (g qd_k + L_z,k e^{−u_k})/2, b_f = −1/h + (g qd_k+1 + L_z,k+1 e^{−u_k+1})/2.
@inline coef_a(pr, f, j, i) = 1 / pr.h[f] + (pr.g * pr.qd[f, j, i] + pr.lz[f, j, i] * pr.ex[f, j, i]) / 2
@inline coef_b(pr, f, j, i) = -1 / pr.h[f] + (pr.g * pr.qd[f+1, j, i] + pr.lz[f+1, j, i] * pr.ex[f+1, j, i]) / 2

"""out = Jᵀ W (Rx, Ry, Rz)."""
function JtW!(pr::Prob, out, Rx, Ry, Rz)
    wc = pr.wc; wf = pr.wf
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, k in 1:pr.n
            pr.tx[k, j, i] = wc[k] * Rx[k, j, i]; pr.ty[k, j, i] = wc[k] * Ry[k, j, i]
        end
    end
    sddx!(out, pr.tx, pr.dx, pr.par); sddy!(pr.jx, pr.ty, pr.dy)               # (−D)ᵀ = D
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, k in 1:pr.n
            out[k, j, i] += pr.jx[k, j, i] - pr.ex[k, j, i] * (pr.lx[k, j, i] * pr.tx[k, j, i] + pr.ly[k, j, i] * pr.ty[k, j, i])
        end
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
    sddx!(pr.jx, v, pr.dx, pr.par); sddy!(pr.jy, v, pr.dy)
    part = zeros(pr.nx)                                   # anchor: Σ v over the anchor cells, per x slice
    Threads.@threads for i in 1:pr.nx
        sa = 0.0
        @inbounds for j in 1:pr.ny
            for k in 1:pr.n
                e = pr.ex[k, j, i] * v[k, j, i]
                pr.jx[k, j, i] = -pr.jx[k, j, i] - pr.lx[k, j, i] * e
                pr.jy[k, j, i] = -pr.jy[k, j, i] - pr.ly[k, j, i] * e
                pr.anc[k, j, i] && (sa += v[k, j, i])
            end
            for f in 1:pr.n-1
                pr.jz[f, j, i] = -(coef_a(pr, f, j, i) * v[f, j, i] + coef_b(pr, f, j, i) * v[f+1, j, i])
            end
        end
        part[i] = sa
    end
    JtW!(pr, out, pr.jx, pr.jy, pr.jz)
    as = pr.Aanc * gsum(pr.par, sum(part))
    Threads.@threads for i in 1:pr.nx                    # damping and anchor (threaded: masked indexing allocates)
        @inbounds for j in 1:pr.ny, k in 1:pr.n
            out[k, j, i] += pr.ereg[k] * v[k, j, i] + (pr.anc[k, j, i] ? as : 0.0)
        end
    end
    return out
end

"""Vertical part of the preconditioner from global layer medians of the current coefficients."""
function setprec!(pr::Prob)
    n = pr.n
    A = [coef_a(pr, f, j, i) for f in 1:n-1, j in 1:pr.ny, i in 1:pr.nx]
    B = [coef_b(pr, f, j, i) for f in 1:n-1, j in 1:pr.ny, i in 1:pr.nx]
    am = gmedians(pr.par, A); bm = gmedians(pr.par, B)
    fill!(pr.cdiag, 0.0)
    for f in 1:n-1
        a, b = am[f], bm[f]
        pr.cdiag[f] += pr.wf[f] * a^2; pr.cdiag[f+1] += pr.wf[f] * b^2; pr.coff[f] = pr.wf[f] * a * b
    end
end

"""Thomas solve, in place on x (rhs in, solution out), of the vertical system of the mode with horizontal |k|² = s2."""
@inline function thomas!(x, cp, dp, pr::Prob, s2)
    n = pr.n
    @inbounds for k in 1:n
        d = pr.cdiag[k] + pr.wc[k] * s2 + pr.ereg[k]
        if k == 1
            cp[1] = pr.coff[1] / d; dp[1] = x[1] / d
        else
            den = d - pr.coff[k-1] * cp[k-1]
            cp[k] = k < n ? pr.coff[k] / den : 0.0
            dp[k] = (x[k] - pr.coff[k-1] * dp[k-1]) / den
        end
    end
    @inbounds x[n] = dp[n]
    @inbounds for k in n-1:-1:1
        x[k] = dp[k] - cp[k] * x[k+1]
    end
    return x
end

function Minv!(pr::Prob, out, r)
    n = pr.n; S = pr.S; par = pr.par; Ncol = pr.ny * pr.NX
    T = fwd!(S, par, r)
    nyhl = size(T, 3)
    Threads.@threads for jl in 1:nyhl
        cp = zeros(n); dp = zeros(ComplexF64, n)
        ky = S.jr[par.rank+1][1] + jl - 1
        for kx in 1:S.nx
            x = @view T[:, kx, jl]
            thomas!(x, cp, dp, pr, pr.sx2[kx] + pr.sy2[ky])
            if kx == 1 && ky == 1
                # mean mode: the anchor adds (Aanc/Ncol) c cᵀ with c_k = anchor cells in layer k (rank 1, coupling the
                # layers); Sherman–Morrison on top of the tridiagonal solve
                w = ComplexF64.(pr.acnt); thomas!(w, cp, dp, pr, 0.0)
                α = pr.Aanc / Ncol
                x .-= (α * sum(pr.acnt .* x) / (1 + α * sum(pr.acnt .* w))) .* w
            end
        end
    end
    bwd!(S, par, out)
    return out
end

"""Threaded element-wise update: outs[m][I] = f(outs..., ins...)[m] (CG vector updates; plain broadcasts run serially)."""
function tbcast!(f, outs, ins...)
    N = length(outs[1])
    Threads.@threads for c in collect(Iterators.partition(1:N, cld(N, 4 * Threads.nthreads())))
        @inbounds for I in c
            y = f(map(o -> o[I], outs)..., map(a -> a[I], ins)...)
            for m in eachindex(outs)
                outs[m][I] = y[m]
            end
        end
    end
end

"""Preconditioned CG on Aop!/Minv! with global inner products (generic over the problem type). Starts from xs = 0, or
from the given xs with `x0 = true` (one extra operator product). Returns (iterations, ‖b − A xs‖/‖b‖): from the
recursion on a normal exit (tol reached or maxit), the true residual of the returned xs on an abnormal one. b = 0 gives
xs = 0 (the exact solution, whatever the start). A start already within `tol` returns at once with its true residual.

Each update is formed out of place and kept only if the new x and r are finite (tested in the residual-norm
reduction: ‖r‖² and the number of non-finite entries of x, one collective, so every rank takes the same decision), so
xs is always the last good iterate. Three global reductions per iteration (pᵀAp, ‖r‖² with the count, rᵀM⁻¹r).
`flag` (a Ref{Symbol} or nothing) receives the outcome: :converged, :maxit, :zero_rhs, :start_within_tol, or the
abnormal exits :nonfinite_start (the start residual is not finite; xs unchanged), :breakdown_rz (rᵀM⁻¹r non-positive or
non-finite), :breakdown_pAp (pᵀAp non-positive or non-finite), :nonfinite_step (α = rᵀM⁻¹r/pᵀAp not finite) and
:nonfinite_update (x + αp or r − αAp overflows). `cnt` (a 2-vector or nothing) is incremented by the operator and
preconditioner applications made (the true-residual check of an abnormal exit included)."""
function pcg!(Aop!, Minv!, par, pr, xs, b; tol = 1e-3, maxit = 80, x0 = false, cnt = nothing, flag = nothing)
    setf(f) = (flag === nothing || (flag[] = f); nothing)
    nb = gnorm(par, b)
    nb == 0 && (fill!(xs, 0.0); setf(:zero_rhs); return 0, 0.0)
    zv = similar(b); p = similar(b); Ap = similar(b)
    na = 0; nm = 0
    tally() = (cnt === nothing || (cnt[1] += na; cnt[2] += nm); nothing)
    function abnormal(f)                              # true residual of the returned (last good) xs
        setf(f)
        Aop!(pr, Ap, xs); na += 1
        t = gnorm(par, b .- Ap) / nb
        tally()
        return isfinite(t) ? t : Inf
    end
    if x0
        Aop!(pr, Ap, xs); r = b .- Ap; na += 1
    else
        fill!(xs, 0.0); r = copy(b)
    end
    rel = gnorm(par, r) / nb; it = 0
    isfinite(rel) || (setf(:nonfinite_start); tally(); return 0, Inf)
    rel < tol && (setf(:start_within_tol); tally(); return 0, rel)
    Minv!(pr, zv, r); nm += 1; p .= zv; rz = gdot(par, r, zv)
    (isfinite(rz) && rz > 0) || return 0, abnormal(:breakdown_rz)
    setf(:maxit)
    for outer it in 1:maxit
        Aop!(pr, Ap, p); na += 1
        pAp = gdot(par, p, Ap)
        (isfinite(pAp) && pAp > 0) || return it - 1, abnormal(:breakdown_pAp)
        αc = rz / pAp
        (isfinite(αc) && αc > 0) || return it - 1, abnormal(:nonfinite_step)
        let a = αc                                    # zv = x + αp, Ap = r − αAp (out of place)
            tbcast!((zz, pp, x) -> (x + a * pp,), (zv,), p, xs)
            tbcast!((ap, rr) -> (rr - a * ap,), (Ap,), r)
        end
        rs = gsum(par, [dot(Ap, Ap), Float64(count(!isfinite, zv))])   # one reduction: ‖r‖² and the non-finite x
        reln = sqrt(rs[1]) / nb
        (isfinite(reln) && rs[2] == 0) || return it - 1, abnormal(:nonfinite_update)
        tbcast!((x, rr, zz, ap) -> (zz, ap), (xs, r), zv, Ap); rel = reln
        rel < tol && (setf(:converged); break)
        Minv!(pr, zv, r); nm += 1; rz2 = gdot(par, r, zv)
        (isfinite(rz2) && rz2 > 0) || return it, abnormal(:breakdown_rz)
        βc = rz2 / rz; rz = rz2
        let b = βc; tbcast!((pp, z) -> (z + b * pp,), (p,), zv); end
    end
    tally()
    return it, rel
end

"""Lorentz force J×B [N m⁻³] from cell-centred B on the (z, y, x) slabs (centred periodic x, y; np.gradient-type z)."""
function lorentz(par, Bx, By, Bz, z, dx, dy)
    Lx, Ly, Lz = similar(Bx), similar(Bx), similar(Bx)
    a = similar(Bx); b = similar(Bx)
    sddy!(a, Bz, dy); ddz!(b, By, z); Jx = (a .- b) ./ MU0
    ddz!(a, Bx, z); sddx!(b, Bz, dx, par); Jy = (a .- b) ./ MU0
    sddx!(a, By, dx, par); sddy!(b, Bx, dy); Jz = (a .- b) ./ MU0
    @. Lx = Jy * Bz - Jz * By; @. Ly = Jz * Bx - Jx * Bz; @. Lz = Jx * By - Jy * Bx
    return Lx, Ly, Lz
end

"""
    solve_pressure(par, tab, g, lnT, Lx, Ly, Lz, P0, z, dx, dy; zcut, maxouter = 30, rtol = 1e-4, anchor = nothing,
                   anchor_w = 1.0, logf = s -> nothing)

Gauss–Newton Poisson step from the start pressure P0 (z, y, x slabs; z top first) with the Lorentz force fixed. Solves
z ≤ zcut (the mean of ln P over the anchor cells, a Bool array like P0 or the bottom plane by default, stays that of
P0) and integrates each column hydrostatically above zcut. Returns (P, Φ0, Φ, steps, kc, u0, u, reason, gn_decrement,
grad_norm, cg_its, cg_rel, rms0, rms, phi0_zero, cont_bad): `reason` is :converged, :no_descent or :max_steps;
`gn_decrement` the relative decrease of Φ predicted by one further Gauss–Newton step (a scale-free stationarity measure)
and `grad_norm` = ‖Jᵀ W R‖ at the returned state; `cont_bad` the global number of non-finite pressures left by the
hydrostatic continuation above zcut. The anchor and `ereg` terms act on the increment system only (they damp changes of
the anchor-cell mean ln P per step; Φ has no anchor term). `S` reuses a cached SlabFFT; `rms_atol` > 0 accepts a start
with √(Φ₀/W) ≤ rms_atol as converged in zero steps.
"""
function solve_pressure(par, tab, g, lnT, Lx, Ly, Lz, P0, z, dx, dy; zcut, maxouter = 30, rtol = 1e-4, anchor = nothing,
                        anchor_w = 1.0, cg_tol = 1e-3, cg_maxit = 80, n_consec = 3, logf = s -> nothing, S = nothing,
                        rms_atol = 0.0, stationarity = true)
    nz, ny, nx = size(P0); NX = par.NX
    kc = findfirst(<=(zcut), z)
    kc === nothing && H.invalid("z_cut = $zcut m lies below the grid")
    R = kc:nz; n = length(R)
    zr = z[R]
    h = zr[1:end-1] .- zr[2:end]
    zf = vcat(zr[1] + h[1] / 2, (zr[1:end-1] .+ zr[2:end]) ./ 2, zr[end] - h[end] / 2)
    w = zf[1:end-1] .- zf[2:end]
    u0 = log.(P0[R, :, :]); u = copy(u0)
    lT = lnT[R, :, :]
    q = similar(u); qd = similar(u); ex = similar(u)
    qq = similar(u)
    Threads.@threads for idx in eachindex(u)
        lr, _, _ = H.lnrho_dY(tab, lT[idx], u[idx]); qq[idx] = 1 / (g * exp(lr - u[idx]))
    end
    Hl = gmedians(par, qq)                                        # layer median scale height H̄
    Hf = sqrt.(Hl[1:end-1] .* Hl[2:end])
    wc = w .* Hl .^ 2; wf = h .* Hf .^ 2
    Ncol = ny * NX
    Wtot = (2 * sum(w) + sum(h)) * Ncol
    κ = 2π / (NX * dx)
    ereg = 1e-2 * κ^2 .* wc
    if anchor === nothing
        anc = falses(n, ny, nx); anc[n, :, :] .= true
    else
        anc = BitArray(anchor[R, :, :])
    end
    acnt = gsum(par, [Float64(count(@view anc[k, :, :])) for k in 1:n])
    Aanc = anchor_w * wf[end] / h[end]^2 / sum(acnt)
    nyh = ny ÷ 2 + 1
    sy2 = [sin(2π * (j - 1) / ny)^2 / dy^2 for j in 1:nyh]
    sx2 = [sin(2π * fftfreq(NX)[i])^2 / dx^2 for i in 1:NX]
    S === nothing && (S = SlabFFT(par, n, ny))
    (S.nk, S.ny, S.nxl) == (n, ny, nx) || error("solve_pressure: cached SlabFFT has sizes $((S.nk, S.ny, S.nxl)), expected $((n, ny, nx))")
    pr = Prob(par, S, tab, g, n, ny, nx, NX, dx, dy, h, wc, wf, ereg, Aanc, anc, acnt, lT,
              Lx[R, :, :], Ly[R, :, :], Lz[R, :, :], q, qd, ex,
              similar(u), similar(u), similar(u), similar(u), zeros(n - 1, ny, nx), sx2, sy2, zeros(n), zeros(n - 1))

    Rx, Ry, Rz = similar(u), similar(u), zeros(n - 1, ny, nx)
    eos!(pr, q, qd, ex, u); resid!(pr, Rx, Ry, Rz, u, q, ex)
    Φ = phi(pr, Rx, Ry, Rz); Φ0 = Φ
    logf(@sprintf("start: Φ %.6e, RMS |R| H̄ %.4e over %d layers (z %.3f … %.3f Mm)", Φ, sqrt(Φ / Wtot), n, zr[1] / 1e6, zr[end] / 1e6))
    δ = similar(u); rhs = similar(u); ut = similar(u); qt = similar(u); qdt = similar(u); ext = similar(u)
    steps = 0; nsmall = 0
    reason = :max_steps
    phi0_zero = Φ0 == 0
    start_ok = phi0_zero || sqrt(Φ0 / Wtot) <= rms_atol
    if start_ok
        reason = :converged
        logf("start state already within rms_atol: no steps")
        maxouter = 0
    end
    for outer in 1:maxouter
        t0 = time()
        setprec!(pr)
        JtW!(pr, rhs, Rx, Ry, Rz); rhs .*= -1
        its, rel = pcg!(Aop!, Minv!, par, pr, δ, rhs; tol = cg_tol, maxit = cg_maxit)
        ncap = gcount(par, count(v -> abs(v) > DU_CAP, δ))
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
            logf(@sprintf("step %2d: no decrease along the step (CG %d its, rel %.1e); stopping", outer, its, rel))
            reason = :no_descent
            break
        end
        rel_dec = (Φ - Φt) / Φ
        u .= ut; q .= qt; qd .= qdt; ex .= ext; Φ = Φt; steps = outer
        logf(@sprintf("step %2d: Φ %.6e (Φ/Φ₀ %.3e, ΔΦ/Φ %.2e), RMS |R| H̄ %.4e | CG %d its (rel %.1e), α %.3g, capped %d, max|αδu| %.3f, %.1f s",
                     outer, Φ, Φ / Φ0, rel_dec, sqrt(Φ / Wtot), its, rel, α, ncap, α * gmax(par, maximum(abs, δ)), time() - t0))
        nsmall = rel_dec < rtol ? nsmall + 1 : 0
        nsmall >= n_consec && (logf(@sprintf("converged: ΔΦ/Φ < %.0e on %d consecutive steps", rtol, nsmall));
                               reason = :converged; break)
    end
    # stationarity at the returned u (Rx, Ry, Rz and q, qd, ex belong to it): undamped Gauss–Newton step and its predicted
    # relative decrease. Diagnostics only: u is not changed.
    gnd = start_ok ? 0.0 : NaN; gn = start_ok ? 0.0 : NaN; sits = 0; srel = start_ok ? 0.0 : NaN
    if stationarity && !start_ok
        setprec!(pr)
        JtW!(pr, rhs, Rx, Ry, Rz); rhs .*= -1
        gn = gnorm(par, rhs)
        sits, srel = pcg!(Aop!, Minv!, par, pr, δ, rhs; tol = cg_tol, maxit = cg_maxit)
        gnd = gdot(par, δ, rhs) / Φ
        logf(@sprintf("stationarity: Gauss–Newton decrement %.3e, ‖JᵀWR‖ %.3e (CG %d its, rel %.1e)", gnd, gn, sits, srel))
    end

    # ---- full column: solved region below z_cut, hydrostatic integration above it ----
    Pfull = copy(P0); Pfull[R, :, :] .= exp.(u)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for k in kc-1:-1:1
                Δz = z[k] - z[k+1]
                ub = log(Pfull[k+1, j, i]); lrb, _, _ = H.lnrho_dY(tab, lnT[k+1, j, i], ub); qb = exp(lrb - ub)
                uk = ub - Δz * g * qb
                for _ in 1:30
                    lr, d, _ = H.lnrho_dY(tab, lnT[k, j, i], uk); qk = exp(lr - uk)
                    fu = uk - ub + Δz * g / 2 * (qk + qb); du = fu / (1 + Δz * g / 2 * qk * (d - 1))
                    uk -= du
                    abs(du) < 1e-12 && break
                end
                Pfull[k, j, i] = exp(uk)
            end
        end
    end
    cont_bad = gcount(par, count(!isfinite, @view Pfull[1:kc-1, :, :]))
    return (P = Pfull, Φ0 = Φ0, Φ = Φ, steps = steps, kc = kc, u0 = u0, u = u, reason = reason, gn_decrement = gnd,
            grad_norm = gn, cg_its = sits, cg_rel = srel, rms0 = sqrt(Φ0 / Wtot), rms = sqrt(Φ / Wtot),
            phi0_zero = phi0_zero, cont_bad = cont_bad)
end

"""ρ, τ₅₀₀ (exp_tail top closure) and z(τ₅₀₀ = 1) for every column of (ln T, P) on the height grid z (top first)."""
function tau_columns(tab, lnT, P, z; top = :exp_tail, H_max = 5.0e6, mu = 1.0)
    nz, ny, nx = size(P)
    rho = similar(P); tau = similar(P); zt1 = zeros(ny, nx)
    Threads.@threads for i in 1:nx
        chi = zeros(nz); tc = zeros(nz)
        @inbounds for j in 1:ny
            for k in 1:nz
                r, _, c, _ = H.state(tab, lnT[k, j, i], log(P[k, j, i]))
                rho[k, j, i] = exp(r); chi[k] = exp(c)
            end
            H.tau_column!(tc, chi, z; top = top, H_max = H_max, mu = mu)
            for k in 1:nz
                tau[k, j, i] = tc[k]
            end
            zt1[j, i] = H.z_tau1(tc, z)
        end
    end
    return rho, tau, zt1
end

end # module PoissonZ
