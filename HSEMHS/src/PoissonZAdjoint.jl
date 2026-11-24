# Adjoint of the Poisson step on the height grid (design/adjoint/03): the implicit-function derivative of the stationary
# point of Φ(u; ln T, L) under the anchor constraint (the mean of u over the anchor cells held at its start value), the
# transposes of R with respect to ln T and the Lorentz force, of J×B with respect to B, and of the hydrostatic
# continuation above z_cut. Included in module PoissonZ.

"""G = ∇ᵤΦ at u, exact (2 JᵀW R); sets pr's state buffers (q, qd, e^{−u}) to u."""
function gradphi!(out, pr::Prob, u, Rx, Ry, Rz)
    eos!(pr, pr.q, pr.qd, pr.ex, u)
    resid!(pr, Rx, Ry, Rz, u, pr.q, pr.ex)
    JtW!(pr, out, Rx, Ry, Rz)
    out .*= 2
    return out
end

"""2 JᵀW J v at the state in pr (the Gauss–Newton part of the Hessian; no damping, no anchor)."""
function gnop!(pr::Prob, out, v)
    jx = similar(v); jy = similar(v); jz = zeros(pr.n - 1, pr.ny, pr.nx)
    sddx!(jx, v, pr.dx, pr.par); sddy!(jy, v, pr.dy)
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny
            for k in 1:pr.n
                e = pr.ex[k, j, i] * v[k, j, i]
                jx[k, j, i] = -jx[k, j, i] - pr.lx[k, j, i] * e
                jy[k, j, i] = -jy[k, j, i] - pr.ly[k, j, i] * e
            end
            for f in 1:pr.n-1
                jz[f, j, i] = -(coef_a(pr, f, j, i) * v[f, j, i] + coef_b(pr, f, j, i) * v[f+1, j, i])
            end
        end
    end
    JtW!(pr, out, jx, jy, jz)
    out .*= 2
    return out
end

"""
∇ₐΦ at u for a = (ln T on the solved levels, L on the solved levels): (X̄, L̄x, L̄y, L̄z), and the layer sums of the
residuals (Sc[k] = Σ Rx² + Ry² on level k, Sf[f] = Σ Rz² on face f, global) that Φ weights by wc, wf.
"""
function gradphi_a(pr::Prob, u)
    n, ny, nx = pr.n, pr.ny, pr.nx; g = pr.g
    Rx, Ry, Rz = similar(u), similar(u), zeros(n - 1, ny, nx)
    eos!(pr, pr.q, pr.qd, pr.ex, u); resid!(pr, Rx, Ry, Rz, u, pr.q, pr.ex)
    Xb = zeros(n, ny, nx); Lxb = similar(u); Lyb = similar(u); Lzb = zeros(n, ny, nx)
    sc = zeros(n, nx); sf = zeros(n - 1, nx)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for k in 1:n
                Lxb[k, j, i] = 2 * pr.wc[k] * Rx[k, j, i] * pr.ex[k, j, i]
                Lyb[k, j, i] = 2 * pr.wc[k] * Ry[k, j, i] * pr.ex[k, j, i]
                sc[k, i] += Rx[k, j, i]^2 + Ry[k, j, i]^2
            end
            for f in 1:n-1
                r = 2 * pr.wf[f] * Rz[f, j, i]
                sf[f, i] += Rz[f, j, i]^2
                for (k, s) in ((f, 1), (f + 1, 1))
                    Lzb[k, j, i] += r * pr.ex[k, j, i] / 2
                    rX = H.state_d(pr.tab, pr.lT[k, j, i], u[k, j, i])[4]
                    Xb[k, j, i] -= r * g / 2 * pr.q[k, j, i] * rX
                end
            end
        end
    end
    return Xb, Lxb, Lyb, Lzb, gsum(pr.par, vec(sum(sc; dims = 2))), gsum(pr.par, vec(sum(sf; dims = 2)))
end

"""Transpose of `ddz!` (np.gradient-type ∂/∂z on a nonuniform top-first axis), added into `o`."""
function ddz_t!(o, f, z)
    mz, my, mx = size(f)
    Threads.@threads for i in 1:mx
        @inbounds for j in 1:my
            a = f[1, j, i] / (z[2] - z[1]); o[2, j, i] += a; o[1, j, i] -= a
            b = f[mz, j, i] / (z[mz] - z[mz-1]); o[mz, j, i] += b; o[mz-1, j, i] -= b
            for k in 2:mz-1
                d1 = z[k] - z[k-1]; d2 = z[k+1] - z[k]; v = f[k, j, i]
                o[k-1, j, i] += -d2 / (d1 * (d1 + d2)) * v
                o[k, j, i] += (d2 - d1) / (d1 * d2) * v
                o[k+1, j, i] += d1 / (d2 * (d1 + d2)) * v
            end
        end
    end
    return o
end

"""Adjoint of `lorentz` (J×B from cell-centred B): adds to (B̄x, B̄y, B̄z) for the cotangents (L̄x, L̄y, L̄z)."""
function lorentz_vjp!(Bb, par, Lxb, Lyb, Lzb, Bx, By, Bz, z, dx, dy)
    a = similar(Bx); b = similar(Bx)
    sddy!(a, Bz, dy); ddz!(b, By, z); Jx = (a .- b) ./ MU0
    ddz!(a, Bx, z); sddx!(b, Bz, dx, par); Jy = (a .- b) ./ MU0
    sddx!(a, By, dx, par); sddy!(b, Bx, dy); Jz = (a .- b) ./ MU0
    Bb[1] .+= Jz .* Lyb .- Jy .* Lzb
    Bb[2] .+= Jx .* Lzb .- Jz .* Lxb
    Bb[3] .+= Jy .* Lxb .- Jx .* Lyb
    Jxb = (By .* Lzb .- Bz .* Lyb) ./ MU0; Jyb = (Bz .* Lxb .- Bx .* Lzb) ./ MU0; Jzb = (Bx .* Lyb .- By .* Lxb) ./ MU0
    # Jx = Dy Bz − Dz By, Jy = Dz Bx − Dx Bz, Jz = Dx By − Dy Bx; Dxᵀ = −Dx, Dyᵀ = −Dy (centred, periodic)
    sddy!(a, Jxb, dy); Bb[3] .-= a; ddz_t!(Bb[2], -Jxb, z)
    ddz_t!(Bb[1], Jyb, z); sddx!(a, Jyb, dx, par); Bb[3] .+= a
    sddx!(a, Jzb, dx, par); Bb[2] .-= a; sddy!(a, Jzb, dy); Bb[1] .+= a
    return Bb
end

"""
Adjoint of the hydrostatic continuation above the cut (each u_k, k < kc, solves u_k − u_{k+1} + Δz g (q_k + q_{k+1})/2 = 0
from u_{k+1}): moves ū on levels 1…kc−1 onto ū[kc] and adds the ln T cotangents. u, lT: (nz, ny, nx) of the solve.
"""
function continuation_vjp!(Xb, ub, lT, u, z, tab, g, kc)
    nz, ny, nx = size(u)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, k in 1:kc-1
            Δz = z[k] - z[k+1]
            r0, _, _, rX0, rY0, _, _, _, _, _ = H.state_d(tab, lT[k, j, i], u[k, j, i])
            r1, _, _, rX1, rY1, _, _, _, _, _ = H.state_d(tab, lT[k+1, j, i], u[k+1, j, i])
            q0 = exp(r0 - u[k, j, i]); q1 = exp(r1 - u[k+1, j, i]); c = Δz * g / 2
            λ = ub[k, j, i] / (1 + c * q0 * (rY0 - 1))
            ub[k+1, j, i] -= λ * (-1 + c * q1 * (rY1 - 1))
            Xb[k, j, i] -= λ * c * q0 * rX0; Xb[k+1, j, i] -= λ * c * q1 * rX1
            ub[k, j, i] = 0.0
        end
    end
    return Xb, ub
end

"""
Implicit-function adjoint of the solved region (design/adjoint/03): for the cotangent ūR of u on the solved levels at the
returned state u, returns (X̄R, L̄x, L̄y, L̄z, ū0) — the cotangents of ln T and L on the solved levels and of the start ln P
(through the weights' median columns and the anchor constraint) — and the CG outcome. `gauss_newton = true` replaces
the Hessian by its Gauss–Newton part.
"""
function stationary_vjp(pr::Prob, u, u0, ub; cg_tol = 1e-10, cg_maxit = 1000, gauss_newton = false)
    par = pr.par; n, ny, nx = pr.n, pr.ny, pr.nx
    op = newton_ops(pr, u; gauss_newton = gauss_newton)
    c = op.c; cc = op.cc
    scale = max(gmax(par, maximum(abs, u)), 1.0)
    b = op.proj!(copy(ub))
    λ = zeros(n, ny, nx)
    flag = Ref(:none)
    its, rel = pcg!(op.Hop!, op.Mop!, par, pr, λ, b; tol = cg_tol, maxit = cg_maxit, flag = flag)
    κ = gsum(par, sum(c .* (ub .- op.hv!(similar(u), λ)))) / cc        # the anchor multiplier: H λ + κ c = ū
    # stationarity of u: the relative decrease ½ GᵀΔ/Φ one undamped Newton step (anchor mean fixed) predicts
    G = similar(u); Rx, Ry, Rz = similar(u), similar(u), zeros(n - 1, ny, nx)
    gradphi!(G, pr, u, Rx, Ry, Rz); Φ = phi(pr, Rx, Ry, Rz)
    δ = zeros(n, ny, nx)
    pcg!(op.Hop!, op.Mop!, par, pr, δ, op.proj!(-G); tol = cg_tol, maxit = cg_maxit)
    dec = -gdot(par, G, δ) / (2Φ)
    # mixed term −(∂G/∂a)ᵀ λ = −d/dε ∇ₐΦ(u + ελ), and the weights' share −d/dε (Sc, Sf)(u + ελ)
    vm = gmax(par, maximum(abs, λ)); ε = 1e-7 * max(scale, 1.0) / max(vm, 1e-300)
    p = gradphi_a(pr, u .+ ε .* λ); m = gradphi_a(pr, u .- ε .* λ)
    d(i) = (p[i] .- m[i]) ./ (2ε)
    Xb = -d(1); Lxb = -d(2); Lyb = -d(3); Lzb = -d(4); wcb = -d(5); wfb = -d(6)
    eos!(pr, pr.q, pr.qd, pr.ex, u)
    # weights: wc_k = w_k H̄_k², wf_f = h_f H̄_f H̄_f+1 with H̄ the layer medians of P/(ρ g) at the start
    qq = similar(u0); rX = similar(u0); rY = similar(u0)
    Threads.@threads for idx in eachindex(u0)
        @inbounds begin
            r, _, _, a, bb, _, _, _, _, _ = H.state_d(pr.tab, pr.lT[idx], u0[idx])
            qq[idx] = 1 / (pr.g * exp(r - u0[idx])); rX[idx] = a; rY[idx] = bb
        end
    end
    Hl = gmedians(par, qq)
    w = pr.wc ./ Hl .^ 2
    Hb = 2 .* w .* Hl .* wcb
    for f in 1:n-1
        Hb[f] += wfb[f] * pr.h[f] * Hl[f+1]; Hb[f+1] += wfb[f] * pr.h[f] * Hl[f]
    end
    u0b = κ .* c
    for k in 1:n
        wk = reshape(gquantile_w(par, vec(qq[k, :, :]), Hb[k]), ny, nx)
        @inbounds for i in 1:nx, j in 1:ny
            wk[j, i] == 0 && continue
            Xb[k, j, i] -= wk[j, i] * qq[k, j, i] * rX[k, j, i]
            u0b[k, j, i] -= wk[j, i] * qq[k, j, i] * (rY[k, j, i] - 1)
        end
    end
    return Xb, Lxb, Lyb, Lzb, u0b, (its = its, rel = rel, flag = flag[], kappa = κ, newton_decrement = dec)
end

"""The weights of one global median (`gquantiles` type 7 at p = 0.5) in the local values, times q̄ (PoissonTau's
`gquantile_vjp`, for this module)."""
function gquantile_w(par, vals::Vector{Float64}, qbar::Float64)
    N = Int(gsum(par, Float64(count(isfinite, vals))))
    h = (N - 1) * 0.5 + 1
    k1 = clamp(floor(Int, h), 1, max(1, N)); k2 = min(k1 + 1, max(1, N))
    a = Slab.gkth(par, [vals], [k1])[1]; b = Slab.gkth(par, [vals], [k2])[1]
    wa = 1 - (h - k1); wb = h - k1
    a == b && (wa += wb; wb = 0.0)
    na = gsum(par, Float64(count(==(a), vals))); nb = b == a ? 1.0 : gsum(par, Float64(count(==(b), vals)))
    return [v == a ? qbar * wa / na : v == b ? qbar * wb / nb : 0.0 for v in vals]
end

"""Φ at u (sets pr's state to u); Inf outside the table."""
function phi_at(pr::Prob, u)
    Rx, Ry, Rz = similar(u), similar(u), zeros(pr.n - 1, pr.ny, pr.nx)
    eos!(pr, pr.q, pr.qd, pr.ex, u) == 0 || return Inf
    resid!(pr, Rx, Ry, Rz, u, pr.q, pr.ex)
    return phi(pr, Rx, Ry, Rz)
end

"""
The projected Newton system at u: (P H P, P M⁻¹ P, P) with P the projection off the anchor indicator c (the anchor mean
fixed) and H the exact Hessian of Φ (central differences of G; `gauss_newton` its Gauss–Newton part).
"""
function newton_ops(pr::Prob, u; gauss_newton = false)
    par = pr.par; n, ny, nx = pr.n, pr.ny, pr.nx
    c = Float64.(pr.anc); cc = gsum(par, sum(c))
    proj!(v) = (a = gsum(par, sum(c .* v)) / cc; v .-= a .* c; v)
    Gp = similar(u); Gm = similar(u); Rx, Ry, Rz = similar(u), similar(u), zeros(n - 1, ny, nx)
    scale = max(gmax(par, maximum(abs, u)), 1.0)
    function hv!(out, v)
        vm = gmax(par, maximum(abs, v)); vm == 0 && return fill!(out, 0.0)
        if gauss_newton
            eos!(pr, pr.q, pr.qd, pr.ex, u); return gnop!(pr, out, v)
        end
        ε = 1e-7 * scale / vm
        gradphi!(Gp, pr, u .+ ε .* v, Rx, Ry, Rz); gradphi!(Gm, pr, u .- ε .* v, Rx, Ry, Rz)
        @. out = (Gp - Gm) / (2ε)
        return out
    end
    Hop!(_, out, v) = proj!(hv!(out, v))
    eos!(pr, pr.q, pr.qd, pr.ex, u); setprec!(pr)
    Mop!(_, out, r) = (Minv!(pr, out, proj!(copy(r))); proj!(out))
    return (Hop! = Hop!, Mop! = Mop!, hv! = hv!, proj! = proj!, c = c, cc = cc)
end

"""
    newton_polish!(pr, u, u0; steps = 5, cg_tol = 1e-10, cg_maxit = 1000) -> (steps, decrement)

Brings the solve's u to the stationary point that the adjoint differentiates (design/adjoint/03): the anchor-cell mean of
u back at that of the start u0 (a uniform shift), then Newton steps on Φ with the exact Hessian in the subspace that keeps
it, each backtracked on Φ. Returns the steps taken and the last relative Newton decrement ½ GᵀΔ/Φ.
"""
function newton_polish!(pr::Prob, u, u0; steps = 5, cg_tol = 1e-10, cg_maxit = 1000)
    par = pr.par
    c = Float64.(pr.anc); cc = gsum(par, sum(c))
    u .+= gsum(par, sum(c .* (u0 .- u))) / cc                   # restore the anchor mean
    G = similar(u); Rx, Ry, Rz = similar(u), similar(u), zeros(pr.n - 1, pr.ny, pr.nx)
    dec = NaN; taken = 0
    for it in 1:steps
        op = newton_ops(pr, u)
        gradphi!(G, pr, u, Rx, Ry, Rz); Φ = phi(pr, Rx, Ry, Rz)
        b = op.proj!(-G)
        δ = zeros(size(u))
        pcg!(op.Hop!, op.Mop!, par, pr, δ, b; tol = cg_tol, maxit = cg_maxit)
        dec = -gdot(par, G, δ) / (2Φ)
        (isfinite(dec) && dec > 0) || break
        α = 1.0
        while α > 1e-4 && !(phi_at(pr, u .+ α .* δ) < Φ)
            α /= 2
        end
        α > 1e-4 || break
        u .+= α .* δ; taken = it
        dec < 1e-14 && break
    end
    eos!(pr, pr.q, pr.qd, pr.ex, u)
    return taken, dec
end
