# Adjoint of the column HSE on the τ and mass axes (design/adjoint/02): the Heun column with its substeps frozen, the
# τ-hydrostatic top closure by the implicit function theorem, the ptot alignment (with its median reference), the gauge,
# and log τ on the mass axis. Included in module PoissonTau.

"""
Substeps of one `hse_tau!` column, in order: the interval k, which bound set the step (0: the end of the interval, b − x;
1: the slope limit 0.05/|f₁|) and how often it was then quartered. With these the steps are a smooth function of the
state, so the replay and its adjoint follow the solver itself, step sizes included.
"""
struct HeunTape
    k::Vector{Int32}
    br::Vector{UInt8}
    nq::Vector{Int8}
end
HeunTape() = HeunTape(Int32[], UInt8[], Int8[])
Base.empty!(t::HeunTape) = (empty!(t.k); empty!(t.br); empty!(t.nq); t)
record!(t::HeunTape, k, br, nq) = (push!(t.k, k); push!(t.br, br); push!(t.nq, nq); t)

"""The right-hand sides of `hse_tau!` (d ln P/dℓ, dz/dℓ) and their X and ln P partials, from the table interpolant (both
are ∝ e^ℓ, so their ℓ partials are themselves)."""
@inline function rhs_d(tab, X, lτ, l, mu, g, mass)
    r, _, c, rX, rY, _, _, cX, cY, _ = H.state_d(tab, X, l)
    cc, ccX, ccY = mass ? (r, rX, rY) : (c, cX, cY)
    χ = exp(cc)
    f = mu * g * exp(lτ) * exp(r - l) / χ
    gz = -mu * exp(lτ) / χ
    return f, gz, f * (rX - ccX), f * (rY - 1 - ccY), -gz * ccX, -gz * ccY
end

"""ln P at the top level of the τ axis: P = μ g τ ρ/χ, as `hse_tau!` iterates it, then Newton to round-off."""
function top_closure(tab, X, τ1, mu, g)
    lp = log(1e2)
    for _ in 1:200
        q, χ = q_chi(tab, X, lp)
        new = log(mu * g * τ1 * q / χ) + lp
        abs(new - lp) < 1e-10 && break
        lp = 0.5 * lp + 0.5 * new
    end
    for _ in 1:4
        r, _, c, _, rY, _, _, _, cY, _ = H.state_d(tab, X, lp)
        G = lp - log(mu * g * τ1) - r + c
        lp -= G / (1 - rY + cY)
    end
    return lp
end

"""Step length of a recorded substep at (x, f₁): the bound `br` divided by 4^nq, as `hse_tau!` computes it."""
@inline function heun_h(br, nq, b, x, f1)
    h = br == 0 ? b - x : 0.05 / max(abs(f1), 1e-30)
    for _ in 1:nq
        h /= 4
    end
    return h
end

"""
    hse_tau_replay!(lnP, z, lT, lt, tab, g, tape; mu=1.0, mass=false) -> Bool

`hse_tau!` with the step structure of `tape` (the bound and the quartering of each substep; the lengths follow the state)
and the top closure solved to round-off: the map whose derivative `hse_tau_vjp!` is. False if the state leaves the table.
`st` (optional) receives (x, ln P) at the start of every substep.
"""
function hse_tau_replay!(lnP, z, lT, lt, tab, g, tape::HeunTape; mu = 1.0, mass = false, st = nothing)
    nt = length(lt); ns = length(tape.k)
    τ1 = exp(LN10 * lt[1])
    l = mass ? log(mu * g * τ1) : top_closure(tab, lT[1], τ1, mu, g)
    lnP[1] = l; z[1] = 0.0; zz = 0.0
    st === nothing || (resize!(st[1], ns); resize!(st[2], ns))
    x = 0.0
    @inbounds for s in 1:ns
        k = tape.k[s]
        a, b = LN10 * lt[k], LN10 * lt[k+1]
        (s == 1 || tape.k[s-1] != k) && (x = a)
        st === nothing || (st[1][s] = x; st[2][s] = l)
        sx = (x - a) / (b - a)
        f1, g1 = rhs_d(tab, lT[k] + (lT[k+1] - lT[k]) * sx, x, l, mu, g, mass)
        h = heun_h(tape.br[s], tape.nq[s], b, x, f1)
        sy = (x + h - a) / (b - a)
        f2, g2 = rhs_d(tab, lT[k] + (lT[k+1] - lT[k]) * sy, x + h, l + h * f1, mu, g, mass)
        l = l + h / 2 * (f1 + f2); zz += h / 2 * (g1 + g2); x += h
        if s == ns || tape.k[s+1] != k
            lnP[k+1] = l; z[k+1] = zz
        end
    end
    z .-= interp_lt(z, lt, 0.0)
    return all(isfinite, lnP) && all(isfinite, z)
end

"""(k, t) of `interp_lt(v, lt, L)`: the value is (1 − t) v[k] + t v[k+1]."""
@inline function lt_bracket(lt, L)
    n = length(lt)
    k = clamp(searchsortedlast(lt, L), 1, n - 1)
    return k, (L - lt[k]) / (lt[k+1] - lt[k])
end

"""Adjoint of `interp_lt(v, lt, L)` for the cotangent ȳ: adds to v̄ and (if given) to the axis cotangent l̄t."""
function interp_lt_vjp!(vbar, ltbar, ybar, v, lt, L)
    k, t = lt_bracket(lt, L)
    @inbounds begin
        vbar[k] += ybar * (1 - t); vbar[k+1] += ybar * t
        if ltbar !== nothing
            D = lt[k+1] - lt[k]; dv = v[k+1] - v[k]
            ltbar[k] += ybar * dv * (t - 1) / D; ltbar[k+1] -= ybar * dv * t / D
        end
    end
    return vbar
end

"""
    hse_tau_vjp!(Xbar, ubar, zbar, lT, lt, tab, g, tape; mu=1.0, mass=false) -> Xbar

Adjoint of `hse_tau_replay!` for the cotangents (ū, z̄) of its (ln P, z): adds ∂ℓ/∂(ln T) to `Xbar`. The substep state is
(x, ln P, z); x restarts at the left end of every interval, so its cotangent does not cross intervals.
"""
function hse_tau_vjp!(Xbar, ubar, zbar, lT, lt, tab, g, tape::HeunTape; mu = 1.0, mass = false)
    nt = length(lt); ns = length(tape.k)
    lnP = zeros(nt); z = zeros(nt); st = (Float64[], Float64[])
    hse_tau_replay!(lnP, z, lT, lt, tab, g, tape; mu = mu, mass = mass, st = st)
    zr = collect(Float64, zbar)                         # z_out = z_raw − interp(z_raw, lt, 0)
    k0, t0 = lt_bracket(lt, 0.0); S = sum(zbar)
    zr[k0] -= S * (1 - t0); zr[k0+1] -= S * t0
    lb = 0.0; zb = 0.0; xb = 0.0
    @inbounds for s in ns:-1:1
        k = tape.k[s]
        if s == ns || tape.k[s+1] != k                  # the last substep of interval k: its outputs are node k+1
            lb += ubar[k+1]; zb += zr[k+1]; xb = 0.0
        end
        a, b = LN10 * lt[k], LN10 * lt[k+1]; dXs = (lT[k+1] - lT[k]) / (b - a)
        x = st[1][s]; l = st[2][s]
        sx = (x - a) / (b - a)
        f1, g1, f1X, f1l, g1X, g1l = rhs_d(tab, lT[k] + (lT[k+1] - lT[k]) * sx, x, l, mu, g, mass)
        h = heun_h(tape.br[s], tape.nq[s], b, x, f1)
        x2 = x + h; sy = (x2 - a) / (b - a); l2 = l + h * f1
        f2, g2, f2X, f2l, g2X, g2l = rhs_d(tab, lT[k] + (lT[k+1] - lT[k]) * sy, x2, l2, mu, g, mass)
        # outputs l' = l + h/2 (f1 + f2), z' = z + h/2 (g1 + g2), x' = x2
        fb1 = h / 2 * lb; fb2 = fb1; gb1 = h / 2 * zb; gb2 = gb1
        hb = (f1 + f2) / 2 * lb + (g1 + g2) / 2 * zb
        x2b = xb
        # stage 2 at (X2, x2, l2)
        l2b = fb2 * f2l + gb2 * g2l
        X2b = fb2 * f2X + gb2 * g2X
        x2b += fb2 * f2 + gb2 * g2 + X2b * dXs
        Xbar[k] += X2b * (1 - sy); Xbar[k+1] += X2b * sy
        # l2 = l + h f1, x2 = x + h
        hb += l2b * f1 + x2b; fb1 += h * l2b
        xbn = x2b; lbn = lb + l2b
        # h = bound / 4^nq
        q = h / (tape.br[s] == 0 ? b - x : 0.05 / max(abs(f1), 1e-30))
        if tape.br[s] == 0
            xbn -= hb * q
        elseif abs(f1) > 1e-30
            fb1 -= hb * h / f1
        end
        # stage 1 at (X1, x, l)
        X1b = fb1 * f1X + gb1 * g1X
        lbn += fb1 * f1l + gb1 * g1l
        xbn += fb1 * f1 + gb1 * g1 + X1b * dXs
        Xbar[k] += X1b * (1 - sx); Xbar[k+1] += X1b * sx
        lb = lbn; xb = xbn                                # zb passes unchanged
    end
    lb += ubar[1]
    if !mass                                            # top: lp − ln(μ g τ₁) − r + c = 0
        _, _, _, rX, rY, _, _, cX, cY, _ = H.state_d(tab, lT[1], lnP[1])
        Xbar[1] += lb * (rX - cX) / (1 - rY + cY)
    end
    return Xbar
end

"""
    logtau_mass_vjp!(Xbar, Ybar, lbar, lT, lnP, lm, tab)

Adjoint of `logtau_mass` (all (nt, ny, nx)): adds ∂ℓ/∂(ln T) and ∂ℓ/∂(ln P) for the cotangent `lbar` of log10 τ.
"""
function logtau_mass_vjp!(Xbar, Ybar, lbar, lT, lnP, lm, tab)
    nt, ny, nx = size(lT)
    lnm = LN10 .* lm
    Threads.@threads for i in 1:nx
        f = zeros(nt); fX = zeros(nt); fY = zeros(nt); τ = zeros(nt); fb = zeros(nt)
        @inbounds for j in 1:ny
            for k in 1:nt
                r, _, c, rX, rY, _, _, cX, cY, _ = H.state_d(tab, lT[k, j, i], lnP[k, j, i])
                f[k] = exp(c - r + lnm[k]); fX[k] = f[k] * (cX - rX); fY[k] = f[k] * (cY - rY)
                τ[k] = k == 1 ? f[1] : τ[k-1] + (lnm[k] - lnm[k-1]) / 2 * (f[k-1] + f[k])
            end
            S = 0.0; fill!(fb, 0.0)
            for k in nt:-1:1
                S += lbar[k, j, i] / (τ[k] * LN10)
                if k == 1
                    fb[1] += S
                else
                    w = (lnm[k] - lnm[k-1]) / 2 * S
                    fb[k] += w; fb[k-1] += w
                end
            end
            for k in 1:nt
                Xbar[k, j, i] += fb[k] * fX[k]; Ybar[k, j, i] += fb[k] * fY[k]
            end
        end
    end
    return Xbar, Ybar
end

"""
Adjoint of one global quantile (`gquantiles`, type 7) for the cotangent q̄ (the same on every rank): the weight of each
local value in `vals` (the order statistics' share, split equally among exact ties).
"""
function gquantile_vjp(par, vals::Vector{Float64}, p::Real, qbar::Float64)
    N = Int(gsum(par, Float64(count(isfinite, vals))))
    h = (N - 1) * p + 1
    k1 = clamp(floor(Int, h), 1, max(1, N)); k2 = min(k1 + 1, max(1, N))
    a = Slab.gkth(par, [vals], [k1])[1]; b = Slab.gkth(par, [vals], [k2])[1]
    wa = 1 - (h - k1); wb = h - k1
    a == b && (wa += wb; wb = 0.0)
    na = gsum(par, Float64(count(==(a), vals))); nb = b == a ? 1.0 : gsum(par, Float64(count(==(b), vals)))
    return [v == a ? qbar * wa / na : v == b ? qbar * wb / nb : 0.0 for v in vals]
end

"""
Adjoint of `align_ptot!` (collective). `zbar` is the cotangent of the aligned heights and becomes that of the heights
before alignment (`z_col`); the cotangents of ln P, ln T and B are added to `ubar`, `Xbar`, `Bbar` (a 3-tuple).
"""
function align_ptot_vjp!(par, zbar, ubar, Xbar, Bbar, z_col, lnP, lT, Bx, By, Bz, tab, g)
    nt, ny, nx = size(z_col)
    Bm(k, j, i) = (Bx[k, j, i]^2 + By[k, j, i]^2 + Bz[k, j, i]^2) / (2MU0)
    lpt(k, j, i) = log(exp(lnP[k, j, i]) + Bm(k, j, i))
    function lpt_vjp!(w, k, j, i)                        # adds w ∂lpt/∂(u, B)
        P = exp(lnP[k, j, i]); D = P + Bm(k, j, i)
        ubar[k, j, i] += w * P / D
        Bbar[1][k, j, i] += w * Bx[k, j, i] / (MU0 * D)
        Bbar[2][k, j, i] += w * By[k, j, i] / (MU0 * D)
        Bbar[3][k, j, i] += w * Bz[k, j, i] / (MU0 * D)
    end
    bottoms = [lpt(nt, j, i) for j in 1:ny, i in 1:nx][:]
    ref = gquantiles(par, [bottoms], 0.5)[1]
    refbar = zeros(nx)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            zrb = -sum(@view zbar[:, j, i])                  # z_out = z_col − z_r
            lb = lpt(nt, j, i)
            if lb < ref
                r, _, _, rX, rY, _, _, _, _, _ = H.state_d(tab, lT[nt, j, i], lnP[nt, j, i])
                q = exp(r - lnP[nt, j, i]); d = (ref - lb) / (g * q)
                zbar[nt, j, i] += zrb
                refbar[i] -= zrb / (g * q)
                lpt_vjp!(zrb / (g * q), nt, j, i)
                Xbar[nt, j, i] += zrb * d * rX; ubar[nt, j, i] += zrb * d * (rY - 1)
            else
                k = nt
                while k > 1 && lpt(k - 1, j, i) >= ref
                    k -= 1
                end
                if k == 1
                    zbar[1, j, i] += zrb
                else
                    l0 = lpt(k - 1, j, i); l1 = lpt(k, j, i); D = l1 - l0; θ = (ref - l0) / D
                    dz = z_col[k, j, i] - z_col[k-1, j, i]
                    zbar[k-1, j, i] += zrb * (1 - θ); zbar[k, j, i] += zrb * θ
                    refbar[i] += zrb * dz / D
                    lpt_vjp!(zrb * dz * (θ - 1) / D, k - 1, j, i)
                    lpt_vjp!(-zrb * dz * θ / D, k, j, i)
                end
            end
        end
    end
    rb = gsum(par, sum(refbar))
    w = gquantile_vjp(par, bottoms, 0.5, rb)
    @inbounds for i in 1:nx, j in 1:ny
        c = (i - 1) * ny + j
        w[c] == 0 || lpt_vjp!(w[c], nt, j, i)
    end
    return zbar
end

"""
Adjoint of `gauge!` (collective): `zbar` (cotangent of the gauged heights) is unchanged (the shift is subtracted from
every height) and each column's z(τ = 1) receives −Σ z̄/(NY NX), passed to its bracket in `z` (the heights before the
gauge) and, on the mass axis, to the cotangent `ltbar` of the log10 τ it is interpolated on.
"""
function gauge_vjp!(par, zbar, z, lt, logtau = nothing, ltbar = nothing)
    nt, ny, nx = size(z)
    sb = -gsum(par, sum(zbar)) / (ny * par.NX)
    @inbounds for i in 1:nx, j in 1:ny
        lc = logtau === nothing ? lt : @view(logtau[:, j, i])
        interp_lt_vjp!(@view(zbar[:, j, i]), logtau === nothing ? nothing : @view(ltbar[:, j, i]), sb,
                       @view(z[:, j, i]), lc, 0.0)
    end
    return zbar
end
