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

# ======================================================================================================================
# Poisson on the τ and mass axes (design/adjoint/03, Phase 5): reverse mode of the residuals of `evaluate!`
# ======================================================================================================================

"""Adjoint of `dzcol!` (o = ∂f/∂z down each column): adds to f̄ and z̄ for the cotangent ō."""
function dzcol_vjp!(fb, zb, ob, f, z)
    nt, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, k in 1:nt
            a, b = k == 1 ? (1, 2) : k == nt ? (nt - 1, nt) : (k - 1, k + 1)
            D = z[a, j, i] - z[b, j, i]; o = (f[a, j, i] - f[b, j, i]) / D; w = ob[k, j, i]
            fb[a, j, i] += w / D; fb[b, j, i] -= w / D
            zb[a, j, i] -= w * o / D; zb[b, j, i] += w * o / D
        end
    end
    return fb, zb
end

"""
Adjoint of `lorentz_tau!` at the heights pr.z: for the cotangents L̄ (nt, ny, nx, the three components) returns (B̄x, B̄y,
B̄z) and adds the height cotangent to `zb` (the curvilinear derivatives ∂/∂x|_z = ∂/∂x|_τ − ∂ₓz ∂_ray and the ray
derivative depend on z).
"""
function lorentz_tau_vjp(pr::TP, Lxb, Lyb, Lzb, zb)
    z = pr.z; par = pr.par; tx = pr.tx; ty = pr.ty; sl = inclined(pr)
    zx = sddx!(similar(z), z, pr.dx, par); zy = sddy!(similar(z), z, pr.dy)
    function d(B)
        fz = dzcol!(similar(B), B, z); fx = sddx!(similar(B), B, pr.dx, par); fy = sddy!(similar(B), B, pr.dy)
        @. fx -= zx * fz; @. fy -= zy * fz
        return fx, fy, sl ? (@. fz - tx * fx - ty * fy) : copy(fz), fz
    end
    Bx, By, Bz = pr.Bx, pr.By, pr.Bz
    (Bxx, Bxy, Bxv, _), (Byx, Byy, Byv, _), (Bzx, Bzy, Bzv, _) = d(Bx), d(By), d(Bz)
    Jx = (Bzy .- Byv) ./ MU0; Jy = (Bxv .- Bzx) ./ MU0; Jz = (Byx .- Bxy) ./ MU0
    Bbx = Jz .* Lyb .- Jy .* Lzb; Bby = Jx .* Lzb .- Jz .* Lxb; Bbz = Jy .* Lxb .- Jx .* Lyb
    Jxb = (By .* Lzb .- Bz .* Lyb) ./ MU0; Jyb = (Bz .* Lxb .- Bx .* Lzb) ./ MU0; Jzb = (Bx .* Lyb .- By .* Lxb) ./ MU0
    Z() = zeros(size(z))
    # (f̄x, f̄y, f̄v) of each component's derivatives
    fbx = (Z(), -Jzb, Jyb); fby = (Jzb, Z(), -Jxb); fbz = (-Jyb, Jxb, Z())
    zxb = Z(); zyb = Z(); t = Z()
    for (B, Bb, (fxb, fyb, fvb)) in ((Bx, Bbx, fbx), (By, Bby, fby), (Bz, Bbz, fbz))
        fz = dzcol!(similar(B), B, z)
        fzb = copy(fvb)
        if sl
            fxb = fxb .- tx .* fvb; fyb = fyb .- ty .* fvb
        end
        sddx!(t, fxb, pr.dx, par); Bb .-= t; @. zxb -= fxb * fz; @. fzb -= zx * fxb
        sddy!(t, fyb, pr.dy); Bb .-= t; @. zyb -= fyb * fz; @. fzb -= zy * fyb
        dzcol_vjp!(Bb, zb, fzb, B, z)
    end
    sddx!(t, zxb, pr.dx, par); zb .-= t; sddy!(t, zyb, pr.dy); zb .-= t
    return Bbx, Bby, Bbz
end

"""
    resid_vjp(pr, u, c) -> NamedTuple

Φ at (u, c) (`evaluate!`, which sets pr's state) and its exact gradient in every input, in one reverse pass through the
residuals: `ubar` (nt, ny, nx; nonzero on k ≥ kc, ln P at kc being fixed), `cbar` (ny, nx; the cut heights), `Xbar` (ln T),
`Bbar` (3-tuple), `zrelbar` (kc, ny, nx; the fixed relative heights above the cut), and the layer sums of the residuals
that the weights multiply, `Sc` (n1: Σ Rx² + Ry²) and `Sf` (n1 − 1: Σ R_z², with the vertical balance), global.
"""
function resid_vjp(pr::TP, u, c)
    Φ, bad = evaluate!(pr, u, c)
    bad == 0 || H.invalid("resid_vjp: the state is outside the EOS table")
    nt, kc, n1, ny, nx = pr.nt, pr.kc, pr.n1, pr.ny, pr.nx
    tx = pr.tx; ty = pr.ty; sl = inclined(pr); vt = vertical(pr); g = pr.g; par = pr.par
    rx = pr.wc .* pr.Rx .* 2; ry = pr.wc .* pr.Ry .* 2
    rz = zeros(n1, ny, nx); @views rz[1:n1-1, :, :] .= 2 .* pr.wf .* pr.Rz[1:n1-1, :, :]
    Z3() = zeros(nt, ny, nx); Zn() = zeros(n1, ny, nx)
    ub = Z3(); Xb = Z3(); zb = Z3(); cb = zeros(ny, nx)
    Lxb = Z3(); Lyb = Z3(); Lzb = Z3()
    Wb = Zn(); ab = Zn(); eb = Zn(); qb = Zn()
    un = u[kc:nt, :, :]; zn = pr.z[kc:nt, :, :]
    zx = sddx!(similar(zn), zn, pr.dx, par); zy = sddy!(similar(zn), zn, pr.dy)
    sc = zeros(n1, nx); sf = zeros(n1, nx)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1
                sc[m, i] += pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2
                m < n1 && (sf[m, i] += pr.Rz[m, j, i]^2)
            end
            if vt                                       # R_v = R_z − Δz t·R̄_h
                for f in 1:n1-1
                    k = node(pr, f)
                    db = -rz[f, j, i] * pr.vq[f, j, i]; qbv = -rz[f, j, i] * pr.vd[f, j, i]
                    rx[f, j, i] += 0.5 * tx * qbv; rx[f+1, j, i] += 0.5 * tx * qbv
                    ry[f, j, i] += 0.5 * ty * qbv; ry[f+1, j, i] += 0.5 * ty * qbv
                    zb[k, j, i] += db; zb[k+1, j, i] -= db
                end
            end
            for f in 1:n1-1                              # R_z = Δu − Δℓ/2 (W a + W a)
                k = node(pr, f); r = rz[f, j, i]; h = pr.dl[k] / 2
                ub[k+1, j, i] += r; ub[k, j, i] -= r
                for m in (f, f + 1)
                    Wb[m, j, i] -= h * r * pr.a[m, j, i]; ab[m, j, i] -= h * r * pr.W[m, j, i]
                end
            end
            for m in 1:n1                                # R_x = −∂ₓu − W ∂ₓz + Lₓ e, R_y likewise
                k = node(pr, m); e = exp(-u[k, j, i])
                Wb[m, j, i] -= rx[m, j, i] * zx[m, j, i] + ry[m, j, i] * zy[m, j, i]
                Lxb[k, j, i] += rx[m, j, i] * e; Lyb[k, j, i] += ry[m, j, i] * e
                eb[m, j, i] += rx[m, j, i] * pr.Lx[k, j, i] + ry[m, j, i] * pr.Ly[k, j, i]
                Ln = sl ? pr.Lz[k, j, i] + tx * pr.Lx[k, j, i] + ty * pr.Ly[k, j, i] : pr.Lz[k, j, i]
                qb[m, j, i] += g * Wb[m, j, i]                # W = g q − Lₙ e
                Lnb = -Wb[m, j, i] * e; eb[m, j, i] -= Wb[m, j, i] * Ln
                Lzb[k, j, i] += Lnb
                sl && (Lxb[k, j, i] += tx * Lnb; Lyb[k, j, i] += ty * Lnb)
            end
        end
    end
    # the differences: −Dₓu and −W Dₓz (Dᵀ = −D)
    t = similar(un)
    sddx!(t, rx, pr.dx, par); @views ub[kc:nt, :, :] .+= t
    sddy!(t, ry, pr.dy); @views ub[kc:nt, :, :] .+= t
    sddx!(t, rx .* pr.W, pr.dx, par); @views zb[kc:nt, :, :] .+= t
    sddy!(t, ry .* pr.W, pr.dy); @views zb[kc:nt, :, :] .+= t
    Bb = lorentz_tau_vjp(pr, Lxb, Lyb, Lzb, zb)
    zrelb = zeros(kc, ny, nx)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for k in 1:kc-1                              # z_k = c + zrel_k above the cut
                cb[j, i] += zb[k, j, i]; zrelb[k, j, i] = zb[k, j, i]
            end
            Zs = 0.0                                     # z_{k+1} = z_k − Δℓ/2 (a_m + a_{m+1}) below
            for k in nt-1:-1:kc
                Zs += zb[k+1, j, i]; m = k - kc + 1; h = pr.dl[k] / 2
                ab[m, j, i] -= h * Zs; ab[m+1, j, i] -= h * Zs
            end
            cb[j, i] += zb[kc, j, i] + Zs
            for m in 1:n1                                # q = ρ/P, a = μ τ/χ (μ m/ρ), e = e^{−u}
                k = node(pr, m)
                _, _, _, rX, rY, _, _, cX, cY, _ = H.state_d(pr.tab, pr.lT[k, j, i], u[k, j, i])
                ccX, ccY = pr.mass ? (rX, rY) : (cX, cY)
                rb = qb[m, j, i] * pr.q[m, j, i]; ccb = -ab[m, j, i] * pr.a[m, j, i]
                Xb[k, j, i] += rb * rX + ccb * ccX
                ub[k, j, i] += rb * (rY - 1) + ccb * ccY - eb[m, j, i] * exp(-u[k, j, i])
            end
        end
    end
    return (phi = Φ, ubar = ub, cbar = cb, Xbar = Xb, Bbar = Bb, zrelbar = zrelb,
            Sc = gsum(par, vec(sum(sc; dims = 2))), Sf = gsum(par, vec(sum(sf; dims = 2)))[1:n1-1])
end

"""Relative step of the central differences that give Hessian–vector and mixed products on the τ grid."""
const HVP_STEP = Ref(1e-5)

"""The unknowns of the τ-grid solve as one array (n1, ny, nx): slot 1 the cut heights c, slots 2… ln P below the cut."""
function to_slots(pr::TP, cb, ub)
    v = zeros(pr.n1, pr.ny, pr.nx)
    v[1, :, :] .= cb
    @views v[2:end, :, :] .= ub[pr.kc+1:pr.nt, :, :]
    return v
end
"""(u, c) moved by ε along the slots v."""
function along(pr::TP, u, c, v, ε)
    u2 = copy(u); @views u2[pr.kc+1:pr.nt, :, :] .+= ε .* v[2:end, :, :]
    return u2, c .+ ε .* v[1, :, :]
end

"""
Adjoint of the heights at the evaluated state of pr (z_kc = c, z_k = c + zrel_k above, the trapezoid of a below) for the
height cotangent zb (nt): returns (c̄, z̄rel, ū, X̄).
"""
function heights_vjp(pr::TP, u, zb)
    nt, kc, n1, ny, nx = pr.nt, pr.kc, pr.n1, pr.ny, pr.nx
    cb = zeros(ny, nx); zrelb = zeros(kc, ny, nx); ub = zeros(nt, ny, nx); Xb = zeros(nt, ny, nx)
    Threads.@threads for i in 1:nx
        ab = zeros(n1)
        @inbounds for j in 1:ny
            fill!(ab, 0.0)
            for k in 1:kc-1
                cb[j, i] += zb[k, j, i]; zrelb[k, j, i] = zb[k, j, i]
            end
            Zs = 0.0
            for k in nt-1:-1:kc
                Zs += zb[k+1, j, i]; m = k - kc + 1; h = pr.dl[k] / 2
                ab[m] -= h * Zs; ab[m+1] -= h * Zs
            end
            cb[j, i] += zb[kc, j, i] + Zs
            for m in 1:n1
                k = node(pr, m)
                _, _, _, rX, rY, _, _, cX, cY, _ = H.state_d(pr.tab, pr.lT[k, j, i], u[k, j, i])
                ccX, ccY = pr.mass ? (rX, rY) : (cX, cY)
                w = -ab[m] * pr.a[m, j, i]
                Xb[k, j, i] += w * ccX; ub[k, j, i] += w * ccY
            end
        end
    end
    return cb, zrelb, ub, Xb
end

"""
The null directions of Φ in the cut heights c (ny, nx each, this slab): c enters Φ only through the centred horizontal
differences of the heights, which annihilate the uniform shift and the Nyquist modes (−1)^x, (−1)^y, (−1)^(x+y) of an even
global dimension. They are mutually orthogonal over the global grid.
"""
function c_nulls(pr::TP)
    par = pr.par; b = par.block; ny, nx = pr.ny, pr.nx
    gx(i) = b.i0 + i - 1; gy(j) = b.j0 + j - 1
    N = [ones(ny, nx)]
    iseven(par.NY) && push!(N, [(-1.0)^gy(j) for j in 1:ny, i in 1:nx])
    iseven(par.NX) && push!(N, [(-1.0)^gx(i) for j in 1:ny, i in 1:nx])
    iseven(par.NX) && iseven(par.NY) && push!(N, [(-1.0)^(gx(i) + gy(j)) for j in 1:ny, i in 1:nx])
    return N
end

"""
    tau_newton_ops(pr, u, c; gauss_newton=false)

The Newton system of the τ-grid solve at (u, c) on the slots (`to_slots`): G (the exact gradient, from `resid_vjp`), the
Hessian–vector product (central differences of G; `gauss_newton`: 2 JᵀWJ from the solver's own Jacobian), the solver's
preconditioner, and the projection off the null direction (a uniform shift of every cut height leaves Φ unchanged).
"""
function tau_newton_ops(pr::TP, u, c; gauss_newton = false)
    par = pr.par
    grad(uu, cc) = (g = resid_vjp(pr, uu, cc); to_slots(pr, g.cbar, g.ubar))
    nulls = c_nulls(pr); nn = [gsum(par, sum(N .^ 2)) for N in nulls]
    proj!(v) = (for (N, n2) in zip(nulls, nn); a = gsum(par, sum(N .* @view v[1, :, :])) / n2; @views v[1, :, :] .-= a .* N; end; v)
    su = max(gmax(par, maximum(abs, @view u[pr.kc:pr.nt, :, :])), 1.0); sc = 1e5      # ln P and height scales
    function hv!(out, v)
        vm = max(gmax(par, maximum(abs, @view v[2:end, :, :])) / su, gmax(par, maximum(abs, @view v[1, :, :])) / sc)
        vm == 0 && return fill!(out, 0.0)
        if gauss_newton
            evaluate!(pr, u, c)
            jx = similar(v); jy = similar(v); jz = similar(v)
            Jop!(pr, jx, jy, jz, v); jx .*= pr.wc; jy .*= pr.wc; jz .*= vcat(pr.wf, 0.0)
            JTop!(pr, out, jx, jy, jz); out .*= 2
            return out
        end
        ε = HVP_STEP[] / vm
        up, cp = along(pr, u, c, v, ε); um, cm = along(pr, u, c, v, -ε)
        out .= (grad(up, cp) .- grad(um, cm)) ./ (2ε)
        return out
    end
    Hop!(_, out, v) = proj!(hv!(out, v))
    # Preconditioner: the exact column blocks of the Gauss–Newton matrix (block Jacobi). The solver's blocks carry its
    # damping ereg, which over-stiffens the cut heights by orders of magnitude for this undamped system (their curvature
    # is ~1e10 below that of ln P per unit); here ln P is undamped and c gets its measured curvature, a Rayleigh quotient
    # of H along a random zero-mean direction of c (H = 2 JᵀWJ + …, the blocks JᵀWJ: half of it).
    evaluate!(pr, u, c)
    if pr.n1 >= 2 && pr.nx >= 3 && pr.ny >= 3
        rng = H.Random.MersenneTwister(20261124)
        v = zeros(pr.n1, pr.ny, pr.nx); v[1, :, :] .= randn(rng, pr.ny, pr.nx); proj!(v)
        κc = gdot(par, v, hv!(similar(v), v)) / gdot(par, v, v)
        evaluate!(pr, u, c)
        e0 = copy(pr.ereg); m0 = pr.mlm; p0 = pr.prec
        pr.ereg .= 0.0; pr.ereg[1] = max(κc / 2, 0.0); pr.mlm = 0.0; pr.prec = :block
        colblocks!(pr)
        pr.ereg .= e0; pr.mlm = m0
        Mop! = (_, out, r) -> (block_solve!(out, pr.blk, proj!(copy(r))); proj!(out))
    else
        setprec!(pr)
        Mop! = (_, out, r) -> (Minv!(pr, out, proj!(copy(r))); proj!(out))
    end
    return (Hop! = Hop!, Mop! = Mop!, hv! = hv!, proj! = proj!, grad = grad, su = su, sc = sc, nulls = nulls, nn = nn)
end

"""
    tau_newton_polish!(pr, u, c; steps=5, cg_tol=1e-10, cg_maxit=1000) -> (steps, decrement)

Newton steps on Φ (exact Hessian, the null direction projected out), each backtracked on Φ, from the solve's (u, c):
the stationary point the adjoint differentiates. u and c are updated in place.
"""
function tau_newton_polish!(pr::TP, u, c; steps = 5, cg_tol = 1e-10, cg_maxit = 1000)
    dec = NaN; taken = 0
    for it in 1:steps
        op = tau_newton_ops(pr, u, c)
        G = op.grad(u, c); Φ = evaluate!(pr, u, c)[1]
        δ = zeros(size(G))
        pcg!(op.Hop!, op.Mop!, pr.par, pr, δ, op.proj!(-G); tol = cg_tol, maxit = cg_maxit)
        dec = -gdot(pr.par, G, δ) / (2Φ)
        (isfinite(dec) && dec > 0) || break
        α = 1.0
        while α > 1e-4
            ut, ct = along(pr, u, c, δ, α)
            evaluate!(pr, ut, ct)[1] < Φ && break
            α /= 2
        end
        α > 1e-4 || break
        ut, ct = along(pr, u, c, δ, α); u .= ut; c .= ct; taken = it
        dec < 1e-14 && break
    end
    evaluate!(pr, u, c)
    return taken, dec
end

"""
Implicit-function adjoint of the τ-grid solve at (u, c) for the slot cotangent xb (design/adjoint/03): returns the
cotangents (X̄ (nt), ū₀ (nt: ln P at k ≤ kc, fixed, and the start state through the weights), z̄rel (kc), B̄) and the solve
details. (u_start, c_start) is the state the solve started from, where the weights were formed.
"""
function tau_stationary_vjp(pr::TP, u, c, xb, u_start, c_start; cg_tol = 1e-10, cg_maxit = 1000, gauss_newton = false)
    par = pr.par; nt, kc, n1, ny, nx = pr.nt, pr.kc, pr.n1, pr.ny, pr.nx
    op = tau_newton_ops(pr, u, c; gauss_newton = gauss_newton)
    # the null directions keep the start's value (the solve never moves them; the uniform one the gauge removes): their
    # share of the cotangent goes to the start's cut heights
    cnull = zeros(ny, nx)
    for (N, n2) in zip(op.nulls, op.nn)
        cnull .+= (gsum(par, sum(N .* @view xb[1, :, :])) / n2) .* N
    end
    nullb = gsum(par, sum(@view xb[1, :, :]))
    λ = zeros(n1, ny, nx); flag = Ref(:none)
    its, rel = pcg!(op.Hop!, op.Mop!, par, pr, λ, op.proj!(copy(xb)); tol = cg_tol, maxit = cg_maxit, flag = flag)
    # stationarity: ½ GᵀΔ/Φ of one undamped Newton step
    G = op.grad(u, c); Φ = evaluate!(pr, u, c)[1]
    δ = zeros(size(G)); pcg!(op.Hop!, op.Mop!, par, pr, δ, op.proj!(-G); tol = 1e-3, maxit = 200)   # its size only
    dec = -gdot(par, G, δ) / (2Φ)
    # mixed term −d/dε ∇ₐΦ along λ (a = ln T, B, ln P at the cut, the heights above it; the weights' layer sums)
    vm = max(gmax(par, maximum(abs, @view λ[2:end, :, :])) / op.su, gmax(par, maximum(abs, @view λ[1, :, :])) / op.sc)
    ε = HVP_STEP[] / max(vm, 1e-300)
    up, cp = along(pr, u, c, λ, ε); um, cm = along(pr, u, c, λ, -ε)
    p = resid_vjp(pr, up, cp); m = resid_vjp(pr, um, cm)
    D(a, b) = (a .- b) ./ (2ε)
    Xb = -D(p.Xbar, m.Xbar)
    u0b = zeros(nt, ny, nx); @views u0b[kc, :, :] .= .-D(p.ubar[kc, :, :], m.ubar[kc, :, :])
    zrelb = -D(p.zrelbar, m.zrelbar)
    Bb = ntuple(i -> -D(p.Bbar[i], m.Bbar[i]), 3)
    wcb = -D(p.Sc, m.Sc); wfb = -D(p.Sf, m.Sf)
    # weights (tp_weights! at the start): wc_m = dzn_m H̄_m², wf_f = H̄_f H̄_f+1 / Δz̄_f, H̄ the layer medians of P/(ρ g),
    # Δz̄ of the face heights, dzn from Δz̄
    evaluate!(pr, u_start, c_start)
    Hq = 1 ./ (pr.g .* pr.q)
    dz = pr.z[kc:nt-1, :, :] .- pr.z[kc+1:nt, :, :]
    Hb = gmedians(par, Hq); dzf = gmedians(par, dz)
    dzn = [m == 1 ? dzf[1] / 2 : m == n1 ? dzf[end] / 2 : (dzf[m-1] + dzf[m]) / 2 for m in 1:n1]
    Hbb = 2 .* dzn .* Hb .* wcb
    dznb = Hb .^ 2 .* wcb
    dzfb = zeros(n1 - 1)
    for f in 1:n1-1
        wf = Hb[f] * Hb[f+1] / dzf[f]
        Hbb[f] += wfb[f] * Hb[f+1] / dzf[f]; Hbb[f+1] += wfb[f] * Hb[f] / dzf[f]
        dzfb[f] -= wfb[f] * wf / dzf[f]
    end
    dzfb[1] += dznb[1] / 2; dzfb[end] += dznb[n1] / 2
    for mm in 2:n1-1
        dzfb[mm-1] += dznb[mm] / 2; dzfb[mm] += dznb[mm] / 2
    end
    Hw = [reshape(gquantile_vjp(par, vec(Hq[mm, :, :]), 0.5, Hbb[mm]), ny, nx) for mm in 1:n1]
    Zw = [reshape(gquantile_vjp(par, vec(dz[f, :, :]), 0.5, dzfb[f]), ny, nx) for f in 1:n1-1]
    Threads.@threads for i in 1:nx
        ab = zeros(n1)
        @inbounds for j in 1:ny
            fill!(ab, 0.0)
            for f in 1:n1-1                             # Δz_f = Δℓ/2 (a_f + a_f+1) at the start
                w = Zw[f][j, i]; w == 0 && continue
                h = pr.dl[node(pr, f)] / 2; ab[f] += w * h; ab[f+1] += w * h
            end
            for mm in 1:n1
                k = node(pr, mm)
                _, _, _, rX, rY, _, _, cX, cY, _ = H.state_d(pr.tab, pr.lT[k, j, i], u_start[k, j, i])
                ccX, ccY = pr.mass ? (rX, rY) : (cX, cY)
                hw = Hw[mm][j, i] * Hq[mm, j, i]         # ∂(1/(g q)) = −(1/(g q)) ∂ln q
                aw = -ab[mm] * pr.a[mm, j, i]            # ∂a = −a ∂ln χ
                Xb[k, j, i] += -hw * rX + aw * ccX
                u0b[k, j, i] += -hw * (rY - 1) + aw * ccY
            end
        end
    end
    evaluate!(pr, u, c)
    return Xb, u0b, zrelb, Bb, (its = its, rel = rel, flag = flag[], newton_decrement = dec, null_component = nullb, cnull = cnull)
end
