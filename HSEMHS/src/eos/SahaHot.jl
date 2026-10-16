# Hot branch (implementation doc §7.4, as built; see doc §20).
#
# Exactly the direct Wittmann model (H, H+, H-, H2, H2+ with the Wittmann partition functions
# and molecular constants; electron donors = elements 2..28 with two ionization stages;
# identical Saha expressions and constants), solved without the C++ solver's 1e±15 clamps,
# with cancellation-free roots and a bracketed search in ln pe. Where both converge the two
# branches agree to the solver tolerance; the direct solver fails in the hot, low-pressure
# corner, where n(H+)/n(H) exceeds its 1e15 clamp.

const NCONTR = 28
const SAHA_C = 9.0804625434325867      # Wittmann Saha constant (log10 form, cgs)
const BK_CGS = 1.3806488e-16
const HH_CGS = 6.62606957e-27
const ME_CGS = 9.10938188e-28
const EV_CGS = 1.602176565e-12
const SAHA_FAC = ((2π * ME_CGS * BK_CGS) / (HH_CGS * HH_CGS))^1.5

struct HotEOS
    ctx::WittCtx
    abund::Vector{Float64}          # normalized fractions (sum 1), as in the C++ code
    amass::Vector{Float64}
    alfai::Vector{Float64}          # ABUND[ii]/ABUND[1], ii = 2..NCONTR
    ab_others::Float64
    rho_from_H::Float64             # cgs, as in the C++ code
    eion::Vector{Vector{Float64}}   # eV, per element, per stage
end

function HotEOS(ctx::WittCtx)
    ab, am = witt_abundances(ctx)
    eion = [[witt_eion_ev(ctx, a, s) for s in 1:witt_nstage(ctx, a)] for a in 1:99]
    return HotEOS(ctx, ab, am, [ab[i] / ab[1] for i in 2:NCONTR], witt_ab_others(ctx),
                  witt_rho_from_H(ctx), eion)
end

# Wittmann's pressure-form Saha ratio n_{r+1}/n_r
@inline saha_w(theta, eion, u1, u2, pe) = u2 * exp(2.302585093 * (SAHA_C - theta * eion)) / (u1 * pe * theta^2.5)

"""Brent's method on [a, b] with f(a) < 0 < f(b). Returns the root."""
function brent(f, a, b; rtol = 1e-15, maxit = 300)
    fa, fb = f(a), f(b)
    (fa < 0 && fb > 0) || error("brent: root not bracketed (f(a) = $fa, f(b) = $fb)")
    c, fc = a, fa
    d = e = b - a
    for _ in 1:maxit
        if (fb > 0) == (fc > 0)
            c, fc = a, fa
            d = e = b - a
        end
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end
        tol = 2eps() * abs(b) + 0.5rtol * abs(b)
        m = 0.5 * (c - b)
        (abs(m) <= tol || fb == 0) && return b
        if abs(e) >= tol && abs(fa) > abs(fb)
            s = fb / fa
            if a == c
                p = 2m * s; q = 1 - s
            else
                q = fa / fc; r = fb / fc
                p = s * (2m * q * (q - r) - (b - a) * (r - 1)); q = (q - 1) * (r - 1) * (s - 1)
            end
            p > 0 ? (q = -q) : (p = -p)
            if 2p < min(3m * q - abs(tol * q), abs(e * q))
                e = d; d = p / q
            else
                d = m; e = m
            end
        else
            d = m; e = m
        end
        a, fa = b, fb
        b += abs(d) > tol ? d : (m > 0 ? tol : -tol)
        fb = f(b)
    end
    return b
end

# H2 / H2+ constants of the Wittmann code (molecb), clamped as in pe_pg
@inline function molecb(theta)
    y1 = -11.206998 + theta * (2.7942767 + theta * (7.9196803e-2 - theta * 2.4790744e-2))
    y2 = -12.533505 + theta * (4.9251644 + theta * (-5.6191273e-2 + theta * 3.2687661e-3))
    return clamp(y1, -30.0, 30.0), clamp(y2, -30.0, 30.0)
end

"""
Hydrogen/donor balance at given pe (cgs), the exact system solved by Wittmann's pe_pg,
written without its 1e±15 clamps and with cancellation-free roots.
Returns (f1, f2, f3, f4, f5, fe, g1).
"""
function hydrogen_fractions(h::HotEOS, theta, uH, us, pe)
    g2 = saha_w(theta, h.eion[1][1], uH[1], uH[2], pe)          # p(H+)/p(H)
    g3 = 1.0 / saha_w(theta, 0.754, 1.0, uH[1], pe)             # p(H-)/p(H)
    g1 = 0.0
    @inbounds for q in 1:NCONTR-1
        u = us[q]; el = h.eion[q+1]
        aa = saha_w(theta, el[1], u[1], u[2], pe)
        bb = saha_w(theta, el[2], u[2], u[3], pe)
        g1 += h.alfai[q] * aa * (1 + 2bb) / (1 + aa * (1 + bb))
    end
    cm1, cm2 = molecb(theta)
    g4 = pe * 10.0^cm1
    g5 = pe * 10.0^cm2
    a = 1 + g2 + g3
    e = g2 * g4 / g5
    b = 2 * (1 + e)
    c = g5
    d = g2 - g3
    c1 = c * b * b + a * d * b - e * a * a
    c2 = 2 * a * e - d * b + a * b * g1
    c3 = -(e + b * g1)
    D = sqrt(max(c2 * c2 - 4 * c1 * c3, 0.0))
    # Wittmann's root: -c2/(2c1) + sign(c1) sqrt((c2/2c1)^2 - c3/c1) = (-c2 + D)/(2 c1)
    # for either sign of c1; for c2 > 0 use the cancellation-free equivalent.
    f1 = c2 > 0 ? -2c3 / (c2 + D) : (-c2 + D) / (2c1)
    # f5 from f5·fe = c f1² with fe = d f1 + e f5 + g1 (positive root, no cancellation)
    s = d * f1 + g1
    f5 = 2 * c * f1^2 / (s + sqrt(s * s + 4 * e * c * f1^2))
    f4 = e * f5
    f3 = g3 * f1
    f2 = g2 * f1
    fe = f2 - f3 + f4 + g1
    return f1, f2, f3, f4, f5, fe, g1
end

function hot_pf(h::HotEOS, T)
    return witt_pf(h.ctx, 1, T), [witt_pf(h.ctx, ii, T) for ii in 2:NCONTR]
end

"""
    hot_solve(h, T, P) -> (rho [kg m-3], ne [m-3], pe_cgs, ok)

Robust solution of the direct solver's equations: root of ln pe' (pe) − ln pe with
pe' = pg / (1 + (Σf + ab_others)/fe), bracketed in [1e-30 pg, pg).
"""
function hot_solve(h::HotEOS, T::Float64, P::Float64; pf = hot_pf(h, T))
    pg = P * 10.0
    theta = 5040.0 / T
    uH, us = pf
    function G(lp)
        pe = exp(lp)
        f1, f2, f3, f4, f5, fe, _ = hydrogen_fractions(h, theta, uH, us, pe)
        pnew = pg / (1 + (f1 + f2 + f3 + f4 + f5 + h.ab_others) / fe)
        return lp - log(pnew)            # increasing in lp
    end
    Gs(lp) = (g = try G(lp) catch; NaN end; isfinite(g) ? g : NaN)
    lo = log(1e-30 * pg)
    hi = log(pg * (1 - 1e-12))
    # near pe = pg the fractions can be undefined (fe <= 0) at extreme densities: step the
    # upper bracket down by factors of 2 until G is finite and positive
    k = 0
    while !(Gs(hi) > 0) && k < 200
        hi -= log(2.0); k += 1
    end
    (Gs(lo) < 0 && Gs(hi) > 0) || return NaN, NaN, NaN, false
    lp = try
        brent(Gs, lo, hi)
    catch
        return NaN, NaN, NaN, false
    end
    pe = exp(lp)
    _, _, _, _, _, fe, _ = hydrogen_fractions(h, theta, uH, us, pe)
    rho_cgs = pe * h.rho_from_H / (fe * T)
    return rho_cgs * 1e3, pe / (BK_CGS * T) * 1e6, pe, isfinite(rho_cgs) && rho_cgs > 0
end

"""Background partials (same layout and conventions as the C++ background_partials)."""
function hot_partials!(n::Vector{Float64}, h::HotEOS, T::Float64, P::Float64, pe::Float64; pf = hot_pf(h, T))
    pg = P * 10.0
    theta = 5040.0 / T
    tk = BK_CGS * T
    xna = (pg - pe) / tk
    xne = pe / tk
    uH, us = pf
    function ion!(x, atom, requested)
        u = atom == 1 ? uH : us[atom-1]               # elements 2..28 are cached in pf
        cnt = requested > 0 ? min(length(u), requested) : length(u)
        ntot = xna * h.abund[atom]
        x[1] = 1.0
        for s in 2:cnt
            x[s] = 2.0 * SAHA_FAC * (u[s] / u[s-1]) * T^1.5 * exp(-h.eion[atom][s-1] * EV_CGS / tk) / xne
        end
        for s in cnt:-1:2
            x[1] = 1.0 + x[1] * x[s]
        end
        x[1] = 1.0 / x[1]
        for s in 2:cnt
            x[s] *= x[s-1]
        end
        for s in 1:cnt
            x[s] *= ntot / u[s]
        end
        return x
    end
    x = zeros(8)
    ion!(x, 2, 3); n[4] = x[1]; n[5] = x[2]; n[6] = x[3]
    ion!(x, 6, 0); n[7] = x[1]
    ion!(x, 13, 0); n[8] = x[1]
    ion!(x, 14, 0); n[9] = x[1]; n[10] = x[2]
    ion!(x, 20, 0); n[11] = x[1]; n[12] = x[2]
    ion!(x, 12, 0); n[13] = x[1]; n[14] = x[2]
    ion!(x, 26, 0); n[15] = x[1]
    ion!(x, 7, 0); n[16] = x[1]
    ion!(x, 8, 0); n[17] = x[1]
    f1, f2, f3, _, _, fe, _ = hydrogen_fractions(h, theta, uH, us, pe)
    phtot = pe / fe
    n[1] = f1 * phtot / tk * 0.5
    n[2] = f2 * phtot / tk
    n[3] = f3 * phtot / tk
    return xna, xne
end
