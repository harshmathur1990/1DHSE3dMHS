# EOS model = direct Wittmann solver + robust hot branch, blended at T_sw
# (implementation doc §7.3, §7.5; as built, see doc §20).
#
# Branch rule for (T, P):
#   T >= 1.1 T_sw            : hot branch only                     (EOS_HOT_BRANCH)
#   T_sw <= T < 1.1 T_sw     : smoothstep blend of direct and hot   (EOS_BLEND);
#                              hot only where the direct solver fails (EOS_HOT_BRANCH)
#   T < T_sw                 : direct; where it fails, the hot branch (same model, robust
#                              solve) is used and flagged EOS_ROBUST_FALLBACK
# All three return ln ρ [kg m-3], ln nₑ [m-3], ln χ₅₀₀ [m-1].

const EOS_ROBUST_FALLBACK = 0x20

const LN_FIELDS = ("lnrho", "lnne", "lnchi")

struct EOSModel
    ctxs::Vector{WittCtx}          # one per thread (contexts are immutable, also thread-safe)
    hots::Vector{HotEOS}
    T_sw::Float64                  # Inf: direct everywhere
    hot_branch::String             # "none" | "robust_witt"
    mass_per_H::Float64
    pf_path::String
end

function EOSModel(pf_path::AbstractString; prec = 1e-10, maxit = 5000, log_abund = nothing,
                  T_sw = Inf, hot_branch = "none")
    n = Threads.maxthreadid()
    ctxs = [WittCtx(pf_path; prec = prec, maxit = maxit, log_abund = log_abund) for _ in 1:n]
    hots = [HotEOS(c) for c in ctxs]
    return EOSModel(ctxs, hots, T_sw, hot_branch, witt_mass_per_H(ctxs[1]), String(pf_path))
end

with_switch(m::EOSModel, T_sw, hot_branch) = EOSModel(m.ctxs, m.hots, T_sw, hot_branch, m.mass_per_H, m.pf_path)

struct BranchOut
    lnrho::Float64
    lnne::Float64
    lnchi::Float64
    ok::Bool
end

function direct_branch(m::EOSModel, tid, T, P, n17)
    c = m.ctxs[tid]
    s, code = witt_state(c, T, P)
    code == 0 || return BranchOut(NaN, NaN, NaN, false)
    xna, xne, pc = witt_partials!(n17, c, T, P, s.pe * 10.0)
    pc == 0 || return BranchOut(NaN, NaN, NaN, false)
    chi = chi500_m(T, n17, xna, xne)
    return BranchOut(log(s.rho), log(s.ne), log(chi), chi > 0 && isfinite(chi))
end

function hot_branch(m::EOSModel, tid, T, P, n17; pf = nothing)
    h = m.hots[tid]
    pfv = pf === nothing ? hot_pf(h, T) : pf
    rho, ne, pe, ok = hot_solve(h, T, P; pf = pfv)
    ok || return BranchOut(NaN, NaN, NaN, false)
    xna, xne = hot_partials!(n17, h, T, P, pe; pf = pfv)
    chi = chi500_m(T, n17, xna, xne)
    return BranchOut(log(rho), log(ne), log(chi), chi > 0 && isfinite(chi))
end

"""
    model_eval(m, X, Y; n17, pf=nothing) -> (lnrho, lnne, lnchi, flags, ok)

The function the table interpolates, at X = ln T, Y = ln P.
"""
function model_eval(m::EOSModel, X::Float64, Y::Float64; n17 = zeros(17), pf = nothing)
    tid = Threads.threadid()
    T = exp(X); P = exp(Y)
    flags = T >= 3.0e4 ? OPACITY_INCOMPLETE : EOS_OK
    if T >= 1.1 * m.T_sw
        b = hot_branch(m, tid, T, P, n17; pf = pf)
        return b.lnrho, b.lnne, b.lnchi, flags | EOS_HOT_BRANCH, b.ok
    end
    d = direct_branch(m, tid, T, P, n17)
    if !d.ok
        m.hot_branch == "none" && return NaN, NaN, NaN, flags | EOS_DIRECT_FAILED, false
        b = hot_branch(m, tid, T, P, n17; pf = pf)
        fl = T >= m.T_sw ? EOS_HOT_BRANCH : EOS_ROBUST_FALLBACK
        return b.lnrho, b.lnne, b.lnchi, flags | fl, b.ok
    end
    if T >= m.T_sw
        b = hot_branch(m, tid, T, P, n17; pf = pf)
        b.ok || return NaN, NaN, NaN, flags | EOS_DIRECT_FAILED, false
        t = (X - log(m.T_sw)) / log(1.1)
        w = t * t * (3 - 2t)
        return (1 - w) * d.lnrho + w * b.lnrho, (1 - w) * d.lnne + w * b.lnne,
               (1 - w) * d.lnchi + w * b.lnchi, flags | EOS_BLEND, true
    end
    return d.lnrho, d.lnne, d.lnchi, flags, true
end

# ---------------------------------------------------------------------------------------
# Envelope (§7.3)
# ---------------------------------------------------------------------------------------

struct Envelope
    T_lo::Float64
    T_hi::Float64
    log10P_lo::Float64
    log10P_hi::Float64
    source::String
end

"""Envelope from the Gate 0 report (`/meta/envelope_inputs`), or explicit overrides."""
function envelope_from_gate0(path::AbstractString; logP_range = Float64[])
    h5open(path, "r") do f
        g = f["meta/envelope_inputs"]
        a(k) = read_attribute(g, k)
        T_lo = 0.9 * a("T_min"); T_hi = 1.1 * a("T_max")
        if isempty(logP_range)
            lo = min(a("log10P_toprun_min"), a("log10P_botrun_min"), a("log10P_native_min")) - 1.0
            hi = max(a("log10P_toprun_max"), a("log10P_botrun_max"), a("log10P_native_max")) + 1.0
            src = "Gate 0 proxy anchor runs ±1 dex ($(path))"
        else
            lo, hi = logP_range
            src = "eos.logP_range override"
        end
        return Envelope(T_lo, T_hi, lo, hi, src)
    end
end

# ---------------------------------------------------------------------------------------
# Coverage scan and T_sw selection (§7.5)
# ---------------------------------------------------------------------------------------

"""
    coverage_scan(m, env; nT=400, nP=200) -> NamedTuple

Direct solver over a (log T, log P) grid: convergence, closure residual, and the
fully-ionized-limit test of §7.5 step 3.
"""
function coverage_scan(m::EOSModel, env::Envelope; nT = 400, nP = 200)
    lT = collect(range(log10(env.T_lo), log10(env.T_hi); length = nT))
    lP = collect(range(env.log10P_lo, env.log10P_hi; length = nP))
    conv = zeros(Bool, nT, nP); clos = fill(NaN, nT, nP); lne = fill(NaN, nT, nP)
    mu = fill(NaN, nT, nP); lrho = fill(NaN, nT, nP)
    Threads.@threads :static for a in 1:nT
        c = m.ctxs[Threads.threadid()]
        for b in 1:nP
            s, code = witt_state(c, 10.0^lT[a], 10.0^lP[b])
            conv[a, b] = code == 0
            clos[a, b] = s.closure_residual
            if code == 0
                lne[a, b] = log(s.ne); mu[a, b] = s.mu_bar; lrho[a, b] = log(s.rho)
            end
        end
    end
    ab, am = witt_abundances(m.ctxs[1])
    sAZ = sum(ab .* (1:99))
    mufull = sum(ab .* am) / (1 + sAZ)
    full_ok = true
    for a in 1:nT, b in 1:nP
        T = 10.0^lT[a]
        (T >= 2.0e5 && conv[a, b]) || continue
        nef = 10.0^lP[b] / (K_B_WITT * T) / (1 + sAZ) * sAZ
        (abs(lne[a, b] - log(nef)) <= 0.02 && abs(mu[a, b] / mufull - 1) <= 0.02) || (full_ok = false)
    end
    mono = true
    for b in 1:nP, a in 2:nT
        10.0^lT[a] > 3.0e4 || continue
        (conv[a, b] && conv[a-1, b]) || continue
        lne[a, b] < lne[a-1, b] - 1e-9 && (mono = false)
    end
    return (lT = lT, lP = lP, conv = conv, closure = clos, lne = lne, lrho = lrho, mu = mu,
            all_converged = all(conv), full_ionization_ok = full_ok, monotone = mono)
end

"""
    select_T_sw(m, scan) -> (T_sw, hot_branch, report)

§7.5: direct everywhere if it converges everywhere and reaches the fully ionized limit;
otherwise the smallest scan temperature T_sw in [1.5e4, 2e5] K such that for all P samples
and T' in [T_sw, 1.1 T_sw] the branches agree to 1e-3 in ln ρ and ln nₑ wherever the direct
solver converges (where it does not, the hot branch is used; see doc §20).
"""
function select_T_sw(m::EOSModel, scan)
    if scan.all_converged && scan.full_ionization_ok && scan.monotone
        return Inf, "none", "direct solver converged everywhere and reaches the fully ionized limit"
    end
    n17 = zeros(17)
    for a in eachindex(scan.lT)
        Tc = 10.0^scan.lT[a]
        (1.5e4 <= Tc <= 2.0e5) || continue
        ok = true; worst = 0.0
        for Tp in range(Tc, 1.1Tc; length = 5), lp in scan.lP
            P = 10.0^lp
            d = direct_branch(m, 1, Tp, P, n17)
            d.ok || continue
            h = hot_branch(m, 1, Tp, P, n17)
            h.ok || (ok = false; break)
            dev = max(abs(d.lnrho - h.lnrho), abs(d.lnne - h.lnne))
            worst = max(worst, dev)
            dev <= 1e-3 || (ok = false; break)
        end
        ok && return Tc, "robust_witt", @sprintf("T_sw = %.4g K; max branch difference in [T_sw, 1.1 T_sw] = %.2e", Tc, worst)
    end
    return NaN, "rejected", "no T_sw in [1.5e4, 2e5] K satisfies the 1e-3 agreement"
end
