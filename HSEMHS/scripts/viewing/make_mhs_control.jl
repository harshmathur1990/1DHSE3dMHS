#=
Known-equilibrium control for mhs-poisson-ltau (design/CODEX_REVIEW_solver_time_round4_2026-10-29.md, priorities 1–3): an
exact 2-D magnetostatic atmosphere (x periodic, invariant in y) mapped onto inclined rays the way make_slanted_tau.jl
maps Bifrost, but from analytic fields on a fine height grid. The equations are exact; the reference, f and the ray
optical depths are numerical (RK4, linear interpolation, the τ quadrature of Tau.tau_column!), with their own
construction error, which `dz_m` refines.

Reference: T₀(τ) grey Eddington–Hopf (T_eff 5772 K) in a vertical column HSE in τ gives T₀(z) (linear in z between the
RK4 nodes); P₀(z) is integrated by RK4 in z, dP₀/dz = −ρ(T₀, P₀) g, ρ₀ = ρ(T₀, P₀), z = 0 at τ₅₀₀ = 1.

Field (Grad–Shafranov, B = ∇ × (A ŷ) + B_y ŷ, B_y constant): A = A₀ cos(kx) f(z), p(A, z) = P₀(z) − m(z) A²/2. Force balance
−∇p − ρ g ẑ + J × B = 0 holds exactly when ∇²A = μ₀ m A, i.e. f'' = (k² + μ₀ m) f, and ρ = −∂_z p|_A / g. m(z) is chosen so
that the pressure deficit is a fraction D(z) of P₀: m A₀² f²/2 = D P₀, so

    P = P₀ (1 − D cos² kx),    ρ = ρ₀ (1 − D cos² kx) + (P₀ D'/g) cos² kx − (2 D P₀ f'/(f g)) cos² kx,
    f'' = k² f + k² D P₀/(P_m f),   A₀ = √(2 μ₀ P_m)/k,   P_m = P₀(0),
    B_x = −A₀ cos(kx) f',  B_z = −A₀ k sin(kx) f,

D(z) = D/(1 + exp((z − D_top)/D_width)) (constant without D_top_km). f is integrated down from the top, starting on the
isothermal self-similar solution f ∝ √P₀ (f'/f = −1/2H, where ρ = ρ₀ and P_m f² = D P₀ (2kH)²/(1 − (2kH)²)).
T = EOS⁻¹(ρ, P) (`invT`: bracketed from T₀ inside the valid EOS domain, failures classified). field = 0 is the
field-free hydrostatic control (horizontally uniform).

Audit before mapping, over one period (`audit_nx` points) and every interior fine level: EOS inversion failures by kind,
ln P margins to the table edges, the T range, and the Cartesian force balance and ∇·B by centred differences (independent
of the derivation). The run stops if any point fails the inversion; the audit is stored as truth attributes.

Rays: x = ξ + t_x z through footprints (ξ, η) (the atmosphere is y-invariant, so t_y only moves the ray along y), sampled on
the fine z grid; χ₅₀₀ = χ(T, P), τ_LOS by Tau.tau_column! (dz/μ), every quantity interpolated linearly in log₁₀ τ_LOS onto
the axis lt_min:dlt:lt_max (as make_slanted_tau.jl; log τ = 0 is on the axis when lt_min is a multiple of dlt).

usage: julia --project=. -t N scripts/viewing/make_mhs_control.jl MU AZIMUTH_DEG OUT_INPUT.nc OUT_TRUTH.h5 [KEY=VALUE ...]
KEY: nx (256), ny (8), box_mm (12), period_mm (3), dlt (0.1), lt_min (−8), lt_max (1.4), D (0.3), D_top_km (none),
D_width_km (100), field (1), by_gauss (0), dz_m (250), audit_nx (512), waves_x, waves_y (0: the y-invariant arcade along x with period_mm; integers
n_x, n_y: the arcade rotated to k = 2π (n_x/L_x, n_y/L_y), L_x = box_mm, L_y = ny · dx, B_s split into B_x, B_y; ray profiles on
ns0 (2048) footprints per period, interpolated linearly to each footprint). Outputs as make_slanted_tau.jl, with the keys
(control_*) and the audit (audit_*) as attributes.
=#
using HSEMHS
using HDF5
using MPI
using Printf

const H = HSEMHS
const MU0 = 4π * 1e-7

"""As make_slanted_tau.jl: linear in log10 τ (NaN below the bottom; above the top hold or extrapolate)."""
function interp_at(v, lτ, L; top = :hold)
    L > lτ[end] && return NaN
    if L < lτ[1]
        top == :hold && return v[1]
        return v[1] + (L - lτ[1]) * (v[2] - v[1]) / (lτ[2] - lτ[1])
    end
    k = searchsortedlast(lτ, L)
    k == length(lτ) && return v[end]
    w = (L - lτ[k]) / (lτ[k+1] - lτ[k])
    return v[k] + w * (v[k+1] - v[k])
end

"""Linear interpolation of v(x) on increasing x."""
function lin(xs, v, x)
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    w = (x - xs[k]) / (xs[k+1] - xs[k])
    return v[k] + w * (v[k+1] - v[k])
end

lnrho(tab, lT, lP) = H.state(tab, lT, lP)[1]

const INV_CODES = ("ok", "ln P outside the table", "target ρ non-positive or non-finite",
                   "needs T below the valid EOS domain at this P", "needs T above the valid EOS domain at this P")

"""
(ln T, code) with ln ρ(ln T, ln P) = lr (ρ decreases with T at fixed P), bracketed by stepping 0.05 in ln T from
`guess` while the EOS is valid (the table has invalid corners: its T range alone does not bracket), then bisected.
code (INV_CODES): 1 ok, 2 ln P outside the table, 3 target non-positive/non-finite, 4/5 the solution lies below/above
the valid domain at this P.
"""
function invT(tab, lr, lP, guess)
    (tab.Y[1] <= lP <= tab.Y[end]) || return NaN, 2
    isfinite(lr) || return NaN, 3
    F(x) = lnrho(tab, x, lP) - lr
    a = clamp(guess, tab.X[1], tab.X[end]); fa = F(a)
    isfinite(fa) || return NaN, 4
    b = a; fb = fa
    if fa > 0                                     # ρ too high: T must rise
        while fb > 0
            b = a + 0.05; b > tab.X[end] && return NaN, 5
            fb = F(b); isfinite(fb) || return NaN, 5
            fb > 0 && (a = b; fa = fb)
        end
    else                                          # T must fall
        while fa <= 0
            fa == 0 && return a, 1
            b = a; fb = fa; a = b - 0.05; a < tab.X[1] && return NaN, 4
            fa = F(a); isfinite(fa) || return NaN, 4
        end
    end
    for _ in 1:60
        c = (a + b) / 2; fc = F(c)
        isfinite(fc) || return NaN, 4
        fc > 0 ? (a = c) : (b = c)
        b - a < 1e-13 && break
    end
    return (a + b) / 2, 1
end

T_edd(τ; Teff = 5772.0) = Teff * (0.75 * (τ + 0.7104 - 0.1331 * exp(-3.4488τ)))^0.25

function main(mu_s, az_s, out_in, out_truth, kv...)
    H.ensure_mpi()
    o = Dict("nx" => 256.0, "ny" => 8.0, "box_mm" => 12.0, "period_mm" => 3.0, "dlt" => 0.1, "lt_min" => -8.0, "lt_max" => 1.4,
             "D" => 0.3, "field" => 1.0, "by_gauss" => 0.0, "dz_m" => 250.0, "D_top_km" => Inf, "D_width_km" => 100.0,
             "audit_nx" => 512.0, "waves_x" => 0.0, "waves_y" => 0.0, "ns0" => 2048.0)
    for a in kv
        k, v = split(a, "="); haskey(o, k) || error("unknown key $k"); o[k] = parse(Float64, v)
    end
    μ = parse(Float64, mu_s); φ = parse(Float64, az_s)
    v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
    isempty(H.option_errors(v)) || error(join(H.option_errors(v), "; "))
    t = H.slant(v); g = 274.0
    tab = H.load_table("cache/eos_table_fullz.h5")
    NX = Int(o["nx"]); NY = Int(o["ny"]); Lb = o["box_mm"] * 1e6; dx = Lb / NX
    rot = o["waves_x"] != 0 || o["waves_y"] != 0       # rotated arcade: varies along ê_s = k/|k|, k = 2π (n_x/L_x, n_y/L_y)
    if rot
        kv = 2π .* (o["waves_x"] / Lb, o["waves_y"] / (NY * dx)); k = hypot(kv...); es = kv ./ k
        all(isinteger, (o["waves_x"], o["waves_y"])) || error("waves_x, waves_y must be integers")
    else
        k = 2π / (o["period_mm"] * 1e6); es = (1.0, 0.0)
        isapprox(Lb * k / 2π, round(Lb * k / 2π); atol = 1e-9) || error("box must hold whole periods")
    end
    per = 2π / k
    nlev = round(Int, (o["lt_max"] - o["lt_min"]) / o["dlt"]) + 1
    lt = collect(range(o["lt_min"], o["lt_max"]; length = nlev)); nt = length(lt)
    D0 = o["field"] != 0 ? o["D"] : 0.0
    zD = o["D_top_km"] * 1e3; wD = o["D_width_km"] * 1e3
    Dz(z) = isfinite(zD) ? D0 / (1 + exp((z - zD) / wD)) : D0           # deficit fraction, tapered above D_top_km
    Dpz(z) = isfinite(zD) ? -Dz(z) * (1 - Dz(z) / D0) / wD : 0.0        # dD/dz

    # 1. vertical reference column HSE in τ (RK4 in ln τ): T₀(z)
    lτv = collect(range(-11.5 * log(10), 3.0 * log(10); step = 0.002 * log(10))); nv = length(lτv)
    function rhs(lτ, lP)
        τ = exp(lτ); lT = log(T_edd(τ)); r, _, c, _ = H.state(tab, lT, lP)
        return g * τ * exp(r - lP - c), -τ * exp(-c)               # d ln P/d ln τ, dz/d ln τ
    end
    lPtop = log(1e-3)
    for _ in 1:200                                                   # top closure P = g τ ρ/χ
        τ = exp(lτv[1]); r, _, c, _ = H.state(tab, log(T_edd(τ)), lPtop); lPtop = log(g * τ) + r - c
    end
    Pv = zeros(nv); zv = zeros(nv); Pv[1] = lPtop
    for n in 1:nv-1
        h = lτv[n+1] - lτv[n]; l = lτv[n]; p = Pv[n]
        a1, b1 = rhs(l, p); a2, b2 = rhs(l + h / 2, p + h / 2 * a1); a3, b3 = rhs(l + h / 2, p + h / 2 * a2); a4, b4 = rhs(l + h, p + h * a3)
        Pv[n+1] = p + h / 6 * (a1 + 2a2 + 2a3 + a4); zv[n+1] = zv[n] + h / 6 * (b1 + 2b2 + 2b3 + b4)
    end
    zv .-= lin(lτv, zv, 0.0)
    Tv = T_edd.(exp.(lτv))
    rz = reverse(zv); rT = reverse(Tv)                               # increasing z
    T0f(z) = lin(rz, rT, z)

    # 2. fine height grid, P₀ by RK4 in z from the top
    dz = o["dz_m"]; ztop = zv[1]; zbot = zv[end]
    zf = collect(ztop:-dz:zbot); nz = length(zf)                     # top first
    P0rhs(z, lP) = -g * exp(lnrho(tab, log(T0f(z)), lP) - lP)       # d ln P₀/dz
    lP0 = zeros(nz); lP0[1] = Pv[1]
    for n in 1:nz-1
        h = -dz; z = zf[n]; p = lP0[n]
        a1 = P0rhs(z, p); a2 = P0rhs(z + h / 2, p + h / 2 * a1); a3 = P0rhs(z + h / 2, p + h / 2 * a2); a4 = P0rhs(z + h, p + h * a3)
        lP0[n+1] = p + h / 6 * (a1 + 2a2 + 2a3 + a4)
    end
    P0 = exp.(lP0); lT0 = log.(T0f.(zf)); ρ0 = [exp(lnrho(tab, lT0[n], lP0[n])) for n in 1:nz]
    any(!isfinite, ρ0) && error("reference outside the EOS table")

    # 3. field: f'' = k² f + k² D P₀/(P_m f) integrated downward (h = −dz), f' = −k f at the top
    Pm = exp(lin(reverse(zf), reverse(lP0), 0.0)); A0 = sqrt(2MU0 * Pm) / k
    f = zeros(nz); fp = zeros(nz)
    Dn = Dz.(zf); Dpn = Dpz.(zf)
    if D0 > 0
        H1 = P0[1] / (ρ0[1] * g); q = 2k * H1                       # top: the isothermal self-similar f ∝ √P₀ (ρ = ρ₀)
        q < 0.95 || error("2 k H at the top is $q: use a longer period")
        Dn[1] > 1e-3D0 || error("D at the top is $(Dn[1]): the self-similar start needs a non-zero deficit there")
        f[1] = sqrt(Dn[1] * q^2 / (1 - q^2) * P0[1] / Pm); fp[1] = -f[1] / (2H1)
        P0z = z -> exp(lin(reverse(zf), reverse(lP0), z))
        acc = (z, y) -> k^2 * y + k^2 * Dz(z) * P0z(z) / (Pm * y)
        for n in 1:nz-1
            h = -dz; z = zf[n]; y = f[n]; yp = fp[n]
            k1y = yp; k1p = acc(z, y)
            k2y = yp + h / 2 * k1p; k2p = acc(z + h / 2, y + h / 2 * k1y)
            k3y = yp + h / 2 * k2p; k3p = acc(z + h / 2, y + h / 2 * k2y)
            k4y = yp + h * k3p; k4p = acc(z + h, y + h * k3y)
            f[n+1] = y + h / 6 * (k1y + 2k2y + 2k3y + k4y); fp[n+1] = yp + h / 6 * (k1p + 2k2p + 2k3p + k4p)
        end
    end
    by = o["by_gauss"] * 1e-4
    c2(x) = cos(k * x)^2
    function fields(n, x)                                            # (ln P, ln ρ, Bx, Bz) at fine level n (all locals)
        local cc = c2(x)
        local ρx = ρ0[n] * (1 - Dn[n] * cc) + (D0 > 0 ? (P0[n] * Dpn[n] - 2Dn[n] * P0[n] * fp[n] / f[n]) / g * cc : 0.0)
        return lP0[n] + log1p(-Dn[n] * cc), log(ρx), -A0 * cos(k * x) * fp[n], -A0 * k * sin(k * x) * f[n]
    end
    i0 = argmin(abs.(zf))
    @printf("μ %.3f φ %g t (%.3f, %.3f); box %.1f Mm, period %.2f Mm, %d × %d, dx %.1f km; τ axis %g:%g:%g (%d levels); fine dz %.0f m, %d levels %.0f … %.0f km\n",
            μ, φ, t..., Lb / 1e6, per / 1e6, NX, NY, dx / 1e3, lt[1], o["dlt"], lt[end], nt, dz, nz, zf[1] / 1e3, zf[end] / 1e3)
    @printf("D %.2f; at z = 0: P₀ %.4g Pa, T₀ %.0f K; max |B| at z = 0: %.0f G, at the top %.0f G; f(0) %.3g\n",
            Dn[i0], P0[i0], exp(lT0[i0]), D0 > 0 ? A0 * max(abs(fp[i0]), k * f[i0]) * 1e4 : 0.0,
            D0 > 0 ? A0 * max(abs(fp[1]), k * f[1]) * 1e4 : 0.0, f[i0])

    # 4. construction audit over one period and every interior fine level: EOS inversion (by failure kind), positivity,
    # table margins, and the Cartesian force balance and ∇·B by centred differences (independent of the derivation)
    nxf = Int(o["audit_nx"]); hx = per / nxf; xs = (0:nxf-1) .* hx
    codes = zeros(Int, length(INV_CODES), nz); rxm = zeros(nz); rzm = zeros(nz); dvm = zeros(nz); dvs = zeros(nz)
    pmin = fill(Inf, nz); pmax = fill(-Inf, nz); tmin = fill(Inf, nz); tmax = fill(-Inf, nz); firstbad = fill((NaN, NaN, NaN, NaN), nz)
    Threads.@threads for n in 2:nz-1
        for x in xs
            lPc, lrc, bxc, bzc = fields(n, x)
            lTc, code = isfinite(lrc) ? invT(tab, lrc, lPc, lT0[n]) : (NaN, 3)
            codes[code, n] += 1
            code > 1 && isnan(firstbad[n][1]) && (firstbad[n] = (x, lPc, lrc, lT0[n]))
            pmin[n] = min(pmin[n], lPc); pmax[n] = max(pmax[n], lPc)
            isfinite(lTc) && (tmin[n] = min(tmin[n], lTc); tmax[n] = max(tmax[n], lTc))
            D0 > 0 || continue
            fu = fields(n - 1, x); fd = fields(n + 1, x); fe = fields(n, x + hx); fw = fields(n, x - hx)
            dPdx = (exp(fe[1]) - exp(fw[1])) / 2hx; dPdz = (exp(fu[1]) - exp(fd[1])) / 2dz
            dBxdx = (fe[3] - fw[3]) / 2hx; dBzdz = (fu[4] - fd[4]) / 2dz
            dBxdz = (fu[3] - fd[3]) / 2dz; dBzdx = (fe[4] - fw[4]) / 2hx
            Jy = (dBxdz - dBzdx) / MU0; ρg = exp(lrc) * g
            rxm[n] = max(rxm[n], abs(-dPdx + Jy * bzc) / ρg)
            rzm[n] = max(rzm[n], abs(-dPdz - ρg - Jy * bxc) / ρg)
            dvm[n] = max(dvm[n], abs(dBxdx + dBzdz)); dvs[n] = max(dvs[n], abs(dBxdx), abs(dBzdz))
        end
    end
    nbadc = vec(sum(codes; dims = 2))
    @printf("audit (%d × %d points, dx %.2f km, dz %.0f m): EOS inversion %s\n", nxf, nz - 2, hx / 1e3, dz,
            join(["$(INV_CODES[c]) $(nbadc[c])" for c in eachindex(INV_CODES)], ", "))
    for n in 2:nz-1
        isnan(firstbad[n][1]) && continue
        x, a, b, c = firstbad[n]
        @printf("  first failure at z %.1f km, x %.1f km: ln P %.3f, target ln ρ %.3f, ln ρ at T₀ %.3f, T₀ %.0f K\n",
                zf[n] / 1e3, x / 1e3, a, b, lnrho(tab, c, a), exp(c))
        break
    end
    mP = (minimum(pmin[2:nz-1]) - tab.Y[1], tab.Y[end] - maximum(pmax[2:nz-1]))
    rT = (exp(minimum(tmin[2:nz-1])), exp(maximum(tmax[2:nz-1])))
    @printf("  ln P margins to the table edges: %.2f (below) / %.2f (above); T %.0f … %.0f K (table %.0f … %.0f K)\n",
            mP..., rT..., exp(tab.X[1]), exp(tab.X[end]))
    divb = maximum(dvm[2:nz-1] ./ max.(dvs[2:nz-1], 1e-300))           # per level, relative to that level's largest term
    nzm = argmax(rzm); nxm = argmax(rxm)
    D0 > 0 && @printf("  force balance max |R_x|/ρg %.2e (z %.0f km), |R_z|/ρg %.2e (z %.0f km; rms over levels %.2e); max_level |∇·B| / max_level |∂B| %.2e\n",
                      rxm[nxm], zf[nxm] / 1e3, rzm[nzm], zf[nzm] / 1e3, sqrt(sum(abs2, rzm) / (nz - 2)), divb)
    nbadc[1] == sum(nbadc) || error("the construction leaves the EOS domain (see the audit)")
    audit = Dict("audit_rx_max" => maximum(rxm), "audit_rz_max" => maximum(rzm), "audit_divb_max" => D0 > 0 ? divb : 0.0,
                 "audit_lnP_margin_lo" => mP[1], "audit_lnP_margin_hi" => mP[2], "audit_T_min" => rT[1], "audit_T_max" => rT[2],
                 "audit_nx" => nxf, "audit_dz_m" => dz)

    # 5. rays: the atmosphere depends on (s, z), s = ê_s·(x, y); the ray through footprint (ξ, η) has s = s₀ + σ z with
    # s₀ = ê_s·(ξ, η), σ = ê_s·t. Profiles on the τ axis per s₀: y-invariant, one per footprint ξ (exact); rotated, on
    # `ns0` points over a period, interpolated linearly in s₀ to each footprint
    xi = collect((0:NX-1) .* dx); y = collect((0:NY-1) .* dx)
    σ = es[1] * t[1] + es[2] * t[2]
    S0 = rot ? collect((0:Int(o["ns0"])-1) .* (per / o["ns0"])) : xi; nS = length(S0)
    pT = fill(NaN, nt, nS); pBs = similar(pT); pBz = similar(pT); pz = similar(pT); pP = similar(pT); pr_ = similar(pT)
    pz1 = fill(NaN, nS)
    nmiss = Threads.Atomic{Int}(0); ntop = Threads.Atomic{Int}(0); nbad = Threads.Atomic{Int}(0)
    lTmin = Threads.Atomic{Float64}(Inf); lTmax = Threads.Atomic{Float64}(-Inf)
    Threads.@threads for q in 1:nS
        lT = zeros(nz); lPr = zeros(nz); lr = zeros(nz); Bs = zeros(nz); Bz = zeros(nz); χ = zeros(nz); tc = zeros(nz)
        for n in 1:nz
            x = S0[q] + σ * zf[n]
            lPr[n], lr[n], Bs[n], Bz[n] = fields(n, x)
            lT[n] = D0 > 0 ? invT(tab, lr[n], lPr[n], lT0[n])[1] : lT0[n]
            c = isfinite(lT[n]) ? H.state(tab, lT[n], lPr[n])[3] : NaN
            isfinite(c) || (Threads.atomic_add!(nbad, 1); c = -700.0;
                            nbad[] < 5 && @printf("outside: z %.0f km s %.0f km ln P %.3f ln ρ %.3f ln T %.3f (T₀ %.0f K)\n",
                                                  zf[n] / 1e3, x / 1e3, lPr[n], lr[n], lT[n], exp(lT0[n])))
            χ[n] = exp(c)
            Threads.atomic_min!(lTmin, lT[n]); Threads.atomic_max!(lTmax, lT[n])
        end
        H.tau_column!(tc, χ, zf; mu = μ)
        lτ = log10.(max.(tc, 1e-300))
        lt[end] > lτ[end] && Threads.atomic_add!(nmiss, 1)
        lt[1] < lτ[1] && Threads.atomic_add!(ntop, 1)
        pz1[q] = H.z_tau1(tc, zf)
        for kk in 1:nt
            pT[kk, q] = interp_at(lT, lτ, lt[kk]); pBs[kk, q] = interp_at(Bs, lτ, lt[kk]); pBz[kk, q] = interp_at(Bz, lτ, lt[kk])
            pz[kk, q] = interp_at(zf, lτ, lt[kk]; top = :extrapolate)
            pP[kk, q] = interp_at(lPr, lτ, lt[kk]; top = :extrapolate); pr_[kk, q] = interp_at(lr, lτ, lt[kk]; top = :extrapolate)
        end
    end
    T_in = fill(NaN, nt, NY, NX); bx = similar(T_in); byv = similar(T_in); bz = similar(T_in)
    zt = similar(T_in); uP = similar(T_in); ur = similar(T_in); z1 = fill(NaN, NY, NX)
    Threads.@threads for i in 1:NX
        for j in 1:NY
            if rot                                     # periodic linear interpolation in s₀
                sq = mod(es[1] * xi[i] + es[2] * y[j], per) / (per / nS)
                q1 = floor(Int, sq) + 1; w = sq - (q1 - 1); q2 = q1 == nS ? 1 : q1 + 1
            else
                q1 = q2 = i; w = 0.0
            end
            L(A, kk) = (1 - w) * A[kk, q1] + w * A[kk, q2]
            for kk in 1:nt
                T_in[kk, j, i] = exp(L(pT, kk)); b_s = L(pBs, kk)
                bx[kk, j, i] = b_s * es[1]; byv[kk, j, i] = b_s * es[2] + by; bz[kk, j, i] = L(pBz, kk)
                zt[kk, j, i] = L(pz, kk); uP[kk, j, i] = L(pP, kk); ur[kk, j, i] = L(pr_, kk)
            end
            z1[j, i] = (1 - w) * pz1[q1] + w * pz1[q2]
        end
    end
    nbad[] == 0 || error("$(nbad[]) fine points outside the EOS table")
    @printf("T range on the rays %.0f … %.0f K; rays with the top above the axis minimum %d; not reaching the bottom %d\n",
            exp(lTmin[]), exp(lTmax[]), ntop[], nmiss[])
    nmiss[] == 0 || error("rays do not reach the bottom of the τ axis")
    @printf("truth z(τ_LOS = 1): mean %.1f km, min %.1f, max %.1f (range %.1f km)\n",
            sum(z1) / length(z1) / 1e3, minimum(z1) / 1e3, maximum(z1) / 1e3, (maximum(z1) - minimum(z1)) / 1e3)
    meta(a) = (a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2];
               a["source"] = "manufactured Grad-Shafranov control (make_mhs_control.jl)"; a["gravity_m_s2"] = g;
               for (kk, vv) in o; a["control_" * kk] = vv; end; for (kk, vv) in audit; a[kk] = vv; end)
    h5open(out_in, "w") do fo
        fo["log10tau500"] = lt
        fo["x"] = xi; attributes(fo["x"])["units"] = "m"
        fo["y"] = y; attributes(fo["y"])["units"] = "m"
        fo["temperature"] = T_in; fo["B_x"] = bx; fo["B_y"] = byv; fo["B_z"] = bz
        meta(attributes(fo))
        attributes(fo)["description"] = "one line of sight per footprint (x, y at z = 0) on log10 tau500 along the line of sight; B local frame"
    end
    h5open(out_truth, "w") do fo
        p(A) = permutedims(A, ndims(A):-1:1)
        fo["truth/z_m"] = p(zt); fo["truth/lnP"] = p(uP); fo["truth/lnrho"] = p(ur); fo["truth/T"] = p(T_in)
        fo["truth/z_tau1_m"] = p(z1)
        fo["logtau500"] = lt; fo["x_m"] = xi; fo["y_m"] = y
        meta(attributes(fo)); attributes(fo)["top_extrapolated_rays"] = ntop[]
    end
    println("wrote $out_in and $out_truth")
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
