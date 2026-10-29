#=
Known-equilibrium control for mhs-poisson-ltau (design/CODEX_REVIEW_solver_time_round3_2026-10-29.md, priority
experiment step 3): an exact 2-D magnetostatic atmosphere (x periodic, invariant in y) mapped onto inclined rays the way
make_slanted_tau.jl maps Bifrost, but from analytic fields on a fine height grid, so the input and its truth are
consistent with the solver's continuous model and the only errors left are the solver's discretisation.

Reference: T₀(τ) grey Eddington-Hopf (T_eff 5772 K) in a vertical column HSE in τ gives T₀(z); P₀(z) is then integrated
exactly in height (RK4, dP₀/dz = −ρ(T₀, P₀) g), ρ₀ = ρ(T₀, P₀), z = 0 at τ₅₀₀ = 1.

Field (Grad–Shafranov, B = ∇ × (A ŷ) + B_y ŷ, B_y constant): A = A₀ cos(kx) f(z), p(A, z) = P₀(z) − m(z) A²/2. Force balance
−∇p − ρ g ẑ + J × B = 0 holds exactly when ∇²A = μ₀ m A, i.e. f'' = (k² + μ₀ m) f, and ρ = −∂_z p|_A / g. m(z) is chosen so
that the pressure deficit is a fixed fraction D of P₀: m A₀² f²/2 = D P₀, so

    P = P₀ (1 − D cos² kx),    ρ = ρ₀ (1 − D cos² kx) − 2 D P₀ f'/(f g) cos² kx,
    f'' = k² f + k² D P₀/(P_m f),   A₀ = √(2 μ₀ P_m)/k,   P_m = P₀(0),
    B_x = −A₀ cos(kx) f',  B_z = −A₀ k sin(kx) f,

f integrated down from the top, starting on the isothermal self-similar solution f ∝ √P₀ (f'/f = −1/2H, where
ρ = ρ₀ and P_m f² = D P₀ (2kH)²/(1 − (2kH)²)). T = EOS⁻¹(ρ, P)
(bisection in ln T). D = 0 with field = 0 is the field-free hydrostatic control (horizontally uniform).

Rays: x = ξ + t_x z through footprints ξ (y-invariant), sampled on the fine z grid; χ₅₀₀ = χ(T, P), τ_LOS by
Tau.tau_column! (dz/μ), every quantity interpolated linearly in log₁₀ τ_LOS onto the axis lt_min:dlt:lt_max (as
make_slanted_tau.jl). Also checks the force balance of the construction by finite differences on a fine (x, z) grid.

usage: julia --project=. -t N scripts/viewing/make_mhs_control.jl MU AZIMUTH_DEG OUT_INPUT.nc OUT_TRUTH.h5 [KEY=VALUE ...]
KEY: nx (256), ny (8), box_mm (12), period_mm (3), dlt (0.1), lt_min (−8), lt_max (1.4), D (0.3), field (1),
by_gauss (0), dz_m (250). Outputs as make_slanted_tau.jl.
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

"""ln T with ln ρ(ln T, ln P) = lr (ρ decreases with T at fixed P)."""
function invT(tab, lr, lP)
    a = tab.X[1] + 1e-9; b = tab.X[end] - 1e-9
    fa = lnrho(tab, a, lP) - lr; fb = lnrho(tab, b, lP) - lr
    (isfinite(fa) && isfinite(fb) && fa > 0 > fb) || return NaN
    for _ in 1:80
        c = (a + b) / 2; fc = lnrho(tab, c, lP) - lr
        fc > 0 ? (a = c) : (b = c)
        b - a < 1e-13 && break
    end
    return (a + b) / 2
end

T_edd(τ; Teff = 5772.0) = Teff * (0.75 * (τ + 0.7104 - 0.1331 * exp(-3.4488τ)))^0.25

function main(mu_s, az_s, out_in, out_truth, kv...)
    H.ensure_mpi()
    o = Dict("nx" => 256.0, "ny" => 8.0, "box_mm" => 12.0, "period_mm" => 3.0, "dlt" => 0.1, "lt_min" => -8.0, "lt_max" => 1.4,
             "D" => 0.3, "field" => 1.0, "by_gauss" => 0.0, "dz_m" => 250.0)
    for a in kv
        k, v = split(a, "="); haskey(o, k) || error("unknown key $k"); o[k] = parse(Float64, v)
    end
    μ = parse(Float64, mu_s); φ = parse(Float64, az_s)
    v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
    isempty(H.option_errors(v)) || error(join(H.option_errors(v), "; "))
    abs(sind(φ)) < 1e-12 || error("the control is y-invariant: azimuth 0 or 180 only")
    t = H.slant(v); g = 274.0
    tab = H.load_table("cache/eos_table_fullz.h5")
    NX = Int(o["nx"]); NY = Int(o["ny"]); Lb = o["box_mm"] * 1e6; dx = Lb / NX
    k = 2π / (o["period_mm"] * 1e6)
    isapprox(Lb * k / 2π, round(Lb * k / 2π); atol = 1e-9) || error("box must hold whole periods")
    nlev = round(Int, (o["lt_max"] - o["lt_min"]) / o["dlt"]) + 1
    lt = collect(range(o["lt_min"], o["lt_max"]; length = nlev)); nt = length(lt)
    D = o["field"] != 0 ? o["D"] : 0.0

    # 1. vertical reference column HSE in τ (RK4 in ln τ): T₀(z)
    lτv = collect(range(-11.5 * log(10), 3.0 * log(10); step = 0.002 * log(10))); nv = length(lτv)
    function rhs(lτ, lP)
        τ = exp(lτ); lT = log(T_edd(τ)); r, _, c, _ = H.state(tab, lT, lP)
        return g * τ * exp(r - lP - c), -τ * exp(-c)               # d ln P/d ln τ, dz/d ln τ
    end
    lP = log(1e-3)
    for _ in 1:200                                                   # top closure P = g τ ρ/χ
        τ = exp(lτv[1]); r, _, c, _ = H.state(tab, log(T_edd(τ)), lP); lP = log(g * τ) + r - c
    end
    Pv = zeros(nv); zv = zeros(nv); Pv[1] = lP
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
    if D > 0
        H1 = P0[1] / (ρ0[1] * g); q = 2k * H1                       # top: the isothermal self-similar f ∝ √P₀ (ρ = ρ₀)
        q < 0.95 || error("2 k H at the top is $q: use a longer period")
        f[1] = sqrt(D * q^2 / (1 - q^2) * P0[1] / Pm); fp[1] = -f[1] / (2H1)
        P0z = z -> exp(lin(reverse(zf), reverse(lP0), z))
        acc = (z, y) -> k^2 * y + k^2 * D * P0z(z) / (Pm * y)
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
    fields(n, x) = (lP = lP0[n] + log1p(-D * c2(x));                 # (ln P, ln ρ, Bx, Bz) at fine level n
                    ρ = ρ0[n] * (1 - D * c2(x)) - (D > 0 ? 2D * P0[n] * fp[n] / (f[n] * g) * c2(x) : 0.0);
                    (lP, log(ρ), -A0 * cos(k * x) * fp[n], -A0 * k * sin(k * x) * f[n]))
    i0 = argmin(abs.(zf))
    @printf("μ %.3f φ %g t (%.3f, %.3f); box %.1f Mm, period %.1f Mm, %d × %d, dx %.1f km; τ axis %g:%g:%g (%d levels); fine dz %.0f m, %d levels %.0f … %.0f km\n",
            μ, φ, t..., Lb / 1e6, o["period_mm"], NX, NY, dx / 1e3, lt[1], o["dlt"], lt[end], nt, dz, nz, zf[1] / 1e3, zf[end] / 1e3)
    @printf("D %.2f; at z = 0: P₀ %.4g Pa, T₀ %.0f K; max |B| at z = 0: %.0f G, at the top %.0f G; f(0) %.3g\n",
            D, P0[i0], exp(lT0[i0]), D > 0 ? A0 * max(abs(fp[i0]), k * f[i0]) * 1e4 : 0.0,
            D > 0 ? A0 * max(abs(fp[1]), k * f[1]) * 1e4 : 0.0, f[i0])

    # 4. construction check: force balance by centred differences on a fine (x, z) grid, relative to ρ g
    if D > 0
        nxf = 512; xs = (0:nxf-1) .* (o["period_mm"] * 1e6 / nxf); hx = xs[2]
        worst = (0.0, 0.0)
        for n in 3:20:nz-2
            for x in xs
                lPc, lrc, bx, bz = fields(n, x)
                dPdx = (exp(fields(n, x + hx)[1]) - exp(fields(n, x - hx)[1])) / 2hx
                dPdz = (exp(fields(n - 1, x)[1]) - exp(fields(n + 1, x)[1])) / 2dz
                dBxdz = (fields(n - 1, x)[3] - fields(n + 1, x)[3]) / 2dz
                dBzdx = (fields(n, x + hx)[4] - fields(n, x - hx)[4]) / 2hx
                Jy = (dBxdz - dBzdx) / MU0
                Rx = -dPdx + Jy * bz; Rz = -dPdz - exp(lrc) * g - Jy * bx
                s = exp(lrc) * g
                worst = (max(worst[1], abs(Rx) / s), max(worst[2], abs(Rz) / s))
            end
        end
        @printf("construction force balance (centred differences, dx %.1f km, dz %.0f m): max |R_x|/ρg %.2e, |R_z|/ρg %.2e\n",
                hx / 1e3, dz, worst...)
    end

    # 5. rays through footprints ξ_i (y-invariant)
    xi = collect((0:NX-1) .* dx)
    T_in = fill(NaN, nt, NY, NX); bx = similar(T_in); byv = similar(T_in); bz = similar(T_in)
    zt = similar(T_in); uP = similar(T_in); ur = similar(T_in); z1 = fill(NaN, NY, NX)
    nmiss = Threads.Atomic{Int}(0); ntop = Threads.Atomic{Int}(0); nbad = Threads.Atomic{Int}(0)
    lTmin = Threads.Atomic{Float64}(Inf); lTmax = Threads.Atomic{Float64}(-Inf)
    Threads.@threads for i in 1:NX
        lT = zeros(nz); lPr = zeros(nz); lr = zeros(nz); Bx = zeros(nz); Bz = zeros(nz); χ = zeros(nz); tc = zeros(nz)
        for n in 1:nz
            x = xi[i] + t[1] * zf[n]
            lPr[n], lr[n], Bx[n], Bz[n] = fields(n, x)
            lT[n] = D > 0 ? invT(tab, lr[n], lPr[n]) : lT0[n]
            c = isfinite(lT[n]) ? H.state(tab, lT[n], lPr[n])[3] : NaN
            isfinite(c) || (Threads.atomic_add!(nbad, 1); c = -700.0;
                            nbad[] < 5 && @printf("outside: z %.0f km x %.0f km ln P %.3f ln ρ %.3f ln T %.3f (T₀ %.0f K)\n",
                                                  zf[n] / 1e3, x / 1e3, lPr[n], lr[n], lT[n], exp(lT0[n])))
            χ[n] = exp(c)
            Threads.atomic_min!(lTmin, lT[n]); Threads.atomic_max!(lTmax, lT[n])
        end
        H.tau_column!(tc, χ, zf; mu = μ)
        lτ = log10.(max.(tc, 1e-300))
        lt[end] > lτ[end] && Threads.atomic_add!(nmiss, 1)
        lt[1] < lτ[1] && Threads.atomic_add!(ntop, 1)
        zz = H.z_tau1(tc, zf)
        for kk in 1:nt
            Tk = exp(interp_at(lT, lτ, lt[kk])); bxk = interp_at(Bx, lτ, lt[kk]); bzk = interp_at(Bz, lτ, lt[kk])
            zk = interp_at(zf, lτ, lt[kk]; top = :extrapolate)
            pk = interp_at(lPr, lτ, lt[kk]; top = :extrapolate); rk = interp_at(lr, lτ, lt[kk]; top = :extrapolate)
            for j in 1:NY
                T_in[kk, j, i] = Tk; bx[kk, j, i] = bxk; byv[kk, j, i] = by; bz[kk, j, i] = bzk
                zt[kk, j, i] = zk; uP[kk, j, i] = pk; ur[kk, j, i] = rk
            end
        end
        z1[:, i] .= zz
    end
    nbad[] == 0 || error("$(nbad[]) fine points outside the EOS table")
    @printf("T range on the rays %.0f … %.0f K; rays with the top above the axis minimum %d; not reaching the bottom %d\n",
            exp(lTmin[]), exp(lTmax[]), ntop[], nmiss[])
    nmiss[] == 0 || error("rays do not reach the bottom of the τ axis")
    @printf("truth z(τ_LOS = 1): mean %.1f km, min %.1f, max %.1f (range %.1f km)\n",
            sum(z1) / length(z1) / 1e3, minimum(z1) / 1e3, maximum(z1) / 1e3, (maximum(z1) - minimum(z1)) / 1e3)
    y = collect((0:NY-1) .* dx)
    meta(a) = (a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2];
               a["source"] = "manufactured Grad-Shafranov control (make_mhs_control.jl)"; a["gravity_m_s2"] = g;
               for (kk, vv) in o; a["control_" * kk] = vv; end)
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
