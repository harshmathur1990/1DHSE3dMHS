# Poisson pressure step for an atmosphere given on a log10 τ500 grid (as an inversion delivers it): T and B are known at
# fixed log10 τ500 in every column; the heights z(τ), the gas pressure and the density follow from magnetohydrostatics.
#
# 0. Start: each column hydrostatic in τ, d ln P/d ln τ = g τ ρ/(χ P), dz/d ln τ = −τ/χ (χ = χ₅₀₀ per metre from the EOS
#    table), the top pressure from P = g τ/κ with κ = χ/ρ. Columns are then shifted vertically so that the total
#    pressure P + B²/2μ₀ is the same at one depth (thin-flux-tube estimate of the Wilson depression; MHS_ALIGN = tau1
#    instead puts τ = 1 at z = 0 in every column).
# 1. Fixed-point iteration:
#    a. map T, B and ln P from z(τ) onto a uniform height grid covering all columns (outside a column's τ range: T and B
#       held at the end values, ln P isothermal-hydrostatic, Lorentz force zero; the padding is then in hydrostatic
#       balance, which matters because the Poisson step holds the bottom-plane mean of ln P);
#    b. Poisson step on that grid (poisson_core.jl: T and B fixed, ln P by Gauss–Newton, column HSE above z_cut);
#    c. from the new pressure, a = 1 + H d ln P/dz (the non-hydrostatic part of the vertical pressure gradient: Lorentz
#       force and the coupling between columns; zero above z_cut), interpolated to each τ level's height; each column
#       re-integrated in τ from its top with d ln P/d ln τ = (1 − a) g τ ρ/(χ P), dz/d ln τ = −τ/χ (the τ grid's own
#       representation: ln T linear in log10 τ, so the vertical structure never depends on the height-grid mapping),
#       then shifted vertically so that its ln P matches the Poisson pressure field below z_cut; there the levels then take
#       the Poisson pressure at their new height (above z_cut the τ-hydrostatic pressure stays);
#    d. gauge: the mean over columns of z(τ = 1) is 0.
#    Stop when the 99th percentile of |Δz| over all levels and columns is below MHS_TAU_TOL_KM (default 1 km).
#
# input (h5py shapes; see make_tau_input.py): logtau500 (nt, increasing), x_m, y_m, T [K], Bx, By, Bz [T] (nt, ny, nx);
#   optional ref_z_m, ref_Pgas (nt, ny, nx) are only compared against at the end.
# output: /tau_grid/{logtau500, z_m, Pgas, rho, T, Bx, By, Bz} (nt, ny, nx) and the last height-grid state in the MHS
#   checkpoint layout (/atmosphere, /coords), so force_split.py and the map scripts read it.
#
# usage (from HSEMHS/): julia --project=. -t 64 scripts/poisson/poisson_tau.jl IN.h5 OUT.h5 [z_cut_Mm = 1.0]
#   ENV MHS_CROP = n            : only the first n × n columns (tests)
#   ENV MHS_DZ_KM = 10          : height-grid spacing
#   ENV MHS_TAU_ITERS = 20      : maximum fixed-point iterations
#   ENV MHS_TAU_TOL_KM = 1      : stop tolerance on p99 |Δz|
#   ENV MHS_TAU_RELAX = 1       : under-relaxation of z(τ) and ln P(τ) updates
#   ENV MHS_POISSON_OUTER = 30  : maximum Gauss–Newton steps per Poisson step
#   ENV MHS_ALIGN = ptot | tau1 : initial vertical alignment of the columns
#   ENV MHS_TAU_FROMREF = 1     : test: start from the input's ref_z_m, ref_Pgas instead of hydrostatic columns
#   ENV MHS_TAU_DEBUG = 1       : log ln P_in − ln P_Poisson by height band and τ level
include(joinpath(@__DIR__, "poisson_core.jl"))

const LN10 = log(10.0)

"""ρ/P [s² m⁻²] (= 1/(g H)) and χ₅₀₀ [m⁻¹] at (ln T, ln P)."""
@inline function q_chi(tab, lT, lP)
    r, _, c, _ = H.state(tab, lT, lP)
    return exp(r - lP), exp(c)
end

"""Column on the τ grid, integrated down from the top: d ln P/d ln τ = fac · g τ ρ/(χ P), dz/d ln τ = −τ/χ, with
fac = 1 − a (a = 1 + H d ln P/dz, the non-hydrostatic part of the vertical pressure gradient; fac ≡ 1 is hydrostatic).
Fills lnP and z (z = 0 where log10 τ = 0). Returns the number of level intervals where ln P had to be capped (rise > 3:
a hot-over-cool jump that linear ln T in log τ cannot represent)."""
function hse_tau!(lnP, z, lT, lt, tab, fac = nothing)
    nt = length(lt)
    τ = exp.(LN10 .* lt)
    # top: P = g τ ρ/χ (κ constant above the top level), a damped fixed point in ln P
    lp = log(1e2)
    for _ in 1:200
        q, χ = q_chi(tab, lT[1], lp)
        new = log(G0 * τ[1] * q / χ) + lp                          # ln(g τ ρ/χ)
        abs(new - lp) < 1e-10 && break
        lp = 0.5 * lp + 0.5 * new
    end
    lnP[1] = lp
    # d ln P/d ln τ = g τ ρ/(χ P) and dz/d ln τ = −τ/χ, with ln T linear in log10 τ between levels. Heun substeps that
    # change ln P by at most 0.05: across a hot-over-cool jump P can grow tenfold within 0.1 dex in τ (κ drops by orders).
    rhs(lTs, lτ, l, φ) = (s = q_chi(tab, lTs, l); (φ * G0 * exp(lτ) * s[1] / s[2], -exp(lτ) / s[2]))
    fk(k) = fac === nothing ? 1.0 : fac[k]
    z[1] = 0.0
    ncap = 0
    for k in 1:nt-1
        a, b = LN10 * lt[k], LN10 * lt[k+1]
        l = lnP[k]; zz = z[k]; x = a
        while x < b
            sx = (x - a) / (b - a)
            f1, g1 = rhs(lT[k] + (lT[k+1] - lT[k]) * sx, x, l, fk(k) + (fk(k + 1) - fk(k)) * sx)
            h = min(b - x, 0.05 / max(abs(f1), 1e-30))
            sy = (x + h - a) / (b - a)
            f2, g2 = rhs(lT[k] + (lT[k+1] - lT[k]) * sy, x + h, l + h * f1, fk(k) + (fk(k + 1) - fk(k)) * sy)
            if !isfinite(f2) || l + h / 2 * (f1 + f2) - lnP[k] > 3
                l = lnP[k] + 3; ncap += 1; break
            end
            l += h / 2 * (f1 + f2); zz += h / 2 * (g1 + g2); x += h
        end
        lnP[k+1] = l; z[k+1] = zz
    end
    z .-= interp_lt(z, lt, 0.0)
    return ncap
end

"""Value of a column quantity at log10 τ = L (linear in log10 τ; extrapolated from the end segments)."""
function interp_lt(v, lt, L)
    n = length(lt)
    k = clamp(searchsortedlast(lt, L), 1, n - 1)
    t = (L - lt[k]) / (lt[k+1] - lt[k])
    return (1 - t) * v[k] + t * v[k+1]
end

"""Shift columns so that P + B²/2μ₀ is equal at one height (the median bottom total pressure)."""
function align_ptot!(z, lnP, lT, Bx, By, Bz, tab)
    nt, ny, nx = size(z)
    lpt(k, j, i) = log(exp(lnP[k, j, i]) + (Bx[k, j, i]^2 + By[k, j, i]^2 + Bz[k, j, i]^2) / (2MU0))
    ref = median(lpt(nt, j, i) for j in 1:ny, i in 1:nx)
    zr = zeros(ny, nx); nbelow = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            if lpt(nt, j, i) < ref                                 # reference lies below the column: hydrostatic extrapolation
                q, _ = q_chi(tab, lT[nt, j, i], lnP[nt, j, i])
                zr[j, i] = z[nt, j, i] - (ref - lpt(nt, j, i)) / (G0 * q)
                Threads.atomic_add!(nbelow, 1)
            else
                k = nt
                while k > 1 && lpt(k - 1, j, i) >= ref
                    k -= 1
                end
                if k == 1
                    zr[j, i] = z[1, j, i]
                else
                    a, b = lpt(k - 1, j, i), lpt(k, j, i)
                    zr[j, i] = z[k-1, j, i] + (z[k, j, i] - z[k-1, j, i]) * (ref - a) / (b - a)
                end
            end
        end
    end
    for i in 1:nx, j in 1:ny
        @views z[:, j, i] .-= zr[j, i]
    end
    @printf("%s aligned on P + B²/2μ₀ = %.4e Pa (%d columns extrapolated below their bottom level)\n", stamp(), exp(ref), nbelow[])
end

"""Gauge: the mean over columns of z(τ = 1) becomes 0. Returns the shift applied."""
function gauge!(z, lt)
    nt, ny, nx = size(z)
    s = mean(interp_lt(@view(z[:, j, i]), lt, 0.0) for j in 1:ny, i in 1:nx)
    z .-= s
    return s
end

"""Map ln T, B, ln P from each column's z(τ) onto the height grid zg (top first). Returns the inside mask."""
function to_height!(gT, gP, gBx, gBy, gBz, inside, zg, z, lT, lnP, Bx, By, Bz, tab)
    nt, ny, nx = size(z); ng = length(zg)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            ztop = z[1, j, i]; zbot = z[nt, j, i]
            qt, _ = q_chi(tab, lT[1, j, i], lnP[1, j, i]); qb, _ = q_chi(tab, lT[nt, j, i], lnP[nt, j, i])
            k = 1
            for g in 1:ng
                zz = zg[g]
                if zz >= ztop
                    gT[g, j, i] = lT[1, j, i]; gP[g, j, i] = lnP[1, j, i] - G0 * qt * (zz - ztop)
                    gBx[g, j, i] = Bx[1, j, i]; gBy[g, j, i] = By[1, j, i]; gBz[g, j, i] = Bz[1, j, i]
                    inside[g, j, i] = false
                elseif zz <= zbot
                    gT[g, j, i] = lT[nt, j, i]
                    gP[g, j, i] = lnP[nt, j, i] + G0 * qb * (zbot - zz)
                    gBx[g, j, i] = Bx[nt, j, i]; gBy[g, j, i] = By[nt, j, i]; gBz[g, j, i] = Bz[nt, j, i]
                    inside[g, j, i] = false
                else
                    while z[k+1, j, i] > zz
                        k += 1
                    end
                    t = (z[k, j, i] - zz) / (z[k, j, i] - z[k+1, j, i])
                    gT[g, j, i] = (1 - t) * lT[k, j, i] + t * lT[k+1, j, i]
                    gP[g, j, i] = (1 - t) * lnP[k, j, i] + t * lnP[k+1, j, i]
                    gBx[g, j, i] = (1 - t) * Bx[k, j, i] + t * Bx[k+1, j, i]
                    gBy[g, j, i] = (1 - t) * By[k, j, i] + t * By[k+1, j, i]
                    gBz[g, j, i] = (1 - t) * Bz[k, j, i] + t * Bz[k+1, j, i]
                    inside[g, j, i] = true
                end
            end
        end
    end
end

"""Value at height zz of a column v on the uniform, top-first height grid zg (clamped to the grid)."""
@inline function zinterp(v, zg, zz)
    ng = length(zg)
    s = clamp((zg[1] - zz) / (zg[1] - zg[2]) + 1, 1.0, Float64(ng))
    g = min(floor(Int, s), ng - 1); t = s - g
    return (1 - t) * v[g] + t * v[g+1]
end

"""
New z(τ), ln P(τ) from the Poisson pressure P on the height grid: a = 1 + H d ln P/dz (zero above z_cut and outside the
column's τ range) at each level's current height, the column re-integrated in τ from its top with fac = 1 − a, and then
shifted vertically so that its ln P matches ln P_Poisson(z) in the mean over the levels below z_cut (H-weighted Newton).
Below z_cut the levels then take ln P_Poisson at their new height: the re-integrated pressure agrees with it to 1–3 %,
but that much column-to-column jitter alone costs ~0.07 ρg of horizontal force.
Returns (clamped a values, capped level intervals).
"""
function retau!(znew, lPnew, z, lT, lt, zg, P, rho, inside, zcut, tab)
    nt, ny, nx = size(z); ng = length(zg)
    lP = log.(P); dl = similar(lP); ddz!(dl, lP, zg)
    ag = similar(lP)
    @. ag = ifelse(inside & (zg <= zcut), 1 + dl * P / (rho * G0), 0.0)
    ncl = Threads.Atomic{Int}(0); ncap = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        fac = zeros(nt)
        @inbounds for j in 1:ny
            agc = @view ag[:, j, i]; lPc = @view lP[:, j, i]
            for k in 1:nt
                a = zinterp(agc, zg, z[k, j, i])
                (a < -5 || a > 0.95) && Threads.atomic_add!(ncl, 1)
                fac[k] = 1 - clamp(a, -5.0, 0.95)
            end
            zc = @view znew[:, j, i]; pc = @view lPnew[:, j, i]
            Threads.atomic_add!(ncap, hse_tau!(pc, zc, @view(lT[:, j, i]), lt, tab, fac))
            s = interp_lt(@view(z[:, j, i]), lt, 0.0)                    # start from the old height of τ = 1
            for _ in 1:3
                num = 0.0; cnt = 0
                for k in 1:nt
                    zz = zc[k] + s
                    (zz > zcut || zz > zg[1] || zz < zg[end]) && continue
                    q, _ = q_chi(tab, lT[k, j, i], pc[k])
                    num += (zinterp(lPc, zg, zz) - pc[k]) / (G0 * q); cnt += 1
                end
                cnt > 0 && (s += num / cnt)
            end
            zc .+= s
            for k in 1:nt                                          # below z_cut the pressure is the Poisson pressure itself
                (zc[k] <= zcut && zg[end] <= zc[k] <= zg[1]) && (pc[k] = zinterp(lPc, zg, zc[k]))
            end
        end
    end
    return ncl[], ncap[]
end

"""RMS of |F|/(layer median ρg) over the cells with mask (F = −∇P + L − ρg ẑ, centred differences at cell centres)."""
function force_rms(tab, lnT, P, Lx, Ly, Lz, zg, dx, dy, mask)
    ng, ny, nx = size(P)
    rho = similar(P)
    Threads.@threads for idx in eachindex(P)
        r, _, _ = H.lnrho_dY(tab, lnT[idx], log(P[idx])); rho[idx] = exp(r)
    end
    gx = similar(P); gy = similar(P); gz = similar(P)
    ddx!(gx, P, dx); ddy!(gy, P, dy); ddz!(gz, P, zg)
    s = 0.0; c = 0
    for g in 1:ng
        m = @view mask[g, :, :]
        any(m) || continue
        S = median(@view(rho[g, :, :])[m]) * G0
        for i in 1:nx, j in 1:ny
            m[j, i] || continue
            fx = -gx[g, j, i] + Lx[g, j, i]; fy = -gy[g, j, i] + Ly[g, j, i]
            fz = -gz[g, j, i] + Lz[g, j, i] - rho[g, j, i] * G0
            s += (fx^2 + fy^2 + fz^2) / S^2; c += 1
        end
    end
    return sqrt(s / c)
end

pct(v, p) = quantile(vec(v), p)

function main()
    infile, outfile = ARGS[1], ARGS[2]
    zcut = parse(Float64, get(ARGS, 3, "1.0")) * 1e6
    crop = parse(Int, get(ENV, "MHS_CROP", "0"))
    dzg = parse(Float64, get(ENV, "MHS_DZ_KM", "10")) * 1e3
    maxit = parse(Int, get(ENV, "MHS_TAU_ITERS", "20"))
    tol = parse(Float64, get(ENV, "MHS_TAU_TOL_KM", "1")) * 1e3
    ω = parse(Float64, get(ENV, "MHS_TAU_RELAX", "1"))
    maxouter = parse(Int, get(ENV, "MHS_POISSON_OUTER", "30"))
    align = get(ENV, "MHS_ALIGN", "ptot")
    FFTW.set_num_threads(1)              # FFTW's threaded execution segfaults under Julia 1.13 (spawn_apply)

    # ---- input on the τ grid: Julia arrays (nt, ny, nx), level 1 = top ----
    zyx(A) = permutedims(A, (3, 2, 1))
    lt, x, y, T, Bx, By, Bz, zref, Pref = h5open(infile, "r") do f
        c = crop > 0 ? (1:crop) : Colon()
        rd(nm) = zyx(Float64.(read(f[nm])))[:, c, c]
        opt(nm) = haskey(f, nm) ? rd(nm) : nothing
        read(f["logtau500"]), read(f["x_m"])[c], read(f["y_m"])[c], rd("T"), rd("Bx"), rd("By"), rd("Bz"),
            opt("ref_z_m"), opt("ref_Pgas")
    end
    @assert issorted(lt) "logtau500 must increase (top first)"
    nt, ny, nx = size(T)
    dx, dy = x[2] - x[1], y[2] - y[1]
    tab = H.load_table(TAB)
    lT = log.(T)
    println(stamp(), " input $infile: $nt levels log10 τ500 $(lt[1]) … $(lt[end]), $(ny)×$(nx) columns, dx $(round(dx/1e3, digits = 2)) km; ",
            "height grid dz $(dzg/1e3) km, z_cut $(zcut/1e6) Mm, threads $(Threads.nthreads()), EOS $TAB")

    # ---- 0: hydrostatic columns in τ, then vertical alignment ----
    lnP = zeros(nt, ny, nx); z = zeros(nt, ny, nx)
    ncap = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            c = hse_tau!(@view(lnP[:, j, i]), @view(z[:, j, i]), @view(lT[:, j, i]), lt, tab)
            Threads.atomic_add!(ncap, c)
        end
    end
    any(!isfinite, lnP) && error("column HSE in τ left the EOS table in $(count(!isfinite, lnP)) points")
    println(stamp(), " hydrostatic columns in τ: ln P capped in $(ncap[]) level intervals")
    align == "ptot" && align_ptot!(z, lnP, lT, Bx, By, Bz, tab)
    if get(ENV, "MHS_TAU_FROMREF", "0") == "1" && zref !== nothing && Pref !== nothing   # test: start from the reference
        z .= zref; lnP .= log.(Pref)
        println(stamp(), " start replaced by the reference z(τ), P(τ) (MHS_TAU_FROMREF)")
    end
    gauge!(z, lt)
    k0 = argmin(abs.(lt))
    @printf("%s start: z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km; z(top level) max %.1f km, z(bottom level) min %.1f km\n",
            stamp(), pct(z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3..., maximum(z[1, :, :]) / 1e3, minimum(z[nt, :, :]) / 1e3)

    # ---- 1: fixed-point iteration ----
    znew = similar(z); lPnew = similar(lnP)
    local zg, gT, gP, gBx, gBy, gBz, P, rho, tau, zt1, inside
    hist = Float64[]
    for it in 1:maxit
        t0 = time()
        ztop = maximum(@view z[1, :, :]) + 2dzg; zbot = minimum(@view z[nt, :, :]) - 2dzg
        ng = ceil(Int, (ztop - zbot) / dzg) + 1
        zg = collect(range(ztop, ztop - (ng - 1) * dzg; length = ng))
        gT, gP, gBx, gBy, gBz = (zeros(ng, ny, nx) for _ in 1:5)
        inside = falses(ng, ny, nx)
        to_height!(gT, gP, gBx, gBy, gBz, inside, zg, z, lT, lnP, Bx, By, Bz, tab)
        Lx, Ly, Lz = lorentz(gBx, gBy, gBz, zg, dx, dy)
        @. Lx *= inside; @. Ly *= inside; @. Lz *= inside                 # no magnetic force in the padding
        @printf("%s iteration %d: height grid %d layers, z %.3f … %.3f Mm, %.1f %% of cells inside the τ range\n",
                stamp(), it, ng, zg[1] / 1e6, zg[end] / 1e6, 100 * count(inside) / length(inside))
        sol = solve_pressure(tab, gT, Lx, Ly, Lz, exp.(gP), zg, dx, dy; zcut = zcut, maxouter = maxouter)
        P = sol.P
        msk = inside .& (zg .<= zcut)
        msk[1, :, :] .= false; msk[end, :, :] .= false
        fin = force_rms(tab, gT, exp.(gP), Lx, Ly, Lz, zg, dx, dy, msk); fout = force_rms(tab, gT, P, Lx, Ly, Lz, zg, dx, dy, msk)
        if it > 1                                                  # how well the τ-grid state reproduced the last Poisson pressure
            d = gP .- log.(P)
            cm = [any(@view msk[:, j, i]) ? mean(@view(d[:, j, i])[@view msk[:, j, i]]) : 0.0 for j in 1:ny, i in 1:nx]
            w = [d[g, j, i] - cm[j, i] for g in 1:size(d, 1), j in 1:ny, i in 1:nx if msk[g, j, i]]
            @printf("%s   ln P_in − ln P_Poisson inside: all p1/p50/p99 %.3f / %.3f / %.3f; column means p1/p99 %.3f / %.3f; within columns p1/p99 %.3f / %.3f\n",
                    stamp(), pct(d[msk], [0.01, 0.5, 0.99])..., pct(cm, [0.01, 0.99])..., pct(w, [0.01, 0.99])...)
            if get(ENV, "MHS_TAU_DEBUG", "0") == "1"
                for (lo, hi) in ((-1e9, 0.0), (0.0, 2e5), (2e5, 4e5), (4e5, 6e5), (6e5, 8e5), (8e5, 1e9))
                    sel = [d[g, j, i] for g in eachindex(zg), j in 1:ny, i in 1:nx if msk[g, j, i] && lo <= zg[g] < hi]
                    isempty(sel) || @printf("%s     z in [%.1f, %.1f) Mm: n %d, p1/p50/p99 %.3f / %.3f / %.3f\n", stamp(), max(lo, -9e6) / 1e6, min(hi, 9e6) / 1e6, length(sel), pct(sel, [0.01, 0.5, 0.99])...)
                end
                # by τ level: Poisson ln P interpolated to each level's height vs the level's ln P (before the update)
                for L in (1.5, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0)
                    k = argmin(abs.(lt .- L))
                    v = [lnP[k, j, i] - zinterp(log.(@view(P[:, j, i])), zg, z[k, j, i]) for j in 1:ny, i in 1:nx if z[k, j, i] <= zcut]
                    isempty(v) || @printf("%s     log τ %.1f: ln P(τ) − ln P_Poisson(z(τ)) p1/p50/p99 %.3f / %.3f / %.3f (n %d)\n", stamp(), lt[k], pct(v, [0.01, 0.5, 0.99])..., length(v))
                end
            end
        end
        rho, tau, zt1 = tau_columns(tab, gT, P, zg)
        ncl, ncap = retau!(znew, lPnew, z, lT, lt, zg, P, rho, inside, zcut, tab)
        @. znew = z + ω * (znew - z); @. lPnew = lnP + ω * (lPnew - lnP)
        gauge!(znew, lt)
        dz = abs.(znew .- z); dlp = abs.(lPnew .- lnP)
        z .= znew; lnP .= lPnew
        push!(hist, pct(dz, 0.99))
        @printf("%s iteration %d: |Δz| p50/p99/max %.3f / %.3f / %.3f km, |Δ ln P(τ)| p99 %.2e, z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km | RMS |F|/ρg inside: handed to Poisson %.4f, after %.4f | Poisson Φ_end/Φ_start %.3f in %d steps, a clamped %d, ln P capped %d | %.1f s\n",
                stamp(), it, pct(dz, 0.5) / 1e3, pct(dz, 0.99) / 1e3, maximum(dz) / 1e3, pct(dlp, 0.99),
                pct(z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3..., fin, fout, sol.Φ / sol.Φ0, sol.steps, ncl, ncap, time() - t0)
        hist[end] < tol && (println(stamp(), " converged (p99 |Δz| < $(tol/1e3) km)"); break)
    end

    # ---- density on the τ grid from the EOS ----
    rhot = similar(lnP)
    Threads.@threads for idx in eachindex(lnP)
        r, _, _ = H.lnrho_dY(tab, lT[idx], lnP[idx]); rhot[idx] = exp(r)
    end
    if zref !== nothing
        zr = copy(zref); gauge!(zr, lt)
        for L in (0.0, -1.0, -2.0, -3.0, -4.0)
            L < lt[1] && continue
            d = [interp_lt(@view(z[:, j, i]), lt, L) - interp_lt(@view(zr[:, j, i]), lt, L) for j in 1:ny, i in 1:nx]
            @printf("%s vs reference at log10 τ = %4.1f: z − z_ref (both gauged) p1/p50/p99 %.1f / %.1f / %.1f km\n",
                    stamp(), L, pct(d, [0.01, 0.5, 0.99]) ./ 1e3...)
        end
        if Pref !== nothing
            d = lnP .- log.(Pref)
            @printf("%s vs reference: ln(P/P_ref) p1/p50/p99 %.3f / %.3f / %.3f\n", stamp(), pct(d, [0.01, 0.5, 0.99])...)
        end
    end

    xyz(A) = permutedims(A, (3, 2, 1))
    h5open(outfile, "w") do f
        t = create_group(f, "tau_grid")
        t["logtau500"] = lt; t["z_m"] = xyz(z); t["Pgas"] = xyz(exp.(lnP)); t["rho"] = xyz(rhot)
        t["T"] = xyz(T); t["Bx"] = xyz(Bx); t["By"] = xyz(By); t["Bz"] = xyz(Bz)
        a = create_group(f, "atmosphere"); c = create_group(f, "coords")
        a["Pgas"] = xyz(P); a["rho"] = xyz(rho); a["T"] = xyz(exp.(gT)); a["tau500"] = xyz(tau)
        a["Bx_centre"] = xyz(gBx); a["By_centre"] = xyz(gBy); a["Bz_centre"] = xyz(gBz)
        a["inside_tau_range"] = xyz(UInt8.(inside))
        a["z_tau1_m"] = permutedims(zt1, (2, 1))
        c["x_m"] = x; c["y_m"] = y; c["z_m"] = zg
        attributes(f)["accepted"] = 0
        attributes(f)["kind"] = "poisson_tau"
        attributes(f)["input"] = infile
        attributes(f)["z_cut_m"] = zcut
        attributes(f)["dz_m"] = dzg
        attributes(f)["eos_table"] = TAB
        attributes(f)["p99_dz_history_m"] = hist
        attributes(f)["seconds"] = time() - t_start
    end
    println(stamp(), " wrote $outfile")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
