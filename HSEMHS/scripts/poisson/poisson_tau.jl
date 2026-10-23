# Poisson pressure step for an atmosphere given on a log10 τ500 grid (as an inversion delivers it): T and B are known at
# fixed log10 τ500 in every column; the heights z(τ), the gas pressure and the density follow from magnetohydrostatics.
#
# 0. Start: each column hydrostatic in τ, d ln P/d ln τ = g τ ρ/(χ P), dz/d ln τ = −τ/χ (χ = χ₅₀₀ per metre from the EOS
#    table), the top pressure from P = g τ/κ with κ = χ/ρ. Columns are then shifted vertically so that the total
#    pressure P + B²/2μ₀ is the same at one depth (thin-flux-tube estimate of the Wilson depression; MHS_ALIGN = tau1
#    instead puts τ = 1 at z = 0 in every column).
# 1. Fixed-point iteration:
#    a. map T, B and ln P from z(τ) onto a uniform height grid covering all columns (outside a column's τ range: T held
#       at the top value / extrapolated linearly in z below, ln P hydrostatic, B constant, Lorentz force zero there);
#    b. Poisson step on that grid (poisson_core.jl: T and B fixed, ln P by Gauss–Newton, column HSE above z_cut);
#    c. τ500 from the new pressure, and z(τ), ln P(τ) at the input τ levels by interpolation in log10 τ;
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
include(joinpath(@__DIR__, "poisson_core.jl"))

const LN10 = log(10.0)

"""ρ/P [s² m⁻²] (= 1/(g H)) and χ₅₀₀ [m⁻¹] at (ln T, ln P)."""
@inline function q_chi(tab, lT, lP)
    r, _, c, _ = H.state(tab, lT, lP)
    return exp(r - lP), exp(c)
end

"""Hydrostatic column on the τ grid: fills lnP and z (z = 0 where log10 τ = 0)."""
function hse_tau!(lnP, z, lT, lt, tab)
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
    function f(k, l)                                               # d ln P/d ln τ = g τ ρ/(χ P)
        q, χ = q_chi(tab, lT[k], l)
        return G0 * τ[k] * q / χ
    end
    for k in 1:nt-1
        Δ = LN10 * (lt[k+1] - lt[k])
        fk = f(k, lnP[k]); l = lnP[k] + Δ * fk
        for _ in 1:50
            ln = lnP[k] + Δ / 2 * (fk + f(k + 1, l))
            abs(ln - l) < 1e-12 && (l = ln; break)
            l = ln
        end
        lnP[k+1] = l
    end
    z[1] = 0.0
    for k in 1:nt-1
        Δ = LN10 * (lt[k+1] - lt[k])
        _, χa = q_chi(tab, lT[k], lnP[k]); _, χb = q_chi(tab, lT[k+1], lnP[k+1])
        z[k+1] = z[k] - Δ / 2 * (τ[k] / χa + τ[k+1] / χb)
    end
    z .-= interp_lt(z, lt, 0.0)
    return lnP, z
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
            sT = (lT[nt-1, j, i] - lT[nt, j, i]) / (z[nt-1, j, i] - zbot)       # d ln T/dz at the bottom
            k = 1
            for g in 1:ng
                zz = zg[g]
                if zz >= ztop
                    gT[g, j, i] = lT[1, j, i]; gP[g, j, i] = lnP[1, j, i] - G0 * qt * (zz - ztop)
                    gBx[g, j, i] = Bx[1, j, i]; gBy[g, j, i] = By[1, j, i]; gBz[g, j, i] = Bz[1, j, i]
                    inside[g, j, i] = false
                elseif zz <= zbot
                    gT[g, j, i] = lT[nt, j, i] + clamp(sT * (zz - zbot), -0.4, 0.4)
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

"""z and ln P at the input τ levels from τ(zg) (linear in log10 τ). Returns the number of extrapolated values."""
function to_tau!(znew, lPnew, zg, tau, P, lt)
    nt, ny, nx = size(znew); ng = length(zg)
    nex = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        lg = zeros(ng)
        @inbounds for j in 1:ny
            for g in 1:ng
                lg[g] = log10(tau[g, j, i])
            end
            g = 1
            for k in 1:nt
                L = lt[k]
                while g < ng - 1 && lg[g+1] < L
                    g += 1
                end
                (L < lg[1] || L > lg[ng]) && Threads.atomic_add!(nex, 1)
                t = (L - lg[g]) / (lg[g+1] - lg[g])
                znew[k, j, i] = zg[g] + t * (zg[g+1] - zg[g])
                lPnew[k, j, i] = log(P[g, j, i]) + t * (log(P[g+1, j, i]) - log(P[g, j, i]))
            end
        end
    end
    return nex[]
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
    Threads.@threads for i in 1:nx
        for j in 1:ny
            a, b = hse_tau!(zeros(nt), zeros(nt), @view(lT[:, j, i]), lt, tab)
            lnP[:, j, i] .= a; z[:, j, i] .= b
        end
    end
    any(!isfinite, lnP) && error("column HSE in τ left the EOS table in $(count(!isfinite, lnP)) points")
    align == "ptot" && align_ptot!(z, lnP, lT, Bx, By, Bz, tab)
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
        rho, tau, zt1 = tau_columns(tab, gT, P, zg)
        nex = to_tau!(znew, lPnew, zg, tau, P, lt)
        @. znew = z + ω * (znew - z); @. lPnew = lnP + ω * (lPnew - lnP)
        gauge!(znew, lt)
        dz = abs.(znew .- z); dlp = abs.(lPnew .- lnP)
        z .= znew; lnP .= lPnew
        push!(hist, pct(dz, 0.99))
        @printf("%s iteration %d: |Δz| p50/p99/max %.3f / %.3f / %.3f km, |Δ ln P(τ)| p99 %.2e, z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km, %d values extrapolated, %.1f s\n",
                stamp(), it, pct(dz, 0.5) / 1e3, pct(dz, 0.99) / 1e3, maximum(dz) / 1e3, pct(dlp, 0.99),
                pct(z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3..., nex, time() - t0)
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

main()
