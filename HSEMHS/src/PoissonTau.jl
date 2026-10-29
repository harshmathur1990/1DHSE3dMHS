# run.mode = "column-hse-ltau" and "mhs-poisson-ltau": atmospheres given on a log10 τ500 grid (inversion output), from
# scripts/poisson/poisson_tau.jl (column HSE in τ) and scripts/poisson/poisson_tau_direct.jl (the direct Poisson solve).
#
# Column HSE in τ: d ln P/d ln τ = g τ ρ/(χ P), dz/d ln τ = −τ/χ, integrated down from the top (P = g τ ρ/χ there) with
# adaptive Heun substeps (ln P changes ≤ 0.05 per substep), no limit on the rise of ln P (with P = g m this is
# d ln m/d ln τ = τ/(κ m), and dz = −dm/ρ: heights strictly decrease); a column whose pressure leaves the EOS table is
# returned as failed (NaN below that level), not altered; columns aligned so that P + B²/2μ₀ is equal at one height;
# gauge mean z(τ = 1) = 0.
#
# Poisson in τ: one Gauss–Newton (Levenberg–Marquardt) solve. Unknowns per column: ln P on the levels below the cut level
# k_c (log10 τ ≥ logtau_cut) and the height c of that level; heights from the pressure, z_k = c + Σ Δz,
# Δz = −∫ τ/χ(T, P) d ln τ (trapezoid): a lower-triangular operator down each column. Above k_c the column is
# τ-hydrostatic (fixed). Residuals (relative, R = F/P): vertical Δ ln P − ∫ (g ρ − L_z) τ/(χ P) d ln τ per interval,
# horizontal −∂_x ln P|_τ − (g ρ − L_z)/P ∂_x z|_τ + L_x/P per level. L = J×B from B on the τ surfaces at their heights
# (curvilinear derivatives); the Jacobian includes L's dependence on the heights. CG on the normal equations,
# preconditioned per horizontal Fourier mode by the layer-median operator A + |k|² B, diagonalised once per step.
# Arrays are (τ, y, x) slabs in x; every reduction is global (Slab.jl).
#
# Viewing angle (design/tau/VIEWING_ANGLE_DESIGN.md): the axis is τ along the line of sight, μ = cos θ. Heights use
# dz = −μ τ/χ d ln τ in both geometries, and column HSE along the ray has d ln P/d ln τ = μ g τ ρ/(χ P). The inclined
# geometry treats each column as a slanted ray x = ξ + t_x z, y = η + t_y z (t = tan θ (cos φ, sin φ)) at its footprint
# (ξ, η): ∂_z|_(x,y) = ∂_ray − t·∇_h in the Lorentz force, and the column balance is the one along the ray,
# W_n = g ρ/P − (L_z + t·L_h)/P in place of W. Plane-parallel (or μ = 1) has t = 0, i.e. the vertical-column code.
# With balance = :vertical (default; SOLVER_TIME_DESIGN.md D0) the column residual is the vertical balance
# R_v = R_ray − Δz t·R̄_h (R̄_h the mean horizontal residual of the interval's two nodes): the along-ray residual mixes the
# horizontal imbalance in with weight tan θ, which swamps the vertical balance at strong inclination.
module PoissonTau

using MPI, FFTW, LinearAlgebra, Printf, HDF5
import ..HSEMHS as H
using ..Slab
import ..Slab: xhalo
import ..HSEMHS: ParCtx, rank0, MU0
using ProgressMeter: Progress, next!, finish!
import ..PoissonZ: tbcast!, pcg!

const LN10 = log(10.0)

"""ρ/P [s² m⁻²] and χ₅₀₀ [m⁻¹] at (ln T, ln P)."""
@inline function q_chi(tab, lT, lP)
    r, _, c, _ = H.state(tab, lT, lP)
    return exp(r - lP), exp(c)
end

"""Value of a column quantity at log10 τ = L (linear in log10 τ; extrapolated from the end segments)."""
function interp_lt(v, lt, L)
    n = length(lt)
    k = clamp(searchsortedlast(lt, L), 1, n - 1)
    t = (L - lt[k]) / (lt[k+1] - lt[k])
    return (1 - t) * v[k] + t * v[k+1]
end

"""
Column on the τ grid, integrated down from the top: d ln P/d ln τ = g τ ρ/(χ P), dz/d ln τ = −τ/χ (z = 0 at log10 τ = 0).
With P = g m (m the column mass) this is d ln m/d ln τ = τ/(κ m), κ = χ/ρ, and dz = −dm/ρ, so heights strictly decrease.
The rise of ln P is not limited: below a hot-over-cool interface the low opacity of the cool layer genuinely requires
a large column mass per unit τ. Steps are limited by the slope at both ends (Heun), so that rise is resolved rather than
averaged over a whole level. Returns true, or false if the pressure leaves the EOS table (or the EOS fails) in some
interval; then lnP and z are NaN from that level down and the column is reported as failed.
"""
function hse_tau!(lnP, z, lT, lt, tab, g; mu = 1.0)
    nt = length(lt)
    τ = exp.(LN10 .* lt)
    lp = log(1e2)
    for _ in 1:200                                                 # top: P = g τ ρ/χ (κ constant above the top level)
        q, χ = q_chi(tab, lT[1], lp)
        new = log(mu * g * τ[1] * q / χ) + lp
        abs(new - lp) < 1e-10 && break
        lp = 0.5 * lp + 0.5 * new
    end
    lnP[1] = lp
    lPmax = tab.Y[end] - 1.0
    rhs(lTs, lτ, l) = (s = q_chi(tab, lTs, l); (mu * g * exp(lτ) * s[1] / s[2], -mu * exp(lτ) / s[2]))
    z[1] = 0.0
    isfinite(lp) || (lnP .= NaN; z .= NaN; return false)
    for k in 1:nt-1
        a, b = LN10 * lt[k], LN10 * lt[k+1]
        l = lnP[k]; zz = z[k]; x = a
        while x < b
            sx = (x - a) / (b - a)
            f1, g1 = rhs(lT[k] + (lT[k+1] - lT[k]) * sx, x, l)
            isfinite(f1) || (@views lnP[k+1:end] .= NaN; z[k+1:end] .= NaN; z .= NaN; return false)
            # The step must also be small for the slope at its end: across a hot-over-cool interval the slope can
            # rise ~1000x within one 0.1 dex level (κ drops with T), and a step sized from f1 alone then averaged a
            # huge f2 over the whole level (Δ ln P ≈ 21, P × 1e9 in one column of a 504² MURaM run).
            h = min(b - x, 0.05 / max(abs(f1), 1e-30))
            local f2, g2
            while true
                sy = (x + h - a) / (b - a)
                f2, g2 = rhs(lT[k] + (lT[k+1] - lT[k]) * sy, x + h, l + h * f1)
                (isfinite(f2) && h * abs(f2) > 0.1 && h > 1e-9 * (b - a)) || break
                h /= 4
            end
            ln = l + h / 2 * (f1 + f2)
            if !isfinite(f2) || ln > lPmax                         # left the EOS table: the column fails
                @views lnP[k+1:end] .= NaN
                z .= NaN
                return false
            end
            l = ln; zz += h / 2 * (g1 + g2); x += h
        end
        lnP[k+1] = l; z[k+1] = zz
    end
    z .-= interp_lt(z, lt, 0.0)
    return true
end

"""Shift columns so that P + B²/2μ₀ is equal at one height (the global median bottom total pressure)."""
function align_ptot!(par, z, lnP, lT, Bx, By, Bz, tab, g)
    nt, ny, nx = size(z)
    lpt(k, j, i) = log(exp(lnP[k, j, i]) + (Bx[k, j, i]^2 + By[k, j, i]^2 + Bz[k, j, i]^2) / (2MU0))
    ref = gquantiles(par, [Float64[lpt(nt, j, i) for j in 1:ny, i in 1:nx][:]], 0.5)[1]
    nbelow = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            if lpt(nt, j, i) < ref                                 # reference below the column: hydrostatic extrapolation
                q, _ = q_chi(tab, lT[nt, j, i], lnP[nt, j, i])
                zr = z[nt, j, i] - (ref - lpt(nt, j, i)) / (g * q)
                Threads.atomic_add!(nbelow, 1)
            else
                k = nt
                while k > 1 && lpt(k - 1, j, i) >= ref
                    k -= 1
                end
                zr = k == 1 ? z[1, j, i] :
                     z[k-1, j, i] + (z[k, j, i] - z[k-1, j, i]) * (ref - lpt(k - 1, j, i)) / (lpt(k, j, i) - lpt(k - 1, j, i))
            end
            @views z[:, j, i] .-= zr
        end
    end
    return exp(ref), gcount(par, nbelow[])
end

"""Gauge: global mean z(τ = 1) = 0. Returns the shift subtracted."""
function gauge!(par, z, lt)
    nt, ny, nx = size(z)
    s = gsum(par, sum(interp_lt(@view(z[:, j, i]), lt, 0.0) for j in 1:ny, i in 1:nx)) / (ny * par.NX)
    z .-= s
    return s
end

"""
Column HSE in τ for every owned column, aligned and gauged; `mu` = cos θ of the line of sight (τ is τ_LOS). Returns (lnP, z, nfailed, Pref_total, nbelow, nbad):
nfailed is the global number of columns whose pressure left the EOS table, nbad the global number of non-finite ln P;
when nbad > 0 the alignment and gauge are skipped (they would spread the NaNs) and every rank returns the same values,
so callers decide collectively.
"""
function column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", progress = false, mu = 1.0)
    nt, ny, nx = size(lT)
    lnP = zeros(nt, ny, nx); z = zeros(nt, ny, nx)
    nfail = Threads.Atomic{Int}(0)
    bar = Progress(nx * ny; desc = "Column HSE, tau grid (this rank's columns) ", dt = 10, output = stdout,
                   enabled = progress)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            hse_tau!(@view(lnP[:, j, i]), @view(z[:, j, i]), @view(lT[:, j, i]), lt, tab, g; mu = mu) ||
                Threads.atomic_add!(nfail, 1)
            next!(bar)
        end
    end
    finish!(bar)
    nfailed = gcount(par, nfail[])
    nbad = gcount(par, count(!isfinite, lnP))
    nbad > 0 && return lnP, z, nfailed, NaN, 0, nbad
    Pt, nb = align == "ptot" ? align_ptot!(par, z, lnP, lT, Bx, By, Bz, tab, g) : (NaN, 0)
    gauge!(par, z, lt)
    return lnP, z, nfailed, Pt, nb, 0
end

"""
Read a τ-grid input (IO/input-mhs-ltau.nc layout: log10tau500 (nt), x (NX), y (NY), temperature, B_x, B_y, B_z as h5py
(x, y, τ); or the make_tau_input.py layout with logtau500, x_m, y_m and h5py (τ, y, x)); this rank's x slab, Julia (τ, y, x).
Only x and y honour a units attribute; temperature and the field are read as K and T without conversion.
"""
function read_tau_input(par, path; xname = "x", yname = "y")
    ir = owned_x(par)
    h5open(path, "r") do f
        if haskey(f, "log10tau500")
            for nm in (xname, yname)
                haskey(f, nm) || H.invalid("$path has no dataset \"$nm\" (τ-grid inputs carry x and y like the height-grid input)")
            end
            ax(nm) = (a = attributes(f[nm]); fac = haskey(a, "units") ? H.UNIT_FACTORS[:length][H.attr_string(read(a["units"]))] : 1.0;
                      Float64.(read(f[nm])) .* fac)
            rd(nm) = Float64.(f[nm][:, :, ir])
            lt = read(f["log10tau500"])
            x, y = ax(xname), ax(yname)
            (length(x), length(y)) == (size(f["temperature"], 3), size(f["temperature"], 2)) ||
                H.invalid("$path: x, y lengths $((length(x), length(y))) do not match temperature $(size(f["temperature"]))")
            return lt, x, y, rd("temperature"), rd("B_x"), rd("B_y"), rd("B_z")
        else
            rz(nm) = permutedims(Float64.(f[nm][ir, :, :]), (3, 2, 1))
            return read(f["logtau500"]), read(f["x_m"]), read(f["y_m"]), rz("T"), rz("Bx"), rz("By"), rz("Bz")
        end
    end
end

"""Global (NX, NY, nt) of a τ-grid input file."""
function tau_input_size(path)
    h5open(path, "r") do f
        if haskey(f, "log10tau500")
            s = size(f["temperature"]); return s[3], s[2], s[1]
        else
            s = size(f["T"]); return s[1], s[2], s[3]
        end
    end
end

# ======================================================================================================================
# direct Poisson solve on the τ grid
# ======================================================================================================================
mutable struct TP
    par::ParCtx; S::SlabFFT
    tab::H.EOSTable; g::Float64
    nt::Int; kc::Int; n1::Int; ny::Int; nx::Int     # n1 nodes k = kc … nt; slot 1 = c, slots 2…n1 = ln P at k > kc
    dx::Float64; dy::Float64
    lτ::Vector{Float64}; dl::Vector{Float64}
    lT::Array{Float64,3}; Bx::Array{Float64,3}; By::Array{Float64,3}; Bz::Array{Float64,3}
    zrel::Array{Float64,3}                          # z_k − z_kc of the fixed levels k ≤ kc
    z::Array{Float64,3}
    Lx::Array{Float64,3}; Ly::Array{Float64,3}; Lz::Array{Float64,3}
    a::Array{Float64,3}; aχ::Array{Float64,3}; q::Array{Float64,3}
    W::Array{Float64,3}; dw::Array{Float64,3}
    lx::Array{Float64,3}; ly::Array{Float64,3}
    Rx::Array{Float64,3}; Ry::Array{Float64,3}; Rz::Array{Float64,3}
    wc::Vector{Float64}; wf::Vector{Float64}; ereg::Vector{Float64}
    Q::Matrix{ComplexF64}; Qt::Matrix{ComplexF64}; λ::Vector{Float64}; s2::Vector{Float64}
    Yh::Matrix{ComplexF64}
    px::Array{Float64,3}; pXx::Array{Float64,3}; pYx::Array{Float64,3}
    py::Array{Float64,3}; pXy::Array{Float64,3}; pYy::Array{Float64,3}
    zζ::Array{Float64,3}; zX::Array{Float64,3}; zY::Array{Float64,3}; zden::Array{Float64,3}
    mu::Float64; tx::Float64; ty::Float64           # μ = cos θ; slant t (0, 0 unless inclined)
    vert::Bool                                      # inclined: vertical-balance column residual (D0)
    vq::Array{Float64,3}; vd::Array{Float64,3}      # per interval: t·R̄_h and Δz (vertical balance)
    prec::Symbol                                    # :fourier (layer-median modes), :block (exact column blocks, D3b) or
                                                    # :hybrid (column blocks + the Fourier modes with |k| ≤ klow, D3c)
    blk::Array{Float64,4}                           # :block, :hybrid: Cholesky factors L of the column blocks, (n1, n1, ny, nx)
    wk::Vector{Float64}                             # per T column: 1 for the modes the Fourier part acts on, else 0
    dlm::Array{Float64,3}                           # :block, :hybrid: diag(JᵀWJ), set by colblocks! (Marquardt damping)
    mlm::Float64                                    # Marquardt damping: the operator adds mlm · dlm
    fdamp::Bool                                     # :hybrid: the Fourier part includes mlm × the layer medians of dlm
end

inclined(pr::TP) = pr.tx != 0 || pr.ty != 0
vertical(pr::TP) = pr.vert && inclined(pr)

node(pr, m) = pr.kc + m - 1

"""∂f/∂z down each column (centred in level index, one-sided at the ends), z top first."""
function dzcol!(o, f, z)
    nt, ny, nx = size(f)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            o[1, j, i] = (f[1, j, i] - f[2, j, i]) / (z[1, j, i] - z[2, j, i])
            o[nt, j, i] = (f[nt-1, j, i] - f[nt, j, i]) / (z[nt-1, j, i] - z[nt, j, i])
            for k in 2:nt-1
                o[k, j, i] = (f[k-1, j, i] - f[k+1, j, i]) / (z[k-1, j, i] - z[k+1, j, i])
            end
        end
    end
    return o
end

"""
J×B on the τ levels at heights z: ∂_ray along the column, ∂/∂x|_z = ∂/∂x|_τ − ∂_x z|_τ ∂_ray, and
∂/∂z|_(x,y) = ∂_ray − t·∇_h|_z (= ∂_ray for vertical columns). Returns the ray derivatives of Bx, By, Bz and ∂z/∂x|_τ,
∂z/∂y|_τ for the Jacobian.
"""
function lorentz_tau!(pr::TP)
    z = pr.z; par = pr.par; tx = pr.tx; ty = pr.ty; sl = inclined(pr)
    zx = sddx!(similar(z), z, pr.dx, par); zy = sddy!(similar(z), z, pr.dy)
    function d(B)
        fz = dzcol!(similar(B), B, z); fx = sddx!(similar(B), B, pr.dx, par); fy = sddy!(similar(B), B, pr.dy)
        @. fx -= zx * fz; @. fy -= zy * fz
        return fx, fy, sl ? (@. fz - tx * fx - ty * fy) : fz, fz
    end
    (Bxx, Bxy, Bxv, Bxz), (Byx, Byy, Byv, Byz), (Bzx, Bzy, Bzv, Bzz) = d(pr.Bx), d(pr.By), d(pr.Bz)
    Bx, By, Bz = pr.Bx, pr.By, pr.Bz
    Threads.@threads for idx in eachindex(z)
        @inbounds begin
            Jx = (Bzy[idx] - Byv[idx]) / MU0; Jy = (Bxv[idx] - Bzx[idx]) / MU0; Jz = (Byx[idx] - Bxy[idx]) / MU0
            pr.Lx[idx] = Jy * Bz[idx] - Jz * By[idx]; pr.Ly[idx] = Jz * Bx[idx] - Jx * Bz[idx]
            pr.Lz[idx] = Jx * By[idx] - Jy * Bx[idx]
        end
    end
    return Bxz, Byz, Bzz, zx, zy
end

"""State, residuals and Jacobian coefficients at (u, c); u holds ln P of all levels (k ≤ kc fixed). Returns (Φ, bad):
bad = the global number of cells whose state or opacity-derivative probes (ln P ± EPS_PROBE) are outside the table
(Φ = Inf then); `eos_cells` separates the two."""
function evaluate!(pr::TP, u, c)
    nt, kc, n1, ny, nx = pr.nt, pr.kc, pr.n1, pr.ny, pr.nx
    bad = Threads.Atomic{Int}(0); ε = EPS_PROBE; g = pr.g
    ρu = similar(pr.a)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1
                k = node(pr, m); lT = pr.lT[k, j, i]; lp = u[k, j, i]
                r, dr, _ = H.lnrho_dY(pr.tab, lT, lp)
                _, _, c0, _ = H.state(pr.tab, lT, lp)
                _, _, cp, _ = H.state(pr.tab, lT, lp + ε); _, _, cm, _ = H.state(pr.tab, lT, lp - ε)   # ε = EPS_PROBE
                isfinite(r + c0 + cp + cm) || (Threads.atomic_add!(bad, 1); r = 0.0; dr = 1.0; c0 = cp = cm = 0.0)
                pr.q[m, j, i] = exp(r - lp); ρu[m, j, i] = dr
                a = pr.mu * exp(pr.lτ[k] - c0); pr.a[m, j, i] = a; pr.aχ[m, j, i] = a * (cp - cm) / (2ε)
            end
            pr.z[kc, j, i] = c[j, i]
            for k in 1:kc-1
                pr.z[k, j, i] = c[j, i] + pr.zrel[k, j, i]
            end
            for m in 1:n1-1
                k = node(pr, m)
                pr.z[k+1, j, i] = pr.z[k, j, i] - pr.dl[k] / 2 * (pr.a[m, j, i] + pr.a[m+1, j, i])
            end
        end
    end
    nbad = gcount(pr.par, bad[]); nbad > 0 && return Inf, nbad
    Bxz, Byz, Bzz, zxf, zyf = lorentz_tau!(pr)
    tx = pr.tx; ty = pr.ty; sl = inclined(pr)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, m in 1:n1
            k = node(pr, m); e = exp(-u[k, j, i]); bx = pr.Bx[k, j, i]; by = pr.By[k, j, i]; bz = pr.Bz[k, j, i]
            xz = Bxz[k, j, i]; yz = Byz[k, j, i]; zz = Bzz[k, j, i]; sx = zxf[k, j, i]; sy = zyf[k, j, i]
            cr(a1, a2, a3) = (a2 * bz - a3 * by, a3 * bx - a1 * bz, a1 * by - a2 * bx)
            # δL per unit ζ (relative change of the column spacing), X = ∂_xδz, Y = ∂_yδz: δJ from δ(∂_x, ∂_y, ∂_z)B
            # (inclined: ∂_z = (1 + t·∇z) ∂_ray − t·∂_h|_τ); the column balance uses the ray component n(L) = L_z + t·L_h
            if sl
                s = 1 + tx * sx + ty * sy
                Lζ = cr((sy * zz + s * yz) / MU0, (-s * xz - sx * zz) / MU0, (sx * yz - sy * xz) / MU0)
                LX = cr(-tx * yz / MU0, (tx * xz + zz) / MU0, -yz / MU0)
                LY = cr((-zz - ty * yz) / MU0, ty * xz / MU0, xz / MU0)
                nζ = Lζ[3] + tx * Lζ[1] + ty * Lζ[2]; nX = LX[3] + tx * LX[1] + ty * LX[2]; nY = LY[3] + tx * LY[1] + ty * LY[2]
            else
                Lζ = cr((sy * zz + yz) / MU0, (-xz - sx * zz) / MU0, (sx * yz - sy * xz) / MU0)
                LX = cr(0.0, zz / MU0, -yz / MU0); LY = cr(-zz / MU0, 0.0, xz / MU0)
                nζ = Lζ[3]; nX = LX[3]; nY = LY[3]
            end
            pr.px[m, j, i] = e * (Lζ[1] + sx * nζ); pr.pXx[m, j, i] = e * (LX[1] + sx * nX); pr.pYx[m, j, i] = e * (LY[1] + sx * nY)
            pr.py[m, j, i] = e * (Lζ[2] + sy * nζ); pr.pXy[m, j, i] = e * (LX[2] + sy * nX); pr.pYy[m, j, i] = e * (LY[2] + sy * nY)
            ea = e * pr.a[m, j, i]
            pr.zζ[m, j, i] = ea * nζ; pr.zX[m, j, i] = ea * nX; pr.zY[m, j, i] = ea * nY
            pr.zden[m, j, i] = k < nt ? pr.z[k-1, j, i] - pr.z[k+1, j, i] : pr.z[nt-1, j, i] - pr.z[nt, j, i]
        end
    end
    un = u[kc:nt, :, :]; zn = pr.z[kc:nt, :, :]
    ux = sddx!(similar(un), un, pr.dx, pr.par); uy = sddy!(similar(un), un, pr.dy)
    zx = sddx!(similar(zn), zn, pr.dx, pr.par); zy = sddy!(similar(zn), zn, pr.dy)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1
                k = node(pr, m); e = exp(-u[k, j, i]); q = pr.q[m, j, i]; a = pr.a[m, j, i]
                Ln = sl ? pr.Lz[k, j, i] + tx * pr.Lx[k, j, i] + ty * pr.Ly[k, j, i] : pr.Lz[k, j, i]
                W = g * q - Ln * e
                dW = g * q * (ρu[m, j, i] - 1) + Ln * e
                pr.W[m, j, i] = W; pr.dw[m, j, i] = dW * a - W * pr.aχ[m, j, i]
                pr.Rx[m, j, i] = -ux[m, j, i] - W * zx[m, j, i] + pr.Lx[k, j, i] * e
                pr.Ry[m, j, i] = -uy[m, j, i] - W * zy[m, j, i] + pr.Ly[k, j, i] * e
                pr.lx[m, j, i] = -zx[m, j, i] * dW - pr.Lx[k, j, i] * e
                pr.ly[m, j, i] = -zy[m, j, i] * dW - pr.Ly[k, j, i] * e
            end
            for f in 1:n1-1
                k = node(pr, f)
                pr.Rz[f, j, i] = u[k+1, j, i] - u[k, j, i] -
                                 pr.dl[k] / 2 * (pr.W[f, j, i] * pr.a[f, j, i] + pr.W[f+1, j, i] * pr.a[f+1, j, i])
            end
            if vertical(pr)
                for f in 1:n1-1
                    k = node(pr, f)
                    q = 0.5 * (tx * (pr.Rx[f, j, i] + pr.Rx[f+1, j, i]) + ty * (pr.Ry[f, j, i] + pr.Ry[f+1, j, i]))
                    d = pr.z[k, j, i] - pr.z[k+1, j, i]
                    pr.vq[f, j, i] = q; pr.vd[f, j, i] = d
                    pr.Rz[f, j, i] -= d * q
                end
            end
        end
    end
    return phi(pr), 0
end

"""Horizontal (Σ wc (Rx² + Ry²)) and vertical (Σ wf Rz²) parts of Φ at the evaluated state, and the part of the vertical
one on the first `nface` faces below the cut (where a start's pressure jump at the cut shows), global."""
function phi_parts(pr::TP; nface = 3)
    h = 0.0; v = 0.0; vt = 0.0
    @inbounds for i in 1:pr.nx, j in 1:pr.ny
        for m in 1:pr.n1
            h += pr.wc[m] * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2)
        end
        for f in 1:pr.n1-1
            r = pr.wf[f] * pr.Rz[f, j, i]^2
            v += r; f <= nface && (vt += r)
        end
    end
    return gsum(pr.par, h), gsum(pr.par, v), gsum(pr.par, vt)
end

"""
Residual of the evaluated state in fixed physical units, for refinement studies (the weights of Φ are not resolution
invariant: wf ∝ H²/Δz): the horizontal residuals R_x, R_y (per unit length) at the nodes and the vertical R_z/Δz (R_z is
integrated over its face) at the faces, each made dimensionless with `Href` (default 100 km, held fixed across grids)
and averaged with the volume weights (node thickness (z_m−1 − z_m+1)/2, face thickness Δz) over the first `nlayer`
levels below the cut (`cut`), the bottom `nlayer` (`bottom`), the rest (`interior`) and all (`all`): rms values,
global. Returns (horizontal = (all, cut, interior, bottom), vertical = (...)).
"""
function defect_parts(pr::TP; Href = 1e5, nlayer = 3)
    n1 = pr.n1; kc = pr.kc
    acc = zeros(2, 4, 2)                                   # (horizontal / vertical, layer, Σ w r² / Σ w)
    layer(m, n) = m <= nlayer ? 2 : m > n - nlayer ? 4 : 3
    @inbounds for i in 1:pr.nx, j in 1:pr.ny
        for m in 1:n1
            k = node(pr, m)
            w = m == 1 ? (pr.z[k, j, i] - pr.z[k+1, j, i]) / 2 : m == n1 ? (pr.z[k-1, j, i] - pr.z[k, j, i]) / 2 :
                (pr.z[k-1, j, i] - pr.z[k+1, j, i]) / 2
            r2 = Href^2 * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2) / 2
            for l in (1, layer(m, n1)); acc[1, l, 1] += w * r2; acc[1, l, 2] += w; end
        end
        for f in 1:n1-1
            k = node(pr, f); d = pr.z[k, j, i] - pr.z[k+1, j, i]
            r2 = Href^2 * (pr.Rz[f, j, i] / d)^2
            for l in (1, layer(f, n1 - 1)); acc[2, l, 1] += d * r2; acc[2, l, 2] += d; end
        end
    end
    a = reshape(gsum(pr.par, vec(acc)), 2, 4, 2)
    r(c) = Tuple(sqrt(a[c, l, 1] / a[c, l, 2]) for l in 1:4)
    nt(t) = (all = t[1], cut = t[2], interior = t[3], bottom = t[4])
    return (horizontal = nt(r(1)), vertical = nt(r(2)))
end

const ROUNDOFF_K = 64.0
"""`pcg!` outcomes that are not a tolerance reached or an iteration budget spent (see `pcg!`)."""
const PCG_ABNORMAL = (:nonfinite_start, :breakdown_rz, :breakdown_pAp, :nonfinite_step, :nonfinite_update)

"""
Round-off floor of Φ at the evaluated state of `pr` (state u, heights pr.z): each residual with its terms replaced by
their magnitudes, times K·eps (K = ROUNDOFF_K; differences of neighbours estimated by the local value):

    R_x: |u|/dx + |W| |z|/dx + |L_x e|,   R_y: the same with dy,
    R_z: |u_k| + |u_k+1| + Δln τ/2 (|W a|_k + |W a|_k+1) (+ |Δz| (|t_x| s_x + |t_y| s_y)/2 in the vertical balance),

summed with the weights of Φ. Φ ≤ this floor means the residuals are at the precision of the arithmetic that forms them
(the state is a discrete equilibrium to round-off); the comparison uses Φ's own weights, so it does not depend on the
grid. Global.
"""
function phi_floor(pr::TP, u)
    ε = ROUNDOFF_K * eps(Float64); tx = abs(pr.tx); ty = abs(pr.ty); vt = vertical(pr)
    s = zeros(pr.nx)
    Threads.@threads for i in 1:pr.nx
        acc = 0.0
        @inbounds for j in 1:pr.ny
            for m in 1:pr.n1
                k = node(pr, m); e = exp(-u[k, j, i]); au = abs(u[k, j, i]); az = abs(pr.z[k, j, i]); aW = abs(pr.W[m, j, i])
                sx = au / pr.dx + aW * az / pr.dx + abs(pr.Lx[k, j, i] * e)
                sy = au / pr.dy + aW * az / pr.dy + abs(pr.Ly[k, j, i] * e)
                acc += pr.wc[m] * ε^2 * (sx^2 + sy^2)
            end
            for f in 1:pr.n1-1
                k = node(pr, f)
                sz = abs(u[k, j, i]) + abs(u[k+1, j, i]) +
                     pr.dl[k] / 2 * (abs(pr.W[f, j, i] * pr.a[f, j, i]) + abs(pr.W[f+1, j, i] * pr.a[f+1, j, i]))
                if vt
                    au = abs(u[k, j, i]); az = abs(pr.z[k, j, i]); aW = abs(pr.W[f, j, i])
                    sz += abs(pr.vd[f, j, i]) * (tx * (au / pr.dx + aW * az / pr.dx) + ty * (au / pr.dy + aW * az / pr.dy))
                end
                acc += pr.wf[f] * (ε * sz)^2
            end
        end
        s[i] = acc
    end
    return gsum(pr.par, sum(s))
end

function phi(pr::TP)
    s = zeros(pr.nx)
    Threads.@threads for i in 1:pr.nx
        acc = 0.0
        @inbounds for j in 1:pr.ny
            for m in 1:pr.n1
                acc += pr.wc[m] * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2)
            end
            for f in 1:pr.n1-1
                acc += pr.wf[f] * pr.Rz[f, j, i]^2
            end
        end
        s[i] = acc
    end
    return gsum(pr.par, sum(s))
end

"""(jx, jy, jz) = J v; v slot 1 = δc, slots 2… = δ ln P."""
function Jop!(pr::TP, jx, jy, jz, v)
    n1, ny, nx = pr.n1, pr.ny, pr.nx
    du = copy(v); du[1, :, :] .= 0.0
    dz = similar(v)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            dz[1, j, i] = v[1, j, i]
            for f in 1:n1-1
                h = pr.dl[node(pr, f)] / 2
                dz[f+1, j, i] = dz[f, j, i] + h * (pr.aχ[f, j, i] * du[f, j, i] + pr.aχ[f+1, j, i] * du[f+1, j, i])
                jz[f, j, i] = du[f+1, j, i] - du[f, j, i] - h * (pr.dw[f, j, i] * du[f, j, i] + pr.dw[f+1, j, i] * du[f+1, j, i])
            end
            jz[n1, j, i] = 0.0
        end
    end
    t = similar(v)
    sddx!(jx, du, pr.dx, pr.par); sddx!(t, dz, pr.dx, pr.par)
    Threads.@threads for idx in eachindex(jx)
        @inbounds jx[idx] = -jx[idx] - pr.W[idx] * t[idx] + pr.lx[idx] * du[idx]
    end
    sddy!(jy, du, pr.dy); t2 = sddy!(similar(v), dz, pr.dy)
    Threads.@threads for idx in eachindex(jy)
        @inbounds jy[idx] = -jy[idx] - pr.W[idx] * t2[idx] + pr.ly[idx] * du[idx]
    end
    # Lorentz force through the heights: t = ∂_xδz, t2 = ∂_yδz, ζ from the vertical neighbours
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            Tp = 0.0
            for m in 1:n1
                zm = m == 1 ? dz[1, j, i] : dz[m-1, j, i]          # δz above the cut = δc
                ζ = m < n1 ? (zm - dz[m+1, j, i]) / pr.zden[m, j, i] : (dz[n1-1, j, i] - dz[n1, j, i]) / pr.zden[m, j, i]
                X = t[m, j, i]; Y = t2[m, j, i]
                jx[m, j, i] += pr.px[m, j, i] * ζ + pr.pXx[m, j, i] * X + pr.pYx[m, j, i] * Y
                jy[m, j, i] += pr.py[m, j, i] * ζ + pr.pXy[m, j, i] * X + pr.pYy[m, j, i] * Y
                T = pr.zζ[m, j, i] * ζ + pr.zX[m, j, i] * X + pr.zY[m, j, i] * Y
                m >= 2 && (jz[m-1, j, i] += pr.dl[node(pr, m - 1)] / 2 * (Tp + T))
                Tp = T
            end
        end
    end
    if vertical(pr)                                    # δR_v = δR_ray − δΔz t·R̄_h − Δz t·δR̄_h
        tx = pr.tx; ty = pr.ty
        Threads.@threads for i in 1:nx
            @inbounds for j in 1:ny, f in 1:n1-1
                jz[f, j, i] -= (dz[f, j, i] - dz[f+1, j, i]) * pr.vq[f, j, i] +
                               pr.vd[f, j, i] * 0.5 * (tx * (jx[f, j, i] + jx[f+1, j, i]) + ty * (jy[f, j, i] + jy[f+1, j, i]))
            end
        end
    end
end

"""Horizontal weights of JᵀR with the R̄_h part of the vertical-balance residual added (inclined, D0)."""
function vertical_weights(pr::TP, rx0, ry0, rz)
    rx = copy(rx0); ry = copy(ry0)
    Threads.@threads for i in 1:pr.nx
        @inbounds for j in 1:pr.ny, f in 1:pr.n1-1
            h = -0.5 * pr.vd[f, j, i] * rz[f, j, i]
            rx[f, j, i] += h * pr.tx; rx[f+1, j, i] += h * pr.tx
            ry[f, j, i] += h * pr.ty; ry[f+1, j, i] += h * pr.ty
        end
    end
    return rx, ry
end

"""out = Jᵀ (rx, ry, rz)."""
function JTop!(pr::TP, out, rx0, ry0, rz)
    n1, ny, nx = pr.n1, pr.ny, pr.nx; par = pr.par
    rx, ry = vertical(pr) ? vertical_weights(pr, rx0, ry0, rz) : (rx0, ry0)   # one assignment: no boxed captures
    a = similar(out); b = similar(out)
    sddx!(out, rx, pr.dx, par); sddy!(a, ry, pr.dy)            # (−D)ᵀ = D
    @. out += a + pr.lx * rx + pr.ly * ry
    @. a = pr.W * rx; @. b = pr.W * ry
    t = similar(out); sddx!(t, a, pr.dx, par); sddy!(a, b, pr.dy); t .+= a   # gradient with respect to δz on the nodes
    gX = similar(out); gY = similar(out); gζ = similar(out)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny, m in 1:n1
            sm = (m >= 2 ? pr.dl[node(pr, m - 1)] / 2 * rz[m-1, j, i] : 0.0) + (m < n1 ? pr.dl[node(pr, m)] / 2 * rz[m, j, i] : 0.0)
            X = rx[m, j, i]; Y = ry[m, j, i]
            gζ[m, j, i] = (pr.px[m, j, i] * X + pr.py[m, j, i] * Y + pr.zζ[m, j, i] * sm) / pr.zden[m, j, i]
            gX[m, j, i] = pr.pXx[m, j, i] * X + pr.pXy[m, j, i] * Y + pr.zX[m, j, i] * sm
            gY[m, j, i] = pr.pYx[m, j, i] * X + pr.pYy[m, j, i] * Y + pr.zY[m, j, i] * sm
        end
    end
    sddx!(a, gX, pr.dx, par); t .-= a; sddy!(a, gY, pr.dy); t .-= a     # Dᵀ = −D
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for m in 1:n1-1
                gg = gζ[m, j, i]
                t[m == 1 ? 1 : m - 1, j, i] += gg; t[m+1, j, i] -= gg
            end
            gg = gζ[n1, j, i]; t[n1-1, j, i] += gg; t[n1, j, i] -= gg
            if vertical(pr)                                # the δΔz part of R_v
                for f in 1:n1-1
                    h = pr.vq[f, j, i] * rz[f, j, i]
                    t[f, j, i] -= h; t[f+1, j, i] += h
                end
            end
        end
    end
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            G = 0.0; tc = t[1, j, i]
            for f in n1-1:-1:1
                h = pr.dl[node(pr, f)] / 2
                r = rz[f, j, i]
                out[f+1, j, i] += r * (1 - h * pr.dw[f+1, j, i])
                out[f, j, i] += r * (-1 - h * pr.dw[f, j, i])
                G += t[f+1, j, i]; tc += t[f+1, j, i]
                out[f, j, i] += h * pr.aχ[f, j, i] * G
                out[f+1, j, i] += h * pr.aχ[f+1, j, i] * G
            end
            out[1, j, i] = tc                                  # slot 1 is δc (ln P at k_c is fixed)
        end
    end
    return out
end

"""Σ w (Jδ)² of Jop! output (the weights of Φ)."""
function wnorm2(pr::TP, jx, jy, jz)
    acc = zeros(size(jx, 3))
    Threads.@threads for i in axes(jx, 3)
        a = 0.0
        @inbounds for j in axes(jx, 2), m in 1:pr.n1
            a += pr.wc[m] * (jx[m, j, i]^2 + jy[m, j, i]^2)
            m < pr.n1 && (a += pr.wf[m] * jz[m, j, i]^2)
        end
        acc[i] = a
    end
    return gsum(pr.par, sum(acc))
end

function Aop!(pr::TP, out, v)
    jx = similar(v); jy = similar(v); jz = similar(v)
    Jop!(pr, jx, jy, jz, v)
    jx .*= pr.wc; jy .*= pr.wc; jz .*= vcat(pr.wf, 0.0)
    JTop!(pr, out, jx, jy, jz)
    @. out += pr.ereg * v
    pr.mlm > 0 && (@. out += pr.mlm * pr.dlm * v)
    return out
end

"""Preconditioner at the current state and regularisation of `pr`: :fourier, the layer-median operator A + s² B per
horizontal mode, diagonalised (M⁻¹ = Qᵀ diag(1/(1 + s²λ)) Q); :block, the exact column blocks (block Jacobi);
:hybrid, both added, the Fourier part on the modes |k| ≤ klow only (the large scales, which block Jacobi cannot
propagate and which barely show in the residual: design/tau/SOLVER_TIME_DESIGN.md D3c). With Marquardt damping
(mlm > 0) the Fourier part adds mlm × the layer medians of diag(JᵀWJ) unless `pr.fdamp` is false."""
function setprec!(pr::TP)
    pr.prec == :fourier || colblocks!(pr)
    pr.prec == :block && return pr
    n1 = pr.n1; par = pr.par
    aχ = gmedians(par, pr.aχ); dw = gmedians(par, pr.dw); W = gmedians(par, pr.W)
    Gz = zeros(n1 - 1, n1)
    for f in 1:n1-1
        h = pr.dl[node(pr, f)] / 2
        Gz[f, f+1] = 1 - h * dw[f+1]
        f >= 2 && (Gz[f, f] = -1 - h * dw[f])
    end
    S = zeros(n1, n1)                                          # δz_m = δc + Σ S[m, m'] δu_m'
    for m in 1:n1
        S[m, 1] = 1.0
        for f in 1:m-1
            h = pr.dl[node(pr, f)] / 2
            f >= 2 && (S[m, f] += h * aχ[f])
            S[m, f+1] += h * aχ[f+1]
        end
    end
    Gh = -(W .* S); for m in 2:n1; Gh[m, m] -= 1; end
    d = pr.fdamp && pr.mlm > 0 ? pr.ereg .+ pr.mlm .* gmedians(par, pr.dlm) : pr.ereg   # the damping of Aop!, layer medians
    A = Gz' * Diagonal(pr.wf) * Gz + Diagonal(d)
    B = Gh' * Diagonal(pr.wc) * Gh
    Li = inv(Matrix(cholesky(Symmetric(A)).L))
    E = eigen(Symmetric(Li * B * Li'))
    Q = E.vectors' * Li
    pr.Q = ComplexF64.(Q); pr.Qt = ComplexF64.(Matrix(Q')); pr.λ = max.(E.values, 0.0)
end

function Minv!(pr::TP, out, r)
    pr.prec == :block && return block_solve!(out, pr.blk, r)
    if pr.prec == :hybrid
        b = block_solve!(similar(out), pr.blk, r)
        fourier_minv!(pr, out, r)
        out .+= b
        return out
    end
    return fourier_minv!(pr, out, r)
end

function fourier_minv!(pr::TP, out, r)
    n1 = pr.n1
    T = fwd!(pr.S, pr.par, r)
    X = reshape(T, n1, :)
    size(pr.Yh) == size(X) || (pr.Yh = similar(X))
    mul!(pr.Yh, pr.Q, X)
    Threads.@threads for col in axes(pr.Yh, 2)
        s = pr.s2[col]; w = pr.wk[col]
        @inbounds for m in 1:n1
            pr.Yh[m, col] *= w / (1 + s * pr.λ[m])
        end
    end
    mul!(X, pr.Qt, pr.Yh)
    bwd!(pr.S, pr.par, out)
    return out
end

"""
    colblocks!(pr)

Block-Jacobi preconditioner (design/tau/SOLVER_TIME_DESIGN.md D3b): the diagonal blocks Σ_R J_RPᵀ W J_RP + diag(ereg)
of JᵀWJ + ereg, one n₁ × n₁ block per column P, assembled from the Jacobian coefficients of the evaluated state and
Cholesky-factored into pr.blk. The rows R that depend on column P are its own and, through the centred differences
(no self term), those of its four horizontal neighbours: ∂_x of ln P and of the heights, X = ∂_xδz and Y = ∂_yδz in the
Lorentz terms and, in the vertical balance, R̄_h. Needs NX, ny ≥ 3 (distinct neighbours; tp_setup falls back to
:fourier). The field-dependent terms that stall the Fourier preconditioner (the slant, B gradients) are exact here.
"""
function colblocks!(pr::TP)
    n1, ny, nx = pr.n1, pr.ny, pr.nx; par = pr.par
    size(pr.blk) == (n1, n1, ny, nx) || (pr.blk = zeros(n1, n1, ny, nx))
    size(pr.dlm) == (n1, ny, nx) || (pr.dlm = zeros(n1, ny, nx))
    vt = vertical(pr); tx = pr.tx; ty = pr.ty
    Wh = xhalo(par, pr.W); Xxh = xhalo(par, pr.pXx); Xyh = xhalo(par, pr.pXy); zXh = xhalo(par, pr.zX); vdh = xhalo(par, pr.vd)
    hf = [pr.dl[node(pr, f)] / 2 for f in 1:n1-1]
    nfail = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:nx
        Zm = zeros(n1, n1); Zt = zeros(n1, n1)                  # δz = Zm v; ζ = Zt v
        Jx = zeros(n1, n1); Jy = zeros(n1, n1); Jz = zeros(n1, n1 - 1); T = zeros(n1, n1)   # rows stored as columns
        cW = zeros(n1); c1 = zeros(n1); c2 = zeros(n1); cT = zeros(n1); cV = zeros(n1)        # neighbour coefficients
        @inbounds for j in 1:ny
            B = @view pr.blk[:, :, j, i]; fill!(B, 0.0)
            fill!(Zm, 0.0); Zm[1, 1] = 1.0                      # Zm[:, m] = row m of the δz map (transposed)
            for f in 1:n1-1
                for l in 1:f; Zm[l, f+1] = Zm[l, f]; end
                f >= 2 && (Zm[f, f+1] += hf[f] * pr.aχ[f, j, i])
                Zm[f+1, f+1] += hf[f] * pr.aχ[f+1, j, i]
            end
            for m in 1:n1
                a, b = m == n1 ? (n1 - 1, n1) : (m == 1 ? 1 : m - 1, m + 1)
                d = pr.zden[m, j, i]
                for l in 1:n1; Zt[l, m] = (Zm[l, a] - Zm[l, b]) / d; end
            end
            # rows of column P itself
            for m in 1:n1, l in 1:n1
                Jx[l, m] = pr.px[m, j, i] * Zt[l, m]; Jy[l, m] = pr.py[m, j, i] * Zt[l, m]; T[l, m] = pr.zζ[m, j, i] * Zt[l, m]
            end
            for m in 2:n1; Jx[m, m] += pr.lx[m, j, i]; Jy[m, m] += pr.ly[m, j, i]; end
            for f in 1:n1-1
                for l in 1:n1; Jz[l, f] = hf[f] * (T[l, f] + T[l, f+1]); end
                f >= 2 && (Jz[f, f] += -1 - hf[f] * pr.dw[f, j, i])
                Jz[f+1, f] += 1 - hf[f] * pr.dw[f+1, j, i]
                if vt
                    q = pr.vq[f, j, i]; vd = 0.5 * pr.vd[f, j, i]
                    for l in 1:n1
                        Jz[l, f] -= (Zm[l, f] - Zm[l, f+1]) * q + vd * (tx * (Jx[l, f] + Jx[l, f+1]) + ty * (Jy[l, f] + Jy[l, f+1]))
                    end
                end
            end
            addgram!(B, Jx, Jy, Jz, pr.wc, pr.wf)
            # rows of the four neighbours R: their centred differences take column P with weight s. Along x,
            # δR_x = −∂_xδu + (pXx − W) X + pXy-part in R_y, T from zX; along y the same with the roles of x and y swapped
            for nb in 1:4
                if nb <= 2                                      # nb 1: R = P + x̂ (P is R's left neighbour)
                    s = (nb == 1 ? -1 : 1) / (2pr.dx); ii = nb == 1 ? i + 1 : i - 1
                    for m in 1:n1
                        if 1 <= ii <= nx
                            cW[m] = pr.W[m, j, ii]; c1[m] = pr.pXx[m, j, ii]; c2[m] = pr.pXy[m, j, ii]
                            cT[m] = pr.zX[m, j, ii]; cV[m] = pr.vd[m, j, ii]
                        else
                            h = ii > nx ? 2 : 1
                            cW[m] = Wh[h][m, j]; c1[m] = Xxh[h][m, j]; c2[m] = Xyh[h][m, j]; cT[m] = zXh[h][m, j]; cV[m] = vdh[h][m, j]
                        end
                    end
                    Ja, Jb = Jx, Jy                             # Ja: the component whose derivative is along the step
                else
                    s = (nb == 3 ? -1 : 1) / (2pr.dy); jj = nb == 3 ? (j == ny ? 1 : j + 1) : (j == 1 ? ny : j - 1)
                    for m in 1:n1
                        cW[m] = pr.W[m, jj, i]; c1[m] = pr.pYy[m, jj, i]; c2[m] = pr.pYx[m, jj, i]
                        cT[m] = pr.zY[m, jj, i]; cV[m] = pr.vd[m, jj, i]
                    end
                    Ja, Jb = Jy, Jx
                end
                for m in 1:n1
                    ca = s * (c1[m] - cW[m]); cb = s * c2[m]; ct = s * cT[m]
                    for l in 1:n1; Ja[l, m] = ca * Zm[l, m]; Jb[l, m] = cb * Zm[l, m]; T[l, m] = ct * Zm[l, m]; end
                    m >= 2 && (Ja[m, m] -= s)
                end
                for f in 1:n1-1
                    vd = vt ? 0.5 * cV[f] : 0.0
                    for l in 1:n1
                        Jz[l, f] = hf[f] * (T[l, f] + T[l, f+1]) - vd * (tx * (Jx[l, f] + Jx[l, f+1]) + ty * (Jy[l, f] + Jy[l, f+1]))
                    end
                end
                addgram!(B, Jx, Jy, Jz, pr.wc, pr.wf)
            end
            for m in 1:n1
                pr.dlm[m, j, i] = B[m, m]
                B[m, m] += pr.ereg[m] + pr.mlm * B[m, m]
            end
            chol_lower!(B) || (Threads.atomic_add!(nfail, 1))
        end
    end
    gcount(par, nfail[]) == 0 || error("colblocks!: a column block is not positive definite")
    return pr
end

"""B += Jxᵀ diag(wc) Jx + Jyᵀ diag(wc) Jy + Jzᵀ diag(wf) Jz (lower triangle; J rows stored as the columns of Jx, Jy, Jz)."""
function addgram!(B, Jx, Jy, Jz, wc, wf)
    n1 = size(B, 1)
    @inbounds for (J, w) in ((Jx, wc), (Jy, wc), (Jz, wf)), r in 1:size(J, 2)
        wr = w[r]
        for b in 1:n1
            t = wr * J[b, r]
            t == 0.0 && continue
            @simd for a in b:n1
                B[a, b] += t * J[a, r]
            end
        end
    end
    return B
end

"""In-place Cholesky of the lower triangle of B (B = L Lᵀ, L in the lower triangle); false if not positive definite."""
function chol_lower!(B)
    n = size(B, 1)
    @inbounds for k in 1:n
        d = B[k, k]
        d > 0 || return false
        d = sqrt(d); B[k, k] = d
        for a in k+1:n; B[a, k] /= d; end
        for b in k+1:n
            t = B[b, k]
            @simd for a in b:n
                B[a, b] -= t * B[a, k]
            end
        end
    end
    return true
end

"""out = (L Lᵀ)⁻¹ r per column, L the factors of `colblocks!`."""
function block_solve!(out, F, r)
    n1 = size(F, 1)
    Threads.@threads for i in axes(r, 3)
        @inbounds for j in axes(r, 2)
            L = @view F[:, :, j, i]
            for m in 1:n1; out[m, j, i] = r[m, j, i]; end
            for k in 1:n1                                       # L y = r
                y = out[k, j, i] / L[k, k]; out[k, j, i] = y
                for a in k+1:n1; out[a, j, i] -= L[a, k] * y; end
            end
            for k in n1:-1:1                                    # Lᵀ x = y
                t = out[k, j, i]
                for a in k+1:n1; t -= L[a, k] * out[a, j, i]; end
                out[k, j, i] = t / L[k, k]
            end
        end
    end
    return out
end

"""Residual summary by log10 τ band: RMS H̄|R_h| (horizontal) and H̄ R_z/Δz̄ (vertical), ≈ |F|/ρg."""
function bands(pr::TP, lt, Hb, dzb)
    out = String[]; N = pr.ny * pr.par.NX
    for (lo, hi) in ((-9.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 9.0))
        ms = [m for m in 1:pr.n1 if lo <= lt[node(pr, m)] < hi]
        isempty(ms) && continue
        h = sqrt(gsum(pr.par, sum(Hb[m]^2 * (pr.Rx[m, j, i]^2 + pr.Ry[m, j, i]^2) for m in ms, j in 1:pr.ny, i in 1:pr.nx)) / (length(ms) * N))
        fs = [f for f in ms if f < pr.n1]
        v = isempty(fs) ? NaN : sqrt(gsum(pr.par, sum((Hb[f] * pr.Rz[f, j, i] / dzb[f])^2 for f in fs, j in 1:pr.ny, i in 1:pr.nx)) / (length(fs) * N))
        push!(out, @sprintf("log τ [%.0f, %.0f): horiz %.3f, vert %.3f", lo, hi, h, v))
    end
    return join(out, "; ")
end

"""
    warm_start!(u, z0, kc, uprev, cprev[, ucol, zcol])

Start of a solve from a previous solution (design/tau/SOLVER_TIME_DESIGN.md D4): `u`, `z0` are the new column HSE
(levels above the cut and ln P at the fixed cut level kc are kept); `uprev` = ln P on levels kc … nt and `cprev` the
cut heights of the previous solution. Without `ucol`, ln P below the cut is the previous one shifted per column by the
change of ln P at the cut level, and the heights above the cut are moved as a block onto the previous cut height (D4a).
With the previous column HSE (`ucol` = its ln P on levels kc … nt, `zcol` its cut heights; D4b), the previous solution
is corrected by the column-HSE response to the change of T: u = uprev + (u − ucol) level by level, and the cut height
by the change of the column-HSE cut height.
"""
function warm_start!(u, z0, kc, uprev, cprev, ucol = nothing, zcol = nothing)
    nt = size(u, 1)
    @inbounds for i in axes(u, 3), j in axes(u, 2)
        if ucol === nothing
            d = u[kc, j, i] - uprev[1, j, i]
            for k in kc+1:nt
                u[k, j, i] = uprev[k-kc+1, j, i] + d
            end
            h = cprev[j, i] - z0[kc, j, i]
        else
            for k in kc+1:nt
                u[k, j, i] = uprev[k-kc+1, j, i] + (u[k, j, i] - ucol[k-kc+1, j, i])
            end
            h = cprev[j, i] - zcol[j, i]
        end
        for k in 1:kc
            z0[k, j, i] += h
        end
    end
    return u, z0
end

"""
Problem on the τ grid from the start heights z0 (the cut level from `logtau_cut`); returns (pr, c) with c the heights
of the cut level. Weights and regularisation are set by `solve_tau`; `evaluate!`, `Jop!` and `JTop!` work on `pr`.
"""
function tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z0, dx, dy; logtau_cut = -4.0, S = nothing, mu = 1.0,
                  slant = (0.0, 0.0), balance = :vertical, precond = :fourier, klow = 8)
    nt, ny, nx = size(lT); NX = par.NX
    kc = argmin(abs.(lt .- logtau_cut)); n1 = nt - kc + 1
    zrel = z0[1:kc, :, :] .- z0[kc:kc, :, :]
    c = z0[kc, :, :]
    nyh = ny ÷ 2 + 1
    S === nothing && (S = SlabFFT(par, n1, ny))
    (S.nk, S.ny, S.nxl) == (n1, ny, nx) || error("solve_tau: cached SlabFFT has sizes $((S.nk, S.ny, S.nxl)), expected $((n1, ny, nx))")
    sy2 = [sin(2π * (j - 1) / ny)^2 / dy^2 for j in 1:nyh]
    sx2 = [sin(2π * fftfreq(NX)[i])^2 / dx^2 for i in 1:NX]
    jr = S.jr[par.rank+1]
    s2 = vec([sx2[kx] + sy2[jr[1]+jl-1] for kx in 1:NX, jl in 1:length(jr)])   # T columns: (kx, local ky)
    prec = precond in (:block, :hybrid) && NX >= 3 && ny >= 3 ? precond : :fourier
    wk = prec == :hybrid ? vec([Float64((fftfreq(NX)[kx] * NX)^2 + (jr[1] + jl - 2)^2 <= klow^2) for kx in 1:NX, jl in 1:length(jr)]) :
         ones(length(s2))
    Z() = zeros(n1, ny, nx); N3() = zeros(nt, ny, nx)
    pr = TP(par, S, tab, g, nt, kc, n1, ny, nx, dx, dy, LN10 .* lt, diff(LN10 .* lt), lT, Bx, By, Bz, zrel, N3(),
            N3(), N3(), N3(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(),
            ones(n1), ones(n1 - 1), zeros(n1), zeros(ComplexF64, 0, 0), zeros(ComplexF64, 0, 0), Float64[], s2,
            zeros(ComplexF64, n1, NX * length(jr)), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z(), ones(n1, ny, nx),
            Float64(mu), Float64(slant[1]), Float64(slant[2]), balance == :vertical, Z(), Z(),
            prec, zeros(0, 0, 0, 0), wk, zeros(0, 0, 0), 0.0, true)
    return pr, c
end

"""Weights of Φ (wc, wf: H̄² per volume) and the regularisation ereg from the evaluated state of `pr` (in a solve: the
start, fixed for the whole solve). Returns (H̄, Δz̄ per face, total weight)."""
function tp_weights!(pr::TP, g, dx)
    par = pr.par; kc, n1, nt, ny = pr.kc, pr.n1, pr.nt, pr.ny
    Hb = gmedians(par, 1 ./ (g .* pr.q))
    dzf = gmedians(par, pr.z[kc:nt-1, :, :] .- pr.z[kc+1:nt, :, :])
    dzn = [m == 1 ? dzf[1] / 2 : m == n1 ? dzf[end] / 2 : (dzf[m-1] + dzf[m]) / 2 for m in 1:n1]
    pr.wc = dzn .* Hb .^ 2
    pr.wf = [Hb[f] * Hb[f+1] / dzf[f] for f in 1:n1-1]
    κ = 2π / (par.NX * dx)
    Wb = gmedians(par, pr.W)
    pr.ereg = 1e-2 * κ^2 .* vcat(sum(pr.wc .* Wb .^ 2), pr.wc[2:end])
    return Hb, dzf, (2 * sum(pr.wc) + sum(pr.wf)) * ny * par.NX
end

# EOS domain of `evaluate!`: ln P within the table, and ln P ± EPS_PROBE too (the opacity derivative probes)
const EPS_PROBE = 1e-4

"""
Largest α ≤ ∞ that keeps every solved ln P of u + α δ inside its bound in the direction it moves (global). Three ln P
domains of the EOS table [Y₁, Y_end]: the physical table; the evaluator's, [Y₁ + EPS_PROBE, Y_end − EPS_PROBE] (the
opacity-derivative probes of `evaluate!` must stay in the table, an artificial margin); and the safety interval, `margin`
(≥ EPS_PROBE) inside the table edges. Per moving cell (δ ≠ 0), against the edge it moves toward:
- before the safety bound: the fraction to that bound (this includes a cell outside the opposite edge moving inward,
  which the near edge does not limit: entry is left to `evaluate!` and the line search);
- in the strip between the safety and the evaluator bounds: the fraction to the evaluator bound;
- at or beyond the evaluator bound: 0 (an outward step has no feasible fraction).
A non-finite ln P or δ in a moving cell gives 0. Cells that do not move are ignored; Inf when nothing limits. The
caller takes 0.99 of it, so a step never lands on a bound.
"""
function feas_scale(pr::TP, u, δ; margin = 2 * EPS_PROBE)
    margin >= EPS_PROBE || throw(ArgumentError("margin must be ≥ EPS_PROBE"))
    Y1 = pr.tab.Y[1]; Y2 = pr.tab.Y[end]; a = Inf
    @inbounds for i in 1:pr.nx, j in 1:pr.ny, m in 2:pr.n1
        d = δ[m, j, i]; v = u[node(pr, m), j, i]
        d == 0 && continue
        (isfinite(d) && isfinite(v)) || (a = 0.0; continue)
        s, e = d > 0 ? (Y2 - margin, Y2 - EPS_PROBE) : (Y1 + margin, Y1 + EPS_PROBE)   # safety, evaluator bound
        dist = (d > 0 ? v < s : v > s) ? s - v : (d > 0 ? v < e : v > e) ? e - v : 0.0
        a = min(a, dist / d)
    end
    return gmin(pr.par, a)
end

"""
Cells of (u, c) that `evaluate!` rejects (global counts): ln P below / above the table (`lo`, `hi`), in it but with a
derivative probe outside (`probe_lo`, `probe_hi`), other non-finite states (`other`, e.g. T outside); and the cell
furthest outside (largest distance in ln P from the nearer edge of the probe-valid range): global x, y, level, log τ,
T, ln P, distance, `du` = its δ (when given). For diagnostics after a failed trial; collective.
"""
function eos_cells(pr::TP, u; δ = nothing)
    tab = pr.tab; Y1 = tab.Y[1]; Y2 = tab.Y[end]; gi = owned_x(pr.par)
    n = zeros(Int, 5); worst = fill(-Inf, 9)
    @inbounds for i in 1:pr.nx, j in 1:pr.ny, m in 1:pr.n1
        k = node(pr, m); lT = pr.lT[k, j, i]; lp = u[k, j, i]
        r, _, c0, _ = H.state(tab, lT, lp)
        _, _, cp, _ = H.state(tab, lT, lp + EPS_PROBE); _, _, cm, _ = H.state(tab, lT, lp - EPS_PROBE)
        isfinite(r + c0 + cp + cm) && continue
        cls = lp < Y1 ? 1 : lp > Y2 ? 2 : !isfinite(r + c0) ? 5 : !isfinite(cm) ? 3 : !isfinite(cp) ? 4 : 5
        n[cls] += 1
        dist = max(Y1 + EPS_PROBE - lp, lp - (Y2 - EPS_PROBE))
        dist > worst[7] && (worst .= (gi[i], j, k, pr.lτ[k], exp(lT), lp, dist, δ === nothing ? NaN : δ[m, j, i], 0.0))
    end
    n = [gcount(pr.par, x) for x in n]
    gw = gmax(pr.par, worst[7])
    mine = worst[7] == gw && gw > -Inf ? pr.par.rank : pr.par.nranks
    owner = round(Int, gmin(pr.par, mine))                # nranks: no rejected cell anywhere (w = 0 below)
    w = gsum(pr.par, pr.par.rank == owner ? worst : zeros(9))
    owner == pr.par.nranks && (w .= NaN; w[1:3] .= -1)
    return (lo = n[1], hi = n[2], probe_lo = n[3], probe_hi = n[4], other = n[5], total = sum(n),
            worst = (x = round(Int, w[1]), y = round(Int, w[2]), level = round(Int, w[3]), logtau = w[4], T = w[5], lnP = w[6], dist = w[7], du = w[8]),
            Y = (Y1, Y2))
end

eos_str(e) = e.total == 0 ? "none" :
    @sprintf("%d (ln P below %d, above %d; probe only %d/%d; other %d; worst x %d y %d level %d log τ %.2f T %.0f ln P %.3f, %.2g beyond the edge [%.3f, %.3f], δu %.3g)",
             e.total, e.lo, e.hi, e.probe_lo, e.probe_hi, e.other, e.worst.x, e.worst.y, e.worst.level, e.worst.logtau,
             e.worst.T, e.worst.lnP, e.worst.dist, e.Y..., e.worst.du)

"""
Stop qualification at the evaluated state of `pr` (design/CODEX_REVIEW_solver_time_round2_2026-10-28.md, finding 1):
an interval for the regularised Gauss–Newton decrement d = bᵀA⁻¹b / Φ, A = JᵀWJ + D, D = diag(ereg0) > 0, b = −JᵀWR.
For any approximate solution x with r = b − Ax, L = 2bᵀx − xᵀAx ≤ bᵀA⁻¹b ≤ U = L + rᵀD⁻¹r (the missing term is
rᵀA⁻¹r and A ⪰ D). This bounds the regularised metric (H + D, not H alone), not heights or global optimality.
PCG (undamped, mlm = 0) to `tol`; with `refine`, the tolerance is cut ×10 from the current x until the interval decides
`gn_tol` or `maxit` CG recurrence iterations in all are spent. `outcome`:
:qualified (U (1 + margin) < gn_tol), :not_stationary (L (1 − margin) > gn_tol), :undecided (the interval straddles
gn_tol within the budget) or :invalid (Φ, ereg0 or b not finite and positive; a non-finite or inconsistent interval,
e.g. a negative-curvature breakdown). b = 0 qualifies exactly (L = U = 0). Returns a NamedTuple with the outcome,
`lower`, `upper` (/ Φ), `pred` = bᵀx/Φ, `model` = (2bᵀx − ‖Jx‖²_W)/Φ, `ereg` = xᵀDx/Φ, `its`, `true_rel` = ‖r‖/‖b‖,
`grad_norm` = ‖b‖, `gn_tol`, `qual_tol`, the `step` and `phi` at which it was evaluated, and its work: `aprod` and
`minv` (operator and preconditioner applications, restarts and interval residuals included) and `seconds`. Uses `rhs` and `δ` as work
arrays and leaves pr.ereg = ereg0, pr.mlm = 0.
"""
function qualify_stop(pr::TP, par, rhs, δ, ereg0, Φ; tol = 1e-2, maxit = 400, gn_tol = 1e-3, refine = true,
                      margin = 1e-6, step = 0)
    t0 = time(); cnt = zeros(Int, 2); fl = Ref(:none)
    pr.ereg = copy(ereg0); pr.mlm = 0.0
    JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
    nb = gnorm(par, rhs)
    q(o, l, u, p, m, e, its, tr) = (outcome = o, lower = l, upper = u, pred = p, model = m, ereg = e, its = its,
                                    true_rel = tr, grad_norm = nb, gn_tol = gn_tol, qual_tol = tol, step = step, phi = Φ,
                                    aprod = cnt[1], minv = cnt[2], seconds = time() - t0, pcg = fl[])
    (isfinite(Φ) && Φ > 0 && all(isfinite, ereg0) && all(>(0), ereg0) && isfinite(nb)) ||
        return q(:invalid, NaN, NaN, NaN, NaN, NaN, 0, NaN)
    fill!(δ, 0.0)
    nb == 0 && return q(:qualified, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    setprec!(pr)
    t = similar(δ); its = 0; tol_k = tol
    while true
        k, _ = pcg!(Aop!, Minv!, par, pr, δ, rhs; tol = tol_k, maxit = maxit - its, x0 = its > 0, cnt = cnt, flag = fl)
        its += k
        abn = fl[] in PCG_ABNORMAL
        Aop!(pr, t, δ); cnt[1] += 1
        bx = gdot(par, δ, rhs); xAx = gdot(par, δ, t)
        rD = gsum(par, sum((rhs .- t) .^ 2 ./ ereg0)); tr = gnorm(par, rhs .- t) / nb
        e = gsum(par, sum(δ .^ 2 .* ereg0))
        L = 2bx - xAx; U = L + rD
        lo = L / Φ; up = U / Φ
        vals = (lo, up, bx / Φ, (2bx - (xAx - e)) / Φ, e / Φ, its, tr)
        all(isfinite, (L, U, xAx)) && L >= 0 && xAx >= 0 || return q(:invalid, vals...)
        up * (1 + margin) < gn_tol && return q(:qualified, vals...)
        lo * (1 - margin) > gn_tol && return q(:not_stationary, vals...)
        (refine && its < maxit && tol_k > 1e-12 && !abn) || return q(:undecided, vals...)   # no refining after a breakdown
        tol_k /= 10
    end
end

"""Reused weights (`solve_tau` keyword): lengths n1, n1 − 1, n1, all finite and positive, and the cut level and grid
of the solve that made them when recorded (info.weights carries kc, n1, NX, ny)."""
function check_weights(w, n1, kc, NX, ny)
    length(w.wc) == n1 && length(w.wf) == n1 - 1 && length(w.ereg) == n1 ||
        error("solve_tau: weights have lengths $((length(w.wc), length(w.wf), length(w.ereg))), expected $((n1, n1 - 1, n1))")
    all(v -> all(isfinite, v) && all(>(0), v), (w.wc, w.wf, w.ereg)) || error("solve_tau: weights must be finite and positive")
    for (k, v) in ((:kc, kc), (:n1, n1), (:NX, NX), (:ny, ny))
        hasproperty(w, k) && getproperty(w, k) != v && error("solve_tau: weights were made with $k = $(getproperty(w, k)), this solve has $v")
    end
    return nothing
end

"""
    solve_tau(par, tab, g, lt, lT, Bx, By, Bz, u, z0, dx, dy; logtau_cut, max_steps, rtol, n_consec, cg_maxit,
              dc_cap, lm_mu0, logf, mu, slant)

Direct Poisson solve from the column HSE in τ (u = ln P, z0 aligned and gauged). Stops on the objective only: when the
relative decrease (Φ_prev − Φ)/Φ_prev of accepted steps is below rtol on n_consec consecutive steps (a single small
step is not enough: Φ is nearly flat along the heights of a few strong-field columns). Height changes are logged, not
used for stopping.
Returns (u, z, Φ0, Φ, steps, kc, info). `info` (a NamedTuple) carries the stopping `reason` (:converged, :no_descent,
:domain_limited (no decrease, and the last step had no feasible fraction: `feas_scale`), :max_steps, :eos_out_of_envelope for a start state outside the table, in which case u and z are the inputs), the
Gauss–Newton decrement `gn_decrement` (the relative decrease of Φ a further undamped Gauss–Newton step would predict:
a scale-free stationarity measure), the gradient norm `grad_norm` = ‖Jᵀ W R‖, the last linear solve (`cg_its`,
`cg_rel`), the final damping `lm_mu`, `rms0`/`rms` = √(Φ/W) and `phi0_zero`. `S` reuses a cached SlabFFT for these
grid sizes; `info.work`: step solves (CG recurrence iterations `cg`, linear solves with retries `solves`, operator and
preconditioner applications `aprod`, `minv`; the ‖Jδ‖² product per solve is not counted), `evals` (trial and start
evaluations of R), `restores` (re-evaluations of the linearisation state after a failed try), qualification
(`qual_cg`, `qual_aprod`, `qual_minv`), the final stationarity diagnostic (`final_*`, zero when it reuses the stop
check at the returned state) and seconds (`step_s`, `qual_s`, `final_s`, `total_s`); `rms_atol` > 0 accepts a start state with √(Φ₀/W) ≤ rms_atol as converged in zero steps. `mu` = cos θ
(lt is τ along the line of sight) and `slant` = (t_x, t_y) of the inclined geometry ((0, 0): vertical columns).
`on_step(s, u, z, info)` (diagnostics) is called after every accepted step with the state (z not yet gauged) and
`info = (phi, pred, cg_its, cg_rel, alpha, seconds)`; `pred` is the step's predicted relative decrease δ·(−JᵀWR)/Φ.
`stop = :gn` (design/tau/SOLVER_TIME_DESIGN.md D1) stops instead when `pred` < gn_tol on gn_consec consecutive
accepted steps: no Gauss–Newton step could lower Φ by more than that fraction (scale-free; independent of the tail).
`balance` (inclined geometry only): :vertical (default, D0) or :ray (the column residual along the ray).
D2 (design/tau/SOLVER_TIME_DESIGN.md): `lm = :gain` sets the damping from the gain ratio ρ = actual / model decrease of
the accepted step (model 2α g·δ − α²‖Jδ‖²_W, g = −JᵀWR): ρ > 0.75 lowers μ ×3, ρ < 0.25 raises it to max(4μ, 1);
`lm = :alpha` is the earlier rule (from the line-search length); `lm = :fixed` keeps μ = lm_mu0 (raised only when no
step decreases Φ; with `damping = :marquardt` and `cap_frac = 0`: fixed Marquardt damping, no cap retries). `forcing = :ew` sets the CG tolerance from the outer
progress, η = clamp(pred of the previous step, eta_min, eta_max), with at most `cg_cap` iterations; `forcing = :fixed`
uses cg_tol and cg_maxit. `precond` (D3b): :fourier (layer-median Fourier modes), :block (exact column blocks, block
Jacobi; `colblocks!`) or :hybrid (both, the Fourier part on the modes |k| ≤ `klow` (mode index) only; D3c).
`damping = :marquardt` (block preconditioners only) damps with μ diag(JᵀWJ) instead of μ times the regularisation
shape, from `mlm_min` up, and treats a step the caps would scale below `cap_frac` as outside the trust region (μ is
raised and the step solved again): the column blocks expose near-null directions (ln P just below the cut) that the
undamped step fills with huge values. A failed line search re-evaluates the state before the next solve.
`stop = :gn` with `qualify` (default): when pred < gn_tol on gn_consec steps, `qualify_stop` bounds the undamped
regularised decrement at the current state (PCG from `qual_tol`, refined, at most `qual_maxit` products): converged if
the upper bound is below gn_tol; the run continues if the lower bound is above; reason :unresolved if the interval
still straddles gn_tol or is invalid (`info.stop_check`: the outcome, bounds, step and work of the last check; at the
returned state it also supplies the stationarity diagnostics instead of a second, weaker solve).
Round-off (`phi_floor`): a start state whose Φ is at or below the round-off floor of its own residual terms returns at
once with :converged (info.roundoff = :start); a step that finds no decrease from such a state ends :converged
(info.roundoff = :no_descent) instead of :no_descent. A step that finds no decrease from a state above the floor ends
:no_descent (or :domain_limited) as before; the gn_tol decrement test and the step budget are unchanged.
`weights` = (wc, wf, ereg) replaces the weights and regularisation computed from the start state (the objective of
another solve of the same cube, so that different starts minimise the same Φ; checked by `check_weights`);
`info.weights` returns those used, with the cut level and grid. `feas` (default true): each trial starts at no more
than 0.99 of the fraction of δ that keeps every solved ln P inside the table less 2 EPS_PROBE (`feas_scale`; the steps
it limited are counted in `info.work.feas_limited`); failed trials log their out-of-table cells (`eos_cells`).
"""
function solve_tau(par, tab, g, lt, lT, Bx, By, Bz, u, z0, dx, dy; logtau_cut = -4.0, max_steps = 60, rtol = 1e-4,
                   n_consec = 3, cg_maxit = 80,
                   cg_tol = 1e-3, dc_cap = 3e5, du_cap = 2.0, lm_mu0 = 0.0, logf = s -> nothing, S = nothing,
                   rms_atol = 0.0, stationarity = true, mu = 1.0, slant = (0.0, 0.0), on_step = nothing,
                   stop = :phi, gn_tol = 1e-3, gn_consec = 2, balance = :vertical, lm = :alpha, forcing = :fixed,
                   cg_cap = 200, eta_min = 0.005, eta_max = 0.1, precond = :fourier, klow = 8,
                   damping = :reg, cap_frac = 0.5, mlm_min = 0.01, weights = nothing, qualify = true, qual_tol = 1e-2,
                   qual_maxit = 400, feas = true)
    nt, ny, nx = size(lT); NX = par.NX
    pr, c = tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z0, dx, dy; logtau_cut = logtau_cut, S = S, mu = mu, slant = slant,
                     balance = balance, precond = precond, klow = klow)
    kc, n1 = pr.kc, pr.n1
    WORK0 = (cg = 0, solves = 0, aprod = 0, minv = 0, evals = 1, restores = 0, qual_cg = 0, qual_aprod = 0, qual_minv = 0,
             final_cg = 0, final_aprod = 0, final_minv = 0, feas_limited = 0, cg_abnormal = 0, step_s = 0.0, qual_s = 0.0, final_s = 0.0, total_s = 0.0)
    tstart = time()
    damping == :marquardt && pr.prec == :fourier && error("solve_tau: damping = :marquardt needs precond = :block or :hybrid")
    mq = damping == :marquardt
    Z() = zeros(n1, ny, nx)
    _, bad = evaluate!(pr, u, c)
    if bad > 0                                       # gcount inside evaluate!: every rank takes this branch
        info = (reason = :eos_out_of_envelope, gn_decrement = NaN, grad_norm = NaN, cg_its = 0, cg_rel = NaN,
                lm_mu = lm_mu0, rms0 = NaN, rms = NaN, phi0_zero = false, weights = nothing, stop_check = nothing,
                work = WORK0, phi_floor = NaN, roundoff = :none)
        return u, copy(z0), NaN, NaN, 0, kc, info
    end
    Hb, dzf, Wtot = tp_weights!(pr, g, dx)
    if weights !== nothing                           # the objective of another solve (same cube, grid and cut)
        check_weights(weights, n1, kc, NX, ny)
        pr.wc = copy(weights.wc); pr.wf = copy(weights.wf); pr.ereg = copy(weights.ereg)
        Wtot = (2 * sum(pr.wc) + sum(pr.wf)) * ny * NX
    end
    wts = (wc = copy(pr.wc), wf = copy(pr.wf), ereg = copy(pr.ereg), kc = kc, n1 = n1, NX = NX, ny = ny)
    k0 = argmin(abs.(lt))
    zq() = gquantiles(par, pr.z[k0, :, :], [0.01, 0.5, 0.99]) ./ 1e3
    Φ, _ = evaluate!(pr, u, c); Φ0 = Φ
    logf(@sprintf("start: cut at log10 τ = %.2f (level %d, %d ln P levels + 1 height per column), Φ %.6e, RMS %.4e | %s | z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km",
                  lt[kc], kc, n1 - 1, Φ, sqrt(Φ / Wtot), bands(pr, lt, Hb, dzf), zq()...))
    reason = :max_steps
    Φfl = phi_floor(pr, u)
    if sqrt(Φ0 / Wtot) <= rms_atol || Φ0 == 0 || Φ0 <= Φfl
        info = (reason = :converged, gn_decrement = 0.0, grad_norm = 0.0, cg_its = 0, cg_rel = 0.0, lm_mu = lm_mu0,
                rms0 = sqrt(Φ0 / Wtot), rms = sqrt(Φ0 / Wtot), phi0_zero = Φ0 == 0, weights = wts, stop_check = nothing,
                work = merge(WORK0, (evals = 2, total_s = time() - tstart)), phi_floor = Φfl,
                roundoff = Φ0 <= Φfl ? :start : :none)
        logf(Φ0 <= Φfl ? @sprintf("start state at round-off (Φ %.3e ≤ floor %.3e): no steps", Φ0, Φfl) :
                         "start state already within rms_atol: no steps")
        return u, copy(pr.z), Φ0, Φ0, 0, kc, info
    end
    rnd = :none
    # Levenberg–Marquardt: damping μ D (D the regularisation's diagonal shape), raised ×4 when a step fails, lowered ×3
    # after a full step; steps over the caps are scaled as a whole (an element-wise clamp can make an ascent direction)
    δ = Z(); rhs = Z(); ut = copy(u); ct = copy(c)
    ereg0 = copy(pr.ereg); Dlm = ereg0 ./ 1e-2
    μ = lm_mu0; steps = 0; nsmall = 0; npred = 0; qinfo = nothing
    # work: CG recurrence iterations, linear solves (retries included), operator / preconditioner applications, trial
    # evaluations (start evaluations included), restorations of the linearisation state, qualification, seconds
    ncg = 0; nsolve = 0; neval = 2; nrest = 0; nqual = 0; cstep = zeros(Int, 2); cqual = zeros(Int, 2)
    tsteps = 0.0; tqual = 0.0; nfeas = 0; nabn = 0; pfl = Ref(:none)              # nfeas: tries whose start α the EOS boundary limited
    η = eta_max; tol_k, maxit_k = forcing == :ew ? (η, cg_cap) : (cg_tol, cg_maxit)
    jx, jy, jz = Z(), Z(), Z()
    for s in 1:max_steps
        t0 = time()
        JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
        accepted = false; its = 0; rel = 0.0; α = 1.0; Φt = Inf; sc = 1.0; pred = NaN; jj = NaN
        cg_s = 0; tries_s = 0; ev_s = 0
        fails = String[]; αf = Inf                        # per failed try: μ, CG, pred, cap scale, trial Φ/Φ (bad = EOS out of range)
        for tries in 1:8
            tries_s = tries
            if mq
                pr.mlm = μ
            else
                pr.ereg = ereg0 .+ μ .* Dlm
            end
            setprec!(pr)
            its, rel = pcg!(Aop!, Minv!, par, pr, δ, rhs; tol = tol_k, maxit = maxit_k, cnt = cstep, flag = pfl)
            pfl[] in PCG_ABNORMAL && (nabn += 1; logf(@sprintf("step %2d: PCG ended %s after %d its (true rel %.2e)", s, pfl[], its, rel)))
            cg_s += its
            pred = gdot(par, δ, rhs) / Φ
            Jop!(pr, jx, jy, jz, δ)                   # ‖Jδ‖²_W at the linearisation state (before the line search)
            jj = wnorm2(pr, jx, jy, jz)
            mu_ = gmax(par, maximum(abs, @view δ[2:end, :, :])); mc = gmax(par, maximum(abs, @view δ[1, :, :]))
            sc = min(1.0, du_cap / mu_, dc_cap / mc)
            if mq && sc < cap_frac && tries < 8           # the step leaves the trust region set by the caps: damp more
                μ = max(4μ, mlm_min)
                continue
            end
            αf = feas ? feas_scale(pr, u, δ) : Inf          # fraction to the EOS-table boundary (ln P)
            α = min(sc, 0.99αf); trial = String[]; αf < sc / 0.99 && (nfeas += 1)
            for ls in 1:12                                # 4 halvings, more while the trial leaves the EOS table
                ls > 4 && bad == 0 && break
                ut .= u; @views ut[kc+1:nt, :, :] .+= α .* δ[2:end, :, :]
                @views ct .= c .+ α .* δ[1, :, :]
                Φt, bad = evaluate!(pr, ut, ct); ev_s += 1
                bad == 0 && Φt < Φ && (accepted = true; break)
                push!(trial, bad == 0 ? @sprintf("%.4g", Φt / Φ) : "bad " * eos_str(eos_cells(pr, ut; δ = δ)))
                α /= 2
            end
            accepted && break
            nrest += 1
            push!(fails, @sprintf("μ %.3g: CG %d rel %.2g pred %.3g cap %.3g feasible %.3g, Φ/Φ [%s]", μ, its, rel, pred, sc, αf,
                                  join(trial, " | ")))
            evaluate!(pr, u, c)                           # back to the linearisation state before solving again
            μ = max(4μ, mq ? mlm_min : 1.0)
        end
        ncg += cg_s; nsolve += tries_s; neval += ev_s
        if !accepted
            evaluate!(pr, u, c); nrest += 1
            Φfl = phi_floor(pr, u)
            if Φ <= Φfl                                   # no decrease because the state is a discrete equilibrium to round-off
                logf(@sprintf("step %2d: no decrease, and Φ %.3e is at the round-off floor %.3e: converged (round-off)", s, Φ, Φfl))
                reason = :converged; rnd = :no_descent
                break
            end
            dom = αf == 0
            logf(@sprintf("step %2d: no decrease up to μ = %.3g%s; stopping", s, μ,
                          dom ? " (the last step had no feasible fraction: a solved ln P at the EOS-domain bound it moves toward)" : ""))
            foreach(f -> logf("    failed try " * f), fails)
            reason = dom ? :domain_limited : :no_descent
            break
        end
        dec = (Φ - Φt) / Φ
        model = 2α * pred - α^2 * jj / Φ             # relative model decrease of the step αδ
        ρg = dec / model
        if lm == :fixed                              # constant damping (raised only by a failed line search above)
        elseif lm == :gain
            μ = !(model > 0) || ρg < 0.25 ? max(4μ, mq ? mlm_min : 1.0) : ρg > 0.75 ? μ / 3 : μ
        else
            μ = α == 1.0 ? μ / 3 : (α >= sc / 2 ? μ : 2μ)
        end
        μ < (mq ? mlm_min / 10 : 1e-3) && (μ = 0.0)
        u .= ut; c .= ct; Φ = Φt; steps = s
        dzmax = α * gmax(par, maximum(abs, @view δ[1, :, :]))
        nsmall = dec < rtol ? nsmall + 1 : 0
        logf(@sprintf("step %2d: Φ/Φ₀ %.3e (ΔΦ/Φ %.2e, predicted %.2e, model %.2e, ρ %.2f), RMS %.4e | CG %d its (rel %.1e, tol %.1e; step %d its in %d solves, %d evaluations), α %.3g (cap scale %.3g, feasible %.3g), μ next %.3g, max|δc| %.1f km | z(τ=1) p1/p50/p99 %.1f / %.1f / %.1f km | %.1f s",
                      s, Φ / Φ0, dec, pred, model, ρg, sqrt(Φ / Wtot), its, rel, tol_k, cg_s, tries_s, ev_s, α, sc, αf, μ, dzmax / 1e3, zq()..., time() - t0))
        tsteps += time() - t0
        φh, φv, φt = on_step === nothing ? (NaN, NaN, NaN) : phi_parts(pr)
        on_step === nothing || on_step(s, u, pr.z, (phi = Φ, pred = pred, cg_its = its, cg_rel = rel, alpha = α,
                                                    seconds = time() - t0, model = model, rho = ρg, cg_tol = tol_k, lm_mu = μ,
                                                    cg_step = cg_s, solves = tries_s, evals = ev_s, cap_scale = sc,
                                                    feas_scale = αf, phi_h = φh, phi_v = φv, phi_v3 = φt))
        forcing == :ew && isfinite(pred) && (tol_k = clamp(pred, eta_min, eta_max))
        npred = pred < gn_tol ? npred + 1 : 0
        if stop == :gn
            if npred >= gn_consec
                qualify || (logf(@sprintf("converged: predicted decrease < %.0e on %d consecutive steps (last max|δc| %.1f m)",
                                          gn_tol, npred, dzmax)); reason = :converged; break)
                # the steps' pred is that of the damped, inexactly solved system at the previous state: qualify at this
                # state with an interval for the undamped regularised decrement (qualify_stop)
                q = qualify_stop(pr, par, rhs, δ, ereg0, Φ; tol = qual_tol, maxit = qual_maxit, gn_tol = gn_tol, step = s)
                qinfo = q; nqual += q.its; cqual[1] += q.aprod; cqual[2] += q.minv; tqual += q.seconds
                logf(@sprintf("stop check (step %d): %s, regularised decrement in [%.2e, %.2e] (tol %.0e) | pred %.2e, model %.2e, ereg part %.2e | CG %d its, true rel %.1e",
                              s, q.outcome, q.lower, q.upper, gn_tol, q.pred, q.model, q.ereg, q.its, q.true_rel))
                if q.outcome == :qualified
                    logf(@sprintf("converged: decrement < %.0e qualified (last max|δc| %.1f m)", gn_tol, dzmax))
                    reason = :converged; break
                elseif q.outcome == :not_stationary
                    npred = 0                                    # the decrement is above the threshold: carry on
                else
                    logf("stop unresolved: the qualification did not decide the threshold within its budget; stopping")
                    reason = :unresolved; break
                end
            end
        else
            nsmall >= n_consec && (logf(@sprintf("converged: ΔΦ/Φ < %.0e on %d consecutive steps (last max|δc| %.1f m)", rtol, nsmall, dzmax));
                                   reason = :converged; break)
        end
    end
    pr.ereg = ereg0; pr.mlm = 0.0
    logf("end: " * bands(pr, lt, Hb, dzf))
    # stationarity at the returned state (pr holds its evaluation): undamped Gauss–Newton step δ from (JᵀWJ + ereg) δ =
    # −JᵀWR and the predicted relative decrease δ·(−JᵀWR)/Φ. Diagnostics only: u and z are not changed.
    gnd = NaN; gn = NaN; its = 0; rel = NaN; cfin = zeros(Int, 2); tf = time(); fin_its = 0
    if qinfo !== nothing && qinfo.step == steps && qinfo.outcome != :invalid   # a qualification at the returned state
        gnd = qinfo.pred; gn = qinfo.grad_norm; its = qinfo.its; rel = qinfo.true_rel
        logf(@sprintf("stationarity: from the stop check at step %d, decrement in [%.3e, %.3e]", steps, qinfo.lower, qinfo.upper))
    elseif stationarity
        JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
        gn = gnorm(par, rhs)
        setprec!(pr)
        its, rel = pcg!(Aop!, Minv!, par, pr, δ, rhs; tol = cg_tol, maxit = cg_maxit, cnt = cfin, flag = pfl); fin_its = its
        pfl[] in PCG_ABNORMAL && logf("stationarity: PCG ended $(pfl[])")
        gnd = gdot(par, δ, rhs) / Φ
        logf(@sprintf("stationarity: Gauss–Newton decrement %.3e, ‖JᵀWR‖ %.3e (CG %d its, rel %.1e)", gnd, gn, its, rel))
    end
    info = (reason = reason, gn_decrement = gnd, grad_norm = gn, cg_its = its, cg_rel = rel, lm_mu = μ,
            rms0 = sqrt(Φ0 / Wtot), rms = sqrt(Φ / Wtot), phi0_zero = false, weights = wts, stop_check = qinfo,
            work = (cg = ncg, solves = nsolve, aprod = cstep[1], minv = cstep[2], evals = neval, restores = nrest,
                    qual_cg = nqual, qual_aprod = cqual[1], qual_minv = cqual[2], final_cg = fin_its, final_aprod = cfin[1],
                    final_minv = cfin[2], feas_limited = nfeas, cg_abnormal = nabn, step_s = tsteps, qual_s = tqual, final_s = time() - tf, total_s = time() - tstart),
            phi_floor = phi_floor(pr, u), roundoff = rnd)
    w = info.work
    logf(@sprintf("work: steps %d CG its in %d solves (%d A, %d M⁻¹ applications), %d evaluations + %d restorations, %.1f s; qualification %d CG (%d A, %d M⁻¹), %.1f s; final diagnostic %d CG (%d A), %.1f s%s; total %.1f s",
                  w.cg, w.solves, w.aprod, w.minv, w.evals, w.restores, w.step_s, w.qual_cg, w.qual_aprod, w.qual_minv, w.qual_s,
                  w.final_cg, w.final_aprod, w.final_s, fin_its == 0 && qinfo !== nothing && qinfo.step == steps ? " (reused the stop check)" : "",
                  w.total_s))
    return u, copy(pr.z), Φ0, Φ, steps, kc, info
end

end # module PoissonTau
