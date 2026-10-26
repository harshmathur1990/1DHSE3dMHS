# Column hydrostatic equilibrium (spec §5, implementation doc §9).
#
# du/dz = −g exp(Lρ(X(z), u) − u),  dm/dz = −exp(Lρ(X(z), u)),  u = ln(P / 1 Pa),
# X = ln T linear in z inside each interval, classical RK4 with n_sub substeps and
# interval-wise substep doubling (n vs 2n) up to n_sub_max. Column mass is co-integrated (E2).

struct HSEOpts
    anchor::Symbol          # :bottom | :top | :height
    P_ref::Float64
    z_ref::Float64
    g::Float64
    n_sub::Int
    adapt::Bool
    rk_tol::Float64
    n_sub_max::Int
    tau_top::Symbol
    H_max::Float64
    mu::Float64
end

HSEOpts(; anchor = :bottom, P_ref, z_ref = NaN, g = 274.0, n_sub = 4, adapt = true,
        rk_tol = 1e-6, n_sub_max = 64, tau_top = :exp_tail, H_max = 5.0e6, mu = 1.0) =
    HSEOpts(anchor, P_ref, z_ref, g, n_sub, adapt, rk_tol, n_sub_max, tau_top, H_max, mu)

"""X at height z on the interval (za, Xa)–(zb, Xb); exact at the endpoints."""
@inline function xlin(za, Xa, zb, Xb, z)
    z == za && return Xa
    z == zb && return Xb
    return Xa + (Xb - Xa) * ((z - za) / (zb - za))
end

"""
RK4 from zs to ze in n substeps (X linear on (za, Xa)–(zb, Xb)). Returns (u, m, status);
stops early if the EOS leaves its envelope.
"""
function rk4_seg(eos, za, Xa, zb, Xb, zs, ze, u, m, n::Int, g)
    h = (ze - zs) / n
    st = EOS_OK
    for s in 0:n-1
        z0 = zs + s * h
        z1 = s == n - 1 ? ze : zs + (s + 1) * h
        zh = z0 + 0.5 * (z1 - z0)
        X0 = xlin(za, Xa, zb, Xb, z0); Xh = xlin(za, Xa, zb, Xb, zh); X1 = xlin(za, Xa, zb, Xb, z1)
        hs = z1 - z0
        l1, s1 = lnrho(eos, X0, u)
        k1u = -g * exp(l1 - u); k1m = -exp(l1)
        u2 = u + 0.5hs * k1u
        l2, s2 = lnrho(eos, Xh, u2)
        k2u = -g * exp(l2 - u2); k2m = -exp(l2)
        u3 = u + 0.5hs * k2u
        l3, s3 = lnrho(eos, Xh, u3)
        k3u = -g * exp(l3 - u3); k3m = -exp(l3)
        u4 = u + hs * k3u
        l4, s4 = lnrho(eos, X1, u4)
        k4u = -g * exp(l4 - u4); k4m = -exp(l4)
        st |= s1 | s2 | s3 | s4
        (st & EOS_OUT_OF_ENVELOPE != 0) && return u, m, st
        u += hs / 6 * (k1u + 2k2u + 2k3u + k4u)
        m += hs / 6 * (k1m + 2k2m + 2k3m + k4m)
    end
    return u, m, st
end

"""One interval with substep doubling. Returns (u, m, n_used, column_flags, eos_status)."""
function rk4_adaptive(eos, za, Xa, zb, Xb, zs, ze, u, m, o::HSEOpts)
    if !o.adapt
        u1, m1, st = rk4_seg(eos, za, Xa, zb, Xb, zs, ze, u, m, o.n_sub, o.g)
        return u1, m1, o.n_sub, COL_OK, st
    end
    n = o.n_sub
    uc, mc, st = rk4_seg(eos, za, Xa, zb, Xb, zs, ze, u, m, n, o.g)
    uf, mf, st2 = rk4_seg(eos, za, Xa, zb, Xb, zs, ze, u, m, 2n, o.g)
    st |= st2
    while abs(uf - uc) > o.rk_tol && 2n < o.n_sub_max && (st & EOS_OUT_OF_ENVELOPE == 0)
        n *= 2
        uc, mc = uf, mf
        uf, mf, st3 = rk4_seg(eos, za, Xa, zb, Xb, zs, ze, u, m, 2n, o.g)
        st |= st3
    end
    fl = COL_OK
    n > o.n_sub && (fl |= COL_RK_REFINED)
    abs(uf - uc) > o.rk_tol && (fl |= COL_RK_TOL_NOT_MET)
    return uf, mf, 2n, fl, st
end

"""
    hse_column!(u, m, X, z, eos, o) -> (column_flags, max_substeps, anchor_k)

Integrate one column (X = ln T, z top-first). `u` and `m` receive ln P and the raw column
mass (zero at the anchor). anchor_k is the anchor node (0 for :height).
"""
function hse_column!(u::AbstractVector, m::AbstractVector, X::AbstractVector, z::AbstractVector, eos, o::HSEOpts)
    NZ = length(z)
    fl = COL_OK
    nmax = 0
    lp = log(o.P_ref)
    function step!(ka, kb, zs, uin, min)       # interval between nodes ka, kb, from zs to z[kb]
        u1, m1, nu, f, st = rk4_adaptive(eos, z[ka], X[ka], z[kb], X[kb], zs, z[kb], uin, min, o)
        fl |= f
        nmax = max(nmax, nu)
        if st & EOS_OUT_OF_ENVELOPE != 0
            fl |= COL_EOS_FAIL
            return false
        end
        u[kb] = u1; m[kb] = m1
        return true
    end
    if o.anchor == :bottom
        u[NZ] = lp; m[NZ] = 0.0
        for k in NZ-1:-1:1
            step!(k + 1, k, z[k+1], u[k+1], m[k+1]) || break
        end
        ak = NZ
    elseif o.anchor == :top
        u[1] = lp; m[1] = 0.0
        for k in 2:NZ
            step!(k - 1, k, z[k-1], u[k-1], m[k-1]) || break
        end
        ak = 1
    elseif o.anchor == :height
        (z[NZ] <= o.z_ref <= z[1]) || invalid("z_ref = $(o.z_ref) m is outside the grid")
        kr = findlast(k -> z[k] >= o.z_ref, 1:NZ)
        kr == NZ && (kr = NZ - 1)
        ok = step!(kr + 1, kr, o.z_ref, lp, 0.0)               # up to node kr
        if ok
            for k in kr-1:-1:1
                step!(k + 1, k, z[k+1], u[k+1], m[k+1]) || break
            end
        end
        ok2 = step!(kr, kr + 1, o.z_ref, lp, 0.0)               # down to node kr+1
        if ok2
            for k in kr+2:NZ
                step!(k - 1, k, z[k-1], u[k-1], m[k-1]) || break
            end
        end
        ak = 0
    else
        invalid("column_hse anchor must be bottom, top or height; got $(o.anchor)")
    end
    return fl, nmax, ak
end

"""
    column_state!(P, rho, ne, chi, tau, est, X, u, z, ak, eos, o) -> (tau_top_method, z_tau1)

State of one integrated column (u = ln P from `hse_column!`, anchor node `ak`): P (exactly `o.P_ref` at the anchor
node), ρ, nₑ and χ₅₀₀ [m⁻¹] from one `state` call per node on the EOS table, the EOS flags, τ₅₀₀ by `tau_column!` on z
and z(τ = 1). Shared by `column_hse` (2-D blocks) and the embedded slab path, so both run identical arithmetic.
"""
function column_state!(P, rho, ne, chi, tau, est, X, u, z, ak, eos, o::HSEOpts)
    @inbounds for k in eachindex(z)
        lr, ln, lc, st = state(eos, X[k], u[k])
        est[k] = st
        P[k] = (k == ak) ? o.P_ref : exp(u[k])
        rho[k] = exp(lr); ne[k] = exp(ln); chi[k] = exp(lc)
    end
    ttm = tau_column!(tau, chi, z; top = o.tau_top, H_max = o.H_max, mu = o.mu)
    return ttm, z_tau1(tau, z)
end

"""Output fields of column_hse (local blocks)."""
struct ColumnHSEResult
    fields::Dict{String,Field}
    eos_status::Mask3
    column_status::Matrix{UInt8}
    tau_top_method::Matrix{UInt8}
    rk_max_substeps::Matrix{Int16}
    z_tau1::Matrix{Float64}
    status::SolverStatus
    messages::Vector{String}
    stats::Dict{String,Any}
end

const COLHSE_FIELDS = (
    ("Pgas", "Pa", "gas pressure"), ("rho", "kg m-3", "mass density"), ("ne", "m-3", "electron number density"),
    ("Pe", "Pa", "electron pressure"), ("nH_total", "m-3", "total hydrogen nuclei density"),
    ("mu_bar", "1", "mean molecular weight"), ("colmass", "kg m-2", "column mass from the top plane (RK4 co-integrated)"),
    ("colmass_trap", "kg m-2", "column mass from the top plane (trapezoid of rho)"),
    ("chi500", "m-1", "extinction at 500 nm (absorption + scattering)"), ("tau500", "1", "optical depth at 500 nm"),
    ("logtau500", "1", "log10 optical depth at 500 nm"))

"""
    column_hse(par, grid, T, eos, o) -> ColumnHSEResult

Every owned column independently; threaded over column chunks. Bitwise independent of the
thread count and of the MPI decomposition (no reductions enter the values).
"""
function column_hse(par::ParCtx, grid::Grid, T::Field, eos, o::HSEOpts)
    b = par.block
    NZ = b.nz
    z = grid.z.s
    F = Dict{String,Field}(nm => newfield(par; fill = NaN) for (nm, _, _) in COLHSE_FIELDS)
    est = newmask(par)
    cst = zeros(UInt8, b.nx, b.ny); ttm = zeros(UInt8, b.nx, b.ny)
    nsub = zeros(Int16, b.nx, b.ny); zt1 = fill(NaN, b.nx, b.ny)
    mph = mass_per_H(eos)
    ncol = b.nx * b.ny
    nchunk = min(ncol, 8 * Threads.nthreads())
    Threads.@threads :static for ch in 1:nchunk
        Xc = zeros(NZ); u = zeros(NZ); m = zeros(NZ); chi = zeros(NZ); tau = zeros(NZ)
        Pc = zeros(NZ); rc = zeros(NZ); nc = zeros(NZ); sc = zeros(UInt8, NZ)
        for c in ch:nchunk:ncol
            i = (c - 1) % b.nx + 1
            j = (c - 1) ÷ b.nx + 1
            @inbounds for k in 1:NZ
                Xc[k] = log(T[i, j, k])
            end
            fill!(u, NaN); fill!(m, NaN)
            fl, nmax, ak = hse_column!(u, m, Xc, z, eos, o)
            cst[i, j] = fl
            nsub[i, j] = nmax
            if fl & COL_EOS_FAIL != 0
                continue                                     # outputs stay NaN
            end
            mtop = m[1]
            ttm[i, j], zt1[i, j] = column_state!(Pc, rc, nc, chi, tau, sc, Xc, u, z, ak, eos, o)
            for k in 1:NZ
                Tk = T[i, j, k]
                P = Pc[k]; rho = rc[k]; ne = nc[k]
                est[i, j, k] = sc[k]
                F["Pgas"][i, j, k] = P
                F["rho"][i, j, k] = rho
                F["ne"][i, j, k] = ne
                F["Pe"][i, j, k] = ne * K_B_WITT * Tk
                F["nH_total"][i, j, k] = rho / mph
                F["mu_bar"][i, j, k] = rho * K_B_WITT * Tk / (P * M_U_WITT)
                F["colmass"][i, j, k] = m[k] - mtop
                F["chi500"][i, j, k] = chi[k]
            end
            acc = 0.0
            F["colmass_trap"][i, j, 1] = 0.0
            for k in 1:NZ-1
                acc += 0.5 * (F["rho"][i, j, k] + F["rho"][i, j, k+1]) * (z[k] - z[k+1])
                F["colmass_trap"][i, j, k+1] = acc
            end
            for k in 1:NZ
                F["tau500"][i, j, k] = tau[k]
                F["logtau500"][i, j, k] = log10(tau[k])
            end
        end
    end
    nfail = global_count(par, count(f -> f & COL_EOS_FAIL != 0, cst))
    st = nfail > 0 ? eos_failure : converged
    msgs = nfail > 0 ? ["$nfail columns left the EOS table envelope (T or P); rebuild the table with a wider envelope (gate0 T range, eos.logP_range)"] : String[]
    stats = Dict{String,Any}(
        "columns_eos_failed" => nfail,
        "columns_rk_refined" => global_count(par, count(f -> f & COL_RK_REFINED != 0, cst)),
        "columns_rk_tol_not_met" => global_count(par, count(f -> f & COL_RK_TOL_NOT_MET != 0, cst)),
        "max_substeps" => Int(global_max(par, maximum(nsub))),
        "tau_exp_tail_columns" => global_count(par, count(==(TAU_EXP_TAIL), ttm)),
        "tau_half_cell_columns" => global_count(par, count(==(TAU_HALF_CELL), ttm)),
        "tau_zero_columns" => global_count(par, count(==(TAU_ZERO), ttm)))
    return ColumnHSEResult(F, est, cst, ttm, nsub, zt1, st, msgs, stats)
end

"""Force residual of the column solution with B = 0: R = (∂ₓP, ∂ᵧP, ∂zP + ρg)."""
function column_residual(par::ParCtx, ops::Ops, P::Field, rho::Field, g)
    exchange_halos!(par, [P])
    Rx = newfield(par); Ry = newfield(par); Rz = newfield(par)
    b = par.block
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            Rx[i, j, k] = dx(P, ops.X, i, j, k)
            Ry[i, j, k] = dy(P, ops.Y, i, j, k)
            Rz[i, j, k] = dz(P, ops.Z, i, j, k) + rho[i, j, k] * g
        end
    end
    return Rx, Ry, Rz
end
