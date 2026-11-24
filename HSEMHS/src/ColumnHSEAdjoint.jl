# Adjoint of the height-grid column HSE (design/adjoint/02): the RK4 integration of `hse_column!` with the substep count
# of every interval frozen (the solution `rk4_adaptive` returned), in the order the forward took the intervals.

"""Intervals of one `hse_column!` column in forward order: nodes (ka → kb), start height zs, the node whose ln P is the
input (0: ln P_ref) and the substeps of the returned solution."""
struct RK4Tape
    ka::Vector{Int32}
    kb::Vector{Int32}
    zs::Vector{Float64}
    src::Vector{Int32}
    n::Vector{Int32}
end
RK4Tape() = RK4Tape(Int32[], Int32[], Float64[], Int32[], Int32[])
Base.empty!(t::RK4Tape) = (foreach(empty!, (t.ka, t.kb, t.zs, t.src, t.n)); t)
record!(t::RK4Tape, ka, kb, zs, src, n) = (push!(t.ka, ka); push!(t.kb, kb); push!(t.zs, zs); push!(t.src, src); push!(t.n, n); t)

"""Anchor node of an `HSEOpts` (0 for :height)."""
anchor_node(o::HSEOpts, NZ) = o.anchor == :bottom ? NZ : o.anchor == :top ? 1 : 0

"""
    hse_column_replay!(u, X, z, eos, o, tape) -> u

`hse_column!`'s ln P with the recorded intervals and substep counts: the same arithmetic (bit for bit) as the recorded
solve, as a function of X alone.
"""
function hse_column_replay!(u::AbstractVector, X::AbstractVector, z::AbstractVector, eos, o::HSEOpts, tape::RK4Tape)
    lp = log(o.P_ref); ak = anchor_node(o, length(z))
    ak > 0 && (u[ak] = lp)
    @inbounds for e in eachindex(tape.ka)
        ka = tape.ka[e]; kb = tape.kb[e]; src = tape.src[e]
        u[kb] = rk4_seg(eos, z[ka], X[ka], z[kb], X[kb], tape.zs[e], z[kb], src == 0 ? lp : u[src], 0.0, Int(tape.n[e]), o.g)[1]
    end
    return u
end

"""Weight of Xb in `xlin(za, Xa, zb, Xb, z)`."""
@inline xlin_w(za, zb, z) = z == za ? 0.0 : z == zb ? 1.0 : (z - za) / (zb - za)

"""k = −g e^{ln ρ − u} and its X and u partials."""
@inline function rk4_k(eos, X, u, g)
    r, _, _, rX, rY, _, _, _, _, _ = state_d(eos, X, u)
    k = -g * exp(r - u)
    return k, k * rX, k * (rY - 1)
end

"""Adjoint of `rk4_seg`'s u for the cotangent ū of its output: returns (ū of the input, X̄a, X̄b)."""
function rk4_seg_vjp(eos, za, Xa, zb, Xb, zs, ze, u0, n::Int, g, ub)
    h = (ze - zs) / n
    us = zeros(n)                                       # u at the start of each substep (forward pass)
    u = u0
    @inbounds for s in 0:n-1
        us[s+1] = u
        z0 = zs + s * h; z1 = s == n - 1 ? ze : zs + (s + 1) * h; zh = z0 + 0.5 * (z1 - z0); hs = z1 - z0
        X0 = xlin(za, Xa, zb, Xb, z0); Xh = xlin(za, Xa, zb, Xb, zh); X1 = xlin(za, Xa, zb, Xb, z1)
        k1 = rk4_k(eos, X0, u, g)[1]; u2 = u + 0.5hs * k1
        k2 = rk4_k(eos, Xh, u2, g)[1]; u3 = u + 0.5hs * k2
        k3 = rk4_k(eos, Xh, u3, g)[1]; u4 = u + hs * k3
        k4 = rk4_k(eos, X1, u4, g)[1]
        u += hs / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    Xab = 0.0; Xbb = 0.0
    @inbounds for s in n-1:-1:0
        u = us[s+1]
        z0 = zs + s * h; z1 = s == n - 1 ? ze : zs + (s + 1) * h; zh = z0 + 0.5 * (z1 - z0); hs = z1 - z0
        X0 = xlin(za, Xa, zb, Xb, z0); Xh = xlin(za, Xa, zb, Xb, zh); X1 = xlin(za, Xa, zb, Xb, z1)
        k1, k1X, k1u = rk4_k(eos, X0, u, g); u2 = u + 0.5hs * k1
        k2, k2X, k2u = rk4_k(eos, Xh, u2, g); u3 = u + 0.5hs * k2
        k3, k3X, k3u = rk4_k(eos, Xh, u3, g); u4 = u + hs * k3
        _, k4X, k4u = rk4_k(eos, X1, u4, g)
        b1 = hs / 6 * ub; b2 = hs / 3 * ub; b3 = hs / 3 * ub; b4 = hs / 6 * ub
        X1b = b4 * k4X; u4b = b4 * k4u
        b3 += hs * u4b
        Xhb = b3 * k3X; u3b = b3 * k3u
        b2 += 0.5hs * u3b
        Xhb += b2 * k2X; u2b = b2 * k2u
        b1 += 0.5hs * u2b
        X0b = b1 * k1X
        ub = ub + u4b + u3b + u2b + b1 * k1u
        w0 = xlin_w(za, zb, z0); wh = xlin_w(za, zb, zh); w1 = xlin_w(za, zb, z1)
        Xab += X0b * (1 - w0) + Xhb * (1 - wh) + X1b * (1 - w1)
        Xbb += X0b * w0 + Xhb * wh + X1b * w1
    end
    return ub, Xab, Xbb
end

"""
    hse_column_vjp!(Xbar, ubar, X, z, eos, o, tape) -> Xbar

Adjoint of `hse_column_replay!` for the cotangent ū of ln P: adds ∂ℓ/∂(ln T) to `Xbar` (ln P_ref is a constant).
"""
function hse_column_vjp!(Xbar::AbstractVector, ubar::AbstractVector, X::AbstractVector, z::AbstractVector, eos, o::HSEOpts,
                         tape::RK4Tape)
    NZ = length(z)
    u = fill(NaN, NZ); hse_column_replay!(u, X, z, eos, o, tape)
    lp = log(o.P_ref)
    ub = collect(Float64, ubar)
    @inbounds for e in reverse(eachindex(tape.ka))
        ka = tape.ka[e]; kb = tape.kb[e]; src = tape.src[e]
        b = ub[kb]; ub[kb] = 0.0
        b == 0 && continue
        uin, Xab, Xbb = rk4_seg_vjp(eos, z[ka], X[ka], z[kb], X[kb], tape.zs[e], z[kb], src == 0 ? lp : u[src],
                                    Int(tape.n[e]), o.g, b)
        Xbar[ka] += Xab; Xbar[kb] += Xbb
        src == 0 || (ub[src] += uin)
    end
    return Xbar
end
