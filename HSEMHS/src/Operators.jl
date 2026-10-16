# Discrete operators (implementation doc §6.2–6.7).
#
# `q` arguments of the transposed (gather) operators may be arrays or callables
# `q(i, j, k)`, so adjoint kernels can form products such as B × a on the fly.
# Transposes need `q` valid in the x/y halo.

@inline _q(q::AbstractArray, i, j, k) = @inbounds q[i, j, k]
@inline _q(q::Function, i, j, k) = q(i, j, k)

# ---- first derivatives -----------------------------------------------------------------
@inline dx(f, X::LocalHAxis, i, j, k) =
    @inbounds X.c1[i, 1] * _q(f, i - 1, j, k) + X.c1[i, 2] * _q(f, i, j, k) + X.c1[i, 3] * _q(f, i + 1, j, k)
@inline dy(f, Y::LocalHAxis, i, j, k) =
    @inbounds Y.c1[j, 1] * _q(f, i, j - 1, k) + Y.c1[j, 2] * _q(f, i, j, k) + Y.c1[j, 3] * _q(f, i, j + 1, k)
@inline dz(f, Z::LocalZAxis, i, j, k) =
    @inbounds Z.c1[k, 1] * _q(f, i, j, Z.idx[k, 1]) + Z.c1[k, 2] * _q(f, i, j, Z.idx[k, 2]) +
              Z.c1[k, 3] * _q(f, i, j, Z.idx[k, 3])

# ---- second derivatives ----------------------------------------------------------------
@inline d2x(f, X::LocalHAxis, i, j, k) =
    @inbounds X.c2[i, 1] * _q(f, i - 1, j, k) + X.c2[i, 2] * _q(f, i, j, k) + X.c2[i, 3] * _q(f, i + 1, j, k)
@inline d2y(f, Y::LocalHAxis, i, j, k) =
    @inbounds Y.c2[j, 1] * _q(f, i, j - 1, k) + Y.c2[j, 2] * _q(f, i, j, k) + Y.c2[j, 3] * _q(f, i, j + 1, k)
@inline d2z(f, Z::LocalZAxis, i, j, k) =
    @inbounds Z.c2[k, 1] * _q(f, i, j, Z.idx[k, 1]) + Z.c2[k, 2] * _q(f, i, j, Z.idx[k, 2]) +
              Z.c2[k, 3] * _q(f, i, j, Z.idx[k, 3])

# ---- transposes, gather form -----------------------------------------------------------
# (Dᵀq)_i = c[i+1,1] q_{i+1} + c[i,2] q_i + c[i-1,3] q_{i-1}   (periodic x, y)
@inline dxT(q, X::LocalHAxis, i, j, k) =
    @inbounds X.c1[i+1, 1] * _q(q, i + 1, j, k) + X.c1[i, 2] * _q(q, i, j, k) + X.c1[i-1, 3] * _q(q, i - 1, j, k)
@inline dyT(q, Y::LocalHAxis, i, j, k) =
    @inbounds Y.c1[j+1, 1] * _q(q, i, j + 1, k) + Y.c1[j, 2] * _q(q, i, j, k) + Y.c1[j-1, 3] * _q(q, i, j - 1, k)
@inline d2xT(q, X::LocalHAxis, i, j, k) =
    @inbounds X.c2[i+1, 1] * _q(q, i + 1, j, k) + X.c2[i, 2] * _q(q, i, j, k) + X.c2[i-1, 3] * _q(q, i - 1, j, k)
@inline d2yT(q, Y::LocalHAxis, i, j, k) =
    @inbounds Y.c2[j+1, 1] * _q(q, i, j + 1, k) + Y.c2[j, 2] * _q(q, i, j, k) + Y.c2[j-1, 3] * _q(q, i, j - 1, k)
@inline function dzT(q, Z::LocalZAxis, i, j, k)
    s = 0.0
    @inbounds for m in axes(Z.tidx, 2)
        s += Z.tc1[k, m] * _q(q, i, j, Z.tidx[k, m])
    end
    return s
end
@inline function d2zT(q, Z::LocalZAxis, i, j, k)
    s = 0.0
    @inbounds for m in axes(Z.tidx, 2)
        s += Z.tc2[k, m] * _q(q, i, j, Z.tidx[k, m])
    end
    return s
end

# ---- vector operators ------------------------------------------------------------------
"""Curl of (Bx, By, Bz) at an owned node (needs B halos)."""
@inline function curl(Bx, By, Bz, o::Ops, i, j, k)
    cx = dy(Bz, o.Y, i, j, k) - dz(By, o.Z, i, j, k)
    cy = dz(Bx, o.Z, i, j, k) - dx(Bz, o.X, i, j, k)
    cz = dx(By, o.X, i, j, k) - dy(Bx, o.Y, i, j, k)
    return cx, cy, cz
end

@inline divergence(Bx, By, Bz, o::Ops, i, j, k) =
    dx(Bx, o.X, i, j, k) + dy(By, o.Y, i, j, k) + dz(Bz, o.Z, i, j, k)

@inline cross(a1, a2, a3, b1, b2, b3) = (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

"""Lorentz force L = (curl B / μ0) × B at an owned node."""
@inline function lorentz(Bx, By, Bz, o::Ops, i, j, k)
    cx, cy, cz = curl(Bx, By, Bz, o, i, j, k)
    @inbounds bx, by, bz = Bx[i, j, k], By[i, j, k], Bz[i, j, k]
    return cross(cx / MU0, cy / MU0, cz / MU0, bx, by, bz)
end

# ---- whole-field helpers (tests, diagnostics) ------------------------------------------
"""Apply a pointwise stencil function `op(f, axis, i, j, k)` to all owned nodes of `out`."""
function apply!(out, op, f, ax, par::ParCtx)
    b = par.block
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j = (jk - 1) % b.ny + 1
        k = (jk - 1) ÷ b.ny + 1
        @inbounds for i in 1:b.nx
            out[i, j, k] = op(f, ax, i, j, k)
        end
    end
    return out
end
