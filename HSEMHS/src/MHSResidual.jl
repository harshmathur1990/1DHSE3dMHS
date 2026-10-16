# MHS objective and its adjoint gradient (spec §6.1–6.3, implementation doc §13.2–13.5).
#
# Optimizer variables x = (v, b̃x, b̃y, b̃z):
#   u   = v − v̄ + ln P_ref,  v̄ = Σ α_ij v[i, j, k_a]   (scalar gauge on the anchor plane, D3)
#   B_c = Bobs_c + σ_c b̃_c                              (E4)
# Objective:
#   Φ_F = (λ_F/N) Σ w |R|² / S²,   R = ∇P + ρgẑ − (∇×B/μ₀)×B
#   Φ_D = (λ_D/N) Σ w h² (∇·B)² / Bs²
#   Φ_B = (1/N_B) Σ b̃²
#   Φ_S = (λ_S/N) Σ w Σ_a Σ_c (h_a² ∂²_a δB_c)² / σ_c²
#   Φ_U = (λ_U/N) Σ w Σ_{a=x,y} (h_a² ∂²_a u)²

const NTERMS = 5        # F, D, B, S, U

mutable struct MHSProblem{E}
    par::ParCtx
    grid::Grid
    ops::Ops
    eos::E
    X::Field                        # ln T (fixed, D4)
    Bobs::NTuple{3,Field}           # halos valid
    sig::NTuple{3,Field}            # σ per component (halos valid), floored at 1e-12 T
    invBs2::Field                   # 1 / max(|Bobs|, 1 G)²
    invS2::Field                    # 1 / S², fixed for the solve
    alpha::Matrix{Float64}          # (nx, ny) anchor-plane area weights, Σ α = 1 globally
    ka::Int                         # anchor-plane index (NZ bottom, 1 top)
    lnPref::Float64
    g::Float64
    lamF::Float64
    lamD::Float64
    lamS::Float64
    lamU::Float64
    N::Float64
    NB::Float64
end

struct MHSWork
    u::Field; P::Field; rho::Field; rhou::Field
    B::NTuple{3,Field}; dB::NTuple{3,Field}
    aR::NTuple{3,Field}; aD::Field
    qS::NTuple{9,Field}             # index (a−1)·3 + c, a = x,y,z axis, c = component
    qU::NTuple{2,Field}
    gu::Field
    partials::Array{Float64,3}      # (NTERMS + 1, ny, nz); last slot counts out-of-envelope nodes
end

MHSWork(par::ParCtx) = MHSWork(newfield(par), newfield(par), newfield(par), newfield(par),
    ntuple(_ -> newfield(par), 3), ntuple(_ -> newfield(par), 3), ntuple(_ -> newfield(par), 3),
    newfield(par), ntuple(_ -> newfield(par), 9), ntuple(_ -> newfield(par), 2), newfield(par),
    zeros(NTERMS + 1, par.block.ny, par.block.nz))

"""Optimizer state: four haloed fields (v, b̃x, b̃y, b̃z)."""
const MHSState = Vector{Field}
newstate(par::ParCtx) = Field[newfield(par) for _ in 1:4]

"""
    MHSProblem(par, grid, T, Bobs, sigma, eos; anchor_plane, P_ref, g, lambda_F, ...)

`T`, `Bobs`, `sigma` are local fields (owned values set). S is set later by `set_scales!`.
"""
function MHSProblem(par::ParCtx, grid::Grid, T::Field, Bobs::NTuple{3,Field}, sigma::NTuple{3,Field}, eos;
                    anchor_plane::Symbol = :bottom, P_ref::Float64, g::Float64 = 274.0,
                    lambda_F::Float64 = 1.0, lambda_D::Float64 = 1.0, lambda_S::Float64 = 1e-3,
                    lambda_U::Float64 = 1e-6)
    ops = Ops(grid, par)
    b = par.block
    X = newfield(par); fill_owned!((i, j, k) -> log(T[i, j, k]), X, par)
    sig = ntuple(c -> (s = newfield(par); fill_owned!((i, j, k) -> max(sigma[c][i, j, k], 1e-12), s, par); s), 3)
    exchange_halos!(par, [Bobs..., sig...])        # invBs2 / invS2 halos: see set_scales!
    invBs2 = newfield(par)
    fill_owned!((i, j, k) -> 1 / max(Bobs[1][i, j, k]^2 + Bobs[2][i, j, k]^2 + Bobs[3][i, j, k]^2, ONE_GAUSS^2), invBs2, par)
    ka = anchor_plane == :bottom ? b.nz : anchor_plane == :top ? 1 : invalid("anchor_plane must be bottom or top")
    hx = [ops.X.h[i] for i in 1:b.nx]; hy = [ops.Y.h[j] for j in 1:b.ny]
    Atot = sum(grid.x.h) * sum(grid.y.h)
    alpha = [hx[i] * hy[j] / Atot for i in 1:b.nx, j in 1:b.ny]
    N = Float64(par.NX) * par.NY * par.NZ
    return MHSProblem(par, grid, ops, eos, X, Bobs, sig, invBs2, newfield(par; fill = 1.0), alpha, ka,
                      log(P_ref), g, lambda_F, lambda_D, lambda_S, lambda_U, N, 3N)
end

"""S = max(ρ₀g, 0.01·median_k(ρ₀g)) from the initial pressure u0 (fixed for the solve)."""
function set_scales!(pb::MHSProblem, u0::Field)
    par = pb.par; b = par.block
    rg = newfield(par)
    fill_owned!((i, j, k) -> exp(lnrho(pb.eos, pb.X[i, j, k], u0[i, j, k])[1]) * pb.g, rg, par)
    med = layer_quantiles(par, rg, [0.5])[:, 1]
    fill_owned!((i, j, k) -> 1 / max(rg[i, j, k], 0.01 * med[k])^2, pb.invS2, par)
    exchange_halos!(par, [pb.invS2, pb.invBs2])    # neighbours' weights enter the GN diagonal
    return pb
end

"""Initial state from a pressure u0 (e.g. column_hse) and B = Bobs."""
function init_state(pb::MHSProblem, u0::Field)
    x = newstate(pb.par)
    fill_owned!((i, j, k) -> u0[i, j, k], x[1], pb.par)
    return x
end

"""Deterministic global Σ α v on the anchor plane."""
function anchor_mean(pb::MHSProblem, v::Field)
    b = pb.par.block
    s = 0.0
    @inbounds for j in 1:b.ny, i in 1:b.nx
        s += pb.alpha[i, j] * v[i, j, pb.ka]
    end
    return MPI.Allreduce(s, +, pb.par.comm)
end

@inline rowjk(jk, ny) = ((jk - 1) % ny + 1, (jk - 1) ÷ ny + 1)

"""
    objective!(pb, w, x) -> (Φ, terms::NTuple{5}, oob::Bool)

Forward pass; leaves in `w` everything the adjoint needs (halos of the sources are
exchanged by `gradient!`).
"""
function objective!(pb::MHSProblem, w::MHSWork, x::MHSState)
    par = pb.par; o = pb.ops; b = par.block
    v, bx, by, bz = x
    vbar = anchor_mean(pb, v)
    bt = (bx, by, bz)
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j, k = rowjk(jk, b.ny)
        @inbounds for i in 1:b.nx
            w.u[i, j, k] = v[i, j, k] - vbar + pb.lnPref
            for c in 1:3
                d = pb.sig[c][i, j, k] * bt[c][i, j, k]
                w.dB[c][i, j, k] = d
                w.B[c][i, j, k] = pb.Bobs[c][i, j, k] + d
            end
        end
    end
    exchange_halos!(par, [w.u, w.B..., w.dB...])
    u = w.u; P = w.P
    Threads.@threads :static for k in 1:b.nz
        @inbounds for j in axes(u, 2), i in axes(u, 1)
            P[i, j, k] = exp(u[i, j, k])
        end
    end
    Bx, By, Bz = w.B
    g = pb.g; lF, lD, lS, lU, N = pb.lamF, pb.lamD, pb.lamS, pb.lamU, pb.N
    pt = w.partials
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j, k = rowjk(jk, b.ny)
        eF = 0.0; eD = 0.0; eB = 0.0; eS = 0.0; eU = 0.0; noob = 0.0
        hy = o.Y.h[j]; hz = o.Z.h[k]
        @inbounds for i in 1:b.nx
            lr, lu, st = lnrho_dY(pb.eos, pb.X[i, j, k], u[i, j, k])
            if st & EOS_OUT_OF_ENVELOPE != 0
                noob += 1; lr = 0.0; lu = 0.0
            end
            rho = exp(lr)
            w.rho[i, j, k] = rho; w.rhou[i, j, k] = lu
            gx = dx(P, o.X, i, j, k); gy = dy(P, o.Y, i, j, k); gz = dz(P, o.Z, i, j, k)
            cx, cy, cz = curl(Bx, By, Bz, o, i, j, k)
            Lx, Ly, Lz = cross(cx / MU0, cy / MU0, cz / MU0, Bx[i, j, k], By[i, j, k], Bz[i, j, k])
            Rx = gx - Lx; Ry = gy - Ly; Rz = gz + rho * g - Lz
            dv = divergence(Bx, By, Bz, o, i, j, k)
            wn = weight(o, i, j, k); hm = hmean(o, i, j, k); hx = o.X.h[i]
            is2 = pb.invS2[i, j, k]; ib2 = pb.invBs2[i, j, k]
            eF += wn * (Rx^2 + Ry^2 + Rz^2) * is2
            eD += wn * hm^2 * dv^2 * ib2
            eB += bt[1][i, j, k]^2 + bt[2][i, j, k]^2 + bt[3][i, j, k]^2
            aF = 2lF * wn * is2 / N
            w.aR[1][i, j, k] = aF * Rx; w.aR[2][i, j, k] = aF * Ry; w.aR[3][i, j, k] = aF * Rz
            w.aD[i, j, k] = 2lD * wn * hm^2 * ib2 * dv / N
            for c in 1:3
                s2 = pb.sig[c][i, j, k]^2
                ddx = d2x(w.dB[c], o.X, i, j, k); ddy = d2y(w.dB[c], o.Y, i, j, k); ddz = d2z(w.dB[c], o.Z, i, j, k)
                eS += wn * ((hx^2 * ddx)^2 + (hy^2 * ddy)^2 + (hz^2 * ddz)^2) / s2
                f = 2lS * wn / (N * s2)
                w.qS[c][i, j, k] = f * hx^4 * ddx
                w.qS[3+c][i, j, k] = f * hy^4 * ddy
                w.qS[6+c][i, j, k] = f * hz^4 * ddz
            end
            uxx = d2x(u, o.X, i, j, k); uyy = d2y(u, o.Y, i, j, k)
            eU += wn * ((hx^2 * uxx)^2 + (hy^2 * uyy)^2)
            w.qU[1][i, j, k] = 2lU * wn * hx^4 * uxx / N
            w.qU[2][i, j, k] = 2lU * wn * hy^4 * uyy / N
        end
        pt[1, j, k] = eF; pt[2, j, k] = eD; pt[3, j, k] = eB; pt[4, j, k] = eS; pt[5, j, k] = eU
        pt[6, j, k] = noob
    end
    s = reduce_rows(par, pt)
    terms = (lF * s[1] / N, lD * s[2] / N, s[3] / pb.NB, lS * s[4] / N, lU * s[5] / N)
    oob = s[6] > 0
    return oob ? Inf : sum(terms), terms, oob
end

"""
    gradient!(gx, pb, w, x)

∇Φ with respect to x = (v, b̃x, b̃y, b̃z), written to the owned part of `gx`. Must follow
`objective!(pb, w, x)` at the same x.
"""
function gradient!(gx::MHSState, pb::MHSProblem, w::MHSWork, x::MHSState)
    par = pb.par; o = pb.ops; b = par.block
    exchange_halos!(par, [w.aR..., w.aD, w.qS..., w.qU...])
    Bx, By, Bz = w.B
    aRx, aRy, aRz = w.aR
    # c = B × aR, evaluated on the fly at referenced neighbours
    cxf = (i, j, k) -> @inbounds By[i, j, k] * aRz[i, j, k] - Bz[i, j, k] * aRy[i, j, k]
    cyf = (i, j, k) -> @inbounds Bz[i, j, k] * aRx[i, j, k] - Bx[i, j, k] * aRz[i, j, k]
    czf = (i, j, k) -> @inbounds Bx[i, j, k] * aRy[i, j, k] - By[i, j, k] * aRx[i, j, k]
    g = pb.g; NB = pb.NB
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j, k = rowjk(jk, b.ny)
        @inbounds for i in 1:b.nx
            gu = w.P[i, j, k] * (dxT(aRx, o.X, i, j, k) + dyT(aRy, o.Y, i, j, k) + dzT(aRz, o.Z, i, j, k)) +
                 aRz[i, j, k] * g * w.rho[i, j, k] * w.rhou[i, j, k] +
                 d2xT(w.qU[1], o.X, i, j, k) + d2yT(w.qU[2], o.Y, i, j, k)
            w.gu[i, j, k] = gu
            cx, cy, cz = curl(Bx, By, Bz, o, i, j, k)
            Jx, Jy, Jz = cx / MU0, cy / MU0, cz / MU0
            ax, ay, az = aRx[i, j, k], aRy[i, j, k], aRz[i, j, k]
            pJx, pJy, pJz = cross(ax, ay, az, Jx, Jy, Jz)
            gB1 = -pJx - (dzT(cyf, o.Z, i, j, k) - dyT(czf, o.Y, i, j, k)) / MU0 + dxT(w.aD, o.X, i, j, k)
            gB2 = -pJy - (dxT(czf, o.X, i, j, k) - dzT(cxf, o.Z, i, j, k)) / MU0 + dyT(w.aD, o.Y, i, j, k)
            gB3 = -pJz - (dyT(cxf, o.Y, i, j, k) - dxT(cyf, o.X, i, j, k)) / MU0 + dzT(w.aD, o.Z, i, j, k)
            gBs = (gB1, gB2, gB3)
            for c in 1:3
                gdB = d2xT(w.qS[c], o.X, i, j, k) + d2yT(w.qS[3+c], o.Y, i, j, k) + d2zT(w.qS[6+c], o.Z, i, j, k)
                gx[c+1][i, j, k] = pb.sig[c][i, j, k] * (gBs[c] + gdB) + 2x[c+1][i, j, k] / NB
            end
        end
    end
    # gauge chain: ∂u_n/∂v_m = δ_nm − α_m [m on the anchor plane]
    G = global_sum(par, w.gu)
    gv = gx[1]
    Threads.@threads :static for k in 1:b.nz
        @inbounds for j in 1:b.ny, i in 1:b.nx
            gv[i, j, k] = w.gu[i, j, k] - (k == pb.ka ? pb.alpha[i, j] * G : 0.0)
        end
    end
    return gx
end

"""Φ and ∇Φ at x (convenience for the optimizer)."""
function fg!(gx::MHSState, pb::MHSProblem, w::MHSWork, x::MHSState)
    f, terms, oob = objective!(pb, w, x)
    oob && return f, terms, oob
    gradient!(gx, pb, w, x)
    return f, terms, oob
end

"""
    gn_diagonal!(D, pb, w)

Gauss–Newton diagonal of ∇²Φ with respect to x = (v, b̃x, b̃y, b̃z) at the state of the last
`objective!` call (gauge coupling ignored). Exact for the pressure, divergence, data and
regularizer terms; for the Lorentz term it uses the upper bound |∂J×B|² ≤ |∂J|²|B|² and keeps
the diagonal J×e_c part. Used as the L-BFGS initial inverse Hessian (Jacobi preconditioner).
"""
function gn_diagonal!(D::MHSState, pb::MHSProblem, w::MHSWork)
    par = pb.par; o = pb.ops; b = par.block
    lF, lD, lS, lU, N = pb.lamF, pb.lamD, pb.lamS, pb.lamU, pb.N
    Bx, By, Bz = w.B
    A(i, j, k) = @inbounds lF * weight(o, i, j, k) * pb.invS2[i, j, k] / N
    AB2(i, j, k) = @inbounds A(i, j, k) * (Bx[i, j, k]^2 + By[i, j, k]^2 + Bz[i, j, k]^2) / MU0^2
    C(i, j, k) = @inbounds lD * weight(o, i, j, k) * hmean(o, i, j, k)^2 * pb.invBs2[i, j, k] / N
    Wx(i, j, k) = @inbounds lS * weight(o, i, j, k) * o.X.h[i]^4 / N
    Wy(i, j, k) = @inbounds lS * weight(o, i, j, k) * o.Y.h[j]^4 / N
    Wz(i, j, k) = @inbounds lS * weight(o, i, j, k) * o.Z.h[k]^4 / N
    Ux(i, j, k) = @inbounds lU * weight(o, i, j, k) * o.X.h[i]^4 / N
    Uy(i, j, k) = @inbounds lU * weight(o, i, j, k) * o.Y.h[j]^4 / N
    sc = pb.sig
    Es = ntuple(c -> ((i, j, k) -> Wx(i, j, k) / sc[c][i, j, k]^2, (i, j, k) -> Wy(i, j, k) / sc[c][i, j, k]^2,
                      (i, j, k) -> Wz(i, j, k) / sc[c][i, j, k]^2), 3)
    g = pb.g
    Threads.@threads :static for jk in 1:(b.ny*b.nz)
        j, k = rowjk(jk, b.ny)
        @inbounds for i in 1:b.nx
            P = w.P[i, j, k]; t = g * w.rho[i, j, k] * w.rhou[i, j, k]
            du = 2 * (P^2 * (dxT2(A, o.X, i, j, k) + dyT2(A, o.Y, i, j, k) + dzT2(A, o.Z, i, j, k)) +
                      A(i, j, k) * (t^2 + 2P * dz_self(o.Z, k) * t)) +
                 2 * (d2xT2(Ux, o.X, i, j, k) + d2yT2(Uy, o.Y, i, j, k))
            D[1][i, j, k] = abs(du)
            cx, cy, cz = curl(Bx, By, Bz, o, i, j, k)
            J2 = (cx^2 + cy^2 + cz^2) / MU0^2
            ax = dxT2(AB2, o.X, i, j, k); ay = dyT2(AB2, o.Y, i, j, k); az = dzT2(AB2, o.Z, i, j, k)
            cxx = dxT2(C, o.X, i, j, k); cyy = dyT2(C, o.Y, i, j, k); czz = dzT2(C, o.Z, i, j, k)
            lor = (ay + az, ax + az, ax + ay)              # Bx enters J via Dz, Dy; By via Dz, Dx; Bz via Dy, Dx
            dvv = (cxx, cyy, czz)
            for c in 1:3
                s2 = sc[c][i, j, k]^2
                reg = d2xT2(Es[c][1], o.X, i, j, k) + d2yT2(Es[c][2], o.Y, i, j, k) + d2zT2(Es[c][3], o.Z, i, j, k)
                D[c+1][i, j, k] = 2s2 * (lor[c] + A(i, j, k) * J2 + dvv[c] + reg) + 2 / pb.NB
            end
        end
    end
    for f in 1:4
        Df = D[f]
        mx = global_max(par, maximum(Df[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
        fill_owned!((i, j, k) -> max(Df[i, j, k], 1e-12 * mx, 1e-300), Df, par)
    end
    return D
end
