# MHS objective and adjoint on the staggered vector-potential mesh
# (HSE_MHS_D9_STAGGERED_ADDENDUM.md §3–7).
#
# State x = (v, Ãx, Ãy, Ãz, b̄):  u = v − v̄ + ln P_ref (centres);  A = A_ref Ã (edges);
# B = curl A + b̄ ẑ (faces).  b̄ is a scalar stored at [1,1,1] of a 1-level field on rank 0
# only (zero elsewhere), so that sums over ranks give its value once.
#
# Φ = Φ_F + Φ_B + Φ_S + Φ_U   (terms reported as (F, 0, B, S, U) to match the collocated layout)
#   Φ_F = (λ_F/N_F) Σ_faces w_f |R_f|²/S_f²   on x-, y- and interior z-faces
#   Φ_B = (1/N_B) Σ ((HB)_c − Bobs_c)²/σ_c²   at centres
#   Φ_S = (λ_S/N) Σ w Σ_a Σ_c (h_a² ∂²_a δB_c)²/σ_c²,  δB = HB − Bobs
#   Φ_U = (λ_U/N) Σ w Σ_{a=x,y} (h_a² ∂²_a u)²

mutable struct SMHSProblem{E}
    par::ParCtx
    grid::Grid
    s::StagOps
    ops::Ops
    eos::E
    X::Field
    Bobs::NTuple{3,Field}
    sig::NTuple{3,Field}
    invS2f::NTuple{3,Field}           # x-faces, y-faces (1:NZ), z-faces (0:NZ, interior used)
    alpha::Matrix{Float64}
    ka::Int
    lnPref::Float64
    g::Float64
    lamF::Float64
    lamS::Float64
    lamU::Float64
    Aref::Float64
    N::Float64
    NF::Float64
    NB::Float64
    wmean::Float64
end

newscalar(par::ParCtx) = OffsetArray(zeros(par.block.nx + 2, par.block.ny + 2, 1), 0:par.block.nx+1, 0:par.block.ny+1, 1:1)
"""Value of the distributed scalar (stored on rank 0 only)."""
getscalar(par::ParCtx, f::Field) = MPI.Allreduce(f[1, 1, 1], +, par.comm)
function setscalar!(par::ParCtx, f::Field, v)
    fill!(parent(f), 0.0)
    rank0(par) && (f[1, 1, 1] = v)
    return f
end

"""State layout for the staggered problem: (v, Ãx, Ãy, Ãz, b̄)."""
snewstate(par::ParCtx) = Field[newfield(par), newfield0(par), newfield0(par), newfield(par), newscalar(par)]

struct SMHSWork
    u::Field; P::Field; rho::Field; rhou::Field
    A::NTuple{3,Field}; B::NTuple{3,Field}; J::NTuple{3,Field}
    HB::NTuple{3,Field}; dB::NTuple{3,Field}
    aR::NTuple{3,Field}
    fX::NTuple{4,Field}               # x-faces: ∂Φ/∂(Jy_f, Jz_f, Bz_f, By_f)
    fY::NTuple{4,Field}               # y-faces: ∂Φ/∂(Jz_f, Jx_f, Bx_f, Bz_f)
    fZ::NTuple{4,Field}               # z-faces: ∂Φ/∂(Jx_f, Jy_f, By_f, Bx_f)
    qS::NTuple{9,Field}; qU::NTuple{2,Field}
    gHB::NTuple{3,Field}; gB::NTuple{3,Field}; gBJ::NTuple{3,Field}; gJ::NTuple{3,Field}; gA::NTuple{3,Field}
    gP::Field
end

function SMHSWork(par::ParCtx)
    c() = newfield(par); z() = newfield0(par)
    edges() = (z(), z(), c()); faces() = (c(), c(), z())
    return SMHSWork(c(), c(), c(), c(), edges(), faces(), edges(), (c(), c(), c()), (c(), c(), c()), faces(),
                    ntuple(_ -> c(), 4), ntuple(_ -> c(), 4), ntuple(_ -> z(), 4), ntuple(_ -> c(), 9), ntuple(_ -> c(), 2),
                    (c(), c(), c()), faces(), faces(), edges(), edges(), c())
end

"""
    SMHSProblem(par, grid, T, Bobs, sigma, eos; anchor_plane, P_ref, g, lambda_F, lambda_S, lambda_U)

Bobs, sigma at centres. S is set by `set_scales!` (from an initial u).
"""
function SMHSProblem(par::ParCtx, grid::Grid, T::Field, Bobs::NTuple{3,Field}, sigma::NTuple{3,Field}, eos;
                     anchor_plane::Symbol = :bottom, P_ref::Float64, g::Float64 = 274.0,
                     lambda_F::Float64 = 1.0, lambda_S::Float64 = 1e-3, lambda_U::Float64 = 0.0)
    ops = Ops(grid, par); s = StagOps(grid, par); b = par.block
    X = newfield(par); fill_owned!((i, j, k) -> log(T[i, j, k]), X, par)
    sig = ntuple(c -> (q = newfield(par); fill_owned!((i, j, k) -> max(sigma[c][i, j, k], 1e-12), q, par); q), 3)
    exchange_halos!(par, [Bobs..., sig...])
    ka = anchor_plane == :bottom ? b.nz : anchor_plane == :top ? 1 : invalid("anchor_plane must be bottom or top")
    Atot = sum(grid.x.h) * sum(grid.y.h)
    alpha = [ops.X.h[i] * ops.Y.h[j] / Atot for i in 1:b.nx, j in 1:b.ny]
    NX, NY, NZ = par.NX, par.NY, par.NZ
    N = Float64(NX) * NY * NZ
    hzs = [s.zf[k-1] - s.zf[k] for k in 1:NZ]
    wmean = sum(grid.x.h) * sum(grid.y.h) * sum(hzs) / N
    smed = layer_quantiles(par, sig[1], [0.5])[:, 1]
    Aref = median(smed) * 0.5 * (sum(grid.x.h) / NX + sum(grid.y.h) / NY)
    invS2f = (newfield(par; fill = 1.0), newfield(par; fill = 1.0), newfield0(par; fill = 1.0))
    return SMHSProblem(par, grid, s, ops, eos, X, Bobs, sig, invS2f, alpha, ka, log(P_ref), g,
                       lambda_F, lambda_S, lambda_U, Aref, N, Float64(NX) * NY * (3NZ - 1), 3N, wmean)
end

@inline wxf(pb::SMHSProblem, i, j, k) = @inbounds pb.s.dxf[i] * pb.s.dyc[j] * pb.s.hz[k] / pb.wmean
@inline wyf(pb::SMHSProblem, i, j, k) = @inbounds pb.s.dxc[i] * pb.s.dyf[j] * pb.s.hz[k] / pb.wmean
@inline wzf(pb::SMHSProblem, i, j, k) = @inbounds pb.s.dxc[i] * pb.s.dyc[j] * pb.s.dzf[k] / pb.wmean
@inline wcc(pb::SMHSProblem, i, j, k) = @inbounds pb.s.dxc[i] * pb.s.dyc[j] * pb.s.hz[k] / pb.wmean

"""S at centres = max(ρ₀g, 0.01 median_k(ρ₀g)); S_f² = S_a S_b on faces (fixed for the solve)."""
function set_scales!(pb::SMHSProblem, u0::Field)
    par = pb.par; b = par.block; NZ = b.nz
    S = newfield(par)
    rg = newfield(par)
    fill_owned!((i, j, k) -> exp(lnrho(pb.eos, pb.X[i, j, k], u0[i, j, k])[1]) * pb.g, rg, par)
    med = layer_quantiles(par, rg, [0.5])[:, 1]
    fill_owned!((i, j, k) -> max(rg[i, j, k], 0.01 * med[k]), S, par)
    exchange_halos!(par, [S])
    fill_owned!((i, j, k) -> 1 / (S[i, j, k] * S[i+1, j, k]), pb.invS2f[1], par)
    fill_owned!((i, j, k) -> 1 / (S[i, j, k] * S[i, j+1, k]), pb.invS2f[2], par)
    owned_map!((i, j, k) -> (1 <= k <= NZ - 1) ? 1 / (S[i, j, k] * S[i, j, k+1]) : 1.0, pb.invS2f[3], par)
    exchange_halos!(par, collect(pb.invS2f))
    return pb
end

function anchor_mean(pb::SMHSProblem, v::Field)
    b = pb.par.block
    s = 0.0
    @inbounds for j in 1:b.ny, i in 1:b.nx
        s += pb.alpha[i, j] * v[i, j, pb.ka]
    end
    return MPI.Allreduce(s, +, pb.par.comm)
end

# ---------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------
function objective!(pb::SMHSProblem, w::SMHSWork, x::MHSState)
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    v, Atx, Aty, Atz, xb = x
    bbar = getscalar(par, xb)
    vbar = anchor_mean(pb, v)
    fill_owned!((i, j, k) -> v[i, j, k] - vbar + pb.lnPref, w.u, par)
    Ar = pb.Aref
    owned_map!((i, j, k) -> Ar * Atx[i, j, k], w.A[1], par)
    owned_map!((i, j, k) -> Ar * Aty[i, j, k], w.A[2], par)
    owned_map!((i, j, k) -> Ar * Atz[i, j, k], w.A[3], par)
    exchange_halos!(par, [w.u, w.A...])
    u = w.u; P = w.P
    Threads.@threads :static for k in 1:NZ
        @inbounds for j in axes(u, 2), i in axes(u, 1)
            P[i, j, k] = exp(u[i, j, k])
        end
    end
    noob = zeros(Int, Threads.maxthreadid())
    @rows 1:NZ begin
        lr, lu, st = lnrho_dY(pb.eos, pb.X[i, j, k], u[i, j, k])
        if st & EOS_OUT_OF_ENVELOPE != 0
            noob[Threads.threadid()] += 1; lr = 0.0; lu = 0.0
        end
        w.rho[i, j, k] = exp(lr); w.rhou[i, j, k] = lu
    end
    B = w.B; J = w.J
    curlA!(B, w.A, bbar, s, par)
    exchange_halos!(par, collect(B))
    curlB!(J, B, s, par)
    for c in 1:3
        Jc = J[c]; owned_map!((i, j, k) -> Jc[i, j, k] / MU0, Jc, par)
    end
    exchange_halos!(par, collect(J))
    H!(w.HB, B, s, par)
    for c in 1:3
        HBc = w.HB[c]; Bo = pb.Bobs[c]
        fill_owned!((i, j, k) -> HBc[i, j, k] - Bo[i, j, k], w.dB[c], par)
    end
    exchange_halos!(par, collect(w.dB))
    Bx, By, Bz = B; Jx, Jy, Jz = J
    g = pb.g; lF, lS, lU = pb.lamF, pb.lamS, pb.lamU; N, NF, NB = pb.N, pb.NF, pb.NB
    # ---- x-faces ----
    pX = zeros(b.ny, NZ)
    forrows(1:NZ, par) do i, j, k
        Jyf = 0.5 * (Jy[i, j, k-1] + Jy[i, j, k]); Jzf = 0.5 * (Jz[i, j-1, k] + Jz[i, j, k])
        Bzf = 0.25 * (Bz[i, j, k-1] + Bz[i, j, k] + Bz[i+1, j, k-1] + Bz[i+1, j, k])
        Byf = 0.25 * (By[i, j-1, k] + By[i, j, k] + By[i+1, j-1, k] + By[i+1, j, k])
        R = (P[i+1, j, k] - P[i, j, k]) / s.dxf[i] - (Jyf * Bzf - Jzf * Byf)
        wf = wxf(pb, i, j, k) * pb.invS2f[1][i, j, k]
        pX[j, k] += wf * R^2
        a = 2lF * wf * R / NF
        w.aR[1][i, j, k] = a
        w.fX[1][i, j, k] = -a * Bzf; w.fX[2][i, j, k] = a * Byf
        w.fX[3][i, j, k] = -a * Jyf; w.fX[4][i, j, k] = a * Jzf
    end
    # ---- y-faces ----
    pY = zeros(b.ny, NZ)
    forrows(1:NZ, par) do i, j, k
        Jzf = 0.5 * (Jz[i-1, j, k] + Jz[i, j, k]); Jxf = 0.5 * (Jx[i, j, k-1] + Jx[i, j, k])
        Bxf = 0.25 * (Bx[i-1, j, k] + Bx[i, j, k] + Bx[i-1, j+1, k] + Bx[i, j+1, k])
        Bzf = 0.25 * (Bz[i, j, k-1] + Bz[i, j, k] + Bz[i, j+1, k-1] + Bz[i, j+1, k])
        R = (P[i, j+1, k] - P[i, j, k]) / s.dyf[j] - (Jzf * Bxf - Jxf * Bzf)
        wf = wyf(pb, i, j, k) * pb.invS2f[2][i, j, k]
        pY[j, k] += wf * R^2
        a = 2lF * wf * R / NF
        w.aR[2][i, j, k] = a
        w.fY[1][i, j, k] = -a * Bxf; w.fY[2][i, j, k] = a * Bzf
        w.fY[3][i, j, k] = -a * Jzf; w.fY[4][i, j, k] = a * Jxf
    end
    # ---- z-faces (equations on interior faces only) ----
    pZ = zeros(b.ny, NZ + 1)
    forrows(0:NZ, par) do i, j, k
        if 1 <= k <= NZ - 1
            Jxf = 0.5 * (Jx[i, j-1, k] + Jx[i, j, k]); Jyf = 0.5 * (Jy[i-1, j, k] + Jy[i, j, k])
            Byf = 0.25 * (By[i, j-1, k] + By[i, j, k] + By[i, j-1, k+1] + By[i, j, k+1])
            Bxf = 0.25 * (Bx[i-1, j, k] + Bx[i, j, k] + Bx[i-1, j, k+1] + Bx[i, j, k+1])
            rb = sqrt(w.rho[i, j, k] * w.rho[i, j, k+1])
            R = (P[i, j, k] - P[i, j, k+1]) / s.dzf[k] + g * rb - (Jxf * Byf - Jyf * Bxf)
            wf = wzf(pb, i, j, k) * pb.invS2f[3][i, j, k]
            pZ[j, k+1] += wf * R^2
            a = 2lF * wf * R / NF
            w.aR[3][i, j, k] = a
            w.fZ[1][i, j, k] = -a * Byf; w.fZ[2][i, j, k] = a * Bxf
            w.fZ[3][i, j, k] = -a * Jxf; w.fZ[4][i, j, k] = a * Jyf
        else
            w.aR[3][i, j, k] = 0.0
            for q in 1:4
                w.fZ[q][i, j, k] = 0.0
            end
        end
    end
    # ---- centres: data and regularizers ----
    pC = zeros(3, b.ny, NZ)
    forrows(1:NZ, par) do i, j, k
        wc = wcc(pb, i, j, k)
        hx = o.X.h[i]; hy = o.Y.h[j]; hz = s.hz[k]
        eB = 0.0; eS = 0.0
        for c in 1:3
            s2 = pb.sig[c][i, j, k]^2
            d = w.dB[c][i, j, k]
            eB += d^2 / s2
            w.gHB[c][i, j, k] = 2d / (NB * s2)
            ddx = d2x(w.dB[c], o.X, i, j, k); ddy = d2y(w.dB[c], o.Y, i, j, k); ddz = d2z(w.dB[c], o.Z, i, j, k)
            eS += wc * ((hx^2 * ddx)^2 + (hy^2 * ddy)^2 + (hz^2 * ddz)^2) / s2
            f = 2lS * wc / (N * s2)
            w.qS[c][i, j, k] = f * hx^4 * ddx; w.qS[3+c][i, j, k] = f * hy^4 * ddy; w.qS[6+c][i, j, k] = f * hz^4 * ddz
        end
        uxx = d2x(u, o.X, i, j, k); uyy = d2y(u, o.Y, i, j, k)
        pC[1, j, k] += eB; pC[2, j, k] += eS; pC[3, j, k] += wc * ((hx^2 * uxx)^2 + (hy^2 * uyy)^2)
        w.qU[1][i, j, k] = 2lU * wc * hx^4 * uxx / N
        w.qU[2][i, j, k] = 2lU * wc * hy^4 * uyy / N
    end
    loc = [0.0, 0.0, 0.0, 0.0, 0.0, Float64(sum(noob))]
    for k in 1:NZ, j in 1:b.ny
        loc[1] += pX[j, k]; loc[2] += pY[j, k]
    end
    for k in 0:NZ, j in 1:b.ny
        loc[3] += pZ[j, k+1]
    end
    for k in 1:NZ, j in 1:b.ny
        loc[4] += pC[1, j, k]; loc[5] += pC[2, j, k]
    end
    eU = 0.0
    for k in 1:NZ, j in 1:b.ny
        eU += pC[3, j, k]
    end
    push!(loc, eU)
    t = MPI.Allreduce(loc, +, par.comm)
    terms = (lF * (t[1] + t[2] + t[3]) / NF, 0.0, t[4] / NB, lS * t[5] / N, lU * t[7] / N)
    oob = t[6] > 0
    return oob ? Inf : sum(terms), terms, oob
end

# ---------------------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------------------
function gradient!(gx::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState)
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    exchange_halos!(par, [w.aR..., w.fX..., w.fY..., w.fZ..., w.qS..., w.qU...])
    fX, fY, fZ = w.fX, w.fY, w.fZ
    # ---- u ----
    G_T!(w.gP, w.aR, s, par)
    g = pb.g
    forrows(1:NZ, par) do i, j, k
        grav = 0.0
        k <= NZ - 1 && (grav += w.aR[3][i, j, k] * sqrt(w.rho[i, j, k] * w.rho[i, j, k+1]))
        k >= 2 && (grav += w.aR[3][i, j, k-1] * sqrt(w.rho[i, j, k-1] * w.rho[i, j, k]))
        gx[1][i, j, k] = w.P[i, j, k] * w.gP[i, j, k] + 0.5 * g * w.rhou[i, j, k] * grav +
                         d2xT(w.qU[1], o.X, i, j, k) + d2yT(w.qU[2], o.Y, i, j, k)
    end
    # ---- edges: ∂Φ/∂J from the face averages ----
    gJ = w.gJ
    owned_map!(gJ[1], par) do i, j, k             # x-edges (i, j+½, k+½), k = 0:NZ
        t = 0.0
        k >= 1 && (t += 0.5 * fY[2][i, j, k])
        k + 1 <= NZ && (t += 0.5 * fY[2][i, j, k+1])
        (1 <= k <= NZ - 1) && (t += 0.5 * (fZ[1][i, j, k] + fZ[1][i, j+1, k]))
        t
    end
    owned_map!(gJ[2], par) do i, j, k             # y-edges (i+½, j, k+½)
        t = 0.0
        k >= 1 && (t += 0.5 * fX[1][i, j, k])
        k + 1 <= NZ && (t += 0.5 * fX[1][i, j, k+1])
        (1 <= k <= NZ - 1) && (t += 0.5 * (fZ[2][i, j, k] + fZ[2][i+1, j, k]))
        t
    end
    owned_map!(gJ[3], par) do i, j, k             # z-edges (i+½, j+½, k)
        0.5 * (fX[2][i, j, k] + fX[2][i, j+1, k]) + 0.5 * (fY[1][i, j, k] + fY[1][i+1, j, k])
    end
    exchange_halos!(par, collect(gJ))
    curlB_T!(w.gBJ, gJ, s, par)
    # ---- centres → faces through H: data + δB regularizer ----
    for c in 1:3
        gH = w.gHB[c]; q1 = w.qS[c]; q2 = w.qS[3+c]; q3 = w.qS[6+c]
        owned_map!(gH, par) do i, j, k
            gH[i, j, k] + d2xT(q1, o.X, i, j, k) + d2yT(q2, o.Y, i, j, k) + d2zT(q3, o.Z, i, j, k)
        end
    end
    exchange_halos!(par, collect(w.gHB))
    H_T!(w.gB, w.gHB, s, par)
    # ---- faces: averages of B in L, plus C*ᵀ/μ0 and Hᵀ ----
    gB = w.gB; gBJ = w.gBJ
    owned_map!(gB[1], par) do i, j, k             # x-faces
        t = 0.25 * (fY[3][i, j, k] + fY[3][i+1, j, k] + fY[3][i, j-1, k] + fY[3][i+1, j-1, k])
        k <= NZ - 1 && (t += 0.25 * (fZ[4][i, j, k] + fZ[4][i+1, j, k]))
        k >= 2 && (t += 0.25 * (fZ[4][i, j, k-1] + fZ[4][i+1, j, k-1]))
        gB[1][i, j, k] + t + gBJ[1][i, j, k] / MU0
    end
    owned_map!(gB[2], par) do i, j, k             # y-faces
        t = 0.25 * (fX[4][i, j, k] + fX[4][i-1, j, k] + fX[4][i, j+1, k] + fX[4][i-1, j+1, k])
        k <= NZ - 1 && (t += 0.25 * (fZ[3][i, j, k] + fZ[3][i, j+1, k]))
        k >= 2 && (t += 0.25 * (fZ[3][i, j, k-1] + fZ[3][i, j+1, k-1]))
        gB[2][i, j, k] + t + gBJ[2][i, j, k] / MU0
    end
    owned_map!(gB[3], par) do i, j, k             # z-faces, k = 0:NZ
        t = 0.0
        if k >= 1
            t += 0.25 * (fX[3][i, j, k] + fX[3][i-1, j, k]) + 0.25 * (fY[4][i, j, k] + fY[4][i, j-1, k])
        end
        if k + 1 <= NZ
            t += 0.25 * (fX[3][i, j, k+1] + fX[3][i-1, j, k+1]) + 0.25 * (fY[4][i, j, k+1] + fY[4][i, j-1, k+1])
        end
        gB[3][i, j, k] + t + gBJ[3][i, j, k] / MU0
    end
    exchange_halos!(par, collect(gB))
    gbar = curlA_T!(w.gA, gB, s, par)
    Ar = pb.Aref
    for c in 1:3
        gAc = w.gA[c]; owned_map!((i, j, k) -> Ar * gAc[i, j, k], gx[c+1], par)
    end
    setscalar!(par, gx[5], gbar)
    # ---- gauge chain for v ----
    G = sdot_ones(par, gx[1])
    gv = gx[1]
    forrows(pb.ka:pb.ka, par) do i, j, k
        gv[i, j, k] -= pb.alpha[i, j] * G
    end
    return gx
end

function fg!(gx::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState)
    f, terms, oob = objective!(pb, w, x)
    oob && return f, terms, oob
    gradient!(gx, pb, w, x)
    return f, terms, oob
end

# ---------------------------------------------------------------------------------------
# Preconditioner: exact Gauss–Newton diagonal for u; per-layer Hutchinson means for Ã; exact for b̄
# ---------------------------------------------------------------------------------------
function gn_diagonal!(D::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState; probes::Int = 4)
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    lF, NF, g = pb.lamF, pb.NF, pb.g
    Ax(i, j, k) = @inbounds lF * wxf(pb, i, j, k) * pb.invS2f[1][i, j, k] / NF
    Ay(i, j, k) = @inbounds lF * wyf(pb, i, j, k) * pb.invS2f[2][i, j, k] / NF
    Az(i, j, k) = @inbounds lF * wzf(pb, i, j, k) * pb.invS2f[3][i, j, k] / NF
    Ux(i, j, k) = @inbounds pb.lamU * wcc(pb, i, j, k) * o.X.h[i]^4 / pb.N
    Uy(i, j, k) = @inbounds pb.lamU * wcc(pb, i, j, k) * o.Y.h[j]^4 / pb.N
    forrows(1:NZ, par) do i, j, k
        P = w.P[i, j, k]; hr = 0.5 * g * w.rhou[i, j, k]
        d = Ax(i, j, k) * (P / s.dxf[i])^2 + Ax(i - 1, j, k) * (P / s.dxf[i-1])^2 +
            Ay(i, j, k) * (P / s.dyf[j])^2 + Ay(i, j - 1, k) * (P / s.dyf[j-1])^2
        if k <= NZ - 1
            rb = sqrt(w.rho[i, j, k] * w.rho[i, j, k+1]); d += Az(i, j, k) * (P / s.dzf[k] + hr * rb)^2
        end
        if k >= 2
            rb = sqrt(w.rho[i, j, k-1] * w.rho[i, j, k]); d += Az(i, j, k - 1) * (-P / s.dzf[k-1] + hr * rb)^2
        end
        D[1][i, j, k] = 2d + 2 * (d2xT2(Ux, o.X, i, j, k) + d2yT2(Uy, o.Y, i, j, k))
    end
    # Ã and b̄: Hessian–vector products by central differences of ∇Φ, averaged per layer
    w2 = SMHSWork(par)
    z = similar_state(x); xt = similar_state(x); g1 = similar_state(x); g2 = similar_state(x)
    ε = 1e-4
    acc = [zeros(length(axes(x[f], 3))) for f in 2:4]
    cnt = [zeros(length(axes(x[f], 3))) for f in 2:4]
    for p in 1:probes
        for f in 2:4
            zf = z[f]
            owned_map!((i, j, k) -> isodd(hash((b.i0 + i - 1, b.j0 + j - 1, k, f, p, 0xa11)) >> 7) ? 1.0 : -1.0, zf, par)
        end
        lincomb!(xt, 1.0, x, ε, z, par); fg!(g1, pb, w2, xt)
        lincomb!(xt, 1.0, x, -ε, z, par); fg!(g2, pb, w2, xt)
        for (q, f) in enumerate(2:4)
            k0 = first(axes(x[f], 3))
            for k in axes(x[f], 3), j in 1:b.ny, i in 1:b.nx
                acc[q][k-k0+1] += z[f][i, j, k] * (g1[f][i, j, k] - g2[f][i, j, k]) / (2ε)
                cnt[q][k-k0+1] += 1
            end
        end
    end
    for (q, f) in enumerate(2:4)
        a = MPI.Allreduce(acc[q], +, par.comm) ./ max.(MPI.Allreduce(cnt[q], +, par.comm), 1)
        k0 = first(axes(x[f], 3))
        mx = maximum(abs, a)
        owned_map!((i, j, k) -> max(abs(a[k-k0+1]), 1e-8 * mx, 1e-300), D[f], par)
    end
    # b̄: exact second derivative along the scalar
    fill!(parent(z[1]), 0.0); for f in 2:4; fill!(parent(z[f]), 0.0); end
    setscalar!(par, z[5], 1.0)
    eb = 1e-6 * max(abs(getscalar(par, x[5])), 1e-4)
    lincomb!(xt, 1.0, x, eb, z, par); fg!(g1, pb, w2, xt)
    lincomb!(xt, 1.0, x, -eb, z, par); fg!(g2, pb, w2, xt)
    hbb = (getscalar(par, g1[5]) - getscalar(par, g2[5])) / (2eb)
    fill!(parent(D[5]), 1.0); rank0(par) && (D[5][1, 1, 1] = max(abs(hbb), 1e-300))
    mx = global_max(par, maximum(D[1][i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx))
    fill_owned!((i, j, k) -> max(D[1][i, j, k], 1e-12 * mx, 1e-300), D[1], par)
    return D
end

# ---------------------------------------------------------------------------------------
# Initialization (addendum §7)
# ---------------------------------------------------------------------------------------
"""
    staggered_hse!(u, pb; P_bottom)

Exact discrete vertical equilibrium for B = 0 on the staggered mesh:
(P_k − P_{k+1})/Δz_{k+½} + g √(ρ_k ρ_{k+1}) = 0, upward from u[NZ] = ln P_bottom.
"""
function staggered_hse!(u::Field, pb::SMHSProblem; P_bottom::Float64 = exp(pb.lnPref))
    par = pb.par; s = pb.s; b = par.block; NZ = b.nz
    nbad = zeros(Int, Threads.maxthreadid())
    Threads.@threads :static for c in 1:(b.nx*b.ny)
        i = (c - 1) % b.nx + 1; j = (c - 1) ÷ b.nx + 1
        u[i, j, NZ] = log(P_bottom)
        for k in NZ-1:-1:1
            up = u[i, j, k+1]; Pp = exp(up)
            rp = exp(lnrho(pb.eos, pb.X[i, j, k+1], up)[1])
            F(uk) = (exp(uk) - Pp) / s.dzf[k] + pb.g * sqrt(exp(lnrho(pb.eos, pb.X[i, j, k], uk)[1]) * rp)
            # bracket: F(up) = g ρ̄ > 0; step down until F < 0 (stays inside the EOS envelope)
            lo = up - 0.25; while lo > up - 60.0 && !(F(lo) < 0); lo -= 0.25; end
            uk = try
                brent(F, lo, up)
            catch
                nbad[Threads.threadid()] += 1; up
            end
            u[i, j, k] = uk
        end
    end
    sum(nbad) > 0 && @warn "staggered_hse!: $(sum(nbad)) column steps without a bracketed root"
    return u
end

"""Initial state: v from the staggered vertical equilibrium, Ã = 0, b̄ = mean Bobs_z."""
function init_state(pb::SMHSProblem)
    par = pb.par
    x = snewstate(par)
    staggered_hse!(x[1], pb)
    bz = sdot_ones(par, pb.Bobs[3]) / pb.N
    setscalar!(par, x[5], bz)
    return x
end

"""
    init_A!(pb, w, x; max_iter, rtol) -> (iterations, rnorm/rnorm0)

A₀: fit (Ã, b̄) to Bobs with v frozen and λ_F = λ_U = 0. Φ_B + Φ_S is then quadratic in
(Ã, b̄), so this is conjugate gradients on the normal equations Q y = −∇Φ(0), with the
Hessian–vector product Q p = ∇Φ(p) − ∇Φ(0) (∇Φ is affine; one gradient per iteration).
Converges to the minimum-norm solution in the gauge null space of curl when started at 0.
"""
function init_A!(pb::SMHSProblem, w::SMHSWork, x::MHSState; max_iter::Int = 2000, rtol::Float64 = 1e-10)
    par = pb.par
    lF, lU = pb.lamF, pb.lamU
    pb.lamF = 0.0; pb.lamU = 0.0
    try
        zfields!(y) = (fill!(parent(y[1]), 0.0); y)
        y = similar_state(x); copyto!(parent(y[1]), parent(x[1]))      # v kept; Ã = b̄ = 0
        g0 = similar_state(x); fg!(g0, pb, w, y); zfields!(g0)
        r = similar_state(x); lincomb!(r, -1.0, g0, 0.0, g0, par)
        p = similar_state(x); lincomb!(p, 1.0, r, 0.0, r, par)
        q = similar_state(x); yt = similar_state(x)
        sol = similar_state(x)                                         # (Ã, b̄) increments from 0
        rr = dotN(par, r, r); rr0 = rr
        it = 0
        while it < max_iter && rr > rtol^2 * rr0
            it += 1
            lincomb!(yt, 1.0, y, 1.0, p, par); fg!(q, pb, w, yt)
            lincomb!(q, 1.0, q, -1.0, g0, par); zfields!(q)
            pq = dotN(par, p, q)
            pq > 0 || break
            α = rr / pq
            axpy!(sol, α, p, par); axpy!(r, -α, q, par)
            rn = dotN(par, r, r)
            lincomb!(p, 1.0, r, rn / rr, p, par)
            rr = rn
        end
        for f in 2:5
            copyto!(parent(x[f]), parent(sol[f]))
        end
        return it, sqrt(rr / max(rr0, 1e-300))
    finally
        pb.lamF = lF; pb.lamU = lU
    end
end
