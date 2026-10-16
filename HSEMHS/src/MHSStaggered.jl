# MHS objective and adjoint on the staggered vector-potential mesh
# (HSE_MHS_D9_STAGGERED_ADDENDUM.md §3–7).
#
# State x = (v, Ãx, Ãy, Ãz, b̄):  u = v − v̄ + ln P_ref (centres);  A = A_ref Ã (edges);
# B = curl A + b̄ ẑ (faces).  b̄ is a scalar stored at [1,1,1] of a 1-level field on rank 0
# only (zero elsewhere), so that sums over ranks give its value once.
#
# Φ = Φ_F + Φ_N + Φ_B + Φ_S + Φ_U
#   Φ_F = (λ_F/N_F) Σ_faces w_f |R_f|²/S_f²   on x-, y- and interior z-faces
#   Φ_B = (1/N_B) Σ ((HB)_c − Bobs_c)²/σ_c²   at centres
#   Φ_S = (λ_S/N) Σ w Σ_a Σ_c (h_a² ∂²_a δB_c)²/σ_c²,  δB = HB − Bobs
#   Φ_U = (λ_U/N) Σ w Σ_{a=x,y} (h_a² ∂²_a u)²
#   Φ_N = (λ_N/N) Σ_faces w_f (¼ δ²_n B_n)²/σ²   (second difference along the face normal; M6c option)
#   terms are reported as (F, N, B, S, U)

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
    lamN::Float64                     # face-normal Nyquist regularizer (M6c option; default 0)
    dfaces::Bool                      # data term on faces against Bobs interpolated to faces (M6c option)
    Bobsf::NTuple{3,Field}
    invsf2::NTuple{3,Field}
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
    qS::NTuple{9,Field}; qU::NTuple{2,Field}; qN::NTuple{3,Field}; gBd::NTuple{3,Field}
    gHB::NTuple{3,Field}; gB::NTuple{3,Field}; gBJ::NTuple{3,Field}; gJ::NTuple{3,Field}; gA::NTuple{3,Field}
    gP::Field
end

function SMHSWork(par::ParCtx)
    c() = newfield(par); z() = newfield0(par)
    edges() = (z(), z(), c()); faces() = (c(), c(), z())
    return SMHSWork(c(), c(), c(), c(), edges(), faces(), edges(), (c(), c(), c()), (c(), c(), c()), faces(),
                    ntuple(_ -> c(), 4), ntuple(_ -> c(), 4), ntuple(_ -> z(), 4), ntuple(_ -> c(), 9), ntuple(_ -> c(), 2), faces(), faces(),
                    (c(), c(), c()), faces(), faces(), edges(), edges(), c())
end

"""
    SMHSProblem(par, grid, T, Bobs, sigma, eos; anchor_plane, P_ref, g, lambda_F, lambda_S, lambda_U)

Bobs, sigma at centres. S is set by `set_scales!` (from an initial u).
"""
function SMHSProblem(par::ParCtx, grid::Grid, T::Field, Bobs::NTuple{3,Field}, sigma::NTuple{3,Field}, eos;
                     anchor_plane::Symbol = :bottom, P_ref::Float64, g::Float64 = 274.0,
                     lambda_F::Float64 = 1.0, lambda_S::Float64 = 1e-3, lambda_U::Float64 = 0.0, lambda_N::Float64 = 0.0, data_on_faces::Bool = false)
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
    # Bobs and 1/σ² interpolated to faces (used when data_on_faces)
    Bobsf = (newfield(par), newfield(par), newfield0(par)); invsf2 = (newfield(par), newfield(par), newfield0(par))
    NZ = b.nz
    fill_owned!((i, j, k) -> 0.5 * (Bobs[1][i, j, k] + Bobs[1][i+1, j, k]), Bobsf[1], par)
    fill_owned!((i, j, k) -> 1 / (sig[1][i, j, k] * sig[1][i+1, j, k]), invsf2[1], par)
    fill_owned!((i, j, k) -> 0.5 * (Bobs[2][i, j, k] + Bobs[2][i, j+1, k]), Bobsf[2], par)
    fill_owned!((i, j, k) -> 1 / (sig[2][i, j, k] * sig[2][i, j+1, k]), invsf2[2], par)
    kl(k) = clamp(k, 1, NZ); ku(k) = clamp(k + 1, 1, NZ)
    owned_map!((i, j, k) -> 0.5 * (Bobs[3][i, j, kl(k)] + Bobs[3][i, j, ku(k)]), Bobsf[3], par)
    owned_map!((i, j, k) -> 1 / (sig[3][i, j, kl(k)] * sig[3][i, j, ku(k)]), invsf2[3], par)
    NB = data_on_faces ? Float64(NX) * NY * (3NZ + 1) : 3N
    return SMHSProblem(par, grid, s, ops, eos, X, Bobs, sig, invS2f, alpha, ka, log(P_ref), g,
                       lambda_F, lambda_S, lambda_U, lambda_N, data_on_faces, Bobsf, invsf2, Aref, N, Float64(NX) * NY * (3NZ - 1), NB, wmean)
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

# Face Lorentz force from the averages M (addendum §4): returns (L, the two J averages, the two B averages)
@inline function lor_x(J, B, i, j, k)
    @inbounds begin
        Jyf = 0.5 * (J[2][i, j, k-1] + J[2][i, j, k]); Jzf = 0.5 * (J[3][i, j-1, k] + J[3][i, j, k])
        Bzf = 0.25 * (B[3][i, j, k-1] + B[3][i, j, k] + B[3][i+1, j, k-1] + B[3][i+1, j, k])
        Byf = 0.25 * (B[2][i, j-1, k] + B[2][i, j, k] + B[2][i+1, j-1, k] + B[2][i+1, j, k])
    end
    return Jyf * Bzf - Jzf * Byf, Jyf, Jzf, Bzf, Byf
end
@inline function lor_y(J, B, i, j, k)
    @inbounds begin
        Jzf = 0.5 * (J[3][i-1, j, k] + J[3][i, j, k]); Jxf = 0.5 * (J[1][i, j, k-1] + J[1][i, j, k])
        Bxf = 0.25 * (B[1][i-1, j, k] + B[1][i, j, k] + B[1][i-1, j+1, k] + B[1][i, j+1, k])
        Bzf = 0.25 * (B[3][i, j, k-1] + B[3][i, j, k] + B[3][i, j+1, k-1] + B[3][i, j+1, k])
    end
    return Jzf * Bxf - Jxf * Bzf, Jzf, Jxf, Bxf, Bzf
end
@inline function lor_z(J, B, i, j, k)          # interior z-faces 1 ≤ k ≤ NZ−1
    @inbounds begin
        Jxf = 0.5 * (J[1][i, j-1, k] + J[1][i, j, k]); Jyf = 0.5 * (J[2][i-1, j, k] + J[2][i, j, k])
        Byf = 0.25 * (B[2][i, j-1, k] + B[2][i, j, k] + B[2][i, j-1, k+1] + B[2][i, j, k+1])
        Bxf = 0.25 * (B[1][i-1, j, k] + B[1][i, j, k] + B[1][i-1, j, k+1] + B[1][i, j, k+1])
    end
    return Jxf * Byf - Jyf * Bxf, Jxf, Jyf, Byf, Bxf
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
    g = pb.g; lF, lS, lU = pb.lamF, pb.lamS, pb.lamU; N, NF, NB = pb.N, pb.NF, pb.NB
    # ---- x-faces ----
    pX = zeros(b.ny, NZ)
    forrows(1:NZ, par) do i, j, k
        L, Jyf, Jzf, Bzf, Byf = lor_x(J, B, i, j, k)
        R = (P[i+1, j, k] - P[i, j, k]) / s.dxf[i] - L
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
        L, Jzf, Jxf, Bxf, Bzf = lor_y(J, B, i, j, k)
        R = (P[i, j+1, k] - P[i, j, k]) / s.dyf[j] - L
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
            L, Jxf, Jyf, Byf, Bxf = lor_z(J, B, i, j, k)
            rb = sqrt(w.rho[i, j, k] * w.rho[i, j, k+1])
            R = (P[i, j, k] - P[i, j, k+1]) / s.dzf[k] + g * rb - L
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
    # ---- faces: Nyquist regularizer along each face normal (the part of B that H averages away) ----
    lN = pb.lamN; Bx, By, Bz = B; sg = pb.sig
    pN = zeros(b.ny, NZ + 1)
    if lN > 0
        forrows(1:NZ, par) do i, j, k
            s2 = sg[1][i, j, k] * sg[1][i+1, j, k]; q = 0.25 * (Bx[i+1, j, k] - 2Bx[i, j, k] + Bx[i-1, j, k]); wc = wxf(pb, i, j, k)
            pN[j, k+1] += wc * q^2 / s2; w.qN[1][i, j, k] = 2lN * wc * q / (s2 * N)
        end
        forrows(1:NZ, par) do i, j, k
            s2 = sg[2][i, j, k] * sg[2][i, j+1, k]; q = 0.25 * (By[i, j+1, k] - 2By[i, j, k] + By[i, j-1, k]); wc = wyf(pb, i, j, k)
            pN[j, k+1] += wc * q^2 / s2; w.qN[2][i, j, k] = 2lN * wc * q / (s2 * N)
        end
        forrows(0:NZ, par) do i, j, k
            if 1 <= k <= NZ - 1
                s2 = sg[3][i, j, k] * sg[3][i, j, k+1]; q = 0.25 * (Bz[i, j, k-1] - 2Bz[i, j, k] + Bz[i, j, k+1]); wc = wzf(pb, i, j, k)
                pN[j, k+1] += wc * q^2 / s2; w.qN[3][i, j, k] = 2lN * wc * q / (s2 * N)
            else
                w.qN[3][i, j, k] = 0.0
            end
        end
    else
        for c in 1:3
            fill!(parent(w.qN[c]), 0.0)
        end
    end
    # ---- faces: data term against Bobs interpolated to faces (option) ----
    pD = zeros(b.ny, NZ + 1)
    if pb.dfaces
        for c in 1:3
            Bc = B[c]; Bo = pb.Bobsf[c]; iv = pb.invsf2[c]; gd = w.gBd[c]
            forrows(collect(axes(Bc, 3)), par) do i, j, k
                d = Bc[i, j, k] - Bo[i, j, k]
                pD[j, k-first(axes(Bc, 3))+1] += d^2 * iv[i, j, k]
                gd[i, j, k] = 2d * iv[i, j, k] / NB
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
            if pb.dfaces
                w.gHB[c][i, j, k] = 0.0
            else
                eB += d^2 / s2
                w.gHB[c][i, j, k] = 2d / (NB * s2)
            end
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
    eN = 0.0
    for k in 0:NZ, j in 1:b.ny
        eN += pN[j, k+1]
    end
    push!(loc, eN)
    eD = 0.0
    for k in 1:NZ+1, j in 1:b.ny
        eD += pD[j, k]
    end
    push!(loc, eD)
    t = MPI.Allreduce(loc, +, par.comm)
    terms = (lF * (t[1] + t[2] + t[3]) / NF, lN * t[8] / N, (t[4] + t[9]) / NB, lS * t[5] / N, lU * t[7] / N)
    oob = t[6] > 0
    return oob ? Inf : sum(terms), terms, oob
end

# ---------------------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------------------
function gradient!(gx::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState)
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    exchange_halos!(par, [w.aR..., w.fX..., w.fY..., w.fZ..., w.qS..., w.qU..., w.qN...])
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
    if pb.dfaces
        for c in 1:3
            gBc = w.gB[c]; gd = w.gBd[c]
            owned_map!((i, j, k) -> gBc[i, j, k] + gd[i, j, k], gBc, par)
        end
    end
    # ---- faces: averages of B in L, plus C*ᵀ/μ0 and Hᵀ ----
    gB = w.gB; gBJ = w.gBJ; qN = w.qN
    owned_map!(gB[1], par) do i, j, k             # x-faces
        t = 0.25 * (fY[3][i, j, k] + fY[3][i+1, j, k] + fY[3][i, j-1, k] + fY[3][i+1, j-1, k])
        k <= NZ - 1 && (t += 0.25 * (fZ[4][i, j, k] + fZ[4][i+1, j, k]))
        k >= 2 && (t += 0.25 * (fZ[4][i, j, k-1] + fZ[4][i+1, j, k-1]))
        t += 0.25 * (qN[1][i-1, j, k] - 2qN[1][i, j, k] + qN[1][i+1, j, k])
        gB[1][i, j, k] + t + gBJ[1][i, j, k] / MU0
    end
    owned_map!(gB[2], par) do i, j, k             # y-faces
        t = 0.25 * (fX[4][i, j, k] + fX[4][i-1, j, k] + fX[4][i, j+1, k] + fX[4][i-1, j+1, k])
        k <= NZ - 1 && (t += 0.25 * (fZ[3][i, j, k] + fZ[3][i, j+1, k]))
        k >= 2 && (t += 0.25 * (fZ[3][i, j, k-1] + fZ[3][i, j+1, k-1]))
        t += 0.25 * (qN[2][i, j-1, k] - 2qN[2][i, j, k] + qN[2][i, j+1, k])
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
        t += -0.5 * qN[3][i, j, k] + (k >= 1 ? 0.25 * qN[3][i, j, k-1] : 0.0) + (k + 1 <= NZ ? 0.25 * qN[3][i, j, k+1] : 0.0)
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

# ---------------------------------------------------------------------------------------
# Gate C helpers (staggered)
# ---------------------------------------------------------------------------------------
"""
Per-layer RMS and p99 of |R|/S from the last `objective!`. Each centre (i, j, k) carries its
own +½ faces: r² = (R_x/S)² + (R_y/S)² + (R_z/S)² (R_z only for k ≤ NZ−1). Needs λ_F > 0.
"""
function layer_residual_stats(pb::SMHSProblem, w::SMHSWork)
    par = pb.par; b = par.block; NZ = b.nz
    c = pb.NF / (2pb.lamF)
    rs(i, j, k) = @inbounds begin
        rx = w.aR[1][i, j, k] * c / (wxf(pb, i, j, k) * sqrt(pb.invS2f[1][i, j, k]))
        ry = w.aR[2][i, j, k] * c / (wyf(pb, i, j, k) * sqrt(pb.invS2f[2][i, j, k]))
        rz = k <= NZ - 1 ? w.aR[3][i, j, k] * c / (wzf(pb, i, j, k) * sqrt(pb.invS2f[3][i, j, k])) : 0.0
        sqrt(rx^2 + ry^2 + rz^2)
    end
    s2 = zeros(2, NZ)
    for k in 1:NZ, j in 1:b.ny, i in 1:b.nx
        s2[1, k] += rs(i, j, k)^2; s2[2, k] += 1
    end
    t = MPI.Allreduce(s2, +, par.comm)
    return sqrt.(t[1, :] ./ t[2, :]), layer_quantiles(par, rs, [0.99])[:, 1]
end

"""Global RMS of f − fref over owned i, j and the whole z axis of f."""
function rms_diff(par::ParCtx, f::Field, fref::Field)
    b = par.block
    s = 0.0
    for k in axes(f, 3), j in 1:b.ny, i in 1:b.nx
        s += (f[i, j, k] - fref[i, j, k])^2
    end
    return sqrt(MPI.Allreduce(s, +, par.comm) / (par.NX * par.NY * length(axes(f, 3))))
end

"""
Staggered C1: the same atmosphere, σ and uniform Bobs as `c1_problem`. Returns the problem,
the initial state (v from column_hse at default substeps, Ã from `init_A!`), the exact
staggered discrete equilibrium u_s (`staggered_hse!`), the column_hse reference u (n_sub = 64)
and the grid.
"""
function sc1_problem(par::ParCtx, NX, NY, NZ; Bobs = (0.0, 0.0, 0.0), lambda_F = 1.0, lambda_N = 0.0, data_on_faces = false, P_ref = 1.0e4)
    grid, x, y, z = synthetic_grid(par, NX, NY, NZ)
    eos = IdealGasMu(1.3)
    T = newfield(par); fill_owned!((i, j, k) -> 6000 + 3000 * tanh((z[k] - 1.2e6) / 2.0e5), T, par)
    Bo = ntuple(c -> newfield(par; fill = Bobs[c]), 3)
    Bm = sqrt(sum(abs2, Bobs))
    sig = ntuple(_ -> newfield(par; fill = sigma_synth(Bm, 0.05, 5e-4)), 3)
    pb = SMHSProblem(par, grid, T, Bo, sig, eos; P_ref = P_ref, lambda_F = lambda_F, lambda_N = lambda_N, data_on_faces = data_on_faces)
    us = newfield(par); staggered_hse!(us, pb; P_bottom = P_ref)
    ex = column_hse(par, grid, T, eos, HSEOpts(anchor = :bottom, P_ref = P_ref, n_sub = 64, adapt = false))
    ini = column_hse(par, grid, T, eos, HSEOpts(anchor = :bottom, P_ref = P_ref))
    uc = newfield(par); fill_owned!((i, j, k) -> log(ex.fields["Pgas"][i, j, k]), uc, par)
    set_scales!(pb, us)
    x0 = snewstate(par)
    fill_owned!((i, j, k) -> log(ini.fields["Pgas"][i, j, k]), x0[1], par)
    setscalar!(par, x0[5], Bobs[3])
    w = SMHSWork(par)
    init_A!(pb, w, x0)
    return pb, x0, us, uc, grid
end

"""
Vector potential of the C2 potential field (`potential_field!`) sampled on the edges:
A = (−∂_yχ, ∂_xχ, 0), χ = −(B0/k²) cos(k_x x) cos(k_y y) e^{k(z − z_top)}, so curl A = ∇φ.
"""
function potential_A!(A::NTuple{3,Field}, par, s::StagOps, x, y; B0 = 0.01, Lx = 4.0e6, Ly = 4.0e6)
    kx = 2π / Lx; ky = 4π / Ly; kk = sqrt(kx^2 + ky^2); zt = s.zc[1]
    b = par.block
    xc(i) = x[mod1(b.i0 + i - 1, par.NX)] + (b.i0 + i - 1 > par.NX ? Lx : 0.0) - (b.i0 + i - 1 < 1 ? Lx : 0.0)
    yc(j) = y[mod1(b.j0 + j - 1, par.NY)] + (b.j0 + j - 1 > par.NY ? Ly : 0.0) - (b.j0 + j - 1 < 1 ? Ly : 0.0)
    for k in axes(A[1], 3), j in 0:b.ny+1, i in 0:b.nx+1
        e = exp(kk * (s.zf[k] - zt))
        xe = xc(i); ye = yc(j) + 0.5 * s.dyf[j]                    # x-edge (i, j+½, k+½)
        A[1][i, j, k] = -(B0 / kk^2) * ky * cos(kx * xe) * sin(ky * ye) * e
        xe = xc(i) + 0.5 * s.dxf[i]; ye = yc(j)                    # y-edge (i+½, j, k+½)
        A[2][i, j, k] = (B0 / kk^2) * kx * sin(kx * xe) * cos(ky * ye) * e
    end
    fill!(parent(A[3]), 0.0)
    return kk
end
