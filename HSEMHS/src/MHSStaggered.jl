# MHS objective and adjoint on the staggered vector-potential mesh
# (HSE_MHS_D9_STAGGERED_ADDENDUM.md §3–7).
#
# State x = (v, Ãx, Ãy, Ãz, b̄):  u = v − v̄ + ln P_ref (centres);  A = A_ref Ã (edges);
# B = curl A + b̄ ẑ (faces).  b̄ is a scalar stored at [1,1,1] of a 1-level field on rank 0
# only (zero elsewhere), so that sums over ranks give its value once.
#
# Φ = Φ_F + Φ_N + Φ_B + Φ_S + Φ_U
#   Φ_F = (λ_F/N_F) Σ_faces w_f |R_f|²/S_f²   on x-, y-faces of layers 2…NZ−1 and interior z-faces
#   Φ_B = (1/N_B) Σ ((HB)_c − Bobs_c)²/σ_c²   at centres
#   Φ_S = (λ_S/N) Σ w Σ_a Σ_c (h_a² ∂²_a δB_c)²/σ_c²,  δB = HB − Bobs
#   Φ_U = (1/N) Σ w [λ_U Σ_{a=x,y} (h_a² ∂²_a u)² + λ_Uz (h_z² ∂²_z u)²]   (λ_Uz: D9b §2.3)
#   Φ_N = (λ_N/N) Σ_faces w_f (¼ δ²_n B_n)²/σ²   (second difference along the face normal; M6c option)
#   Φ_Q = (λ_F λ_Q/N_F) Σ_edges w_e Σ_{a ∈ avg(e)} (B_ref ¼δ²_a J_e / S_e)²   (D9b §2.2; × λ_F since M5)
#   terms are reported as (F, N + Q, B, S, U)

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
    lamUz::Float64                    # vertical grid-scale term in Φ_U (D9b §2.3)
    lamN::Float64                     # face-normal Nyquist regularizer (M6c option; default 0)
    dfaces::Bool                      # data term on faces against Bobs interpolated to faces (M6c option)
    Bobsf::NTuple{3,Field}
    invsf2::NTuple{3,Field}
    lamQ::Float64                     # force-weighted current-Nyquist term Φ_Q (D9b §2.2)
    cQe::Vector{Float64}              # (B_ref/S)² at x/y-edge levels k+½, index k+1 (k = 0:NZ)
    cQc::Vector{Float64}              # (B_ref/S)² at z-edge levels k (1:NZ)
    Aref::Float64
    N::Float64
    NF::Float64
    NB::Float64
    wmean::Float64
    pc_atilde::Symbol                 # Ã preconditioner diagonal: :layer (Hutchinson layer means) | :colored (exact per node, M5)
    pc_period::Int                    # coloring period for :colored (≥ the GN Hessian reach)
    pc_ufloor::Float64                # u diagonal floor, as a fraction of its layer median (M5)
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
    qS::NTuple{9,Field}; qU::NTuple{3,Field}; qN::NTuple{3,Field}; gBd::NTuple{3,Field}
    qQ::NTuple{6,Field}               # Φ_Q seeds: x-edge (y, z), y-edge (x, z), z-edge (x, y)
    gHB::NTuple{3,Field}; gB::NTuple{3,Field}; gBJ::NTuple{3,Field}; gJ::NTuple{3,Field}; gA::NTuple{3,Field}
    gP::Field
end

function SMHSWork(par::ParCtx)
    c() = newfield(par); z() = newfield0(par)
    edges() = (z(), z(), c()); faces() = (c(), c(), z())
    return SMHSWork(c(), c(), c(), c(), edges(), faces(), edges(), (c(), c(), c()), (c(), c(), c()), faces(),
                    ntuple(_ -> c(), 4), ntuple(_ -> c(), 4), ntuple(_ -> z(), 4), ntuple(_ -> c(), 9), ntuple(_ -> c(), 3), faces(), faces(), (z(), z(), z(), z(), c(), c()),
                    (c(), c(), c()), faces(), faces(), edges(), edges(), c())
end

"""
    SMHSProblem(par, grid, T, Bobs, sigma, eos; anchor_plane, P_ref, g, lambda_F, lambda_S, lambda_U,
                lambda_N, data_on_faces, lambda_Q, lambda_Uz)

Bobs, sigma at centres. S (and the Φ_Q layer weights) are set by `set_scales!` (from an initial u).
Defaults follow D9b: data on faces, λ_Q = 1; λ_Uz = 0 (M6d: 1e-4 biases C1a beyond the gate, and with face data
the top additive-pressure mode it targets does not occur).
"""
function SMHSProblem(par::ParCtx, grid::Grid, T::Field, Bobs::NTuple{3,Field}, sigma::NTuple{3,Field}, eos;
                     anchor_plane::Symbol = :bottom, P_ref::Float64, g::Float64 = 274.0,
                     lambda_F::Float64 = 1.0, lambda_S::Float64 = 1e-3, lambda_U::Float64 = 0.0, lambda_N::Float64 = 0.0, data_on_faces::Bool = true, lambda_Q::Float64 = 1.0,
                     lambda_Uz::Float64 = 0.0, precond_atilde::Symbol = :layer, precond_period::Int = 5, precond_ufloor::Float64 = 0.0)
    precond_atilde in (:layer, :colored) || invalid("precond_atilde must be :layer or :colored")
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
                       lambda_F, lambda_S, lambda_U, lambda_Uz, lambda_N, data_on_faces, Bobsf, invsf2, lambda_Q, zeros(NZ + 1), zeros(NZ), Aref, N, Float64(NX) * NY * (3NZ - 5), NB, wmean,
                       precond_atilde, precond_period, precond_ufloor)
end

"""Horizontal (x-, y-face) force equations exist on layers 2…NZ−1 only: the top and bottom layers
average a boundary-edge J, which lacks the ∂_z B term (M6c, option (a))."""
@inline horeq(k, NZ) = (2 <= k <= NZ - 1) ? 1.0 : 0.0

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
    # Φ_Q layer weights (B_ref/S)², B_ref = layer median |Bobs|, S = layer median S
    Bm = layer_quantiles(par, (i, j, k) -> sqrt(pb.Bobs[1][i, j, k]^2 + pb.Bobs[2][i, j, k]^2 + pb.Bobs[3][i, j, k]^2), [0.5])[:, 1]
    Sm = layer_quantiles(par, S, [0.5])[:, 1]
    for k in 0:NZ
        ka = clamp(k, 1, NZ); kb = clamp(k + 1, 1, NZ)
        pb.cQe[k+1] = Bm[ka] * Bm[kb] / (Sm[ka] * Sm[kb])
    end
    for k in 1:NZ
        pb.cQc[k] = (Bm[k] / Sm[k])^2
    end
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
"""
    phiQ_seeds!(qQ, J, pb, pQ = nothing)

Φ_Q seeds ∂Φ_Q/∂(¼δ²_a J_e) for the edge currents J (D9b §2.2): horizontal second differences on
every edge level (boundary edges included: under option (a) no force equation sees those currents),
the z second difference of x/y-edge currents on interior edges only. Φ_Q is linear in J, so the same
seeds applied to δJ give the Gauss–Newton product. With `pQ`, accumulates w_e q² per row (j, k+1).
"""
function phiQ_seeds!(qQ::NTuple{6,Field}, J::NTuple{3,Field}, pb::SMHSProblem, pQ = nothing)
    par = pb.par; s = pb.s; NZ = par.block.nz; lQ = pb.lamQ * pb.lamF; NF = pb.NF   # priced like the force it hides
    if lQ <= 0
        for c in 1:6
            fill!(parent(qQ[c]), 0.0)
        end
        return qQ
    end
    acc = pQ !== nothing
    Jx, Jy, Jz = J
    forrows(0:NZ, par) do i, j, k
        c = pb.cQe[k+1]
        wx = s.dxc[i] * s.dyf[j] * s.dzf[k] / pb.wmean * c
        wy = s.dxf[i] * s.dyc[j] * s.dzf[k] / pb.wmean * c
        e = 0.0
        q = 0.25 * (Jx[i, j+1, k] - 2Jx[i, j, k] + Jx[i, j-1, k]); e += wx * q^2; qQ[1][i, j, k] = 2lQ * wx * q / NF
        q = 0.25 * (Jy[i+1, j, k] - 2Jy[i, j, k] + Jy[i-1, j, k]); e += wy * q^2; qQ[3][i, j, k] = 2lQ * wy * q / NF
        if 2 <= k <= NZ - 2
            q = 0.25 * (Jx[i, j, k-1] - 2Jx[i, j, k] + Jx[i, j, k+1]); e += wx * q^2; qQ[2][i, j, k] = 2lQ * wx * q / NF
            q = 0.25 * (Jy[i, j, k-1] - 2Jy[i, j, k] + Jy[i, j, k+1]); e += wy * q^2; qQ[4][i, j, k] = 2lQ * wy * q / NF
        else
            qQ[2][i, j, k] = 0.0; qQ[4][i, j, k] = 0.0
        end
        acc && (pQ[j, k+1] += e)
    end
    forrows(1:NZ, par) do i, j, k                    # z-edges: all layers (no z difference)
        wz = s.dxf[i] * s.dyf[j] * s.hz[k] / pb.wmean * pb.cQc[k]
        q = 0.25 * (Jz[i+1, j, k] - 2Jz[i, j, k] + Jz[i-1, j, k]); qQ[5][i, j, k] = 2lQ * wz * q / NF; e = wz * q^2
        q = 0.25 * (Jz[i, j+1, k] - 2Jz[i, j, k] + Jz[i, j-1, k]); qQ[6][i, j, k] = 2lQ * wz * q / NF
        acc && (pQ[j, k+1] += e + wz * q^2)
    end
    return qQ
end

"""
Work counters for the staggered problem (M5 solver comparison): `WORK_COUNTS[1]` objective (forward) calls,
`[2]` gradient (adjoint) calls, `[3]` Gauss–Newton products. `work_equivalent()` converts them to
equivalent GN products: an objective+gradient pair counts 1, a GN product 1 (its internal adjoint is not
counted again), a bare objective ½.
"""
const WORK_COUNTS = zeros(Int, 3)
reset_work!() = fill!(WORK_COUNTS, 0)
work_equivalent() = 0.5WORK_COUNTS[1] + 0.5 * (WORK_COUNTS[2] - WORK_COUNTS[3]) + WORK_COUNTS[3]

function objective!(pb::SMHSProblem, w::SMHSWork, x::MHSState)
    WORK_COUNTS[1] += 1
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
    # ---- edges: force-weighted current-Nyquist term Φ_Q (D9b §2.2) ----
    pQ = zeros(b.ny, NZ + 1)
    phiQ_seeds!(w.qQ, J, pb, pQ)
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
        wf = horeq(k, NZ) * wxf(pb, i, j, k) * pb.invS2f[1][i, j, k]
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
        wf = horeq(k, NZ) * wyf(pb, i, j, k) * pb.invS2f[2][i, j, k]
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
        uxx = d2x(u, o.X, i, j, k); uyy = d2y(u, o.Y, i, j, k); uzz = d2z(u, o.Z, i, j, k)
        pC[1, j, k] += eB; pC[2, j, k] += eS
        pC[3, j, k] += wc * (lU * ((hx^2 * uxx)^2 + (hy^2 * uyy)^2) + pb.lamUz * (hz^2 * uzz)^2)
        w.qU[1][i, j, k] = 2lU * wc * hx^4 * uxx / N
        w.qU[2][i, j, k] = 2lU * wc * hy^4 * uyy / N
        w.qU[3][i, j, k] = 2pb.lamUz * wc * hz^4 * uzz / N
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
    eQ = 0.0
    for k in 1:NZ+1, j in 1:b.ny
        eQ += pQ[j, k]
    end
    push!(loc, eQ)
    t = MPI.Allreduce(loc, +, par.comm)
    terms = (lF * (t[1] + t[2] + t[3]) / NF, lN * t[8] / N + pb.lamQ * pb.lamF * t[10] / NF, (t[4] + t[9]) / NB, lS * t[5] / N, t[7] / N)
    oob = t[6] > 0
    return oob ? Inf : sum(terms), terms, oob
end

# ---------------------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------------------
function gradient!(gx::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState; gauge::Bool = true)
    WORK_COUNTS[2] += 1
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
                         d2xT(w.qU[1], o.X, i, j, k) + d2yT(w.qU[2], o.Y, i, j, k) + d2zT(w.qU[3], o.Z, i, j, k)
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
    if pb.lamQ > 0
        qQ = w.qQ
        exchange_halos!(par, collect(qQ))
        owned_map!(gJ[1], par) do i, j, k
            gJ[1][i, j, k] + 0.25 * (qQ[1][i, j-1, k] - 2qQ[1][i, j, k] + qQ[1][i, j+1, k]) +
                0.25 * ((k >= 1 ? qQ[2][i, j, k-1] : 0.0) - 2qQ[2][i, j, k] + (k + 1 <= NZ ? qQ[2][i, j, k+1] : 0.0))
        end
        owned_map!(gJ[2], par) do i, j, k
            gJ[2][i, j, k] + 0.25 * (qQ[3][i-1, j, k] - 2qQ[3][i, j, k] + qQ[3][i+1, j, k]) +
                0.25 * ((k >= 1 ? qQ[4][i, j, k-1] : 0.0) - 2qQ[4][i, j, k] + (k + 1 <= NZ ? qQ[4][i, j, k+1] : 0.0))
        end
        owned_map!(gJ[3], par) do i, j, k
            gJ[3][i, j, k] + 0.25 * (qQ[5][i-1, j, k] - 2qQ[5][i, j, k] + qQ[5][i+1, j, k]) +
                0.25 * (qQ[6][i, j-1, k] - 2qQ[6][i, j, k] + qQ[6][i, j+1, k])
        end
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
    gauge || return gx                               # physical-u gradient (diagonal construction)
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
"""Gauss–Newton diagonal in physical u (force and λ_U terms), with the optional per-layer floor."""
function gn_diag_u!(D1::Field, pb::SMHSProblem, w::SMHSWork)
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    lF, NF, g = pb.lamF, pb.NF, pb.g
    Ax(i, j, k) = @inbounds lF * horeq(k, NZ) * wxf(pb, i, j, k) * pb.invS2f[1][i, j, k] / NF
    Ay(i, j, k) = @inbounds lF * horeq(k, NZ) * wyf(pb, i, j, k) * pb.invS2f[2][i, j, k] / NF
    Az(i, j, k) = @inbounds lF * wzf(pb, i, j, k) * pb.invS2f[3][i, j, k] / NF
    Ux(i, j, k) = @inbounds pb.lamU * wcc(pb, i, j, k) * o.X.h[i]^4 / pb.N
    Uy(i, j, k) = @inbounds pb.lamU * wcc(pb, i, j, k) * o.Y.h[j]^4 / pb.N
    Uz(i, j, k) = @inbounds pb.lamUz * wcc(pb, i, j, k) * s.hz[k]^4 / pb.N
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
        D1[i, j, k] = 2d + 2 * (d2xT2(Ux, o.X, i, j, k) + d2yT2(Uy, o.Y, i, j, k) + d2zT2(Uz, o.Z, i, j, k))
    end
    if pb.pc_ufloor > 0                              # per-layer floor: no node gets a huge D⁻¹∇Φ step in u
        med = layer_quantiles(par, D1, [0.5])[:, 1]
        MPI.Bcast!(med, 0, par.comm)
        fill_owned!((i, j, k) -> max(D1[i, j, k], pb.pc_ufloor * med[k]), D1, par)
    end
    return D1
end

"""
    gn_diag_v!(D1, pb, w, x)

Gauss–Newton diagonal in the optimizer coordinate v (u = v − ᾱ·v + ln P_ref): diag(Gᵀ H_u G) with
G = I − 1αᵀ. Off the anchor plane it equals the physical diagonal; on it,
D_i = (H_u)_ii − 2α_i (H_u 1)_i + α_i² (1ᵀ H_u 1), with H_u 1 from one gauge-free GN product
(Codex review M5, finding 3). The per-layer floor is applied afterwards. `w` must hold the state at x.
"""
function gn_diag_v!(D1::Field, pb::SMHSProblem, w::SMHSWork, x::MHSState)
    par = pb.par; b = par.block; ka = pb.ka
    uf = pb.pc_ufloor; pb.pc_ufloor = 0.0
    gn_diag_u!(D1, pb, w); pb.pc_ufloor = uf
    z = similar_state(x); Hz = similar_state(x); wl = SMHSWork(par)
    for c in 2:5
        fill!(parent(z[c]), 0.0)
    end
    fill_owned!((i, j, k) -> 1.0, z[1], par); exchange_halos!(par, [z[1]])
    gn_hvp!(Hz, pb, w, wl, z; gauge = false)
    S1 = sdot_ones(par, Hz[1])
    forrows(ka:ka, par) do i, j, k
        a = pb.alpha[i, j]
        D1[i, j, k] = max(D1[i, j, k] - 2a * Hz[1][i, j, k] + a^2 * S1, 1e-300)
    end
    if uf > 0
        med = layer_quantiles(par, D1, [0.5])[:, 1]
        MPI.Bcast!(med, 0, par.comm)
        fill_owned!((i, j, k) -> max(D1[i, j, k], uf * med[k]), D1, par)
    end
    return D1
end

function gn_diagonal!(D::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState; probes::Int = 4)
    par = pb.par; b = par.block; NZ = b.nz
    gn_diag_v!(D[1], pb, w, x)
    w2 = SMHSWork(par)
    z = similar_state(x); xt = similar_state(x); g1 = similar_state(x); g2 = similar_state(x)
    if pb.pc_atilde == :colored
        gn_diag_colored!(D, pb, w, x)
    else
    # Ã: Hessian–vector products by central differences of ∇Φ, averaged per layer
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

"""
    gn_diag_colored!(D, pb, w, x)

Exact Gauss–Newton diagonal for Ã by colored probing (M5): one `gn_hvp!` per color class of nodes
spaced `pb.pc_period` apart in i, j and k (and per component), so no two probed nodes share a row of
H_GN. A periodic axis whose length is not a multiple of the period gets extra colors for the
remainder. Uses the state of the last `objective!(pb, w, x)`.
"""
function gn_diag_colored!(D::MHSState, pb::SMHSProblem, w::SMHSWork, x::MHSState)
    par = pb.par; b = par.block; p = pb.pc_period
    NXg = b.i0 + b.nx - 1; NYg = b.j0 + b.ny - 1
    NXg = Int(MPI.Allreduce(NXg, max, par.comm)); NYg = Int(MPI.Allreduce(NYg, max, par.comm))
    col(gi, N) = (q = p * (N ÷ p); gi - 1 < q ? (gi - 1) % p : p + (gi - 1 - q))
    ncol(N) = N % p == 0 ? p : p + N % p
    wl = SMHSWork(par); z = similar_state(x); Hz = similar_state(x)
    for f in 2:4
        k0 = first(axes(x[f], 3))
        for ck in 0:p-1, cj in 0:ncol(NYg)-1, ci in 0:ncol(NXg)-1
            for q in 1:5
                fill!(parent(z[q]), 0.0)
            end
            any = 0
            for k in axes(x[f], 3), j in 1:b.ny, i in 1:b.nx
                if col(b.i0 + i - 1, NXg) == ci && col(b.j0 + j - 1, NYg) == cj && (k - k0) % p == ck
                    z[f][i, j, k] = 1.0; any += 1
                end
            end
            MPI.Allreduce(any, +, par.comm) == 0 && continue
            gn_hvp!(Hz, pb, w, wl, z)
            for k in axes(x[f], 3), j in 1:b.ny, i in 1:b.nx
                z[f][i, j, k] == 1.0 && (D[f][i, j, k] = Hz[f][i, j, k])
            end
        end
        mx = global_max(par, maximum(D[f][i, j, k] for k in axes(x[f], 3), j in 1:b.ny, i in 1:b.nx))
        owned_map!((i, j, k) -> max(D[f][i, j, k], 1e-14 * mx, 1e-300), D[f], par)
    end
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

A₀: fit (Ã, b̄) to Bobs with v frozen and λ_F = λ_U = λ_Uz = λ_Q = 0 (data and λ_S only). Φ_B + Φ_S is then quadratic in
(Ã, b̄), so this is conjugate gradients on the normal equations Q y = −∇Φ(0), with the
Hessian–vector product Q p = ∇Φ(p) − ∇Φ(0) (∇Φ is affine; one gradient per iteration).
Converges to the minimum-norm solution in the gauge null space of curl when started at 0.
"""
function init_A!(pb::SMHSProblem, w::SMHSWork, x::MHSState; max_iter::Int = 2000, rtol::Float64 = 1e-10)
    par = pb.par
    lF, lU, lUz, lQ = pb.lamF, pb.lamU, pb.lamUz, pb.lamQ
    pb.lamF = 0.0; pb.lamU = 0.0; pb.lamUz = 0.0; pb.lamQ = 0.0     # Φ_Q's (B_ref/S)² weights would wreck κ(Q)
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
        pb.lamF = lF; pb.lamU = lU; pb.lamUz = lUz; pb.lamQ = lQ
    end
end

# ---------------------------------------------------------------------------------------
# Solver policy (M6d)
# ---------------------------------------------------------------------------------------
"""Copies of the measured quantities (u = ln P at centres, B on faces) at the state of the last `objective!`."""
measured_snapshot(pb::SMHSProblem, w::SMHSWork) = (copy(parent(w.u)), map(B -> copy(parent(B)), w.B))

"""
    measured_change(pb, w, snap) -> max relative change since `snap`

max(max |ΔP/P|, max |ΔB|/max(|B|, σ)) over owned centres / faces; |ΔP/P| = |e^{Δu} − 1|.
"""
function measured_change(pb::SMHSProblem, w::SMHSWork, snap)
    par = pb.par; b = par.block; u0, B0 = snap
    pu = parent(w.u); ou = w.u.offsets
    m = 0.0
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        I = (i - ou[1], j - ou[2], k - ou[3])
        m = max(m, abs(expm1(pu[I...] - u0[I...])))
    end
    for c in 1:3
        Bc = w.B[c]; pB = parent(Bc); oB = Bc.offsets; iv = pb.invsf2[c]
        for k in axes(Bc, 3), j in 1:b.ny, i in 1:b.nx
            I = (i - oB[1], j - oB[2], k - oB[3])
            m = max(m, abs(pB[I...] - B0[c][I...]) / max(abs(pB[I...]), 1 / sqrt(iv[i, j, k])))
        end
    end
    return global_max(par, m)
end

"""
    layer_median_beta(pb, w) -> Vector

Per-layer median plasma β = 2μ₀P/|B|² at centres (B = HB, unfloored: B = 0 gives β = ∞; the 1 G floor
of §13.10 is for the output field), at the state of the last `objective!`.
"""
function layer_median_beta(pb::SMHSProblem, w::SMHSWork)
    HB = w.HB
    β(i, j, k) = @inbounds 2MU0 * w.P[i, j, k] / max(HB[1][i, j, k]^2 + HB[2][i, j, k]^2 + HB[3][i, j, k]^2, 1e-300)
    med = layer_quantiles(pb.par, β, [0.5])[:, 1]
    MPI.Bcast!(med, 0, pb.par.comm)
    return med
end

"""
    layer_median_lorentz(pb, w) -> Vector

Per-layer median of |L|/S at centres: L from the face Lorentz forces adjacent to the cell (x, y faces
on layers with horizontal equations, the z face below), each normalized by its face S, at the state
of the last `objective!`.
"""
function layer_median_lorentz(pb::SMHSProblem, w::SMHSWork)
    NZ = pb.par.block.nz; J = w.J; B = w.B
    function r(i, j, k)
        q = 0.0
        if horeq(k, NZ) > 0
            q += lor_x(J, B, i, j, k)[1]^2 * pb.invS2f[1][i, j, k] + lor_y(J, B, i, j, k)[1]^2 * pb.invS2f[2][i, j, k]
        end
        k <= NZ - 1 && (q += lor_z(J, B, i, j, k)[1]^2 * pb.invS2f[3][i, j, k])
        return sqrt(q)
    end
    med = layer_quantiles(pb.par, r, [0.5])[:, 1]
    MPI.Bcast!(med, 0, pb.par.comm)
    return med
end

"""
    mhs_solve!(pb, w, x; opts, lorentz_bfirst = 1.0, b_first_iter = 2000) -> (result, bfirst::Bool)

Plain joint L-BFGS, or B-first continuation (Ã relaxed with v frozen, then joint) when the start is
Lorentz-dominated: some layer has median |L|/S > `lorentz_bfirst` (M5, user decision 17 Oct 2026).
Only B can remove such a residual; the joint direction is then dominated by the vacuum direction in
u, which caps the step and leaves B unchanged (reports/M5a_realdata.md). Low β alone is not the
test: a uniform or potential field has β ≪ 1 and L = 0, and B-first then pushes the column-HSE
pressure error into a soft B mode (C1b/C3 references: 1.3e-9 T instead of 3e-13 T).
"""
function mhs_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState; opts::LBFGSOptions = LBFGSOptions(),
                    lorentz_bfirst::Float64 = 1.0, b_first_iter::Int = 2000)
    objective!(pb, w, x)
    if maximum(layer_median_lorentz(pb, w)) > lorentz_bfirst
        return lbfgs_continuation!(pb, w, x; b_first_iter = b_first_iter, opts = opts)[2], true
    end
    return lbfgs!(pb, w, x; opts = opts), false
end

"""
    discrepancy_solve!(pb, w, x; grid, bisect_ratio = 1.5, opts, lorentz_bfirst, b_first_iter)
        -> (x_selected, table, flags)

λ_F selection by the discrepancy principle (implementation doc §13.9). Continuation over `grid`
(increasing), each stage warm-started from the previous one and solved by `mhs_solve!`; χ² = Φ_B.
Selects the largest λ_F with χ² ≤ 1 + 3√(2/N_B), refined by log-bisection to `bisect_ratio`.
`x` is left at the selected solution and `pb.lamF` at the selected λ_F. `table` rows are
(λ_F, status, iterations, bfirst, χ², Φ terms); `flags` has `lambda_F`, `lambda_F_at_upper_bound`,
`lambda_F_at_lower_bound`, `thresh`.
"""
function discrepancy_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState;
                            grid::AbstractVector{<:Real} = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4],
                            bisect_ratio::Float64 = 1.5, opts::LBFGSOptions = LBFGSOptions(),
                            lorentz_bfirst::Float64 = 1.0, b_first_iter::Int = 2000)
    par = pb.par
    issorted(grid) || invalid("discrepancy_solve!: grid must be increasing")
    thresh = 1 + 3 * sqrt(2 / pb.NB)
    table = NamedTuple[]
    function stage!(λ, xs)
        pb.lamF = λ
        r, bf = mhs_solve!(pb, w, xs; opts = opts, lorentz_bfirst = lorentz_bfirst, b_first_iter = b_first_iter)
        t = objective!(pb, w, xs)[2]
        push!(table, (lambda_F = λ, status = r.status, iterations = r.iterations, bfirst = bf, chi2 = t[3], terms = t))
        return t[3] <= thresh
    end
    xok = nothing; xfirst = nothing; λok = NaN; λbad = NaN
    for λ in grid
        ok = stage!(Float64(λ), x)
        xfirst === nothing && (xfirst = deepcopy(x))
        if ok
            xok = deepcopy(x); λok = Float64(λ)
        else
            λbad = Float64(λ); break
        end
    end
    flags = Dict{String,Any}("thresh" => thresh, "lambda_F_at_upper_bound" => false, "lambda_F_at_lower_bound" => false)
    if xok === nothing
        # no stage passes: report the lowest λ_F stage (spec §13.9 step 2)
        flags["lambda_F_at_lower_bound"] = true
        λsel = Float64(first(grid))
        copystate!(x, xfirst, par); pb.lamF = λsel; objective!(pb, w, x)
        flags["lambda_F"] = λsel
        return x, table, flags
    end
    if isnan(λbad)
        flags["lambda_F_at_upper_bound"] = true
    else
        xt = similar_state(x)
        while λbad / λok > bisect_ratio
            λm = sqrt(λok * λbad)
            copystate!(xt, xok, par)
            if stage!(λm, xt)
                copystate!(xok, xt, par); λok = λm
            else
                λbad = λm
            end
        end
    end
    copystate!(x, xok, par); pb.lamF = λok; objective!(pb, w, x)
    flags["lambda_F"] = λok
    return x, table, flags
end

# ---------------------------------------------------------------------------------------
# Pressure identifiability (D9b §2.3)
# ---------------------------------------------------------------------------------------
"""
    dP_unobs!(out, pb, w) -> out

δP_unobs = |B| σ_B/μ₀ at centres (output `/diagnostics/dP_unobs`, Pa): pressure perturbations below
this are balanced by a δB the data cannot see. B is HB at the state of the last `objective!`;
σ_B = sqrt(σx² + σy² + σz²) (E10, as in screening).
"""
function dP_unobs!(out::Field, pb::SMHSProblem, w::SMHSWork)
    HB = w.HB; sg = pb.sig
    fill_owned!(out, pb.par) do i, j, k
        Bm = sqrt(HB[1][i, j, k]^2 + HB[2][i, j, k]^2 + HB[3][i, j, k]^2)
        Bm * sqrt(sg[1][i, j, k]^2 + sg[2][i, j, k]^2 + sg[3][i, j, k]^2) / MU0
    end
    return out
end

"""Per-layer fraction of nodes whose pressure is not identifiable (δP_unobs > P), at the last `objective!`."""
unidentifiable_fraction(pb::SMHSProblem, w::SMHSWork, dPu::Field) =
    layer_fraction(pb.par, (i, j, k) -> dPu[i, j, k] > w.P[i, j, k])

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
function sc1_problem(par::ParCtx, NX, NY, NZ; Bobs = (0.0, 0.0, 0.0), lambda_F = 1.0, lambda_N = 0.0, data_on_faces = true, lambda_Q = 1.0,
                     lambda_Uz = 0.0, P_ref = 1.0e4)
    grid, x, y, z = synthetic_grid(par, NX, NY, NZ)
    eos = IdealGasMu(1.3)
    T = newfield(par); fill_owned!((i, j, k) -> 6000 + 3000 * tanh((z[k] - 1.2e6) / 2.0e5), T, par)
    Bo = ntuple(c -> newfield(par; fill = Bobs[c]), 3)
    Bm = sqrt(sum(abs2, Bobs))
    sig = ntuple(_ -> newfield(par; fill = sigma_synth(Bm, 0.05, 5e-4)), 3)
    pb = SMHSProblem(par, grid, T, Bo, sig, eos; P_ref = P_ref, lambda_F = lambda_F, lambda_N = lambda_N, data_on_faces = data_on_faces, lambda_Q = lambda_Q, lambda_Uz = lambda_Uz)
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
