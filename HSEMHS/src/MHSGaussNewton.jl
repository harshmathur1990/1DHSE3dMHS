# Gauss–Newton / Levenberg–Marquardt with inner preconditioned CG for the staggered MHS problem
# (M6c remedy (i), reports/M6c_gate.md).
#
# Φ is a weighted sum of squares of residuals r(x): force R on faces (nonlinear), and the data and
# regularizer residuals, which are linear in (u, B). The Gauss–Newton product is
#     H_GN v = Jᵀ W J v,
# evaluated exactly: J v is the linearized residual (δu, δB = C(A_ref δÃ) + δb̄ ẑ, δJ = C*δB/μ0,
# δP = P δu, δρ̄ = ½ρ̄(ρ_u,k δu_k + ρ_u,k+1 δu_k+1), δL = M(δJ)×M(B) + M(J)×M(δB)), and Jᵀ W is the
# existing adjoint (`gradient!`) driven by seeds built from J v instead of r.

"""Work that shares the base-state fields (u, P, ρ, ρ_u) of `w` and uses the seeds/scratch of `wl`."""
hybrid_work(w::SMHSWork, wl::SMHSWork) =
    SMHSWork(w.u, w.P, w.rho, w.rhou, wl.A, wl.B, wl.J, wl.HB, wl.dB, wl.aR, wl.fX, wl.fY, wl.fZ,
             wl.qS, wl.qU, wl.qN, wl.gBd, wl.qQ, wl.gHB, wl.gB, wl.gBJ, wl.gJ, wl.gA, wl.gP)

"""
    gn_hvp!(Hv, pb, w, wl, v)

Hv = H_GN v at the state of the last `objective!(pb, w, x)`. `wl` is scratch work.
"""
function gn_hvp!(Hv::MHSState, pb::SMHSProblem, w::SMHSWork, wl::SMHSWork, v::MHSState; gauge::Bool = true)
    WORK_COUNTS[3] += 1
    par = pb.par; s = pb.s; o = pb.ops; b = par.block; NZ = b.nz
    lF, lS, lU, lN = pb.lamF, pb.lamS, pb.lamU, pb.lamN; N, NF, NB, g = pb.N, pb.NF, pb.NB, pb.g
    # ---- linearized state ----
    du = wl.u
    vm = gauge ? anchor_mean(pb, v[1]) : 0.0          # gauge = false: v is a physical δu
    fill_owned!((i, j, k) -> v[1][i, j, k] - vm, du, par)
    Ar = pb.Aref
    for c in 1:3
        vc = v[c+1]; owned_map!((i, j, k) -> Ar * vc[i, j, k], wl.A[c], par)
    end
    exchange_halos!(par, [du, wl.A...])
    dB = wl.B; dJ = wl.J
    curlA!(dB, wl.A, getscalar(par, v[5]), s, par)
    exchange_halos!(par, collect(dB))
    curlB!(dJ, dB, s, par)
    for c in 1:3
        Jc = dJ[c]; owned_map!((i, j, k) -> Jc[i, j, k] / MU0, Jc, par)
    end
    exchange_halos!(par, collect(dJ))
    dP = wl.P
    jr = axes(dP, 2); nj = length(jr)
    Threads.@threads :static for jk in 1:(nj*NZ)                 # (j, k) rows, halos included
        j = first(jr) + (jk - 1) % nj; k = (jk - 1) ÷ nj + 1
        @inbounds for i in axes(dP, 1)
            dP[i, j, k] = w.P[i, j, k] * du[i, j, k]
        end
    end
    B = w.B; J = w.J
    # ---- seeds: force term ----
    forrows(1:NZ, par) do i, j, k
        L, Jyf, Jzf, Bzf, Byf = lor_x(J, B, i, j, k)
        y = (dP[i+1, j, k] - dP[i, j, k]) / s.dxf[i] - lor_x(dJ, B, i, j, k)[1] - lor_x(J, dB, i, j, k)[1]
        a = 2lF * horeq(k, NZ) * wxf(pb, i, j, k) * pb.invS2f[1][i, j, k] * y / NF
        wl.aR[1][i, j, k] = a
        wl.fX[1][i, j, k] = -a * Bzf; wl.fX[2][i, j, k] = a * Byf
        wl.fX[3][i, j, k] = -a * Jyf; wl.fX[4][i, j, k] = a * Jzf
    end
    forrows(1:NZ, par) do i, j, k
        L, Jzf, Jxf, Bxf, Bzf = lor_y(J, B, i, j, k)
        y = (dP[i, j+1, k] - dP[i, j, k]) / s.dyf[j] - lor_y(dJ, B, i, j, k)[1] - lor_y(J, dB, i, j, k)[1]
        a = 2lF * horeq(k, NZ) * wyf(pb, i, j, k) * pb.invS2f[2][i, j, k] * y / NF
        wl.aR[2][i, j, k] = a
        wl.fY[1][i, j, k] = -a * Bxf; wl.fY[2][i, j, k] = a * Bzf
        wl.fY[3][i, j, k] = -a * Jzf; wl.fY[4][i, j, k] = a * Jxf
    end
    forrows(0:NZ, par) do i, j, k
        if 1 <= k <= NZ - 1
            L, Jxf, Jyf, Byf, Bxf = lor_z(J, B, i, j, k)
            rb = sqrt(w.rho[i, j, k] * w.rho[i, j, k+1])
            y = (dP[i, j, k] - dP[i, j, k+1]) / s.dzf[k] +
                0.5 * g * rb * (w.rhou[i, j, k] * du[i, j, k] + w.rhou[i, j, k+1] * du[i, j, k+1]) -
                lor_z(dJ, B, i, j, k)[1] - lor_z(J, dB, i, j, k)[1]
            a = 2lF * wzf(pb, i, j, k) * pb.invS2f[3][i, j, k] * y / NF
            wl.aR[3][i, j, k] = a
            wl.fZ[1][i, j, k] = -a * Byf; wl.fZ[2][i, j, k] = a * Bxf
            wl.fZ[3][i, j, k] = -a * Jxf; wl.fZ[4][i, j, k] = a * Jyf
        else
            wl.aR[3][i, j, k] = 0.0
            for q in 1:4
                wl.fZ[q][i, j, k] = 0.0
            end
        end
    end
    # ---- seeds: data and regularizers (linear residuals) ----
    H!(wl.HB, dB, s, par)
    exchange_halos!(par, collect(wl.HB))
    forrows(1:NZ, par) do i, j, k
        wc = wcc(pb, i, j, k)
        hx = o.X.h[i]; hy = o.Y.h[j]; hz = s.hz[k]
        for c in 1:3
            s2 = pb.sig[c][i, j, k]^2
            d = wl.HB[c][i, j, k]
            wl.gHB[c][i, j, k] = pb.dfaces ? 0.0 : 2d / (NB * s2)
            f = 2lS * wc / (N * s2)
            wl.qS[c][i, j, k] = f * hx^4 * d2x(wl.HB[c], o.X, i, j, k)
            wl.qS[3+c][i, j, k] = f * hy^4 * d2y(wl.HB[c], o.Y, i, j, k)
            wl.qS[6+c][i, j, k] = f * hz^4 * d2z(wl.HB[c], o.Z, i, j, k)
        end
        wl.qU[1][i, j, k] = 2lU * wc * hx^4 * d2x(du, o.X, i, j, k) / N
        wl.qU[2][i, j, k] = 2lU * wc * hy^4 * d2y(du, o.Y, i, j, k) / N
        wl.qU[3][i, j, k] = 2pb.lamUz * wc * hz^4 * d2z(du, o.Z, i, j, k) / N
    end
    Bx, By, Bz = dB; sg = pb.sig
    if lN > 0
        forrows(1:NZ, par) do i, j, k
            wl.qN[1][i, j, k] = 2lN * wxf(pb, i, j, k) * 0.25 * (Bx[i+1, j, k] - 2Bx[i, j, k] + Bx[i-1, j, k]) /
                                (sg[1][i, j, k] * sg[1][i+1, j, k] * N)
            wl.qN[2][i, j, k] = 2lN * wyf(pb, i, j, k) * 0.25 * (By[i, j+1, k] - 2By[i, j, k] + By[i, j-1, k]) /
                                (sg[2][i, j, k] * sg[2][i, j+1, k] * N)
        end
        forrows(0:NZ, par) do i, j, k
            wl.qN[3][i, j, k] = (1 <= k <= NZ - 1) ?
                2lN * wzf(pb, i, j, k) * 0.25 * (Bz[i, j, k-1] - 2Bz[i, j, k] + Bz[i, j, k+1]) / (sg[3][i, j, k] * sg[3][i, j, k+1] * N) : 0.0
        end
    else
        for c in 1:3
            fill!(parent(wl.qN[c]), 0.0)
        end
    end
    phiQ_seeds!(wl.qQ, dJ, pb)                       # Φ_Q is linear in J
    if pb.dfaces
        for c in 1:3
            dBc = dB[c]; iv = pb.invsf2[c]
            hg = pb.hinge
            if hg === nothing
                owned_map!((i, j, k) -> 2dBc[i, j, k] * iv[i, j, k] / NB, wl.gBd[c], par)
            else                                       # Φ_H: GN curvature 2λ σ⁻² on the faces active at the base state
                Bc = w.B[c]; Bo = pb.Bobsf[c]; sf = hg.soft[c]; lH = hg.lam
                owned_map!(wl.gBd[c], par) do i, j, k
                    cf = (Bc[i, j, k] - Bo[i, j, k]) * sqrt(iv[i, j, k])
                    extra = if hg.barrier                  # ½ λ φ''(c), φ = −log(1 − c²/u²)
                        u2 = sf[i, j, k]^2; q = u2 - cf^2
                        q > 0 ? lH * (u2 + cf^2) / q^2 : 0.0
                    else
                        abs(cf) > sf[i, j, k] ? lH : 0.0
                    end
                    2dBc[i, j, k] * iv[i, j, k] * (1 + extra) / NB
                end
            end
        end
    end
    return gradient!(Hv, pb, hybrid_work(w, wl), v; gauge = gauge)
end

Base.@kwdef struct GNOptions
    max_outer::Int = 300
    max_inner::Int = 300
    eta::Float64 = 1e-3              # inner PCG relative residual
    gtol_rel::Float64 = 1e-10
    mu0::Float64 = 1e-3              # LM damping, relative to the GN diagonal
    max_du::Float64 = 0.5            # largest |Δ ln P| per accepted step
    collapse_du::Float64 = 3.0       # safeguard: reject steps with min(u − u_start) < −collapse_du
    max_reject::Int = 30
    verbose::Bool = false
end

"""
    gncg!(pb, w, x; opts) -> LBFGSResult

Levenberg–Marquardt: solve (H_GN + μ D) s = −∇Φ by PCG (preconditioner (1+μ)D, D the GN
diagonal), cap |Δu| ≤ max_du, reject steps that drop u more than collapse_du below the starting
u anywhere (pressure-collapse safeguard), accept on ρ = actual/predicted > 1e-4, adapt μ.
"""
function gncg!(pb::SMHSProblem, w::SMHSWork, x::MHSState; opts::GNOptions = GNOptions())
    par = pb.par; b = par.block
    wl = SMHSWork(par); wt = SMHSWork(par)
    gx = similar_state(x); xt = similar_state(x); sx = similar_state(x)
    r = similar_state(x); z = similar_state(x); p = similar_state(x); q = similar_state(x); Hs = similar_state(x)
    f, terms, oob = fg!(gx, pb, w, x)
    oob && return LBFGSResult(eos_failure, "initial state outside the EOS envelope", 0, 1, f, terms, NaN, NaN, NTuple{4,Float64}[])
    ustart = deepcopy(w.u)
    g0 = sqrt(dotN(par, gx, gx)); gn = g0
    μ = opts.mu0
    hist = NTuple{4,Float64}[]
    it = 0; nev = 1; nrej = 0; status = max_iter; reason = "max_outer"
    D = gn_diagonal!(similar_state(x), pb, w, x)
    while it < opts.max_outer
        if gn <= opts.gtol_rel * g0
            status = converged; reason = "gradient norm ≤ gtol_rel·‖∇Φ₀‖"; break
        end
        it += 1
        # ---- PCG on (H + μD) s = −g ----
        for f_ in eachindex(sx); fill!(parent(sx[f_]), 0.0); end
        lincomb!(r, -1.0, gx, 0.0, gx, par)
        for f_ in eachindex(z)
            rf = r[f_]; Df = D[f_]; owned_map!((i, j, k) -> rf[i, j, k] / ((1 + μ) * Df[i, j, k]), z[f_], par)
        end
        lincomb!(p, 1.0, z, 0.0, z, par)
        rz = dotN(par, r, z); r0 = sqrt(dotN(par, r, r)); ninner = 0
        while ninner < opts.max_inner
            ninner += 1
            gn_hvp!(q, pb, w, wl, p)
            for f_ in eachindex(q)
                qf = q[f_]; pf = p[f_]; Df = D[f_]
                owned_map!((i, j, k) -> qf[i, j, k] + μ * Df[i, j, k] * pf[i, j, k], qf, par)
            end
            pq = dotN(par, p, q)
            pq > 0 || break
            α = rz / pq
            axpy!(sx, α, p, par); axpy!(r, -α, q, par)
            sqrt(dotN(par, r, r)) <= opts.eta * r0 && break
            for f_ in eachindex(z)
                rf = r[f_]; Df = D[f_]; owned_map!((i, j, k) -> rf[i, j, k] / ((1 + μ) * Df[i, j, k]), z[f_], par)
            end
            rzn = dotN(par, r, z)
            lincomb!(p, 1.0, z, rzn / rz, p, par)
            rz = rzn
        end
        # ---- step cap, predicted reduction ----
        dmu = maxabs_all(par, sx[1])
        αc = dmu > opts.max_du ? opts.max_du / dmu : 1.0
        gn_hvp!(Hs, pb, w, wl, sx)
        gs = dotN(par, gx, sx); sHs = dotN(par, sx, Hs)
        pred = -(αc * gs + 0.5 * αc^2 * sHs)
        lincomb!(xt, 1.0, x, αc, sx, par)
        ft, tt, oobt = objective!(pb, wt, xt); nev += 1
        collapse = !oobt && global_max(par, maximum(ustart[i, j, k] - wt.u[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)) > opts.collapse_du
        ρ = (oobt || collapse || pred <= 0) ? -Inf : (f - ft) / pred
        if ρ > 1e-4
            copystate!(x, xt, par)
            f, terms, _ = fg!(gx, pb, w, x); nev += 1
            gn = sqrt(dotN(par, gx, gx))
            μ = ρ > 0.75 ? max(μ / 3, 1e-12) : ρ < 0.25 ? 4μ : μ
            nrej = 0
            D = gn_diagonal!(D, pb, w, x)
        else
            μ *= 4; nrej += 1
            if nrej > opts.max_reject
                status = stalled; reason = collapse ? "steps rejected by the pressure-collapse safeguard" : "repeated step rejection"; break
            end
        end
        push!(hist, (f, gn, αc, Float64(ninner)))
        opts.verbose && rank0(par) && @printf("GN %4d  Φ %.4e  ‖g‖ %.3e  μ %.1e  inner %3d  ρ %.3f%s\n", it, f, gn, μ, ninner, ρ, collapse ? " (collapse)" : "")
    end
    if status == max_iter && gn <= opts.gtol_rel * g0
        status = converged; reason = "gradient norm ≤ gtol_rel·‖∇Φ₀‖"
    end
    return LBFGSResult(status, reason, it, nev, f, terms, g0, gn, hist)
end
