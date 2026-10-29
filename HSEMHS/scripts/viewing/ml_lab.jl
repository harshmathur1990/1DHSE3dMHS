#=
Two-level preconditioner prototype for mhs-poisson-ltau on frozen Gauss–Newton systems
(design/CODEX_REVIEW_solver_time_round5_2026-10-29.md, steps A–C). One rank.

make: julia ... ml_lab.jl make INPUT.nc OUT.h5 MU AZIMUTH_DEG LOGTAU_CUT BALANCE STEPS [crop=NC]
      the state after STEPS steps from column HSE (Fourier, gain LM, EW forcing: the reference trajectory), saved with
      its objective: u, z (all levels, Julia (τ, y, x)), wc, wf, ereg, and the input, crop window (global x, y indices),
      view, cut, balance, steps, Φ/Φ₀ as attributes. crop = NC: an NC × NC periodic window around the strongest |B(τ=1)|;
      tile = n: n × n periodic copies of that window (the same local state on an n× larger domain: domain growth).
lab:  julia ... ml_lab.jl lab FROZEN.h5 [KEY=VALUE ...]
      the frozen system A δ = b (A = JᵀWJ + diag(ereg) + mlm·diag(JᵀWJ), b = −JᵀWR at the saved state and weights) with
      every preconditioner of `precs` (comma list):
        fourier, block, hybrid      the solver's (hybrid: the corrected D3c baseline)
        add:a / mult:a              two-level, horizontal aggregates of a × a whole columns, coarse space = every level of
                                    the aggregate (piecewise constant in x, y), Galerkin A_c = Pᵀ A P probed from the actual
                                    operator (coloured probes), sparse Cholesky; add: M⁻¹ = B⁻¹ + P A_c⁻¹ Pᵀ; mult: damped
                                    block Jacobi pre-smoothing, coarse correction, post-smoothing (symmetric; SPD for
                                    ω λmax(B⁻¹A) < 2: ω = `omega`/λs, λs = 1.2 × the B-Rayleigh
                                    estimate capped at the provable bound 13; omega_safe = 1: λs = 13, proven)
        add:a:k / mult:a:k          the same with k column shapes per aggregate: the k smallest eigenvectors of the
                                    aggregate's diagonal block of A_c (compact coarse space, A_c' = Vᵀ A_c V)
        addb:a[:k] / multb:a[:k]    four branches per aggregate: the smooth basis and its (−1)ⁱ, (−1)ʲ, (−1)ⁱ⁺ʲ modulations
                                    (the centred differences' Nyquist branches; a even)
      For each: setup time, PCG to tol 1e-3 and 1e-5 (iterations, A and M⁻¹ applications, the true relative residual
      recomputed, solve time), and against a tight reference δ (block PCG to 1e-10): scaled ‖δ − δ_ref‖ / ‖δ_ref‖ and
      the model decrease δ·b/Φ against the reference's.
KEY: precs (default fourier,block,hybrid,add:2,mult:2,mult:4,mult:2:8,mult:4:8), mlm (Marquardt damping, 0), omega (1.0), omega_safe (0),
     explicit (1: dense M⁻¹ and A by probes on systems ≤ 6000 unknowns: symmetry and positivity of M⁻¹ and of M⁻¹A),
     gauge (1: the cut-height checkerboard and constant modes: ‖A q‖/‖q‖ and the error reduction ‖(I − M⁻¹A) q‖_A per
     preconditioner), ref_tol (1e-10), rhs (b: the Gauss–Newton right-hand side; smooth: random modes |k| ≤ 4 of the whole
     domain on every level; rand: white noise; both seeded and scaled to ‖b‖ — for domain-growth tests on tiled systems,
     whose b would keep every Krylov iterate periodic with the tile; δ err and pred then refer to that rhs), assembly (probe:
     A_c by coloured probes of the operator; analytic: from the per-column Jacobian blocks, galerkin_analytic), galcheck
     (1: also the other assembly, and their difference), max_coarse (60000: two-level variants with more coarse unknowns are skipped).
=#
using HSEMHS
using HDF5
using MPI
using LinearAlgebra
using SparseArrays
using Printf
using Random

const H = HSEMHS
const PT = H.PoissonTau
const HREF = 1e5

function window(lt, Bx, By, Bz, NC)
    k0 = argmin(abs.(lt)); _, NY, NX = size(Bx)
    _, jc, ic = Tuple(argmax(Bx[k0:k0, :, :] .^ 2 .+ By[k0:k0, :, :] .^ 2 .+ Bz[k0:k0, :, :] .^ 2))
    return mod1.(jc - NC ÷ 2 .+ (0:NC-1), NY), mod1.(ic - NC ÷ 2 .+ (0:NC-1), NX)
end

function load_input(path, J, I)
    par = H.slab_par(PT.tau_input_size(path)...)
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    H.free_parallel!(par)
    J === nothing && (J = 1:size(T, 2); I = 1:size(T, 3))
    return lt, x[2] - x[1], y[2] - y[1], log.(T[:, J, I]), Bx[:, J, I], By[:, J, I], Bz[:, J, I], J, I
end

function make(path, out, mu_s, az_s, cut_s, bal, steps_s, kv...)
    H.ensure_mpi()
    o = Dict("crop" => 0, "tile" => 1); for a in kv; k, v = split(a, "="); o[k] = parse(Int, v); end
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = :inclined)
    cut = parse(Float64, cut_s); nst = parse(Int, steps_s)
    J = I = nothing
    if o["crop"] > 0
        par = H.slab_par(PT.tau_input_size(path)...)
        lt, _, _, _, Bx, By, Bz = PT.read_tau_input(par, path); H.free_parallel!(par)
        J, I = window(lt, Bx, By, Bz, o["crop"])
        J = repeat(J, o["tile"]); I = repeat(I, o["tile"])   # tile × tile periodic copies of the crop (pure domain growth)
    end
    lt, dx, dy, lT, Bx, By, Bz, J, I = load_input(path, J, I)
    _, ny, nx = size(lT); nt = length(lt)
    par = H.slab_par(nx, ny, nt); tab = H.load_table("cache/eos_table_fullz.h5"); g = 274.0
    lnP, z0, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed")
    u, z, Φ0, Φ, s, kc, info = PT.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, dx, dy; logtau_cut = cut,
                                            max_steps = nst, mu = v.mu, slant = H.slant(v), stop = :phi, rtol = 0.0,
                                            balance = Symbol(bal), lm = :gain, forcing = :ew, eta_max = 0.01, cg_cap = 400,
                                            stationarity = false, logf = s -> println(s))
    W = info.weights
    h5open(out, "w") do f
        f["u"] = u; f["z"] = z; f["wc"] = W.wc; f["wf"] = W.wf; f["ereg"] = W.ereg; f["J"] = collect(J); f["I"] = collect(I)
        a = attributes(f)
        a["input"] = abspath(path); a["mu"] = v.mu; a["azimuth_deg"] = v.azimuth_deg; a["logtau_cut"] = cut
        a["balance"] = bal; a["steps"] = s; a["phi_rel"] = Φ / Φ0; a["crop"] = o["crop"]; a["tile"] = o["tile"]; a["kc"] = kc
    end
    @printf("wrote %s: %d × %d × %d, %d steps, Φ/Φ₀ %.3e\n", out, ny, nx, nt, s, Φ / Φ0)
end

# ---------------------------------------------------------------------------------------------------------------------
# two-level preconditioner
# ---------------------------------------------------------------------------------------------------------------------
mutable struct TwoLevel
    a::Int; nay::Int; nax::Int; k::Int            # aggregate size, aggregates in y and x, shapes per group (0: all n1)
    nq::Int                                       # branches: 1 (smooth) or 4 (smooth, (−1)ⁱ, (−1)ʲ, (−1)ⁱ⁺ʲ modulated)
    V::Vector{Matrix{Float64}}                    # per aggregate: n1 × k shapes (k > 0)
    F::Any                                        # Cholesky of A_c
    mult::Bool; ω::Float64
    nA::Int                                       # A applications made inside the preconditioner
    tsetup::Float64; nc::Int; nnzA::Int; nnzF::Int; sym::Float64
    Ac::Any                                       # the (symmetrised) coarse operator, for the explicit check
    tassemble::Float64                            # seconds to assemble A_c
    cmp::Tuple{Float64,Float64}                   # check: ‖A_c − other assembly‖₁/‖·‖₁ and its seconds
end

agg(T::TwoLevel, j, i) = (cld(j, T.a), cld(i, T.a))
aidx(T::TwoLevel, by, bx) = by + T.nay * (bx - 1)
gidx(T::TwoLevel, q, by, bx) = q + T.nq * (aidx(T, by, bx) - 1)          # group (branch q of aggregate (by, bx))
sgn(q, j, i) = q == 1 ? 1.0 : q == 2 ? (-1.0)^i : q == 3 ? (-1.0)^j : (-1.0)^(i + j)
ngroups(T::TwoLevel) = T.nq * T.nay * T.nax

function restrict(T::TwoLevel, r)                  # Pᵀ r
    n1 = size(r, 1); S = zeros(n1, ngroups(T))
    @inbounds for i in axes(r, 3), j in axes(r, 2)
        by, bx = agg(T, j, i)
        for q in 1:T.nq
            g = gidx(T, q, by, bx); sg = sgn(q, j, i)
            for m in 1:n1; S[m, g] += sg * r[m, j, i]; end
        end
    end
    T.k == 0 && return vec(S)
    y = zeros(T.k * ngroups(T))
    for g in 1:ngroups(T); y[(g-1)*T.k+1:g*T.k] = T.V[g]' * @view S[:, g]; end
    return y
end

function prolong!(T::TwoLevel, out, y, n1)         # out += P y
    if T.k == 0
        Y = reshape(y, n1, ngroups(T))
    else
        Y = zeros(n1, ngroups(T))
        for g in 1:ngroups(T); Y[:, g] = T.V[g] * y[(g-1)*T.k+1:g*T.k]; end
    end
    @inbounds for i in axes(out, 3), j in axes(out, 2)
        by, bx = agg(T, j, i)
        for q in 1:T.nq
            g = gidx(T, q, by, bx); sg = sgn(q, j, i)
            for m in 1:n1; out[m, j, i] += sg * Y[m, g]; end
        end
    end
    return out
end

"""Galerkin A_c = Pᵀ A P (every level of each aggregate) by coloured probes: aggregates c apart in each direction share a
probe (A couples columns ≤ 2 apart, i.e. aggregates ≤ 1 apart for a ≥ 2)."""
function galerkin(pr, T::TwoLevel)
    n1 = pr.n1; nay, nax = T.nay, T.nax; nc = n1 * ngroups(T)
    col(n) = n < 3 ? n : first(c for c in 3:n if n % c == 0)
    cy, cx = col(nay), col(nax)
    Is = Int[]; Js = Int[]; Vs = Float64[]
    v = zeros(n1, pr.ny, pr.nx); w = similar(v)
    for oy in 1:cy, ox in 1:cx, q in 1:T.nq, m in 1:n1
        fill!(v, 0.0)
        for i in 1:pr.nx, j in 1:pr.ny
            by, bx = agg(T, j, i)
            ((by - oy) % cy == 0 && (bx - ox) % cx == 0) && (v[m, j, i] = sgn(q, j, i))
        end
        PT.Aop!(pr, w, v)
        S = zeros(n1, ngroups(T))
        for i in 1:pr.nx, j in 1:pr.ny
            by, bx = agg(T, j, i)
            for q2 in 1:T.nq
                g = gidx(T, q2, by, bx); sg = sgn(q2, j, i)
                for l in 1:n1; S[l, g] += sg * w[l, j, i]; end
            end
        end
        for bx in 1:nax, by in 1:nay                # target aggregate (by, bx): its source in this colour is the one
            ys_ = unique(filter(yy -> (yy - oy) % cy == 0, [mod1(by + d, nay) for d in -1:1]))   # within ±1 (periodic)
            xs_ = unique(filter(xx -> (xx - ox) % cx == 0, [mod1(bx + d, nax) for d in -1:1]))
            (isempty(ys_) || isempty(xs_)) && continue
            (length(ys_) == 1 && length(xs_) == 1) || error("galerkin: colouring not unique")
            src = gidx(T, q, ys_[1], xs_[1])
            for q2 in 1:T.nq, l in 1:n1
                val = S[l, gidx(T, q2, by, bx)]; val == 0 && continue
                push!(Is, l + n1 * (gidx(T, q2, by, bx) - 1)); push!(Js, m + n1 * (src - 1)); push!(Vs, val)
            end
        end
    end
    return sparse(Is, Js, Vs, nc, nc)
end


# ---------------------------------------------------------------------------------------------------------------------
# analytic Galerkin assembly (one rank): A_c = Σ_R (J_R P)ᵀ W_R (J_R P) + Pᵀ diag(ereg + mlm·dlm) P, J_RP the Jacobian of
# the residual rows of column R with respect to the unknowns of column P (P = R or one of its 4 neighbours), built as in
# PoissonTau.colblocks! (rows stored as columns: J[l, r] = ∂R_r/∂v_l)
# ---------------------------------------------------------------------------------------------------------------------
"""δz map Zm (Zm[:, m] = row m of the δz map, transposed) and ζ map Zt of column (j, i), as in colblocks!."""
function colmaps!(Zm, Zt, pr, j, i, hf)
    n1 = pr.n1; fill!(Zm, 0.0); Zm[1, 1] = 1.0
    @inbounds for f in 1:n1-1
        for l in 1:f; Zm[l, f+1] = Zm[l, f]; end
        f >= 2 && (Zm[f, f+1] += hf[f] * pr.aχ[f, j, i])
        Zm[f+1, f+1] += hf[f] * pr.aχ[f+1, j, i]
    end
    @inbounds for m in 1:n1
        a, b = m == n1 ? (n1 - 1, n1) : (m == 1 ? 1 : m - 1, m + 1)
        d = pr.zden[m, j, i]
        for l in 1:n1; Zt[l, m] = (Zm[l, a] - Zm[l, b]) / d; end
    end
end

"""J_RP into (Jx, Jy, Jz) (n1 × n1, n1 × n1, n1 × n1−1); R = (jR, iR), P = (jP, iP) with P's maps Zm, Zt."""
function jac_RP!(Jx, Jy, Jz, T_, pr, jR, iR, jP, iP, Zm, Zt, hf)
    n1, ny, nx = pr.n1, pr.ny, pr.nx; vt = PT.vertical(pr); tx = pr.tx; ty = pr.ty
    @inbounds if jR == jP && iR == iP
        j, i = jR, iR
        for m in 1:n1, l in 1:n1
            Jx[l, m] = pr.px[m, j, i] * Zt[l, m]; Jy[l, m] = pr.py[m, j, i] * Zt[l, m]; T_[l, m] = pr.zζ[m, j, i] * Zt[l, m]
        end
        for m in 2:n1; Jx[m, m] += pr.lx[m, j, i]; Jy[m, m] += pr.ly[m, j, i]; end
        for f in 1:n1-1
            for l in 1:n1; Jz[l, f] = hf[f] * (T_[l, f] + T_[l, f+1]); end
            f >= 2 && (Jz[f, f] += -1 - hf[f] * pr.dw[f, j, i])
            Jz[f+1, f] += 1 - hf[f] * pr.dw[f+1, j, i]
            if vt
                q = pr.vq[f, j, i]; vd = 0.5 * pr.vd[f, j, i]
                for l in 1:n1
                    Jz[l, f] -= (Zm[l, f] - Zm[l, f+1]) * q + vd * (tx * (Jx[l, f] + Jx[l, f+1]) + ty * (Jy[l, f] + Jy[l, f+1]))
                end
            end
        end
    else
        j, i = jR, iR                                          # the coefficients are R's
        if jR == jP
            s = (iR == mod1(iP + 1, nx) ? -1 : 1) / (2pr.dx)
            c1 = view(pr.pXx, :, j, i); c2 = view(pr.pXy, :, j, i); cT = view(pr.zX, :, j, i); Ja, Jb = Jx, Jy
        else
            s = (jR == mod1(jP + 1, ny) ? -1 : 1) / (2pr.dy)
            c1 = view(pr.pYy, :, j, i); c2 = view(pr.pYx, :, j, i); cT = view(pr.zY, :, j, i); Ja, Jb = Jy, Jx
        end
        for m in 1:n1
            ca = s * (c1[m] - pr.W[m, j, i]); cb = s * c2[m]; ct = s * cT[m]
            for l in 1:n1; Ja[l, m] = ca * Zm[l, m]; Jb[l, m] = cb * Zm[l, m]; T_[l, m] = ct * Zm[l, m]; end
            m >= 2 && (Ja[m, m] -= s)
        end
        for f in 1:n1-1
            vd = vt ? 0.5 * pr.vd[f, j, i] : 0.0
            for l in 1:n1
                Jz[l, f] = hf[f] * (T_[l, f] + T_[l, f+1]) - vd * (tx * (Jx[l, f] + Jx[l, f+1]) + ty * (Jy[l, f] + Jy[l, f+1]))
            end
        end
    end
    return nothing
end

function galerkin_analytic(pr, T::TwoLevel)
    n1, ny, nx = pr.n1, pr.ny, pr.nx; nq = T.nq; ng = ngroups(T); nagg = T.nay * T.nax
    hf = [pr.dl[PT.node(pr, f)] / 2 for f in 1:n1-1]
    blocks = [Dict{Int,Array{Float64,4}}() for _ in 1:nagg]   # row aggregate → column aggregate → (n1, n1, nq, nq)
    locks = [ReentrantLock() for _ in 1:nagg]
    wc = pr.wc; wf = pr.wf
    Threads.@threads for iR in 1:nx
        Zm = [zeros(n1, n1) for _ in 1:5]; Zt = [zeros(n1, n1) for _ in 1:5]; T_ = zeros(n1, n1)
        Jx = [zeros(n1, n1) for _ in 1:5]; Jy = [zeros(n1, n1) for _ in 1:5]; Jz = [zeros(n1, n1 - 1) for _ in 1:5]
        Wx = [zeros(n1, n1) for _ in 1:5]; Wy = [zeros(n1, n1) for _ in 1:5]; Wz = [zeros(n1, n1 - 1) for _ in 1:5]
        B = zeros(n1, n1)
        for jR in 1:ny
            st = ((jR, iR), (jR, mod1(iR - 1, nx)), (jR, mod1(iR + 1, nx)), (mod1(jR - 1, ny), iR), (mod1(jR + 1, ny), iR))
            for (k, (jP, iP)) in enumerate(st)
                colmaps!(Zm[k], Zt[k], pr, jP, iP, hf)
                jac_RP!(Jx[k], Jy[k], Jz[k], T_, pr, jR, iR, jP, iP, Zm[k], Zt[k], hf)
                Wx[k] .= Jx[k] .* wc'; Wy[k] .= Jy[k] .* wc'; Wz[k] .= Jz[k] .* wf'
            end
            for (k, (jP, iP)) in enumerate(st), (k2, (jP2, iP2)) in enumerate(st)
                mul!(B, Wx[k], Jx[k2]'); mul!(B, Wy[k], Jy[k2]', 1.0, 1.0); mul!(B, Wz[k], Jz[k2]', 1.0, 1.0)
                by, bx = agg(T, jP, iP); by2, bx2 = agg(T, jP2, iP2)
                g = aidx(T, by, bx); g2 = aidx(T, by2, bx2)
                lock(locks[g]) do
                    C = get!(() -> zeros(n1, n1, nq, nq), blocks[g], g2)
                    for q in 1:nq, q2 in 1:nq
                        sg = sgn(q, jP, iP) * sgn(q2, jP2, iP2)
                        @views C[:, :, q, q2] .+= sg .* B
                    end
                end
            end
        end
    end
    for i in 1:nx, j in 1:ny                                   # Pᵀ diag(ereg + mlm dlm) P
        by, bx = agg(T, j, i); g = aidx(T, by, bx)
        C = get!(() -> zeros(n1, n1, nq, nq), blocks[g], g)
        for q in 1:nq, q2 in 1:nq
            sg = sgn(q, j, i) * sgn(q2, j, i)
            for m in 1:n1; C[m, m, q, q2] += sg * (pr.ereg[m] + pr.mlm * pr.dlm[m, j, i]); end
        end
    end
    Is = Int[]; Js = Int[]; Vs = Float64[]
    for g in 1:nagg, (g2, C) in blocks[g], q in 1:nq, q2 in 1:nq, l2 in 1:n1, l in 1:n1
        v = C[l, l2, q, q2]; v == 0 && continue
        push!(Is, l + n1 * (q + nq * (g - 1) - 1)); push!(Js, l2 + n1 * (q2 + nq * (g2 - 1) - 1)); push!(Vs, v)
    end
    return sparse(Is, Js, Vs, n1 * ng, n1 * ng)
end

function twolevel(pr, a, k, mult, ω; nq = 1, assembly = :probe, check = false)
    t0 = time()
    T = TwoLevel(a, cld(pr.ny, a), cld(pr.nx, a), k, nq, Matrix{Float64}[], nothing, mult, ω, 0, 0.0, 0, 0, 0, NaN, nothing,
                 0.0, (NaN, NaN))
    (pr.ny % a == 0 && pr.nx % a == 0) || error("grid $(pr.ny) × $(pr.nx) not divisible by aggregate size $a")
    nq == 4 && isodd(a) && error("branch coarse spaces need an even aggregate size")
    Ac = assembly == :analytic ? galerkin_analytic(pr, T) : galerkin(pr, T)
    T.tassemble = time() - t0
    if check                                                   # the other assembly, for the comparison
        t1 = time(); Ac2 = assembly == :analytic ? galerkin(pr, T) : galerkin_analytic(pr, T)
        T.cmp = (norm(Ac - Ac2, 1) / norm(Ac2, 1), time() - t1)
    end
    T.sym = norm(Ac - Ac', 1) / norm(Ac, 1)
    Ac = (Ac + Ac') / 2
    n1 = pr.n1
    if k > 0
        nb = ngroups(T); Vb = Vector{Matrix{Float64}}(undef, nb)
        for b in 1:nb
            r = (b-1)*n1+1:b*n1
            E = eigen(Symmetric(Matrix(Ac[r, r])))
            Vb[b] = E.vectors[:, 1:k]
        end
        T.V = Vb
        Vs = blockdiag((sparse(V) for V in Vb)...)
        Ac = Vs' * Ac * Vs; Ac = (Ac + Ac') / 2
    end
    T.nc = size(Ac, 1); T.nnzA = nnz(Ac); T.Ac = Ac
    T.F = cholesky(Ac)
    T.nnzF = nnz(T.F)
    T.tsetup = time() - t0
    return T
end

function ml_apply!(pr, T::TwoLevel, out, r)
    n1 = pr.n1
    if !T.mult
        PT.block_solve!(out, pr.blk, r)
        prolong!(T, out, T.F \ restrict(T, r), n1)
        return out
    end
    PT.block_solve!(out, pr.blk, r); out .*= T.ω
    t = similar(r); PT.Aop!(pr, t, out); T.nA += 1
    prolong!(T, out, T.F \ restrict(T, r .- t), n1)
    PT.Aop!(pr, t, out); T.nA += 1
    s = similar(r); PT.block_solve!(s, pr.blk, r .- t)
    @. out += T.ω * s
    return out
end

"""Largest eigenvalue of B⁻¹A (B the column blocks; B⁻¹A is B-self-adjoint, real positive spectrum): power iteration
x ← B⁻¹A x (n steps) with the B-Rayleigh quotient xᵀAx / xᵀBx (B x_k+1 = A x_k, so xᵀBx needs no B application), a
lower bound of λmax. Also the provable upper bound: A couples a column only to the columns of its residuals' centred
stencils (Manhattan distance ≤ 2: LMAX_BOUND = 13 blocks per block row), and by block Cauchy–Schwarz
xᵀAx ≤ 13 xᵀBx."""
const LMAX_BOUND = 13
function lmax_block(pr, n)
    Random.seed!(3); x = randn(size(pr.Rx)); y = similar(x); zz = similar(x); λ = 0.0
    PT.Aop!(pr, y, x)
    for _ in 1:n
        PT.block_solve!(zz, pr.blk, y); s = norm(zz); zz ./= s       # x_k+1 = B⁻¹A x_k / s; B x_k+1 = A x_k / s
        xBx = dot(zz, y) / s
        x .= zz; PT.Aop!(pr, y, x)
        λ = dot(x, y) / xBx
    end
    return λ
end

function lab(path, kv...)
    H.ensure_mpi()
    o = Dict{String,Any}("precs" => "fourier,block,hybrid,add:2,mult:2,mult:4,mult:2:8,mult:4:8", "mlm" => 0.0,
                         "omega" => 1.0, "omega_safe" => 0, "explicit" => 1, "gauge" => 1, "ref_tol" => 1e-10, "rhs" => "b",
                         "assembly" => "probe", "galcheck" => 0,
                         "max_coarse" => 60000)
    for a in kv
        k, v = split(a, "="); haskey(o, k) || error("unknown key $k")
        o[k] = o[k] isa String ? String(v) : o[k] isa Int ? parse(Int, v) : parse(Float64, v)
    end
    f = h5open(path, "r"); at = attributes(f)
    u = read(f["u"]); z = read(f["z"]); wc = read(f["wc"]); wf = read(f["wf"]); ereg = read(f["ereg"])
    J = read(f["J"]); I = read(f["I"]); inp = read(at["input"]); mu = read(at["mu"]); az = read(at["azimuth_deg"])
    cut = read(at["logtau_cut"]); bal = Symbol(read(at["balance"])); close(f)
    lt, dx, dy, lT, Bx, By, Bz, _, _ = load_input(inp, J, I)
    _, ny, nx = size(lT); nt = length(lt)
    par = H.slab_par(nx, ny, nt); tab = H.load_table("cache/eos_table_fullz.h5"); g = 274.0
    v = H.ViewOptions(mu = mu, azimuth_deg = az, geometry = :inclined)
    t0 = time(); logf(s) = (@printf("[%7.1fs] %s\n", time() - t0, s); flush(stdout))
    BLAS.set_num_threads(1)
    function frozen(pc)                                  # all locals: `pr` of lab must not be rebound
        local p, c, Φf, rh, bad
        p, c = PT.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z, dx, dy; logtau_cut = cut, mu = mu, slant = H.slant(v),
                            balance = bal, precond = pc)
        p.wc = copy(wc); p.wf = copy(wf); p.ereg = copy(ereg); p.mlm = o["mlm"]
        Φf, bad = PT.evaluate!(p, u, c); bad == 0 || error("frozen state outside the EOS table")
        o["mlm"] > 0 && PT.colblocks!(p)                  # dlm for the Marquardt term of Aop!
        rh = zeros(p.n1, ny, nx)
        PT.JTop!(p, rh, p.Rx .* p.wc, p.Ry .* p.wc, p.Rz .* vcat(p.wf, 0.0)); rh .*= -1
        return p, Φf, rh
    end
    pr, Φ, b = frozen(:block)
    n1 = pr.n1; N = n1 * ny * nx
    if o["rhs"] != "b"                                    # a right-hand side that is not the Gauss–Newton one
        Random.seed!(11); nb0 = norm(b)
        if o["rhs"] == "rand"
            b = randn(size(b))
        elseif o["rhs"] == "smooth"                       # random modes |k| ≤ 4 of the whole domain, every level
            b = zeros(size(b))
            for ky in -4:4, kx in -4:4
                ((kx == 0 && ky == 0) || kx^2 + ky^2 > 16) && continue
                a = randn(n1); φ = 2π * rand()
                for i in 1:nx, j in 1:ny
                    c = cos(2π * (kx * (i - 1) / nx + ky * (j - 1) / ny) + φ)
                    for m in 1:n1; b[m, j, i] += a[m] * c; end
                end
            end
        else
            error("rhs: b, rand or smooth")
        end
        b .*= nb0 / norm(b)
    end
    logf(@sprintf("frozen %s: %s, μ %.3f φ %g, cut %.2f (n₁ %d), %s balance, %d × %d columns, %d unknowns; Marquardt %.3g; rhs %s, ‖b‖ %.3e",
                  path, inp, mu, az, cut, n1, bal, ny, nx, N, o["mlm"], o["rhs"], norm(b)))
    Z() = zeros(n1, ny, nx)
    snorm(δ) = sqrt(sum(abs2, @view δ[2:end, :, :]) + sum(abs2, @view δ[1, :, :]) / HREF^2)
    truerel(δ) = (t = Z(); PT.Aop!(pr, t, δ); norm(b .- t) / norm(b))
    PT.setprec!(pr)                                       # column blocks
    PT.Aop!(pr, Z(), b); PT.block_solve!(Z(), pr.blk, b)                 # compile
    ta = time(); for _ in 1:5; PT.Aop!(pr, Z(), b); end; tA = (time() - ta) / 5
    tb = time(); for _ in 1:5; PT.block_solve!(Z(), pr.blk, b); end; tB = (time() - tb) / 5
    logf(@sprintf("one A application %.2f ms, one block solve %.2f ms", 1e3 * tA, 1e3 * tB))
    δr = Z(); fl = Ref(:none)
    kr, _ = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, δr, b; tol = o["ref_tol"], maxit = 20000, flag = fl)
    rr = truerel(δr); pref = dot(δr, b) / Φ
    logf(@sprintf("reference δ: block PCG %d its (%s), true rel %.2e%s; pred %.6e", kr, fl[], rr,
                  rr > 10 * o["ref_tol"] ? " (UNRESOLVED reference)" : "", pref))
    λq = lmax_block(pr, 60)
    # mult is SPD for ω λmax < 2. omega_safe = 1: ω = omega/LMAX_BOUND (proven for omega < 2); otherwise ω = omega/λs
    # with λs = 1.2 × the Rayleigh lower bound, proven only when 2λs ≥ LMAX_BOUND (else breakdown_rz in PCG is the
    # fallback signal)
    λB = o["omega_safe"] == 1 ? Float64(LMAX_BOUND) : min(1.2λq, Float64(LMAX_BOUND))
    logf(@sprintf("λmax(B⁻¹A) ≥ %.3f (B-Rayleigh, power iteration 60), ≤ %d (block stencil): smoother ω = %.3f/%.3f = %.4f, SPD of mult %s",
                  λq, LMAX_BOUND, o["omega"], λB, o["omega"] / λB, o["omega"] / λB * LMAX_BOUND < 2 ? "proven" : "not proven (estimate)"))
    precs = split(o["precs"], ",")
    for pname in precs
        M! = nothing; setup = 0.0; T = nothing; prx = pr
        if pname in ("fourier", "block", "hybrid")
            p2, _, _ = frozen(Symbol(pname)); tq = time(); PT.setprec!(p2); setup = time() - tq
            prx = p2; M! = PT.Minv!
        else
            parts = split(pname, ":"); a = parse(Int, parts[2]); kk = length(parts) > 2 ? parse(Int, parts[3]) : 0
            nq = endswith(parts[1], "b") ? 4 : 1
            ncoarse = nq * (kk == 0 ? n1 : kk) * cld(ny, a) * cld(nx, a)
            ncoarse > o["max_coarse"] && (logf(@sprintf("%-10s skipped: %d coarse unknowns > max_coarse", pname, ncoarse)); continue)
            T = try twolevel(pr, a, kk, startswith(parts[1], "mult"), o["omega"] / λB; nq = nq,
                             assembly = Symbol(o["assembly"]), check = o["galcheck"] == 1) catch e
                logf("$pname: setup failed: $(sprint(showerror, e))\n" * join(string.(stacktrace(catch_backtrace())[1:min(8, end)]), "\n")); continue end
            setup = T.tsetup; prx = pr; M! = (p, out, r) -> ml_apply!(p, T, out, r)
        end
        row = String[]
        let δw = Z(); PT.pcg!(PT.Aop!, M!, par, prx, δw, b; tol = 1e-3, maxit = 3); end   # compile before timing
        for tol in (1e-3, 1e-5)
            δ = Z(); cn = zeros(Int, 2); fl = Ref(:none); T === nothing || (T.nA = 0)
            ts = time(); k, _ = PT.pcg!(PT.Aop!, M!, par, prx, δ, b; tol = tol, maxit = 5000, cnt = cn, flag = fl); ts = time() - ts
            nA = cn[1] + (T === nothing ? 0 : T.nA)
            ovh = pname == "block" && k > 0 ? @sprintf(", PCG overhead %.2f ms/it (iteration %.2f ms − A − block solve)",
                                                      1e3 * (ts / k - tA - tB), 1e3 * ts / k) : ""
            push!(row, @sprintf("tol %.0e: %4d its (%s) true %.1e, %5d A + %4d M⁻¹, %6.2f s; δ err %.1e, pred %+.1e%s",
                                tol, k, fl[], truerel(δ), nA, cn[2], ts, snorm(δ .- δr) / snorm(δr),
                                (dot(δ, b) / Φ - pref) / pref, ovh))
        end
        extra = T === nothing ? "" : @sprintf(" [%s assembly %.2f s, coarse %d unknowns, nnz A_c %d, factor %d (%.0f MB), A_c asym %.1e%s]",
                                              o["assembly"], T.tassemble, T.nc, T.nnzA, T.nnzF, 8e-6 * T.nnzF, T.sym,
                                              isnan(T.cmp[1]) ? "" : @sprintf("; vs the other assembly %.1e (%.2f s)", T.cmp...))
        logf(@sprintf("%-10s setup %6.2f s%s", pname, setup, extra))
        foreach(r -> logf("    " * r), row)
        if o["explicit"] == 1 && N <= 6000                  # dense M⁻¹ and M⁻¹A by probes
            Mm = zeros(N, N); Am = zeros(N, N); e = Z(); w = Z()
            for c in 1:N
                fill!(e, 0.0); e[c] = 1.0
                M!(prx, w, e); Mm[:, c] = vec(w)
                PT.Aop!(prx, w, e); Am[:, c] = vec(w)
            end
            sM = norm(Mm - Mm') / norm(Mm); λM = eigmin(Symmetric((Mm + Mm') / 2))
            λs = eigvals(Symmetric(Am), Symmetric(inv(Symmetric((Mm + Mm') / 2))))
            logf(@sprintf("    explicit: ‖M⁻¹ − M⁻ᵀ‖/‖M⁻¹‖ %.1e, λmin(M⁻¹) %.2e (> 0: SPD), λ(M⁻¹A) %.3e … %.3e, κ %.3g",
                          sM, λM, minimum(λs), maximum(λs), maximum(λs) / minimum(λs)))
            if T !== nothing                                # Galerkin check: A_c against Pᵀ A P from the dense A
                nc = T.nc; Pd = zeros(N, nc); ec = zeros(nc)
                for c in 1:nc; fill!(ec, 0.0); ec[c] = 1.0; w = Z(); prolong!(T, w, ec, n1); Pd[:, c] = vec(w); end
                G = Pd' * Am * Pd
                logf(@sprintf("    Galerkin: ‖A_c − PᵀAP‖/‖PᵀAP‖ %.1e", norm(Matrix(T.Ac) - (G + G') / 2) / norm(G)))
            end
        end
        if o["gauge"] == 1                                  # checkerboard / constant cut-height modes
            out = String[]
            for (nm, ph) in (("const", (j, i) -> 1.0), ("(−1)ⁱ", (j, i) -> (-1.0)^i), ("(−1)ʲ", (j, i) -> (-1.0)^j),
                             ("(−1)ⁱ⁺ʲ", (j, i) -> (-1.0)^(i + j)))
                q = Z(); for i in 1:nx, j in 1:ny; q[1, j, i] = ph(j, i); end
                Aq = Z(); PT.Aop!(prx, Aq, q); Mq = Z(); M!(prx, Mq, Aq)
                eA(x) = (t = Z(); PT.Aop!(prx, t, x); sqrt(max(dot(x, t), 0.0)))
                push!(out, @sprintf("%s: ‖(I − M⁻¹A)q‖_A/‖q‖_A %.3f", nm, eA(q .- Mq) / eA(q)))
            end
            logf("    gauge modes: " * join(out, "; "))
        end
    end
end

ARGS[1] == "make" ? make(ARGS[2:end]...) : lab(ARGS[2:end]...)
MPI.Finalized() || MPI.Finalize()
