#=
Frozen-system lab for mhs-poisson-ltau (design/CODEX_REVIEW_solver_time_2026-10-28.md, steps 2–3): one Gauss–Newton
system, the linearisation at the state after STEPS D2 steps with the trajectory's own weights and regularisation, and
what the solvers and preconditioners do with it.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/frozen_lab.jl INPUT.nc MU AZIMUTH_DEG GEOMETRY LOGTAU_CUT_LOS BALANCE STEPS [KEY=VALUE ...]
KEY: crop (0: the full cube; NC > 0: an NC × NC crop around the strongest |B| at τ = 1, periodic, one rank only),
parts (comma list of checks, gauge, tight, direct, marq, coarse; default checks,gauge,tight,marq; direct needs a crop),
traj (preconditioner of the trajectory, default fourier), tight_tol (1e-8), tight_maxit (2000), marq_maxit (600),
ritz_its (Lanczos iterations, 300), nritz (Ritz vectors kept, 32).

checks  adjoint ⟨Jv, w⟩ = ⟨v, Jᵀw⟩; finite differences of R against Jv for ε 1e-2 … 1e-6 (a random v and one
        supported on the 5 levels below the cut); symmetry and positivity of A = JᵀWJ + ereg and of each M⁻¹;
        the column block of one column against operator probes with Marquardt damping 1.
gauge   ‖W^½ J v‖ / ‖v‖ for cut-height modes 1, (−1)^i, (−1)^j, (−1)^(i+j) (zero centred-difference symbol at
        Nyquist) against a smooth mode and random v; heights scaled by Href = 100 km.
tight   δ from PCG to tight_tol with each preconditioner (fourier, block, hybrid): iterations, recursive and true
        residual, pred, max|δu|, max|δc|, cap scale, and the differences between the δ (scaled norm; δc rms and the
        |k| ≤ 2 part of δc).
direct  dense A by probes, Cholesky solve (compared with the PCG δ) and the smallest eigenpairs of the scaled H = JᵀWJ
        and of H + ereg: eigenvalue / largest, c share, depth (log τ of the largest ln P part), |k| of the c part.
marq    Marquardt damping μ ∈ 0, 0.01, 0.1, 1, 10 for block, hybrid without and with the damping in its Fourier part:
        build time, CG iterations and residuals (tol 0.1 and 1e-3), pred, cap scale.
coarse  slow modes of the block-preconditioned system (Ritz pairs from Lanczos) and balanced deflation with
        several coarse spaces (`coarse_lab`).
=#
using HSEMHS
using MPI
using LinearAlgebra
using Printf
using Random
using FFTW

const H = HSEMHS
const PT = H.PoissonTau
const HREF = 1e5

function main(path, mu_s, az_s, geom, cut_s, bal, steps_s, kv...)
    H.ensure_mpi()
    o = Dict{String,Any}("crop" => 0, "parts" => "checks,gauge,tight,marq", "traj" => "fourier", "tight_tol" => 1e-8,
                         "tight_maxit" => 2000, "marq_maxit" => 600, "ritz_its" => 300, "nritz" => 32)
    for a in kv
        k, v = split(a, "="); haskey(o, k) || error("unknown key $k")
        o[k] = o[k] isa String ? String(v) : o[k] isa Int ? parse(Int, v) : parse(Float64, v)
    end
    parts = Set(split(o["parts"], ","))
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = Symbol(geom))
    cut = parse(Float64, cut_s); balance = Symbol(bal); nsteps = parse(Int, steps_s)
    t0 = time()
    NX, NY, nt = PT.tau_input_size(path)
    par = H.slab_par(NX, NY, nt)
    logf(s) = H.rprint(par, @sprintf("[%7.1fs] %s", time() - t0, s))
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    g = 274.0; dx = x[2] - x[1]; dy = y[2] - y[1]
    NC = o["crop"]
    if NC > 0
        par.nranks == 1 || error("crop needs one rank")
        k0 = argmin(abs.(lt))
        _, jc, ic = Tuple(argmax(Bx[k0:k0, :, :] .^ 2 .+ By[k0:k0, :, :] .^ 2 .+ Bz[k0:k0, :, :] .^ 2))
        J = mod1.(jc - NC ÷ 2 .+ (0:NC-1), NY); I = mod1.(ic - NC ÷ 2 .+ (0:NC-1), NX)
        T, Bx, By, Bz = (A[:, J, I] for A in (T, Bx, By, Bz))
        NX = NY = NC
        par = H.slab_par(NX, NY, nt)
        logf(@sprintf("crop %d × %d around column (y %d, x %d), |B(τ=1)| %.0f G", NC, NC, jc, ic,
                      1e4 * sqrt(Bx[k0, NC÷2+1, NC÷2+1]^2 + By[k0, NC÷2+1, NC÷2+1]^2 + Bz[k0, NC÷2+1, NC÷2+1]^2)))
    end
    lT = log.(T)
    BLAS.set_num_threads(1)
    lnP, z0, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed")
    u, z, Φ0, Φs, s, _, info = PT.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, dx, dy; logtau_cut = cut,
                                            max_steps = nsteps, mu = v.mu, slant = H.slant(v), stop = :phi, rtol = 0.0,
                                            balance = balance, lm = :gain, forcing = :ew, precond = Symbol(o["traj"]),
                                            stationarity = false)
    W = info.weights
    logf(@sprintf("state after %d steps (%s): Φ/Φ₀ %.3e; μ %.3f φ %g %s, cut %.2f, balance %s, grid %d × %d × %d",
                  s, o["traj"], Φs / Φ0, v.mu, v.azimuth_deg, geom, cut, balance, NY, NX, nt))

    # the frozen system: linearisation at (u, z) with the trajectory's weights
    function frozen(pc)
        pr, c = PT.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z, dx, dy; logtau_cut = cut, mu = v.mu, slant = H.slant(v),
                            balance = balance, precond = pc)
        pr.wc = copy(W.wc); pr.wf = copy(W.wf); pr.ereg = copy(W.ereg)
        Φ, _ = PT.evaluate!(pr, u, c)
        rhs = zeros(pr.n1, pr.ny, pr.nx)
        PT.JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
        return pr, c, Φ, rhs
    end
    pr, c, Φ, rhs = frozen(:hybrid)
    n1, ny, nx, kc = pr.n1, pr.ny, pr.nx, pr.kc
    logf(@sprintf("frozen system: Φ %.6e (trajectory %.6e), n₁ %d (cut level %d, log τ %.2f), %d unknowns, ‖b‖ %.3e",
                  Φ, Φs, n1, kc, lt[kc], n1 * ny * NX, H.gnorm(par, rhs)))
    Z() = zeros(n1, ny, nx)
    sc_(δ) = (du = H.gmax(par, maximum(abs, @view δ[2:end, :, :])); dc = H.gmax(par, maximum(abs, @view δ[1, :, :]));
              (min(1.0, 2.0 / du, 3e5 / dc), du, dc))
    snorm(δ) = sqrt(H.gsum(par, sum(abs2, @view δ[2:end, :, :]) + sum(abs2, @view δ[1, :, :]) / HREF^2))
    truerel(p, δ) = (t = Z(); PT.Aop!(p, t, δ); H.gnorm(par, rhs .- t) / H.gnorm(par, rhs))
    Wjv(p, δ) = begin jx, jy, jz = Z(), Z(), Z(); PT.Jop!(p, jx, jy, jz, δ); sqrt(PT.wnorm2(p, jx, jy, jz)) end
    gi = H.owned_x(par)                                        # global x indices (1-based) of this rank
    function lowk_rms(a; kmax = 2)                             # rms of the |k| ≤ kmax part of a (y, x) map (Parseval)
        e2 = 0.0
        for ky in -kmax:kmax, kx in -kmax:kmax
            ky^2 + kx^2 <= kmax^2 || continue
            ph = [cis(-2π * (kx * (gi[i] - 1) / NX + ky * (j - 1) / ny)) for j in 1:ny, i in 1:nx]
            F = H.gsum(par, real(sum(a .* ph))) + im * H.gsum(par, imag(sum(a .* ph)))
            e2 += abs2(F)
        end
        return sqrt(e2) / (ny * NX)
    end

    if "checks" in parts
        rng = MersenneTwister(7 + par.rank)
        dv = 0.01 .* randn(rng, n1, ny, nx); dv[1, :, :] .= 100.0 .* randn(rng, ny, nx)
        jx, jy, jz = Z(), Z(), Z(); PT.Jop!(pr, jx, jy, jz, dv)
        wx, wy, wz = randn(rng, n1, ny, nx), randn(rng, n1, ny, nx), randn(rng, n1, ny, nx); wz[n1, :, :] .= 0
        out = Z(); PT.JTop!(pr, out, wx, wy, wz)
        l = H.gsum(par, sum(jx .* wx) + sum(jy .* wy) + sum(jz .* wz)); r = H.gdot(par, dv, out)
        logf(@sprintf("adjoint: |⟨Jv,w⟩ − ⟨v,Jᵀw⟩| / |⟨Jv,w⟩| = %.2e", abs(l - r) / abs(l)))
        dn = zeros(n1, ny, nx); dn[2:min(6, n1), :, :] .= 0.01 .* randn(rng, min(5, n1 - 1), ny, nx)
        for (name, d) in (("random", dv), ("5 levels below the cut", dn))
            PT.evaluate!(pr, u, c); PT.Jop!(pr, jx, jy, jz, d)
            den = sqrt(H.gsum(par, sum(abs2, jx) + sum(abs2, jy) + sum(abs2, @view jz[1:n1-1, :, :])))
            errs = String[]
            for ε in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
                R(sg) = (ut = copy(u); @views ut[kc+1:end, :, :] .+= sg * ε .* d[2:end, :, :];
                         PT.evaluate!(pr, ut, c .+ sg * ε .* d[1, :, :]); (copy(pr.Rx), copy(pr.Ry), copy(pr.Rz)))
                Rp = R(1); Rm = R(-1)
                num = sqrt(H.gsum(par, sum(abs2, (Rp[1] .- Rm[1]) ./ 2ε .- jx) + sum(abs2, (Rp[2] .- Rm[2]) ./ 2ε .- jy) +
                                       sum(abs2, @views (Rp[3][1:n1-1, :, :] .- Rm[3][1:n1-1, :, :]) ./ 2ε .- jz[1:n1-1, :, :])))
                push!(errs, @sprintf("ε %.0e: %.1e", ε, num / den))
            end
            logf("finite differences (" * name * "): " * join(errs, ", "))
        end
        PT.evaluate!(pr, u, c)
        a, b = randn(rng, n1, ny, nx), randn(rng, n1, ny, nx)
        Aa, Ab = Z(), Z(); PT.Aop!(pr, Aa, a); PT.Aop!(pr, Ab, b)
        logf(@sprintf("A: symmetry %.2e, ⟨Aa,a⟩ > 0: %s", abs(H.gdot(par, Aa, b) - H.gdot(par, a, Ab)) / abs(H.gdot(par, Aa, b)),
                      H.gdot(par, Aa, a) > 0))
        for (pc, mlm, fd) in ((:fourier, 0.0, true), (:block, 0.0, true), (:hybrid, 0.0, true), (:block, 1.0, true),
                              (:hybrid, 1.0, false), (:hybrid, 1.0, true))
            p, = frozen(pc); p.mlm = mlm; p.fdamp = fd; PT.setprec!(p)
            Ma, Mb = Z(), Z(); PT.Minv!(p, Ma, a); PT.Minv!(p, Mb, b)
            logf(@sprintf("M⁻¹ %-7s mlm %g fdamp %-5s: symmetry %.2e, ⟨M⁻¹a,a⟩ > 0: %s", pc, mlm, fd,
                          abs(H.gdot(par, Ma, b) - H.gdot(par, a, Mb)) / abs(H.gdot(par, Ma, b)), H.gdot(par, Ma, a) > 0))
        end
        p, = frozen(:block); p.mlm = 1.0; PT.setprec!(p)
        jj, ii = ny ÷ 2 + 1, nx ÷ 2 + 1; e = Z(); out = Z(); A = zeros(n1, n1)
        for m in 1:n1
            fill!(e, 0.0); par.rank == 0 && (e[m, jj, ii] = 1.0)
            PT.Aop!(p, out, e); A[:, m] = out[:, jj, ii]
        end
        if par.rank == 0
            L = LowerTriangular(p.blk[:, :, jj, ii])
            logf(@sprintf("column block with Marquardt damping 1 against probes: max |LLᵀ − A| / max |A| = %.2e",
                          maximum(abs, L * L' .- A) / maximum(abs, A)))
        end
    end

    if "gauge" in parts
        ii = reshape(collect(gi), 1, :); jj = collect(1:ny)
        modes = [("constant", ones(ny, nx)), ("(-1)^i", repeat((-1.0) .^ ii, ny, 1)), ("(-1)^j", repeat((-1.0) .^ jj, 1, nx)),
                 ("(-1)^(i+j)", (-1.0) .^ (jj .+ ii)), ("cos kx=1", repeat(cos.(2π .* (ii .- 1) ./ NX), ny, 1)),
                 ("cos kx=NX/4", repeat(cos.(2π .* (ii .- 1) ./ 4), ny, 1))]
        for (name, m) in modes
            d = Z(); d[1, :, :] .= HREF .* m
            logf(@sprintf("height mode %-12s ‖W½Jv‖/‖v‖ %.3e (NX %d %s)", name, Wjv(pr, d) / snorm(d), NX, iseven(NX) ? "even" : "odd"))
        end
        rng = MersenneTwister(11 + par.rank); d = randn(rng, n1, ny, nx); d[1, :, :] .*= HREF
        logf(@sprintf("random v                ‖W½Jv‖/‖v‖ %.3e", Wjv(pr, d) / snorm(d)))
    end

    δs = Dict{Symbol,Array{Float64,3}}()
    if "tight" in parts || "direct" in parts
        for pc in (:fourier, :block, :hybrid)
            p, = frozen(pc); tb = time(); PT.setprec!(p); tb = time() - tb
            δ = Z(); ts = time()
            its, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, p, δ, rhs; tol = o["tight_tol"], maxit = o["tight_maxit"])
            ts = time() - ts
            sc, du, dc = sc_(δ)
            logf(@sprintf("tight %-7s: build %.1f s, CG %4d its %.0f s, rel %.1e, true rel %.1e | pred %.4f | max|δu| %.3g max|δc| %.1f km sc %.3g",
                          pc, tb, its, ts, rel, truerel(p, δ), H.gdot(par, δ, rhs) / Φ, du, dc / 1e3, sc))
            δs[pc] = δ
        end
        ks = collect(keys(δs))
        for a in 1:length(ks), b in a+1:length(ks)
            A_, B_ = δs[ks[a]], δs[ks[b]]
            dc = A_[1, :, :] .- B_[1, :, :]
            logf(@sprintf("δ(%s) − δ(%s): scaled %.2e of ‖δ(%s)‖; δc rms %.3f km (|k| ≤ 2: %.3f km); δu max %.3g",
                          ks[a], ks[b], snorm(A_ .- B_) / snorm(B_), ks[b], sqrt(H.gsum(par, sum(abs2, dc)) / (ny * NX)) / 1e3,
                          lowk_rms(dc) / 1e3, H.gmax(par, maximum(abs, @view (A_ .- B_)[2:end, :, :]))))
        end
    end

    if "direct" in parts
        NC > 0 || error("direct needs a crop")
        n = n1 * ny * nx; logf(@sprintf("direct: probing the %d × %d operator", n, n))
        A = zeros(n, n); e = Z(); out = Z()
        for col in 1:n
            fill!(e, 0.0); e[col] = 1.0; PT.Aop!(pr, out, e); A[:, col] = vec(out)
        end
        logf(@sprintf("direct: asymmetry %.2e", maximum(abs, A .- A') / maximum(abs, A)))
        x = cholesky(Symmetric((A .+ A') ./ 2)) \ vec(rhs); δd = reshape(x, n1, ny, nx)
        logf(@sprintf("direct: true rel %.1e, pred %.4f, max|δu| %.3g, max|δc| %.1f km", truerel(pr, δd), dot(δd, rhs) / Φ,
                      sc_(δd)[2], sc_(δd)[3] / 1e3))
        for (k, d) in δs
            logf(@sprintf("direct − %-7s: scaled %.2e", k, snorm(d .- δd) / snorm(δd)))
        end
        S = ones(n); S[1:n1:end] .= HREF
        for (name, ereg) in (("H", zeros(n1)), ("H + ereg", pr.ereg))
            M = S .* (A .- Diagonal(repeat(pr.ereg, ny * nx)) .+ Diagonal(repeat(ereg, ny * nx))) .* S'
            E = eigen(Symmetric((M .+ M') ./ 2))
            λmax = E.values[end]
            logf(@sprintf("eigenpairs of scaled %s: λmax %.3e; ereg (scaled) per level %.1e … %.1e", name, λmax,
                          minimum(pr.ereg[2:end]), maximum(pr.ereg[2:end])))
            for q in 1:10
                vq = reshape(E.vectors[:, q], n1, ny, nx)
                cs = sum(abs2, vq[1, :, :])
                lev = vec(sum(abs2, vq[2:end, :, :], dims = (2, 3)))
                mlev = argmax(lev); ltm = lt[kc+mlev]
                ltw = sum(lt[kc+1:kc+n1-1] .* lev) / max(sum(lev), eps())
                cmap = vq[1, :, :]; F = abs2.(fft(cmap)); kx = fftfreq(nx) .* nx; ky = fftfreq(ny) .* ny
                kmag = [sqrt(ky[j]^2 + kx[i]^2) for j in 1:ny, i in 1:nx]
                kc_ = sum(F .* kmag) / max(sum(F), eps())
                ue = vec(sum(abs2, vq[2:end, :, :], dims = 1)); ue = reshape(ue, ny, nx)
                Fu = abs2.(fft(ue .- sum(ue) / length(ue)))
                logf(@sprintf("  λ%-2d %.3e (λ/λmax %.1e): c share %.2f, ln P peak at log τ %.2f (mean %.2f), c |k| %.1f, |c| pattern %s",
                              q, E.values[q], E.values[q] / λmax, cs, ltm, ltw, kc_,
                              maximum(abs, cmap) > 0 ? @sprintf("max %.2f at (%d,%d)", maximum(abs, cmap), Tuple(argmax(abs.(cmap)))...) : "–"))
            end
        end
    end

    if "marq" in parts
        for μ in (0.0, 0.01, 0.1, 1.0, 10.0), (pc, fd) in ((:block, true), (:hybrid, false), (:hybrid, true))
            pc == :block && !fd && continue
            p, = frozen(pc); p.mlm = μ; p.fdamp = fd
            tb = time(); PT.setprec!(p); tb = time() - tb
            row = String[]
            for tol in (0.1, 1e-3)
                δ = Z(); ts = time()
                its, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, p, δ, rhs; tol = tol, maxit = o["marq_maxit"])
                sc, du, dc = sc_(δ)
                push!(row, @sprintf("tol %.0e: %3d its %5.0f s rel %.1e true %.1e pred %.3f sc %.3f (|δu| %.2g |δc| %.0f km)",
                                    tol, its, time() - ts, rel, truerel(p, δ), H.gdot(par, δ, rhs) / Φ, sc, du, dc / 1e3))
            end
            logf(@sprintf("marq μ %-5g %-7s fdamp %-5s build %.1f s | %s", μ, pc, fd, tb, join(row, " | ")))
        end
    end
    if "coarse" in parts
        coarse_lab(par, pr, rhs, Φ, lt, NX, o, logf, Z, snorm, sc_, truerel, gi)
    end
    logf("done")
end

"""Columns of a matrix (local part) for a list of fields."""
cols(fs) = reduce(hcat, (vec(f) for f in fs))

"""
Coarse-space part (Codex review step 5): the slow modes of the block-preconditioned system from the Lanczos process of
`ritz_its` PCG iterations (Ritz pairs of M⁻¹A, the smallest `nritz` characterised: θ, residual, c share, |k| ≤ 2 share,
depth) and balanced deflation M⁻¹_BNN = Q + (I − QA) B (I − AQ), Q = Z (ZᵀAZ)⁻¹ Zᵀ, B the column blocks, for
coarse spaces Z: the cut-height gauge modes, low-k cut-height Fourier modes, low-k modes on every level, Ritz vectors.
"""
function coarse_lab(par, pr, rhs, Φ, lt, NX, o, logf, Z, snorm, sc_, truerel, gi)
    n1, ny, nx, kc = pr.n1, pr.ny, pr.nx, pr.kc
    p = pr; p.prec = :block; p.mlm = 0.0; PT.setprec!(p)
    N = ny * NX; nloc = n1 * ny * nx
    hs = ones(n1); hs[1] = HREF                                 # scaled field: c / HREF
    ii = reshape(collect(gi), 1, :); jj = collect(1:ny)
    # Lanczos from PCG: q_j = z_j / √(r_jᵀ z_j) M-orthonormal (signs folded into T), T from α, β
    m = o["ritz_its"]; V = zeros(nloc, m)
    xs = Z(); r = copy(rhs); zv = Z(); pv = Z(); Ap = Z()
    PT.Minv!(p, zv, r); pv .= zv; rz = H.gdot(par, r, zv); α = Float64[]; β = Float64[]
    t0 = time()
    for j in 1:m
        V[:, j] .= vec(zv) ./ sqrt(rz)
        PT.Aop!(p, Ap, pv); a = rz / H.gdot(par, pv, Ap); push!(α, a)
        @. xs += a * pv; @. r -= a * Ap
        PT.Minv!(p, zv, r); rz2 = H.gdot(par, r, zv); b = rz2 / rz; push!(β, b); rz = rz2
        @. pv = zv + b * pv
    end
    rel = H.gnorm(par, r) / H.gnorm(par, rhs)
    T = zeros(m, m)
    for j in 1:m
        T[j, j] = 1 / α[j] + (j > 1 ? β[j-1] / α[j-1] : 0.0)
        j < m && (T[j, j+1] = T[j+1, j] = -sqrt(β[j]) / α[j])
    end
    E = eigen(Symmetric(T))
    logf(@sprintf("Lanczos (block PCG) %d its %.0f s, rel %.1e: Ritz θ of M⁻¹A min %.3e max %.3e (κ ≈ %.2e); smallest: %s",
                  m, time() - t0, rel, E.values[1], E.values[end], E.values[end] / E.values[1],
                  join((@sprintf("%.2e", t) for t in E.values[1:min(12, m)]), " ")))
    nr = min(o["nritz"], m)
    BLAS.set_num_threads(min(32, Threads.nthreads()))
    X = V * E.vectors[:, 1:nr]
    BLAS.set_num_threads(1)
    V = nothing; GC.gc()
    # low-k phases on the global grid for the |k| ≤ 2 shares
    ks = [(ky, kx) for ky in -2:2 for kx in -2:2 if ky^2 + kx^2 <= 4]
    phs = [vec(cis.(-2π .* (kx .* (ii .- 1) ./ NX .+ ky .* (jj .- 1) ./ ny))) for (ky, kx) in ks]
    gauge = [ones(ny, nx), repeat((-1.0) .^ ii, ny, 1), repeat((-1.0) .^ jj, 1, nx), (-1.0) .^ (jj .+ ii)]
    function describe(x)
        s = reshape(x, n1, ny, nx) ./ hs                      # scaled field
        tot = H.gsum(par, sum(abs2, s))
        cs = H.gsum(par, sum(abs2, @view s[1, :, :])) / tot
        S2 = reshape(s, n1, ny * nx)
        lk = 0.0
        for ph in phs
            F = S2 * ph; Fr = H.gsum(par, real(F)); Fi = H.gsum(par, imag(F))
            lk += sum(abs2, Fr) + sum(abs2, Fi)
        end
        lev = [H.gsum(par, sum(abs2, @view s[q, :, :])) for q in 2:n1]
        dep = sum(lt[kc+1:kc+n1-1] .* lev) / max(sum(lev), eps())
        gov = sum(abs2, [H.gsum(par, sum(s[1, :, :] .* gm)) for gm in gauge]) / N / tot   # c part in the gauge modes
        return cs, lk / N / tot, dep, gov
    end
    w = Z(); Ax = Z()
    for q in 1:nr
        x = reshape(view(X, :, q), n1, ny, nx)
        PT.Aop!(p, Ax, x); PT.Minv!(p, w, Ax)
        res = snorm(w .- E.values[q] .* x) / (E.values[q] * snorm(x))
        cs, lk, dep, gov = describe(vec(x))
        logf(@sprintf("  Ritz %-2d θ %.3e res %.1e: c share %.2f, |k| ≤ 2 share %.2f, gauge share %.2f, ln P mean log τ %.2f",
                      q, E.values[q], res, cs, lk, gov, dep))
    end

    # coarse spaces
    function fourier_fields(K; levels)
        fs = Array{Float64,3}[]
        pats = Matrix{Float64}[ones(ny, nx)]
        for ky in 0:K, kx in -K:K
            (ky == 0 && kx <= 0) && continue
            ky^2 + kx^2 <= K^2 || continue
            ang = 2π .* (kx .* (ii .- 1) ./ NX .+ ky .* (jj .- 1) ./ ny)
            push!(pats, cos.(ang), sin.(ang))
        end
        for q in levels, pt in pats
            f = Z(); f[q, :, :] .= pt .* hs[q]; push!(fs, f)
        end
        for gm in gauge[2:4]
            f = Z(); f[1, :, :] .= gm .* HREF; push!(fs, f)
        end
        return fs
    end
    sets = Any[("gauge (4)", () -> (fs = Array{Float64,3}[]; for gm in gauge; f = Z(); f[1, :, :] .= gm .* HREF; push!(fs, f); end; cols(fs))),
               ("c |k| ≤ 2", () -> cols(fourier_fields(2; levels = 1:1))),
               ("c |k| ≤ 4", () -> cols(fourier_fields(4; levels = 1:1))),
               ("all levels |k| ≤ 1", () -> cols(fourier_fields(1; levels = 1:n1))),
               ("all levels |k| ≤ 2", () -> cols(fourier_fields(2; levels = 1:n1))),
               ("Ritz 8", () -> X[:, 1:min(8, nr)]),
               ("Ritz $nr", () -> X)]
    for (name, mk) in sets
        tb = time()
        Zm = mk(); k = size(Zm, 2)
        AZ = similar(Zm); t = Z()
        for q in 1:k
            PT.Aop!(p, t, reshape(view(Zm, :, q), n1, ny, nx)); AZ[:, q] .= vec(t)
        end
        BLAS.set_num_threads(min(32, Threads.nthreads()))
        Eg = reshape(H.gsum(par, vec(Zm' * AZ)), k, k); Eg = (Eg .+ Eg') ./ 2
        BLAS.set_num_threads(1)
        Ee = eigen(Symmetric(Eg)); keep = Ee.values .> 1e-12 * Ee.values[end]
        Ei = Ee.vectors[:, keep] * Diagonal(1 ./ Ee.values[keep]) * Ee.vectors[:, keep]'   # (ZᵀAZ)⁻¹ on its range
        tb = time() - tb
        gz(v) = H.gsum(par, Zm' * vec(v))
        gaz(v) = H.gsum(par, AZ' * vec(v))
        bt = Z(); y = Z()
        function bnn!(_, out, rr)                               # Q r + (I − QA) B (I − AQ) r
            c1 = Ei * gz(rr)
            bt .= rr .- reshape(AZ * c1, n1, ny, nx)
            PT.Minv!(p, y, bt)
            c2 = Ei * gaz(y)
            out .= y .- reshape(Zm * c2, n1, ny, nx) .+ reshape(Zm * c1, n1, ny, nx)
            return out
        end
        row = String[]
        for (tol, mi) in ((1e-3, o["marq_maxit"]), (1e-5, 2 * o["marq_maxit"]))
            δ = Z(); ts = time()
            its, rel = PT.pcg!(PT.Aop!, bnn!, par, p, δ, rhs; tol = tol, maxit = mi)
            sc, du, dc = sc_(δ)
            push!(row, @sprintf("tol %.0e: %4d its %5.0f s true %.1e pred %.3f sc %.3f", tol, its, time() - ts, truerel(p, δ),
                                H.gdot(par, δ, rhs) / Φ, sc))
        end
        logf(@sprintf("BNN %-20s k %4d (rank %4d, cond(ZᵀAZ) %.1e) build %.0f s | %s", name, k, count(keep),
                      Ee.values[end] / Ee.values[findfirst(keep)], tb, join(row, " | ")))
    end
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
