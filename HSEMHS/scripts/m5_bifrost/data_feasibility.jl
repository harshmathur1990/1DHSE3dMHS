# Data-only magnetic feasibility (CODEX_REVIEW_M5c_2026-10-18.md §4): how small can
#   t = max |B − Bobs| / σ
# be for a divergence-free field on the staggered grid? B = curl(A_ref Ã) + b̄ ẑ, so ∇·B = 0 exactly and the
# problem is unconstrained in (Ã, b̄). The L∞ fit is approached by minimizing the L_p mean of the normalized
# residual with p continued 2 → 128; every stage gives a WITNESS (an upper bound on the minimum t), not a
# lower bound. Lower bounds come from necessary conditions (codex_m5c_result_review.jl: t ≥ 1.92 on faces).
# Observation models:
#   face   — the current one: r_f = (B_f − I·y)/σ_f on faces (centre data y interpolated to faces; at the top
#            and bottom z-faces I takes the nearest centre value);
#   centre — r_c = (H B_f − y)/σ_c at cell centres (H: the face→centre average), the observation residual for
#            cell-centred data (the input B is cell-centred: scripts/m5_bifrost/stagger_test.jl).
# No pressure, EOS, force or Q term. usage: julia --project=. -t 8 scripts/m5_bifrost/data_feasibility.jl face|centre [iters_per_p]
include(joinpath(@__DIR__, "bif_setup.jl"))
mode = Symbol(ARGS[1]); iters = parse(Int, get(ARGS, 2, "400"))
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)    # x: initializer (A₀ fit)
s = pb.s; Ar = pb.Aref
A = (H.newfield0(par), H.newfield0(par), H.newfield(par)); Bf = (H.newfield(par), H.newfield(par), H.newfield0(par))
HB = (H.newfield(par), H.newfield(par), H.newfield(par)); gB = (H.newfield(par), H.newfield(par), H.newfield0(par))
gH = (H.newfield(par), H.newfield(par), H.newfield(par)); gA = (H.newfield0(par), H.newfield0(par), H.newfield(par))
# pack/unpack the owned values of (Ã_x, Ã_y, Ã_z, b̄)
idx = [(c, i, j, k) for c in 1:3 for k in axes(x[c+1], 3) for j in 1:b.ny for i in 1:b.nx]
pack(xs) = vcat([xs[c+1][i, j, k] for (c, i, j, k) in idx], H.getscalar(par, xs[5]))
z0 = pack(x)
σf(c, i, j, k) = 1 / sqrt(pb.invsf2[c][i, j, k])
function fields!(z)
    for (n, (c, i, j, k)) in enumerate(idx)
        A[c][i, j, k] = Ar * z[n]
    end
    H.exchange_halos!(par, collect(A))
    H.curlA!(Bf, A, z[end], s, par); H.exchange_halos!(par, collect(Bf))
    mode == :centre && (H.H!(HB, Bf, s, par); H.exchange_halos!(par, collect(HB)))
end
# residual iterator: (value, c, i, j, k, location) over the chosen observation set
function foreach_res(fun)
    if mode == :face
        for c in 1:3, k in axes(Bf[c], 3), j in 1:b.ny, i in 1:b.nx
            fun((Bf[c][i, j, k] - pb.Bobsf[c][i, j, k]) / σf(c, i, j, k), c, i, j, k)
        end
    else
        for c in 1:3, k in 1:NZ, j in 1:b.ny, i in 1:b.nx
            fun((HB[c][i, j, k] - pb.Bobs[c][i, j, k]) / pb.sig[c][i, j, k], c, i, j, k)
        end
    end
end
function stats()
    mx = 0.0; loc = (0, 0, 0, 0); n = 0; n3 = 0; n5 = 0; s2 = 0.0
    foreach_res((r, c, i, j, k) -> begin
        n += 1; a = abs(r); s2 += r^2; a > 3 && (n3 += 1); a > 5 && (n5 += 1)
        a > mx && (mx = a; loc = (c, i, j, k))
    end)
    return (t = mx, loc = loc, frac3 = n3 / n, n5 = n5, chi2 = s2 / n)
end
# F_p(z) = (1/N) Σ (|r|/τ)^p with a fixed scale τ (the stage's starting max) to keep it in range
function fg(z, p, τ, g)
    fields!(z)
    for c in 1:3
        fill!(parent(gB[c]), 0.0); fill!(parent(gH[c]), 0.0)
    end
    F = 0.0; N = 0
    foreach_res((r, c, i, j, k) -> begin
        N += 1; q = abs(r) / τ; F += q^p
        d = p * q^(p - 1) * sign(r) / τ                        # ∂(q^p)/∂r
        if mode == :face
            gB[c][i, j, k] = d / σf(c, i, j, k)
        else
            gH[c][i, j, k] = d / pb.sig[c][i, j, k]
        end
    end)
    if mode == :centre
        H.exchange_halos!(par, collect(gH)); H.H_T!(gB, gH, s, par)
    end
    H.exchange_halos!(par, collect(gB))
    gbar = H.curlA_T!(gA, gB, s, par)
    for (n, (c, i, j, k)) in enumerate(idx)
        g[n] = Ar * gA[c][i, j, k] / N
    end
    g[end] = gbar / N
    return F / N
end
# plain L-BFGS (m = 10) with Armijo backtracking
function lbfgs(z, p, τ, iters)
    m = 10; S = Vector{Vector{Float64}}(); Y = Vector{Vector{Float64}}()
    g = similar(z); f = fg(z, p, τ, g); gt = similar(z)
    for it in 1:iters
        q = copy(g); α = zeros(length(S))
        for k in length(S):-1:1
            α[k] = dot(S[k], q) / dot(Y[k], S[k]); q .-= α[k] .* Y[k]
        end
        γ = isempty(S) ? 1 / norm(g) : dot(S[end], Y[end]) / dot(Y[end], Y[end])
        q .*= γ
        for k in eachindex(S)
            β = dot(Y[k], q) / dot(Y[k], S[k]); q .+= (α[k] - β) .* S[k]
        end
        d = -q; dg = dot(d, g)
        dg >= 0 && (d = -g / norm(g); dg = dot(d, g); empty!(S); empty!(Y))
        st = 1.0; zt = z .+ st .* d; ft = fg(zt, p, τ, gt)
        while !(ft <= f + 1e-4 * st * dg) && st > 1e-12
            st /= 2; zt = z .+ st .* d; ft = fg(zt, p, τ, gt)
        end
        st <= 1e-12 && break
        sk = zt .- z; yk = gt .- g
        if dot(sk, yk) > 1e-12 * norm(sk) * norm(yk)
            push!(S, sk); push!(Y, yk); length(S) > m && (popfirst!(S); popfirst!(Y))
        end
        z = zt; f = ft; g .= gt
    end
    return z
end
using LinearAlgebra
fields!(z0); s0 = stats()
@printf("%s: initializer (A₀): t = %.3f at %s, frac>3σ %.2e, #>5σ %d, χ²/N %.4f\n", mode, s0.t, string(s0.loc), s0.frac3, s0.n5, s0.chi2)
z = copy(z0)
for p in (get(ARGS, 3, "128") == "256" ? (2, 4, 8, 16, 32, 64, 128, 256) : (2, 4, 8, 16, 32, 64, 128))
    fields!(z); τ = stats().t
    global z = lbfgs(z, p, τ, iters)
    fields!(z); st = stats()
    @printf("%s: p = %3d: witness t = %.3f at %s, frac>3σ %.2e, #>5σ %d, χ²/N %.4f\n", mode, p, st.t, string(st.loc), st.frac3, st.n5, st.chi2)
    flush(stdout)
    # save this stage's witness: normalized residuals per component, in the layout of export_observations.jl
    if !isempty(get(ENV, "WITNESS_DIR", ""))
        fn = joinpath(ENV["WITNESS_DIR"], "witness_$(mode)_p$(p).h5")
        h5open(fn, "w") do f
            if mode == :face
                for c in 1:3
                    f["c$c"] = [(Bf[c][i, j, k] - pb.Bobsf[c][i, j, k]) / σf(c, i, j, k) for i in 1:b.nx, j in 1:b.ny, k in collect(axes(Bf[c], 3))]
                    f["B$c"] = [Bf[c][i, j, k] for i in 1:b.nx, j in 1:b.ny, k in collect(axes(Bf[c], 3))]
                end
            else
                for c in 1:3
                    f["c$c"] = [(HB[c][i, j, k] - pb.Bobs[c][i, j, k]) / pb.sig[c][i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
                    f["B$c"] = [Bf[c][i, j, k] for i in 1:b.nx, j in 1:b.ny, k in collect(axes(Bf[c], 3))]
                end
            end
            attributes(f)["p"] = p; attributes(f)["t"] = st.t
        end
    end
end
