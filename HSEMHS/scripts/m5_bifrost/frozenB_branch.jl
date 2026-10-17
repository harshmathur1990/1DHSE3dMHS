# Codex review M5, experiment steps 2–3 (revised after CODEX_REVIEW_M5_FOLLOWUP.md): from the frozen-B
# checkpoint, solve the pressure problem with B fixed, in ln P (`lnP`) or scaled pressure p = P/P_ref (`P`;
# a scalar reference scale, not a per-node P₀). Reporting windows of 100 accepted updates OBSERVE the solve:
# the trust-region state (Δ, μ) is carried across windows. The final state is saved for resuming.
# usage: julia --project=. -t 6 scripts/m5_bifrost/frozenB_branch.jl CHECKPOINT.h5 lnP|P [max_accepted] [perturb] [resume.h5]
include(joinpath(@__DIR__, "bif_setup.jl"))
ck = ARGS[1]; coord = Symbol(ARGS[2]); maxacc = parse(Int, get(ARGS, 3, "500"))
pert = parse(Float64, get(ARGS, 4, "0")); resume = get(ARGS, 5, "")
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
w = H.SMHSWork(par)
uinit = H.newfield(par)
h5open(ck) do f
    for q in 1:5
        copyto!(parent(x[q]), read(f["x$q"]))
    end
    copyto!(parent(uinit), read(f["u_init"]))
end
H.exchange_halos!(par, collect(x))
# collapse bound from the initial u (computed once, on the main thread)
ma = H.anchor_mean(pb, uinit)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> uinit[i, j, k] - ma + pb.lnPref - 3.0, ulo, par)
# fixed identifiability mask from the UNPERTURBED checkpoint
H.objective!(pb, w, x)
dPu = H.dP_unobs!(H.newfield(par), pb, w)
ident = [dPu[i, j, k] <= 0.1 * w.P[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
if pert != 0                                                  # ±pert multiplicative perturbation, projected into bounds ∩ anchor
    t = H.newfield(par); δ = H.newfield(par)
    H.fill_owned!((i, j, k) -> log1p(pert * (isodd(hash((b.i0 + i, b.j0 + j, k, 0x77))) ? 1.0 : -1.0)), t, par)
    m = H.anchor_mean(pb, t); H.owned_map!((i, j, k) -> t[i, j, k] - m, t, par)
    H.anchor_box_project!(δ, t, (i, j, k) -> min(ulo[i, j, k] - w.u[i, j, k], 0.0), (i, j, k) -> Inf, (i, j, k) -> 1.0, pb)
    H.owned_map!((i, j, k) -> x[1][i, j, k] + δ[i, j, k], x[1], par)
end
Δ = NaN; μ = 1e-8; total = 0
if !isempty(resume)                                           # resume a previous branch
    h5open(resume) do f
        copyto!(parent(x[1]), read(f["x1"])); global Δ = read(f["Delta"]); global μ = read(f["mu"]); global total = read(f["accepted"])
    end
    H.exchange_halos!(par, collect(x))
end
f0, t0, _ = H.objective!(pb, w, x)
cst = f0 - t0[1]                                              # B frozen: Φ_B, Φ_Q, Φ_S are constant
lF = pb.lamF
@printf("%s: start raw force %.6e; identifiable fraction %.3f; bound margin %.3e; perturbation %g; resumed at %d\n", coord, t0[1] / lF,
        count(ident) / length(ident), minimum(w.u[i, j, k] - ulo[i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx), pert, total)
quiet = 0; tm = time()
while total < maxacc
    uref = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    fw = H.objective!(pb, w, x)[1]
    r = H.u_gauss_newton!(pb, w, x, ulo; steps = min(100, maxacc - total), cg_iter = 300, cg_rtol = 1e-4, du_max = 0.5, coord = coord,
                          opt_tol = 1e-6, Δ0 = Δ, μ0 = μ, record = true)
    global Δ = r.Δ; global μ = r.μ; global total += r.accepted
    dP = [abs(expm1(w.u[i, j, k] - uref[i, j, k])) for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    act = [count(w.u[i, j, k] - ulo[i, j, k] <= 1e-6 for i in 1:b.nx, j in 1:b.ny) / (b.nx * b.ny) for k in 1:NZ]
    h = r.history
    stat(fn, op) = isempty(h) ? NaN : op(fn(q) for q in h)
    @printf("%s window: %s, accepted %d (total %d), candidates %d, Cauchy fallbacks %d, trust scalings (incl. Cauchy trials) %d, cap hits %d | raw force %.6e | opt %.2e→%.2e | max ΔP/P %.2e (ident %.2e, complement %.2e) | min margin %.2e, anchor err %.1e | trust use mean %.2f | ρ mean %.2f | model decrease Σ %.3e, actual Σ %.3e | max PCG rel %.1e | ε-active max %d | bound-active layers %d | Δ %.2e | %.0f s\n",
            coord, string(r.status), r.accepted, total, r.counts.cand, r.counts.cauchy, r.counts.trust, r.counts.cap, (r.f - cst) / lF, r.opt0, r.opt,
            maximum(dP), maximum(dP[ident]), any(.!ident) ? maximum(dP[.!ident]) : 0.0, stat(q -> q.margin, minimum), stat(q -> q.anchor_err, maximum),
            stat(q -> q.ynorm / q.Delta_used, x -> sum(x) / length(h)), stat(q -> q.rho, x -> sum(x) / length(h)),
            stat(q -> q.pred, sum), fw - r.f, stat(q -> q.cg_rel, maximum),
            stat(q -> q.nactive, maximum), count(>(0), act), Δ, time() - tm)
    flush(stdout)
    global quiet = maximum(dP) <= 1e-3 ? quiet + 1 : 0
    r.status == :failed && (println("$coord: block failed (no accepted step)"); break)
    r.status == :converged && (println("$coord: converged (optimality ≤ 1e-6)"); quiet >= 0 && break)
    quiet >= 3 && r.opt <= 1e-6 && (println("$coord: PASS (3 quiet windows, optimality ≤ 1e-6)"); break)
end
out = replace(ck, ".h5" => "_$(coord)$(pert != 0 ? "_p$(pert)" : "").h5")
h5open(out, "w") do f
    f["u"] = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    f["x1"] = parent(x[1]); f["Delta"] = Δ; f["mu"] = μ; f["accepted"] = total
end
println("$coord: state saved to $out (resumable)")
