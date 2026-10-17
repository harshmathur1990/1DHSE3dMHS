# Codex review M5, experiment steps 2–3: from the frozen-B checkpoint, solve the pressure problem with B
# fixed, in ln P (`lnP`) or bounded scaled pressure p = P/P_ref (`P`). Chunks of 100 accepted updates
# (trust radius restarts per chunk), up to 500 accepted; PCG rtol 1e-4, ≤ 300 iterations.
# usage: julia --project=. -t 6 scripts/m5_bifrost/frozenB_branch.jl CHECKPOINT.h5 lnP|P [perturb]
include(joinpath(@__DIR__, "bif_setup.jl"))
ck = ARGS[1]; coord = Symbol(ARGS[2]); pert = parse(Float64, get(ARGS, 3, "0"))
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
w = H.SMHSWork(par)
h5open(ck) do f
    for q in 1:5
        copyto!(parent(x[q]), read(f["x$q"]))
    end
    global uinit = H.newfield(par); copyto!(parent(uinit), read(f["u_init"]))
end
H.exchange_halos!(par, collect(x))
H.objective!(pb, w, x)
ulo = H.newfield(par); H.owned_map!((i, j, k) -> w.u[i, j, k] * 0 + (uinit[i, j, k] - H.anchor_mean(pb, uinit) + pb.lnPref) - 3.0, ulo, par)
if pert != 0                                                  # ±pert multiplicative pressure perturbation (repeatability)
    H.owned_map!((i, j, k) -> x[1][i, j, k] + log1p(pert * (isodd(hash((b.i0 + i, b.j0 + j, k, 0x77))) ? 1.0 : -1.0)), x[1], par)
    H.objective!(pb, w, x)
end
f0, t0, _ = H.objective!(pb, w, x)
cst = f0 - t0[1]                                              # B frozen: Φ_B, Φ_Q, Φ_S are constant
lF = pb.lamF
dPu = H.dP_unobs!(H.newfield(par), pb, w)
ident = [dPu[i, j, k] <= 0.1 * w.P[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]   # fixed reference mask
@printf("%s: start raw force %.6e; identifiable fraction %.3f; bound margin %.3e\n", coord, t0[1] / lF, count(ident) / length(ident),
        minimum(w.u[i, j, k] - ulo[i, j, k] for k in 1:NZ, j in 1:b.ny, i in 1:b.nx))
quiet = 0; total = 0; tm = time()
for chunk in 1:8
    uref = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    r = H.u_gauss_newton!(pb, w, x, ulo; steps = 100, cg_iter = 300, cg_rtol = 1e-4, du_max = 0.5, coord = coord, opt_tol = 1e-6, record = true)
    global total += r.accepted
    dP = [abs(expm1(w.u[i, j, k] - uref[i, j, k])) for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    act = [count(w.u[i, j, k] - ulo[i, j, k] <= 1e-6 for i in 1:b.nx, j in 1:b.ny) / (b.nx * b.ny) for k in 1:NZ]
    h = r.history
    mm = isempty(h) ? NaN : minimum(q.margin for q in h); ae = isempty(h) ? NaN : maximum(q.anchor_err for q in h)
    cgm = isempty(h) ? NaN : maximum(q.cg_rel for q in h)
    @printf("%s chunk %d: %s, accepted %d (total %d) tries %d | raw force %.6e | opt %.2e | max ΔP/P %.2e (ident %.2e, complement %.2e) | min margin %.2e anchor err %.1e | max PCG rel %.1e | bound-active layers %d (max frac %.2f) | %.0f s\n",
            coord, chunk, string(r.status), r.accepted, total, r.tries, (r.f - cst) / lF, r.opt, maximum(dP), maximum(dP[ident]),
            any(.!ident) ? maximum(dP[.!ident]) : 0.0, mm, ae, cgm, count(>(0), act), maximum(act), time() - tm)
    flush(stdout)
    global quiet = maximum(dP) <= 1e-3 ? quiet + 1 : 0
    (r.status == :failed) && (println("$coord: block failed"); break)
    quiet >= 3 && r.opt <= 1e-6 && (println("$coord: PASS window criterion (3 quiet windows) with optimality ≤ 1e-6"); break)
    total >= 500 && break
end
h5open(replace(ck, ".h5" => "_$(coord)$(pert != 0 ? "_p$(pert)" : "").h5"), "w") do f
    f["u"] = [w.u[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
end
