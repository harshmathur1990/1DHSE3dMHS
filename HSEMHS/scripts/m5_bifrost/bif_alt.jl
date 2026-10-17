# ARGS: lambda_F, lambda_Q, max_outer, a_iter
include(joinpath(@__DIR__, "bif_setup.jl"))
lF = parse(Float64, ARGS[1]); lq = parse(Float64, ARGS[2]); mo = parse(Int, ARGS[3]); ai = parse(Int, ARGS[4])
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = lF, lambda_Q = lq)
w = H.SMHSWork(par); f0, t0, _ = H.objective!(pb, w, x); tm = time()
st, why, no, hist = H.alternating_solve!(pb, w, x; max_outer = mo, a_iter = ai, u_steps = 5, verbose = true)
for h in hist[1:0]   # per-cycle lines are printed by verbose = true
    @printf("  cycle %3d: Φ %.4e  Φ_F %.3e Φ_Q %.3e χ² %.4f  change %.2e  u steps %d\n", h.outer, h.f, h.terms[1], h.terms[2], h.terms[3], h.change, h.usteps)
end
f, t, _ = H.objective!(pb, w, x); rms, p99 = H.layer_residual_stats(pb, w)
nb = 0; c3 = 0; mx = 0.0
for q in 1:3, k in axes(w.B[q], 3), j in 1:b.ny, i in 1:b.nx
    d = (w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]); global nb += 1; abs(d) > 3 && (global c3 += 1); global mx = max(mx, abs(d))
end
@printf("λ_F %g λ_Q %g: %s (%s) after %d cycles, %.0f s | Φ_F %.3e→%.3e | χ² %.4f frac3 %.1e max %.1f | Φ_Q/Φ %.2e | RMS|R|/S every 8th: %s\n",
        lF, lq, st, why, no, time() - tm, t0[1], t[1], t[3], c3 / nb, mx, t[2] / f, join((@sprintf("%.1e", rms[k]) for k in 1:8:NZ), " "))
