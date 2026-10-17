include(joinpath(@__DIR__, "bif_setup.jl"))
pb, x, par, b, NZ = bif_problem(12, 2; precond_ufloor = 1e-2, lambda_Q = parse(Float64, ARGS[1]))
w = H.SMHSWork(par); tm = time()
xs, tab, flags = H.discrepancy_solve!(pb, w, x; opts = H.LBFGSOptions(max_iter = 20000, xtol_rel = 1e-3))
for r in tab
    @printf("λ_F %.3g: %s (%s) it %d bfirst %s χ² %.4f Φ_F %.3e Φ_Q %.3e Φ_S %.3e\n", r.lambda_F, r.status, "", r.iterations, r.bfirst, r.chi2, r.terms[1], r.terms[2], r.terms[4])
end
f, t, _ = H.objective!(pb, w, xs); rms, p99 = H.layer_residual_stats(pb, w)
@printf("selected λ_F %.3g flags %s (%.0f s)\n", flags["lambda_F"], string(flags), time() - tm)
@printf("fractions: Φ_Q/Φ %.2e, (Φ_S+Φ_U)/Φ %.2e; RMS|R|/S max %.2e, layers meeting real-data tol (1e-2): %d of %d\n", t[2] / f, (t[4] + t[5]) / f, maximum(rms), count(<=(1e-2), rms), NZ)
dPu = H.dP_unobs!(H.newfield(par), pb, w); fr = H.unidentifiable_fraction(pb, w, dPu)
layers_ok = [k for k in 1:NZ if rms[k] <= 1e-2]
@printf("per-layer RMS|R|/S (top→bottom, every 8th): %s\n", join((@sprintf("%.1e", rms[k]) for k in 1:8:NZ), " "))
@printf("unidentifiable fraction (every 8th): %s\n", join((@sprintf("%.2f", fr[k]) for k in 1:8:NZ), " "))
