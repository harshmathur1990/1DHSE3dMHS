# Codex review M5, experiment step 1: one alternating cycle at λ_F = 1e-4, λ_Q = 1e-3 (300 Ã iterations,
# 5 pressure steps), saved as the frozen-B checkpoint together with the original collapse bound.
# usage (from HSEMHS/): julia --project=. -t 8 scripts/m5_bifrost/frozenB_checkpoint.jl OUT.h5
include(joinpath(@__DIR__, "bif_setup.jl"))
out = ARGS[1]
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
w = H.SMHSWork(par)
st, why, no, hist = H.alternating_solve!(pb, w, x; max_outer = 1, a_iter = 300, u_steps = 5, verbose = true)
u0 = H.init_state(pb)[1]                                      # the initial u that defines the collapse bound
h5open(out, "w") do f
    for q in 1:5
        f["x$q"] = parent(x[q])
    end
    f["u_init"] = parent(u0)
end
@printf("checkpoint written: %s (cycle: %s, Φ = %.6e)\n", out, string(st), hist[end].f)
