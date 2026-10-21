# Full-resolution kernel timing: objective+gradient, GN product, 20 init_A! CG iterations, gn_diagonal!, vector update.
include(joinpath(@__DIR__, "bif_setup.jl"))
pb, x, par, b, NZ = bif_problem(1, 1; init_iter = 0, lambda_F = 1e-4, lambda_Q = 1e-3)
w = H.SMHSWork(par); g = H.similar_state(x); wl = H.SMHSWork(par)
H.fg!(g, pb, w, x); H.gn_hvp!(g, pb, w, wl, x); H.init_A!(pb, w, deepcopy(x); max_iter = 2)   # compile
t1 = minimum(@elapsed(H.fg!(g, pb, w, x)) for _ in 1:3)
t2 = minimum(@elapsed(H.gn_hvp!(g, pb, w, wl, x)) for _ in 1:3)
t3 = @elapsed H.init_A!(pb, w, deepcopy(x); max_iter = 20)
y = H.similar_state(x); t4 = minimum(@elapsed(H.lincomb!(y, 1.0, x, 2.0, g, par)) for _ in 1:5)
H.objective!(pb, w, x); t5 = @elapsed H.gn_diagonal!(H.similar_state(x), pb, w, x)
@printf("BENCH threads %d: fg! %.3f s | gn_hvp! %.3f s | init_A! %.3f s/iteration | lincomb! %.4f s | gn_diagonal! %.2f s\n", Threads.nthreads(), t1, t2, t3 / 20, t4, t5)
