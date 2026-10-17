include(joinpath(@__DIR__, "bif_setup.jl"))
pb, x, par, b, NZ = bif_problem(12, 2; precond_ufloor = 1e-2, lambda_Q = 1e-3, lambda_F = 1e-4)
w = H.SMHSWork(par)
H.lbfgs!(pb, w, x; opts = H.LBFGSOptions(max_iter = 2000, frozen = [1]))   # B-first stage
H.objective!(pb, w, x)
dPu = H.dP_unobs!(H.newfield(par), pb, w)
for chunk in 1:8
    H.objective!(pb, w, x); u0 = copy(parent(w.u)); B0 = map(B -> copy(parent(B)), w.B)
    H.lbfgs!(pb, w, x; opts = H.LBFGSOptions(max_iter = 100, gtol_rel = 1e-14))
    H.objective!(pb, w, x)
    ou = w.u.offsets
    dP = [abs(expm1(w.u[i, j, k] - u0[i-ou[1], j-ou[2], k-ou[3]])) for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    idn = [dPu[i, j, k] <= 0.1 * w.P[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    dB = 0.0; kB = 0
    for c in 1:3, k in axes(w.B[c], 3), j in 1:b.ny, i in 1:b.nx
        oB = w.B[c].offsets
        d = abs(w.B[c][i, j, k] - B0[c][i-oB[1], j-oB[2], k-oB[3]]) / max(abs(w.B[c][i, j, k]), 1 / sqrt(pb.invsf2[c][i, j, k]))
        d > dB && (dB = d; kB = k)
    end
    kP = argmax(dP)[3]
    @printf("after +%4d: max|ΔP/P| %.2e (layer %d), on identifiable nodes %.2e; max|ΔB|/max(|B|,σ) %.2e (layer %d); layers with max|ΔP/P|>1e-3: %d\n",
            100chunk, maximum(dP), kP, maximum(dP[idn]), dB, kB, count(k -> maximum(dP[:, :, k]) > 1e-3, 1:NZ))
end
