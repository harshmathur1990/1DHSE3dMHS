# M5g: incidence of outside-domain preconditioner probes at saved M5f barrier checkpoints (Codex M5f review finding 1).
# For every checkpoint (x1..x5) given, with the M5f barrier (λ, hard bounds 5/3 − margin on FIELD's support):
#   - min face slack u − |c| and the number of faces with slack < 1e-3;
#   - the PRE-M5g probes (gn_diagonal!(barrier_safe = false)): the 2 × probes Ã probes x ± 1e-4 z and the two b̄ probes
#     x ± ε_b e_b̄ — how many have Φ = Inf (outside the barrier domain);
#   - b̄ curvature: pre-M5g |FD|, M5g (barrier-free FD + exact local barrier term), and the exact GN product e_b̄ᵀ H e_b̄;
#   - Ã diagonal at the node of every component touching the minimum-slack face: pre-M5g layer mean, M5g, and the exact
#     GN diagonal (unit-vector GN product).
# usage (from HSEMHS/): julia --project=. -t 16 scripts/m5_bifrost/m5g_probe_audit.jl FIELD.h5 LAMBDA MARGIN CK1.h5 [CK2.h5 ...]
include(joinpath(@__DIR__, "bif_setup.jl"))
fieldf = ARGS[1]; lamH = parse(Float64, ARGS[2]); margin = parse(Float64, ARGS[3]); cks = ARGS[4:end]
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
sup = h5open(fieldf) do f; [read(f["support$q"]) for q in 1:3]; end
hard = (H.newfield(par), H.newfield(par), H.newfield0(par))
for q in 1:3
    k0 = first(axes(hard[q], 3))
    H.owned_map!((i, j, k) -> (sup[q][i, j, k-k0+1] > 0.5 ? 5.0 : 3.0) - margin, hard[q], par)
end
H.exchange_halos!(par, collect(hard))
H.set_face_hinge!(pb, hard; lambda = lamH, kind = :barrier)
w = H.SMHSWork(par); w2 = H.SMHSWork(par); wl = H.SMHSWork(par)
probes = 4; ε = 1e-4
for ckf in cks
    h5open(ckf) do f; for q in 1:5; copyto!(parent(x[q]), read(f["x$q"])); end; end
    H.exchange_halos!(par, collect(x))
    f0 = H.objective!(pb, w, x)[1]
    # slack and the minimum-slack face
    best = (Inf, 0, 0, 0, 0); nsl = 0
    for q in 1:3, k in axes(w.B[q], 3), j in 1:b.ny, i in 1:b.nx
        c = (w.B[q][i, j, k] - pb.Bobsf[q][i, j, k]) * sqrt(pb.invsf2[q][i, j, k]); sl = hard[q][i, j, k] - abs(c)
        sl < 1e-3 && (nsl += 1)
        sl < best[1] && (best = (sl, q, i, j, k))
    end
    # pre-M5g probes: replicate the Ã probe vectors of _gn_diagonal! (:layer) and the b̄ probe
    z = H.similar_state(x); xt = H.similar_state(x); ninf = 0
    for p in 1:probes
        for f in 2:4
            H.owned_map!((i, j, k) -> isodd(hash((b.i0 + i - 1, b.j0 + j - 1, k, f, p, 0xa11)) >> 7) ? 1.0 : -1.0, z[f], par)
        end
        for sgn in (1.0, -1.0)
            H.lincomb!(xt, 1.0, x, sgn * ε, z, par); isfinite(H.objective!(pb, w2, xt)[1]) || (ninf += 1)
        end
    end
    zb = H.similar_state(x); H.setscalar!(par, zb[5], 1.0)
    eb = 1e-6 * max(abs(H.getscalar(par, x[5])), 1e-4); binf = 0
    for sgn in (1.0, -1.0)
        H.lincomb!(xt, 1.0, x, sgn * eb, zb, par); isfinite(H.objective!(pb, w2, xt)[1]) || (binf += 1)
    end
    H.objective!(pb, w, x)
    Dold = H.gn_diagonal!(H.similar_state(x), pb, w, x; barrier_safe = false); H.objective!(pb, w, x)
    Dnew = H.gn_diagonal!(H.similar_state(x), pb, w, x); H.objective!(pb, w, x)
    hv = H.similar_state(x); H.gn_hvp!(hv, pb, w, wl, zb); exact_b = H.getscalar(par, hv[5])
    @printf("%s: Φ %.6f | min slack %.3e (comp %d at %d,%d,%d) #slack<1e-3 %d | pre-M5g probes outside the domain: Ã %d/%d, b̄ %d/2 (ε_b %.2e) | b̄ curvature: pre-M5g %.4e  M5g %.4e  exact GN %.4e (ratios %.3g, %.3g)\n",
            ckf, f0, best[1], best[2], best[3], best[4], best[5], nsl, ninf, 2probes, binf, eb,
            H.getscalar(par, Dold[5]), H.getscalar(par, Dnew[5]), exact_b, H.getscalar(par, Dold[5]) / exact_b, H.getscalar(par, Dnew[5]) / exact_b)
    # Ã nodes touching the minimum-slack face (q, i, j, k): exact GN diagonal by unit-vector products
    _, q, i, j, k = best
    nodes = q == 1 ? [(3, i, j, k - 1), (3, i, j, k), (4, i, j, k), (4, i, j - 1, k)] :
            q == 2 ? [(2, i, j, k - 1), (2, i, j, k), (4, i, j, k), (4, i - 1, j, k)] :
                     [(2, i, j, k), (2, i, j - 1, k), (3, i, j, k), (3, i - 1, j, k)]
    for (fld, ii, jj, kk) in nodes
        (1 <= ii <= b.nx && 1 <= jj <= b.ny && kk in axes(x[fld], 3)) || continue
        e = H.similar_state(x); e[fld][ii, jj, kk] = 1.0
        H.gn_hvp!(hv, pb, w, wl, e)
        @printf("    Ã%s node (%d,%d,%d): pre-M5g %.4e  M5g %.4e  exact GN %.4e (ratios %.3g, %.3g)\n", "xyz"[fld-1], ii, jj, kk,
                Dold[fld][ii, jj, kk], Dnew[fld][ii, jj, kk], hv[fld][ii, jj, kk], Dold[fld][ii, jj, kk] / hv[fld][ii, jj, kk], Dnew[fld][ii, jj, kk] / hv[fld][ii, jj, kk])
    end
    flush(stdout)
end
