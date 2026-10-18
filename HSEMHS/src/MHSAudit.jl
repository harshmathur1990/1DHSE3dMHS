# M5: solver-independent optimality audits (CODEX_REVIEW_M5_SECOND_FOLLOWUP.md, Q1 and Q3).
#
# A fixed reference metric D_ref is taken once, at a common reference state, and used for every iterate
# and every solver that is compared. The audit correction is NOT truncated by any optimizer's trust
# radius or per-step cap:
#   pressure:  s* = argmin gᵀs + ½ sᵀD_ref s  subject to u + s ≥ u_lo and Σ_anchor α s = 0;
#   Ã, b̄:      the unconstrained diagonal correction −g/D_ref, b̄ measured in units of σ_B.
# The full-state measure is the predicted decrease of these corrections, in the units of Φ:
#   m = (−gᵀs* − ½ s*ᵀD s*) + ½ Σ_Ã g²/D + ½ g_b̄²/D_b̄.

"""All ranks' values of a local vector, concatenated (local lengths may differ between ranks)."""
function _allgatherv(par::ParCtx, v::Vector{Float64})
    counts = MPI.Allgather(Int32(length(v)), par.comm)
    out = Vector{Float64}(undef, sum(counts))
    MPI.Allgatherv!(v, MPI.VBuffer(out, counts), par.comm)
    return out
end

"""
    reference_metric(pb, w, x; floor_rel = 1e-6) -> D_ref::MHSState

Positive reference diagonal at the state of `x` (`w` must hold it): `gn_diagonal!` — the v-coordinate GN
diagonal for the pressure, per-layer Hutchinson means for Ã (of the full Hessian with the default
`precond_atilde = :layer`, of H_GN with `:layer_gn`) and the b̄ curvature. Each field is floored at
`floor_rel` × its global median, so a few near-zero entries cannot dominate the audit.
"""
function reference_metric(pb::SMHSProblem, w::SMHSWork, x::MHSState; floor_rel::Float64 = 1e-6)
    par = pb.par; b = par.block
    D = gn_diagonal!(similar_state(x), pb, w, x)
    for f in 1:4
        vals = Float64[]
        for k in axes(D[f], 3), j in 1:b.ny, i in 1:b.nx
            push!(vals, D[f][i, j, k])
        end
        allv = _allgatherv(par, vals)
        med = median(allv)
        Df = D[f]; owned_map!((i, j, k) -> max(Df[i, j, k], floor_rel * med), Df, par)
    end
    objective!(pb, w, x)
    return D
end

"""
    full_audit(pb, w, x, Dref, ulo; ident = nothing, nworst = 5) -> NamedTuple

Fixed-metric audit at the state of `x` (`w` must hold it; it holds it on exit). Returns the pressure
correction statistics (max, p50/p95/p99 of |s*|, D-norm, model decrease, worst nodes with their g, D,
P/P_ref, bound distance and active status), the Ã and b̄ components (gradient norms; b̄ in σ_B units)
and the full-state measure `m`. With a fixed `ident` mask, the |s*| maximum is also split by it.
"""
function full_audit(pb::SMHSProblem, w::SMHSWork, x::MHSState, Dref::MHSState, ulo::Field;
                    ident = nothing, nworst::Int = 5)
    par = pb.par; b = par.block
    g = similar_state(x)
    f = fg!(g, pb, w, x)[1]
    s = newfield(par); t = newfield(par)
    owned_map!((i, j, k) -> -g[1][i, j, k] / Dref[1][i, j, k], t, par)
    anchor_box_project!(s, t, (i, j, k) -> min(ulo[i, j, k] - w.u[i, j, k], 0.0), (i, j, k) -> Inf,
                        (i, j, k) -> pb.alpha[i, j] / Dref[1][i, j, k], pb)
    # pressure part
    nodes = [(abs(s[i, j, k]), i, j, k) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx]
    absS = _allgatherv(par, Float64[q[1] for q in vec(nodes)])
    q(p) = quantile(absS, p)
    gs = _sum1(par, (i, j, k) -> g[1][i, j, k] * s[i, j, k]); sDs = _sum1(par, (i, j, k) -> Dref[1][i, j, k] * s[i, j, k]^2)
    mu = -gs - 0.5sDs
    # global worst nodes: each rank contributes its local top `nworst`; every rank selects the global top
    loc = sort(vec(nodes); rev = true)[1:min(nworst, length(nodes))]
    flat = Float64[]
    for (_, i, j, k) in loc
        append!(flat, (abs(s[i, j, k]), b.i0 + i - 1, b.j0 + j - 1, k, s[i, j, k], g[1][i, j, k], Dref[1][i, j, k],
                       exp(w.u[i, j, k] - pb.lnPref), w.u[i, j, k] - ulo[i, j, k],
                       (w.u[i, j, k] - ulo[i, j, k] <= 1e-6 && g[1][i, j, k] > 0) ? 1.0 : 0.0))
    end
    allf = reshape(_allgatherv(par, flat), 10, :)
    order = sortperm(allf[1, :]; rev = true)[1:min(nworst, size(allf, 2))]
    wl = [(i = Int(allf[2, n]), j = Int(allf[3, n]), k = Int(allf[4, n]), s = allf[5, n], g = allf[6, n], D = allf[7, n],
           P_over_Pref = allf[8, n], bound_dist = allf[9, n], active = allf[10, n] == 1.0) for n in order]
    split = ident === nothing ? (NaN, NaN) :
        (global_max(par, maximum((ident[i, j, k] ? abs(s[i, j, k]) : 0.0) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)),
         global_max(par, maximum((ident[i, j, k] ? 0.0 : abs(s[i, j, k])) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)))
    # Ã and b̄ parts
    gA2 = 0.0; mA = 0.0
    for fld in 2:4, k in axes(g[fld], 3), j in 1:b.ny, i in 1:b.nx
        gA2 += g[fld][i, j, k]^2; mA += 0.5 * g[fld][i, j, k]^2 / Dref[fld][i, j, k]
    end
    gA2 = MPI.Allreduce(gA2, +, par.comm); mA = MPI.Allreduce(mA, +, par.comm)
    gb = getscalar(par, g[5])
    Db = max(MPI.Allreduce(rank0(par) ? Dref[5][1, 1, 1] : 0.0, +, par.comm), 1e-300)   # the scalar lives on rank 0
    σB = sqrt(_sum1(par, (i, j, k) -> pb.sig[3][i, j, k]^2) / pb.N)
    mb = 0.5gb^2 / Db
    objective!(pb, w, x)
    return (f = f, m = mu + mA + mb, m_u = mu, m_A = mA, m_b = mb,
            u_max = q(1.0), u_p50 = q(0.5), u_p95 = q(0.95), u_p99 = q(0.99), u_Dnorm = sqrt(sDs),
            u_max_ident = split[1], u_max_rest = split[2], gA = sqrt(gA2), gb_sigma = abs(gb) * σB, worst = wl)
end
