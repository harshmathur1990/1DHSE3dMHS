# M5: joint damped trust-region Gauss–Newton on the full state (v, Ã, b̄) — the challenger of the solver
# comparison proposed in CODEX_REVIEW_M5_FOLLOWUP.md (Q3) and CODEX_REVIEW_M5_SECOND_FOLLOWUP.md (Q2).
#
# One model per trust-region iteration, at the accepted base state (trial points in a separate workspace):
#   1. Steihaug–Toint PCG on (H_GN + μD) y = −g, D the preconditioner diagonal (`gn_diagonal!`: exact GN
#      v-diagonal; per-layer Hutchinson means for Ã — of the full Hessian by default, of H_GN with
#      precond_atilde = :layer_gn; b̄ curvature), ε-active pressure nodes fixed, D-norm trust ball, preconditioned
#      residual stop, exit reason recorded (the residual is recomputed for the returned direction);
#   2. candidate: the pressure part re-centred and projected exactly onto bounds ∩ {Σ_anchor α δu = 0}; the Ã
#      and b̄ parts unconstrained; the WHOLE step then scaled toward 0 until it fits the D-norm ball (feasible:
#      the box contains 0, the anchor constraint is linear);
#   3. projected-Cauchy safeguard on the full state (≥ 0.1 of the Cauchy model decrease, else the Cauchy step);
#   4. accept on the actual/predicted ratio; radius updated from the final step.
# The gauge null directions of Ã (curl-free changes of A) carry no curvature; the Levenberg damping μD keeps
# the system positive definite, and gauge components of the step change neither B nor Φ. They are not
# removed by a quotient solve; all comparisons are made through physical quantities (P, B) and Φ.

"""
    joint_gn_solve!(pb, w, x, ulo; work_budget, max_accepted, max_candidates, cg_iter, cg_rtol, du_cap, μ0,
                    Δ_init, Δ0, block_scale, magnetic_only, diagnostics, callback, verbose)
        -> (status, reason, accepted, (history, candidates, counts))

Joint trust-region GN until `work_budget` equivalent GN products (`work_equivalent`) are spent (or `max_accepted`
steps accepted, or `max_candidates` candidates tried), measured from the call.
- `Δ_init`: `:pressure` (all pressure nodes moving 0.5 in ln P; the original rule) or `:full` (0.5 in every
  coordinate of the D-norm); `Δ0` overrides both.
- `block_scale = (s_u, s_A, s_b)`: multiplies the pressure / Ã / b̄ parts of D in the trust norm and the
  preconditioner (s_u > 1 makes pressure stiffer in the step geometry); `magnetic_only` freezes the pressure.
- `anchor_tangent`: keep the pressure part of the PCG iterates in {Σ_anchor α δv = 0} by the preconditioned
  projection P = I − M⁻¹a(aᵀM⁻¹a)⁻¹aᵀ (projected CG, Gould–Hribar–Nocedal), so the step measured by the trust
  norm is the physical δu: the anchor re-centring in the candidate then changes nothing (Codex M5d review §4).
- `diagnostics`: per candidate (not charged to the work budget) the D-norm shares of the pressure, Ã and b̄
  parts before and after projection/capping, the actual max |ΔP/P| and max |ΔB|/max(|B|, σ), and the split
  of the quadratic model into pressure, magnetic and cross terms.
- `face_hard`: per-face hard data bounds |c_f| ≤ face_hard[q][i, j, k] (σ units, layout of `Bobsf`; M5f pilot). B is
  linear in (Ã, b̄), so every candidate is scaled by the fraction to the boundary (`bound_tau` × the largest
  feasible fraction) before its trial; a rejected truncated candidate shrinks the radius from Δ, not from its
  truncated length; the base state must satisfy the bounds. B stays exactly divergence-free (no clipping).
  Combine with `set_face_hinge!` so that the model itself sees the bounds. M5g (Codex M5f review finding 2): the GN
  and Cauchy candidates are each made face-feasible BEFORE their model decreases are compared, the Cauchy search
  truncates every trial, and the fraction is computed from the exact linear map δB = A_ref curl δÃ + δb̄ ẑ. With the
  barrier's local curvature in D (`gn_diagonal!`), the Cauchy direction −D⁻¹g is barrier-scaled, so it moves little in
  coordinates that push active faces outward; radial truncation alone guarantees no tangent motion. Inputs are
  validated (0 < `bound_tau` < 1, finite positive bounds, a feasible base — strictly inside with a barrier), and the
  trial state is re-checked against the bounds (`bound_check_tol`, σ units) before acceptance.
`callback(acc, row)` runs after every accepted step with the work counters saved and restored; returning `:stop` ends
the solve (status `stalled`, reason "callback stop": a declared stagnation stop). `cand_sink` (a Ref) receives the live
candidate vector. `cand_callback(record)` runs after every REJECTED candidate (x and w unchanged, so the base state can be
checkpointed); `:stop` ends the solve likewise.
The result tuple also carries `μ` and `Δ` (restart state).
"""
function joint_gn_solve!(pb::SMHSProblem, w::SMHSWork, x::MHSState, ulo::Field; work_budget::Float64 = 3000.0,
                         max_accepted::Int = typemax(Int), max_candidates::Int = typemax(Int),
                         cg_iter::Int = 100, cg_rtol::Float64 = 1e-2, du_cap::Float64 = 5.0, μ0::Float64 = 1e-6,
                         Δ_init::Symbol = :pressure, Δ0::Float64 = NaN, block_scale = (1.0, 1.0, 1.0),
                         magnetic_only::Bool = false, anchor_tangent::Bool = false, diagnostics::Bool = false,
                         face_hard = nothing, bound_tau::Float64 = 0.999, bound_check_tol::Float64 = 1e-9,
                         callback = nothing, cand_sink = nothing, cand_callback = nothing, verbose::Bool = false)
    par = pb.par; b = par.block
    if face_hard !== nothing
        0 < bound_tau < 1 || invalid("joint_gn_solve!: bound_tau must lie in (0, 1)")
        pb.dfaces || invalid("joint_gn_solve!: face_hard needs data_on_faces")
        objective!(pb, w, x)
        v, nbad = face_violation(pb, w, face_hard)
        nbad == 0 || invalid("joint_gn_solve!: face_hard bounds must be finite and > 0 ($nbad faces are not)")
        strict = pb.hinge !== nothing && pb.hinge.barrier
        (strict ? v < 0 : v <= 0) || invalid("joint_gn_solve!: the base state violates face_hard (max(|c| − u) = $v)")
    end
    w0 = work_equivalent()
    wl = SMHSWork(par); wt = SMHSWork(par)
    g = similar_state(x); D = similar_state(x); Dinv = similar_state(x)
    r = similar_state(x); zr = similar_state(x); p = similar_state(x); q = similar_state(x); y = similar_state(x)
    δ = similar_state(x); δc = similar_state(x); cdir = similar_state(x); xt = similar_state(x); act = newfield(par)
    t1 = newfield(par); s1 = newfield(par)
    f = 0.0
    function rebuild!()
        f = fg!(g, pb, w, x)[1]
        gn_diagonal!(D, pb, w, x)                        # leaves w at x
        for (fld, sc) in ((1, block_scale[1]), (2, block_scale[2]), (3, block_scale[2]), (4, block_scale[2]), (5, block_scale[3]))
            sc == 1.0 && continue
            Df = D[fld]; owned_map!((i, j, k) -> sc * Df[i, j, k], Df, par)
        end
        for fld in eachindex(D)
            Df = D[fld]; Di = Dinv[fld]
            owned_map!((i, j, k) -> 1 / Df[i, j, k], Di, par)
        end
        fill_owned!((i, j, k) -> (magnetic_only || (w.u[i, j, k] - ulo[i, j, k] <= 1e-6 && g[1][i, j, k] > 0)) ? 1.0 : 0.0, act, par)
        objective!(pb, w, x)
        return f
    end
    ones_state = similar_state(x)
    for f in 1:4
        of = ones_state[f]; owned_map!((i, j, k) -> 1.0, of, par)
    end
    rebuild!()
    free!(v) = (vf = v[1]; owned_map!((i, j, k) -> act[i, j, k] == 1.0 ? 0.0 : vf[i, j, k], vf, par); v)
    Dnorm2(v) = (divdiag!(zr, v, Dinv, par); dotN(par, v, zr))       # vᵀ D v   (zr is scratch here)
    Hv!(out, v) = gn_hvp!(out, pb, w, wl, v)
    function predicted(dlt)
        Hv!(q, dlt)
        return -(dotN(par, g, dlt) + 0.5 * dotN(par, dlt, q))
    end
    # face bounds: largest feasible fraction of dlt from the exact linear map (scratch: wt.A, wt.B), then scale by
    # bound_tau × that fraction when it is < 1; returns the fraction (1 without bounds)
    function bound_truncate!(dlt)
        face_hard === nothing && return 1.0
        a = face_fraction(pb, w, wt, dlt, face_hard)
        a < 1 && lincomb!(dlt, bound_tau * a, dlt, 0.0, dlt, par)
        return a
    end
    # actual step from a direction: project the pressure part, keep Ã/b̄, fit the whole step into the ball
    ntrust = 0
    function candidate!(dlt, dir, τ, Δ)
        lincomb!(dlt, τ, dir, 0.0, dir, par)
        owned_map!((i, j, k) -> dlt[1][i, j, k], t1, par)
        m = anchor_mean(pb, t1); owned_map!((i, j, k) -> t1[i, j, k] - m, t1, par)
        ae = anchor_box_project!(s1, t1, (i, j, k) -> min(max(ulo[i, j, k] - w.u[i, j, k], -du_cap), 0.0), (i, j, k) -> du_cap,
                                 (i, j, k) -> 1.0, pb)
        copyto!(parent(dlt[1]), parent(s1))
        nrm = sqrt(max(Dnorm2(dlt), 0.0))
        if nrm > Δ * (1 + 1e-12)
            ntrust += 1; lincomb!(dlt, Δ / nrm, dlt, 0.0, dlt, par)
        end
        return ae
    end
    Δ = if !isnan(Δ0)
        Δ0
    elseif Δ_init == :full                               # 0.5 in every coordinate of the D-norm
        0.5 * sqrt(max(sum(sdot_local(par, D[f], ones_state[f]) for f in 1:4) |> v -> MPI.Allreduce(v, +, par.comm), 1e-300))
    else                                                 # as in the pressure block: all nodes moving 0.5 in ln P
        0.5 * sqrt(_sum1(par, (i, j, k) -> D[1][i, j, k]))
    end
    Δstart = Δ; μ = μ0
    Mia = similar_state(x)                                 # M⁻¹a for the anchor constraint (pressure field only)
    function prec!(zz, rr, μc)                             # zz = P M⁻¹ rr,  M = (1 + μc) D on the free set
        divdiag!(zz, rr, D, par); lincomb!(zz, 1 / (1 + μc), zz, 0.0, zz, par); free!(zz)
        if anchor_tangent
            for f in 2:5; fill!(parent(Mia[f]), 0.0); end
            m1 = Mia[1]; fill!(parent(m1), 0.0)
            forrows(pb.ka:pb.ka, par) do i, j, k
                m1[i, j, k] = act[i, j, k] == 1.0 ? 0.0 : pb.alpha[i, j] / ((1 + μc) * D[1][i, j, k])
            end
            aMa = MPI.Allreduce(sum(pb.alpha[i, j] * Mia[1][i, j, pb.ka] for j in 1:b.ny, i in 1:b.nx), +, par.comm)
            az = MPI.Allreduce(sum(pb.alpha[i, j] * zz[1][i, j, pb.ka] for j in 1:b.ny, i in 1:b.nx), +, par.comm)
            aMa > 0 && lincomb!(zz, 1.0, zz, -az / aMa, Mia, par)
        end
        return zz
    end
    nacc = 0; ncand = 0; ncauchy = 0; nbound = 0; nexit = Dict{Symbol,Int}(); hist = NamedTuple[]
    cands = cand_sink === nothing ? NamedTuple[] : (cand_sink[] = NamedTuple[])   # cand_sink: a Ref the caller can read live
    status = max_iter; reason = "work budget"
    while work_equivalent() - w0 < work_budget && nacc < max_accepted && ncand < max_candidates
        # ---- 1. Steihaug–Toint PCG on the full state ----
        lincomb!(r, -1.0, g, 0.0, g, par); free!(r)
        prec!(zr, r, μ)
        lincomb!(p, 1.0, zr, 0.0, zr, par); lincomb!(y, 0.0, y, 0.0, y, par)
        rz = dotN(par, r, zr); rz0 = rz; cg_rel = 1.0; cg_exit = :budget
        rz0 <= 0 && (status = converged; reason = "zero preconditioned gradient"; break)
        for _ in 1:cg_iter
            Hv!(q, p); free!(q)
            divdiag!(zr, p, Dinv, par)                    # D p
            lincomb!(q, 1.0, q, μ, zr, par)               # (H + μD) p
            pq = dotN(par, p, q)
            yDy = Dnorm2(y); pDp = Dnorm2(p)
            divdiag!(zr, p, Dinv, par); yDp = dotN(par, y, zr)
            τt = pDp > 0 ? max((-yDp + sqrt(max(yDp^2 - pDp * (yDy - Δ^2), 0.0))) / pDp, 0.0) : Inf
            α = pq > 0 ? rz / pq : Inf
            if α >= τt
                lincomb!(y, 1.0, y, τt, p, par); cg_exit = pq > 0 ? :boundary : :negcurv
                # residual of the RETURNED direction: r = −g − (H + μD) y on the free set
                Hv!(q, y); free!(q); divdiag!(zr, y, Dinv, par); lincomb!(q, 1.0, q, μ, zr, par)
                lincomb!(r, -1.0, g, -1.0, q, par); free!(r)
                prec!(zr, r, μ)
                cg_rel = sqrt(max(dotN(par, r, zr), 0.0) / rz0)
                break
            end
            lincomb!(y, 1.0, y, α, p, par)
            lincomb!(r, 1.0, r, -α, q, par)
            prec!(zr, r, μ)
            rzn = dotN(par, r, zr)
            cg_rel = sqrt(max(rzn, 0.0) / rz0)
            cg_rel <= cg_rtol && (cg_exit = :residual; break)
            lincomb!(p, 1.0, zr, rzn / rz, p, par); free!(p)
            rz = rzn
        end
        nexit[cg_exit] = get(nexit, cg_exit, 0) + 1
        # ---- 2. candidate: projected, fitted into the ball, then made face-feasible ----
        ncand += 1
        aerr = candidate!(δ, y, 1.0, Δ)
        αg = bound_truncate!(δ)
        pred = predicted(δ)
        # ---- 3. projected-Cauchy safeguard, every trial face-feasible before its model decrease is computed ----
        lincomb!(r, -1.0, g, 0.0, g, par); free!(r); prec!(cdir, r, 0.0)   # projected steepest descent −P D⁻¹ g
        cn = sqrt(max(Dnorm2(cdir), 0.0))
        predc = -Inf; αc = 1.0; τc = cn > 0 ? Δ / cn : 0.0
        for _ in 1:20
            candidate!(xt, cdir, τc, Δ); at = bound_truncate!(xt); pc = predicted(xt)
            if pc > predc
                predc = pc; αc = at; copystate!(δc, xt, par)
            elseif predc > 0
                break
            end
            τc /= 2
        end
        usedc = false; pred_gn = pred
        if predc > 0 && pred < 0.1 * predc
            ncauchy += 1; copystate!(δ, δc, par); pred = predc; usedc = true
        end
        αb = usedc ? αc : αg
        αb < 1 && (nbound += 1)
        nfin = sqrt(max(Dnorm2(δ), 0.0))
        diag = NamedTuple()
        if diagnostics                                    # of the FINAL (selected, face-feasible) step
            saved = copy(WORK_COUNTS)
            blk(v, fs) = (tmp = similar_state(x); for f in 1:5; ff = tmp[f]; vf = v[f]; owned_map!((i, j, k) -> f in fs ? vf[i, j, k] : 0.0, ff, par); end; tmp)
            share(v) = (nt = max(Dnorm2(v), 1e-300); (u = Dnorm2(blk(v, (1,))) / nt, A = Dnorm2(blk(v, (2, 3, 4))) / nt, b = Dnorm2(blk(v, (5,))) / nt))
            δu = blk(δ, (1,)); δA = blk(δ, (2, 3, 4, 5))
            Hv!(q, δu); quu = 0.5 * dotN(par, δu, q); qua = dotN(par, δA, q)
            Hv!(q, δA); qaa = 0.5 * dotN(par, δA, q)
            gu = dotN(par, g, δu); ga = dotN(par, g, δA)
            lincomb!(xt, 1.0, x, 1.0, δ, par); objective!(pb, wt, xt)
            dPmax = global_max(par, maximum(abs(expm1(wt.u[i, j, k] - w.u[i, j, k])) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
            dBmax = measured_change_B(pb, wt, (copy(parent(wt.u)), map(B -> copy(parent(B)), w.B)))
            diag = (share_y = share(y), share_final = share(δ), dPmax = dPmax, dBmax = dBmax,
                    pred_u = -(gu + quu), pred_A = -(ga + qaa), pred_cross = -qua)
            copyto!(WORK_COUNTS, saved)
        end
        # ---- 4. trial in a separate workspace ----
        lincomb!(xt, 1.0, x, 1.0, δ, par)
        ft, tt, oob = objective!(pb, wt, xt)
        margin = -global_max(par, -minimum(wt.u[i, j, k] - ulo[i, j, k] for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))
        bviol = face_hard === nothing ? -Inf : face_violation(pb, wt, face_hard)[1]
        ρ = (oob || pred <= 0) ? -Inf : (f - ft) / pred
        Δused = Δ
        ok = ρ > 1e-4 && ft < f && margin >= -1e-12 && bviol <= bound_check_tol
        push!(cands, (candidate = ncand, accepted = ok, rho = ρ, pred = pred, pred_gn = pred_gn, predc = predc, actual = f - ft,
                      Delta = Δused, ynorm = nfin, cauchy = usedc, cg_exit = cg_exit, cg_rel = cg_rel, bound_frac = αb,
                      bound_frac_gn = αg, bound_frac_cauchy = αc, bound_viol = bviol, mu = μ, diag = diag))
        if ok
            copystate!(x, xt, par)
            rebuild!()
            nacc += 1
            ρ > 0.75 && nfin >= 0.99Δ && (Δ *= 2)
            ρ < 0.25 && (Δ = nfin / 4)
            μ = ρ > 0.75 ? max(μ / 3, 1e-12) : (ρ < 0.25 ? min(4μ, 1e2) : μ)
            row = (accepted = nacc, f = f, terms = tt, rho = ρ, pred = pred, predc = predc, cauchy = usedc, ynorm = nfin,
                   Delta_used = Δused, Delta = Δ, mu = μ, cg_exit = cg_exit, cg_rel = cg_rel, margin = margin, anchor_err = aerr,
                   bound_frac = αb, bound_viol = bviol, work = work_equivalent() - w0)
            push!(hist, row)
            if verbose && rank0(par) && nacc % 10 == 0
                @printf("  jgn %4d: Φ %.6e  Φ_F %.3e  ρ %.2f  ‖δ‖_D/Δ %.2f  CG %s  Cauchy %d/%d  work %.0f\n", nacc, f, tt[1], ρ, nfin / Δused,
                        string(cg_exit), ncauchy, ncand, row.work)
                flush(stdout)
            end
            if callback !== nothing
                saved = copy(WORK_COUNTS); cbr = callback(nacc, row); copyto!(WORK_COUNTS, saved)
                cbr === :stop && (status = stalled; reason = "callback stop"; break)
            end
        else
            if cand_callback !== nothing                  # rejected candidate: x and w are still the accepted base state
                saved = copy(WORK_COUNTS); cbr = cand_callback(cands[end]); copyto!(WORK_COUNTS, saved)
                cbr === :stop && (status = stalled; reason = "callback stop"; break)
            end
            Δ = (αb < 1 ? Δused : nfin) / 4; μ = min(4μ, 1e2)
            if Δ < 1e-14 * Δstart
                status = stalled; reason = "trust radius collapsed"; break
            end
        end
    end
    reason == "callback stop" || (nacc >= max_accepted && (reason = "max_accepted"))
    reason == "callback stop" || (ncand >= max_candidates && (reason = "max_candidates"))
    return status, reason, nacc, (history = hist, candidates = cands, counts = (cand = ncand, cauchy = ncauchy, trust = ntrust, bound = nbound, cg_exit = nexit), Δ = Δ, μ = μ)
end

"""
    face_violation(pb, w, u) -> (max(|c_f| − u_f), number of non-finite or non-positive u_f)

At the state of the last `objective!(pb, w, ·)`, c_f = (B_f − I·y_f)/σ_f over every owned face (all ranks).
"""
function face_violation(pb::SMHSProblem, w::SMHSWork, u)
    par = pb.par; b = par.block
    m = -Inf; nbad = 0
    for q in 1:3
        Bc = w.B[q]; Bo = pb.Bobsf[q]; iv = pb.invsf2[q]; uq = u[q]
        for k in axes(Bc, 3), j in 1:b.ny, i in 1:b.nx
            t = uq[i, j, k]
            (isfinite(t) && t > 0) || (nbad += 1; continue)
            m = max(m, abs((Bc[i, j, k] - Bo[i, j, k]) * sqrt(iv[i, j, k])) - t)
        end
    end
    return global_max(par, m), MPI.Allreduce(nbad, +, par.comm)
end

"""
    fraction_to_boundary(c0, dc, u) -> α

Largest α ∈ [0, 1] with |c0 + α dc| ≤ u, for a face inside its bound (|c0| ≤ u).
"""
@inline function fraction_to_boundary(c0::Float64, dc::Float64, u::Float64)
    dc == 0 && return 1.0
    a = dc > 0 ? (u - c0) / dc : (-u - c0) / dc
    return clamp(a, 0.0, 1.0)
end

"""
    face_fraction(pb, w, ws, δ, u) -> α

Largest α ∈ [0, 1] such that the state at `w` (its B) moved by α δ satisfies every face bound u, computed from the
exact linear map δB = A_ref curl δÃ + δb̄ ẑ (ws.A, ws.B are scratch; no objective evaluation, no work counted).
"""
function face_fraction(pb::SMHSProblem, w::SMHSWork, ws::SMHSWork, δ::MHSState, u)
    par = pb.par; b = par.block; Ar = pb.Aref
    for c in 1:3
        δc = δ[c+1]; owned_map!((i, j, k) -> Ar * δc[i, j, k], ws.A[c], par)
    end
    exchange_halos!(par, collect(ws.A))
    curlA!(ws.B, ws.A, getscalar(par, δ[5]), pb.s, par)
    am = fill(1.0, Threads.maxthreadid())
    for q in 1:3
        B0 = w.B[q]; dB = ws.B[q]; Bo = pb.Bobsf[q]; iv = pb.invsf2[q]; uq = u[q]
        forrows(collect(axes(B0, 3)), par) do i, j, k
            isg = sqrt(iv[i, j, k])
            a = fraction_to_boundary((B0[i, j, k] - Bo[i, j, k]) * isg, dB[i, j, k] * isg, uq[i, j, k])
            a < am[Threads.threadid()] && (am[Threads.threadid()] = a)
        end
    end
    return -global_max(par, -minimum(am))
end
