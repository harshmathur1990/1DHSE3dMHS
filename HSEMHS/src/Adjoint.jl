# Adjoint of the embedded equilibrium (design/adjoint): cotangents of the outputs of a recorded `solve_embedded!` pulled
# back to its inputs (ln T, B), for every embedded mode. Column modes: the exact derivative of the recorded computation
# with its discrete choices frozen (A1); Poisson modes: the implicit-function derivative at the returned state (A3).
#
#     res = solve_embedded!(ws, lnT, Bx, By, Bz; record = true)
#     bar = EquilibriumCotangent(ws); bar.rho .= …                    # any subset of the outputs
#     adj = adjoint_embedded!(ws, bar)                                # collective; adj.lnT, adj.Bx, adj.By, adj.Bz

"""
Cotangents of the outputs of one solve on this rank's slab, (vertical, y, x): `Pgas` [Pa⁻¹ of the caller's scalar],
`lnP`, `rho`, `ne`, `chi500`, `tau500`, `logtau500`, `z`. Fields left at zero do not contribute. Outputs that are constant
in a mode (z on the height grid, log τ on the τ axis, ln P on the mass axis) may carry any cotangent; it has no effect.
"""
mutable struct EquilibriumCotangent
    Pgas::Array{Float64,3}
    lnP::Array{Float64,3}
    rho::Array{Float64,3}
    ne::Array{Float64,3}
    chi500::Array{Float64,3}
    tau500::Array{Float64,3}
    logtau500::Array{Float64,3}
    z::Array{Float64,3}
end
const COTANGENT_FIELDS = (:Pgas, :lnP, :rho, :ne, :chi500, :tau500, :logtau500, :z)

"""Zero cotangents on the workspace's slab."""
function EquilibriumCotangent(ws::EmbeddedWorkspace)
    s = (length(ws.axis), ws.par.block.ny, ws.par.block.nx)
    return EquilibriumCotangent((zeros(s) for _ in COTANGENT_FIELDS)...)
end

"""
Result of `adjoint_embedded!`: the cotangents of ln T and of the cell-centred B on this rank's slab (vertical, y, x) and
`status` (:exact for the column modes; :stationary or :not_stationary for the Poisson modes), with details in `info`.
"""
struct EquilibriumAdjoint
    lnT::Array{Float64,3}
    Bx::Array{Float64,3}
    By::Array{Float64,3}
    Bz::Array{Float64,3}
    status::Symbol
    info::Dict{String,Any}
end

# ---- record --------------------------------------------------------------------------------------------------------

"""The per-column tapes and buffers a recorded solve fills (Embedded.jl passes it to the column stages)."""
function new_record(ws::EmbeddedWorkspace)
    b = ws.par.block; s = (length(ws.axis), b.ny, b.nx)
    if height_mode(ws.mode)
        return (rk4 = [RK4Tape() for _ in 1:b.ny, _ in 1:b.nx],)
    end
    return (tapes = [PoissonTau.HeunTape() for _ in 1:b.ny, _ in 1:b.nx], z_col = zeros(s), z_al = zeros(s),
            u_start = zeros(s), z_start = zeros(s), z_column = zeros(s))
end

"""The record of a finished solve: inputs, a snapshot of the result and the stage tapes."""
function finish_record!(ws::EmbeddedWorkspace, rec, lT, bx, by, bz)
    return (ne_given = false, valid = ws.res.status.valid, mode = ws.mode, lnT = copy(lT), Bx = copy(bx), By = copy(by),
            Bz = copy(bz), res = snapshot(ws.res), rec = rec)
end

# ---- the adjoint -----------------------------------------------------------------------------------------------------

"""
    adjoint_embedded!(ws, bar) -> EquilibriumAdjoint

Collective. The cotangents of ln T and B of the last solve on `ws`, which must have been `solve_embedded!(...; record =
true)` and valid, for the output cotangents `bar` (an `EquilibriumCotangent` on this rank's slab). The workspace and its
record are not changed, so several cotangents can be pulled back through one solve.
"""
function adjoint_embedded!(ws::EmbeddedWorkspace, bar::EquilibriumCotangent; kw...)
    t0 = time()
    shp = (length(ws.axis), ws.par.block.ny, ws.par.block.nx)
    agree(ws.par.comm, "adjoint_embedded!($(ws.mode))") do
        ws.tape === nothing && invalid("no recorded solve on this workspace: call solve_embedded!(...; record = true) first " *
                                       "(any later solve ends the record)")
        ws.tape.ne_given && invalid("the last solve had a given electron density (lnne): it is forward only and has no adjoint")
        ws.tape.valid || invalid("the recorded solve is not valid ($(ws.tape.res.status.reason)); there is nothing to differentiate")
        for nm in COTANGENT_FIELDS
            a = getfield(bar, nm)
            size(a) == shp || invalid("cotangent $nm has size $(size(a)); this rank's slab is $shp (vertical, y, x)")
            all(isfinite, a) || invalid("cotangent $nm has non-finite values")
        end
        nothing
    end
    X̄ = zeros(shp); B̄ = (zeros(shp), zeros(shp), zeros(shp))
    info = Dict{String,Any}()
    status = if ws.mode in (:column_hse_ltau, :column_hse_logm)
        column_tau_adjoint!(X̄, B̄, ws, bar); :exact
    elseif ws.mode == :column_hse_z
        column_z_adjoint!(X̄, ws, bar); :exact
    else
        poisson_adjoint!(X̄, B̄, ws, bar, info; kw...)
    end
    info["seconds"] = time() - t0
    return EquilibriumAdjoint(X̄, B̄..., status, info)
end

"""Pointwise pull-back of the state outputs (P, ln P, ρ, nₑ, χ₅₀₀) at (X, u) into (X̄, ū); returns χ's (X, u) partials."""
function state_vjp!(X̄, ū, bar, lT, lnP, eos; chibar = nothing)
    Threads.@threads for idx in eachindex(lT)
        @inbounds begin
            r, n, c, rX, rY, nX, nY, cX, cY, _ = state_d(eos, lT[idx], lnP[idx])
            ρb = bar.rho[idx] * exp(r); nb = bar.ne[idx] * exp(n)
            χb = bar.chi500[idx] * exp(c) + (chibar === nothing ? 0.0 : chibar[idx] * exp(c))
            ū[idx] += bar.Pgas[idx] * exp(lnP[idx]) + bar.lnP[idx] + ρb * rY + nb * nY + χb * cY
            X̄[idx] += ρb * rX + nb * nX + χb * cX
        end
    end
end

"""Column HSE on the τ or mass axis: outputs → (ln P, z) → alignment → columns (design/adjoint/02)."""
function column_tau_adjoint!(X̄, B̄, ws::EmbeddedWorkspace, bar)
    t = ws.tape; res = t.res; lT = t.lnT
    ū = zeros(size(lT))
    state_vjp!(X̄, ū, bar, lT, res.lnP, ws.eos)
    l̄ = mass_mode(ws.mode) ? bar.logtau500 .+ bar.tau500 .* res.tau500 .* log(10.0) : nothing
    return column_tau_core!(X̄, B̄, ws, ū, copy(bar.z), res.lnP; l̄ = l̄)
end

"""
The column HSE of the recorded solve pulled back from the cotangents of its ln P (ū) and of its gauged heights (z̄), and on
the mass axis of the log10 τ the gauge used (l̄): gauge → alignment → Heun columns. `lnP` is the column's ln P (the start
of a Poisson solve). ū and z̄ are consumed.
"""
function column_tau_core!(X̄, B̄, ws::EmbeddedWorkspace, ū, z̄, lnP; l̄ = nothing)
    t = ws.tape; rec = t.rec; lT = t.lnT; par = ws.par; mass = mass_mode(ws.mode)
    nk, ny, nx = size(lT)
    if mass
        l̄ === nothing && (l̄ = zeros(size(lT)))
        PoissonTau.gauge_vjp!(par, z̄, rec.z_al, ws.axis, PoissonTau.logtau_mass(lT, lnP, ws.axis, ws.eos), l̄)
        PoissonTau.logtau_mass_vjp!(X̄, ū, l̄, lT, lnP, ws.axis, ws.eos)
    else
        PoissonTau.gauge_vjp!(par, z̄, rec.z_al, ws.axis)
    end
    ws.column.align == "ptot" && PoissonTau.align_ptot_vjp!(par, z̄, ū, X̄, B̄, rec.z_col, lnP, lT, t.Bx, t.By, t.Bz,
                                                            ws.eos, ws.g)
    mu = ws.view.mu
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            PoissonTau.hse_tau_vjp!(@view(X̄[:, j, i]), @view(ū[:, j, i]), @view(z̄[:, j, i]), @view(lT[:, j, i]), ws.axis,
                                    ws.eos, ws.g, rec.tapes[j, i]; mu = mu, mass = mass)
        end
    end
    return X̄
end

"""The τ-grid problem of the recorded Poisson solve, at its start (weights formed there) and its cut level."""
function tau_problem(ws::EmbeddedWorkspace)
    t = ws.tape; rec = t.rec; o = ws.poisson
    pre = o.precond == :switch ? :block : o.precond
    pr, c0 = PoissonTau.tp_setup(ws.par, ws.eos, ws.g, ws.axis, copy(t.lnT), copy(t.Bx), copy(t.By), copy(t.Bz), rec.z_start,
                                 ws.dx, ws.dy; logtau_cut = los_cut(o, ws.view.mu), S = ws.fft, mu = ws.view.mu,
                                 slant = slant(ws.view), precond = pre, klow = o.klow, mass = mass_mode(ws.mode))
    PoissonTau.evaluate!(pr, rec.u_start, c0); PoissonTau.tp_weights!(pr, ws.g, ws.dx)
    return pr, c0
end

"""mhs-poisson-ltau, -logm: outputs → gauge → heights and the stationary region → the fixed part and the weights → the
column-HSE start (design/adjoint/03)."""
function poisson_tau_adjoint!(X̄, B̄, ws, bar, info; cg_tol, cg_maxit, gauss_newton, stationary_tol)
    t = ws.tape; rec = t.rec; res = t.res; lT = t.lnT; par = ws.par; mass = mass_mode(ws.mode)
    st = res.status; kc = st.stats["cut_level"]; nk, ny, nx = size(lT)
    ū = zeros(size(lT))
    state_vjp!(X̄, ū, bar, lT, res.lnP, ws.eos)
    z̄ = copy(bar.z)
    shift = get(st.stats, "gauge_shift_m", 0.0)
    zpre = res.z .+ shift                                          # the heights before the gauge
    if mass
        l̄ = bar.logtau500 .+ bar.tau500 .* res.tau500 .* log(10.0)
        PoissonTau.gauge_vjp!(par, z̄, zpre, ws.axis, res.logtau500, l̄)
        PoissonTau.logtau_mass_vjp!(X̄, ū, l̄, lT, res.lnP, ws.axis, ws.eos)
    else
        PoissonTau.gauge_vjp!(par, z̄, zpre, ws.axis)
    end
    pr, c0 = tau_problem(ws)
    c = zpre[kc, :, :]
    PoissonTau.evaluate!(pr, res.lnP, c)
    cb, zrelb, ub, Xh = PoissonTau.heights_vjp(pr, res.lnP, z̄)           # the heights of the solution
    X̄ .+= Xh; ū .+= ub
    ū0 = zeros(size(lT)); z̄0 = zeros(size(lT))
    xb = PoissonTau.to_slots(pr, cb, ū)
    if st.steps == 0 && get(st.stats, "polish_steps", 0) == 0          # start accepted: the solution is the start
        ū0 .= ū; z̄0[kc, :, :] .+= cb
        @views ū0[kc+1:nk, :, :] .= ū[kc+1:nk, :, :]
    else
        Xb, u0b, zrelb2, Bb, cg = PoissonTau.tau_stationary_vjp(pr, res.lnP, c, xb, rec.u_start, c0; cg_tol = cg_tol,
                                                                cg_maxit = cg_maxit, gauss_newton = gauss_newton)
        X̄ .+= Xb; ū0 .+= u0b; zrelb .+= zrelb2
        for k in 1:3
            B̄[k] .+= Bb[k]
        end
        @views ū0[1:kc, :, :] .+= ū[1:kc, :, :]                        # ln P at and above the cut: the start's
        info["cg_iterations"] = cg.its; info["cg_rel"] = cg.rel; info["cg_flag"] = String(cg.flag)
        info["newton_decrement"] = cg.newton_decrement; info["null_component"] = cg.null_component
        z̄0[kc, :, :] .+= cg.cnull                                     # the null modes of c are the start's
    end
    # zrel_k = z0_k − z0_kc above the cut (the start's heights)
    @views z̄0[1:kc-1, :, :] .+= zrelb[1:kc-1, :, :]
    @views z̄0[kc, :, :] .-= dropdims(sum(zrelb[1:kc-1, :, :]; dims = 1); dims = 1)
    # the start: the column HSE of this solve (a warm start adds the change of that column to an earlier solution, so
    # its derivative is the column's)
    column_tau_core!(X̄, B̄, ws, ū0, z̄0, log.(res.Pgas_start))
    dec = get(info, "newton_decrement", 0.0)
    return (isfinite(dec) && dec <= stationary_tol) ? :stationary : :not_stationary
end
"""
τ₅₀₀ of the height grid from χ (vertical columns or inclined rays): its adjoint for the cotangent τ̄ of `res.tau500` adds
χ̄ (design/adjoint/01).
"""
function height_tau_vjp!(χ̄, τ̄, ws::EmbeddedWorkspace, chi)
    o = ws.column; z = ws.axis; nk, ny, nx = size(chi)
    t = slant(ws.view)
    if t == (0.0, 0.0)
        Threads.@threads for i in 1:nx
            tc = zeros(nk)
            @inbounds for j in 1:ny
                m = tau_column!(tc, @view(chi[:, j, i]), z; top = o.tau_top, H_max = o.H_max, mu = o.mu)
                tau_column_vjp!(@view(χ̄[:, j, i]), @view(τ̄[:, j, i]), @view(chi[:, j, i]), z, m; mu = o.mu)
            end
        end
    else
        ttm = zeros(UInt8, ny, nx)
        inclined_tau!(similar(chi), zeros(ny, nx), ttm, ws.par, chi, z, ws.dx, ws.dy, ws.view.mu, t; top = o.tau_top, H_max = o.H_max)
        inclined_tau_vjp!(χ̄, τ̄, ws.par, chi, z, ws.dx, ws.dy, ws.view.mu, t, ttm)
    end
    return χ̄
end

"""Column HSE on the height grid: outputs (τ through χ) → ln P → columns."""
function column_z_adjoint!(X̄, ws::EmbeddedWorkspace, bar)
    t = ws.tape; rec = t.rec; res = t.res; lT = t.lnT
    nk, ny, nx = size(lT)
    τ̄ = bar.tau500 .+ ifelse.(bar.logtau500 .== 0, 0.0, bar.logtau500 ./ (res.tau500 .* log(10.0)))   # τ₁ = 0: top = :zero
    χ̄ = height_tau_vjp!(zeros(size(lT)), τ̄, ws, res.chi500)
    ū = zeros(size(lT))
    state_vjp!(X̄, ū, bar, lT, res.lnP, ws.eos; chibar = χ̄)     # the state at ln P (ln P_ref at the anchor node)
    o = hse_opts(ws.column, ws.g)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            hse_column_vjp!(@view(X̄[:, j, i]), @view(ū[:, j, i]), @view(lT[:, j, i]), ws.axis, ws.eos, o, rec.rk4[j, i];
                            u = @view(res.lnP[:, j, i]))      # the recorded ln P (the replay's, bit for bit)
        end
    end
    return X̄
end

"""
Poisson modes: the column-HSE start, the implicit-function adjoint of the solved region and the fixed part (design/adjoint/03).
Keywords: `cg_tol` (1e-8), `cg_maxit` (1000) of the adjoint solve, `gauss_newton` (false), `stationary_tol` (1e-6: the
largest undamped Newton decrement ½ GᵀΔ/Φ of the returned state reported as :stationary).
"""
function poisson_adjoint!(X̄, B̄, ws, bar, info; cg_tol = 1e-8, cg_maxit = 1000, gauss_newton = false, stationary_tol = 1e-6)
    f = ws.mode == :mhs_poisson_z ? poisson_z_adjoint! : poisson_tau_adjoint!
    return f(X̄, B̄, ws, bar, info; cg_tol = cg_tol, cg_maxit = cg_maxit, gauss_newton = gauss_newton,
             stationary_tol = stationary_tol)
end

"""mhs-poisson-z: outputs → continuation above z_cut → the stationary region (bordered by the anchor) → B through J×B and
the start → the column-HSE adjoint of the start."""
function poisson_z_adjoint!(X̄, B̄, ws, bar, info; cg_tol, cg_maxit, gauss_newton, stationary_tol)
    t = ws.tape; rec = t.rec; res = t.res; lT = t.lnT; o = ws.poisson; z = ws.axis; par = ws.par
    nk, ny, nx = size(lT)
    st = res.status
    τ̄ = bar.tau500 .+ ifelse.(bar.logtau500 .== 0, 0.0, bar.logtau500 ./ (res.tau500 .* log(10.0)))
    χ̄ = height_tau_vjp!(zeros(size(lT)), τ̄, ws, res.chi500)
    ū = zeros(size(lT))
    state_vjp!(X̄, ū, bar, lT, res.lnP, ws.eos; chibar = χ̄)
    ū0 = zeros(size(lT))                                              # cotangent of the column-HSE start ln P
    kc = st.stats["cut_level"]; R = kc:nk
    PoissonZ.continuation_vjp!(X̄, ū, lT, res.lnP, z, ws.eos, ws.g, kc)   # above z_cut: integrated from u at kc
    if st.steps == 0                                                  # start accepted as it is below the cut
        ū0[R, :, :] .+= ū[R, :, :]
    else
        Lx, Ly, Lz = PoissonZ.lorentz(par, t.Bx, t.By, t.Bz, z, ws.dx, ws.dy)
        pr, u0R, _, _, _ = PoissonZ.zprob(par, ws.eos, ws.g, lT, Lx, Ly, Lz, res.Pgas_start, z, ws.dx, ws.dy;
                                         zcut = o.z_cut_m, anchor_w = o.anchor_w, S = ws.fft)
        Xb, Lxb, Lyb, Lzb, u0b, cg = PoissonZ.stationary_vjp(pr, res.lnP[R, :, :], u0R, ū[R, :, :]; cg_tol = cg_tol,
                                                            cg_maxit = cg_maxit, gauss_newton = gauss_newton)
        X̄[R, :, :] .+= Xb
        ū0[R, :, :] .+= u0b
        L̄ = (zeros(size(lT)), zeros(size(lT)), zeros(size(lT)))
        L̄[1][R, :, :] .= Lxb; L̄[2][R, :, :] .= Lyb; L̄[3][R, :, :] .= Lzb
        PoissonZ.lorentz_vjp!(B̄, par, L̄..., t.Bx, t.By, t.Bz, z, ws.dx, ws.dy)
        info["cg_iterations"] = cg.its; info["cg_rel"] = cg.rel; info["cg_flag"] = String(cg.flag)
        info["anchor_multiplier"] = cg.kappa; info["newton_decrement"] = cg.newton_decrement
    end
    oo = hse_opts(ws.column, ws.g)
    lnP0 = log.(res.Pgas_start)                                       # the column-HSE start (to rounding)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            hse_column_vjp!(@view(X̄[:, j, i]), @view(ū0[:, j, i]), @view(lT[:, j, i]), z, ws.eos, oo, rec.rk4[j, i];
                            u = @view(lnP0[:, j, i]))
        end
    end
    dec = get(info, "newton_decrement", 0.0)
    return st.steps == 0 || (isfinite(dec) && dec <= stationary_tol) ? :stationary : :not_stationary
end
