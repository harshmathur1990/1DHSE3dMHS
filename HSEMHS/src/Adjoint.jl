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
    return (tapes = [PoissonTau.HeunTape() for _ in 1:b.ny, _ in 1:b.nx], z_col = zeros(s), z_al = zeros(s))
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
    t = ws.tape; rec = t.rec; res = t.res; lT = t.lnT; par = ws.par; mass = mass_mode(ws.mode)
    nk, ny, nx = size(lT)
    ū = zeros(size(lT))
    state_vjp!(X̄, ū, bar, lT, res.lnP, ws.eos)
    z̄ = copy(bar.z)
    if mass
        l̄ = bar.logtau500 .+ bar.tau500 .* res.tau500 .* log(10.0)
        PoissonTau.gauge_vjp!(par, z̄, rec.z_al, ws.axis, res.logtau500, l̄)
        PoissonTau.logtau_mass_vjp!(X̄, ū, l̄, lT, res.lnP, ws.axis, ws.eos)
    else
        PoissonTau.gauge_vjp!(par, z̄, rec.z_al, ws.axis)
    end
    ws.column.align == "ptot" && PoissonTau.align_ptot_vjp!(par, z̄, ū, X̄, B̄, rec.z_col, res.lnP, lT, t.Bx, t.By, t.Bz,
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
            hse_column_vjp!(@view(X̄[:, j, i]), @view(ū[:, j, i]), @view(lT[:, j, i]), ws.axis, ws.eos, o, rec.rk4[j, i])
        end
    end
    return X̄
end

poisson_adjoint!(X̄, B̄, ws, bar, info; kw...) = invalid("the adjoint of $(ws.mode) is not implemented yet")
