# Shared helpers for the equilibrium-adjoint tests (test_adjoint.jl, mpi/mpi_adjoint.jl); design/adjoint/04-validation.md.
#
# The check: for random output cotangents ȳ (every output, scaled relative to its value) and a random input direction
# δa = (δ ln T, δB), compare ⟨ȳ, [F(a + hδa) − F(a − hδa)]/2h⟩ (two solves) with ⟨adjoint(ȳ), δa⟩ (one recorded solve
# and one adjoint), over several h; the error is the smallest relative difference over h.

@isdefined(ETABLE) || include("embedded_helpers.jl")
using Random

const ADJ_OUTPUTS = (:Pgas, :rho, :ne, :chi500, :tau500, :logtau500, :z)
"""Outputs given a cotangent (all of ADJ_OUTPUTS unless a diagnostic sets `ADJ_OUT1` in Main)."""
adj_outputs() = isdefined(Main, :ADJ_OUT1) ? Main.ADJ_OUT1 : ADJ_OUTPUTS

"""Smooth global field (nk, NY, NX): a few low modes in every direction with random phases, plus one localised bump."""
function smooth_field(rng, nk, NY, NX; nmodes = 3)
    f = zeros(nk, NY, NX)
    for _ in 1:nmodes
        a, b, c = rand(rng, 0:2), rand(rng, 0:2), rand(rng, 1:3)
        p1, p2, p3 = 2π .* rand(rng, 3)
        A = randn(rng)
        for i in 1:NX, j in 1:NY, k in 1:nk
            f[k, j, i] += A * cos(2π * a * i / NX + p1) * cos(2π * b * j / NY + p2) * cos(π * c * k / nk + p3)
        end
    end
    k0, j0, i0 = rand(rng, 1:nk), rand(rng, 1:NY), rand(rng, 1:NX)
    for i in 1:NX, j in 1:NY, k in 1:nk
        d2 = (k - k0)^2 / 9 + (min(abs(j - j0), NY - abs(j - j0)))^2 / 2 + (min(abs(i - i0), NX - abs(i - i0)))^2 / 2
        f[k, j, i] += 2 * exp(-d2)
    end
    return f
end

"""Global inputs of a fixture: (axis, x, y, lnT, Bx, By, Bz, extra) for the height (`:z`) or τ fixtures."""
function adjoint_inputs(grid::Symbol, tag = "11x12")
    if grid == :z
        f = load_z_fixture(tag, Colon())
        return (axis = f.z, x = f.x, y = f.y, g = f.g, P_ref = f.P_ref, lnT = f.lnT, Bx = f.Bx, By = f.By, Bz = f.Bz)
    end
    f = load_tau_fixture(tag, Colon())
    ax = grid == :logm ? f.lt .+ 1.0 : f.lt
    return (axis = ax, x = f.x, y = f.y, g = 274.0, P_ref = NaN, lnT = f.lnT, Bx = f.Bx, By = f.By, Bz = f.Bz)
end

"""
Anchor pressure for a height-grid anchor: the fixture's P_ref is a bottom pressure; for :top and :height the pressure that
the bottom-anchored column (1, 1) has at the top or at the middle height (z_ref), so every column stays in the table.
"""
function anchor_pressure(lnT, z, P_ref, g, tab, anchor)
    anchor == :bottom && return P_ref
    n = length(z); u = fill(NaN, n); m = fill(NaN, n)
    H.hse_column!(u, m, lnT[:, 1, 1], z, tab, H.HSEOpts(anchor = :bottom, P_ref = P_ref, g = g))
    anchor == :top && return exp(u[1])
    zr = (z[1] + z[end]) / 2; k = findlast(>=(zr), z)
    return exp(u[k] + (u[k+1] - u[k]) * (zr - z[k]) / (z[k+1] - z[k]))
end

"""
ln T moved 5e-6 off the table's slope breaks (cell pairs 1e-9 apart at partition-function nodes) where it lies within 2e-6
of one: there the derivative is one-sided and any finite difference averages the two sides. (The height fixture has a
node at T = 4365.2 K exactly, a break.)
"""
function off_breaks(lnT, tab)
    br = [tab.X[i] for i in 1:length(tab.X)-1 if tab.X[i+1] - tab.X[i] < 1e-6]
    return map(x -> any(b -> abs(x - b) < 2e-6, br) ? x + 5e-6 : x, lnT)
end

"""Workspace for a case on `comm` (a NamedTuple from `adjoint_cases`)."""
function adjoint_workspace(comm, case, inp, tab)
    m = case.mode
    view = case.view
    if H.height_mode(m)
        zr = (inp.axis[1] + inp.axis[end]) / 2
        col = H.ColumnZOptions(anchor = case.anchor, P_ref = anchor_pressure(inp.lnT, inp.axis, inp.P_ref, inp.g, tab, case.anchor),
                               z_ref = zr, adapt = case.adapt, tau_top = case.top)
        poi = m == :mhs_poisson_z ? H.PoissonZOptions(; merge((z_cut_m = 1.0e6,), case.poisson)...) : nothing
    else
        col = H.ColumnTauOptions(align = case.align)
        poi = m == :mhs_poisson_ltau ? H.PoissonTauOptions(; logtau_cut = -4.0, case.poisson...) :
              m == :mhs_poisson_logm ? H.PoissonMassOptions(; logm_cut = -1.5, case.poisson...) : nothing
    end
    return H.prepare_embedded(comm, m; axis = inp.axis, x = inp.x, y = inp.y, eos = tab, g = inp.g, column = col,
                              poisson = poi, view = view)
end

"""Global sum over the workspace's communicator."""
gsum_ws(ws, v) = MPI.Allreduce(v, +, ws.par.comm)

"""
    adjoint_dot_error(comm, case, inp, tab; seed, hs) -> (err, lhs, rhs, adj_status)

The dot-product check of the module header for one case (collective). `inp` holds global inputs; every rank takes its
owned x slab. Random fields are drawn globally from `seed`, so every rank count sees the same problem.
"""
function adjoint_dot_error(comm, case, inp, tab; seed = 20261124, hs = (1e-4, 1e-5, 1e-6))
    ws = adjoint_workspace(comm, case, inp, tab)
    ir = H.owned_columns(ws)
    nk = length(inp.axis); NY = length(inp.y); NX = length(inp.x)
    rng = MersenneTwister(seed)
    dT = 0.05 .* smooth_field(rng, nk, NY, NX)[:, :, ir]
    Bs = max(maximum(abs, inp.Bx), maximum(abs, inp.By), maximum(abs, inp.Bz), 1e-3)
    dB = case.B ? ntuple(_ -> 0.05 * Bs .* smooth_field(rng, nk, NY, NX)[:, :, ir], 3) : ntuple(_ -> zeros(nk, NY, length(ir)), 3)
    lnT = off_breaks(inp.lnT[:, :, ir], tab)
    B0 = case.B ? (inp.Bx[:, :, ir], inp.By[:, :, ir], inp.Bz[:, :, ir]) : ntuple(_ -> zeros(nk, NY, length(ir)), 3)

    res = H.snapshot(H.solve_embedded!(ws, lnT, B0...; record = true))
    res.status.valid || error("$(case.name): the centre solve is invalid ($(res.status.reason))")
    bar = H.EquilibriumCotangent(ws)
    for nm in ADJ_OUTPUTS
        nm in adj_outputs() || continue
        v = getfield(res, nm)
        w = randn(rng, nk, NY, NX)[:, :, ir]
        scale = nm == :z ? fill(1e5, size(v)) : nm == :logtau500 ? ones(size(v)) : ifelse.(v .== 0, 1.0, abs.(v))
        getfield(bar, nm) .= ifelse.(isfinite.(v), w ./ scale, 0.0)          # log τ₁ = −∞ with top = :zero
    end
    adj = H.adjoint_embedded!(ws, bar)
    rhs = gsum_ws(ws, sum(adj.lnT .* dT) + sum(adj.Bx .* dB[1]) + sum(adj.By .* dB[2]) + sum(adj.Bz .* dB[3]))

    function D(h)                                    # central difference of the pairing, NaN if a solve is invalid
        rp = H.snapshot(H.solve_embedded!(ws, lnT .+ h .* dT, (B0 .+ h .* dB)...))
        rm = H.snapshot(H.solve_embedded!(ws, lnT .- h .* dT, (B0 .- h .* dB)...))
        (rp.status.valid && rm.status.valid) || return NaN
        loc = 0.0
        for nm in ADJ_OUTPUTS
            b = getfield(bar, nm); d = getfield(rp, nm) .- getfield(rm, nm)
            loc += sum(ifelse.(b .== 0, 0.0, b .* d)) / (2h)
        end
        return gsum_ws(ws, loc)
    end
    best = (Inf, NaN)
    for h in hs
        d1 = D(h); d2 = D(h / 2)
        for lhs in (d1, d2, (4d2 - d1) / 3)       # and the Richardson extrapolation (fourth order)
            isfinite(lhs) || continue
            e = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
            e < best[1] && (best = (e, lhs))
        end
    end
    H.free_embedded!(ws)
    return (err = best[1], lhs = best[2], rhs = rhs, status = adj.status)
end

const PPV = H.ViewOptions()
const PP8 = H.ViewOptions(mu = 0.8)
const INC8 = H.ViewOptions(mu = 0.8, azimuth_deg = 30.0, geometry = :inclined)
const TIGHT_TAU = (max_steps = 80, stop = :gn, gn_tol = 1e-9, gn_consec = 1, cg_tol = 1e-10, cg_maxit = 400, qual_maxit = 400)
const TIGHT_Z = (max_steps = 30, polish = 10)        # the Gauss–Newton solve stalls short of stationarity; the polish reaches it

"""The case matrix of design/adjoint/04-validation.md. `tol` is the target of each case."""
function adjoint_cases(; poisson = true)
    c(name, mode, grid; view = PPV, align = "ptot", anchor = :bottom, adapt = true, top = :exp_tail, B = true, tol, poisson = (;)) =
        (name = name, mode = mode, grid = grid, view = view, align = align, anchor = anchor, adapt = adapt, top = top, B = B,
         tol = tol, poisson = poisson)
    cases = Any[
        c("column-hse-ltau", :column_hse_ltau, :ltau; tol = 1e-6),
        c("column-hse-ltau align none, B = 0", :column_hse_ltau, :ltau; align = "none", B = false, tol = 1e-6),
        c("column-hse-ltau μ 0.8", :column_hse_ltau, :ltau; view = PP8, tol = 1e-6),
        c("column-hse-ltau μ 0.8 inclined", :column_hse_ltau, :ltau; view = INC8, tol = 1e-6),
        c("column-hse-logm", :column_hse_logm, :logm; tol = 1e-6),
        c("column-hse-logm align none, B = 0", :column_hse_logm, :logm; align = "none", B = false, tol = 1e-6),
        c("column-hse-logm μ 0.8 inclined", :column_hse_logm, :logm; view = INC8, tol = 1e-6),
        c("column-hse-z anchor bottom", :column_hse_z, :z; B = false, tol = 1e-7),
        c("column-hse-z anchor top", :column_hse_z, :z; anchor = :top, B = false, tol = 1e-7),
        c("column-hse-z anchor height", :column_hse_z, :z; anchor = :height, B = false, tol = 1e-7),
        c("column-hse-z adapt off, half_cell", :column_hse_z, :z; adapt = false, top = :half_cell, B = false, tol = 1e-7),
        c("column-hse-z τ top zero", :column_hse_z, :z; top = :zero, B = false, tol = 1e-7),
        c("column-hse-z μ 0.8 inclined", :column_hse_z, :z; view = INC8, B = false, tol = 1e-7),
    ]
    poisson || return cases
    append!(cases, [
        c("mhs-poisson-z", :mhs_poisson_z, :z; tol = 1e-5, poisson = TIGHT_Z),
        c("mhs-poisson-ltau", :mhs_poisson_ltau, :ltau; tol = 1e-5, poisson = TIGHT_TAU),
        c("mhs-poisson-ltau μ 0.8 inclined", :mhs_poisson_ltau, :ltau; view = INC8, tol = 1e-5, poisson = TIGHT_TAU),
        c("mhs-poisson-ltau block preconditioner", :mhs_poisson_ltau, :ltau; tol = 1e-5, poisson = merge(TIGHT_TAU, (precond = :block,))),
        c("mhs-poisson-logm", :mhs_poisson_logm, :logm; tol = 1e-5, poisson = TIGHT_TAU),
    ])
    return cases
end
