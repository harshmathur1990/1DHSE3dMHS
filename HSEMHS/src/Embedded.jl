# In-memory API for the four fixed-B modes (column-hse-z, column-hse-ltau, mhs-poisson-z, mhs-poisson-ltau), for callers
# that embed HSEMHS in their own MPI program (FFNOInversion). The caller owns MPI and threading: nothing here initialises
# or finalises MPI, touches COMM_WORLD, sets BLAS/FFTW threads, reads or writes files (except loading an EOS table when
# given a path) or prints (logging goes to a caller-supplied `logf`).
#
#     ws  = prepare_embedded(comm, mode; axis, x, y, eos, g, column, poisson)   # collective, once
#     res = solve_embedded!(ws, lnT, Bx, By, Bz)                                # collective, every evaluation
#     free_embedded!(ws)                                                        # collective, releases communicators
#
# Arrays are rank-local x slabs in (vertical, y, x) order, vertical index 1 = top: z decreasing [m] or log10 τ500
# increasing. Ranks own contiguous x ranges given by `block_range` (rank r of `comm` owns slab r) and all of y.
#
# Collective safety: every prepare and solve first runs its local checks (shapes, finiteness, options, files) inside
# `try`, agrees on the outcome over an already valid communicator, and only then enters collective construction or a
# solver; a failure on any rank becomes the same InvalidInput on every rank. Physics failures inside a solve (EOS
# envelope, non-finite fields, folded heights, failed continuation) are returned in the status, identical on all ranks.
# This does not cover process death or failures inside MPI itself.

using .Slab
import .PoissonZ, .PoissonTau

const EMBEDDED_MODES = (:column_hse_z, :column_hse_ltau, :mhs_poisson_z, :mhs_poisson_ltau)

# ---------------------------------------------------------------------------------------------------------------------
# options (names, defaults and meaning as in the standalone [column_hse], [tau], [tau_grid] and [poisson] config sections)
# ---------------------------------------------------------------------------------------------------------------------

"""Column HSE on the height grid (`[column_hse]` and `[tau]`). `P_ref` is the resolved anchor pressure [Pa]."""
Base.@kwdef struct ColumnZOptions
    anchor::Symbol = :bottom          # :bottom | :top | :height
    P_ref::Float64 = NaN              # required
    z_ref::Float64 = NaN              # anchor height [m] for anchor = :height
    n_sub::Int = 4
    adapt::Bool = true
    rk_tol::Float64 = 1e-6
    n_sub_max::Int = 64
    tau_top::Symbol = :exp_tail       # :exp_tail | :half_cell | :zero
    H_max::Float64 = 5.0e6
    mu::Float64 = 1.0
end

"""Column HSE on the τ grid (`[tau_grid]`): no pressure anchor, the top closure P = g τ ρ/χ sets the pressure."""
Base.@kwdef struct ColumnTauOptions
    cap_per_dex::Float64 = 50.0
    align::String = "ptot"            # "ptot" | "none"
end

"""Poisson step on the height grid (`[poisson]`). `rms_atol` > 0 accepts an already balanced start in zero steps."""
Base.@kwdef struct PoissonZOptions
    z_cut_m::Float64 = 1.0e6
    max_steps::Int = 100
    rtol::Float64 = 1e-4
    n_consec::Int = 3
    anchor_w::Float64 = 1.0
    cg_tol::Float64 = 1e-3
    cg_maxit::Int = 80
    rms_atol::Float64 = 0.0
end

"""Gauss–Newton solve on the τ grid (`[poisson]`)."""
Base.@kwdef struct PoissonTauOptions
    logtau_cut::Float64 = -4.0
    max_steps::Int = 100
    rtol::Float64 = 1e-4
    n_consec::Int = 3
    cg_tol::Float64 = 1e-3
    cg_maxit::Int = 80
    dc_cap_m::Float64 = 3.0e5
    du_cap::Float64 = 2.0
    lm_mu0::Float64 = 0.0
    rms_atol::Float64 = 0.0
end

hse_opts(o::ColumnZOptions, g) = HSEOpts(o.anchor, o.P_ref, o.z_ref, g, o.n_sub, o.adapt, o.rk_tol, o.n_sub_max,
                                         o.tau_top, o.H_max, o.mu)

"""Smallest number of solved levels at or below the cut (Poisson z: `h[1]`, `h[end]` and the anchor face need ≥ 2)."""
const POISSON_Z_MIN_LEVELS = 3
"""Smallest number of solved τ levels (k_c … nt); k_c ≥ 2 because the height stencil at k_c uses level k_c − 1."""
const POISSON_TAU_MIN_LEVELS = 3

push_if!(errs, cond, msg) = (cond || push!(errs, msg); errs)

"""Problems with column-z options on the height axis z (top first); empty when valid. Pure: same on every rank."""
function option_errors(o::ColumnZOptions, z::AbstractVector)
    e = String[]
    push_if!(e, o.anchor in (:bottom, :top, :height), "column anchor must be :bottom, :top or :height, got :$(o.anchor)")
    push_if!(e, isfinite(o.P_ref) && o.P_ref > 0, "column P_ref must be a finite positive pressure [Pa], got $(o.P_ref)")
    o.anchor == :height && push_if!(e, isfinite(o.z_ref) && z[end] <= o.z_ref <= z[1],
                                    "column z_ref = $(o.z_ref) m is outside the height axis [$(z[end]), $(z[1])] m")
    push_if!(e, o.n_sub >= 1, "column n_sub must be ≥ 1")
    push_if!(e, o.n_sub_max >= o.n_sub, "column n_sub_max must be ≥ n_sub")
    push_if!(e, o.rk_tol > 0, "column rk_tol must be positive")
    push_if!(e, o.tau_top in (:exp_tail, :half_cell, :zero), "tau_top must be :exp_tail, :half_cell or :zero")
    push_if!(e, o.H_max > 0 && o.mu > 0, "tau H_max and mu must be positive")
    return e
end

option_errors(o::ColumnTauOptions, lt::AbstractVector) =
    push_if!(push_if!(String[], o.cap_per_dex > 0, "cap_per_dex must be positive"),
             o.align in ("ptot", "none"), "align must be \"ptot\" or \"none\", got \"$(o.align)\"")

function solver_errors(e, o)
    push_if!(e, o.max_steps >= 1, "poisson max_steps must be ≥ 1")
    push_if!(e, o.rtol > 0, "poisson rtol must be positive")
    push_if!(e, o.n_consec >= 1, "poisson n_consec must be ≥ 1")
    push_if!(e, 0 < o.cg_tol < 1, "poisson cg_tol must be in (0, 1)")
    push_if!(e, o.cg_maxit >= 1, "poisson cg_maxit must be ≥ 1")
    push_if!(e, o.rms_atol >= 0, "poisson rms_atol must be ≥ 0")
end

"""Index of the cut level and the number of solved levels for the height axis (`nothing` if the cut is below it)."""
function cut_levels(o::PoissonZOptions, z)
    kc = findfirst(<=(o.z_cut_m), z)
    return kc, kc === nothing ? 0 : length(z) - kc + 1
end
cut_levels(o::PoissonTauOptions, lt) = (kc = argmin(abs.(lt .- o.logtau_cut)); (kc, length(lt) - kc + 1))

function option_errors(o::PoissonZOptions, z::AbstractVector)
    e = solver_errors(String[], o)
    push_if!(e, o.anchor_w > 0, "poisson anchor_w must be positive")
    kc, n = cut_levels(o, z)
    if kc === nothing
        push!(e, "z_cut_m = $(o.z_cut_m) m lies below the height axis (bottom $(z[end]) m)")
    else
        push_if!(e, n >= POISSON_Z_MIN_LEVELS,
                 "z_cut_m = $(o.z_cut_m) m leaves $n solved level(s) at or below the cut; at least $POISSON_Z_MIN_LEVELS are required")
    end
    return e
end

function option_errors(o::PoissonTauOptions, lt::AbstractVector)
    e = solver_errors(String[], o)
    push_if!(e, o.dc_cap_m > 0 && o.du_cap > 0, "poisson dc_cap_m and du_cap must be positive")
    push_if!(e, o.lm_mu0 >= 0, "poisson lm_mu0 must be ≥ 0")
    push_if!(e, lt[1] <= o.logtau_cut <= lt[end], "logtau_cut = $(o.logtau_cut) is outside the axis [$(lt[1]), $(lt[end])]")
    kc, n = cut_levels(o, lt)
    push_if!(e, kc >= 2, "logtau_cut = $(o.logtau_cut) selects the top level; the cut must leave at least one fixed level above it")
    push_if!(e, n >= POISSON_TAU_MIN_LEVELS,
             "logtau_cut = $(o.logtau_cut) leaves $n solved level(s); at least $POISSON_TAU_MIN_LEVELS are required")
    return e
end

# ---------------------------------------------------------------------------------------------------------------------
# two-stage collective checks
# ---------------------------------------------------------------------------------------------------------------------

"""All ranks' messages (empty strings dropped), gathered on every rank of `comm`."""
function gather_messages(comm::MPI.Comm, msg::AbstractString)
    bytes = Vector{UInt8}(codeunits(String(msg)))
    counts = MPI.Allgather(Int32(length(bytes)), comm)
    all = Vector{UInt8}(undef, sum(counts))
    MPI.Allgatherv!(bytes, MPI.VBuffer(all, counts), comm)
    out = String[]; off = 0
    for c in counts
        c > 0 && push!(out, String(all[off+1:off+c]))
        off += c
    end
    return out
end

"""
    agree(f, comm, stage) -> value of f()

Run the rank-local checks `f()` on every rank, catching any error, then agree over `comm` (one Allreduce, plus an
Allgatherv only on failure). If any rank failed, every rank throws the same InvalidInput listing each failing rank's
message; otherwise every rank returns its own value. `comm` must already be valid on all ranks.
"""
function agree(f, comm::MPI.Comm, stage::AbstractString)
    val = nothing; msg = ""
    try
        val = f()
    catch e
        e isa InterruptException && rethrow()
        msg = "rank $(MPI.Comm_rank(comm)): " * (e isa InvalidInput ? e.msg : sprint(showerror, e))
    end
    nfail = MPI.Allreduce(isempty(msg) ? 0 : 1, +, comm)
    nfail == 0 && return val
    invalid("$stage failed on $nfail rank(s): " * join(gather_messages(comm, msg), "; "))
end

# ---------------------------------------------------------------------------------------------------------------------
# status, result and workspace
# ---------------------------------------------------------------------------------------------------------------------

"""
Outcome of one embedded solve, identical on every rank.

`valid` is true when the returned fields are physically usable: EOS clean, all finite, P, ρ, χ₅₀₀ > 0, heights strictly
decreasing and τ strictly increasing down every column, and (Poisson z) the hydrostatic continuation above the cut
succeeded. `reason` is the first failed check (:eos_out_of_envelope, :continuation_failed, :nonfinite, :nonpositive,
:folded) or else the solver's stopping reason (:converged, :no_descent, :max_steps; column modes: :converged). The
Poisson stopping rule is on relative decreases of Φ, which does not prove stationarity: `gn_decrement` (the relative
decrease of Φ one further undamped Gauss–Newton step would predict) and `grad_norm` = ‖JᵀWR‖ measure it, `cg_its` and
`cg_rel` describe that last linear solve, and `lm_mu` is the final Levenberg–Marquardt damping (τ grid). `phi0`/`phi`
are the absolute objective at the start/end, `rms0`/`rms` = √(Φ/ΣW); `phi0_zero` marks Φ₀ = 0 (ratios undefined).
`counts` holds the global number of nodes or columns per failed check, `stats` the column-stage statistics.
"""
Base.@kwdef mutable struct EquilibriumStatus
    status::SolverStatus = converged
    reason::Symbol = :converged
    valid::Bool = true
    steps::Int = 0
    phi0::Float64 = NaN
    phi::Float64 = NaN
    rms0::Float64 = NaN
    rms::Float64 = NaN
    phi0_zero::Bool = false
    gn_decrement::Float64 = NaN
    grad_norm::Float64 = NaN
    cg_its::Int = 0
    cg_rel::Float64 = NaN
    lm_mu::Float64 = NaN
    eos_violations::Int = 0
    counts::Dict{Symbol,Int} = Dict{Symbol,Int}()
    stats::Dict{String,Any} = Dict{String,Any}()
    messages::Vector{String} = String[]
    seconds::Float64 = 0.0
end

"""
Fields of one solve on this rank's slab, (vertical, y, x), SI: `lnP` = ln(P/Pa), `Pgas` [Pa], `rho` [kg m⁻³], `ne`
[m⁻³], `chi500` [m⁻¹] (extinction per length at 5000 Å, absorption + scattering, from the same EOS table as ρ and nₑ),
`tau500`, `logtau500`, `z` [m] (the native axis broadcast for height modes; the reconstructed, gauged heights for τ
modes), `z_tau1` (y, x) [m], `eos_flags` (EOS bit mask per node), `Pgas_start` (the column-HSE start of the Poisson
modes, else equal to Pgas). The arrays are workspace buffers, overwritten by the next `solve_embedded!` on the same
workspace; keep a result across solves with `snapshot`.
"""
mutable struct EquilibriumResult
    lnP::Array{Float64,3}
    Pgas::Array{Float64,3}
    rho::Array{Float64,3}
    ne::Array{Float64,3}
    chi500::Array{Float64,3}
    tau500::Array{Float64,3}
    logtau500::Array{Float64,3}
    z::Array{Float64,3}
    z_tau1::Matrix{Float64}
    eos_flags::Array{UInt8,3}
    Pgas_start::Array{Float64,3}
    status::EquilibriumStatus
end

function EquilibriumResult(nk::Int, ny::Int, nx::Int)
    A() = fill(NaN, nk, ny, nx)
    return EquilibriumResult(A(), A(), A(), A(), A(), A(), A(), A(), fill(NaN, ny, nx), zeros(UInt8, nk, ny, nx), A(),
                             EquilibriumStatus())
end

"""Owned deep copy of a result (independent of later solves on the workspace)."""
snapshot(r::EquilibriumResult) = deepcopy(r)

mutable struct EmbeddedWorkspace
    mode::Symbol
    par::ParCtx
    axis::Vector{Float64}             # z [m] top first, or log10 τ500 increasing
    x::Vector{Float64}
    y::Vector{Float64}
    dx::Float64
    dy::Float64
    eos::EOSTable
    g::Float64
    column::Union{ColumnZOptions,ColumnTauOptions}
    poisson::Union{Nothing,PoissonZOptions,PoissonTauOptions}
    fft::Union{Nothing,SlabFFT}
    res::EquilibriumResult
    logf::Any
end

height_mode(mode) = mode in (:column_hse_z, :mhs_poisson_z)
poisson_mode(mode) = mode in (:mhs_poisson_z, :mhs_poisson_ltau)

"""Owned global x indices of this workspace's slab."""
owned_columns(ws::EmbeddedWorkspace) = owned_x(ws.par)
"""Global x range rank `r` of `nranks` owns for NX columns (the slab layout `prepare_embedded` uses)."""
slab_range(NX::Int, nranks::Int, r::Int) = (b = block_range(NX, nranks, r); b[1]:b[1]+b[2]-1)

function uniform_spacing(name, v)
    length(v) >= 2 || invalid("$name needs at least 2 points")
    all(isfinite, v) || invalid("$name has non-finite values")
    d = diff(v)
    all(>(0), d) || invalid("$name must be strictly increasing")
    maximum(abs.(d .- d[1])) <= 1e-9 * abs(d[1]) || invalid("$name must be uniformly spaced (to 1e-9)")
    return d[1]
end

"""
    prepare_embedded(comm, mode; axis, x, y, eos, g, column, poisson=nothing, logf=s->nothing) -> EmbeddedWorkspace

Collective over `comm`. `mode` ∈ $(EMBEDDED_MODES). `axis` is the height axis z [m] (top first, decreasing) for the
height modes or the log10 τ500 axis (increasing) for the τ modes; `x`, `y` are the global, uniform, increasing horizontal
axes [m]; `eos` an EOSTable or the path of one (loaded once per rank); `g` [m s⁻²]; `column` a ColumnZOptions (height
modes) or ColumnTauOptions (τ modes); `poisson` a PoissonZOptions / PoissonTauOptions for the Poisson modes. The slab
context duplicates `comm`; release it with `free_embedded!`.
"""
function prepare_embedded(comm::MPI.Comm, mode::Symbol; axis, x, y, eos, g, column, poisson = nothing,
                          logf = s -> nothing)
    # stage 1: local checks and the EOS table, agreed before any collective construction
    local_state = agree(comm, "prepare_embedded($mode)") do
        mode in EMBEDDED_MODES || invalid("mode must be one of $(EMBEDDED_MODES), got :$mode")
        ax = Float64.(collect(axis)); xs = Float64.(collect(x)); ys = Float64.(collect(y))
        length(ax) >= 3 || invalid("vertical axis needs at least 3 levels, got $(length(ax))")
        all(isfinite, ax) || invalid("vertical axis has non-finite values")
        if height_mode(mode)
            all(<(0), diff(ax)) || invalid("height axis must be strictly decreasing (top first) [m]")
            column isa ColumnZOptions || invalid("height modes take column::ColumnZOptions")
        else
            all(>(0), diff(ax)) || invalid("log10 τ500 axis must be strictly increasing (top first)")
            column isa ColumnTauOptions || invalid("τ modes take column::ColumnTauOptions")
        end
        dx = uniform_spacing("x", xs); dy = uniform_spacing("y", ys)
        isfinite(g) && g > 0 || invalid("gravity g must be finite and positive, got $g")
        errs = option_errors(column, ax)
        if poisson_mode(mode)
            want = height_mode(mode) ? PoissonZOptions : PoissonTauOptions
            poisson isa want || invalid("mode :$mode takes poisson::$want")
            append!(errs, option_errors(poisson, ax))
        else
            poisson === nothing || invalid("mode :$mode takes no poisson options")
        end
        isempty(errs) || invalid(join(errs, "; "))
        msg = slab_check(length(xs), MPI.Comm_size(comm))
        msg === nothing || invalid(msg)
        tab = eos isa EOSTable ? eos : eos isa AbstractString ? load_table(eos) :
              invalid("eos must be an EOSTable or a table path")
        (ax, xs, ys, dx, dy, tab)
    end
    ax, xs, ys, dx, dy, tab = local_state
    # stage 2: collective construction (all ranks reach this point together)
    par = slab_par(comm, length(xs), length(ys), length(ax))
    b = par.block
    fft = nothing
    if poisson_mode(mode)
        _, n = cut_levels(poisson, ax)
        fft = SlabFFT(par, n, length(ys))
    end
    res = EquilibriumResult(length(ax), length(ys), b.nx)
    return EmbeddedWorkspace(mode, par, ax, xs, ys, dx, dy, tab, Float64(g), column, poisson, fft, res, logf)
end

"""Release the communicators of the workspace's slab context (collective). The workspace cannot be used afterwards."""
free_embedded!(ws::EmbeddedWorkspace) = (free_parallel!(ws.par); ws.fft = nothing; nothing)

prepare_column_hse_z(comm; kw...) = prepare_embedded(comm, :column_hse_z; kw...)
prepare_column_hse_ltau(comm; kw...) = prepare_embedded(comm, :column_hse_ltau; kw...)
prepare_mhs_poisson_z(comm; kw...) = prepare_embedded(comm, :mhs_poisson_z; kw...)
prepare_mhs_poisson_ltau(comm; kw...) = prepare_embedded(comm, :mhs_poisson_ltau; kw...)

# ---------------------------------------------------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------------------------------------------------

"""
    solve_embedded!(ws, lnT, Bx=nothing, By=nothing, Bz=nothing) -> EquilibriumResult

Collective. `lnT` = ln(T/K) and the cell-centred field [T] on this rank's slab, (vertical, y, x). B is required for the
Poisson modes and for column-hse-ltau with align = "ptot"; `nothing` means zero otherwise. T and B are not modified.
Returns `ws.res` (buffers reused by the next solve; see `snapshot`).
"""
function solve_embedded!(ws::EmbeddedWorkspace, lnT::AbstractArray{<:Real,3}, Bx = nothing, By = nothing, Bz = nothing)
    t0 = time()
    par = ws.par; b = par.block; shp = (length(ws.axis), b.ny, b.nx)
    needB = poisson_mode(ws.mode) || (ws.mode == :column_hse_ltau && ws.column.align == "ptot")
    inputs = agree(par.comm, "solve_embedded!($(ws.mode)) input") do
        size(lnT) == shp || invalid("lnT has size $(size(lnT)); this rank's slab is $shp (vertical, y, x)")
        all(isfinite, lnT) || invalid("lnT has non-finite values")
        Bs = (Bx, By, Bz)
        if all(isnothing, Bs)
            needB && invalid("mode :$(ws.mode)$(ws.mode == :column_hse_ltau ? " with align = \"ptot\"" : "") needs Bx, By, Bz")
            Bs = (zeros(shp), zeros(shp), zeros(shp))
        else
            any(isnothing, Bs) && invalid("give all of Bx, By, Bz or none")
            for (nm, B) in zip(("Bx", "By", "Bz"), Bs)
                size(B) == shp || invalid("$nm has size $(size(B)); expected $shp")
                all(isfinite, B) || invalid("$nm has non-finite values")
            end
        end
        (Float64.(lnT), map(B -> B isa Array{Float64,3} ? B : Float64.(B), Bs))
    end
    lT, (bx, by, bz) = inputs
    st = EquilibriumStatus()
    res = ws.res
    if ws.mode == :column_hse_z
        column_z!(ws, lT, st)
        res.Pgas_start .= res.Pgas
    elseif ws.mode == :mhs_poisson_z
        column_z!(ws, lT, st)
        res.Pgas_start .= res.Pgas
        st.valid && poisson_z!(ws, lT, bx, by, bz, st)
    elseif ws.mode == :column_hse_ltau
        column_ltau!(ws, lT, bx, by, bz, st)
        res.Pgas_start .= res.Pgas
    else
        column_ltau!(ws, lT, bx, by, bz, st)
        res.Pgas_start .= res.Pgas
        st.valid && poisson_ltau!(ws, lT, bx, by, bz, st)
    end
    st.valid && check_fields!(ws, st)
    st.status = !st.valid ? eos_failure : st.reason == :converged ? converged :
                st.reason == :max_steps ? max_iter : stalled
    st.seconds = time() - t0
    res.status = st
    return res
end

"""Mark the status invalid with `reason` (only the first failure is kept as the reason)."""
function fail!(st::EquilibriumStatus, reason::Symbol, n::Int, msg::AbstractString)
    st.counts[reason] = n
    push!(st.messages, msg)
    if st.valid
        st.valid = false
        st.reason = reason
    end
    return st
end

"""Column HSE on the height grid, every owned column independently (threaded); fills lnP, Pgas, ρ, nₑ, χ, τ, z."""
function column_z!(ws::EmbeddedWorkspace, lT, st)
    par = ws.par; res = ws.res; z = ws.axis
    o = hse_opts(ws.column, ws.g)
    nk, ny, nx = size(lT)
    ncol = ny * nx
    cst = zeros(UInt8, ny, nx); nsub = zeros(Int16, ny, nx); ttm = zeros(UInt8, ny, nx)
    nchunk = min(ncol, 8 * Threads.nthreads())
    Threads.@threads :static for ch in 1:nchunk
        u = zeros(nk); m = zeros(nk); chi = zeros(nk); tau = zeros(nk)
        Pc = zeros(nk); rc = zeros(nk); nc = zeros(nk); sc = zeros(UInt8, nk)
        for c in ch:nchunk:ncol
            j = (c - 1) % ny + 1
            i = (c - 1) ÷ ny + 1
            X = @view lT[:, j, i]
            fill!(u, NaN); fill!(m, NaN)
            fl, nmax, ak = hse_column!(u, m, X, z, ws.eos, o)
            cst[j, i] = fl; nsub[j, i] = nmax
            if fl & COL_EOS_FAIL != 0
                @views res.lnP[:, j, i] .= NaN; res.Pgas[:, j, i] .= NaN; res.rho[:, j, i] .= NaN
                @views res.ne[:, j, i] .= NaN; res.chi500[:, j, i] .= NaN; res.tau500[:, j, i] .= NaN
                @views res.logtau500[:, j, i] .= NaN; res.eos_flags[:, j, i] .= EOS_OUT_OF_ENVELOPE
                res.z_tau1[j, i] = NaN
                continue
            end
            ttm[j, i], res.z_tau1[j, i] = column_state!(Pc, rc, nc, chi, tau, sc, X, u, z, ak, ws.eos, o)
            @inbounds for k in 1:nk
                res.lnP[k, j, i] = u[k]; res.Pgas[k, j, i] = Pc[k]; res.rho[k, j, i] = rc[k]; res.ne[k, j, i] = nc[k]
                res.chi500[k, j, i] = chi[k]; res.tau500[k, j, i] = tau[k]; res.logtau500[k, j, i] = log10(tau[k])
                res.eos_flags[k, j, i] = sc[k]
            end
        end
    end
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nk
        res.z[k, j, i] = z[k]
    end
    nfail = gcount(par, count(f -> f & COL_EOS_FAIL != 0, cst))
    st.stats["columns_eos_failed"] = nfail
    st.stats["columns_rk_refined"] = gcount(par, count(f -> f & COL_RK_REFINED != 0, cst))
    st.stats["columns_rk_tol_not_met"] = gcount(par, count(f -> f & COL_RK_TOL_NOT_MET != 0, cst))
    st.stats["max_substeps"] = Int(gmax(par, Float64(maximum(nsub))))
    st.stats["tau_exp_tail_columns"] = gcount(par, count(==(TAU_EXP_TAIL), ttm))
    st.stats["tau_half_cell_columns"] = gcount(par, count(==(TAU_HALF_CELL), ttm))
    nfail > 0 && fail!(st, :eos_out_of_envelope, nfail, "$nfail columns of the height-grid column HSE left the EOS table")
    return st
end

"""ρ, nₑ, χ₅₀₀, EOS flags, τ₅₀₀ and z(τ = 1) of every owned column from (ln T, P) on the height grid (one `state` call
per node, as `PoissonZ.tau_columns`)."""
function height_state!(ws::EmbeddedWorkspace, lT)
    res = ws.res; z = ws.axis; o = ws.column
    nk, ny, nx = size(lT)
    Threads.@threads for i in 1:nx
        chi = zeros(nk); tc = zeros(nk)
        @inbounds for j in 1:ny
            for k in 1:nk
                r, n, c, s = state(ws.eos, lT[k, j, i], log(res.Pgas[k, j, i]))
                res.rho[k, j, i] = exp(r); res.ne[k, j, i] = exp(n); chi[k] = exp(c)
                res.chi500[k, j, i] = chi[k]; res.eos_flags[k, j, i] = s
            end
            tau_column!(tc, chi, z; top = o.tau_top, H_max = o.H_max, mu = o.mu)
            for k in 1:nk
                res.tau500[k, j, i] = tc[k]; res.logtau500[k, j, i] = log10(tc[k])
            end
            res.z_tau1[j, i] = z_tau1(tc, z)
        end
    end
end

"""Poisson step on the height grid from the column-HSE start in ws.res."""
function poisson_z!(ws::EmbeddedWorkspace, lT, bx, by, bz, st)
    par = ws.par; res = ws.res; o = ws.poisson; z = ws.axis
    Lx, Ly, Lz = PoissonZ.lorentz(par, bx, by, bz, z, ws.dx, ws.dy)
    sol = PoissonZ.solve_pressure(par, ws.eos, ws.g, lT, Lx, Ly, Lz, res.Pgas_start, z, ws.dx, ws.dy;
                                  zcut = o.z_cut_m, maxouter = o.max_steps, rtol = o.rtol, anchor_w = o.anchor_w,
                                  cg_tol = o.cg_tol, cg_maxit = o.cg_maxit, n_consec = o.n_consec, logf = ws.logf,
                                  S = ws.fft, rms_atol = o.rms_atol)
    res.Pgas .= sol.P
    res.lnP .= log.(sol.P)
    st.reason = sol.reason; st.steps = sol.steps; st.phi0 = sol.Φ0; st.phi = sol.Φ; st.rms0 = sol.rms0; st.rms = sol.rms
    st.phi0_zero = sol.phi0_zero; st.gn_decrement = sol.gn_decrement; st.grad_norm = sol.grad_norm
    st.cg_its = sol.cg_its; st.cg_rel = sol.cg_rel
    st.stats["cut_level"] = sol.kc
    sol.cont_bad > 0 && fail!(st, :continuation_failed, sol.cont_bad,
                              "hydrostatic continuation above z_cut left $(sol.cont_bad) non-finite pressures")
    st.valid && height_state!(ws, lT)
    return st
end

"""ρ, nₑ, χ₅₀₀ and EOS flags on the τ grid; τ and log τ are the native axis; z(τ = 1) by interpolation in log τ."""
function tau_state!(ws::EmbeddedWorkspace, lT)
    res = ws.res; lt = ws.axis
    nk, ny, nx = size(lT)
    Threads.@threads for i in 1:nx
        @inbounds for j in 1:ny
            for k in 1:nk
                r, n, c, s = state(ws.eos, lT[k, j, i], res.lnP[k, j, i])
                res.rho[k, j, i] = exp(r); res.ne[k, j, i] = exp(n); res.chi500[k, j, i] = exp(c)
                res.eos_flags[k, j, i] = s
                res.logtau500[k, j, i] = lt[k]; res.tau500[k, j, i] = exp10(lt[k])
            end
            res.z_tau1[j, i] = PoissonTau.interp_lt(@view(res.z[:, j, i]), lt, 0.0)
        end
    end
end

"""Column HSE on the τ grid (aligned on P + B²/2μ₀ if requested, gauged to mean z(τ = 1) = 0)."""
function column_ltau!(ws::EmbeddedWorkspace, lT, bx, by, bz, st)
    res = ws.res; o = ws.column
    lnP, z, ncap, Pt, nb, nbad = PoissonTau.column_hse_tau(ws.par, ws.axis, lT, bx, by, bz, ws.eos, ws.g;
                                                           cap_per_dex = o.cap_per_dex, align = o.align)
    res.lnP .= lnP; res.z .= z; res.Pgas .= exp.(lnP)
    st.stats["capped_intervals"] = ncap
    st.stats["align_total_pressure_Pa"] = Pt
    st.stats["align_extrapolated_columns"] = nb
    if nbad > 0
        fill!(res.eos_flags, EOS_OUT_OF_ENVELOPE)
        return fail!(st, :eos_out_of_envelope, nbad, "column HSE in τ left the EOS table in $nbad points")
    end
    poisson_mode(ws.mode) || tau_state!(ws, lT)
    return st
end

"""Gauss–Newton solve on the τ grid from the column-HSE start in ws.res; gauged afterwards (mean z(τ = 1) = 0)."""
function poisson_ltau!(ws::EmbeddedWorkspace, lT, bx, by, bz, st)
    par = ws.par; res = ws.res; o = ws.poisson; lt = ws.axis
    u, z, Φ0, Φ, steps, kc, info = PoissonTau.solve_tau(par, ws.eos, ws.g, lt, lT, bx, by, bz, copy(res.lnP), res.z,
                                                        ws.dx, ws.dy; logtau_cut = o.logtau_cut, max_steps = o.max_steps,
                                                        rtol = o.rtol, n_consec = o.n_consec, cg_maxit = o.cg_maxit,
                                                        cg_tol = o.cg_tol, dc_cap = o.dc_cap_m, du_cap = o.du_cap,
                                                        lm_mu0 = o.lm_mu0, logf = ws.logf, S = ws.fft,
                                                        rms_atol = o.rms_atol)
    if info.reason == :eos_out_of_envelope
        fill!(res.eos_flags, EOS_OUT_OF_ENVELOPE)
        return fail!(st, :eos_out_of_envelope, 1, "the τ-grid Poisson start state lies outside the EOS table")
    end
    st.stats["gauge_shift_m"] = PoissonTau.gauge!(par, z, lt)
    res.lnP .= u; res.z .= z; res.Pgas .= exp.(u)
    st.reason = info.reason; st.steps = steps; st.phi0 = Φ0; st.phi = Φ; st.rms0 = info.rms0; st.rms = info.rms
    st.phi0_zero = info.phi0_zero; st.gn_decrement = info.gn_decrement; st.grad_norm = info.grad_norm
    st.cg_its = info.cg_its; st.cg_rel = info.cg_rel; st.lm_mu = info.lm_mu
    st.stats["cut_level"] = kc
    tau_state!(ws, lT)
    return st
end

"""Validity of the returned fields (global counts; every rank reaches the same verdict)."""
function check_fields!(ws::EmbeddedWorkspace, st)
    par = ws.par; res = ws.res
    neos = gcount(par, count(f -> f & EOS_OUT_OF_ENVELOPE != 0, res.eos_flags))
    st.eos_violations = neos
    neos > 0 && return fail!(st, :eos_out_of_envelope, neos, "$neos nodes outside the EOS table")
    nnf = gcount(par, sum(count(!isfinite, a) for a in (res.Pgas, res.rho, res.ne, res.chi500, res.tau500, res.z)))
    nnf > 0 && return fail!(st, :nonfinite, nnf, "$nnf non-finite values in the returned fields")
    nnp = gcount(par, count(<=(0), res.Pgas) + count(<=(0), res.rho) + count(<=(0), res.chi500))
    nnp > 0 && return fail!(st, :nonpositive, nnp, "$nnp non-positive pressures, densities or opacities")
    nk, ny, nx = size(res.z)
    nfold = 0
    @inbounds for i in 1:nx, j in 1:ny
        ok = true
        for k in 1:nk-1
            (res.z[k+1, j, i] < res.z[k, j, i] && res.tau500[k+1, j, i] > res.tau500[k, j, i]) || (ok = false)
        end
        ok || (nfold += 1)
    end
    nfold = gcount(par, nfold)
    nfold > 0 && return fail!(st, :folded, nfold, "$nfold columns where z does not decrease or τ does not increase downwards")
    return st
end
