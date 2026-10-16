# ccall wrapper for the Wittmann shim (implementation doc §7.2).
#
# The library path is resolved at run time: ENV["HSEMHS_LIBWITT"] or
# deps/witt/libwitt.<dylib|so> in the package. Symbols are looked up once.

struct WittState
    rho::Float64
    ne::Float64
    pe::Float64
    mu_bar::Float64
    nhtot::Float64
    closure_residual::Float64
    dif::Float64
    iterations::Int32
    converged::Int32
end

const WITT_SYMS = (:witt_ctx_new, :witt_ctx_free, :witt_ctx_state, :witt_ctx_partials,
                   :witt_ctx_mass_per_H_kg, :witt_ctx_abundances, :witt_ctx_nstage,
                   :witt_ctx_eion_ev, :witt_ctx_tpf_max, :witt_ctx_tpf_min, :witt_ctx_pf,
                   :witt_ctx_ab_others, :witt_ctx_rho_from_H, :witt_shim_fix_gasc_h2, :witt_ctx_tpf)

const WITT_LIB = Ref{Ptr{Cvoid}}(C_NULL)
const WITT_FN = Dict{Symbol,Ptr{Cvoid}}()
const WITT_LOCK = ReentrantLock()

function libwitt_path()
    p = get(ENV, "HSEMHS_LIBWITT", "")
    isempty(p) || return p
    ext = Sys.isapple() ? "dylib" : "so"
    return joinpath(pkgdir(@__MODULE__), "deps", "witt", "libwitt.$ext")
end

function wittfn(s::Symbol)
    lock(WITT_LOCK) do
        if WITT_LIB[] == C_NULL
            p = libwitt_path()
            isfile(p) || invalid("Wittmann library not found at $p; run `julia deps/witt/build.jl`")
            WITT_LIB[] = Libdl.dlopen(p)
            for n in WITT_SYMS
                WITT_FN[n] = Libdl.dlsym(WITT_LIB[], n)
            end
        end
        return WITT_FN[s]
    end
end

"""Wittmann EOS context: immutable after construction, safe for concurrent calls."""
mutable struct WittCtx
    ptr::Ptr{Cvoid}
    pf_path::String
    prec::Float64
    maxit::Int
    log_abund::Union{Nothing,Vector{Float64}}
    fstate::Ptr{Cvoid}
    fpartials::Ptr{Cvoid}
end

function WittCtx(pf_path::AbstractString; prec::Float64 = 1e-10, maxit::Int = 5000,
                 log_abund::Union{Nothing,Vector{Float64}} = nothing)
    isfile(pf_path) || invalid("partition-function file $pf_path does not exist")
    log_abund === nothing || length(log_abund) == 99 || invalid("abundance vector must have 99 entries")
    abp = log_abund === nothing ? C_NULL : pointer(log_abund)
    ptr = GC.@preserve log_abund ccall(wittfn(:witt_ctx_new), Ptr{Cvoid},
                                       (Cstring, Float64, Cint, Ptr{Float64}), pf_path, prec, maxit, abp)
    ptr == C_NULL && invalid("could not create the Wittmann EOS context (pf file $pf_path)")
    c = WittCtx(ptr, String(pf_path), prec, maxit, log_abund, wittfn(:witt_ctx_state), wittfn(:witt_ctx_partials))
    finalizer(c) do c
        c.ptr == C_NULL || ccall(wittfn(:witt_ctx_free), Cvoid, (Ptr{Cvoid},), c.ptr)
        c.ptr = C_NULL
    end
    return c
end

"""Direct state at (T [K], P [Pa]). Returns (state, code); code 0 = converged."""
@inline function witt_state(c::WittCtx, T::Float64, P::Float64)
    r = Ref{WittState}()
    code = ccall(c.fstate, Cint, (Ptr{Cvoid}, Float64, Float64, Ref{WittState}), c.ptr, T, P, r)
    return r[], Int(code)
end

"""Background partial densities (cm⁻³, divided by partition functions) for the opacity."""
function witt_partials!(n17::Vector{Float64}, c::WittCtx, T::Float64, P::Float64, pe_cgs::Float64)
    xna = Ref(0.0); xne = Ref(0.0)
    code = ccall(c.fpartials, Cint, (Ptr{Cvoid}, Float64, Float64, Float64, Ref{Float64}, Ref{Float64}, Ptr{Float64}),
                 c.ptr, T, P, pe_cgs, xna, xne, n17)
    return xna[], xne[], Int(code)
end

witt_mass_per_H(c::WittCtx) = ccall(wittfn(:witt_ctx_mass_per_H_kg), Float64, (Ptr{Cvoid},), c.ptr)

function witt_abundances(c::WittCtx)
    ab = zeros(99); am = zeros(99)
    ccall(wittfn(:witt_ctx_abundances), Cvoid, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}), c.ptr, ab, am)
    return ab, am
end

witt_nstage(c::WittCtx, atom::Int) = Int(ccall(wittfn(:witt_ctx_nstage), Cint, (Ptr{Cvoid}, Cint), c.ptr, atom - 1))
witt_eion_ev(c::WittCtx, atom::Int, stage::Int) =
    ccall(wittfn(:witt_ctx_eion_ev), Float64, (Ptr{Cvoid}, Cint, Cint), c.ptr, atom - 1, stage - 1)
witt_tpf_range(c::WittCtx) = (ccall(wittfn(:witt_ctx_tpf_min), Float64, (Ptr{Cvoid},), c.ptr),
                              ccall(wittfn(:witt_ctx_tpf_max), Float64, (Ptr{Cvoid},), c.ptr))

"""Partition functions of all stages of element `atom` (1-based) at T (Wittmann tables)."""
function witt_pf(c::WittCtx, atom::Int, T::Float64)
    u = zeros(8)
    n = ccall(wittfn(:witt_ctx_pf), Cint, (Ptr{Cvoid}, Cint, Float64, Ptr{Float64}), c.ptr, atom - 1, T, u)
    n < 0 && invalid("partition function failed for element $atom")
    return u[1:n]
end
"""Temperatures [K] at which the Wittmann partition functions are tabulated."""
function witt_tpf(c::WittCtx)
    out = zeros(4096)
    n = ccall(wittfn(:witt_ctx_tpf), Cint, (Ptr{Cvoid}, Ptr{Float64}, Cint), c.ptr, out, 4096)
    return out[1:min(n, 4096)]
end
witt_ab_others(c::WittCtx) = ccall(wittfn(:witt_ctx_ab_others), Float64, (Ptr{Cvoid},), c.ptr)
witt_rho_from_H(c::WittCtx) = ccall(wittfn(:witt_ctx_rho_from_H), Float64, (Ptr{Cvoid},), c.ptr)
witt_fix_gasc_h2() = ccall(wittfn(:witt_shim_fix_gasc_h2), Cint, ()) == 1

"""SHA-256 of a file (provenance)."""
file_sha256(path) = bytes2hex(open(sha256, path))
