# NLTE hydrogen ionization (eos.model = "nlte").
#
# The correction D(T, P) = log10(n_p/n_HI)_sim − log10(n_p/n_HI)_Saha(T, p_e,sim) is learned from
# Bifrost non-equilibrium hydrogen (hion) simulations by scripts/nlte/make_nlte_correction.py:
# the median per (log10 T, log10 P) bin, the runs weighted equally, smoothed, and tapered to 0
# where the simulations have no data and above 10^4.6 K. The NLTE EOS is the hot-branch model
# with the hydrogen Saha ratio g₂ = p(H⁺)/p(H) multiplied by 10^D at the self-consistent pₑ;
# everything else (He, metals, H⁻, H₂ and the opacity partials) stays LTE at that pₑ. Where
# D = 0 the model is the LTE model unchanged.
#
# The correction's Saha reference (u_HI = 2, u_HII = 1, χ_H = 13.598434 eV) differs from the
# EOS's (Wittmann partition functions and χ_H) by s(T) = log10(u_HI/(2 u_HII)) − θ (13.598434 eV − χ_H),
# up to 1.2 dex where Wittmann's u_HI grows at the hot end of the covered range. The EOS applies
# dH = D + W s(T), W the correction's coverage weight: the departure from the simulations' Saha
# reference where they have data, and exactly 0 (LTE) where W = 0, as D is.

struct NLTECorrection
    lT0::Float64                 # first log10 T node
    dlT::Float64
    lP0::Float64                 # first log10 P [Pa] node
    dlP::Float64
    D::Matrix{Float64}           # (nT, nP), log10 departure of n_p/n_HI from Saha (reference above), tapered
    W::Matrix{Float64}           # (nT, nP), coverage weight of D, 0 … 1
    path::String
    sha256::String
    meta::Dict{String,Any}
end

"""Read the correction written by scripts/nlte/make_nlte_correction.py."""
function load_nlte_correction(path::AbstractString)
    isfile(path) || invalid("NLTE correction $path does not exist (scripts/nlte/make_nlte_correction.py writes it)")
    h5open(path, "r") do f
        lT = read(f["log10T"]); lP = read(f["log10P"])
        D = permutedims(read(f["D"]))                 # written in C order (log10T, log10P)
        W = permutedims(read(f["weight"]))
        size(D) == size(W) == (length(lT), length(lP)) || invalid("NLTE correction $path: D and weight must have size $(length(lT)) × $(length(lP))")
        all(isfinite, D) && all(isfinite, W) || invalid("NLTE correction $path: D and weight must be finite")
        all(((d, w),) -> w != 0 || d == 0, zip(D, W)) || invalid("NLTE correction $path: D is nonzero where the weight is 0")
        uniform(a) = length(a) >= 4 && maximum(abs.(diff(a) .- (a[2] - a[1]))) < 1e-6 * abs(a[2] - a[1])
        uniform(lT) && uniform(lP) || invalid("NLTE correction $path: the log10 T and log10 P axes must be uniform")
        meta = Dict{String,Any}(k => read_attribute(f, k) for k in keys(attributes(f)))
        return NLTECorrection(lT[1], lT[2] - lT[1], lP[1], lP[2] - lP[1], D, W, abspath(path), file_sha256(path), meta)
    end
end

@inline catmull_rom(p0, p1, p2, p3, t) =
    0.5 * (2p1 + (p2 - p0) * t + (2p0 - 5p1 + 4p2 - p3) * t^2 + (3p1 - p0 - 3p2 + p3) * t^3)

"""Bicubic Catmull–Rom (C¹) of grid `M` at X = ln T, Y = ln P [Pa], edge nodes repeated; 0 outside the grid."""
@inline function grid_value(c::NLTECorrection, M::Matrix{Float64}, X::Float64, Y::Float64)
    nT, nP = size(M)
    u = (X / log(10.0) - c.lT0) / c.dlT
    v = (Y / log(10.0) - c.lP0) / c.dlP
    (0.0 <= u <= nT - 1 && 0.0 <= v <= nP - 1) || return 0.0
    i = min(unsafe_trunc(Int, u), nT - 2); j = min(unsafe_trunc(Int, v), nP - 2)
    t = u - i; s = v - j
    @inbounds begin
        row(ii) = catmull_rom(M[ii+1, max(j - 1, 0)+1], M[ii+1, j+1], M[ii+1, j+2], M[ii+1, min(j + 2, nP - 1)+1], s)
        return catmull_rom(row(max(i - 1, 0)), row(i), row(i + 1), row(min(i + 2, nT - 1)), t)
    end
end

"""
    nlte_departure(c, X, Y, uHf, eion_H) -> dH

log10 factor on the EOS's hydrogen Saha ratio at X = ln T, Y = ln P [Pa]: D + W s(T) (see the
header); `uHf()` returns the EOS's hydrogen partition functions at T (called only where W > 0)
and `eion_H` [eV] is its ionization energy. 0 where W = 0.
"""
@inline function nlte_departure(c::NLTECorrection, X::Float64, Y::Float64, uHf, eion_H::Float64)
    w = clamp(grid_value(c, c.W, X, Y), 0.0, 1.0)
    w == 0 && return 0.0
    T = exp(X); uH = uHf()
    return grid_value(c, c.D, X, Y) + w * (log10(uH[1] / (2 * uH[2])) - 5040.0 / T * (13.598434 - eion_H))
end
