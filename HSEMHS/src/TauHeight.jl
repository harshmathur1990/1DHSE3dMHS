# Height of τ₅₀₀ = 1 for an MHS state (M5h; the user's iteration-stability criterion, REVIEW_M5g_2026-10-21.md §2):
# dz_max[n] = max over columns |z_τ1[n] − z_τ1[n−1]| between adjacent ACCEPTED iterations, threshold < 1e-3 m (absolute).
# χ₅₀₀ comes from the solver's own EOS/opacity table (`state`, the same table column_hse exports), τ from the native-height
# trapezoid `tau_column!` and the crossing from `z_tau1` (linear in log₁₀ τ), with fixed, recorded settings.

const TAU1_OK = 0x00
const TAU1_NO_CROSSING = 0x01         # τ never reaches 1 in the column (or τ₁ > 1 at the top: no bracketing levels)
const TAU1_NONFINITE = 0x02           # non-finite χ or τ somewhere in the column
const TAU1_EOS = 0x04                 # EOS status other than OK/flags (outside the table envelope)

"""
    tau1_heights!(zt, flag, pb, u; top = :exp_tail, H_max = 5.0e6, mu = 1.0, tau = nothing) -> zt

For every owned column (i, j): χ₅₀₀(ln T, u) from `state(pb.eos, ·)` [m⁻¹], τ by `tau_column!` (top-first trapezoid on the
native heights, top closure `top`, H_max, μ), z(τ = 1) by `z_tau1` [m]. `u` is the physical ln P (w.u after `objective!`).
`flag[i, j]` is a bit set of TAU1_*; zt[i, j] is NaN unless the flag is TAU1_OK. With `tau` (a Field), τ is stored too.
"""
function tau1_heights!(zt::AbstractMatrix{Float64}, flag::AbstractMatrix{UInt8}, pb, u::Field;
                       top::Symbol = :exp_tail, H_max::Float64 = 5.0e6, mu::Float64 = 1.0, tau = nothing)
    par = pb.par; b = par.block; NZ = b.nz
    z = pb.grid.z.s
    ncol = b.nx * b.ny
    nchunk = min(ncol, 8 * Threads.nthreads())
    Threads.@threads :static for ch in 1:nchunk
        chi = zeros(NZ); tc = zeros(NZ)
        for c in ch:nchunk:ncol
            i = (c - 1) % b.nx + 1; j = (c - 1) ÷ b.nx + 1
            fl = TAU1_OK
            @inbounds for k in 1:NZ
                _, _, lc, st = state(pb.eos, pb.X[i, j, k], u[i, j, k])
                st & EOS_OUT_OF_ENVELOPE != 0 && (fl |= TAU1_EOS)
                chi[k] = exp(lc)
                isfinite(chi[k]) || (fl |= TAU1_NONFINITE)
            end
            if fl == TAU1_OK
                tau_column!(tc, chi, z; top = top, H_max = H_max, mu = mu)
                all(isfinite, tc) || (fl |= TAU1_NONFINITE)
            else
                fill!(tc, NaN)
            end
            zz = fl == TAU1_OK ? z_tau1(tc, z) : NaN
            fl == TAU1_OK && isnan(zz) && (fl |= TAU1_NO_CROSSING)
            zt[i, j] = zz; flag[i, j] = fl
            if tau !== nothing
                @inbounds for k in 1:NZ
                    tau[i, j, k] = tc[k]
                end
            end
        end
    end
    return zt
end

"""
    tau1_change_stats(par, z1, f1, z0, f0) -> NamedTuple

Global (all ranks) statistics of |z1 − z0| over columns valid in both states: `dz_max` [m] with its global column
(`worst_i`, `worst_j`), `dz_median`, `dz_p99`, `ncomparable`, and the invalid counts of the current state by kind
(`nmissing`, `nnonfinite`, `neos`) and of the previous state (`nprev_invalid`). `all_valid` is true only if every column
of the domain is comparable. With z0 === nothing (initial state) the change statistics are NaN.
"""
function tau1_change_stats(par::ParCtx, z1::AbstractMatrix, f1::AbstractMatrix{UInt8}, z0, f0)
    b = par.block
    ncol = Float64(par.NX) * par.NY
    nmiss = count(f -> f & TAU1_NO_CROSSING != 0, f1); nnf = count(f -> f & TAU1_NONFINITE != 0, f1); neos = count(f -> f & TAU1_EOS != 0, f1)
    nprev = z0 === nothing ? 0 : count(!=(TAU1_OK), f0)
    loc = Float64[]; mx = -Inf; wi = 0; wj = 0
    if z0 !== nothing
        for j in 1:b.ny, i in 1:b.nx
            (f1[i, j] == TAU1_OK && f0[i, j] == TAU1_OK) || continue
            d = abs(z1[i, j] - z0[i, j]); push!(loc, d)
            d > mx && (mx = d; wi = b.i0 + i - 1; wj = b.j0 + j - 1)
        end
    end
    cnt = MPI.Allreduce([Float64(nmiss), Float64(nnf), Float64(neos), Float64(nprev), Float64(length(loc))], +, par.comm)
    all_d = _allgatherv(par, loc)                                  # one Float64 per column
    cand = MPI.Allgather([mx, Float64(wi), Float64(wj)], par.comm)
    best = (-Inf, 0, 0)
    for r in 0:div(length(cand), 3)-1
        v = cand[3r+1]
        v > best[1] && (best = (v, Int(cand[3r+2]), Int(cand[3r+3])))
    end
    q = isempty(all_d) ? (NaN, NaN) : (quantile(all_d, 0.5), quantile(all_d, 0.99))
    ncmp = Int(cnt[5])
    return (dz_max = z0 === nothing || isempty(all_d) ? NaN : best[1], dz_median = q[1], dz_p99 = q[2],
            worst_i = best[2], worst_j = best[3], ncomparable = ncmp, nmissing = Int(cnt[1]), nnonfinite = Int(cnt[2]),
            neos = Int(cnt[3]), nprev_invalid = Int(cnt[4]), all_valid = z0 !== nothing && ncmp == Int(ncol))
end

"""Global min / median / max of the valid z(τ = 1) values and the number of valid columns."""
function tau1_summary(par::ParCtx, z::AbstractMatrix, f::AbstractMatrix{UInt8})
    b = par.block
    loc = [z[i, j] for j in 1:b.ny for i in 1:b.nx if f[i, j] == TAU1_OK]
    all_z = _allgatherv(par, loc)
    isempty(all_z) && return (min = NaN, median = NaN, max = NaN, nvalid = 0)
    return (min = minimum(all_z), median = quantile(all_z, 0.5), max = maximum(all_z), nvalid = length(all_z))
end
