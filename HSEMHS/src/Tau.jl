# τ₅₀₀ on the native height grid (spec §5.4 with erratum E1; implementation doc §10).

"""
    tau_column!(tau, chi, z; top=:exp_tail, H_max=5.0e6, mu=1.0) -> method::UInt8

Vertical, top-first trapezoid of χ [m⁻¹] over z [m] (decreasing). `tau` receives τ/μ.
Top closure: `:exp_tail` uses τ_top = χ₁ H_χ with H_χ = (z₁ − z₂)/ln(χ₂/χ₁) (E1) when
χ₁ > 0, χ₂ > χ₁ and 0 < H_χ < H_max; otherwise the half-cell value ½ χ₁ (z₁ − z₂).
"""
function tau_column!(tau::AbstractVector, chi::AbstractVector, z::AbstractVector;
                     top::Symbol = :exp_tail, H_max::Float64 = 5.0e6, mu::Float64 = 1.0)
    NZ = length(z)
    method = TAU_HALF_CELL
    half = 0.5 * chi[1] * (z[1] - z[2])
    t0 = half
    if top == :exp_tail
        if chi[1] > 0 && chi[2] > chi[1]
            Hc = (z[1] - z[2]) / log(chi[2] / chi[1])
            if 0 < Hc < H_max
                t0 = chi[1] * Hc
                method = TAU_EXP_TAIL
            end
        end
    elseif top == :zero
        t0 = 0.0
        method = TAU_ZERO
    elseif top != :half_cell
        invalid("tau.top must be exp_tail, half_cell or zero; got $top")
    end
    acc = t0
    tau[1] = acc / mu
    @inbounds for k in 1:NZ-1
        acc += 0.5 * (chi[k] + chi[k+1]) * (z[k] - z[k+1])
        tau[k+1] = acc / mu
    end
    return method
end

"""Height where log₁₀τ = 0, linear in log τ between levels (NaN if τ never reaches 1)."""
function z_tau1(tau::AbstractVector, z::AbstractVector)
    for k in 1:length(z)-1
        if tau[k] <= 1.0 <= tau[k+1] && tau[k] > 0
            a = log10(tau[k]); b = log10(tau[k+1])
            return b == a ? z[k] : z[k] + (z[k+1] - z[k]) * (0 - a) / (b - a)
        end
    end
    return NaN
end

"""
    tau_column_vjp!(chibar, taubar, chi, z, method; mu=1.0) -> chibar

Adjoint of `tau_column!` at `chi` with its top closure `method` frozen (design/adjoint/01): adds ∂ℓ/∂χ to `chibar` for
the cotangent `taubar` of the returned `tau` (= τ/μ). z is the fixed axis.
"""
function tau_column_vjp!(chibar::AbstractVector, taubar::AbstractVector, chi::AbstractVector, z::AbstractVector,
                         method::UInt8; mu::Float64 = 1.0)
    NZ = length(z)
    A = 0.0                                              # Σ_{m ≥ k+1} τ̄_m / μ, built from the bottom
    @inbounds for k in NZ-1:-1:1
        A += taubar[k+1] / mu
        w = 0.5 * (z[k] - z[k+1]) * A
        chibar[k] += w; chibar[k+1] += w
    end
    @inbounds A += taubar[1] / mu                        # ∂ℓ/∂t₀
    if method == TAU_EXP_TAIL
        L = log(chi[2] / chi[1]); Hc = (z[1] - z[2]) / L
        chibar[1] += A * Hc * (1 + 1 / L)
        chibar[2] -= A * chi[1] * Hc / (L * chi[2])
    elseif method == TAU_HALF_CELL
        chibar[1] += A * 0.5 * (z[1] - z[2])
    end
    return chibar
end
