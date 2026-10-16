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
