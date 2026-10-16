# Analytic EOSs for tests (implementation doc §7.8). Not physical outside tests.

"""Ideal gas with constant mean molecular weight μ: ρ = P μ m_u /(k T)."""
struct IdealGasMu <: AbstractEOS
    mu::Float64
end
const IdealGasIsothermal = IdealGasMu     # same closure; isothermality comes from T

const KAPPA0_TEST = 1.0e-2                # m² kg⁻¹, test opacity per mass

@inline lnrho(e::IdealGasMu, X::Float64, Y::Float64) = (Y + log(e.mu * M_U / K_B) - X, EOS_OK)
@inline lnrho_dY(e::IdealGasMu, X::Float64, Y::Float64) = (Y + log(e.mu * M_U / K_B) - X, 1.0, EOS_OK)
@inline function state(e::IdealGasMu, X::Float64, Y::Float64)
    lr = Y + log(e.mu * M_U / K_B) - X
    return lr, lr - log(e.mu * M_U), lr + log(KAPPA0_TEST), EOS_OK
end
mass_per_H(e::IdealGasMu) = e.mu * M_U
