# SI constants (implementation doc §3.4).

const K_B       = 1.380649e-23          # J/K   (SI 2019, exact)
const M_U       = 1.66053906660e-27     # kg
const M_E       = 9.1093837015e-31      # kg
const H_PL      = 6.62607015e-34        # J s   (exact)
const E_CH      = 1.602176634e-19       # C     (exact)
const MU0       = 1.25663706212e-6      # N A^-2 (CODATA 2018)
const SIGMA_T   = 6.6524587321e-29      # m^2
const ONE_GAUSS = 1.0e-4                # T

# Constants used inside the Wittmann C++ code (CODATA 2010, CGS there), in SI.
# Quantities that must agree with the C++ EOS state use these.
const K_B_WITT = 1.3806488e-23
const M_U_WITT = 1.660538921e-27

constants_meta() = Dict{String,Any}(
    "K_B" => K_B, "M_U" => M_U, "M_E" => M_E, "H_PL" => H_PL, "E_CH" => E_CH,
    "MU0" => MU0, "SIGMA_T" => SIGMA_T, "K_B_WITT" => K_B_WITT, "M_U_WITT" => M_U_WITT)
