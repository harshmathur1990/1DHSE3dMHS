# Status codes and bit masks (implementation doc §4.5).
# The enum order is severity order; collective_status reduces with `max`.

@enum SolverStatus::UInt8 converged = 0 stalled = 1 max_iter = 2 eos_failure = 3 invalid_input = 4

# per-node EOS bit mask
const EOS_OK              = 0x00
const EOS_HOT_BRANCH      = 0x01
const EOS_BLEND           = 0x02
const EOS_OUT_OF_ENVELOPE = 0x04
const OPACITY_INCOMPLETE  = 0x08
const EOS_DIRECT_FAILED   = 0x10

# per-column column_hse mask
const COL_OK             = 0x00
const COL_RK_REFINED     = 0x01
const COL_RK_TOL_NOT_MET = 0x02
const COL_EOS_FAIL       = 0x04

# per-column tau_top_method
const TAU_EXP_TAIL  = 0x00
const TAU_HALF_CELL = 0x01
const TAU_ZERO      = 0x02

"""Error raised at API boundaries for rejected input. Carries a precise message."""
struct InvalidInput <: Exception
    msg::String
end
Base.showerror(io::IO, e::InvalidInput) = print(io, "invalid_input: ", e.msg)

invalid(msg) = throw(InvalidInput(msg))
