# Status codes and bit masks (implementation doc §4.5).
# The enum order is severity order; collective_status reduces with `max`.

@enum SolverStatus::UInt8 converged = 0 stalled = 1 max_iter = 2 eos_failure = 3 invalid_input = 4

# per-node EOS bit mask (EOS_OK … EOS_DIRECT_FAILED): WittBifrostEOS's

# per-column column_hse mask
const COL_OK             = 0x00
const COL_RK_REFINED     = 0x01
const COL_RK_TOL_NOT_MET = 0x02
const COL_EOS_FAIL       = 0x04

# per-column tau_top_method
const TAU_EXP_TAIL  = 0x00
const TAU_HALF_CELL = 0x01
const TAU_ZERO      = 0x02

# InvalidInput and invalid(msg), the error raised at API boundaries for rejected input: WittBifrostEOS's
