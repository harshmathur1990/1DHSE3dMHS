# The EOS table's envelope from the Gate 0 report (implementation doc §7.3); the EOS model and tables are WittBifrostEOS's.

"""Envelope from the Gate 0 report (`/meta/envelope_inputs`), or explicit overrides."""
function envelope_from_gate0(path::AbstractString; logP_range = Float64[])
    h5open(path, "r") do f
        g = f["meta/envelope_inputs"]
        a(k) = read_attribute(g, k)
        T_lo = 0.9 * a("T_min"); T_hi = 1.1 * a("T_max")
        if isempty(logP_range)
            lo = min(a("log10P_toprun_min"), a("log10P_botrun_min"), a("log10P_native_min")) - 1.0
            hi = max(a("log10P_toprun_max"), a("log10P_botrun_max"), a("log10P_native_max")) + 1.0
            src = "Gate 0 proxy anchor runs ±1 dex ($(path))"
        else
            lo, hi = logP_range
            src = "eos.logP_range override"
        end
        return Envelope(T_lo, T_hi, lo, hi, src)
    end
end
