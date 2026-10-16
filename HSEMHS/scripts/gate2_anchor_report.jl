# Gate 2 anchor report (spec §5 "Bifrost 385 with both anchors": report, not a threshold)
# and Gate 0 item 3 τ = 1 maps.
# usage: julia --project -t N scripts/gate2_anchor_report.jl BOTTOM.h5 TOP.h5 BIFROST.nc OUT.md
using HSEMHS, HDF5, Statistics, Printf
const H = HSEMHS

bot, top, bif, outmd = ARGS
stride = 4
targets = [3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5]

readP(f) = h5open(f, "r") do h
    P = read(h["output/Pgas"]); z = read(h["diagnostics/z"]); zt = read(h["diagnostics/z_tau500_1"])
    cs = read(h["diagnostics/column_status"])
    # Julia sees (z, y, x); return (x, y, z) subsampled
    permutedims(P, (3, 2, 1))[1:stride:end, 1:stride:end, :], z,
    permutedims(zt, (2, 1))[1:stride:end, 1:stride:end], permutedims(cs, (2, 1))[1:stride:end, 1:stride:end]
end
Pb, z, ztb, csb = readP(bot)
Pt, _, ztt, cst = readP(top)
spec = H.open_input(bif; crop_z_m = [-0.5e6, 3.0e6])
@assert maximum(abs.(spec.z .- z)) < 1e-3 "height grids differ"
Pn = h5open(bif, "r") do f
    a = f["gas_pressure"][spec.krange, 1:stride:504, 1:stride:504, 1]     # (z, y, x)
    Float64.(permutedims(a, (3, 2, 1)))
end
io = IOBuffer()
println(io, "# Gate 2 anchor report: column_hse (Wittmann table) vs Bifrost native pressure\n")
println(io, "Statistics on every $(stride)th column in x and y. log₁₀(P_column_hse / P_Bifrost), p1 / p50 / p99. NaN columns (left the EOS envelope) are excluded and counted.\n")
@printf(io, "- bottom anchor run: `%s`, %d of %d sampled columns failed\n", bot, count(c -> c & H.COL_EOS_FAIL != 0, csb), length(csb))
@printf(io, "- top anchor run: `%s`, %d of %d sampled columns failed (P beyond the table envelope at depth)\n\n", top, count(c -> c & H.COL_EOS_FAIL != 0, cst), length(cst))
println(io, "| z [Mm] | bottom anchor (P_ref = 103 509 Pa at −0.494 Mm) | top anchor (0.1 Pa at +2.98 Mm) | proxy (Gate 0) bottom / top p50 |")
println(io, "|---|---|---|---|")
g0 = h5open(joinpath(dirname(outmd), "gate0_bifrost385.h5"), "r") do f
    permutedims(read(f["diagnostics/layer_anchor"]), (2, 1))
end
for zt in targets
    k = argmin(abs.(z .- zt * 1e6))
    q(P) = (v = filter(isfinite, vec(log10.(P[:, :, k] ./ Pn[:, :, k]))); quantile(v, [0.01, 0.5, 0.99]))
    b = q(Pb); t = q(Pt)
    @printf(io, "| %.2f | %+.2f / %+.2f / %+.2f | %+.2f / %+.2f / %+.2f | %+.2f / %+.2f |\n", z[k] / 1e6, b..., t..., g0[k, 5], g0[k, 2])
end
qz(a) = (v = filter(isfinite, vec(a)) ./ 1e3; quantile(v, [0.01, 0.5, 0.99]))
println(io, "\n## τ₅₀₀ = 1 height (Gate 0 item 3)\n")
println(io, "| run | z(τ=1) p1 / p50 / p99 [km] |\n|---|---|")
@printf(io, "| bottom anchor | %.1f / %.1f / %.1f |\n", qz(ztb)...)
@printf(io, "| top anchor | %.1f / %.1f / %.1f |\n", qz(ztt)...)
println(io, "\nBifrost header ZTAU51 (mean τ₅₀₀ = 1 height) = 61 km.")
println(io, "\nImplied opposite-boundary pressure: bottom-anchored top-plane log₁₀P p1/p50/p99 = ",
        join(round.(quantile(filter(isfinite, vec(log10.(Pb[:, :, 1]))), [0.01, 0.5, 0.99]); digits = 2), " / "),
        "; top-anchored bottom-plane = ",
        join(round.(quantile(filter(isfinite, vec(log10.(Pt[:, :, end]))), [0.01, 0.5, 0.99]); digits = 2), " / "), " (finite columns only).")
write(outmd, String(take!(io)))
print(String(read(outmd)))
