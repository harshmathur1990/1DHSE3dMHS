#=
Bifrost truth on the log10 τ500 grid of a τ-grid input.

τ500 of each Bifrost column from its own T and P (χ500 from the EOS table, top-first trapezoid as Tau.tau_column!), then
z, ln P, ln ρ and ln T interpolated linearly in log10 τ onto the input's levels. The input temperature is compared with
the interpolated Bifrost temperature to confirm both grids describe the same atmosphere.

usage: julia --project=. -t N scripts/ltaucut/truth_on_tau.jl BIFROST.nc TAU_INPUT.nc OUT.h5
output (h5py layout (τ, y, x), SI): truth/z_m, truth/lnP, truth/lnrho, truth/T, truth/z_tau1_m, logtau500, x_m, y_m
=#
using HSEMHS
using HDF5
using Printf

const H = HSEMHS

"""Linear interpolation of v at log10 τ = L along increasing lτ (NaN outside)."""
function interp_at(v, lτ, L)
    (L < lτ[1] || L > lτ[end]) && return NaN
    k = searchsortedlast(lτ, L)
    k == length(lτ) && return v[end]
    w = (L - lτ[k]) / (lτ[k+1] - lτ[k])
    return v[k] + w * (v[k+1] - v[k])
end

function main(bfile, tfile, out)
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt = h5open(f -> read(f["log10tau500"]), tfile)
    nt = length(lt)
    bf = h5open(bfile, "r"); tf = h5open(tfile, "r")
    zb = vec(read(bf["z"]))                                   # top first, m
    x = Float64.(read(bf["x"])); y = Float64.(read(bf["y"]))
    nz = length(zb); NX = length(x); NY = length(y)
    zt = fill(NaN, nt, NY, NX); lP = similar(zt); lr = similar(zt); Tt = similar(zt); z1 = fill(NaN, NY, NX)
    dT = zeros(nt, NY, NX); nbad = Threads.Atomic{Int}(0)
    for i in 1:NX                                            # Julia view of Bifrost: (depth, y, x, t)
        T = Float64.(bf["temperature"][:, :, i, 1]); P = Float64.(bf["gas_pressure"][:, :, i, 1])
        R = Float64.(bf["density"][:, :, i, 1]); Tin = Float64.(tf["temperature"][:, :, i])   # input: (τ, y, x)
        Threads.@threads for j in 1:NY
            χ = zeros(nz); τ = zeros(nz)
            for k in 1:nz
                _, _, c, _ = H.state(tab, log(T[k, j]), log(P[k, j]))
                isfinite(c) || (Threads.atomic_add!(nbad, 1); c = -700.0)
                χ[k] = exp(c)
            end
            H.tau_column!(τ, χ, zb)
            lτ = log10.(max.(τ, 1e-300))
            z1[j, i] = H.z_tau1(τ, zb)
            for k in 1:nt
                zt[k, j, i] = interp_at(zb, lτ, lt[k])
                lP[k, j, i] = interp_at(log.(P[:, j]), lτ, lt[k])
                lr[k, j, i] = interp_at(log.(R[:, j]), lτ, lt[k])
                Tt[k, j, i] = exp(interp_at(log.(T[:, j]), lτ, lt[k]))
                dT[k, j, i] = log(Tin[k, j] / Tt[k, j, i])
            end
        end
        i % 63 == 0 && (@printf("%d / %d x columns\n", i, NX); flush(stdout))
    end
    close(bf); close(tf)
    println("χ500 outside the table at $(nbad[]) Bifrost points")
    for k in 1:5:nt
        d = filter(isfinite, vec(dT[k, :, :]))
        @printf("log τ %5.1f: ln(T_input/T_bifrost) median %+.2e, max|.| %.2e, NaN %d\n", lt[k],
                isempty(d) ? NaN : sort(d)[cld(length(d), 2)], isempty(d) ? NaN : maximum(abs, d), NX * NY - length(d))
    end
    h5open(out, "w") do f
        p(a) = permutedims(a, ndims(a):-1:1)                  # Julia (x, y, τ) as the run outputs
        f["truth/z_m"] = p(zt); f["truth/lnP"] = p(lP); f["truth/lnrho"] = p(lr); f["truth/T"] = p(Tt)
        f["truth/z_tau1_m"] = p(z1)
        f["logtau500"] = lt; f["x_m"] = x; f["y_m"] = y
        attributes(f)["source"] = abspath(bfile); attributes(f)["tau_input"] = abspath(tfile)
    end
    println("wrote $out")
end

main(ARGS...)
