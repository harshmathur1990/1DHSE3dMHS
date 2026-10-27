#=
Synthetic off-centre τ-grid input and its truth from a Bifrost snapshot (design/tau/VIEWING_ANGLE_DESIGN.md §7).

Rays toward an observer at μ = cos θ and azimuth φ (from +x toward +y) are x = ξ + t_x z, y = η + t_y z with
t = tan θ (cos φ, sin φ), one per footprint (ξ, η) on the Bifrost x, y grid. ln T, ln P, ln ρ, B and χ₅₀₀ (EOS table,
from Bifrost T and P) are sheared onto the footprint grid level by level (periodic bilinear, ViewTau.shear_levels),
τ_LOS is integrated down each ray (Tau.tau_column!, path length dz/μ), and every quantity is interpolated linearly in
log10 τ_LOS onto the log10 τ500 axis of a template τ input (T exp of the ln T interpolation). B stays in the local frame.

Rays whose τ at the top of the box already exceeds the axis minimum get the top values of T and B and a linear
extrapolation of z, ln P, ln ρ in log τ there (counted and stored as the attribute top_extrapolated_rays); every ray must
reach the bottom of the axis.

usage: julia --project=. -t N scripts/viewing/make_slanted_tau.jl BIFROST.nc TEMPLATE_TAU.nc MU AZIMUTH_DEG OUT_INPUT.nc OUT_TRUTH.h5 [ZMIN_M]
OUT_INPUT: the IO/input-mhs-ltau.nc layout (log10tau500, x, y [m], temperature, B_x, B_y, B_z; h5py (x, y, τ)).
OUT_TRUTH: truth/{z_m, lnP, lnrho, T} h5py (τ, y, x), truth/z_tau1_m (y, x) on footprints, logtau500, x_m, y_m.
=#
using HSEMHS
using HDF5
using MPI
using Printf

const H = HSEMHS

"""
Linear interpolation of v at log10 τ = L along increasing lτ (NaN below the bottom). Above the top (rays whose top τ
exceeds the axis minimum, through the hot top of the box) `top` decides: :hold keeps the top value (T, B), :extrapolate
continues the top segment linearly (z, ln P, ln ρ).
"""
function interp_at(v, lτ, L; top = :hold)
    L > lτ[end] && return NaN
    if L < lτ[1]
        top == :hold && return v[1]
        return v[1] + (L - lτ[1]) * (v[2] - v[1]) / (lτ[2] - lτ[1])
    end
    k = searchsortedlast(lτ, L)
    k == length(lτ) && return v[end]
    w = (L - lτ[k]) / (lτ[k+1] - lτ[k])
    return v[k] + w * (v[k+1] - v[k])
end

function main(bfile, tfile, mu_s, az_s, out_in, out_truth, zmin_s = "-2.4e6")
    H.ensure_mpi()
    μ = parse(Float64, mu_s); φ = parse(Float64, az_s); zmin = parse(Float64, zmin_s)
    v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
    isempty(H.option_errors(v)) || error(join(H.option_errors(v), "; "))
    t = H.slant(v)
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt = h5open(f -> read(f["log10tau500"]), tfile); nt = length(lt)
    bf = h5open(bfile, "r")
    zb = vec(read(bf["z"]))                                         # top first, m
    ks = findall(>=(zmin), zb); z = zb[ks]; nz = length(z)
    x = Float64.(read(bf["x"])); y = Float64.(read(bf["y"])); NX = length(x); NY = length(y)
    dx = x[2] - x[1]; dy = y[2] - y[1]
    @printf("μ %.4f (θ %.2f°), φ %.1f°, t = (%.4f, %.4f); %d levels %.3f … %.3f Mm; max shift %.1f cells\n",
            μ, acosd(μ), φ, t..., nz, z[1] / 1e6, z[end] / 1e6, maximum(abs, z) * hypot(t...) / dx)
    rd(nm) = (a = Float64.(bf[nm][:, :, :, 1]); a[ks, :, :])        # Julia view of Bifrost: (depth, y, x, t)
    lT = log.(rd("temperature")); lP = log.(rd("gas_pressure")); lr = log.(rd("density"))
    Bx = rd("B_x"); By = rd("B_y"); Bz = rd("B_z")
    close(bf)
    chi = similar(lT); nbad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(lT)
        _, _, c, _ = H.state(tab, lT[idx], lP[idx])
        isfinite(c) || (Threads.atomic_add!(nbad, 1); c = -700.0)
        chi[idx] = exp(c)
    end
    println("χ500 outside the table at $(nbad[]) Bifrost points")
    par = H.slab_par(MPI.COMM_WORLD, NX, NY, nz)
    S(A) = H.shear_levels(par, A, z, dx, dy, t, 1.0)
    sχ = S(chi); sT = S(lT); sP = S(lP); sr = S(lr); sBx = S(Bx); sBy = S(By); sBz = S(Bz)
    println("sheared")
    T_in = fill(NaN, nt, NY, NX); bx = similar(T_in); by = similar(T_in); bz = similar(T_in)
    zt = similar(T_in); uP = similar(T_in); ur = similar(T_in); z1 = fill(NaN, NY, NX); nmiss = Threads.Atomic{Int}(0)
    ntop = Threads.Atomic{Int}(0); ltop = fill(NaN, NY, NX)
    Threads.@threads for i in 1:NX
        tc = zeros(nz)
        for j in 1:NY
            H.tau_column!(tc, @view(sχ[:, j, i]), z; mu = μ)
            lτ = log10.(max.(tc, 1e-300))
            z1[j, i] = H.z_tau1(tc, z)
            lt[end] > lτ[end] && Threads.atomic_add!(nmiss, 1)
            lt[1] < lτ[1] && Threads.atomic_add!(ntop, 1)
            ltop[j, i] = lτ[1]
            for k in 1:nt
                T_in[k, j, i] = exp(interp_at(@view(sT[:, j, i]), lτ, lt[k]))
                bx[k, j, i] = interp_at(@view(sBx[:, j, i]), lτ, lt[k])
                by[k, j, i] = interp_at(@view(sBy[:, j, i]), lτ, lt[k])
                bz[k, j, i] = interp_at(@view(sBz[:, j, i]), lτ, lt[k])
                zt[k, j, i] = interp_at(z, lτ, lt[k]; top = :extrapolate)
                uP[k, j, i] = interp_at(@view(sP[:, j, i]), lτ, lt[k]; top = :extrapolate)
                ur[k, j, i] = interp_at(@view(sr[:, j, i]), lτ, lt[k]; top = :extrapolate)
            end
        end
    end
    q = sort(vec(ltop))
    @printf("log10 τ at the top of the rays: p50 %.2f, p99 %.2f, max %.2f; %d rays above the axis minimum %.1f (top extrapolated)\n",
            q[cld(end, 2)], q[ceil(Int, 0.99 * end)], q[end], ntop[], lt[1])
    println("rays not reaching the bottom of the τ axis: $(nmiss[]) of $(NX * NY)")
    nmiss[] == 0 || error("the τ axis bottom is not reached by every ray; lower ZMIN_M")
    h5open(out_in, "w") do f
        f["log10tau500"] = lt
        f["x"] = x; attributes(f["x"])["units"] = "m"
        f["y"] = y; attributes(f["y"])["units"] = "m"
        f["temperature"] = T_in; f["B_x"] = bx; f["B_y"] = by; f["B_z"] = bz     # Julia (τ, y, x) = h5py (x, y, τ)
        a = attributes(f)
        a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2]
        a["source"] = abspath(bfile); a["gravity_m_s2"] = 274.0
        a["description"] = "one line of sight per footprint (x, y at z = 0) on log10 tau500 along the line of sight; B local frame"
    end
    h5open(out_truth, "w") do f
        p(A) = permutedims(A, ndims(A):-1:1)                       # Julia (x, y, τ): h5py (τ, y, x) as the run outputs
        f["truth/z_m"] = p(zt); f["truth/lnP"] = p(uP); f["truth/lnrho"] = p(ur); f["truth/T"] = p(T_in)
        f["truth/z_tau1_m"] = p(z1)
        f["logtau500"] = lt; f["x_m"] = x; f["y_m"] = y
        a = attributes(f)
        a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2]; a["source"] = abspath(bfile)
        a["top_extrapolated_rays"] = ntop[]
    end
    H.free_parallel!(par)
    println("wrote $out_in and $out_truth")
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
