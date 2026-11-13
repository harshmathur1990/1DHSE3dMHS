#=
τ-grid input and its truth on the grid of a MULTI3D run of a Bifrost snapshot (design/tau/VIEWING_ANGLE_DESIGN.md §7):
T and B on a log10 τ500 axis along the line of sight, one line of sight per column of the run, and the snapshot's z,
ln P, ln ρ, T on the same levels.

The atmosphere is the run's: DIR/mesh (cm) and DIR/atm3d (headerless float32, Fortran order (nx, ny, nz, 12); field 2 is
T). Its columns are a subsample of the Bifrost netCDF of the same snapshot, which supplies B, P and ρ (absent from atm3d).
The subsample is found, not assumed: every 2-fold subsampling offset of the netCDF, with x and y flipped or swapped, is
compared with atm3d's T on several levels, and exactly one must match to float32 rounding (relative difference < 1e-5).
The run's nz levels must be the netCDF's top nz levels. Arrays are then ordered with x and y increasing (right-handed, as
the netCDF), so B keeps its components; for en024048_hion 385 the run's column (i, j) (0-based) is netCDF (x = 2i,
y = 2(n_y − 1 − j) + 1).

Rays toward an observer at μ = cos θ and azimuth φ (from +x toward +y) are x = ξ + t_x z, y = η + t_y z with
t = tan θ (cos φ, sin φ), one per column (ξ, η). ln T, ln P, ln ρ, B and χ₅₀₀ (EOS table, from T and P) are sheared onto
the columns level by level (periodic bilinear on the run's grid, ViewTau.shear_levels). log10 τ along each ray is

  tau=eos        τ_LOS integrated down the ray from the table's χ₅₀₀ (Tau.tau_column!, path length dz/μ);
  tau=multi3d    the run's own τ500, DIR/H/tau500 (headerless float32, Fortran order (nx, ny, nz), as
                 read_tau500.py in the run directory reads it); μ = 1 only. The eos τ is computed as well, and the
                 run stops if the two z(τ = 1) maps do not correlate (corr < 0.9).

Every quantity is interpolated linearly in log10 τ onto the axis LT_MIN:DLT:LT_MAX (T as exp of the ln T interpolation; B
in the local frame), as make_logm_input.py does in log10 m. Rays whose τ at the top already exceeds the axis minimum
get the top values of T and B and a linear extrapolation of z, ln P, ln ρ (counted: top_extrapolated_rays); every ray
must reach the bottom of the axis.

usage: julia --project=. -t N scripts/inputs/make_tau_input.jl DIR BIFROST.nc OUT_INPUT.nc OUT_TRUTH.h5 [KEY=VALUE ...]
KEY: mu (1), azimuth_deg (0), tau (eos | multi3d), lt_min (-8), lt_max (1.4), dlt (0.1), table (cache/eos_table_fullz.h5:
the EOS table of the χ₅₀₀ that tau=eos integrates, e.g. cache/eos_table_nlte.h5).
OUT_INPUT: log10tau500, x, y [m], temperature, B_x, B_y, B_z; h5py (x, y, τ); attributes view_*, tau_source, grid, source.
OUT_TRUTH: truth/{z_m, lnP, lnrho, lnne, T} h5py (τ, y, x), truth/z_tau1_m (y, x), logtau500, x_m, y_m.
=#
using HSEMHS
using HDF5
using MPI
using Printf
using Statistics

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

"""(nx, ny, nz, z [m]) of DIR/mesh."""
function multi3d_mesh(dir)
    tok = split(read(joinpath(dir, "mesh"), String))
    nx = parse(Int, tok[1]); i = 2 + nx
    ny = parse(Int, tok[i]); i += 1 + ny
    nz = parse(Int, tok[i])
    return nx, ny, nz, parse.(Float64, tok[i+1:i+nz]) ./ 100
end

"""A headerless float32 cube of the run, Fortran order dims."""
function read_f32(fn, dims...)
    filesize(fn) == 4 * prod(dims) || error("$fn: $(filesize(fn)) bytes, expected float32 $(join(dims, " × "))")
    a = Array{Float32}(undef, dims...)
    read!(fn, a)
    return a
end

"""
Index maps of the run's columns onto the netCDF: returns (sx, sy, swap) such that run column (i, j) is netCDF column
(sx[i], sy[j]) (swap = false) or (sy[j], sx[i]) (swap = true), found by matching T (run (x, y, z), netCDF Julia
(z, y, x)) on `levels`. Exactly one candidate must match to `tol`.
"""
function register(Tm, Tb, levels; tol = 1e-5)
    nx, ny, _ = size(Tm); NZ, NY, NX = size(Tb)
    cand = Tuple{Vector{Int},Vector{Int},Bool,String}[]
    for swap in (false, true), ox in 1:2, oy in 1:2, fx in (false, true), fy in (false, true)
        MX, MY = swap ? (NY, NX) : (NX, NY)
        (2nx == MX && 2ny == MY) || continue
        sx = collect(ox:2:MX); sy = collect(oy:2:MY)
        fx && reverse!(sx); fy && reverse!(sy)
        push!(cand, (sx, sy, swap, "swap $swap, offsets ($ox, $oy), flips ($fx, $fy)"))
    end
    hits = filter(cand) do (sx, sy, swap, _)
        all(levels) do k
            maximum(1:nx) do i
                maximum(1:ny) do j
                    tb = swap ? Tb[k, sx[i], sy[j]] : Tb[k, sy[j], sx[i]]
                    abs(Tm[i, j, k] / tb - 1)
                end
            end < tol
        end
    end
    length(hits) == 1 || error("$(length(hits)) subsamplings of the netCDF match atm3d's T (need exactly 1): " *
                               join(last.(hits), "; "))
    println("registration: ", hits[1][4])
    return hits[1][1:3]
end

function main(dir, bfile, out_in, out_truth, kv...)
    H.ensure_mpi()
    o = Dict("mu" => "1", "azimuth_deg" => "0", "tau" => "eos", "lt_min" => "-8", "lt_max" => "1.4", "dlt" => "0.1",
             "table" => "cache/eos_table_fullz.h5")
    for a in kv
        k, v = split(a, "="; limit = 2)
        haskey(o, k) || error("unknown key $k"); o[k] = v
    end
    μ = parse(Float64, o["mu"]); φ = parse(Float64, o["azimuth_deg"])
    o["tau"] in ("eos", "multi3d") || error("tau = eos or multi3d")
    m3d = o["tau"] == "multi3d"
    m3d && μ != 1 && error("tau = multi3d is the run's vertical τ500: mu must be 1")
    nt = round(Int, (parse(Float64, o["lt_max"]) - parse(Float64, o["lt_min"])) / parse(Float64, o["dlt"])) + 1
    lt = [round(parse(Float64, o["lt_min"]) + (k - 1) * parse(Float64, o["dlt"]); digits = 10) for k in 1:nt]
    v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
    isempty(H.option_errors(v)) || error(join(H.option_errors(v), "; "))
    t = H.slant(v)
    tab = H.load_table(o["table"])

    # the run's atmosphere and the netCDF on the run's columns, ordered x, y increasing: Julia (z, y, x)
    nx, ny, nz, z = multi3d_mesh(dir)
    Tm = read_f32(joinpath(dir, "atm3d"), nx, ny, nz, 12)[:, :, :, 2]
    bf = h5open(bfile, "r")
    zb = vec(read(bf["z"])); xb = Float64.(read(bf["x"])); yb = Float64.(read(bf["y"]))
    length(zb) >= nz && maximum(abs, zb[1:nz] .- z) < 1e-5 * maximum(abs, z) ||
        error("the run's z is not the netCDF's top $nz levels")
    rd(nm) = Float64.(bf[nm][1:nz, :, :, 1])                         # Julia view of the netCDF: (z, y, x, t)
    Tb = rd("temperature")
    sx, sy, swap = register(Tm, Tb, (nz ÷ 4, nz ÷ 2, 3nz ÷ 4))
    swap && error("the run's x is the netCDF's y: B would need its components exchanged (not implemented)")
    px = sortperm(sx); py = sortperm(sy)                            # run index order with netCDF x, y increasing
    cx = sx[px]; cy = sy[py]
    on_cols(A) = A[:, cy, cx]
    x = xb[cx]; y = yb[cy]; NX = nx; NY = ny; dx = x[2] - x[1]; dy = y[2] - y[1]
    lT = log.(permutedims(Float64.(Tm[px, py, :]), (3, 2, 1)))      # the run's T, (z, y, x)
    lP = log.(on_cols(rd("gas_pressure"))); lr = log.(on_cols(rd("density"))); lne = log.(on_cols(rd("electron_density")))
    Bx = on_cols(rd("B_x")); By = on_cols(rd("B_y")); Bz = on_cols(rd("B_z"))
    close(bf)
    @printf("grid %d × %d × %d, dx %.2f km, dy %.2f km; μ %.4f (θ %.2f°), φ %.1f°, t = (%.4f, %.4f), max shift %.1f cells; τ from %s; axis %.2f … %.2f (%d)\n",
            nx, ny, nz, dx / 1e3, dy / 1e3, μ, acosd(μ), φ, t..., maximum(abs, z) * hypot(t...) / dx, o["tau"], lt[1], lt[end], nt)

    chi = similar(lT); nbad = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(lT)
        local lc                                                    # per task: a function-level name would be shared
        _, _, lc, _ = H.state(tab, lT[idx], lP[idx])
        isfinite(lc) || (Threads.atomic_add!(nbad, 1); lc = -700.0)
        chi[idx] = exp(lc)
    end
    println("χ500 outside the table at $(nbad[]) points")
    par = H.slab_par(MPI.COMM_WORLD, NX, NY, nz)
    S(A) = H.shear_levels(par, A, z, dx, dy, t, 1.0)
    sχ = S(chi); sT = S(lT); sP = S(lP); sr = S(lr); sne = S(lne); sBx = S(Bx); sBy = S(By); sBz = S(Bz)
    println("sheared")
    LM = m3d ? permutedims(log10.(Float64.(read_f32(joinpath(dir, "H", "tau500"), nx, ny, nz)[px, py, :])), (3, 2, 1)) :
               nothing                                              # (z, y, x)
    T_in = fill(NaN, nt, NY, NX); bx = similar(T_in); by = similar(T_in); bz = similar(T_in)
    zt = similar(T_in); uP = similar(T_in); ur = similar(T_in); une = similar(T_in); z1 = fill(NaN, NY, NX); z1eos = fill(NaN, NY, NX)
    nmiss = Threads.Atomic{Int}(0); nmono = Threads.Atomic{Int}(0); ntop = Threads.Atomic{Int}(0); ltop = fill(NaN, NY, NX)
    Threads.@threads for i in 1:NX
        tc = zeros(nz)
        for j in 1:NY
            H.tau_column!(tc, @view(sχ[:, j, i]), z; mu = μ)
            z1eos[j, i] = H.z_tau1(tc, z)
            local lτ
            if LM === nothing
                lτ = log10.(max.(tc, 1e-300)); z1[j, i] = z1eos[j, i]
            else
                lτ = LM[:, j, i]
                all(diff(lτ) .> 0) || (Threads.atomic_add!(nmono, 1); continue)
                z1[j, i] = interp_at(z, lτ, 0.0)
            end
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
                une[k, j, i] = interp_at(@view(sne[:, j, i]), lτ, lt[k]; top = :extrapolate)
            end
        end
    end
    nmono[] == 0 || error("$(nmono[]) columns of the run's τ500 do not increase downward")
    q = sort(vec(ltop))
    @printf("log10 τ at the top of the rays: p50 %.2f, p99 %.2f, max %.2f; %d rays above the axis minimum %.1f (top extrapolated)\n",
            q[cld(end, 2)], q[ceil(Int, 0.99 * end)], q[end], ntop[], lt[1])
    println("rays not reaching the bottom of the τ axis: $(nmiss[]) of $(NX * NY)")
    nmiss[] == 0 || error("the τ axis bottom is not reached by every ray")
    if LM !== nothing                                               # registration check against the table's own τ
        ok = isfinite.(z1) .& isfinite.(z1eos)
        @printf("z(τ=1) not found: MULTI3D %d, eos %d columns\n", count(!isfinite, z1), count(!isfinite, z1eos))
        count(!isfinite, z1) == 0 || error("the run's τ500 does not reach 1 in every column")
        d = z1[ok] .- z1eos[ok]; r = cor(z1[ok], z1eos[ok])
        @printf("z(τ=1) MULTI3D − eos over %d columns: median %.2f km, rms about the median %.2f km, corr %.4f\n",
                count(ok), median(d) / 1e3, sqrt(mean((d .- median(d)) .^ 2)) / 1e3, r)
        r >= 0.9 || error("the run's z(τ=1) does not follow the table's own (corr $r): misregistered τ500?")
    end
    src = abspath(bfile)
    tsrc = LM === nothing ? "eos: Tau.tau_column! with chi500 of $(o["table"]) ($(H.eos_model(tab)))" :
                            "multi3d: " * abspath(joinpath(dir, "H", "tau500"))
    gsrc = "MULTI3D run " * abspath(dir) * " (T from atm3d); B, P, rho from the netCDF columns x[$(cx[1]):2:$(cx[end])], y[$(cy[1]):2:$(cy[end])] (1-based)"
    h5open(out_in, "w") do f
        f["log10tau500"] = lt
        f["x"] = x; attributes(f["x"])["units"] = "m"
        f["y"] = y; attributes(f["y"])["units"] = "m"
        f["temperature"] = T_in; f["B_x"] = bx; f["B_y"] = by; f["B_z"] = bz     # Julia (τ, y, x) = h5py (x, y, τ)
        a = attributes(f)
        a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2]
        a["source"] = src; a["tau_source"] = tsrc; a["grid"] = gsrc; a["gravity_m_s2"] = 274.0
        a["created_by"] = "scripts/inputs/make_tau_input.jl"
        a["description"] = "one line of sight per column (x, y at z = 0) on log10 tau500 along the line of sight; B local frame"
    end
    h5open(out_truth, "w") do f
        p(A) = permutedims(A, ndims(A):-1:1)                       # Julia (x, y, τ): h5py (τ, y, x) as the run outputs
        f["truth/z_m"] = p(zt); f["truth/lnP"] = p(uP); f["truth/lnrho"] = p(ur); f["truth/lnne"] = p(une); f["truth/T"] = p(T_in)
        f["truth/z_tau1_m"] = p(z1)
        f["logtau500"] = lt; f["x_m"] = x; f["y_m"] = y
        a = attributes(f)
        a["view_mu"] = μ; a["view_azimuth_deg"] = φ; a["view_tx"] = t[1]; a["view_ty"] = t[2]
        a["source"] = src; a["tau_source"] = tsrc; a["grid"] = gsrc; a["created_by"] = "scripts/inputs/make_tau_input.jl"
        a["top_extrapolated_rays"] = ntop[]
    end
    H.free_parallel!(par)
    println("wrote $out_in and $out_truth")
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
