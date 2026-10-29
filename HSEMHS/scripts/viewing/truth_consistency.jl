#=
How far the truth of a synthetic τ-grid input is from the solver's own model (design/CODEX_REVIEW_solver_time_round3_2026-10-29.md,
priority experiment step 1). The generator (make_slanted_tau.jl) shears χ500 from the Bifrost T, P on the geometric
grid, integrates τ along the ray and interpolates T, P, ρ, z onto the τ axis; the solver instead takes χ from the
interpolated T and the solved P and builds z by the trapezoid of dz = μ τ/χ d ln τ. This script measures, on the truth:

  eos     ln ρ_EOS(T, P_truth) − ln ρ_truth (the snapshot density against the solver's EOS)
  tauz    the solver's height step −Δln τ (a_k + a_k+1)/2, a = μ τ / χ_EOS(T, P_truth), against the truth's Δz per level,
          and z(τ_LOS = 1) rebuilt by that map down from the truth's cut height against the truth z(τ = 1)
  cut     the fixed boundary: column-HSE ln P at the cut − truth, and the column-HSE heights above the cut relative to
          the cut against the truth's

reported by log τ band and by |B(τ = 1)| class (strong > 500 G, quiet < 100 G); cells with ln P outside the table are
counted separately. Force balance in Cartesian components: scripts/viewing/force_diag.jl on the truth file.

usage: julia --project=. -t N scripts/viewing/truth_consistency.jl TAU_INPUT.nc TRUTH.h5 MU AZIMUTH_DEG LOGTAU_CUT [OUT.h5]
OUT.h5 (optional, h5py (τ, y, x)): eos_dlnrho, tauz_ddz (solver − truth Δz per level, m), z1_rebuilt_err (y, x).
=#
using HSEMHS
using HDF5
using MPI
using Printf
using Statistics

const H = HSEMHS
const PT = H.PoissonTau

function main(path, tpath, mu_s, az_s, cut_s, out = "")
    H.ensure_mpi()
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = :inclined)
    μ = v.mu; cut = parse(Float64, cut_s)
    NX, NY, nt = PT.tau_input_size(path)
    par = H.slab_par(NX, NY, nt); par.nranks == 1 || error("one rank")
    tab = H.load_table("cache/eos_table_fullz.h5"); g = 274.0
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    lT = log.(T)
    tf = h5open(tpath, "r")
    isapprox(read(tf["logtau500"]), lt; rtol = 1e-6) || error("τ axes differ")
    rd(nm) = permutedims(Float64.(read(tf[nm])), (3, 2, 1))           # h5py (τ, y, x) → Julia (τ, y, x)
    uP = rd("truth/lnP"); ur = rd("truth/lnrho"); zt = rd("truth/z_m")
    z1t = permutedims(Float64.(read(tf["truth/z_tau1_m"])), (2, 1))
    close(tf)
    kc = argmin(abs.(lt .- cut)); k0 = argmin(abs.(lt))
    B1 = sqrt.(Bx[k0, :, :] .^ 2 .+ By[k0, :, :] .^ 2 .+ Bz[k0, :, :] .^ 2) .* 1e4
    cls = [("all", trues(NY, NX)), ("strong", B1 .> 500), ("quiet", B1 .< 100)]
    bands = [(-9.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 9.0)]
    @printf("input %s, truth %s; μ %.3f φ %g; cut log τ %.2f (level %d); %d × %d × %d; strong %d, quiet %d columns\n",
            path, tpath, μ, v.azimuth_deg, lt[kc], kc, NY, NX, nt, count(cls[2][2]), count(cls[3][2]))

    # EOS: ln ρ and ln χ at the truth (T, P)
    lrE = fill(NaN, nt, NY, NX); lcE = fill(NaN, nt, NY, NX); nout = Threads.Atomic{Int}(0)
    Threads.@threads for i in 1:NX
        for j in 1:NY, k in 1:nt
            r, _, c, _ = H.state(tab, lT[k, j, i], uP[k, j, i])
            isfinite(r + c) || (Threads.atomic_add!(nout, 1); continue)
            lrE[k, j, i] = r; lcE[k, j, i] = c
        end
    end
    @printf("truth (T, P) outside the EOS table: %d cells (excluded below)\n", nout[])
    dr = lrE .- ur
    rmsf(a) = (b = filter(isfinite, a); isempty(b) ? NaN : sqrt(mean(abs2, b)))
    p99(a) = (b = sort(abs.(filter(isfinite, a))); isempty(b) ? NaN : b[ceil(Int, 0.99 * end)])
    function table(name, A; scale = 1.0, levels = 1:nt)
        println(name)
        for (lo, hi) in bands
            ks = [k for k in levels if lo <= lt[k] < hi]
            isempty(ks) && continue
            row = [@sprintf("%s rms %.3g p99 %.3g", c, rmsf(A[ks, findall(vec(msk))]) * scale,
                            p99(A[ks, findall(vec(msk))]) * scale) for (c, msk) in cls]
            @printf("  log τ [%g, %g): %s\n", lo, hi, join(row, " | "))
        end
    end
    A2(A) = reshape(A, nt, :)
    table("eos: ln ρ_EOS(T, P_truth) − ln ρ_truth", A2(dr))

    # τ–z: the solver's height steps on the truth (T, P) against the truth's
    lτ = lt .* log(10); dl = diff(lτ)
    a = μ .* exp.(reshape(lτ, :, 1, 1) .- lcE)
    dzs = fill(NaN, nt, NY, NX); dzt = fill(NaN, nt, NY, NX)
    for k in 1:nt-1
        @views dzs[k, :, :] .= .-dl[k] ./ 2 .* (a[k, :, :] .+ a[k+1, :, :])
        @views dzt[k, :, :] .= zt[k+1, :, :] .- zt[k, :, :]
    end
    ddz = dzs .- dzt
    table("tauz: solver Δz − truth Δz per level [km]", A2(ddz); scale = 1e-3, levels = 1:nt-1)
    rel = A2(ddz ./ abs.(dzt))
    table("tauz: (solver Δz − truth Δz) / |truth Δz|", rel; levels = 1:nt-1)
    # z(τ = 1) rebuilt down from the truth cut height
    zr = copy(zt[kc, :, :])
    for k in kc:k0-1
        zr .+= dzs[k, :, :]
    end
    e1 = zr .- zt[k0, :, :]
    for (c, msk) in cls
        @printf("z(τ_LOS = 1) rebuilt from the truth cut height − truth: %-6s rms %.2f km, mean %+.2f km, p99 |.| %.2f km\n",
                c, rmsf(e1[msk]) / 1e3, mean(filter(isfinite, e1[msk])) / 1e3, p99(e1[msk]) / 1e3)
    end
    @printf("truth z(τ_LOS = 1) on the axis level %d (log τ %.3f) against truth/z_tau1_m: rms %.2f km\n", k0, lt[k0],
            rmsf(zt[k0, :, :] .- z1t) / 1e3)
    s = e1 .- mean(filter(isfinite, e1)); sq = e1[cls[2][2]]; qq = e1[cls[3][2]]
    @printf("  strong − quiet of that error: %+.2f km (the solver map alone shifts the depression by this much)\n",
            (mean(filter(isfinite, sq)) - mean(filter(isfinite, qq))) / 1e3)

    # the fixed boundary from column HSE
    lnP, zc, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = μ)
    nbad == 0 || error("column HSE failed")
    dPc = lnP[kc, :, :] .- uP[kc, :, :]
    for (c, msk) in cls
        @printf("cut: ln P_col − ln P_truth %-6s rms %.4f mean %+.4f p99 %.4f\n", c, rmsf(dPc[msk]), mean(dPc[msk]), p99(dPc[msk]))
    end
    zrel = (zc[1:kc-1, :, :] .- reshape(zc[kc, :, :], 1, NY, NX)) .- (zt[1:kc-1, :, :] .- reshape(zt[kc, :, :], 1, NY, NX))
    table("cut: column-HSE heights above the cut relative to the cut − truth's [km]", A2(cat(zrel, fill(NaN, nt - kc + 1, NY, NX); dims = 1));
          scale = 1e-3, levels = 1:kc-1)
    dPall = lnP .- uP
    table("column HSE ln P − truth (for scale: the start's distance)", A2(dPall))
    if out != ""
        h5open(out, "w") do f
            p(A) = permutedims(A, ndims(A):-1:1)
            f["eos_dlnrho"] = p(dr); f["tauz_ddz"] = p(ddz); f["z1_rebuilt_err"] = p(e1); f["cut_dlnP"] = p(dPc)
            f["logtau500"] = lt
        end
        println("wrote $out")
    end
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
