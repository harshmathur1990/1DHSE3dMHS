# Poisson-type pressure step (FIRTEZ-like, done as a weighted least-squares problem in u = ln P) on a height grid.
# The solver itself is in poisson_core.jl; this script reads a column-HSE file and writes the result.
#
# T and B are fixed (column-HSE input); only the pressure changes. Below z_cut, u = ln P minimises
#     Φ = Σ_centres w H̄² (R_x² + R_y²) + Σ_z-faces h H̄² R_z²,     R = F/P = −∇u − ẑ/H(T, P) + J×B/P,
# with H = P/(ρ g) and H̄ the layer median of H in the starting state (fixed), so R H̄ ≈ F/(ρ g) layer by layer. In u the
# residual is measured against the local pressure: lowering P cannot reduce it (the P-space form with a fixed scale S
# rewards collapsing P where column HSE leaves |∇ₕP| ≫ ρg), and P stays positive. Discretisation as
# scripts/m5_bifrost/force_split.py: cell-centred B, centred periodic x, y differences, R_z on interior z-faces.
# Gauss–Newton; each step solves the normal equations (a weighted Poisson equation in δu plus local terms) by PCG,
# preconditioned by the Laplacian part with layer-median coefficients: FFT in x, y and a tridiagonal solve in z per mode.
# Steps are capped at |δu| ≤ 2 per node and backtracked on Φ. The bottom-plane mean of u is held fixed.
# Above z_cut each column is integrated in hydrostatic equilibrium from the solved pressure at z_cut (FIRTEZ likewise
# drops the magnetic term above the lines' sensitivity). τ₅₀₀ from the EOS table with the exp_tail top closure.
#
# usage (from HSEMHS/): julia --project=. -t 64 scripts/poisson/poisson_step.jl COLHSE.h5 OUT.h5 [z_cut_Mm = 3.0]
#   ENV MHS_CROP = n : only the first n × n columns (test runs; x, y treated as periodic regardless)
#   ENV MHS_POISSON_OUTER = maximum Gauss–Newton steps (default 30); stop when ΔΦ/Φ < 1e-4
# OUT.h5 has the layout of an MHS checkpoint (/atmosphere, /coords), so force_split.py and m5h_isotau*.py read it.
include(joinpath(@__DIR__, "poisson_core.jl"))

function main()
    infile, outfile = ARGS[1], ARGS[2]
    zcut = parse(Float64, get(ARGS, 3, "3.0")) * 1e6
    crop = parse(Int, get(ENV, "MHS_CROP", "0"))
    maxouter = parse(Int, get(ENV, "MHS_POISSON_OUTER", "30"))
    FFTW.set_num_threads(1)              # FFTW's threaded execution segfaults under Julia 1.13 (spawn_apply); 0.4 s per pair anyway

    # ---- input: column HSE output, arrays (z, y, x), z top first ----
    P0, T, Bx, By, Bz, z, x, y = h5open(infile, "r") do f
        c = crop > 0 ? (1:crop) : Colon()
        rd(nm) = Float64.(read(f[nm])[:, c, c])
        rd("output/Pgas"), rd("input/temperature"), rd("input/B_x"), rd("input/B_y"), rd("input/B_z"),
            read(f["input/z"]), read(f["input/x"])[c], read(f["input/y"])[c]
    end
    nz, ny, nx = size(P0)
    @assert length(z) == nz && z[1] > z[end] "expected (z, y, x) arrays with z top first"
    dx, dy = x[2] - x[1], y[2] - y[1]
    tab = H.load_table(TAB)
    lnT = log.(T)
    println(stamp(), " input $infile: $(nz)×$(ny)×$(nx), z $(round(z[1]/1e6, digits = 3)) … $(round(z[end]/1e6, digits = 3)) Mm, ",
            "dx $(round(dx/1e3, digits = 2)) km, threads $(Threads.nthreads()), z_cut $(zcut/1e6) Mm, EOS $TAB")

    # ---- Lorentz force (fixed) ----
    Lx, Ly, Lz = lorentz(Bx, By, Bz, z, dx, dy)
    if get(ENV, "MHS_NO_LORENTZ", "0") == "1"                    # diagnostic: pure hydrostatics with the same T
        fill!(Lx, 0.0); fill!(Ly, 0.0); fill!(Lz, 0.0)
    end
    println(stamp(), " Lorentz force computed")

    sol = solve_pressure(tab, lnT, Lx, Ly, Lz, P0, z, dx, dy; zcut = zcut, maxouter = maxouter)
    t_solve = time() - t_start
    Pfull, Φ0, Φ, u0, u = sol.P, sol.Φ0, sol.Φ, sol.u0, sol.u
    rhofull, tau, zt1 = tau_columns(tab, lnT, Pfull, z)
    lt_top = log10.(tau[1, :, :])
    zt1_in = h5open(f -> read(f["diagnostics/z_tau500_1"]), infile)
    crop > 0 && (zt1_in = zt1_in[1:crop, 1:crop])
    @printf("%s done: solve %.1f s; ln(P/P_colhse) below z_cut p1/p50/p99 %.3f / %.3f / %.3f; z(τ=1) median %.1f km (column HSE %.1f km); log10 τ500 at z_top min/median/max %.2f / %.2f / %.2f\n",
            stamp(), t_solve, quantile(vec(u .- u0), [0.01, 0.5, 0.99])..., median(filter(isfinite, zt1)) / 1e3,
            median(filter(isfinite, zt1_in)) / 1e3, minimum(lt_top), median(lt_top), maximum(lt_top))

    # ---- output in the MHS checkpoint layout: Julia (x, y, z) arrays ----
    xyz(A) = permutedims(A, (3, 2, 1))
    h5open(outfile, "w") do f
        a = create_group(f, "atmosphere"); c = create_group(f, "coords")
        a["Pgas"] = xyz(Pfull); a["rho"] = xyz(rhofull); a["T"] = xyz(T); a["tau500"] = xyz(tau)
        a["Bx_centre"] = xyz(Bx); a["By_centre"] = xyz(By); a["Bz_centre"] = xyz(Bz)
        a["z_tau1_m"] = permutedims(zt1, (2, 1))
        c["x_m"] = x; c["y_m"] = y; c["z_m"] = z
        attributes(f)["accepted"] = 0
        attributes(f)["kind"] = "poisson_step"
        attributes(f)["input"] = infile
        attributes(f)["z_cut_m"] = zcut
        attributes(f)["eos_table"] = TAB
        attributes(f)["phi_start"] = Φ0; attributes(f)["phi_end"] = Φ
        attributes(f)["solve_seconds"] = t_solve
    end
    println(stamp(), " wrote $outfile")
end

main()
