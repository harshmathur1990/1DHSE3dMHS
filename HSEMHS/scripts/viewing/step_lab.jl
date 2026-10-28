#=
Step lab for mhs-poisson-ltau with the column-block preconditioner (design/tau/SOLVER_TIME_DESIGN.md D3b): where the
Gauss–Newton step exceeds the ln P cap, and how Levenberg–Marquardt damping changes it.

usage: mpiexecjl -n R julia --project=. -t T scripts/viewing/step_lab.jl INPUT.nc MU AZIMUTH_DEG GEOMETRY LOGTAU_CUT_LOS BALANCE STEPS
STEPS steps (D2 settings, block preconditioner) give the state; then for damping μ in 0, 0.01, 0.1, 1, 10, 100 the CG
step (η = 0.1 and 1e-3): the cap scale sc = min(1, 2/max|δu|, 3e5 m/max|δc|), the predicted decrease, and the
distribution of |δu| over levels (the levels and columns of the largest entries).
=#
using HSEMHS
using MPI
using LinearAlgebra
using Printf

const H = HSEMHS
const PT = H.PoissonTau

function main(path, mu_s, az_s, geom, cut_s, bal, steps_s)
    H.ensure_mpi()
    v = H.ViewOptions(mu = parse(Float64, mu_s), azimuth_deg = parse(Float64, az_s), geometry = Symbol(geom))
    cut = parse(Float64, cut_s); balance = Symbol(bal); nsteps = parse(Int, steps_s)
    t0 = time()
    NX, NY, nt = PT.tau_input_size(path)
    par = H.slab_par(NX, NY, nt)
    logf(s) = H.rprint(par, @sprintf("[%7.1fs] %s", time() - t0, s))
    tab = H.load_table("cache/eos_table_fullz.h5")
    lt, x, y, T, Bx, By, Bz = PT.read_tau_input(par, path)
    lT = log.(T); g = 274.0; dx = x[2] - x[1]; dy = y[2] - y[1]
    BLAS.set_num_threads(1)
    lnP, z0, _, _, _, nbad = PT.column_hse_tau(par, lt, lT, Bx, By, Bz, tab, g; align = "ptot", mu = v.mu)
    nbad == 0 || error("column HSE failed")
    u, z, Φ0, Φ, s, _, _ = PT.solve_tau(par, tab, g, lt, lT, Bx, By, Bz, lnP, z0, dx, dy; logtau_cut = cut,
                                        max_steps = nsteps, mu = v.mu, slant = H.slant(v), stop = :phi, rtol = 0.0,
                                        balance = balance, lm = :gain, forcing = :ew, precond = :block,
                                        stationarity = false)
    logf(@sprintf("state after %d steps: Φ/Φ₀ %.3e", s, Φ / Φ0))
    for pc in (:block, :fourier)
        pr, c = PT.tp_setup(par, tab, g, lt, lT, Bx, By, Bz, z, dx, dy; logtau_cut = cut, mu = v.mu, slant = H.slant(v),
                            balance = balance, precond = pc)
        PT.evaluate!(pr, u, c); PT.tp_weights!(pr, g, dx); Φs, _ = PT.evaluate!(pr, u, c)
        rhs = zeros(pr.n1, pr.ny, pr.nx)
        PT.JTop!(pr, rhs, pr.Rx .* pr.wc, pr.Ry .* pr.wc, pr.Rz .* vcat(pr.wf, 0.0)); rhs .*= -1
        ereg0 = copy(pr.ereg); Dlm = ereg0 ./ 1e-2; δ = similar(rhs)
        for η in (0.1, 1e-3), μ in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
            pc == :fourier && μ > 0 && η > 0.01 && continue
            pr.ereg = ereg0 .+ μ .* Dlm
            PT.setprec!(pr)
            its, rel = PT.pcg!(PT.Aop!, PT.Minv!, par, pr, δ, rhs; tol = η, maxit = 600)
            pred = H.gdot(par, δ, rhs) / Φs
            du = abs.(δ[2:end, :, :]); mu_ = H.gmax(par, maximum(du)); mc = H.gmax(par, maximum(abs, δ[1, :, :]))
            sc = min(1.0, 2.0 / mu_, 3e5 / mc)
            nbig = H.gcount(par, count(>(2.0), du))
            lev = [H.gmax(par, maximum(@view du[m, :, :])) for m in 1:pr.n1-1]
            top = sortperm(lev, rev = true)[1:3]
            logf(@sprintf("%-7s η %.0e μ %-6g CG %3d (rel %.1e) pred %.3f | max|δu| %.2f max|δc| %.1f km sc %.3f | %d entries |δu|>2; largest at log τ %s (%s)",
                          pc, η, μ, its, rel, pred, mu_, mc / 1e3, sc, nbig,
                          join([@sprintf("%.2f", lt[pr.kc+m]) for m in top], ", "), join([@sprintf("%.1f", lev[m]) for m in top], ", ")))
        end
    end
end

main(ARGS...)
MPI.Finalized() || MPI.Finalize()
