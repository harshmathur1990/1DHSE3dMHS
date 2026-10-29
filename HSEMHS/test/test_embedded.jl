# Embedded (in-memory) API on a single rank: equivalence with the standalone drivers, layout, units, validity checks,
# option validation, buffers and allocations. Multi-rank checks are in mpi/mpi_embedded.jl.
include("embedded_helpers.jl")

@testset "embedded API (single rank)" begin
    comm = MPI.COMM_WORLD
    tab = H.load_table(ETABLE)
    blas0 = BLAS.get_num_threads()

    @testset "equivalence with the stored standalone results ($mode, $tag)" for mode in H.EMBEDDED_MODES, tag in ("24x18", "11x12")
        fx = load_fixture(mode, tag, :)
        o = ref_options(mode, fx)
        ws = H.prepare_embedded(comm, mode; axis = o.axis, x = o.x, y = o.y, eos = tab, g = o.g, column = o.column,
                                poisson = o.poisson)
        # the standalone mhs-poisson-ltau driver raises BLAS to min(32, nthreads) threads around its solve; the result of
        # that iterative solve depends on the BLAS thread count, so reproduce the driver's setting to compare bitwise
        nb = mode == :mhs_poisson_ltau ? min(32, Threads.nthreads()) : 1
        BLAS.set_num_threads(nb)
        res = try
            H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
        finally
            BLAS.set_num_threads(blas0)
        end
        @test res.status.valid
        # column-hse-z: the slab path against the 2-D-block column_hse output (different decomposition, same kernels)
        @test ref_error(res, load_ref(ref_name(mode), tag, :; path = ref_file(mode, tag))) <= 1e-12
        if H.poisson_mode(mode)
            @test res.status.steps > 0 && isfinite(res.status.gn_decrement) && res.status.gn_decrement >= 0
            @test res.status.reason in (:converged, :max_steps, :no_descent)
        end
        H.free_embedded!(ws)
    end

    @testset "layout: (vertical, y, x) on a non-square grid with independent axis variation" begin
        NX, NY, NZ = 7, 5, 40
        z = collect(range(2.0e6, -0.4e6; length = NZ))
        x = collect(0:NX-1) .* 5.0e4; y = collect(0:NY-1) .* 4.0e4 .+ 1.0e3
        T = [6000.0 + 3000.0 * (k / NZ)^2 + 40.0 * j + 7.0 * i for k in 1:NZ, j in 1:NY, i in 1:NX]
        ws = H.prepare_embedded(comm, :column_hse_z; axis = z, x = x, y = y, eos = tab, g = 274.0,
                                column = H.ColumnZOptions(anchor = :top, P_ref = 0.1))
        res = H.solve_embedded!(ws, log.(T))
        @test res.status.valid
        o = H.hse_opts(ws.column, 274.0)
        for (i, j) in ((1, 1), (NX, 2), (3, NY))
            X = log.(T[:, j, i]); u = zeros(NZ); m = zeros(NZ)
            _, _, ak = H.hse_column!(u, m, X, z, tab, o)
            P = zeros(NZ); r = zeros(NZ); n = zeros(NZ); c = zeros(NZ); t = zeros(NZ); s = zeros(UInt8, NZ)
            _, zt = H.column_state!(P, r, n, c, t, s, X, u, z, ak, tab, o)
            @test res.Pgas[:, j, i] == P && res.rho[:, j, i] == r && res.ne[:, j, i] == n && res.tau500[:, j, i] == t
            @test res.z_tau1[j, i] == zt
        end
        @test all(res.z[:, j, i] == z for j in 1:NY, i in 1:NX)
        # χ₅₀₀ in m⁻¹ and z in m: τ is the trapezoid of χ over z plus the exp-tail top closure
        χ = res.chi500[:, 2, 3]; τ = res.tau500[:, 2, 3]
        Hc = (z[1] - z[2]) / log(χ[2] / χ[1])
        τref = cumsum(vcat(χ[1] * Hc, [0.5 * (χ[k] + χ[k+1]) * (z[k] - z[k+1]) for k in 1:NZ-1]))
        @test maximum(abs.(τ .- τref) ./ τref) <= 1e-12
        @test res.logtau500 == log10.(res.tau500)
        H.free_embedded!(ws)
    end

    @testset "option and input validation" begin
        fx = load_z_fixture("11x12", :)
        z = fx.z; NZ = length(z)
        col = H.ColumnZOptions(P_ref = fx.P_ref)
        prep(; kw...) = H.prepare_embedded(comm, :mhs_poisson_z; axis = z, x = fx.x, y = fx.y, eos = tab, g = fx.g,
                                           column = col, kw...)
        # retained levels below the cut: top level fine, bottom and next-to-bottom rejected, three levels accepted
        @test isempty(H.option_errors(H.PoissonZOptions(z_cut_m = z[1]), z))
        @test !isempty(H.option_errors(H.PoissonZOptions(z_cut_m = z[NZ]), z))
        @test !isempty(H.option_errors(H.PoissonZOptions(z_cut_m = z[NZ-1]), z))
        @test isempty(H.option_errors(H.PoissonZOptions(z_cut_m = z[NZ-2]), z))
        @test !isempty(H.option_errors(H.PoissonZOptions(z_cut_m = z[NZ] - 1.0), z))
        lt = collect(range(-8.0, 1.4; length = 20))
        @test !isempty(H.option_errors(H.PoissonTauOptions(logtau_cut = lt[1]), lt))        # cut on the top level
        @test isempty(H.option_errors(H.PoissonTauOptions(logtau_cut = lt[2]), lt))
        @test !isempty(H.option_errors(H.PoissonTauOptions(logtau_cut = lt[end-1]), lt))    # two solved levels
        @test !isempty(H.option_errors(H.PoissonTauOptions(logtau_cut = 3.0), lt))          # outside the axis
        # solver controls: valid choices pass, others are rejected; Marquardt damping needs column blocks
        @test isempty(H.option_errors(H.PoissonTauOptions(logtau_cut = lt[2], lm = :gain, forcing = :ew, precond = :hybrid,
                                                          damping = :marquardt), lt))
        for bad in ((lm = :nope,), (forcing = :nope,), (precond = :nope,), (damping = :nope,), (cg_cap = 0,),
                    (eta_min = 0.2, eta_max = 0.1), (eta_max = 1.0,), (klow = -1,), (damping = :marquardt,))
            @test !isempty(H.option_errors(H.PoissonTauOptions(; logtau_cut = lt[2], bad...), lt))
        end
        @test !isempty(H.option_errors(H.ColumnZOptions(), z))                              # P_ref missing
        @test !isempty(H.option_errors(H.ColumnZOptions(P_ref = 1e5, anchor = :height, z_ref = 9e6), z))
        @test !isempty(H.option_errors(H.ColumnTauOptions(align = "median"), lt))
        @test_throws H.InvalidInput prep(poisson = H.PoissonZOptions(z_cut_m = z[NZ]))
        @test_throws H.InvalidInput prep(poisson = nothing)
        @test_throws H.InvalidInput prep(poisson = H.PoissonZOptions(), eos = "/nonexistent/table.h5")
        @test_throws H.InvalidInput H.prepare_embedded(comm, :column_hse_z; axis = reverse(z), x = fx.x, y = fx.y,
                                                       eos = tab, g = fx.g, column = col)
        @test_throws H.InvalidInput H.prepare_embedded(comm, :column_hse_z; axis = z, x = fx.x .^ 1.1, y = fx.y,
                                                       eos = tab, g = fx.g, column = col)
        ws = prep(poisson = H.PoissonZOptions())
        @test_throws H.InvalidInput H.solve_embedded!(ws, fx.lnT[:, :, 1:end-1], fx.Bx, fx.By, fx.Bz)
        @test_throws H.InvalidInput H.solve_embedded!(ws, fx.lnT)                            # Poisson needs B
        bad = copy(fx.lnT); bad[3, 2, 1] = NaN
        @test_throws H.InvalidInput H.solve_embedded!(ws, bad, fx.Bx, fx.By, fx.Bz)
        H.free_embedded!(ws)
    end

    @testset "physics failures come back in the status" begin
        fx = load_z_fixture("11x12", :)
        ws = H.prepare_embedded(comm, :mhs_poisson_z; axis = fx.z, x = fx.x, y = fx.y, eos = tab, g = fx.g,
                                column = H.ColumnZOptions(P_ref = fx.P_ref), poisson = H.PoissonZOptions())
        hot = copy(fx.lnT); hot[:, 3:4, 2:3] .= log(5.0e7)                                  # far outside the table
        res = H.solve_embedded!(ws, hot, fx.Bx, fx.By, fx.Bz)
        @test !res.status.valid && res.status.reason == :eos_out_of_envelope && res.status.status == H.eos_failure
        # the field checker on injected defects (non-finite, non-positive, folded columns)
        res = H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
        @test res.status.valid
        for (defect, want) in ((r -> r.ne[5, 2, 2] = NaN, :nonfinite), (r -> r.rho[7, 1, 1] = -1.0, :nonpositive),
                               (r -> r.tau500[9, 3, 3] = 0.5 * r.tau500[8, 3, 3], :folded),
                               (r -> r.z[9, 3, 3] = r.z[8, 3, 3] + 1.0, :folded))
            snap = H.snapshot(res); defect(snap); ws2 = ws; old = ws.res; ws.res = snap
            st = H.EquilibriumStatus(); H.check_fields!(ws, st)
            @test !st.valid && st.reason == want
            ws.res = old
        end
        # equal heights of adjacent levels are invalid: heights must strictly decrease downwards
        snap = H.snapshot(res); snap.z[9, 3, 3] = snap.z[8, 3, 3]; old = ws.res; ws.res = snap
        st = H.EquilibriumStatus(); H.check_fields!(ws, st)
        @test !st.valid && st.reason == :folded
        ws.res = old
        H.free_embedded!(ws)
        tau = load_tau_fixture("11x12", :)
        ws = H.prepare_embedded(comm, :column_hse_ltau; axis = tau.lt, x = tau.x, y = tau.y, eos = tab, g = 274.0,
                                column = H.ColumnTauOptions())
        hot = copy(tau.lnT); hot[:, 2, 2] .= log(5.0e7)
        res = H.solve_embedded!(ws, hot, tau.Bx, tau.By, tau.Bz)
        @test !res.status.valid && res.status.reason == :eos_out_of_envelope
        res = H.solve_embedded!(ws, tau.lnT, tau.Bx, tau.By, tau.Bz)                       # recovers on the next solve
        @test res.status.valid
        H.free_embedded!(ws)
    end

    @testset "already balanced start and the stationarity measure" begin
        NX, NY, NZ = 6, 5, 60
        z = collect(range(1.5e6, -0.4e6; length = NZ))
        x = collect(0:NX-1) .* 5.0e4; y = collect(0:NY-1) .* 5.0e4
        T = [5000.0 + 4000.0 * (1 - (k - 1) / (NZ - 1))^3 for k in 1:NZ, j in 1:NY, i in 1:NX]   # horizontally uniform
        B0 = zeros(NZ, NY, NX)
        mk(rms_atol, steps) = H.prepare_embedded(comm, :mhs_poisson_z; axis = z, x = x, y = y, eos = tab, g = 274.0,
                                                 column = H.ColumnZOptions(anchor = :top, P_ref = 0.1),
                                                 poisson = H.PoissonZOptions(z_cut_m = 1.0e6, rms_atol = rms_atol,
                                                                             max_steps = steps))
        ws = mk(1e-3, 30)
        res = H.solve_embedded!(ws, log.(T), B0, B0, B0)
        @test res.status.valid && res.status.reason == :converged && res.status.steps == 0
        @test res.status.gn_decrement == 0.0 && isfinite(res.status.rms0) && res.status.rms0 <= 1e-3
        kc = res.status.stats["cut_level"]                # zero steps: the solved region keeps the start (exp∘log) and
        @test relerr(res.Pgas[kc:end, :, :], res.Pgas_start[kc:end, :, :]) <= 1e-14   # above z_cut it is re-integrated
        @test relerr(res.Pgas, res.Pgas_start) <= 1e-2
        H.free_embedded!(ws)
        # a magnetised fixture: more steps → smaller Gauss–Newton decrement
        fx = load_z_fixture("11x12", :)
        dec = Float64[]
        for steps in (1, 30)
            w = H.prepare_embedded(comm, :mhs_poisson_z; axis = fx.z, x = fx.x, y = fx.y, eos = tab, g = fx.g,
                                   column = H.ColumnZOptions(P_ref = fx.P_ref),
                                   poisson = H.PoissonZOptions(max_steps = steps, cg_tol = 1e-6))
            r = H.solve_embedded!(w, fx.lnT, fx.Bx, fx.By, fx.Bz)
            push!(dec, r.status.gn_decrement)
            steps == 1 && @test r.status.reason == :max_steps && r.status.status == H.max_iter
            H.free_embedded!(w)
        end
        @test all(isfinite, dec) && dec[2] < dec[1]
    end

    @testset "result buffers and snapshot" begin
        tau = load_tau_fixture("11x12", :)
        ws = H.prepare_embedded(comm, :column_hse_ltau; axis = tau.lt, x = tau.x, y = tau.y, eos = tab, g = 274.0,
                                column = H.ColumnTauOptions())
        r1 = H.solve_embedded!(ws, tau.lnT, tau.Bx, tau.By, tau.Bz)
        s1 = H.snapshot(r1); P1 = copy(r1.Pgas)
        r2 = H.solve_embedded!(ws, tau.lnT .+ 0.01, tau.Bx, tau.By, tau.Bz)
        @test r2 === r1                                   # the same workspace buffers ...
        @test r1.Pgas != P1                               # ... now hold the second solve
        @test s1.Pgas == P1                               # the snapshot does not
        H.free_embedded!(ws)
    end

    @testset "allocations per warmed-up solve" begin
        # budget in units of one local volume of Float64 (nk·ny·nx·8 bytes); see reports/embedding_api_gate.md. The
        # threaded kernels allocate per-thread scratch (column chunks, FFT line buffers), so the budget is a + b·threads,
        # fitted on the 24x18 fixtures at 1, 4, 8 and 16 threads (Julia 1.13.1, 29 Oct 2026; at 4 threads the gate's
        # 1.8 / 4.7 / 1114 / 20282) and × 1.5; the Poisson kernels allocate temporaries every CG iteration
        ab = Dict(:column_hse_z => (1.1, 0.14), :column_hse_ltau => (4.5, 0.01), :mhs_poisson_z => (620.0, 90.0),
                  :mhs_poisson_ltau => (17300.0, 520.0))
        budgets = Dict(m => 1.5 * (a + b * Threads.nthreads()) for (m, (a, b)) in ab)
        for mode in H.EMBEDDED_MODES
            fx = load_fixture(mode, "24x18", :)
            o = ref_options(mode, fx)
            ws = H.prepare_embedded(comm, mode; axis = o.axis, x = o.x, y = o.y, eos = tab, g = o.g, column = o.column,
                                    poisson = o.poisson)
            for _ in 1:2
                H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)
            end
            a = minimum(@allocated(H.solve_embedded!(ws, fx.lnT, fx.Bx, fx.By, fx.Bz)) for _ in 1:3)   # min: no one-off compiles
            vol = 8 * length(fx.lnT)
            c = a / vol
            @info "embedded allocations" mode bytes = a volumes = round(c; digits = 1)
            @test c <= budgets[mode]
            H.free_embedded!(ws)
        end
    end

    @test BLAS.get_num_threads() == blas0                 # loading and solving changed no global thread setting
end
