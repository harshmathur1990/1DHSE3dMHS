# Cost of adjoint_embedded! against solve_embedded! at the size of the first FFNO inversion (252² columns; log τ −8 … 1.5,
# 96 levels): julia --project=. -t 64 test/scripts/adjoint_benchmark.jl   (from HSEMHS/; one rank)
using HSEMHS, MPI, HDF5, Random, Statistics, Printf
const H = HSEMHS; H.ensure_mpi()
include(joinpath(@__DIR__, "..", "adjoint_helpers.jl"))
tab = H.load_table(ETABLE); comm = MPI.COMM_WORLD
N = 252
tile(a) = repeat(a, 1, cld(N, size(a, 2)), cld(N, size(a, 3)))[:, 1:N, 1:N]
xs = collect(0:N-1) .* 95238.1
function bench(name, mode, axis, lnT, B; column, poisson = nothing, kw = (;))
    ws = H.prepare_embedded(comm, mode; axis = axis, x = xs, y = xs, eos = tab, g = 274.0, column = column, poisson = poisson)
    println("start ", name); flush(stdout)
    H.solve_embedded!(ws, lnT, B...)                                    # compile
    ts = @elapsed r = H.solve_embedded!(ws, lnT, B...)
    tr = @elapsed H.solve_embedded!(ws, lnT, B...; record = true)
    bar = H.EquilibriumCotangent(ws); res = ws.tape.res
    bar.rho .= 1 ./ res.rho; bar.ne .= 1 ./ res.ne; bar.Pgas .= 1 ./ res.Pgas; bar.z .= 1e-5
    H.adjoint_embedded!(ws, bar)                                       # compile
    ta = @elapsed a = H.adjoint_embedded!(ws, bar)
    @printf("%-34s solve %7.2f s  recorded %7.2f s  adjoint %7.2f s  ratio %.2f  status %s\n", name, ts, tr, ta, ta / ts, a.status); flush(stdout)
    H.free_embedded!(ws)
end
lt = collect(range(-8.0, 1.5; step = 0.1))
falc = h5open("/mn/stornext/d9/data/harshm/ffnoml_data/models/zaug/ffnoinversion/en024048_hion_385_inv_halpha/data/initial_atmosphere.h5") do f
    permutedims(read(f["temperature"]), (3, 2, 1))                      # h5py (z, y, x) -> Julia (z, y, x)
end
lnT = log.(falc) .+ 0.05 .* smooth_field(MersenneTwister(1), length(lt), N, N)
Z3 = zeros(size(lnT))
bench("column-hse-ltau (252², 96)", :column_hse_ltau, lt, lnT, (Z3, Z3, Z3); column = H.ColumnTauOptions(align = "ptot"))
bench("column-hse-logm (252², 96)", :column_hse_logm, lt .+ 1.0, lnT, (Z3, Z3, Z3); column = H.ColumnTauOptions(align = "ptot"))
fz = load_z_fixture("11x12", Colon())
bench("column-hse-z (252², 177)", :column_hse_z, fz.z, tile(fz.lnT), (zeros(length(fz.z), N, N) for _ in 1:3) |> collect |> Tuple;
      column = H.ColumnZOptions(anchor = :bottom, P_ref = fz.P_ref))
ft = load_tau_fixture("11x12", Colon())
bench("mhs-poisson-ltau (252², 95) polish 3", :mhs_poisson_ltau, ft.lt, tile(ft.lnT), (tile(ft.Bx), tile(ft.By), tile(ft.Bz));
      column = H.ColumnTauOptions(align = "ptot"), poisson = H.PoissonTauOptions(polish = 3))
bench("mhs-poisson-z (252², 177) polish 3", :mhs_poisson_z, fz.z, tile(fz.lnT), (tile(fz.Bx), tile(fz.By), tile(fz.Bz));
      column = H.ColumnZOptions(anchor = :bottom, P_ref = fz.P_ref), poisson = H.PoissonZOptions(z_cut_m = 1.0e6, max_steps = 30, polish = 3))
