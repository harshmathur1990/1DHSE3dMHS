# Small structured column_hse run with the Wittmann table, for thread/rank invariance tests.
# usage: [mpiexec -n R] julia --project -t T colhse_small.jl OUT.h5
using HSEMHS, MPI
const H = HSEMHS
out = ARGS[1]
NX, NY, NZ = 24, 20, 60
par = H.init_parallel(NX, NY, NZ)
x = collect((0:NX-1) .* 50.0e3); y = collect((0:NY-1) .* 50.0e3)
z = collect(range(2.4e6, -0.5e6; length = NZ))
grid = H.Grid(H.build_axis("x", x; periodic = true), H.build_axis("y", y; periodic = true), H.build_axis("z", z))
T = H.newfield(par)
b = par.block
for k in 1:NZ, j in 1:b.ny, i in 1:b.nx
    gi, gj = b.i0 + i - 1, b.j0 + j - 1
    zz = z[k] / 1e6
    ztr = 1.9 + 0.3 * sin(2π * gi / NX) * cos(2π * gj / NY)          # TR height [Mm]
    Tc = 4200 + 2500 * exp(-((zz - 0.0) / 0.25)^2) + 1500 * cos(2π * (gi + 2gj) / NX) * exp(-zz^2)
    T[i, j, k] = zz < 0 ? 6500 - 3000 * zz : Tc + (1.0e5 - Tc) * 0.5 * (1 + tanh((zz - ztr) / 0.05))
end
eos = H.load_table(joinpath(@__DIR__, "..", "..", "cache", "eos_table_fullz.h5"))
o = H.HSEOpts(anchor = :bottom, P_ref = 1.0e5, g = 274.0)
res = H.column_hse(par, grid, T, eos, o)
H.create_output(par, out; overwrite = true)
H.write_fields(par, out, [H.FieldOut("/output/Pgas", res.fields["Pgas"], "Pa", ""),
                          H.FieldOut("/output/logtau500", res.fields["logtau500"], "1", ""),
                          H.FieldOut("/output/colmass", res.fields["colmass"], "kg m-2", "")];
               cols = [H.ColumnOut("/diagnostics/column_status", res.column_status, "1", "")])
MPI.Finalize()
