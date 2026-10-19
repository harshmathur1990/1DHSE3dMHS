# Coarsened periodic Bifrost problem (staggered, D9b defaults). bif_problem(c, sz; kw...) -> (pb, x, par, b, NZ)
using HSEMHS, MPI, HDF5, Printf, Statistics
const H = HSEMHS
H.ensure_mpi()
const BIF = normpath(joinpath(@__DIR__, "..", "..", "..", "IO", "input-mhs-hse.nc"))
const TAB = normpath(joinpath(@__DIR__, "..", "..", "cache", "eos_table.h5"))
function bif_problem(c, sz; kw...)
    raw = h5open(BIF) do f
        (x = read(f["x"]), y = read(f["y"]), z = read(f["z"]), T = read(f["temperature"]), Bx = read(f["B_x"]), By = read(f["B_y"]), Bz = read(f["B_z"]),
         Pref = read_attribute(f, "P_ref_Pa"), Lx = read_attribute(f, "period_x_m"), Ly = read_attribute(f, "period_y_m"), g = read_attribute(f, "gravity_m_s2"))
    end
    ks = collect(1:sz:length(raw.z)); ks[end] == length(raw.z) || push!(ks, length(raw.z))
    NX = length(raw.x) ÷ c; NY = length(raw.y) ÷ c; NZ = length(ks)
    coarse(A, k, j, i) = mean(@view A[k, (j-1)*c+1:j*c, (i-1)*c+1:i*c])
    xs = [mean(raw.x[(i-1)*c+1:i*c]) for i in 1:NX]; ys = [mean(raw.y[(j-1)*c+1:j*c]) for j in 1:NY]
    par = H.init_parallel(NX, NY, NZ); b = par.block
    grid = H.Grid(H.build_axis("x", xs; periodic = true, period = raw.Lx), H.build_axis("y", ys; periodic = true, period = raw.Ly), H.build_axis("z", raw.z[ks]))
    T = H.newfield(par); H.fill_owned!((i, j, k) -> coarse(raw.T, ks[k], b.j0 + j - 1, b.i0 + i - 1), T, par)
    Bo = ntuple(_ -> H.newfield(par), 3)
    for (q, A) in enumerate((raw.Bx, raw.By, raw.Bz))
        H.fill_owned!((i, j, k) -> coarse(A, ks[k], b.j0 + j - 1, b.i0 + i - 1), Bo[q], par)
    end
    sig = ntuple(_ -> H.newfield(par), 3)
    for q in 1:3
        H.fill_owned!((i, j, k) -> H.sigma_synth(sqrt(Bo[1][i, j, k]^2 + Bo[2][i, j, k]^2 + Bo[3][i, j, k]^2), 0.05, 5e-4), sig[q], par)
    end
    pb = H.SMHSProblem(par, grid, T, Bo, sig, H.load_table(TAB); P_ref = Float64(raw.Pref), g = Float64(raw.g), kw...)
    x = H.init_state(pb); H.set_scales!(pb, x[1])
    w = H.SMHSWork(par); H.init_A!(pb, w, x; max_iter = 3000)
    return pb, x, par, b, NZ
end
