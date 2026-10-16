# Synthetic benchmarks for Gate C (implementation doc §14) and residual diagnostics.

"""Synthetic grid: periodic x, y (Lx = Ly = 4 Mm); z from z_top down to z_top − Z, with
z_k = z_top − Z ζ_k, ζ_k = s + 0.1 sin(π s), s = (k−1)/(NZ−1)."""
function synthetic_grid(par::ParCtx, NX, NY, NZ; Lx = 4.0e6, Ly = 4.0e6, z_top = 2.0e6, Z = 2.0e6)
    x = collect((0:NX-1) .* (Lx / NX)); y = collect((0:NY-1) .* (Ly / NY))
    s = collect(range(0.0, 1.0; length = NZ))
    z = z_top .- Z .* (s .+ 0.1 .* sin.(π .* s))
    return Grid(build_axis("x", x; periodic = true), build_axis("y", y; periodic = true), build_axis("z", z)), x, y, z
end

"""Per-layer RMS and p99 of |R|/S from the work arrays of the last objective! call."""
function layer_residual_stats(pb::MHSProblem, w::MHSWork)
    par = pb.par; o = pb.ops; b = par.block
    rs(i, j, k) = @inbounds begin
        f = 2pb.lamF * weight(o, i, j, k) * pb.invS2[i, j, k] / pb.N        # aR = f R
        sqrt(w.aR[1][i, j, k]^2 + w.aR[2][i, j, k]^2 + w.aR[3][i, j, k]^2) / f * sqrt(pb.invS2[i, j, k])
    end
    s2 = zeros(2, b.nz)
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        s2[1, k] += rs(i, j, k)^2; s2[2, k] += 1
    end
    t = MPI.Allreduce(s2, +, par.comm)
    rms = sqrt.(t[1, :] ./ t[2, :])
    p99 = layer_quantiles(par, rs, [0.99])[:, 1]
    return rms, p99
end

"""Global max |f| over owned nodes."""
maxabs(par::ParCtx, f) = global_max(par, maximum(abs(f[i, j, k]) for k in 1:par.block.nz, j in 1:par.block.ny, i in 1:par.block.nx))

"""
C1: horizontally uniform T(z) = 6000 + 3000 tanh((z − 1.2 Mm)/0.2 Mm), ideal gas μ = 1.3,
uniform Bobs (zero for C1a, (20 G, 0, 50 G) for C1b), synthetic σ. Returns the problem, the
initial state (column_hse, default substeps), the exact u (column_hse, n_sub = 64), grid.
"""
function c1_problem(par::ParCtx, NX, NY, NZ; Bobs = (0.0, 0.0, 0.0), lambda_F = 1.0, P_ref = 1.0e4)
    grid, x, y, z = synthetic_grid(par, NX, NY, NZ)
    eos = IdealGasMu(1.3)
    T = newfield(par); fill_owned!((i, j, k) -> 6000 + 3000 * tanh((z[k] - 1.2e6) / 2.0e5), T, par)
    Bo = ntuple(c -> newfield(par; fill = Bobs[c]), 3)
    Bm = sqrt(sum(abs2, Bobs))
    sig = ntuple(_ -> newfield(par; fill = sigma_synth(Bm, 0.05, 5e-4)), 3)
    ex = column_hse(par, grid, T, eos, HSEOpts(anchor = :bottom, P_ref = P_ref, n_sub = 64, adapt = false))
    ini = column_hse(par, grid, T, eos, HSEOpts(anchor = :bottom, P_ref = P_ref))
    uex = newfield(par); fill_owned!((i, j, k) -> log(ex.fields["Pgas"][i, j, k]), uex, par)
    u0 = newfield(par); fill_owned!((i, j, k) -> log(ini.fields["Pgas"][i, j, k]), u0, par)
    pb = MHSProblem(par, grid, T, Bo, sig, eos; P_ref = P_ref, lambda_F = lambda_F)
    set_scales!(pb, uex)
    return pb, init_state(pb, u0), uex, grid
end

"""Truncation floor (E3): per-layer RMS / p99 of |R|/S of the exact solution on the grid."""
function residual_floor(pb::MHSProblem, uex::Field)
    w = MHSWork(pb.par)
    objective!(pb, w, init_state(pb, uex))
    return layer_residual_stats(pb, w)
end

"""C2 potential field B = ∇φ, φ = (B0/k) cos(kx x) cos(ky y) e^{k(z − z_top)}."""
function potential_field!(B::NTuple{3,Field}, par, x, y, z; B0 = 0.01, Lx = 4.0e6, Ly = 4.0e6)
    kx = 2π / Lx; ky = 4π / Ly; kk = sqrt(kx^2 + ky^2); zt = z[1]
    b = par.block
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        xx = x[b.i0+i-1]; yy = y[b.j0+j-1]; e = exp(kk * (z[k] - zt))
        B[1][i, j, k] = -B0 * (kx / kk) * sin(kx * xx) * cos(ky * yy) * e
        B[2][i, j, k] = -B0 * (ky / kk) * cos(kx * xx) * sin(ky * yy) * e
        B[3][i, j, k] = B0 * cos(kx * xx) * cos(ky * yy) * e
    end
    exchange_halos!(par, collect(B))
    return kk
end

"""Projection of f − f_ref on the checkerboard (−1)^{gi+gj+k}, divided by N."""
function checker_amp(par::ParCtx, f, fref)
    b = par.block
    s = 0.0
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        s += (f[i, j, k] - fref[i, j, k]) * (iseven(b.i0 + i + b.j0 + j + k) ? 1.0 : -1.0)
    end
    return MPI.Allreduce(s, +, par.comm) / (par.NX * par.NY * par.NZ)
end
