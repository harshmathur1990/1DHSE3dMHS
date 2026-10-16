# Shared helpers for single-rank and multi-rank tests.

"""Nonuniform, strictly decreasing z (top first) with a fixed-seed jitter."""
function test_z(NZ; top = 2.0e6, bot = -0.5e6, jitter = 0.3, seed = 7)
    rng = Xoshiro(seed)
    u = collect(range(top, bot; length = NZ))
    dz = (top - bot) / (NZ - 1)
    for k in 2:NZ-1
        u[k] += jitter * dz * (rand(rng) - 0.5)
    end
    return u
end

function test_setup(NX, NY, NZ; px = 0, py = 0, Lx = 4.0e6, Ly = 3.0e6)
    par = H.init_parallel(NX, NY, NZ; px = px, py = py)
    x = collect((0:NX-1) .* (Lx / NX))
    y = collect((0:NY-1) .* (Ly / NY)) .+ 1.0e3
    z = test_z(NZ)
    grid = H.Grid(H.build_axis("x", x; periodic = true), H.build_axis("y", y; periodic = true),
                  H.build_axis("z", z))
    return par, grid, H.Ops(grid, par), x, y, z
end

"""Fill owned nodes of a field from a function of global coordinates."""
function fill_global!(f, par, x, y, z, fun)
    b = par.block
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        f[i, j, k] = fun(x[b.i0+i-1], y[b.j0+j-1], z[k], b.i0 + i - 1, b.j0 + j - 1, k)
    end
    return f
end

"""Global random field: same values regardless of decomposition (hash of the global index)."""
grand(gi, gj, k; seed = 1) = (hash((gi, gj, k, seed)) % 2^30) / 2^30 - 0.5

"""Check ⟨D f, q⟩ = ⟨f, Dᵀ q⟩ for all six operators; returns max relative error."""
function transpose_errors(par, ops, x, y, z)
    f = H.newfield(par); q = H.newfield(par)
    fill_global!(f, par, x, y, z, (xx, yy, zz, gi, gj, k) -> grand(gi, gj, k; seed = 11))
    fill_global!(q, par, x, y, z, (xx, yy, zz, gi, gj, k) -> grand(gi, gj, k; seed = 12))
    H.exchange_halos!(par, [f, q])
    Df = H.newfield(par); DTq = H.newfield(par)
    errs = Float64[]
    for (op, opT, ax) in ((H.dx, H.dxT, ops.X), (H.dy, H.dyT, ops.Y), (H.dz, H.dzT, ops.Z),
                          (H.d2x, H.d2xT, ops.X), (H.d2y, H.d2yT, ops.Y), (H.d2z, H.d2zT, ops.Z))
        H.apply!(Df, op, f, ax, par)
        H.apply!(DTq, opT, q, ax, par)
        a = H.global_dot(par, Df, q)
        b = H.global_dot(par, f, DTq)
        push!(errs, abs(a - b) / abs(a))
    end
    return errs
end

"""Halo check: f = gi + 1000 gj + 1e6 k; every halo cell must equal the wrapped formula."""
function halo_errors(par)
    b = par.block
    f = H.newfield(par, fill = NaN)
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        f[i, j, k] = (b.i0 + i - 1) + 1000.0 * (b.j0 + j - 1) + 1.0e6 * k
    end
    g2 = H.newfield(par); g2 .= 2 .* f
    H.exchange_halos!(par, [f, g2])
    bad = 0
    val(gi, gj, k) = mod1(gi, par.NX) + 1000.0 * mod1(gj, par.NY) + 1.0e6 * k
    for k in 1:b.nz
        for j in 1:b.ny, i in (0, b.nx + 1)
            e = val(b.i0 + i - 1, b.j0 + j - 1, k)
            (f[i, j, k] == e && g2[i, j, k] == 2e) || (bad += 1)
        end
        for i in 0:b.nx+1, j in (0, b.ny + 1)          # includes the diagonal corners (staggered averages)
            e = val(b.i0 + i - 1, b.j0 + j - 1, k)
            (f[i, j, k] == e && g2[i, j, k] == 2e) || (bad += 1)
        end
    end
    return H.global_count(par, bad)
end

"""Expected fixture values (see oracles/python/make_fixture.py), SI."""
fixT(i, j, k) = Float64(Float32(1000.0 + (i - 1) + 100.0 * (j - 1) + 10000.0 * (k - 1)))
fixBx(i, j, k) = Float64(Float32(1.0 + (i - 1) + 0.5 * (j - 1) - 0.25 * (k - 1))) * 1e-4

function check_fixture_read(path, par)
    spec = H.open_input(path)
    T = H.read_field(spec, par, "temperature", :temperature; positive = true)
    Bx = H.read_field(spec, par, "B_x", :field)
    b = par.block
    bad = 0
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        gi, gj = b.i0 + i - 1, b.j0 + j - 1
        (T[i, j, k] == fixT(gi, gj, k) && Bx[i, j, k] == fixBx(gi, gj, k)) || (bad += 1)
    end
    return H.global_count(par, bad)
end


"""Write a pattern field and a per-column field; return the file path."""
function write_pattern(par, path)
    b = par.block
    f = H.newfield(par)
    for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx
        f[i, j, k] = (b.i0 + i - 1) + 1000.0 * (b.j0 + j - 1) + 1.0e6 * k
    end
    col = [UInt8(((b.i0 + i - 1) + (b.j0 + j - 1)) % 3) for i in 1:b.nx, j in 1:b.ny]
    H.create_output(par, path; overwrite = true)
    H.write_fields(par, path, [H.FieldOut("/output/pattern", f, "1", "test pattern")];
                   cols = [H.ColumnOut("/diagnostics/colmask", col, "1", "test mask")])
    H.write_layer_table(par, path, "/diagnostics/layer_test", [Float64(k + c) for k in 1:par.NZ, c in 1:3],
                        ["a", "b", "c"])
    H.write_meta(par, path, Dict("mode" => "test", "nested" => Dict("x" => 1.5)))
    return path
end

"""Read back with HDF5.jl, convert to the (x, y, z) convention and verify."""
function check_pattern(path, NX, NY, NZ)
    h5open(path, "r") do f
        a = read(f["output/pattern"])
        size(a) == (NZ, NY, NX) || return false
        A = permutedims(a, (3, 2, 1))
        ok = all(A[i, j, k] == i + 1000.0j + 1.0e6k for i in 1:NX, j in 1:NY, k in 1:NZ)
        m = permutedims(read(f["diagnostics/colmask"]), (2, 1))
        ok &= all(m[i, j] == UInt8((i + j) % 3) for i in 1:NX, j in 1:NY)
        t = permutedims(read(f["diagnostics/layer_test"]), (2, 1))
        ok &= t[3, 2] == 5.0
        ok &= read_attribute(f["meta"], "mode") == "test"
        ok &= read_attribute(f["meta/nested"], "x") == 1.5
        ok &= read_attribute(f["output/pattern"], "dims") == ["x", "y", "z"]
        ok
    end
end


# ---- staggered-mesh helpers ----
"""Fill owned i, j and the whole z axis of f with a decomposition-independent random field."""
function fill_rand!(f, par; seed = 1)
    b = par.block
    for k in axes(f, 3), j in 1:b.ny, i in 1:b.nx
        f[i, j, k] = grand(b.i0 + i - 1, b.j0 + j - 1, k; seed = seed)
    end
    return f
end

"""Staggered test fields: A (edges), B (faces), centre field."""
function stag_fields(par)
    A = (H.newfield0(par), H.newfield0(par), H.newfield(par))
    B = (H.newfield(par), H.newfield(par), H.newfield0(par))
    J = (H.newfield0(par), H.newfield0(par), H.newfield(par))
    return A, B, J
end

"""Max relative transpose errors for C, C*, H, G on random data; and max |div curl A|."""
function staggered_checks(par, grid)
    s = H.StagOps(grid, par)
    A, B, J = stag_fields(par)
    for (q, f) in enumerate(A); fill_rand!(f, par; seed = 10 + q); end
    H.exchange_halos!(par, collect(A))
    Bbar = 0.37
    H.curlA!(B, A, Bbar, s, par)
    H.exchange_halos!(par, collect(B))
    b = par.block
    dmax = maximum(abs(H.div_face(B, s, i, j, k)) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)
    bmax = maximum(maximum(abs, B[q][1:b.nx, 1:b.ny, :]) for q in 1:3)
    hmin = min(minimum(s.dxc), minimum(s.dyc), minimum(s.hz))
    divrel = H.global_max(par, dmax) / (H.global_max(par, bmax) / hmin)
    errs = Dict{String,Float64}()
    # C
    gB = (H.newfield(par), H.newfield(par), H.newfield0(par))
    for (q, f) in enumerate(gB); fill_rand!(f, par; seed = 20 + q); end
    H.exchange_halos!(par, collect(gB))
    gA = (H.newfield0(par), H.newfield0(par), H.newfield(par))
    gbar = H.curlA_T!(gA, gB, s, par)
    lhs = sum(H.sdot(par, B[q], gB[q]) for q in 1:3)
    rhs = sum(H.sdot(par, A[q], gA[q]) for q in 1:3) + Bbar * gbar
    nrm(a, c) = sqrt(sum(H.sdot(par, a[q], a[q]) for q in 1:3) * sum(H.sdot(par, c[q], c[q]) for q in 1:3))
    errs["C"] = abs(lhs - rhs) / nrm(B, gB)
    # C* (on an independent random B: B = curl A + 0.37 T would cancel a large constant)
    for (q, f) in enumerate(B); fill_rand!(f, par; seed = 70 + q); end
    H.exchange_halos!(par, collect(B))
    H.curlB!(J, B, s, par)
    gJ = (H.newfield0(par), H.newfield0(par), H.newfield(par))
    for (q, f) in enumerate(gJ); fill_rand!(f, par; seed = 30 + q); end
    H.exchange_halos!(par, collect(gJ))
    gB2 = (H.newfield(par), H.newfield(par), H.newfield0(par))
    H.curlB_T!(gB2, gJ, s, par)
    lhs = sum(H.sdot(par, J[q], gJ[q]) for q in 1:3); rhs = sum(H.sdot(par, B[q], gB2[q]) for q in 1:3)
    errs["Cstar"] = abs(lhs - rhs) / nrm(J, gJ)
    errs["Cstar_rel"] = abs(lhs - rhs) / abs(lhs)
    # H
    HB = (H.newfield(par), H.newfield(par), H.newfield(par))
    H.H!(HB, B, s, par)
    gH = (H.newfield(par), H.newfield(par), H.newfield(par))
    for (q, f) in enumerate(gH); fill_rand!(f, par; seed = 40 + q); end
    H.exchange_halos!(par, collect(gH))
    gB3 = (H.newfield(par), H.newfield(par), H.newfield0(par))
    H.H_T!(gB3, gH, s, par)
    lhs = sum(H.sdot(par, HB[q], gH[q]) for q in 1:3); rhs = sum(H.sdot(par, B[q], gB3[q]) for q in 1:3)
    errs["H"] = abs(lhs - rhs) / nrm(HB, gH)
    # G
    P = H.newfield(par); fill_rand!(P, par; seed = 50); H.exchange_halos!(par, [P])
    GP = (H.newfield(par), H.newfield(par), H.newfield0(par))
    H.G!(GP, P, s, par)
    gR = (H.newfield(par), H.newfield(par), H.newfield0(par))
    for (q, f) in enumerate(gR); fill_rand!(f, par; seed = 60 + q); end
    H.exchange_halos!(par, collect(gR))
    gP = H.newfield(par); H.G_T!(gP, gR, s, par)
    lhs = sum(H.sdot(par, GP[q], gR[q]) for q in 1:3); rhs = H.sdot(par, P, gP)
    errs["G"] = abs(lhs - rhs) / nrm(GP, gR)
    return divrel, errs
end
