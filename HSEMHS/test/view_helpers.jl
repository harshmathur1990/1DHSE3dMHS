# Shared helper for the viewing-angle tests (test_view.jl, mpi/mpi_view.jl).

"""
Exact check of `inclined_tau!` on `par`'s block: χ is built so that every ray passes through grid points (levels spaced
so that t z_k is a whole number of cells), so τ along a ray is the vertical trapezoid of that ray's χ samples and τ at a
grid point is the value of the ray through it. Returns the largest relative errors of τ and z(τ_LOS = 1).
"""
function shear_exact_error(par, φ; μ = 0.6)
    NX, NY, nz = par.NX, par.NY, par.NZ
    v = H.ViewOptions(mu = μ, azimuth_deg = φ, geometry = :inclined)
    t = H.slant(v)
    dx = 50e3
    dy = abs(t[2]) > 1e-12 && abs(t[1]) > 1e-12 ? dx * abs(t[2] / t[1]) : 40e3   # equal cell shifts on a diagonal
    Δ = abs(t[1]) > 1e-12 ? dx / abs(t[1]) : dy / abs(t[2])
    z = [(nz - k - 2) * Δ for k in 1:nz]                                       # top first, some levels below 0
    nsx = [round(Int, t[1] * z[k] / dx) for k in 1:nz]; nsy = [round(Int, t[2] * z[k] / dy) for k in 1:nz]
    rng = MersenneTwister(11)
    C = [2e-5 * (1 + rand(rng)) * exp(-z[k] / 1.0e5) for k in 1:nz, J in 1:NY, I in 1:NX]   # χ of the ray (I, J); τ reaches 1
    chi = zeros(nz, NY, NX)
    for I in 1:NX, J in 1:NY, k in 1:nz
        chi[k, mod1(J + nsy[k], NY), mod1(I + nsx[k], NX)] = C[k, J, I]
    end
    b = par.block; jr = b.j0:b.j0+b.ny-1; ir = b.i0:b.i0+b.nx-1
    tau = zeros(nz, b.ny, b.nx); zt = zeros(b.ny, b.nx); tm = zeros(UInt8, b.ny, b.nx)
    H.inclined_tau!(tau, zt, tm, par, chi[:, jr, ir], z, dx, dy, μ, t)
    tt = zeros(nz, NY, NX); zt1 = zeros(NY, NX); tc = zeros(nz)
    for I in 1:NX, J in 1:NY
        H.tau_column!(tc, C[:, J, I], z; mu = μ)
        tt[:, J, I] .= tc; zt1[J, I] = H.z_tau1(tc, z)
    end
    et = 0.0
    for (i, I) in enumerate(ir), (j, J) in enumerate(jr), k in 1:nz
        ref = tt[k, mod1(J - nsy[k], NY), mod1(I - nsx[k], NX)]
        et = max(et, abs(tau[k, j, i] - ref) / ref)
    end
    isequal(isnan.(zt), isnan.(zt1[jr, ir])) || return et, Inf
    all(isnan, zt1) && return et, Inf                                           # the check needs τ = 1 crossings
    f = .!isnan.(zt)
    ez = maximum(abs.(zt[f] .- zt1[jr, ir][f]) ./ max.(abs.(zt1[jr, ir][f]), Δ); init = 0.0)
    return et, ez
end
