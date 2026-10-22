# RH cross-checks for Gate 1 (500 nm background opacity at 10 photospheric points) and
# Gate 2 (LTE column comparison on 3 Bifrost columns, same m_top, T, nₑ and abundances).
#
# usage: julia --project scripts/rh_crosscheck.jl out/bifrost385_colhse_bottom.h5 [RH_DIR] [RUN_DIR]
#
# RH is run unmodified (cache/rh, built by deps/rh/build_rh.sh). Each column is given to RH as
# a MULTI column-mass atmosphere (lg m [g cm⁻²], T, nₑ [cm⁻³], v = 0, vturb = 0) without
# hydrogen populations; RH then derives n_H from P = g m, builds heights from dm/ρ (trapezoid)
# and τ₅₀₀ from its own continuum opacity (multiatmos.c, convertScales). SOLVE_NE = NONE so RH
# uses our nₑ. Abundances: the Wittmann vector, written in RH's log ε format.
using HSEMHS, HDF5, Statistics, Printf
const H = HSEMHS

const ELEMENTS = split("H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN " *
    "GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE PR ND PM SM " *
    "EU GD TB DY HO ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM " *
    "CM BK CF ES")

outfile = ARGS[1]
rhdir = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "cache", "rh")
rundir = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "..", "cache", "rh_runs")
rhdir = abspath(rhdir); rundir = abspath(rundir)
rundir = joinpath(rundir, splitext(get(ENV, "HSEMHS_RH_HATOM", "H_20.atom"))[1])
isfile(joinpath(rhdir, "rhf1d", "rhf1d")) || error("RH not built: run deps/rh/build_rh.sh")
g = 274.0

# ---------------------------------------------------------------------------------------
# XDR readers (big-endian)
# ---------------------------------------------------------------------------------------
mutable struct XDRReader
    b::Vector{UInt8}
    p::Int
end
XDRReader(path::AbstractString) = XDRReader(read(path), 1)
function xint(r)
    v = ntoh(reinterpret(Int32, r.b[r.p:r.p+3])[1]); r.p += 4; return Int(v)
end
function xdouble(r)
    v = ntoh(reinterpret(Float64, r.b[r.p:r.p+7])[1]); r.p += 8; return v
end
xdoubles(r, n) = [xdouble(r) for _ in 1:n]
function xcstring(r)                     # RH's xdr_counted_string: short, then xdr_string
    n = xint(r)
    n == 0 && return ""
    m = xint(r)
    s = String(r.b[r.p:r.p+m-1]); r.p += 4 * cld(m, 4)
    return s
end

function read_geometry(path)
    r = XDRReader(path)
    xint(r); nr = xint(r); nd = xint(r)
    xdoubles(r, nr); xdoubles(r, nr)
    return (height = xdoubles(r, nd), cmass = xdoubles(r, nd), tau_ref = xdoubles(r, nd))
end

function read_atmos_elements(path, Nspace)
    r = XDRReader(path)
    Nlevel = xint(r); Nelem = xint(r); xint(r)
    xdoubles(r, 3Nspace); xdoubles(r, Nspace * Nlevel)
    xcstring(r)
    w = Float64[]; a = Float64[]
    for _ in 1:Nelem
        xcstring(r); push!(w, xdouble(r)); push!(a, xdouble(r))
    end
    return w, a
end

"""Total background opacity (all absorption and scattering contributions) from backgrcontr."""
function read_bopac(path, Nspace)
    r = XDRReader(path)
    lam = xdouble(r)
    tot = zeros(Nspace); parts = Dict{String,Vector{Float64}}()
    while r.p <= length(r.b)
        lab = xcstring(r); xint(r)
        v = xdoubles(r, Nspace)
        parts[lab] = v
        tot .+= v
    end
    return lam, tot, parts
end

# ---------------------------------------------------------------------------------------
# Columns from the column_hse output
# ---------------------------------------------------------------------------------------
f = h5open(outfile, "r")
z = read(f["diagnostics/z"])
NZ = length(z)
Tin = f["input/temperature"]                     # Julia (z, y, x), float32 as in the input
# column choice on a 4×4 subsample: typical, coolest chromosphere, lowest transition region
k1 = argmin(abs.(z .- 1.0e6)); k15 = argmin(abs.(z .- 1.5e6))
Tsub = Float64.(Tin[:, 1:4:504, 1:4:504])         # (z, y, x)
T1 = Tsub[k1, :, :]
med = median(T1)
c_typ = argmin(abs.(T1 .- med))
c_cool = argmin(dropdims(minimum(Tsub[argmin(abs.(z .- 2.0e6)):argmin(abs.(z .- 0.5e6)), :, :]; dims = 1); dims = 1))
c_hot = argmax(Tsub[k15, :, :])
cols = [(4 * (c[2] - 1) + 1, 4 * (c[1] - 1) + 1) for c in (c_typ, c_cool, c_hot)]   # (gi, gj)
names = ["typical", "cool_chromosphere", "low_transition_region"]
getcol(ds, gi, gj) = Float64.(vec(f[ds][:, gj, gi]))

# ---------------------------------------------------------------------------------------
# RH inputs
# ---------------------------------------------------------------------------------------
tabf = joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5")
ab = h5open(tabf, "r") do t
    m = t["meta"]
    haskey(attributes(m), "abundances") ? nothing : nothing
end
ctx = H.WittCtx(joinpath(@__DIR__, "..", "..", "wittman-eos", "pf_Kurucz.input"))
abund, amass = H.witt_abundances(ctx)
function write_abundance(path)
    open(path, "w") do io
        println(io, "# Wittmann default abundance vector (HSEMHS), log eps = 12 + log10(A_i/A_H)")
        for (i, el) in enumerate(ELEMENTS)
            @printf(io, "  %-3s %8.4f\n", el, 12 + log10(abund[i] / abund[1]))
        end
    end
end

const HATOM = get(ENV, "HSEMHS_RH_HATOM", "H_20.atom")    # H_20 resolves the n >= 6 bound-free edges
const ATOMS = [(HATOM, "PASSIVE", "LTE_POPULATIONS"), ("C.atom", "PASSIVE", "LTE_POPULATIONS"),
               ("O.atom", "PASSIVE", "LTE_POPULATIONS"), ("Si.atom", "PASSIVE", "LTE_POPULATIONS"),
               ("Al.atom", "PASSIVE", "LTE_POPULATIONS"), ("CaII.atom", "ACTIVE", "LTE_POPULATIONS"),
               ("Fe.atom", "PASSIVE", "LTE_POPULATIONS"), ("He.atom", "PASSIVE", "LTE_POPULATIONS"),
               ("Mg.atom", "PASSIVE", "LTE_POPULATIONS"), ("N.atom", "PASSIVE", "LTE_POPULATIONS"),
               ("Na.atom", "PASSIVE", "LTE_POPULATIONS"), ("S.atom", "PASSIVE", "LTE_POPULATIONS")]

function setup_run(dir, T, ne, m; nH = nothing)
    mkpath(dir)
    write_abundance(joinpath(dir, "abundance.input"))
    open(joinpath(dir, "col.atmos"), "w") do io
        println(io, "* HSEMHS column for RH cross-check\n  HSEMHS_column\n  Mass scale\n* log g [cm s^-2]")
        @printf(io, "  %.10f\n* Ndep\n  %d\n*lg column Mass     Temperature        Ne         V              Vturb\n", log10(g * 100), length(T))
        for k in eachindex(T)
            @printf(io, "  %.15e  %.15e  %.15e  0.0  0.0\n", log10(m[k] * 0.1), T[k], ne[k] * 1e-6)
        end
        if nH !== nothing          # matched EOS: RH takes n_H (total H nuclei) from us
            println(io, "*\n* Hydrogen populations (HSEMHS total H nuclei; split only nominal)\n*    nh(1)       nh(2)       nh(3)       nh(4)       nh(5)       np")
            for k in eachindex(T)
                n = nH[k] * 1e-6; np = min(ne[k] * 1e-6, 0.999n); n1 = n - np
                @printf(io, "  %.15e  %.6e  %.6e  %.6e  %.6e  %.15e\n", n1 * (1 - 4e-20), 1e-20n1, 1e-20n1, 1e-20n1, 1e-20n1, np)
            end
        end
    end
    open(joinpath(dir, "atoms.input"), "w") do io
        println(io, "  $(length(ATOMS))\n")
        for (a, s, p) in ATOMS
            println(io, "  $(joinpath(rhdir, "Atoms", a))   $s   $p")
        end
    end
    open(joinpath(dir, "molecules.input"), "w") do io          # H2 and H2+ only, as in Wittmann
        println(io, "  2\n")
        println(io, "  $(joinpath(rhdir, "Molecules", "H2.molecule"))   PASSIVE   LTE_POPULATIONS")
        println(io, "  $(joinpath(rhdir, "Molecules", "H2+.molecule"))  PASSIVE   LTE_POPULATIONS")
    end
    write(joinpath(dir, "keyword.input"), """
  ATMOS_FILE = col.atmos
  ABUND_FILE = abundance.input
  NRAYS = 3
  ATOMS_FILE = atoms.input
  MOLECULES_FILE = molecules.input
  I_SUM = -1
  N_MAX_ITER = 1
  ITER_LIMIT = 1.0
  J_FILE = J.dat
  STARTING_J = NEW_J
  BACKGROUND_FILE = background.dat
  KURUCZ_PF_DATA = $(joinpath(rhdir, "Atoms", "pf_Kurucz.input"))
  KURUCZ_DATA = none
  SOLVE_NE = NONE
  VMICRO_CHAR = 3.0
  VACUUM_TO_AIR = FALSE
  ALLOW_PASSIVE_BB = FALSE
  LAMBDA_REF = 500.0
  GEOMETRY_OUTPUT = geometry.out
  ATMOS_OUTPUT = atmos.out
  SPECTRUM_OUTPUT = spectrum.out
  N_THREADS = 1
""")
    write(joinpath(dir, "contribute.input"), "1\n500.0\n")
end

const RLIST = (1, 4, 16, 64)
const TAB = H.load_table(joinpath(@__DIR__, "..", "cache", "eos_table_fullz.h5"))

"""
Our solution on an R-times refined height grid: ln T linear in z inside each original
interval (the solver's own interpolant), P and m from column_hse's integrator with the same
bottom anchor. Returns (z, T, P, m_absolute, ne).
"""
function refined_solution(z, T, Pbot, R)
    zr = Float64[]; Tr = Float64[]
    for k in 1:length(z)-1, s in 0:R-1
        t = s / R
        push!(zr, z[k] + t * (z[k+1] - z[k])); push!(Tr, exp(log(T[k]) + t * (log(T[k+1]) - log(T[k]))))
    end
    push!(zr, z[end]); push!(Tr, T[end])
    u = fill(NaN, length(zr)); m = fill(NaN, length(zr))
    H.hse_column!(u, m, log.(Tr), zr, TAB, H.HSEOpts(anchor = :bottom, P_ref = Pbot, g = g))
    P = exp.(u); P[end] = Pbot
    mabs = (m .- m[1]) .+ P[1] / g
    st = [H.state(TAB, log(Tr[k]), u[k]) for k in eachindex(zr)]
    ne = [exp(x[2]) for x in st]
    nH = [exp(x[1]) / H.mass_per_H(TAB) for x in st]
    return zr, Tr, P, mabs, ne, nH
end

"""RH heights aligned at the bottom node (the anchor) and RH pressure at the given heights."""
function rh_pressure_at_z(dir, zq, zgrid)
    geo = read_geometry(joinpath(dir, "geometry.out"))
    zrh = zgrid[end] .+ (geo.height .- geo.height[end])
    lm = log.(geo.cmass)
    N = length(zrh)
    Prh = similar(zq)
    for k in eachindex(zq)
        j = clamp(searchsortedlast(-zrh, -zq[k]), 1, N - 1)
        t = (zq[k] - zrh[j]) / (zrh[j+1] - zrh[j])
        Prh[k] = g * exp(lm[j] + t * (lm[j+1] - lm[j]))
    end
    return Prh, zrh, geo
end

# H_20 energies [cm^-1] (levels n = 1..19, continuum)
const H20_E = [0.000, 82258.214, 97491.217, 102822.768, 105290.515, 106631.019, 107439.301, 107963.906,
               108323.575, 108580.843, 108771.193, 108915.969, 109028.639, 109118.040, 109190.163,
               109249.191, 109298.112, 109339.108, 109373.803]
const H20_ECONT = 109677.619

"""
Corrections to RH's 500 nm background opacity for two RH artifacts (reports/rh_opacity_investigation.md):
(1) hydrogen continua whose model-atom wavelength grid starts above 500 nm (n ≥ 5 in H_20) are skipped
by Hydrogen_bf; added back with RH's own formula (hydrogenic σ, Seaton Gaunt factor, RH LTE populations);
(2) ChemicalEquilibrium stores nHmin = nₑ·n_H(atomic, all stages)·Φ instead of the neutral density;
H⁻ bound-free is rescaled by n_HI/(n_HI + n_p). Returns (added_bf, hmin_bf_correction) in m⁻¹.
"""
function rh_artifact_corrections(atmosout, Nspace, T, lam, hmbf)
    r = XDRReader(atmosout)
    Nl = xint(r); xint(r); xint(r)
    xdoubles(r, 3Nspace)
    pops = reshape(xdoubles(r, Nspace * Nl), Nspace, Nl)
    hP = 6.6260755e-34; cL = 2.99792458e8; kB = 1.380658e-23; ER = 2.1798741e-18
    qe = 1.60217733e-19; eps0 = 8.85418782e-12; me = 9.1093897e-31
    s0 = 32 / (3sqrt(3)) * qe^2 / (4π * eps0) / (me * cL) * hP / (2ER)
    add = zeros(Nspace); dh = zeros(Nspace)
    Nl == 20 || return add, dh
    for k in 1:Nspace
        stim = 1 - exp(-hP * cL / (lam * 1e-9) / (kB * T[k]))
        for i in 5:19                                  # n ≥ 5 (λmin of their grids > 500 nm)
            dE = (H20_ECONT - H20_E[i]) * 100 * hP * cL
            ledge = hP * cL / dE * 1e9
            lam <= ledge || continue
            neff = sqrt(ER / dE)
            x = (hP * cL / (lam * 1e-9)) / ER; x3 = x^(1 / 3); nsqx = 1 / (neff^2 * x)
            gbf = 1 + 0.1728x3 * (1 - 2nsqx) - 0.0496x3^2 * (1 - (1 - nsqx) * 0.66666667 * nsqx)
            add[k] += s0 * neff * gbf * (lam / ledge)^3 * stim * pops[k, i]
        end
        nHI = sum(pops[k, 1:19]); np = pops[k, 20]
        dh[k] = -hmbf[k] * np / (nHI + np)
    end
    return add, dh
end

function runrh(dir, exe)
    log = joinpath(dir, "$(exe).log")
    ok = success(pipeline(Cmd(`$(joinpath(rhdir, "rhf1d", exe))`; dir = dir); stdout = log, stderr = log))
    # backgrcontr frees its (unallocated) opacity-fudge matrix after writing the output when
    # OPACITY_FUDGE = none and exits with "Trying to free NULL pointer": accept that case only
    if !ok && exe == "backgrcontr" && occursin("freeMatrix", read(log, String)) &&
       isfile(joinpath(dir, "bopac500.000"))
        return
    end
    ok || error("RH $exe failed in $dir; see $log")
end

# ---------------------------------------------------------------------------------------
# Run and compare
# ---------------------------------------------------------------------------------------
io = IOBuffer()
println(io, "# RH cross-check (Gate 1 opacity row, Gate 2 RH row)\n")
println(io, "RH source commit `$(strip(read(joinpath(rhdir, "SOURCE_COMMIT"), String)))`, unmodified; inputs in `$(rundir)`.")
println(io, "RH hydrogen model atom: `$HATOM` (LTE populations).")
println(io, "Matched: T, nₑ (SOLVE_NE = NONE), abundances (Wittmann vector), m_top (= P_top/g of the column_hse solution), g = $g m s⁻², molecules H₂ and H₂⁺ only, no line opacity (KURUCZ_DATA = none, ALLOW_PASSIVE_BB = FALSE).\n")
gate2 = Float64[]
opac_rows = String[]
for (c, ((gi, gj), nm)) in enumerate(zip(cols, names))
    T = getcol("input/temperature", gi, gj)
    P = getcol("output/Pgas", gi, gj); rho = getcol("output/rho", gi, gj); ne = getcol("output/ne", gi, gj)
    m = getcol("output/colmass", gi, gj); tau = getcol("output/tau500", gi, gj); chi = getcol("output/chi500", gi, gj)
    nH = getcol("output/nH_total", gi, gj)
    mtot = m .+ P[1] / g                                  # absolute column mass, m_top = P_top/g
    dir = joinpath(rundir, "col$(c)_$(nm)")
    setup_run(dir, T, ne, mtot)
    runrh(dir, "rhf1d")
    runrh(dir, "backgrcontr")
    geo = read_geometry(joinpath(dir, "geometry.out"))
    w, a = read_atmos_elements(joinpath(dir, "atmos.out"), NZ)
    totalAbund = sum(a); wght = sum(a .* w)
    kB = 1.380658e-23; amu = 1.6605402e-27            # RH constant.h values
    nH_rh = (g .* geo.cmass .- ne .* kB .* T) ./ (totalAbund .* kB .* T)
    rho_rh = amu * wght .* nH_rh
    # heights: RH shifts h so that tau_ref = 1 at h = 0 (convertScales); refer to the top node
    zrh = z[1] .+ (geo.height .- geo.height[1])
    # P_RH at our heights: interpolate ln m(z_RH) at z (both decreasing)
    lm = log.(geo.cmass)
    Prh = similar(z)
    for k in eachindex(z)
        j = clamp(searchsortedlast(-zrh, -z[k]), 1, NZ - 1)
        t = (z[k] - zrh[j]) / (zrh[j+1] - zrh[j])
        Prh[k] = g * exp(lm[j] + t * (lm[j+1] - lm[j]))
    end
    dP = abs.(Prh ./ P .- 1)
    drho = abs.(rho_rh ./ rho .- 1)
    dz = (zrh .- z) ./ 1e3
    dtau = log10.(geo.tau_ref[2:end] ./ tau[2:end])
    # refinement study: our T(z) profile and solution sampled R times finer; RH's trapezoid
    # height scale should converge as R^-2 towards the RK4 heights (heights aligned at the
    # bottom anchor). Compared on the original nodes up to the transition region.
    ktr = something(findlast(k -> T[k] >= 2.0e4, 1:NZ), 0) + 1      # first node below the TR
    conv = Float64[]; convall = Float64[]; conv_rh = Float64[]
    for R in RLIST
        zr, Tr, Pr, mr, ner, nHr = refined_solution(z, T, P[end], R)
        # (a) matched EOS: RH receives our n_H; only the height integration differs
        dR = joinpath(rundir, "col$(c)_$(nm)_R$(R)_matched")
        setup_run(dR, Tr, ner, mr; nH = nHr)
        runrh(dR, "rhf1d")
        PR, _, _ = rh_pressure_at_z(dR, z, zr)
        e = abs.(PR ./ P .- 1)
        push!(conv, maximum(e[ktr:end])); push!(convall, maximum(e))
        # (b) RH's own n_H from P = g m (no H2): EOS-difference diagnostic
        dE = joinpath(rundir, "col$(c)_$(nm)_R$(R)_rheos")
        setup_run(dE, Tr, ner, mr)
        runrh(dE, "rhf1d")
        PE, _, _ = rh_pressure_at_z(dE, z, zr)
        push!(conv_rh, maximum(abs.(PE ./ P .- 1)[ktr:end]))
    end
    push!(gate2, conv[end])
    kr = argmax(drho)
    @printf(io, "## Column %d (%s): global (i, j) = (%d, %d); T range %.0f – %.0f K\n\n", c, nm, gi, gj, minimum(T), maximum(T))
    @printf(io, "- ρ: RH algebraic n_H (metals neutral, no H₂) vs HSEMHS Wittmann: max |Δρ/ρ| = %.2e (median %.2e)\n", maximum(drho), median(drho))
    @printf(io, "- Height scale (RH trapezoid of dm/ρ vs HSEMHS RK4): bottom-node offset %.3f km, max |Δz| %.3f km\n", dz[end], maximum(abs.(dz)))
    @printf(io, "- ρ: largest difference at z = %.0f km, T = %.0f K, P = %.3g Pa\n", z[kr] / 1e3, T[kr], P[kr])
    @printf(io, "- P at the same height, RH on the native grid: max |ΔP/P| = %.2e (median %.2e)\n", maximum(dP), median(dP))
    @printf(io, "- **Matched EOS (RH given our n_H), profile refined ×1 / ×4 / ×16 / ×64, heights aligned at the bottom anchor, z ≤ %.0f km: max |ΔP/P| = %.2e / %.2e / %.2e / %.2e** (orders %.2f, %.2f, %.2f)\n",
            z[ktr] / 1e3, conv..., [log(conv[i] / conv[i+1]) / log(4) for i in 1:3]...)
    @printf(io, "- Matched EOS including the TR and corona: %.2e / %.2e / %.2e / %.2e\n", convall...)
    @printf(io, "- RH's own n_H (P = g m, no H₂), same refinement: %.2e / %.2e / %.2e / %.2e (EOS-difference floor)\n", conv_rh...)
    @printf(io, "- Mass per H nucleus: RH (AMU·wght_per_H) / HSEMHS − 1 = %+.2e\n", amu * wght / H.mass_per_H(TAB) - 1)
    @printf(io, "- log τ₅₀₀ (RH tau_ref vs HSEMHS, below the top node): median %+.4f, p1/p99 %+.4f / %+.4f dex\n\n",
            median(dtau), quantile(dtau, 0.01), quantile(dtau, 0.99))
    if c == 1
        lam, chirh, parts = read_bopac(joinpath(dir, @sprintf("bopac%.3f", 500.0)), NZ)
        ks = round.(Int, range(argmin(abs.(z .- 0.4e6)), argmin(abs.(z .+ 0.2e6)); length = 10))
        add, dh = rh_artifact_corrections(joinpath(dir, "atmos.out"), NZ, T, lam, parts["HMINUS_BF"])
        chicor = chirh .+ add .+ dh
        for k in ks
            r = chirh[k] / chi[k] - 1
            rc = chicor[k] / chi[k] - 1
            push!(opac_rows, @sprintf("| %.0f | %.0f | %.4g | %.4g | %.4g | %+.2f %% | %+.2f %% | %.2e |", z[k] / 1e3, T[k], P[k], chi[k], chirh[k], 100r, 100rc, nH_rh[k] / nH[k] - 1))
        end
        global opac_err_cor = [abs(chicor[k] / chi[k] - 1) for k in ks]
        global opac_lam = lam
        global opac_parts = parts
        global opac_ks = ks
        global opac_err = [abs(chirh[k] / chi[k] - 1) for k in ks]
    end
end
println(io, "## Gate 1 row: 500 nm background opacity at 10 photospheric points (column 1)\n")
@printf(io, "RH `backgrcontr` at λ = %.3f nm, all absorption and scattering contributions summed; HSEMHS χ₅₀₀ from the table.\n\n", opac_lam)
println(io, "| z [km] | T [K] | P [Pa] | χ HSEMHS [m⁻¹] | χ RH [m⁻¹] | RH/HSEMHS − 1 | RH, artifacts removed / HSEMHS − 1 | n_H RH/HSEMHS − 1 |\n|---|---|---|---|---|---|---|---|")
foreach(r -> println(io, r), opac_rows)
@printf(io, "\nMax |Δχ/χ| vs RH as is = %.2f %%; vs RH with its two artifacts removed = %.2f %% (threshold 5 %%): **%s**\n\n", 100maximum(opac_err), 100maximum(opac_err_cor), maximum(opac_err_cor) <= 0.05 ? "pass (artifact-corrected)" : "fail")
println(io, "RH contributions at the deepest of these points (m⁻¹): ",
        join(["$k = $(@sprintf("%.3g", v[opac_ks[end]]))" for (k, v) in sort(collect(opac_parts); by = x -> -x[2][opac_ks[end]]) if v[opac_ks[end]] > 0], ", "))
@printf(io, "\n\n## Gate 2 row summary\n\nmax |ΔP/P| over the 3 columns, matched EOS, RH on the ×64 refined grid = %.2e (threshold 1e-3): **%s**\n", maximum(gate2), maximum(gate2) <= 1e-3 ? "pass" : "fail")
close(f)
rep = joinpath(@__DIR__, "..", "reports", HATOM == "H_20.atom" ? "rh_crosscheck.md" : "rh_crosscheck_$(splitext(HATOM)[1]).md")
write(rep, String(take!(io)))
print(read(rep, String))
