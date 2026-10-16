# Generate witt_shim.cpp from the verbatim witt_eos_cpp.cpp by exact, counted string
# replacements (implementation doc §7.1), then compile libwitt.
#
# usage: julia deps/witt/build.jl            (from the HSEMHS directory or anywhere)

using SHA

const SRCDIR = @__DIR__
const SRC = joinpath(SRCDIR, "witt_eos_cpp.cpp")
# HSEMHS_WITT_BUILD_DIR builds a variant (e.g. FIX_GASC_H2 = 0) elsewhere without touching
# the default shim and library in deps/witt
const DIR = get(ENV, "HSEMHS_WITT_BUILD_DIR", SRCDIR)
const SHIM = joinpath(DIR, "witt_shim.cpp")
const FIX_GASC_H2 = get(ENV, "HSEMHS_FIX_GASC_H2", "1") == "1"

function replace_exact(s, old, new, count)
    n = length(findall(old, s))
    n == count || error("patch: expected $count occurrence(s) of \"$old\", found $n")
    return replace(s, old => new)
end

const MEMBERS = """
    // ---- HSEMHS shim: tolerances (spec §4.1) ----
    double prec_ = 1.0e-10;   // outer fixed-point tolerance (was PREC = 1e-5)
    int maxit_ = 5000;        // outer iteration cap (was 250)
    int maxit_inner_ = 50;    // f5 refinement cap (was 5)
"""

const PUBLIC_API = """
    // ---- HSEMHS shim: status-returning solve and accessors ----
    struct PeSolve { double pe; int it; bool converged; double dif; };
    PeSolve pe_from_pg_status(double t, double pg) const {
        double dif = 1.1, pe = init_pe_from_pg(t, pg), ope = pe;
        int it = 0;
        while ((std::abs(dif) > prec_) && (it < maxit_)) {
            pe = (ope + pe) * 0.5;
            ope = pe;
            pe = pe_pg(t, pe, pg).first;
            dif = 2.0 * std::abs(pe - ope) / (pe + ope);
            ++it;
        }
        const bool ok = (std::abs(dif) <= prec_) && std::isfinite(pe) && pe > 0.0;
        return {pe, it, ok, dif};
    }
    double shim_pg_from_pe(double t, double pe) const { return gasc(t, pe).pg; }
    double shim_rho_cgs_from_pe(double t, double pe) const { return rho_from_pe(t, pe); }
    void shim_partials(double t, double pg, double pe, double* n17) const { background_partials(t, pg, pe, n17); }
    double shim_mass_per_H_kg() const { return avw / ABUND[0] * 1.0e-3; }   // avw is in g
    void shim_abundances(double* ab, double* am) const {
        for (int i = 0; i < 99; ++i) { ab[i] = ABUND[i]; am[i] = AMASS[i]; }
    }
    int shim_nstage(int atom) const { return int(el[atom].nstage); }
    double shim_eion(int atom, int stage) const { return el[atom].eion[stage]; }
    int shim_pf(int atom, double t, double* u8) const { int c = 0; partition_f(atom, t, 0, u8, c); return c; }
    double shim_ab_others() const { return ab_others; }
    int shim_tpf(double* out, int nmax) const {
        const int n = int(tpf.size());
        for (int i = 0; i < n && i < nmax; ++i) out[i] = tpf[i];
        return n;
    }
    double shim_rho_from_H() const { return rho_from_H; }
    double shim_tpf_max() const { return tpf.back(); }
    double shim_tpf_min() const { return tpf.front(); }

"""

const C_API_HEAD = "\n#define FIX_GASC_H2_FLAG " * (FIX_GASC_H2 ? "1" : "0") * "\n"
const C_API = raw"""

// =========================================================================================
// HSEMHS C API (implementation doc §7.1). All SI in/out; CGS inside.
// =========================================================================================
extern "C" {

struct WittState {
    double rho, ne, pe, mu_bar, nhtot, closure_residual, dif;
    int32_t iterations;
    int32_t converged;
};

void* witt_ctx_new(const char* pf_path, double prec, int maxit, const double* log_abund_or_null) {
    try { return new WittEOS(pf_path, prec, maxit, log_abund_or_null); } catch (...) { return nullptr; }
}

void witt_ctx_free(void* ctx) { delete static_cast<WittEOS*>(ctx); }

int witt_ctx_state(void* ctx, double T, double P_pa, WittState* out) {
    try {
        const WittEOS& e = *static_cast<WittEOS*>(ctx);
        const double pg = P_pa * 10.0;
        const auto s = e.pe_from_pg_status(T, pg);
        const double rho_cgs = e.shim_rho_cgs_from_pe(T, s.pe);
        out->rho = rho_cgs * 1.0e3;
        out->pe = s.pe * 0.1;
        out->ne = s.pe / (BK * T) * 1.0e6;
        out->mu_bar = rho_cgs * BK * T / (pg * AMU);
        out->nhtot = out->rho / e.shim_mass_per_H_kg();
        out->closure_residual = std::abs(e.shim_pg_from_pe(T, s.pe) - pg) / pg;
        out->dif = s.dif;
        out->iterations = s.it;
        // With FIX_GASC_H2 the gasc pressure at the converged pe reproduces P to ~1e-10, so
        // the closure test is part of convergence. Without the fix it is only a diagnostic
        // (pe_pg and gasc then disagree by up to 40 % where H2 matters).
        const bool closure_ok = FIX_GASC_H2_FLAG ? (out->closure_residual <= 1.0e-8) : true;
        out->converged = (s.converged && closure_ok && std::isfinite(out->rho) && out->rho > 0.0 &&
                          std::isfinite(out->ne) && out->ne > 0.0) ? 1 : 0;
        return out->converged ? 0 : 1;
    } catch (...) {
        out->converged = 0;
        return 2;
    }
}

int witt_ctx_partials(void* ctx, double T, double P_pa, double pe_cgs,
                      double* xna_cm3, double* xne_cm3, double* n17) {
    try {
        const WittEOS& e = *static_cast<WittEOS*>(ctx);
        const double pg = P_pa * 10.0;
        *xna_cm3 = (pg - pe_cgs) / (BK * T);
        *xne_cm3 = pe_cgs / (BK * T);
        e.shim_partials(T, pg, pe_cgs, n17);
        return 0;
    } catch (...) { return 2; }
}

double witt_ctx_mass_per_H_kg(void* ctx) { return static_cast<WittEOS*>(ctx)->shim_mass_per_H_kg(); }

void witt_ctx_abundances(void* ctx, double* abund99, double* amass99) {
    static_cast<WittEOS*>(ctx)->shim_abundances(abund99, amass99);
}

int witt_ctx_nstage(void* ctx, int atom0) { return static_cast<WittEOS*>(ctx)->shim_nstage(atom0); }
double witt_ctx_eion_ev(void* ctx, int atom0, int stage0) { return static_cast<WittEOS*>(ctx)->shim_eion(atom0, stage0); }
double witt_ctx_tpf_max(void* ctx) { return static_cast<WittEOS*>(ctx)->shim_tpf_max(); }
int witt_shim_fix_gasc_h2() { return FIX_GASC_H2_FLAG; }
int witt_ctx_pf(void* ctx, int atom0, double T, double* u8) {
    try { return static_cast<WittEOS*>(ctx)->shim_pf(atom0, T, u8); } catch (...) { return -1; }
}
int witt_ctx_tpf(void* ctx, double* out, int nmax) { return static_cast<WittEOS*>(ctx)->shim_tpf(out, nmax); }
double witt_ctx_ab_others(void* ctx) { return static_cast<WittEOS*>(ctx)->shim_ab_others(); }
double witt_ctx_rho_from_H(void* ctx) { return static_cast<WittEOS*>(ctx)->shim_rho_from_H(); }
double witt_ctx_tpf_min(void* ctx) { return static_cast<WittEOS*>(ctx)->shim_tpf_min(); }

}  // extern "C"
"""

function make_shim()
    s = read(SRC, String)
    s = replace_exact(s, "constexpr double PREC = 1.0e-5;",
                      "// HSEMHS shim: PREC replaced by the member prec_ (see WittEOS).", 1)
    s = replace_exact(s, "    explicit WittEOS(const char* pf_path) {\n        for (size_t i = 0; i < ABUND.size(); ++i) ABUND[i] = std::pow(10.0, ABUND_RAW[i]);",
        "    explicit WittEOS(const char* pf_path, double prec = 1.0e-10, int maxit = 5000,\n" *
        "                     const double* log_abund = nullptr) : prec_(prec), maxit_(maxit) {\n" *
        "        for (size_t i = 0; i < ABUND.size(); ++i)\n" *
        "            ABUND[i] = std::pow(10.0, log_abund ? log_abund[i] : ABUND_RAW[i]);", 1)
    s = replace_exact(s, "\nprivate:\n", "\n" * PUBLIC_API * "private:\n" * MEMBERS, 1)
    s = replace_exact(s, "while ((std::abs(dif) > PREC) && (it < 250)) {",
                      "while ((std::abs(dif) > prec_) && (it < maxit_)) {", 1)
    s = replace_exact(s, "while ((diff > 1.0e-5) && (it < 5)) {",
                      "while ((diff > prec_) && (it < maxit_inner_)) {", 2)
    if FIX_GASC_H2
        # gasc() omits the factor pe in the H2 / H2+ constants that pe_pg() uses
        # (g4 = pe*10^cmol0, g5 = pe*10^cmol1). The pe_pg form is the exact solution of the
        # H/H+/H-/H2/H2+ balance (see implementation doc §20), so gasc is aligned with it.
        s = replace_exact(s, "        const double g4 = std::pow(10.0, cmol[0]);\n        const double g5 = std::pow(10.0, cmol[1]);",
            "        // HSEMHS shim (FIX_GASC_H2): same H2 constants as pe_pg\n" *
            "        const double g4 = pe * std::pow(10.0, acota(cmol[0], -30.0, 30.0));\n" *
            "        const double g5 = pe * std::pow(10.0, acota(cmol[1], -30.0, 30.0));", 1)
    end
    s = replace_exact(s, "while ((dif >= PREC) && (it < 100)) {",
                      "while ((dif >= prec_) && (it < 100)) {", 1)
    s = "// GENERATED by deps/witt/build.jl from witt_eos_cpp.cpp (sha256 $(bytes2hex(sha256(read(SRC))))), FIX_GASC_H2 = $(FIX_GASC_H2). Do not edit.\n" * s * C_API_HEAD * C_API
    write(SHIM, s)
end

function build()
    mkpath(DIR)
    make_shim()
    ext = Sys.isapple() ? "dylib" : "so"
    cxx = get(ENV, "CXX", Sys.isapple() ? "clang++" : "g++")
    out = joinpath(DIR, "libwitt.$ext")
    cmd = `$cxx -O2 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off -o $out $SHIM`
    println(cmd)
    run(cmd)
    write(joinpath(DIR, "compiler.txt"), readchomp(`$cxx --version`))
    println("built $out")
end

abspath(PROGRAM_FILE) == (@__FILE__) && build()
