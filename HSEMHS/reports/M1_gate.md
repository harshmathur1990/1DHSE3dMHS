# M1 gate report (16 Oct 2026)

Build: `julia deps/witt/build.jl` (clang 17, `-O2 -ffp-contract=off`, FIX_GASC_H2 = on); table: `scripts/run.jl config/eos_table_bifrost385.toml` (12 threads, 326 s).

| Gate 1 row (spec §4) | Result | Threshold | Pass |
|---|---|---|---|
| Python ↔ C++ shim ↔ Julia direct solver, 500 points, prec 1e-10 | max rel 8.5e-12 | 1e-8 | yes |
| Table vs model, 1e6 withheld points: ln ρ | 5.2e-7 | 1e-5 | yes |
| … ln nₑ | 1.4e-6 | 1e-4 | yes |
| … ln χ₅₀₀ | 3.2e-5 | 1e-4 (doc §7.9) | yes |
| ∂ln ρ/∂ln P vs FD step sweep, 1e4 points | 6.2e-7 | 1e-3 abs | yes |
| Unflagged non-converged calls / failed table nodes | 0 / 0 | 0 | yes |
| Opacity port vs Python `cop` (1000 points) | 4.4e-16 rel | 1e-12 | yes |
| Shim background partials vs Python (same gasc fix) | 4.6e-13 rel | 1e-8 | yes |
| Hot branch vs direct where both converge (2000 points, full envelope) | 2.6e-9 | 1e-8 | yes |
| Opacity within 5 % of RH background opacity at 10 photospheric points | ≤ 2.3 % for T ≤ 6000 K vs RH as is; ≤ 4.3 % at all 10 points vs RH with its two artifacts removed (H continua truncated by model-atom grids; H⁻ density including protons); remainder = H₂⁺ data (ATLAS fit vs Bates 1952). `reports/rh_opacity_investigation.md` | 5 % | yes (accepted by user, spec §10) |

Table: 1733 × 1254 nodes, envelope T = 2172 K – 1.61 MK, log₁₀P = −10.5 – 14.5 (Gate 0 proxy runs ±1 dex), 244 breakpoint temperatures, T_sw = 15.1 kK, 714 nodes flagged EOS_ROBUST_FALLBACK (cold/high-P corner). Lookup: ~5 ns, allocation-free. Full report: `reports/gate1_table.md`.

gasc H₂ correction approved (spec §10). RH opacity comparison accepted with the artifact correction (spec §10). M1 closed.
