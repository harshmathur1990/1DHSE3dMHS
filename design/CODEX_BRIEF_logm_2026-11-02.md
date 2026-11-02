# Codex brief: log10 column-mass modes (branch `log10m-mode`, 2 Nov 2026)

**What changed.** New `run.mode` `column-hse-logm`, `mhs-poisson-logm` and embedded modes `:column_hse_logm`,
`:mhs_poisson_logm` (`PoissonMassOptions(logm_cut)`, `ColumnMassOptions`). `PoissonTau` takes `mass = true`: χ → ρ, τ → m in
`hse_tau!` and `evaluate!` (a = μ m/ρ, and its pressure derivative), so P = μ g m in column HSE; τ becomes an output
(`logtau_mass`), used for the gauge and z(τ = 1). New input key `log10m` (SI, or `units = "g cm-2"`), output group
`/mass_grid`, `make_input_mhs_logm.py`. With μ < 1 the axis is m along the line of sight, as τ in the τ modes.

**Executed.** Full suite on hyades20 (exit 0). Full cube: column on hyades10 (14 s), Poisson on hyades13 (187 s, 5 steps,
qualified). Results and limits: `HSEMHS/reports/logm_2026-11-02/README.md`.

**Uncertain.** (1) 5 qualified steps against `max_iter` at 25 on the τ axis, not a controlled comparison: this input's m uses
the snapshot's true ρ and P. (2) Line-of-sight m for μ < 1 is a choice; m is usually quoted vertically. (3) The top closure
P = g m₁ ignores what lies above the axis; above the cut P/(g m) in Bifrost spans 0.35–1.95 at log m = −2. (4) τ on the
mass axis is one trapezoid per level with no substeps, so coarse levels through hot-over-cool intervals are less accurate
than the heights of column HSE. (5) `eos_cells` reports ln of the axis; its log text changed from "log τ" to "ln axis" (it
was already ln τ under that label).

**Please review.** The `mass` branches in `evaluate!` and the probes; `logtau_mass` and its closure; the gauge after the
solve; whether a separate `PoissonMassOptions` type is the right guard; `check_fields!` on the computed τ.
