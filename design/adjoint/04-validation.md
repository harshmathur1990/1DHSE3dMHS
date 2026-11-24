# 04 — Validation and cost (all phases; Phase 6 collects them)

## The check

For a mode, inputs a = (ln T, B), random output cotangent ȳ (all outputs: P, ρ, nₑ, χ, τ, z) and random input direction δa:

    lhs = ⟨ȳ, [F(a + h δa) − F(a − h δa)]/(2h)⟩       (two recorded solves, global inner product)
    rhs = ⟨adjoint(ȳ), δa⟩                            (one solve + one adjoint)
    err = |lhs − rhs| / max(|lhs|, |rhs|)

Several h (1e-3 … 1e-7 in ln T, matching scales in B) to see the truncation and round-off ends; the reported error is the
minimum over h. δa is smooth (low horizontal and vertical wavenumbers) and also localised (one column, one level) so that
single-column bugs show.

## Matrix

| Mode | Cases |
| --- | --- |
| column-hse-ltau, -logm | μ = 1; μ = 0.8 plane-parallel; μ = 0.8 inclined (azimuth 0, 30°); align ptot / none; B = 0 / B ≠ 0 |
| column-hse-z | anchors bottom / top / height; adapt on / off; τ top exp_tail / half_cell / zero; μ = 1 / 0.8 inclined |
| mhs-poisson-z | B ≠ 0 (the mode needs it); z_cut in the middle and near the top; anchor default |
| mhs-poisson-ltau, -logm | B ≠ 0; μ = 1 / 0.8 inclined; balance vertical; preconditioners fourier / block |

Each on 1, 2 and 3 ranks (x slabs), small synthetic atmospheres (`Synthetic.jl`; 12–24 columns per direction, 40–80
levels) and one real crop (en024048 385, 32 × 32 columns, the inversion's log τ grid −8 … 1.5).

## Targets

- Column modes, frozen-choice finite differences (forward rerun with the recorded choices): err ≤ 1e-8.
- Column modes against the solver itself: err ≤ 1e-4 for Heun (A1), ≤ 1e-8 for RK4 (no state-dependent steps).
- Poisson modes, solves converged tightly (gn_tol 1e-8, cg_tol 1e-10): err ≤ 1e-5; at production tolerances: report err
  and gn_decrement side by side.
- Rank independence: adjoint on 1 vs 3 ranks equal to 1e-12 relative.
- Invariances: z̄ uniform in a column of a τ mode → zero input cotangent (gauge); a cotangent on ρ only at one node →
  nonzero ln T̄ only in that column (column modes).

## Cost

Wall time of `adjoint_embedded!` / `solve_embedded!` per mode on the 252² × 96 inversion grid (hyades, 1 rank × 64 threads,
and 4 ranks). Targets: column modes ≤ 3; Poisson modes ≤ the cost of one full Poisson solve. Memory of the tape reported.

## Reports

`reports/adjoint_phase<N>.md` per phase: the table of cases with err, the h scan for one case per mode, timings, and any
deviation from this plan.
