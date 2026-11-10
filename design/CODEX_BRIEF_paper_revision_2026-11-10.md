# Codex brief: method paper, revision after REVIEW_AA_MANUSCRIPT_2026-11-10 (10 Nov 2026)

Paper: `/mn/stornext/d9/data/harshm/HSEMHS-Paper` branch `results-bifrost-mu1`, commit 5cdd76e (`hsemhs.pdf`, 8 pages incl.
appendices and references). Code: branch `consistent-inputs`, commit 75499a8; runs on main 6e0a52b.
Report: `HSEMHS/reports/paper_252_2026-11-10/README.md`. The user accepted these decisions: three contributions instead of seven
bullets; inversion use labelled as interface capability; the suggested title; a bounded input-error test; length about 6 pages
plus appendices (not a hard limit).

## What changed (by review item)

- **A1 objective.** Sect. 2.3 now defines R_x (levels, vertical balance substituted) and R_z (face-integrated), states
  F_x/P = R_x − (∂_x z) F_z/P away from equilibrium, calls Φ a vector residual in transformed coordinates, and keeps the
  Cartesian force diagnostic as the assessment. Weights described as an approximate scaling; relative residual as a choice.
- **A2 inclined geometry** (new Sect. 2.1–2.3): footprints vs image plane, rays x = ξ + t z, local-frame B, dz/dlnτ = −μτ/χ,
  ∂_z = ∂_ray − t·∇_h, W_n with L_z + t·L_h, the vertical-balance column residual used (R_v = R_ray − Δz t·R̄_h), cut at vertical
  equivalent log τ_LOS = −4 − log10 μ. Detailed discrete equations referred to the code documentation, not derived in an appendix.
- **A3 column mass.** "removes the explicit opacity dependence from the hydrostatic relation and geometric integration";
  P = μ g m; m defined; cut equivalence given as a median with its 1–99 % range in this snapshot.
- **A4 solver/stopping.** E (regularisation on all unknowns, in the step not Φ), finite-difference ∂lnχ/∂lnP, switch to column
  blocks, 100-step budget stated, qualification interval described, μ = 0.5 "undecided" in abstract, table, figure (open symbol)
  and conclusions. "pred" relabelled as the accepted step's prediction. Borrero rule: described as applied to this trajectory
  (mean absolute change with the per-step mean removed); "stricter" removed. The 2021 criterion text was read through ar5iv
  earlier (dz = 12 km; "more than half a vertical grid point between two consecutive iterations"), not re-verified today.
- **A5 headline.** Abstract and conclusions now lead with the depression and state the unchanged/worse rms and lower
  correlation; "set by granulation" removed; masks counted per μ; new mask-threshold (300/500/1000 G) and 6×6 block bootstrap
  (`depression_sensitivity.txt`).
- **A6 cut.** Presented as an empirical result for this field; −5 invalid, −6/no cut stalled = failure outcomes; vertical force
  imbalance shown (MHS trades F_z 0.001 → 0.014 ρg for F_h 0.072 → 0.018 ρg).
- **A7 boundary/gauge.** Top closure, fixed cut pressure, total-pressure alignment without tension, periodicity, the alternating
  null modes suppressed by E (Appendix B).
- **B literature.** Added Borrero et al. 2023 (Paper III) and 2024 (Paper IV); Puschmann 2010 described as minimising ∇·B and
  static-force deviation (title corrected). Runtime comparison with FIRTEZ removed; 504² claims removed. All DOIs checked on
  Crossref; the Wittmann 1974 DOI and the FIRTEZ-dz DOI were wrong before and are corrected.
- **C validation.** Appendix A summarises the existing known-equilibrium controls (SOLVER_TIME_DESIGN §6.12–6.14, incl. the
  rotated φ = 30° case). New input-error test (Table 2): correlated T/B errors, 2 amplitudes × 3 realisations, μ = 1 own τ.
- **D figures/table.** Fig. 1 separate |B| colour bar and panel letters; Fig. 2 restricted to log τ ≥ −5 with four curves;
  Fig. 3 F_h and F_z, β and fraction β<1 on separate axes, cuts marked; Fig. 4 (viewing) markers only, μ = 0.5 open; Fig. 5
  signed depression; Table 1 status column. Running title/author set.
- **E provenance.** Paper README rewritten; `data/` holds metrics, perturbed metrics, sensitivity and step history.

## Executed (hyades11, 12, 15, 17, 18, 20, 21; 3 ranks × 32 threads, 24 on hyades18)

- Perturbed inputs (`make_perturbed_inputs.py`) and 12 runs (column HSE + MHS); force diagnostics; `perturbed_metrics.py`.
- `depression_sensitivity.py`; figures regenerated.

## Results

| input errors | column HSE z(τ=1) rms / depression recovered | MHS | MHS status |
|---|---|---|---|
| none | 20.7 km / 74 % | 22.6 km / 93 % | converged (16) |
| small (2 % T, 10 % B) | 23.6 / 74 % | 23.2 / 94 % | converged (17–19) |
| large (5 % T, 25 % B) | 35.9 / 77 % | 26.9 / 90 % | converged (16–37) |

## Not done / open

- Column HSE on column mass is much worse than on τ (57 vs 21 km); not explained (`\todo` in Sect. 4.1).
- Abundance source of the Wittmann table not identified in the code (log number fractions, He/H ≈ 0.09).
- The inclined Jacobian/transpose is not written out in an appendix (referred to code docs) — say if this is insufficient.
- Embedded-API test fixture is cropped from the deleted scrambled input; its generator points at a missing file.
- Co-authors, acknowledgements, Zenodo DOI.

## Please review

1. Whether Sect. 2.3's description of the residuals and the inclined variant is now accurate to `src/PoissonTau.jl`.
2. Whether the perturbation model (Gaussian-correlated, 2 px / 0.5 dex, multiplicative per component) is defensible for the
   claim made, and whether the claim "horizontal coupling damps the response of the heights to pixel-scale errors" is supported.
3. Whether the abstract/conclusions are now balanced against Table 1 and Table 2.
4. Remaining over-claims anywhere in the manuscript.
