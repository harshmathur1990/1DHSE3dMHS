# A&A manuscript review, revision 2 — 10 November 2026

Reviewed paper commit `5cdd76e`, the revised eight-page PDF (including appendices), LaTeX/bibliography, Claude's revision brief, and code/report commit `75499a8` (current code-repository HEAD `de311c5` adds review documents). Inspected the perturbation generator, mask/bootstrap analysis and saved metrics. No scientific runs or tests performed; no manuscript/scientific source edited.

## Assessment

**This is a substantial improvement and now reads as a scientific methods paper.** Most of the previous major concerns are addressed: the actual residual is distinguished from Cartesian force, inclination and mass coordinates are defined, boundary assumptions and solver details are disclosed, the abstract acknowledges worse global morphology, the unresolved inclined result is marked, the unfair runtime comparison is removed, and the bounded sensitivity experiment is complete.

Do not restart the earlier broad validation programme. The next revision should correct a small set of remaining factual/interpretive problems, answer the clarification requests below, fix the figures and finish the publication metadata. I would not submit this exact PDF yet, but the main scientific narrative no longer needs rebuilding.

## Confirmed corrections

### 1. Threshold-independent percentages are contradicted by the new analysis

The abstract says that 93/91/102% recovery holds “for any strong-field threshold between 300 and 1000 G”; conclusion 1 says the result is independent of threshold. Only three thresholds were tested, and the percentages do change:

| μ | MHS at 300 / 500 / 1000 G |
|---|---|
| 1 | 86 / 93 / 96% |
| 0.707 | 87 / 91 / 90% |
| 0.5 | 100 / 102 / 102% |

The robust conclusion is that **MHS improves the strong-minus-quiet contrast relative to HSE at all three tested thresholds**. The quoted 93/91/102% values specifically use 500 G. Replace the abstract/conclusion statements accordingly. Do not infer a continuous range of thresholds from three samples.

Suggested sentence: “For a 500 G strong-field threshold, the reconstructed contrasts are 93, 91 and 102% of the reference values; the improvement over column hydrostatics persists at the three thresholds tested.” The unresolved μ=0.5 qualification must stay attached to the interpretation.

### 2. The abstract exceeds the template limit

The build log reports approximately 372 words against a 300-word maximum. This can be fixed editorially without losing the scientific balance. Retain the physical question, prescribed-input nature, selected contrast gain, unchanged/worse global error, input-sensitivity result and unresolved inclined case. Move the exact threshold sweep, full angle-by-angle percentages, low-β failures and repository URL to the body/availability statement. Aim below 300 rather than exactly at the boundary.

### 3. Step regularisation does not suppress existing state gauge amplitudes

Sect. 2.4 says the alternating cut-height null modes are “suppressed” by the small regularisation. E is in the step operator, not a state penalty. It makes the linear system definite and discourages increments along null directions; it does not in itself select or remove arbitrary null amplitudes already in the starting c map. The uniform gauge is explicitly fixed afterwards.

Replace the state-suppression claim with the narrower step statement and document what actually happens to alternating components in the starting/final state. No scientific code change is requested unless Claude identifies a different implementation than the one inspected.

### 4. Figure 1 still has a layout defect; the revision fixes several other figures well

In the actual PDF, the bottom-row titles run together: `|B| at logτ=0difference, rms 21 kmdifference...`. Shorten each title to `(f) Δz: 21 km` or move RMS values into the caption. Keep the panel letters and separate colour bars; both are useful improvements. The common height/error range is appropriate. A zoom is optional, not an additional requirement.

Figures 2–5 are clearer: fewer curves, separate force/β axes, open unresolved symbol, signed depression and separate convergence panels. Please preserve those changes. Figures 4 and 5 currently float to page 7, after references on page 6; adjust float placement so scientific results appear near their discussion and before the bibliography. Eight pages is not itself a problem; sparse page layout is.

### 5. A precision detail remains in the stopping description

`pred` is computed from the damped, inexact GN direction before cap/backtracking. “Predicted decrease of the accepted step” can imply the actual accepted αδ model decrease, which is separately computed in the code. Use “direction-based decrease indicator associated with each accepted iteration,” or define `bᵀδ/Φ` and say it is evaluated before scaling. The final qualification interval then has its distinct, properly explained role.

Appendix B's `diag(…)` is a placeholder in an equation. Either give E's actual diagonal and units/scales, or remove that incomplete equation and give a precise prose description with an immutable source reference. Do not leave ellipses as the definition of a numerical method.

## Clarifications for Claude before recommending further changes

Please answer these explicitly in a short response document or in the next brief. I am not treating these as demonstrated implementation errors.

### Q1. What exactly is meant by the inclined “vertical projection”?

Sect. 2.3 gives `∂z|x,y = ∂ray − t·∇h` without defining whether ∇h is at fixed physical z or at fixed τ/footprint. Those choices differ. The code's transformation includes factors involving `1+t·∇τ z`. Also, `Rv=Rray−Δz t·Rbar_h` is called a vertical projection; please show its derivation from the Cartesian forces and specify whether it is an equivalent equilibrium residual or literally a discretisation of Fz/P away from equilibrium.

The μ=1 residual explanation is now correct and useful. A short appendix with the **continuous inclined transformation and exact discrete residuals** is sufficient; the full Jacobian/transpose is not required for this A&A audience. The missing item is physical coordinate clarity, not implementation detail.

### Q2. Why is column-mass HSE much worse, and what part is alignment?

The 57→20 km result remains prominent while its baseline has an `explain` TODO. At m, the pressure is fixed by μgm under HSE, but relative column offsets are supplied by the total-pressure alignment. Is the poor z(τ=1) baseline mainly the offset prescription, different fitted domains, thermodynamic/mapping differences, or another known cause?

First inspect existing maps/diagnostics and explain what is established. If existing evidence does not isolate the cause, say it remains unexplained and limit the claim to improvement over this specific HSE-plus-alignment baseline. Do not present it as proof that the mass formulation is intrinsically more accurate than the τ formulation. A narrowly targeted diagnostic may be justified after the answer; a new suite is not requested now.

### Q3. What common height gauge is used for the verification's −51/−120 km offsets?

The text fixes mean z(τ=1)=0, yet Sect. 3 and Appendix A call near-uniform offsets a boundary-model bias. A uniform height translation is not observable in this problem. Are these numbers measured before gauge alignment to an external reference surface, or after a different anchoring convention? Please state that convention and distinguish the genuine cut-pressure/interior-shape differences from a removable translation. If the offsets disappear under the declared gauge, do not count them as a physical reconstruction error.

Also distinguish the fine geometric construction's force defect (≤5e-4) from the mapped coarse-τ residual: earlier records showed those are different diagnostics. No need to reopen the first-order-defect investigation; simply label which one is quoted. Avoid describing all coarse controls as exact discrete equilibria.

### Q4. What supports the causal input-error explanation?

The experiment supports: **for these prescribed correlated perturbations, at μ=1, MHS gives a smaller global height RMS error than HSE, especially at the larger amplitude.** That is a useful result.

The statement “horizontal coupling damps the response ... to pixel-scale errors” is a plausible explanation, but RMS error against truth mixes existing bias, perturbation response, changed boundary/weights and nonlinear trajectory. Do you already have the difference between perturbed and unperturbed reconstructed maps? If so, report its RMS or scale-dependent summary to support the response claim. Otherwise write “consistent with reduced sensitivity through horizontal coupling.” New runs are not needed merely to qualify this causal wording.

### Q5. How should readers interpret the perturbation widths/amplitudes?

The code smooths white noise with Gaussian kernels of widths 2 pixels and 0.5 dex. These are **smoothing-kernel standard deviations**, not identical to the resulting field's autocorrelation width. Please name them precisely. The independent multiplicative B-component errors also preserve zero components and scale differently from an inversion covariance with additive weak-field uncertainty, correlated inclination/azimuth errors and disambiguation failures.

Are 0.5 dex and the chosen amplitudes intended as illustrative choices or calibrated inversion errors? If illustrative, say so and remove “the width of a response function” as a universal justification. The scalar 25% parameter is a componentwise fractional noise amplitude, not necessarily a 25% RMS error in |B|. Explain the natural-log temperature perturbation similarly. Retain the fixed unperturbed masks and clarify that quoted ± values are scatter over three realisations, not well-established population error bars.

The perturbation test is adequate as a bounded sensitivity demonstration with these qualifications. It does not need expansion to all angles or a full synthetic inversion for this paper.

## Remaining scientific/editorial wording

- Replace “recovers the geometry of magnetic structures” with “improves the mean magnetic-region depression/height contrast” where no morphology-resolved improvement is shown. The body makes this distinction well; preserve it in the short abstract/conclusion.
- The bootstrap intervals are conditional spatial-resampling intervals for one snapshot, fixed masks and chosen blocks. Name the 42×42-pixel (~4 Mm) blocks, 2000 resamples and paired truth/reconstruction resampling. Do not imply uncertainty across solar regions or inversion noise; those are different experiments.
- Avoid “the total imbalance is lower” unless the combined norm was actually computed. Separate medians of |Fh| and |Fz| do not determine the median total vector norm. Reporting the component trade-off already makes the useful point.
- Sect. 2.4's weighting prose says “imbalance ... per unit volume.” Better distinguish volume weighting from force-density normalisation; the squared residual sum has dimensions and includes layer-median approximations. Do not revive the earlier exact Cartesian-norm equivalence.
- The mass-above-top convention deserves one unambiguous sentence: nonzero top m1 represents a prescribed overlying column mass, with its boundary pressure μgm1. It is not zero mass above the first level unless the coordinate origin/boundary is defined differently.
- Resolve abundance provenance, authors and acknowledgements. An archive DOI and a short data/code availability statement should identify the exact released code, corrected builders, configurations and machine-readable results. Local `/mn/...` paths alone are not access for other scientists.
- README's assertion that all pre-November-10 “504² τ” results are invalid appears broader than the repository report's invalidation of the swapped-MULTI3D-derived input. **Claude: confirm exactly which builders/products were affected.** Mark invalid products precisely; do not invalidate independent controls or EOS-built synthetic inputs without provenance evidence.

## Answers to the revision brief

1. **Residual description:** μ=1 is substantially corrected. The inclined projection/derivative notation needs Q1's clarification; a full implementation derivation is unnecessary.
2. **Perturbations:** defensible as an explicitly illustrative sensitivity model; not yet a calibrated model of inversion uncertainty. The observed benefit is supported; its causal explanation should be qualified unless response diagnostics already exist.
3. **Abstract/conclusions:** much better balanced, with the exception of the threshold-independence statement, overlong abstract and residual morphology wording.
4. **Remaining overclaims:** mainly those listed above, especially state null-mode suppression and gauge offsets counted as bias. They can largely be resolved through explanation/prose rather than more simulations.

## Bounded next revision

1. Answer Q1–Q5 and the provenance question; do not initiate new runs until an answer identifies a specific evidence gap.
2. Correct the threshold statement, shorten the abstract below 300 words, repair Fig. 1 and float placement, and complete the incomplete E definition.
3. Resolve the remaining scientific TODOs by an evidenced explanation or an explicitly limited statement; complete availability/author/abundance metadata.
4. Regenerate the PDF for a final consistency read. Keep the existing good figures, balanced results and small sensitivity test. This should be a focused closing revision, not another broad rewrite.
