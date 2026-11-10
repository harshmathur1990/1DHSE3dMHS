# Codex brief: closing manuscript revision (after REVIEW_AA_MANUSCRIPT_ROUND2 and REVIEW_AA_CLARIFICATION_RESPONSE), 10 Nov 2026

Paper: branch `results-bifrost-mu1` (`hsemhs.tex`, `hsemhs.pdf`, 9 pages incl. three appendices; abstract now below the
300-word limit, no class warning, no undefined references). Code/report: branch `consistent-inputs`. No new solver runs;
two stored-state diagnostics were controlled/relabelled as asked.

## Clarification-response items

- **Q1 slips corrected** in the new Appendix C: ∂_ray ln P = −W_n − F_ray/P (the answers had +); the general expansion
  ∂_ξ f|_τ = (1 + t_x z_ξ)∂_x f + t_y z_ξ ∂_y f + z_ξ ∂_z f. Δz = z_k − z_{k+1} > 0 defined; R_h on nodes (centred periodic
  differences), R_ray trapezoidal on intervals and stated to approximate Δz F_ray/P (not an identity); R_v ≈ Δz[F_z −
  (t·∇_ξ z)F_ray]/P; the equivalence F = 0 ⟺ R_h = R_v = 0 stated for the continuous limit. Sect. 2.3 summarises this.
- **Q2 controlled** (`alignment_depth.py/.txt`, rewritten): the offline re-alignment now applies the production policy of
  `align_ptot!` exactly (interpolation in the column, hydrostatic extrapolation below the bottom level), so every column is
  kept at every depth; the bottom-level rows reproduce the stored states exactly (56.6 and 20.7 km). Mass axis: reference
  at log m 2.6 / 2.2 / 1.9 / 1.7 / 1.4 / 1.0 → 56.6 / 39.0 / 21.5 / 22.1 / 24.2 / 26.6 km (extrapolated columns 31752 at
  2.6, none elsewhere). τ axis: 1.4 / 1.0 / 0.5 / 0 / −1 → 20.7 / 20.7 / 21.5 / 23.8 / 26.5 km. Sect. 4.1 explains the
  baseline, Discussion and conclusion 4 say MHS corrects the starting column offsets, not that mass is more accurate.
- **Q3**: offsets removed from Sect. 3 and App. A (translation not observable under the gauge); the 5e-4 is labelled as the
  fine-construction balance; the mapped equilibria are described as leaving a small discrete residual that decreases with
  refinement (no grid-dependent √(Φ/ΣW) number quoted).
- **Q4–Q5**: response rms 5.2/14.0 (MHS) vs 11.4/29.4 km (HSE) reported in Sect. 4.5 with "consistent with" for the
  mechanism; the split is relabelled Gaussian-smoothed/residual in the report and not used in the text. Illustrative error
  model, kernel standard deviations, 1/e autocorrelation at 4 px, realised amplitudes, what is not modelled, scatter over
  three realisations.
- **Provenance**: paper README lists affected/unaffected products explicitly.

## Round-2 corrections

1. Threshold: "for a 500 G threshold … ; the improvement persists at the three thresholds tested" (abstract, Sect. 4.1,
   conclusion 1). 2. Abstract rewritten (URL, threshold sweep, column mass, low-β detail moved to the body). 3. Null modes:
   E keeps steps out of the alternating directions, does not remove existing components; measured amplitudes of the
   alternating cut-height components: < 0.05 km at μ = 1 (reference 0.01–0.03 km), < 1 km at μ = 0.5 (App. B). 4. Fig. 1
   bottom titles "Δz: 21 km rms"; `placeins` keeps floats before the Discussion (page 6 is a float page). 5. Stopping:
   b^T δv/Φ before scaling/backtracking (Sect. 2.6, Fig. 5 caption); E written out: per pixel 1e-2 κ₁² Σ_k w_k W̄_k² for
   the cut height and 1e-2 κ₁² w_k for ln P, w_k = Δz̄_k H̄_k², κ₁ = 2π/(N_x Δx) (`tp_weights!`).
   Also: "recovers the geometry of magnetic structures" → "mean depression / height contrast"; bootstrap described (42² px
   ≈ 4 Mm blocks, 2000 paired resamples, conditional); "total imbalance" replaced by the two components; weighting prose
   separates force-density normalisation (H̄²) from the volume element (Δz̄); mass above the top: m₁ > 0, boundary pressure
   μ g m₁; cut/photosphere sentence (log τ −4 at median 550 km, near the temperature minimum); data/code availability
   paragraph.

## Open (need the author)

Co-authors, acknowledgements, abundance source of the Wittmann table, archived release DOI (`\todo` in the PDF).

## Please review

1. Appendix C: signs, the stated approximation level, and whether Sect. 2.3 now matches it.
2. Sect. 4.1 column-mass paragraph and conclusion 4: does the controlled re-alignment support the wording?
3. Abstract and conclusions against Tables 1–2 (final consistency read).
4. Anything still over-claimed or inconsistent.
