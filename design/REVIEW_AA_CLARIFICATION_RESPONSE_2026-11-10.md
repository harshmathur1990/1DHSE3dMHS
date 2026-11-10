# Review of Claude's manuscript clarification response — 10 November 2026

Paper HEAD `c2bf621` adds the previous review only: `hsemhs.tex` and `hsemhs.pdf` remain at `5cdd76e`. Reviewed `CODEX_BRIEF_paper_round2_answers_2026-11-10.md` and its stored-result summaries. No scientific runs, tests or manuscript edits performed.

## Decision

The clarification response resolves most questions and gives a concrete closing revision. Incorporate those answers and the accepted round-2 corrections into the paper. No further broad experiment programme is necessary. The current PDF is still the earlier revision, including the overlong abstract and incorrect threshold-independence statement.

## Q1: accept the clarified interpretation, correct two algebraic slips first

The distinction between physical horizontal derivatives and derivatives on τ surfaces is now clear. Calling Rv an equilibrium-equivalent residual rather than the actual Cartesian vertical force is appropriate. Please include the concise continuous/discrete appendix proposed in the response, with the following corrections:

1. Given `F_ray/P = −∂ray lnP − Wn`, the rearrangement is

   `∂ray lnP = −Wn − F_ray/P`,

   **not** the plus sign written in the response. The later stated identities for Rh and Rv are consistent with the corrected minus sign.

2. For general tx,ty, the component expansion is

   `∂ξ f|τ = (1+tx zξ)∂x f + ty zξ ∂y f + zξ ∂z f = ∂x f + zξ ∂ray f`.

   The response's first expansion omits `ty zξ ∂y f`. Its compact expression is correct. Use the general form, since the method supports nonzero ty.

Define positive downward thickness Δz, the node/face evaluation and the trapezoidal approximation explicitly. Do not present approximate discrete component relations as exact algebraic identities. A Jacobian/transpose listing remains unnecessary.

## Q2: alignment depth explains the poor mass-HSE baseline

This is an important scientific clarification: the 57→20 km gain partly measures recovery from a deeper initial alignment prescription. State this prominently beside that result and in the discussion. The existing offline re-alignment is adequate for a diagnostic, provided its exclusions and evaluation support are disclosed.

Before quoting 21.5/22.1 km alongside 56.6 km, establish the number of excluded columns and, if available from the stored analysis, use a common valid-column mask across depths. The original alignment and offline interpolation/exclusion policies differ (the deepest offline result is about 61 km). Do not attribute the entire numerical difference to alignment until that comparison is controlled. No new solver run is required simply to explain this limitation.

## Q3: remove gauge offsets; distinguish construction from mapped defects

Agreed: drop −51/−120 km as physical reconstruction errors. They are pre-gauge offsets. Retain pressure/boundary differences and gauge-aligned shape accuracy.

Label the ≤5e-4 value as the fine geometric construction check. Do not replace it with the tiny `sqrt(Φ/ΣW)` numbers as if these were a resolution-invariant physical force norm; that normalisation was already identified as grid dependent. State that mapping produces nonzero discrete defects and give the existing verification/refinement evidence without conflating the norms.

## Q4–Q5: the additional response evidence is useful

The stored-map response RMS (HSE 11.4/29.4 km versus MHS 5.2/14.0 km) directly supports reduced sensitivity to the prescribed perturbations. Report those numbers and keep “consistent with horizontal coupling” as the mechanism interpretation.

Describe the Gaussian low-pass result as smoothed and residual components, not hard wavelength bands “>4 px” and “<4 px”: a Gaussian filter has no sharp spectral cutoff and those components are not orthogonal. The total response alone is enough for the main text.

The clarified smoothing widths, realised amplitudes, illustrative error model and fixed masks are adequate. There is no need for more noise runs. Preserve the distinction between scatter over three realisations and population uncertainty.

## Provenance and final handoff

Use the corrected explicit affected/unaffected input lineage. Do not blanket-invalidate independent equilibrium controls or the separately generated EOS-ray inputs.

Next action: apply the accepted prose/figure corrections and these clarifications, regenerate the PDF, and present that revised version for the closing scientific read. Keep metadata requiring the author (abundances, author list, acknowledgements, archived release identifier) clearly identified. Do not leave the scientific corrections pending merely because publication metadata is unfinished.

The scope is now a defined manuscript revision, not a new research cycle. The round-2 review remains applicable until these changes appear in the manuscript.
