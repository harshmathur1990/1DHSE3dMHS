# Codex brief: answers to REVIEW_AA_MANUSCRIPT_ROUND2_2026-11-10 (Q1–Q5, provenance), 10 Nov 2026

Paper unchanged since `5cdd76e` (branch `results-bifrost-mu1`). No new solver runs: Q2 and Q4 use the stored states of
`out/paper_252_2026-11-10` (scripts and outputs in `HSEMHS/reports/paper_252_2026-11-10/`: `alignment_depth.py/.txt`,
`perturbation_response.py/.txt`, run on hyades). The confirmed corrections 1–5 are accepted as stated; they wait for the
user's go-ahead together with the wording below.

## Q1. Inclined coordinates and the vertical-balance residual (`src/PoissonTau.jl`, `lorentz_tau!`, `evaluate!`)

Coordinates: footprint (ξ, η) and level τ; the ray is x = ξ + t_x z, y = η + t_y z, z = z(ξ, η, τ), t = tan θ (cos φ, sin φ).
For any f(ξ, η, τ), with ∂_ray ≡ ∂_z + t·∇_h (derivative along the ray per unit height, at fixed footprint):

  ∂_ξ f|_τ = (1 + t_x z_ξ) ∂_x f + z_ξ ∂_z f = ∂_x f + z_ξ ∂_ray f,  so  ∂_x f|_z = ∂_ξ f|_τ − z_ξ ∂_ray f.

So ∇_h in the paper is the horizontal derivative **at fixed physical height** (equal to the footprint derivative at fixed
height, because ξ and x differ by t z), and ∂_z|_(x,y) = ∂_ray − t·∇_h|_z. The code computes ∂_ray B by differences down
the ray (Δf/Δz), ∇_h|_z B = ∇_ξ B|_τ − ∇_ξ z|_τ ∂_ray B, then ∂_z = ∂_ray − t·∇_h|_z. The factor (1 + t·∇_τ z) Codex saw is
the same operator written in the τ-surface derivatives, ∂_z = (1 + t·∇_τ z) ∂_ray − t·∇_τ, used for the Jacobian (δL).

Residuals. With F = −∇P − ρg ẑ + L and the ray component F_ray = F_z + t·F_h:
- along the ray ∂_ray ln P = −W_n + F_ray/P, W_n = (ρg − L_z − t·L_h)/P;
- R_h = −∇_ξ ln P|_τ − W_n ∇_ξ z|_τ + L_h/P = F_h/P + (∇_τ z) F_ray/P (exact identity, any state);
- R_ray (per interval, trapezoid) = Δ ln P − ∫ W_n a d ln τ ≈ Δz F_ray/P (Δz > 0 the interval thickness);
- R_v = R_ray − Δz t·R̄_h ≈ Δz [F_z − (t·∇_τ z) F_ray]/P.

R_v is not a discretisation of F_z/P away from equilibrium (it has the extra (t·∇_τ z) F_ray term, which vanishes at μ = 1);
it is an equivalent equilibrium residual: R_h = 0 and R_v = 0 give F_h = −(∇_τ z) F_ray and F_z = (t·∇_τ z) F_ray, hence
F_ray = F_z + t·F_h = 0 and F = 0, for any slope. The name "vertical balance" means only that the t·F_h part of R_ray is
removed. Discretely R̄_h is the mean of the interval's two nodes and the integral is trapezoidal. Proposal: a short
appendix with these four lines (continuous transformation and the discrete R_h, R_ray, R_v); no Jacobian.

## Q2. Why column HSE on column mass is worse (57 vs 21 km)

Established: the alignment depth. Production aligns each column so that P + B²/2μ₀ equals the median total pressure of
the **bottom level of the axis**. On the τ axis that is log τ 1.4 (truth median z −83 km); on the mass axis log m 2.6
(z −581 km, about 0.9 dex in m below τ = 1 at median log m 1.68). Horizontal total-pressure contrasts at that depth (the
deep convective structure is not hydrostatic in each column) map into column offsets that propagate to τ = 1.

Evidence (`alignment_depth.txt`; stored column-HSE state, re-aligned offline, gauged):
- the error is in the heights of the mass levels themselves (rms 59–71 km at log m −0.5 … 1.5), not in the part from a
  mass level to τ = 1 (7 km from log m 1.5);
- aligned at log m 1.9 or 1.7 instead, column HSE on mass gives 21.5 / 22.1 km (corr 0.75 / 0.71), the same as on τ
  (20.7 km, corr 0.74); at log m 2.2 39 km, at 2.6 61 km (offline, columns not reaching the reference excluded).

Consequence for the paper: the 57 → 20 km gain is a property of this baseline (HSE with deep alignment), not evidence
that the mass formulation is more accurate. With comparable alignment depth the two HSE baselines agree, and MHS on mass
(19.5 km) and on τ (22.6 km) are close. What it does show: the MHS solve removes most of a column-offset error introduced
by the starting alignment. Proposed wording accordingly; optional (not run): rerun column HSE on mass with the alignment
at a fixed log m near τ = 1, one run of a few seconds, to replace the offline number.

## Q3. Verification offsets (−51 / −120 km) and the defect values

The offsets were measured in the solver's internal frame, before the gauge: mean z(τ_LOS = 1) of the run minus that of
the truth, both in their absolute frames (`scripts/viewing/measure_analyse.py`, `off`). Under the declared gauge they are
zero, so they are a translation, not a reconstruction error. The genuine boundary-model differences are the ΔlnP 0.05
(μ1) and 0.056 (μ0.3) near the cut and the depression/interior shape, which match the truth-boundary solution. Proposal:
drop the offsets from Sect. 3/App. A and keep the ΔlnP.

Defects: 5 × 10⁻⁴ (max |R_z|/ρg, rms 2 × 10⁻⁵) is the centred-difference balance of the fine geometric construction
(dz 250 m); the mapped coarse-τ defect is √(Φ/ΣW) with the run's weights, 3 × 10⁻⁸ … 5 × 10⁻⁷ depending on the grid
(SOLVER_TIME_DESIGN §6.12). The coarse controls are mapped equilibria with that discrete defect, not exact discrete
equilibria; App. A will say so.

## Q4. Input-error response (`perturbation_response.txt`)

rms of z1(perturbed) − z1(unperturbed), mean of three realisations, split by a Gaussian low-pass of σ = 4 px:

| | column HSE: total / > 4 px / < 4 px | MHS: total / > 4 px / < 4 px |
|---|---|---|
| small | 11.4 / 4.8 / 8.7 km | 5.2 / 2.8 / 3.8 km |
| large | 29.4 / 12.2 / 22.3 km | 14.0 / 7.7 / 9.9 km |

The MHS response is half of the column-HSE response, and the reduction is larger at small scales (factor 2.3 below 4 px
against 1.6–1.7 above). This supports "MHS reduces the response of the heights to these perturbations, most at small
scales"; the mechanism (horizontal coupling) remains an interpretation. Proposal: report the response numbers and use
"consistent with" for the mechanism.

## Q5. Perturbation parameters

Accepted: 2 px and 0.5 dex are smoothing-kernel standard deviations (realised horizontal autocorrelation falls to 1/e at
4 px, as expected for Gaussian smoothing); the amplitudes are illustrative, not calibrated inversion errors; "response
function width" removed as a justification. Realised: rms Δ ln T 0.020 / 0.050; rms Δ|B|/|B| for |B| > 500 G 0.092 / 0.227
(componentwise fractional noise σ_B 0.10 / 0.25, multiplicative, so zero components stay zero; no additive weak-field
noise, no correlated inclination/azimuth errors, no disambiguation failures). ± values are scatter over three
realisations. Masks fixed from the unperturbed input (already so).

## Provenance (README invalidation)

Affected (lineage `IO/input-mhs-ltau.nc`, `_xy`, `out/ltaucut/input-ltau-xy.nc`, x/y-swapped MULTI3D τ): the `ltaucut`
study, the release `out/modes` ltau runs, `reports/mu_mismatch_2026-11-02` (halpha configs), the first paper drafts, the
τ side of comparisons in `reports/logm_2026-11-02`, and the embedded-API fixture (a cropped test case; numerically valid
as a test, not as physics). Not affected: the Grad–Shafranov controls (`make_mhs_control.jl`), the viewing-angle inputs of
`reports/viewing` (`make_slanted_tau.jl`, T, B and τ built from the Bifrost netCDF with our opacity; only the log τ axis
values were taken from the old file; outputs deleted with `out/view`), and the 504² column-mass input
(`IO/input-mhs-logm.nc`, from the netCDF). Proposal: replace the README sentence with this list.

## Proposed next revision (after the user's go-ahead)

Corrections 1–5 of the review; Q1 appendix; Q2–Q5 wording; abstract < 300 words; Fig. 1 titles; floats before the
references; E defined with its actual diagonal; bootstrap described (42² px ≈ 4 Mm blocks, 2000 paired resamples); no
"total imbalance" claim; mass-above-top sentence; photosphere/cut sentence (user question: the force balance is solved
below log τ −4 ≈ 550 km, the temperature minimum); availability statement. Metadata (authors, acknowledgements,
abundance source, Zenodo DOI) needs the user.
