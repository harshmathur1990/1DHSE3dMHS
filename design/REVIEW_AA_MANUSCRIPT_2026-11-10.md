# Scientific and editorial review for A&A — 10 November 2026

Reviewed the complete six-page `HSEMHS-Paper/hsemhs.pdf` visually, its LaTeX and bibliography, paper commit `575d2a4`, and the corresponding repository at `5f7c36b`. Cross-checked the November 10 report, metrics/figure/history scripts, input provenance, run configurations and relevant solver equations. No scientific runs or tests were performed. The manuscript and scientific source were not edited.

## Overall assessment

**This has a credible A&A methods-paper argument, but it is not submission-ready.** Its strongest scientific result is selective: force-balance reconstruction recovers the magnetic-region depression missed by column HSE, while overall optical-depth-surface accuracy changes little or worsens. Inclined geometry matters, and the column-mass formulation provides a useful complementary test. These are meaningful results even though the method does not recover the dynamic snapshot exactly.

The immediate problems are inaccurate descriptions of the objective/pressure assumptions, incomplete inclined-ray mathematics, overly broad conclusions, and limited validation of the use case advertised in the title. The paper also devotes too much introductory space to an implementation feature list while giving too little attention to the physical trade-off in its own results.

Make the scientific question explicit: **What additional atmospheric geometry and thermodynamic information can vector force-balance constraints recover from prescribed T and B on inversion coordinates, and where does this reconstruction cease to be reliable?** Runtime enables the science; it should not be the main scientific conclusion.

## A. Corrections needed before submission

### A1. The implemented residual is not simply the Cartesian force vector divided by P

Sections 2.1–2.2, Eq. (4), Eq. (5), abstract and novelty item 2.

The horizontal equation substitutes vertical equilibrium into the coordinate transformation. Away from equilibrium this substitution changes the residual. At μ=1, writing V=Fz/P, the continuum relation is

`Fx/P = Rx − (∂x z)|τ V`,

with an analogous y equation, whereas the code's Rx is the vertically eliminated horizontal residual. Rz is a face-integrated residual, not the same dimensional quantity as nodal Fz/P. Thus minimising the stated diagonal weights of these residuals is generally different from minimising a diagonal Cartesian force norm when the residual is nonzero. The distinction matters especially because the paper explicitly says an exact equilibrium need not exist.

**Required revision:** define the actual nodal and face residuals, their units and their relation to physical force components. Describe the objective as a weighted vector force-balance residual in transformed coordinates. Retain the independent Cartesian-force diagnostic as the physical assessment. If claiming a Cartesian force norm, derive the induced metric including cross terms and demonstrate equivalence; do not alter the code merely to match the prose.

Replace the strong volume-integral equivalence with a qualified statement about scaling. Layer medians of H and Δz are approximations and do not equal local normalisation/volume elements. “Dividing by P prevents trivial pressure reduction” also needs justification; it is a chosen relative residual, not a general mathematical guarantee against all pressure degeneracies.

### A2. Derive the inclined geometry where the scientific method is introduced

The paper tests μ=0.707 and 0.5 but presents only μ=1 height/force equations, then places inclination in the software section. This makes an essential physical contribution look like a configuration option.

Add a short main-text definition of footprint coordinates, local-frame field components, line-of-sight optical depth and vertical height. For slant t=tanθ(cosφ,sinφ), specify the rays, `dz/dlnτ_LOS = −μ τ_LOS/χ`, and the ray force component involving `Lz+t·Lh`. Derive the horizontal and vertical coordinate transformations and state which balance variant the reported runs use. Put the detailed inclined Jacobian/transpose in an appendix.

Explain whether the −4 cut is on vertical-equivalent optical depth or τ_LOS; the run configs use the vertical-depth convention by default. It is not the same τ_LOS level at every μ. Distinguish the observed image plane from the deprojected periodic footprint grid. Readers need to know what input field orientation and sampling are assumed.

### A3. Column mass removes explicit opacity from hydrostatics, not magnetic dependence from pressure

Abstract, Sect. 2.3 and discussion overstate this as “pressure follows from mass alone” or “pressure does not depend on opacity.” P=gm is the vertical **hydrostatic** relation; at inclination, the code's line-of-sight mass convention gives P=μgm. In the MHS solve pressure is still unknown and responds to Lorentz forces, density, geometry and boundary conditions. Opacity is absent from the m-to-z relation, but is still used to obtain τ500 and its unit surface.

Write: “The column-mass formulation removes the explicit opacity dependence from the hydrostatic pressure relation and geometric integration; optical depth is computed afterwards.” Define m, its units, its direction and the mass above the top boundary. Do not present agreement at τ=1 as an opacity-independent diagnostic.

The −0.5 mass cut and −4 optical-depth cut have similar median heights in this snapshot; they are not a general coordinate identity. State their spatial scatter or acknowledge that the fitted physical domains differ. Likewise RMS errors on uniformly sampled logm and logτ are not directly interchangeable averages over identical material.

### A4. Convergence is numerical stationarity, not physical accuracy

Keep the qualified test, but specify that the interval bounds the regularised local GN decrement for H+ereg. Eq. (6) hides the fixed positive step penalty in an ambiguous parenthesis and calls it damping “for the uniform shift”; the code applies ereg to pressure and height slots, not just the uniform height gauge. It is in the step model, not Φ.

The paper's main runs use `precond=switch`, whereas Sect. 2.4 describes Fourier only. Add one sentence that the solver changes to column-block preconditioning after a capped Fourier solve, and put the policy/details in an appendix. “Exact Jacobian” should be qualified: the structural linearisation is analytic, but pressure derivatives of opacity are evaluated with ±1e-4 probes in the inspected code.

At μ=0.5 the outcome is qualification-undecided. Mark that point visually and keep its provisional status in the abstract and conclusion, not only a table footnote. Change “converges in 8–33 steps” to distinguish the qualified subset from the unresolved 34-step case. The successful runs also override the advertised default 25-step budget with 100; state that clearly.

Figure 5/history calls `pred` the decrease of the **next** step, but callbacks attach the direction prediction at the preceding linearisation to the just-accepted step. It can also be damped and inexact. Label it accordingly; distinguish it from the final qualification interval.

Do not call the test intrinsically “stricter” than another paper's height criterion: the quantities differ and need not imply one another. State what happens when this particular mean-height criterion is applied to this trajectory. I could not retrieve the 2021 full text through the available web endpoint; verify the exact original aggregation, absolute/signed change and comparison stage before attributing the 6 km criterion. The local history script removes each step's mean height; disclose that gauge choice.

### A5. Revise the headline scientific interpretation to match all metrics

The table is honest; the surrounding text needs to be equally balanced:

- Own-τ μ=1: RMS height error worsens 20.7→22.6 km and correlation falls 0.743→0.687.
- μ=0.707: RMS is similar but correlation falls 0.607→0.550.
- μ=0.5: RMS worsens 23.2→28.4 km and correlation falls 0.409→0.259, despite an almost exact strong-minus-quiet mean contrast.

These results support recovery of a selected magnetic-region statistic, not uniformly improved surface morphology. Explain the tension and retain it in the abstract. Replace “same rms” with “comparable rms” where warranted. Do not explain the residual as “set by granulation, where the weak field cannot change the balance” without a mask-resolved error decomposition; pressure and geometry are horizontally coupled even in weak-field pixels.

Define the depression as a strong-minus-quiet **vertical height contrast of a τ_LOS surface**, rather than automatically equating every inclined value to a conventional sunspot Wilson depression. The masks change with viewing angle; the three points are different sampling/selection problems. Give strong/quiet counts at each μ, identify whether masks come from fixed input truth, and add a small mask/threshold sensitivity or spatial-block uncertainty check. The strong mask occupies only about 1.1% of disk-centre columns, so 93/91/102% recovery deserves context rather than apparent percent-level precision.

“At any viewing angle” exceeds tests at three μ values in one azimuth, including one unresolved result. Separate supported code geometry from demonstrated scientific performance.

### A6. A low-β cut is an empirical result here, not a universal physical requirement

Conclusion: “force balance must be restricted to layers with β>1” is too strong. A low-β, sufficiently force-free field can satisfy MHS; high β does not guarantee hydrostatic validity. The reference at the chosen −4 cut even contains a low-β minority.

State that, for this prescribed field/snapshot, extending the fitted region upward produces poor or invalid reconstructions. Separate inability of a non-static fixed field to satisfy MHS, boundary assumptions, EOS/coordinate coupling and optimisation failure. The cut −5 case is invalid; −6/no-cut cases are stalled. Their errors are failure outcomes, not comparisons of equally converged physical models. Lower force residual than the dynamic reference is expected from fitting a static model and does not prove a more realistic atmosphere.

Use the cut test as a useful limitation/sensitivity result, not proof that −4 or β=1 is a universal boundary for inversion applications.

### A7. Be precise about boundary information and gauges

“No external reference pressure is needed” should be followed immediately by the top closure and fixed cut-pressure assumption. Pressure is not independently recovered from data without boundary information. Initial total-pressure alignment neglects magnetic tension; say so. Periodic boundaries also matter for finite observed fields of view.

The inspected centred periodic stencil has additional alternating cut-height null modes on even grids. The manuscript fixes only the uniform gauge. Briefly document numerical gauge handling/initialisation and any effect on relative heights in an appendix; do not claim every relative height is uniquely inferred from Φ. Similarly, opacity need not increase monotonically with pressure in every tabulated regime, and pressure changes affect the accumulated levels below: avoid unconditional “raises opacity/lifts every level” prose without specifying the regime.

## B. Literature, evidence and fair comparisons

The introduction should compare physical assumptions fairly, rather than frame every implementation difference as a first. Compress seven novelty bullets into three contributions: coupled pressure/geometry on inversion coordinates; consistent ray geometry plus a column-mass counterpart; quantified scientific gain and limits at practical field-of-view scale. MPI, a library interface and the stopping implementation support those contributions.

Add the subsequent FIRTEZ work: [Borrero et al. 2024, solenoidal inversion](https://arxiv.org/html/2404.02045v1) directly addresses the divergence limitation the manuscript discusses; [Borrero & Pastor Yabar 2022, current uncertainties](https://arxiv.org/abs/2211.07593) is relevant to noise-sensitive derivatives. Check the original Puschmann treatment before summarising it as offsets alone. Verify whether “ideal-gas EOS” accurately describes the thermodynamic treatment of each cited FIRTEZ version rather than suggesting a constant ionisation/molecular weight without evidence.

Remove the runtime comparison with 500–2000 CPU hours for a full inversion+MHS cycle. It is a different workload, hardware and scientific task. Give your own wall time, cores/ranks/threads, grid, solver settings and inclusion of setup/I/O; benchmark speedups only against a matched workload. Remove the unsubstantiated 504²-in-minutes sentence from the introduction or attach a separate validated result; this paper's table concerns 252². The measured four-node scaling can be a brief appendix/data-availability item, not a motivation for more scaling experiments.

The pressure abstract/conclusion numbers need consistent scope: 3–9% is a reduction of a selected RMS **log-error**, not the fractional error of every pressure measurement; the mass result is a larger reduction. Report selected absolute log errors and their interpretation without conflating them with percent accuracy. The claim that an opacity change scales all pressures proportionally is not generally true for pressure-dependent χ and a coupled MHS solve.

## C. Minimal additional validation, without an open-ended project

The current test is prescribed simulation T/B, not a Stokes inversion, and not a demonstration that changing P/z preserves or improves a fit to spectra. Pressure/electron density can influence opacity, damping and line formation. Replace “not constrained by spectra” with “often weakly or indirectly constrained, depending on diagnostics,” and describe the tested mode as reconstruction of prescribed stratifications intended for inversion use.

For a defensible methods submission, add two compact checks:

1. **Independent numerical verification:** summarise an existing known-equilibrium recovery/refinement case and one rotated-ray result, with a small figure/table or appendix. They establish correct numerical recovery without using a dynamic snapshot as an exact MHS oracle. Reuse valid existing results; no new verification campaign is needed.
2. **A bounded input-sensitivity test:** select one representative μ and apply modest, explicitly justified, spatially/depth-correlated perturbations to T/B at two amplitudes, using a few fixed realisations. Report status, magnetic-region contrast, surface RMS/correlation and remaining force error for HSE and MHS. Derivatives amplify noise; white noise alone is not a model of inversion uncertainty. Either add one limited azimuth-ambiguity example or explicitly assume disambiguated input. A full sunspot study, all angles and a complete synthetic inversion are valuable future work, not simultaneous submission requirements if the present claims are narrowed.

If these checks are deferred, remove the empty robustness section and weaken the title/claims accordingly; leaving a planned experiment in the abstract is not acceptable. Avoid claiming readiness for every inversion iteration until spectral feedback is demonstrated or clearly labelled as interface capability.

## D. Presentation for scientists

### Suggested narrative and title

Suggested title: **“Recovering solar atmospheric geometry with magnetohydrostatic constraints on optical-depth and column-mass grids.”** It matches the tested reconstruction better than suggesting a demonstrated Stokes inversion.

Suggested structure:

1. Introduction: why common geometric heights matter; existing approaches; the precise question and three contributions.
2. Physical formulation: prescribed and inferred quantities, boundary/gauge assumptions, τ/m mapping, inclined rays, actual objective.
3. Validation design: equilibrium verification, dynamic-snapshot input construction, HSE baseline, metrics and stopping outcomes.
4. Results: magnetic-region contrast versus overall morphology, thermodynamics, inclination, cut/input sensitivity; one short cost paragraph.
5. Discussion: why fitting static balance differs from recovering a dynamic snapshot; noise, divergence, boundaries and inversion applicability.
6. Conclusions: a few quantitative, conditional findings.

Move Fourier eigendecomposition, implementation layout, detailed derivative operators and solver configuration to an appendix or code documentation. Keep enough mathematics in the main text to understand what physical problem is actually solved. Scientific readers need the coordinate geometry and assumptions more than the transpose/cumulative-sum implementation.

### Abstract rewrite direction

Use one context sentence, one aim, two method sentences, and a few results with their limits. Lead results with improved strong-field contrast, then disclose unchanged/worse overall surface accuracy and the unresolved most-inclined case. Conclude with demonstrated applicability over the tested range and the prescribed/disambiguated-field assumptions. Do not use 102% as an accuracy certificate or repeat every numerical table entry.

### Figures and table (inspected in the PDF)

- **Fig. 1:** use a separate labelled colour bar for |B|. The common height/error scale is useful, but split labels `z` and `reconstruction−reference` and add panel letters. Consider a magnetic-region zoom/inset because the scientific signal occupies a small fraction of the field.
- **Fig. 2:** six curves and the small legend are hard to distinguish at one-column width. Limit the main panel to HSE/MHS for the main τ case and mass pair, or use a wider figure. The top height errors reach ~10^5 km, far beyond the reference box: explain extrapolated/input-unconstrained upper layers, restrict the main plot to the scientifically interpreted domain, and retain the full-range plot in an appendix. Identify the mapping of mass levels to median τ as descriptive, not pointwise equivalence.
- **Fig. 3:** median β and the fraction below unity share one logarithmic axis labelled β. Use a separate fraction axis restricted to [0,1], and mark the tested cuts. Show or tabulate vertical force residual too if claiming vector balance. Distinguish EOS-consistent diagnostic density from snapshot density in the reference force estimate; the inspected diagnostic recomputes ρ from the solver EOS.
- **Fig. 4:** mark μ=0.5 as unresolved/open symbol and avoid connecting three points as though they establish general angular behaviour. Preserve the worsening RMS result prominently.
- **Fig. 5:** the table defines depression as negative, while the plotted right axis is a positive magnitude. Label `−depression` or use the signed quantity consistently. Separate contrast and height-change panels if needed to avoid mixed log/linear interpretation. Correct the pred timing described above.
- **Table 1:** add a compact explicit convergence/status column instead of relying on superscripts alone. Keep the correlation column: it is scientifically revealing. Explain differing thread counts/node hardware in a note and do not compare time as a controlled scaling result. Avoid counting an invalid run as quantitative physical evidence.
- The first page contains the template message requesting a running title/author list. Set those fields. Remove author/abundance/DOI/bibliography TODOs and make all symbols/units consistent, especially ln versus log10 and mass units.

## E. Reproducibility and bounded completion plan

The November 10 report explicitly invalidates the older τ input/results because x/y registration was wrong. The paper's README still describes older reports and says rows are awaiting reruns, although the manuscript now contains the rebuilt results. Update that provenance immediately. Archive the corrected input mapping/checks, exact paper/code commits, configurations and machine-readable Table 1/figure data. This is essential because new qualified results differ from the earlier budget-limited 504² studies; do not mix them.

Give the actual EOS/opacity composition, abundance source, LTE/partition-function assumptions and treatment of simulation non-equilibrium ionisation. The netCDF snapshot density and table density need not match. Keep physical model discrepancy distinct from numerical error. Verify the Wittmann bibliographic entry and all DOI/title TODOs from authoritative records; do not submit placeholder entries.

**Recommended order for Claude:**

1. Correct A1–A4 first: the objective, inclined geometry, mass-pressure statement and actual solver/stopping description. This is a prose/derivation correction, not a request to change scientific source.
2. Rewrite the abstract/introduction/conclusion around the selective gain and limits in A5–A7; remove unfair runtime and unsupported universal claims. Add the relevant later literature.
3. Add the compact verification/sensitivity evidence in C, or explicitly narrow unsupported inversion-use claims. Do not initiate a broad new sunspot/inversion campaign merely to finish this paper.
4. Revise figures/captions/table for the scientific narrative and update the paper README/provenance. Ensure new uncertainty/status statements appear in the abstract as well as the methods.
5. Finish bibliography, data/code availability, authors and acknowledgements, then regenerate the PDF for a final scientific consistency read.

Do not reduce the contribution to an implementation inventory. The paper is strongest when it explains **which atmospheric information static force constraints recover, which they do not, and why the answer depends on coordinates, viewing geometry and fitted height range.**
