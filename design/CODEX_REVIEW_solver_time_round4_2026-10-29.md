# Solver-time review, round 4 — 2026-10-29

Reviewed HEAD `5d4f353`, the round-4 brief, changes since `d40f9de`, PCG and feasibility changes, truth diagnostics/start construction, and the draft `make_mhs_control.jl`. No tests, scientific runs or cluster access performed. Numerical results are Claude's reported measurements. No scientific source edited. Recommendations remain for the user's decision.

## Main recommendation

Finish a valid known-equilibrium control before rebuilding the Bifrost benchmark or changing the objective. Start with the field-free case, then the y-invariant magnetic case. Separate generator accuracy, solver discretisation, fixed-boundary mismatch and optimisation error. A full 3-D exact equilibrium is not required for this stage.

The new consistency measurements are useful, but they do not yet justify the statements that 23.6 km is an accuracy floor, that the small-μ error is largely explained, or that changing the objective is the next remedy.

## Findings

### 1. High: the measured reconstruction discrepancy is not a lower bound on attainable error

`truth_consistency.jl` measures the solver map at **fixed truth pressures and truth cut heights**. The optimiser is allowed to change pressure and cut height. It can therefore compensate part of that discrepancy, potentially at the expense of force balance. Saying the truth's heights are “not representable” is justified only with those pressures/boundaries held fixed, not for the full state space.

The 23.6 km discrepancy and the 22.6 km solved error have similar magnitudes but are different error fields. Compare their gauge-aligned spatial maps, correlations, covariance and residual difference before attributing one to the other. Neither their RMS equality nor a worse raw-truth start establishes a floor. The 55 km map discrepancy at μ0.3 likewise cannot simply be added to or subtracted from solver error.

**Action:** replace “floor,” “cause,” and “largely explained” with conditional statements until controlled ablations separate the contributions. The density offset, nonlinear interpolation, τ integration/quadrature, fixed cut pressure, upper geometry and non-static snapshot forces remain separate candidates.

### 2. High: neither truth-start variant is an independent test of convergence to the raw truth

Both constructions retain the column-HSE cut pressure and upper relative structure. The shifted variant perturbs all deeper pressures; `truth_raw=1` leaves the first solved face joining column-HSE pressure to raw truth pressure. Both are feasible solver initialisations when within the EOS domain, but neither is the original equilibrium with its own boundary conditions. “Unshifted is the meaningful one” is too strong: its cut-face defect can dominate the vertical residual.

**Action:** report vertical residual contributions separately for the first 1–3 faces and the remaining domain. Add an experimental exact-truth cut/upper-geometry control, clearly separated from production boundary conditions. A smooth taper of the cut-pressure adjustment can be another initialisation ablation, not a new physical prescription. Compare all with identical weights and the same final problem when testing sensitivity to starts.

μ1 endpoints agreeing after 20 steps are evidence of repeatability for those starts, not proof of uniqueness or good conditioning. At μ0.3 a nonstationary trajectory passing a low height-error point does not identify the desired stopping time.

### 3. Medium: `feas_scale` skips an outward step if the state is already at/beyond its safety boundary

The code tests `d>0 && v<hi` and `d<0 && v>lo`. At v≥hi with d>0, or v≤lo with d<0, the limiting cell is ignored. A start can be valid for the ±EPS_PROBE evaluator yet outside the stricter two-probe-width safety range. Returning Inf in that case defeats the intended fraction-to-boundary safeguard.

**Action:** define behaviour explicitly for starts outside the safety interval. An outward step has zero feasible fraction relative to that interval; an inward step needs an entry/restoration policy rather than silent omission. Test equality, the strip between one and two probe widths, inward/outward/zero directions, nonfinite entries, and limiting cells on another MPI rank. Distinguish the artificial derivative margin from the physical EOS domain.

### 4. Medium: PCG breakdown handling still lacks a reliable outcome and full update protection

Initial residual checking fixes the earlier exact-start defect. However, positive finite rz and pAp do not guarantee finite α=rz/pAp or finite updated arrays. After an overflow, the code may return a corrupted x rather than the documented “last good iterate.” Callers also receive no explicit breakdown status, only a recursive residual.

**Action:** guard scalar step/update finiteness, return a distinct diagnostic outcome through a compatible interface, and check the true residual on abnormal exits. Add a deliberately ill-scaled/breakdown test. This is a numerical edge case, not evidence that reported runs encountered it.

## Answers and next experiments

### Synthetic input: retain two complementary validation tracks

Keep the existing Bifrost inputs as a realistic model-mismatch benchmark with quantified defects. Do not redefine their truth silently. Separately build the controlled equilibrium benchmark to measure discretisation and solver error.

- Option (a), rebuilding truth z with the solver map, is a useful **model-projected reference** at stated P/c, but not independent truth. Rebuilding z also changes where supplied T/B live geometrically; the projected fields need not remain in MHS balance.
- Option (b), “compute χ after interpolation and reintegrate τ on the τ axis,” is not a complete construction: τ is already the independent coordinate there. Define a consistent geometric state, compute EOS opacity there, integrate optical depth, and map to τ, or explicitly construct a discrete-map test. Re-evaluate defects after mapping; interpolation and quadrature errors remain until refined.
- Option (c), retaining realistic inconsistencies, is appropriate for application robustness but insufficient by itself to verify the solver's mathematical accuracy.

Define time-to-convergence first on the controlled problem at fixed physical accuracy and stated boundary conditions. Then report time/budget and achieved residuals on Bifrost separately. Neither track alone establishes universal μ-independent performance or performance on FFNOML inputs.

### Known-equilibrium control: 2-D is enough initially; distinguish exact equations from numerical construction

The displayed constant-D Grad–Shafranov formulas are algebraically consistent: `B=(-A_z,0,A_x)`, `p=P0−mA²/2`, `∇²A=μ0 m A`, and `ρ=−∂p/∂z|A/g` give the stated f equation and density. RK4 integration and interpolated T0/P0 make the resulting reference numerical, not “integrated exactly.” Verify its independent construction error rather than asserting only solver discretisation remains.

A y-invariant magnetic test exercises inclined geometry and pressure/height coupling. It does not cover general 3-D structure or arbitrary horizontal coupling. Complete μ1 and μ0.3 first; later use a rotated version `A(kx x+ky y,z)` or another independent case for mixed derivatives/azimuth. The present draft unnecessarily restricts azimuth to 0/180: a y-invariant atmosphere can be observed with nonzero ty; only tx changes which x structures the ray samples. Removing that restriction requires validating the intended geometry, not inventing y dependence.

**EOS failure:** first identify whether the remaining failure is pressure out of range, nonpositive target density, or an unbracketed temperature inversion. `invT` returns NaN for all of them. Report P, target ρ, endpoint EOS densities and attainable T. Do not clamp T, substitute a tiny χ, or widen the table beyond validated physics to make the control pass.

Prefer adjusting the physical domain/arcade parameters while retaining a nonzero magnetic signal. A top truncation is acceptable only if all rays still cover the requested τ range with a demonstrably negligible or explicitly integrated optical-depth tail; extrapolating the missing top is not equivalent. A longer period changes the physics as well as resolution, so hold the chosen physical case fixed during refinement.

If tapering D upward, update every derived equation. For D=D(z), the density becomes

`ρ = ρ0(1−D cos²kx) + (P0 D'/g) cos²kx − (2 D P0 f'/(f g)) cos²kx`.

The f equation uses D(z). Omitting the D' term destroys equilibrium. Check positive P/ρ, finite f, EOS inversion and independent Cartesian force/divergence errors across the complete domain before mapping. The current sampled finite-difference check at every twentieth z level can miss local failures.

### Refinement: the proposed factor-of-four test needs additional controls

First refine the fine generator step (250 m → smaller) and its reference integration/interpolation until those errors are below the coarse solver error. Refine horizontal spacing and Δlogτ separately before a joint sequence. Keep the same physical domain, field, boundary data, gauge and diagnostic locations. Ensure logτ=0 is exactly sampled or use consistent interpolation.

Use two boundary tracks: analytic/exact boundary data to test interior consistency, and production column-HSE boundary data to measure boundary-model bias. Production boundaries need not converge to this non-hydrostatic magnetic equilibrium as resolution increases. Also refine or consistently treat the fixed upper geometry because the cut-level Lorentz derivative uses levels above the cut.

Report raw truth residual defects, reconstructed-height defect, P/height solution errors, actual qualification result and optimisation error. Tighten optimisation until its error is below the spatial error. A fixed 80-step count is not a convergence guarantee. Fourfold reduction is an expected asymptotic result only for second-order terms; boundary derivatives, interpolation and table/model errors can reduce the observed order. It is enough to diagnose consistency, but not alone to attribute Bifrost drift or establish full-cube timings.

### If the control drifts: do not immediately blame Φ

Even an exact continuous equilibrium has nonzero discrete residual. Changing the solution to reduce that residual can initially increase pointwise truth error. The decisive checks are whether the defect and recovered error vanish under refinement with compatible boundaries, whether perturbations return to the same stationary state, and whether optimisation is actually resolved.

If these fail, isolate boundary, discretisation, domain and solver defects before proposing a physical metric. A derived Cartesian-force/volume metric is a reasonable proposal only after that audit; it changes the nonzero-residual compromise and needs scientific justification. Bound-aware steps address feasibility, and multilevel methods address cost: neither substitutes for that metric decision.

### Stopping: stationarity cannot detect the best unknown truth accuracy

Without truth, a prior, or a justified discrepancy/noise model, stationarity cannot tell that step 5 happens to have the smallest Bifrost height error. Preserve the default 25-step budget and report the actual qualified/budget-limited/domain-limited outcome. Do not select early stopping from the known 13.6 km point and present it as a general method. A discrepancy principle would require a defensible residual-error model, not an arbitrary threshold fitted to this snapshot.

## Priority handoff to Claude

1. Repair the feasibility edge case; add separate inversion/domain diagnostics to the control generator and finish the field-free control first.
2. Produce one valid magnetic control, preserving its equilibrium equations and full ray coverage. Save generator parameters, construction defects and EOS margins.
3. Run the boundary-separated, generator/solver refinement checks above on modest grids before new full-cube optimisation work.
4. Correct the attribution and representability claims, update design §6, and separate raw-truth, projected-truth and boundary-matched errors in tables. In particular, ereg adds step curvature; it does not “flatten the linear model” in the ordinary curvature sense. Threshold-dependent spectral counts remain properties of the specified operator and metric, just not invariant counts of undetermined physical modes.
5. Choose the next objective/preconditioner experiment from those results. Do not launch another long reference sweep or introduce a new scientific default merely to meet the time target.
