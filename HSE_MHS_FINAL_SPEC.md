# 3D HSE / MHS on a geometric-height grid: final v1 specification

*Status: **binding implementation handoff**, 16 Oct 2026. It supersedes the design discussion in `HSE_MHS_DESIGN_REVIEW.md` (Codex), `HSE_MHS_DESIGN_CRITIQUE.md` (Claude), `review-response.md` (Codex) and `REVIEW_RESPONSE_CRITIQUE.md` (Claude). Those four remain the rationale. Where this file is silent, Codex's original design applies. Where they conflict, this file wins. Any change to a decision in §1 must be recorded in §10.*

---

## 0. Objective

Given `T[NX,NY,NZ]`, `x[NX]`, `y[NY]`, `z[NZ]` (top-first) and optionally `Bx, By, Bz[NX,NY,NZ]`, compute `Pgas`, `rho`, `ne` and `logtau500` (plus the extra outputs in §2) on the same grid, covering the photosphere through the chromosphere and into the transition region. There are **two modes** (decision D0):

| Mode | Physics | Label on output |
|---|---|---|
| `column_hse` (nonmagnetic) | Each column independently satisfies vertical HSE, `dP/dz = −ρg`; T fixed | `vertical_hse_only`; horizontal force residual reported |
| `mhs` (3D) | 3D force balance `∇P + ρgẑ − J×B = 0`; **T fixed**; P and B adjusted; B fitted to `B_obs` within σ_B | `mhs_fit` plus the flags in §6.7 |

**D0: final mode selection (user, 16 Oct 2026).** Only `column_hse` and `mhs` are built. The plane-parallel "full 3D HSE with B = 0" mode proposed in `HSE_MHS_DESIGN_REVIEW.md` §3 is **dropped**. With B = 0, exact three-direction balance forces P, ρ and T to depend on z only, so it returns a single 1D atmosphere copied across the plane, and it is infeasible for any structured cube. A horizontally structured atmosphere in true 3D balance needs the Lorentz force, which is what `mhs` provides. `column_hse` is the nonmagnetic product and the MHS initializer; it does not claim horizontal balance.

The first real dataset is Bifrost `en024048_hion`, snapshot 385. The authoritative file is `MHS/IO/BIFROST_en024048_hion_0_504_0_504.nc` (6.1 GB, NetCDF4/HDF5, top-first z). It is periodic in x and y, with dx = 47.6 km. In the window [−0.5, +3.0] Mm it has 177 levels spaced 18.6–20 km. Grid sizes are always runtime values.

---

## 1. Binding decisions

| ID | Decision | Notes |
|---|---|---|
| D1 | Two modes, `column_hse` and `mhs`, as in §0 (D0). Neither is ever silently substituted for the other, and no plane-parallel mode exists. | User decision, 16 Oct 2026 |
| D2 | **`column_hse` default anchor: `:bottom`**, a uniform scalar `P_ref` at the lowest level (`z[end]`). Options: `:top` (default value 0.1 Pa) and `:height(z_ref)`. A τ anchor is deferred. | **Overrides the recorded top requirement** (evidence in `REVIEW_RESPONSE_CRITIQUE.md` §2). There is no built-in default value for the bottom `P_ref`; the user must supply it. For Bifrost, P at −0.5 Mm is 10^5.02 Pa with a 0.08 dex spread. |
| D3 | **`mhs` uses a single scalar gauge, not a pressure map**: the mean of `u = ln(P/1 Pa)` over the anchor plane equals `ln P_ref`. The default anchor plane is the bottom; the user may pick the top with 0.1 Pa. | Removes the artificial `Lx = Ly = 0` top-plane constraint and the over-determined-loop problem. |
| D4 | **T is fixed bitwise in `mhs` and `column_hse`.** Joint (T, B) inference is a documented v2 option and is not built. | Recorded requirement kept. |
| D5 | EOS: Wittmann LTE through a status-returning adapter, then a validated table (§4). One EOS state supplies ρ, nₑ and Pₑ. There is no fixed μ anywhere except in analytic test EOSs. | |
| D6 | τ₅₀₀: total extinction (absorption + scattering) at 5000 Å, vertical, top-first trapezoid. The default top closure is `:exp_tail` with `H_χ`, falling back to `:half_cell` with a per-column mask. Exported `logtau500` is finite. | §5 |
| D7 | Viewing angle: v1 implements `:per_column` (`τ_μ = τ_z/μ` in each column, the RH 1.5D convention; this optical-depth option is unrelated to the dropped HSE mode). 3D rays come in M8. Equilibrium is always solved in the local solar frame. | |
| D8 | Float64 everywhere. SI internally. Input units are read from file metadata, never guessed. | NetCDF: m, T, K |
| D9 | MHS v1 discretization: collocated nodes with second-order coordinate-aware centred differences, periodic x and y, one-sided second-order differences at the z faces, and an explicit grid-scale regularizer (§6.3). Staggered B or a vector potential only if Gate C3 fails. | |
| D10 | MHS v1 solver: least squares with L-BFGS and hand-written adjoint gradients (§6). An augmented-Lagrangian outer loop only if synthetic gates fail. **No PETSc in v1.** | |
| D11 | Parallelism: v1 uses Julia threads on one node, with all kernels written on local blocks with halos. MPI.jl (`THREAD_FUNNELED`) is wired in at M7 and is required for production. | |
| D12 | `fieldline_hse` is an optional diagnostic and initializer (M4b). It is not on the critical path. | |
| D13 | Native vertical sampling by default (177 levels for Bifrost). Resampling (e.g. to 357 levels) is an explicit, recorded preprocessing adapter: ln T linear and B linear. | |

---

## 2. Interface

```julia
column_hse(T, x, y, z; eos, anchor=:bottom, P_ref, g=274.0, n_sub=4,
           opacity, tau_top=:exp_tail, mu=1.0) -> Result
mhs(T, Bx, By, Bz, x, y, z; sigma_B, eos, anchor_plane=:bottom, P_ref,
    bc=(x=:periodic, y=:periodic), lambda=..., init=:column_hse, ...) -> Result
```

**Input validation (reject unless an explicit adapter is used):** `z` must be strictly decreasing; `x` and `y` strictly increasing; shapes must match; T > 0 and finite. A multidimensional z in the file must be identical across columns to within 1e-6 relative before it is reduced to `z[NZ]`. For periodic axes, check for a duplicated endpoint.

**Outputs (all `(NX,NY,NZ)`, top-first):**
- Thermodynamics: `Pgas`, `rho`, `ne`, `Pe`, `nH_total`, `mu_bar`, `colmass` (relative to the top plane, plus `m_top` if τ or mass closure is used), `tau500`, `logtau500`. T is never modified in either mode, so no `T_out` is produced; the input T is kept in `/input`.
- MHS only: `Bx_out`, `By_out`, `Bz_out`, `dB/σ`.
- Diagnostics: `R` (3 components), `R_par`, `R_perp`, `divB`, `beta`, EOS status mask, `tau_top_method` mask.
- Metadata: `vertical_order=:top_first`, units, anchor (type, plane, value), gauge, g, EOS/opacity version and hash, composition vector, τ convention, all λ and σ settings, solver history, git hash, and thread/MPI layout.

**File format:** HDF5, with groups `/input`, `/output`, `/diagnostics` and `/meta`. Every dataset carries `units` and `dims` attributes. The original inputs are kept untouched.

---

## 3. Gate 0: input audit (run first, on every new cube)

Produce a report (HDF5 plus a short markdown summary) containing:

1. Units, frame, dimension order, periodicity flags and z-uniqueness check.
2. Per-layer minimum, median and maximum of T, |B| and β (β uses the column-HSE P from both anchors).
3. **Anchor sensitivity:** column HSE with `:top` (0.1 Pa) and with `:bottom`. Report the pressure distribution at common heights, the implied opposite-boundary pressures, and the τ = 1 height maps.
4. If native P and ρ exist (Bifrost lgp/lgr), compute the native residual `R = ∇P + ρgẑ − J×B`. Report each component per layer, normalized by ρg and by |∇P| + ρg + |L|. This is a **baseline**, not a floor.
5. **Force-budget screening:** per layer, `|∇_h P_colHSE|` against `|L_h(B_obs)|` against the characteristic correction scale `|δL| ~ |B|·σ_B/(μ₀ ℓ)`, using ℓ = 2dx. Report the fraction of cells where the correction scale exceeds the mismatch. This is labelled "screening", not "feasibility".

Reference values measured on Bifrost 385. These were revalidated on 16 Oct 2026 directly from `IO/BIFROST_en024048_hion_0_504_0_504.nc`, the authoritative input, using native T, P, ρ, B and v, frozen ρ/P, and 4×4 horizontal subsampling. The FITS-based numbers are reproduced exactly.

**File facts:**
- `z` is a `[1,496]` float64 variable in m, stored **top-first**: +14.367 Mm down to −2.444 Mm.
- Cubes are `[1, 504, 504, 496]` float32, chunked 1×64×64×16, with shuffle and deflate filters.
- Units: T in K, P in Pa, ρ in kg m⁻³, nₑ in m⁻³, B in T, v in m s⁻¹.
- The window [−0.5, +3.0] Mm contains 177 levels spaced 18.6–20 km.
- T ranges over 2.4×10³ – 1.6×10⁶ K.

| z [Mm] | log₁₀P p1 / p50 / p99 [Pa] | spread [dex] | median β | median ρv²/P | fraction T > 3×10⁴ K |
|---|---|---|---|---|---|
| 3.0 | −3.23 / −2.47 / −1.32 | 1.91 | 0.007 | 0.04 | 0.66 |
| 2.0 | −2.66 / −2.06 / −1.24 | 1.42 | 0.013 | 0.51 | 0.12 |
| 1.5 | −2.05 / −1.26 / −0.42 | 1.63 | 0.074 | 0.71 | 0.003 |
| 1.0 | −0.74 / 0.22 / 0.75 | 1.49 | 1.0 | 0.44 | 0 |
| 0.5 | 2.00 / 2.46 / 2.67 | 0.67 | 95 | 0.10 | 0 |
| 0.0 | 4.04 / 4.26 / 4.33 | 0.29 | 2.6×10⁴ | 0.20 | 0 |
| −0.5 | 4.96 / 5.02 / 5.04 | 0.08 | 2.1×10⁴ | 0.04 | 0 |

**Anchor experiment:**
- Top anchor (0.1 Pa uniform): pressure error at z = 0 is +2.5 dex (median) and +5.7 dex (p99).
- Bottom anchor: error at z = 0 is +0.01 / +0.11 dex (median / p99); at z = 1 Mm it is +0.04 / +1.17 dex.
- Neither anchor reproduces z ≥ 2 Mm, where the bottom-anchored error spans −4.2 to +1.3 dex (p1–p99).

**Native vertical-only residual** (median): 0.09 at 0–0.5 Mm, 0.24 at 1 Mm, 0.42 at 1.5 Mm, 0.62 at 2 Mm, about 4 at 2.9 Mm.

**Consequences:**
- β crosses 1 near z ≈ 1 Mm.
- At 1.5–2 Mm, 30–37 % of cells have ρv² > P, so a static fixed-T solution cannot represent those layers faithfully.
- 66 % of columns are above 3×10⁴ K at the top, so the hot EOS branch (§4.3) is mandatory.

The Julia implementation must reproduce these numbers to within about 0.1 dex using the same proxy.

---

## 4. EOS and opacity (M1)

**4.1 Direct adapter.** Wrap the Wittmann C++ entry points (`witt_thermodynamics_from_pgas`) through a persistent context per task, with no per-call table rebuild. Tighten the inner convergence from 1e-5 to **1e-10** relative and return `(rho, ne, Pe, mu_bar, iterations, converged::Bool, charge_residual)`. A non-converged call is an error state that propagates; it is never a plausible value. The Python implementation is a small-case oracle only.

**4.2 Envelope.** Temperature runs from 0.9·T_min to 1.1·T_max of the cube from Gate 0. Pressure covers the union of both anchor runs ±1 dex.

**4.3 High-T and low-T coverage.** Run the direct solver across the envelope.
- If it stays converged and approaches the fully ionized limit (μ̄ → value from composition, nₑ → full ionization) smoothly, use it everywhere.
- Otherwise, above a switch temperature `T_sw`, use an **H + He Saha branch** with the same abundance vector. `T_sw` is chosen where both agree to within 1e-3 in ln ρ and ln nₑ; reject if no such overlap exists.
- At the cold end, test down to T_min with H₂ active.
- Out-of-envelope calls return `eos_out_of_envelope`.

**4.4 Table.** Tabulate `ln ρ(ln T, ln P)` and `ln nₑ(ln T, ln P)` using bicubic Hermite interpolation, with node derivatives from central differences of the direct solver. Start at Δlog₁₀T = 0.002 and Δlog₁₀P = 0.02, then refine locally until the gates pass. Derivatives `∂ ln ρ/∂ ln P|_T` come from the interpolant.

**4.5 Composition.** Use one explicit abundance vector, recorded in the metadata with its source (the Wittmann default unless overridden). Match Bifrost's composition only for comparison runs.

**4.6 Opacity.** Port or wrap `contOpacity` at 5000 Å. It returns extinction **including** scattering; never add scattering twice. Convert from cgs to m⁻¹ once. `HOTOP` is a placeholder, so mark T ranges where hot-opacity terms matter as `opacity_incomplete`.

**Gate 1 (pass criteria):**
- Python, C++ and Julia direct solvers agree to within 1e-8 relative at identical constants.
- On withheld off-node points across the envelope: |Δ ln ρ| ≤ 1e-5 and |Δ ln nₑ| ≤ 1e-4 (table against direct).
- `∂ ln ρ/∂ ln P` agrees with a finite-difference step sweep of the direct solver to within 1e-3 absolute.
- No unflagged non-converged calls.
- Opacity is within 5 % of an RH background-opacity calculation at 10 photospheric (T, P) points with matched EOS; deviations are reported elsewhere.

---

## 5. `column_hse` and τ₅₀₀ (M2: the first usable product)

**5.1 ODE.** `du/dz = −g ρ(T(z), P_ref·e^u)/(P_ref·e^u)` with `u = ln(P/1 Pa)`.
- Anchor `:bottom`: integrate from `k = NZ` toward `k = 1` (upward).
- Anchor `:top`: integrate from `k = 1` toward `k = NZ`.
- Anchor `:height`: integrate both ways from `z_ref`, after locating it by interpolation.

**5.2 Scheme.** Classical RK4 with `n_sub` (default 4) equal substeps per grid interval. T(z) inside an interval is **ln-T linear** in z; no cubic splines. The EOS is called at every stage. The error estimate is the difference between `n_sub` and `2·n_sub` runs. Flag columns where it exceeds 1e-6 in u (expected only at the TR); those cells get an automatic substep doubling, up to 64.

**5.3 Outputs.** P, ρ, nₑ and the other state variables are evaluated at the nodes from the final u. Column mass is the trapezoid of ρ.

**5.4 τ₅₀₀.** `τ[:, :, 1] = τ_top`, and `τ[k+1] = τ[k] + ½(χ_k + χ_{k+1})(z_k − z_{k+1})`.
- `:exp_tail`: `H_χ = (z_1 − z_2)/ln(χ_1/χ_2)`. If `0 < H_χ < 5000 km`, then `τ_top = χ_1·H_χ`.
- Otherwise use `:half_cell`, `τ_top = ½ χ_1 (z_1 − z_2)` (RH convention), and set the column's mask bit.
- `:zero` is selectable, and gives log τ = −Inf at the top.
- Report `tau_top_method` per column.
- μ ≠ 1: `τ_μ = τ/μ`, labelled `per_column`.

**Gate 2:**
- Isothermal ideal-gas test EOS: P matches `P_ref·exp(Δz/H)` to 1e-12 relative.
- Linear-T ideal gas: matches the analytic `∫dz/T = Δz·ln(T₂/T₁)/(T₂−T₁)` to 1e-8, and RK4 shows fourth order under substep halving.
- Top-first index tests: P and τ increase with k, and `Pgas[:, :, anchor_k] == P_ref` exactly.
- Mass identity `|P(z) − P_top − g·m(z)| / P(z) ≤ 1e-6`.
- A τ-convergence test on smooth profiles shows second order.
- A run on Bifrost 385 with the Wittmann table and both anchors produces the Gate 0 anchor table. This is a report, not a pass threshold.
- A matched-EOS LTE comparison with RH on 3 columns shows P agreeing to within 1e-3, given the same `m_top` and abundances.
- Results are bitwise identical for 1 and N threads.

---

## 6. `mhs` (M5, with M6 only if needed)

**6.1 Unknowns.** `u` and `B = (Bx, By, Bz)` at all nodes. T is fixed. The gauge (D3) is implemented by the parameterization `u = v − ⟨v⟩_anchor + ln P_ref`, so the optimizer works on an unconstrained `v`.

**6.2 Operators.**
- First derivative along any axis: derivative of the quadratic Lagrange interpolant through three neighbouring nodes, using **signed coordinates**. This works for decreasing z and nonuniform spacing.
- Periodic x and y wrap with period `L_x = NX·dx` (no duplicated endpoint).
- z faces use the one-sided quadratic through 3 nodes.
- `J = ∇×B/μ₀`, `L = J×B`.
- `∇P` is computed by differencing P itself, **not** as `P∇u`.
- `ρ = ρ_EOS(T, P)` pointwise.
- Residual: `R = ∇P + ρgẑ − L`.
- Split: `R_par = b̂·R` and `R_perp = R − R_par b̂`, where `b̂ = B/max(|B|, 1 G)`.

**6.3 Objective.**

```
Φ = λ_F/N Σ_n w_n |R_n|²/S_n²                         force (all nodes; z-face nodes included, reported separately)
  + λ_D/N Σ_n w_n (h_n ∇·B_n / Bs_n)²                 divergence
  + 1/N_B Σ_{c,n} ((B_c,n − Bobs_c,n)/σ_c,n)²          data
  + λ_S/N Σ_n w_n Σ_axes |h_a² ∂_aa(B − Bobs)|²/σ²     grid-scale regularizer on δB
  + λ_U/N Σ_n w_n Σ_{a∈x,y} (h_a² ∂_aa u)²             grid-scale regularizer on u (odd–even control)
```

| Symbol | Definition |
|---|---|
| `w_n` | Cell volume / mean cell volume |
| `h_n` | Local mean spacing |
| `S_n` | `max(ρ₀,n·g, 0.01·median_layer(ρ₀g))`, from the initializer and **fixed for the whole solve**: the residual is measured in units of the local weight |
| `Bs_n` | `max(\|Bobs_n\|, 1 G)` |
| `σ_c,n` (synthetic default) | `sqrt((0.05·\|Bobs_n\|)² + (5 G)²)` per component (quadrature). Real data: supplied by the user. |
| Defaults | `λ_D = 1`, `λ_S = 1e-3`, `λ_U = 1e-6` |

The regularizer terms must contribute less than 1 % of Φ at convergence, or the run is flagged.

**6.4 Choosing λ_F (discrepancy principle).** Run warm-started continuation with λ_F ∈ {1e-4, 1e-3, …, 1e4}. Select the **largest** λ_F for which `χ²_B/N_B ≤ 1 + 3√(2/N_B)`, then bisect in log λ_F to a factor of 1.5. This makes the field as close to equilibrium as the declared noise allows.

**6.5 Optimizer.**
- L-BFGS with m = 10 and a strong-Wolfe (Hager–Zhang) line search; Optim.jl or an in-house implementation is fine.
- The gradient is a hand-written adjoint of the stencil kernels, including the EOS term through `∂ρ/∂P|_T` from the table.
- Verify it with dot-product and finite-difference tests (≤ 1e-6 relative on 16³ boxes). Enzyme or ForwardDiff may be used to check, not in production.
- Stop when `‖∇Φ‖ ≤ 1e-6‖∇Φ₀‖`, or when the relative change in Φ is below 1e-10 over 20 iterations, or at `max_iter` (default 5000). The reason is always recorded.

**6.6 Initialization.** `u₀` from `column_hse` (same anchor as the gauge), `B₀ = Bobs`. `fieldline_hse` is optional (M4b).

**6.7 Result classification (all flags always present).**
- `solver_status`: `converged | stalled | max_iter | eos_failure | invalid_input`.
- `equilibrium_tol_met`: true only if, per layer, the RMS of `|R|/S` is at most `tol_rms` and its p99 is at most `tol_p99`. Synthetic defaults: 1e-6 and 1e-4. Real-data defaults: 1e-2 and 1e-1. Both are user-overridable and recorded.
- `data_admissible`: true only if the fraction of samples with `|δB|/σ > 3` is at most 1 % **and** `max |δB|/σ ≤ 5`.
- `success_mhs = equilibrium_tol_met ∧ data_admissible ∧ (solver_status == converged)`. Anything else is exported as `approximate_only = true` with full diagnostics.

**6.8 Augmented Lagrangian (M6, conditional).** Use the same kernels, with multipliers on R and ∇·B and penalty growth by a factor of 10 per outer iteration up to 1e8. Build it only if Gate C fails with 6.3.

**Gate C (synthetic, boxes ≤ 128³):**
1. With horizontally uniform T, zero B and uniform B must reproduce `column_hse`. In that case column HSE is exact 3D balance.
2. A known curl-free, solenoidal, nonuniform B gives L = 0 to truncation order.
3. **Checkerboard test:** inject ±ε odd–even modes into u and B and verify that the objective penalizes them (the regularizer works) and that they decay. If this fails, switch to staggered B or a vector potential (D9).
4. Codex's vertical benchmark: `B = (b(z), 0, 0)`.
5. Codex's 2.5D horizontal benchmark: `B = (0, b(x,z), 0)` with `P = Q(z) − b²/2μ₀`.
6. **Neukirch & Wiegelmann (2019)** analytic 3D MHS: a positive-pressure member, with T generated through the test EOS. Recover it from B perturbed by the declared σ, with fixed T. Show second-order convergence of the residual under refinement.
7. Failure cases (incompatible flux, too-small σ, a large Poisson-only-consistent vector residual) must yield `approximate_only` with correct diagnostics.

Targets: the 6.7 synthetic tolerances on resolved grids, and convergence toward the continuum reference under refinement.

**Real-data run (M7):** Bifrost 385 cropped to [−0.5, 3] Mm. Report the full per-layer residual and data tables. No scientific claim of equilibrium is made unless 6.7 passes.

---

## 7. (Removed) plane-parallel HSE

Dropped by decision D0. Codex's original §3.2–3.3 (temperature-bound feasibility and the log-T projection) is not implemented. Temperature is never modified in v1.

---

## 8. Parallelism and performance

- **M2–M6:** `Threads.@threads` over column blocks (HSE) and over z-slabs of local blocks (MHS kernels). Each task gets one EOS context; there is no shared mutable state. BLAS and OpenMP threads are set to 1.
- **Block abstraction:** each kernel acts on `(local array, halo width = 1)`, so that MPI later only adds halo exchange and reductions.
- **Memory preflight** before allocation: `bytes = 8·N·(n_fields)`. For example, `N = 504·504·177 ≈ 45 M` gives about 0.36 GB per field; MHS with L-BFGS needs about 100 field-equivalents, roughly 36 GB. On Olivia, one CPU node (256 cores, 768 GiB) is the M7 target.
- **M7, MPI.jl:** a 2D horizontal process grid (full columns local), `THREAD_FUNNELED`, communication on the main thread only, collective error propagation, and decomposition-invariance tests (identical classification; values within reduction-order tolerance). A GH200 port is a benchmark experiment after M7.

---

## 9. Milestones and ownership

| M | Deliverable | Gate |
|---|---|---|
| M0 | Reader (NetCDF/FITS), units, crop, flip, Gate 0 report | §3 reproduced |
| M1 | EOS adapter, envelope, table, opacity | Gate 1 |
| M2 | `column_hse` + τ₅₀₀ (threaded): **first product** | Gate 2 |
| M3 | Force-budget screening maps (Gate 0 item 5) and the column-HSE horizontal residual report | Screening reproduces hand cases |
| M4a | MHS operators, residual, adjoint gradient | Gate C1–C3 and gradient tests |
| M4b (optional) | `fieldline_hse` (arclength ODE, null mask, loop-compatibility report) | Vertical field matches `column_hse` within tracing/ODE tolerance |
| M5 | MHS L-BFGS + discrepancy-principle λ_F | Gate C4–C7 |
| M6 (conditional) | Augmented Lagrangian, staggered B or vector potential | Gate C re-run |
| M7 | Full Bifrost run, MPI, HDF5 restart | §6 real-data report; decomposition invariance |
| M8 | 3D inclined rays, prescribed-NLTE hook | Plane-parallel limit tests |

**Workflow.**
- One model implements each milestone. The other writes that milestone's oracle tests directly from this spec, before seeing the code.
- Every pull request quotes the gate rows it passes, with numbers.
- No runs larger than 128³ before M5 passes.
- Package layout as in Codex's original §14, with `ColumnHSE.jl` and `Screening.jl` added and PETSc removed.

---

## 10. Decision log

| Date | Decision | By |
|---|---|---|
| 15 Oct 2026 | Top-first storage; arbitrary runtime sizes; LTE Wittmann; MPI + threads; fixed-T MHS with adjustable B | User (recorded in design v1) |
| 16 Oct 2026 | D2 bottom anchor default for `column_hse`; D3 scalar gauge for `mhs` | Claude (delegated by user), with Bifrost evidence; **user may veto** |
| 16 Oct 2026 | D6 finite log τ default; D10 no PETSc in v1 | Claude (delegated) |
| 16 Oct 2026 | **D0/D1: two final modes, `column_hse` (each column independently in HSE) and `mhs` (3D). The plane-parallel full-HSE mode is dropped.** | **User** |
| 16 Oct 2026 | **E1**: the §5.4 `:exp_tail` scale height is `H_χ = (z_1 − z_2)/ln(χ_2/χ_1)` (the earlier `ln(χ_1/χ_2)` had the wrong sign). Accept iff `χ_1 > 0`, `χ_2 > χ_1` and `0 < H_χ < H_max`. | **User** (from `HSE_MHS_IMPLEMENTATION.md` §1) |
| 16 Oct 2026 | **E2**: §5.3 column mass is integrated together with `u` in the same RK4 stages (`dm/dz = −ρ`), so that Gate 2's mass identity can hold; the trapezoid value is kept as the diagnostic `colmass_trap`. | **User** |
| 16 Oct 2026 | **E3**: the collocated discretization (D9) has a nonzero per-column least-squares residual floor (RMS `|R|/S` ≈ 1e-4 at Δz/H = 1/8). Synthetic Gate C passes if per-layer RMS ≤ `max(1e-6, 1.5·R_floor)`, p99 ≤ `max(1e-4, 1.5·R_floor_p99)` and both converge at order ≥ 1.8 under 2× refinement, where `R_floor` is the residual of the exact solution sampled on the grid. Production `equilibrium_tol_met` (§6.7) is unchanged. | **User** |
| 16 Oct 2026 | **EOS correction `FIX_GASC_H2` (default on).** The supplied Wittmann code's `gasc()` omits the factor pe in the H₂/H₂⁺ constants that `pe_pg()` uses (derivation and Bifrost impact: `HSEMHS/reports/gasc_h2_check.md`). The shim aligns `gasc` with `pe_pg`. Effect on Bifrost 385: nₑ unchanged, ρ ≤ 0.26 %, χ₅₀₀ corrected by up to 1.36 dex below 3500 K. The original behaviour can still be built with `HSEMHS_FIX_GASC_H2=0`. | **User** |
| 16 Oct 2026 | **Opacity: keep the Wittmann/ATLAS 500 nm opacity; Gate 1 RH row passed against RH with two RH artifacts removed.** The artifacts: `Hydrogen_bf` skips hydrogen continua whose model-atom wavelength grid starts above λ (n ≥ 5 at 500 nm), and `chemequil.c` stores n(H⁻) with all atomic H instead of the neutral H. After removing them, all 10 points agree within 4.3 %; the remainder is the H₂⁺ data difference (ATLAS fit vs Bates 1952), recorded as a known difference. Evidence: `HSEMHS/reports/rh_opacity_investigation.md`. Gate 2 RH row: 6.8e-4 with matched EOS (`HSEMHS/reports/rh_crosscheck_notes.md`). | **User** |
| 16 Oct 2026 | **D9 invoked: `mhs` uses the staggered vector-potential scheme of `HSE_MHS_D9_STAGGERED_ADDENDUM.md`.** P at cell centres; B = curl A + B̄_z ẑ on faces, so ∇·B = 0 exactly; J on edges; compact force balance on interior faces. For this scheme λ_D is dropped and **E3 is withdrawn**: synthetic tolerances return to 1e-6 / 1e-4. The collocated scheme stays only as `mhs.scheme = "collocated"` for comparison. Trigger: Gate C3(b) failure (`HSEMHS/reports/M4a_c3_remedies.md`). Open: whether the input B is cell-centred or native face-staggered (addendum §12.3). | **User** |
| 16 Oct 2026 | **D9b (`HSE_MHS_D9b_AVERAGING_ADDENDUM.md`) approved for the staggered `mhs` scheme.** (1) The data term compares B on faces with Bobs interpolated to faces (`data_on_faces` default on; B_out stays HB). (2) A force-weighted current-Nyquist term Φ_Q (λ_Q = 1 default) acts on edge currents along the M averaging directions. (3) A vertical grid-scale term λ_Uz in Φ_U, plus an identifiability diagnostic δP_unobs = \|B\|σ_B/μ₀; the C3(b) u criterion is measured on nodes with δP_unobs ≤ 0.1 P. Also in effect: option (a), with horizontal force equations on layers 2…NZ−1 only. Trigger: C3(b) B mode, a joint near-null of H and M (`HSEMHS/reports/M6c_gate.md`, `M6c_gn.md`). Φ_Q is subject to the M6d step-1 experiment. | **User** |
| 17 Oct 2026 | **M6d (D9b) results accepted, with three deviations from the addendum** (`HSEMHS/reports/M6d_step1.md`, `M6d_step2.md`). (1) **λ_Uz defaults to 0**: 1e-4 biases C1a beyond its gate, and with face data and Φ_Q the top pressure mode it targets does not occur. The term stays available. (2) **Solver policy**: plain L-BFGS by default; B-first continuation automatically when Φ_Q/Φ_F at the start exceeds `mhs.bfirst_ratio` (default 10). L-BFGS also gets a pressure-collapse safeguard on u = ln P (`mhs.collapse_du = 3`). (3) **C3(b) B rows**: B-first solves end at the round-off floor (`stalled`, line search fails), so the gate checks decay ≥ 10³ and Φ back at the reference floor instead of `status == converged`. Φ_Q covers the boundary layers (horizontal differences on all edge levels; only the z difference is interior-only), and A₀ excludes Φ_Q. | **User** |
| 17 Oct 2026 | **M5a decisions** (`HSEMHS/reports/M5a_realdata.md`). (1) Φ_Q is multiplied by λ_F. (2) The C1b gate uses the same round-off-floor criterion as C3(b). (3) The real-data convergence test: P and B change by ≤ 1e-3 (relative) over 100 iterations (`mhs.xtol_rel`). Comparison with the native Bifrost state is not an acceptance criterion. (4) B-first when the start is Lorentz-dominated: implemented as median \|L\|/S > 1 in some layer, in place of the requested median β < 1, which misfires on uniform fields (**confirmation pending**). (5) The real-data λ_Q is still open: C3(b) passes down to λ_Q = 1e-4, while Φ_Q/Φ on Bifrost is 6–12 % at λ_Q = 1e-3. | **User** |
| 18 Oct 2026 | **`stalled_at_floor` termination label** (`alternating_solve!`; Codex review M5 second follow-up, Q1). A solve may end `stalled` with reason "stalled_at_floor" only if all four hold. (1) The all-node P/B change test and the feasibility checks hold. (2) Fresh residual, data and pressure audits at the same final state are computed and reported. (3) Independent floor evidence holds, supplied by the caller: for synthetic gates, the validated reference floor and the gate's accuracy tests. Production supplies none, so it cannot obtain the label. (4) The fixed-metric diagnostic correction predicts a decrease ≤ `res_rel` × Φ. It is **not** `converged`; `success_mhs` still requires `solver_status == converged`. | **User** |
| — | Bottom `P_ref` value (suggested 1.0×10⁵ Pa at −0.5 Mm for Bifrost-like cubes) | **User to confirm** |
| — | Real-data σ_B and tolerances; downstream consumer (RH 1.5D / FIRTEZ / other); whether to resample to 357 levels | **User to confirm** before M7 |
