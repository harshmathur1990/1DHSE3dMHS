# Critique of `HSE_MHS_DESIGN_REVIEW.md` (Codex design for 3D HSE / MHS in Julia)

*Reviewer: Claude · 15 Oct 2026 · Scope: physics, numerics, solver choice, validation, parallel plan, and handoff quality. I checked claims against the RH and FIRTEZ sources in `WorkRepo/RH` and `WorkRepo/firtez-dz-mhs-master`. The Wittmann EOS folder (`WorkRepo/MHS/wittman-eos`) was not connected, so I take Codex's §7.3 EOS findings as given.*

---

## 0. Verdict in one paragraph

The Codex document is **mathematically careful and mostly correct**. Its best contribution is the proof that full nonmagnetic 3D HSE forces `P`, `ρ` and `T` to be horizontally uniform (§3.1). Its insistence on judging success by raw vector-force residuals, not by a scalar Poisson residual, is also right. But the document is **risk-averse to the point of being hard to implement**, and it misses the three issues that I think most likely explain why earlier attempts did not converge:

1. **Anchoring pressure at the top (0.1 Pa at +3000 km) puts all of the inconsistency into the high-β photosphere, where no realistic field can absorb it.** Plausible columns that share the same top pressure reach z = 0 with pressures differing by up to **five orders of magnitude** (§2.1).
2. **With fixed T, MHS has a field-aligned structure.** Along every field line, pressure obeys exactly the 1D HSE ODE. Given T and the field-line geometry, P is therefore fixed up to one constant per field line. This makes the problem much easier to understand, initialize and diagnose than a black-box constrained optimizer (§2.4).
3. **The proposed solver stack is the riskiest available option**: full-space SQP or augmented Lagrangian, PETSc.jl, and MPI from day one. There is published, working prior art for exactly this class of problem (Wiegelmann/Neukirch/Zhu-type MHS optimization) that Codex does not cite (§2.5).

My recommendation: keep Codex's mathematical framing and validation philosophy, but **reorder the build**. First an EOS table, then column HSE with a configurable anchor, then cheap feasibility maps (β and force budget), then field-line HSE, then a least-squares MHS optimizer on one node with threads or GPU. MPI comes last.

---

## 1. What Codex got right (keep these)

| Item | Why it's right |
|---|---|
| §3.1: full B = 0 3D HSE ⇒ `P(z)`, `ρ(z)`, `T*(z)` | Exact. The horizontal momentum equations leave no freedom. This alone explains why "3D HSE with small ΔT" cannot converge for any structured cube. |
| §3.2: closed-form feasibility certificate `ε_min,k = (Tmax−Tmin)/(Tmax+Tmin)` | Cheap, exact, and gives a precise failure message instead of a hung solver. |
| §4.1: fixed T **and** fixed B is overdetermined (1 scalar unknown, 3 equations) | Correct. FIRTEZ's Poisson solve only imposes the divergence of the force balance. |
| §6: Poisson for ln P is necessary, not sufficient | Correct, and an important reason not to clone FIRTEZ's loop. |
| §2.3: ρ and nₑ from **one** EOS state; no fixed μ | Correct. FIRTEZ's Poisson THERMO_TERM uses `MW/T` (an ideal-gas form) while the EOS supplies MW; mixing the two creates a lagged nonlinearity. |
| §7.3: Wittmann `contOpacity` already includes scattering; λ is in Å; cgs units | Important catch; double-counting scattering is a classic bug. |
| §7.3: the EOS inner loop can silently stop at 250 iterations | Important. Silent EOS failure is a plausible convergence killer. |
| §10.1: signed differences with top-first storage | Correct and necessary. |
| §11: separate "numerically converged" from "physically in equilibrium"; structured statuses | Good engineering. Keep. |
| §12 Gate B tests (isothermal analytic, false-HSE test, top-first indexing) and Gate C 2.5D benchmarks | Good, cheap, decisive tests. Keep them all. |
| §15 "why earlier attempts failed" table | Reasonable hypotheses. I add the missing ones in §2 and §4. |

---

## 2. Critical issues (ranked by expected impact)

### 2.1 Top-anchored pressure is probably the biggest single cause of non-convergence

The design treats `Pgas(z_top) = 0.1 Pa` (uniform) as fixed and integrates **downward**. Codex correctly notes that this forces `Lx = Ly = 0` on the top plane (§5.1). It does not notice the much larger consequence.

For an ideal gas, `ln P(z) = ln P_top + ∫_z^{z_top} dz′ / H(z′)` with `H = kT/(μ m_u g)`. Scale heights span roughly two orders of magnitude inside the box:

| T [K] | μ (approx.) | H [km] |
|---|---|---|
| 4 000 | 1.27 | 96 |
| 6 500 | 1.25 | 158 |
| 10 000 | 0.75 | 405 |
| 10⁵ | 0.6 | 5 060 |

Three illustrative columns, all starting at 0.1 Pa at +3000 km (crude μ(T), my own quick calculation, illustrative only):

| Column | P(z = 0) | P(z = −500 km) |
|---|---|---|
| TR at 1.8 Mm, 6500 K chromosphere | 1.4 × 10⁴ Pa | 1.3 × 10⁵ Pa |
| TR at 2.5 Mm, 6500 K chromosphere | 4.1 × 10⁵ Pa | 3.8 × 10⁶ Pa |
| TR at 2.5 Mm, cool 4500 K chromosphere | 1.9 × 10⁹ Pa | 1.8 × 10¹⁰ Pa |

The real photosphere at z = 0 is around 10⁴ Pa and roughly horizontally uniform at fixed geometric height (to within about a factor of 2, not 10⁵). **A uniform top pressure therefore produces absurd photospheric densities, wildly scattered τ₅₀₀ = 1 heights, and enormous horizontal pressure gradients in the layer with the highest β.** No magnetic field within any realistic uncertainty can balance them, so any MHS iteration seeded from this state fails, and it fails in the photosphere, not in the chromosphere.

The physics points the other way. Anchor pressure where it is best constrained and where β is high (the bottom, or a photospheric reference height, or τ₅₀₀ = 1). Let it float where β is low (the top), because there a tiny δB can balance any leftover horizontal gradient. FIRTEZ already does this. In the local source, arrays are **bottom-first** (index 1 is the bottom). `RK4_INTEGRATION_LOG_BOTTOM` and `RK4_INTEGRATION_LOG_LTAUM05` (with `IZ=1`) anchor at the bottom. `MHS_SOLVER` with `NBDCND==2` imposes Dirichlet `PZ0` at the bottom and an HSE Neumann gradient at the top.

**Recommendation (decision for the user):** make the anchor a first-class option, `anchor = :top | :bottom | :height(z_ref) | :tau(τ_ref)`, with a reference pressure scalar or 2D map. Default to **bottom/photospheric anchoring** for MHS. Keep `:top` with 0.1 Pa for reproducibility with earlier work. Report the implied pressure at the opposite boundary as a diagnostic either way.

### 2.2 "Full 3D HSE" as specified is a 1D problem; column HSE should be the primary nonmagnetic product

Codex proves that full nonmagnetic HSE is plane-parallel, then demotes per-column HSE to a "diagnostic utility" that must "never be substituted for requested full HSE". That is logically clean but practically backwards:

- With chromospheric temperature contrasts (often 4000 to over 10⁴ K at fixed height, more across the TR), `ε_min` will be tens of percent at most heights. Full HSE will almost always return `hse_infeasible_temperature_bounds`, and when it does succeed it returns **one 1D atmosphere copied across x, y**. That is useless for 3D synthesis or inversion.
- What the community usually means by "HSE for a 3D cube" (RH 1.5D practice, FIRTEZ initialization, inversion codes) is **column-by-column HSE**. It is the RH-equivalent deliverable, it is embarrassingly parallel, and it can be finished and validated in days.

**Recommendation:** define three explicit nonmagnetic modes and keep Codex's honest labelling:

| Mode | What it solves | Status label |
|---|---|---|
| `column_hse` | Vertical balance in each column; horizontal residual reported | `vertical_hse_only` (not "3D equilibrium") |
| `plane_parallel_hse` | Codex's full HSE (feasibility test + projected T*(z)) | `full_hse` / `infeasible` |
| `mhs` | Below | `mhs_*` |

The user should confirm whether "3D HSE" meant column HSE. Codex's §1.1 records "full force balance in all three directions" as confirmed. If that is truly the requirement, the user should know that it is a 1D product.

### 2.3 Fixed-T MHS is likely infeasible in the high-β layers: add a cheap feasibility map before any expensive solve

The design defers feasibility to the end of an SQP run ("`mhs_no_feasible_solution_found`"). That run may cost hours on 90 M points. A simple order-of-magnitude bound tells you most of the answer up front.

- The maximum horizontal force a field can supply is roughly `B²/(μ₀ ℓ)`, where ℓ is a horizontal length scale. The horizontal pressure contrast that fixed T implies (from the anchored column-HSE pressures) is `ΔP_h(z)`.
- Feasibility needs roughly `ΔP_h ≲ B²/μ₀` (equivalently β_eff ≲ 1 for the *contrast*, not for P itself), **and** the required δB must be inside the uncertainty budget.
- In the quiet photosphere, `B ~ 100 G` gives `B²/2μ₀ ≈ 40 Pa` against `P ~ 10⁴ Pa`. B can then absorb only about 0.4 % horizontal pressure contrasts. Granular T contrasts imply far more. **Fixed-T MHS will not exist there.** This is not a solver failure; it is physics (granulation is dynamically supported).
- In the low-β chromosphere the opposite holds. Gas forces are tiny compared with what a 1–5 % change in B can do, so feasibility is easy as far as the cross-field balance is concerned.

**Recommendation:** before any MHS solve, compute and save per-height maps of the following, and use them to decide where MHS is even meaningful:

- `β`;
- the force budget `|∇_h P_colHSE|` against `|L_h(B_obs)|` against `|δL|_max(σ_B)`;
- the fraction of each layer where fixed-T balance is achievable within σ_B.

Then the user must choose one of these scientific options (this is a real decision, not a numerical one):

- **(a)** allow T to vary with an uncertainty (joint `(T, B)` inference; this is implicitly what FIRTEZ does through optical-depth remapping);
- **(b)** restrict MHS to the region where the map says it's feasible and state that the photosphere there is column-HSE only;
- **(c)** keep fixed T everywhere and report the residual as "missing dynamic support".

### 2.4 Missing insight: field-aligned HSE structure (a much better initializer and diagnostic)

Because `b̂ · (J × B) = 0`, the component of force balance along the field contains no magnetic term:

```
b̂·∇P = −ρ g b_z   ⇔   along a field line (b_z ≠ 0):  dP/dz|_line = −ρ(T, P) g
```

This is **exactly the column-HSE ODE**, but with T sampled along the field line instead of along the vertical. Consequences:

1. **Given T and the field-line geometry, P is fixed up to one constant per field line.** Lines that cross the anchor surface get their constant from the anchor. Low-lying closed loops need another rule, such as matching the column-HSE pressure at one footpoint.
2. **Codex's §4.1 compatibility condition `B·∇P = −ρ g B_z` is precisely this**, but the document treats it as a residual to check rather than as a structure to exploit.
3. **This gives a cheap, physically correct low-β initializer, "field-line HSE"**: trace lines through the observed B (or a lightly smoothed version), integrate the 1D EOS-coupled ODE along each, and splat P onto the grid. Column HSE is the special case of a vertical field. In canopies, where the field is strongly inclined, field-line HSE differs from column HSE and is the physically right starting point.
4. It separates the unknowns cleanly: **the magnetic freedom only has to fix the cross-field balance** `J×B = ∇_⊥P + ρ g ẑ_⊥`. In low β this needs only a small δB. That tells the optimizer which way to move.
5. It explains a failure mode. If T is fixed and B is nearly rigid (low β), *the parallel balance cannot be repaired by B at all*. Only the field-line geometry within σ_B can move. Any scheme that tries to repair parallel residuals through the pressure Poisson equation will stagnate.

**Recommendation:** add `fieldline_hse` as a mode between column HSE and full MHS. Use it as the MHS initial guess where β < 1, blended to column HSE where β > 1. Report the parallel residual `b̂·R` and the perpendicular residual `R_⊥` separately in all diagnostics. They have different physical meaning and different fixes.

### 2.5 Solver choice: prefer proven least-squares MHS optimization over full-space SQP plus PETSc.jl

Codex recommends infeasible-start SQP or augmented Lagrangian with trust region, a matrix-free KKT system, Schur-complement and multigrid preconditioners, and PETSc/TAO through PETSc.jl, all distributed. In my judgment this is **research-grade numerical optimization on top of an unvalidated physics problem**: around 4 × 90 M unknowns, a similar number of equality constraints, a nonlinear EOS, and mixed elliptic/hyperbolic structure. Each layer is a plausible failure point, and when it fails you won't know which layer it was. PETSc.jl has also historically lagged PETSc and exposes TAO only partly. Codex itself concedes that availability "must be established by prototype".

There is direct prior art Codex does not cite:

- **Wiegelmann & Neukirch (2006, A&A 457, 1053)**, "Computing nonlinear magnetohydrostatic equilibria via optimization".
- **Zhu & Wiegelmann (2018, ApJ; arXiv:1809.02168)**, MHS extrapolation. It minimizes `L(B,p,ρ) = ∫ w_f |(∇×B)×B/μ₀ − ∇p − ρgẑ|²/B² + w_d |(∇·B) B|²/B² dV`. It handles positivity with `p = Q²`, `ρ = R²/(gH)` and reports good recovery of a reference model. Follow-ups include Zhu, Wiegelmann & Solanki (2020, SUNRISE/IMaX AR 11768) and a multi-layer fast approach (arXiv:2109.12789).
- **Neukirch & Wiegelmann (2019, Sol. Phys., DOI 10.1007/s11207-019-1561-0)**: analytical 3D MHS solutions with an adjustable transition from non-force-free to force-free. These are exactly the "fully 3D independently verified equilibrium benchmark" that Codex's Gate C.6 asks for but does not name.

Our problem differs (T is fixed, so ρ is tied to P through the EOS, and B has a data term instead of boundary-only data), but the structure carries over directly:

```
min over (u = ln P, B):
   ½ Σ w_F |R(u,B;T)|² / S_F²                (force balance, fixed reference scales)
 + ½ Σ w_D (ℓ ∇·B)² / S_B²                   (solenoidality, or remove it via B = ∇×A)
 + ½ (HB − B_obs)ᵀ C_B⁻¹ (HB − B_obs)        (data term)
```

- **Optimizer:** L-BFGS (or nonlinear CG) with the gradient from hand-written or AD (Enzyme.jl) adjoints of local stencil kernels. Memory is about 2 × m × n_state; with m = 10 that is roughly 55 GB at the example size, which fits one node.
- **Exactness upgrade path:** wrap the same objective in an **augmented-Lagrangian** outer loop (multipliers on R and ∇·B). That gives Codex's "exact constraint" semantics without changing the inner machinery. This is a strict superset of the simple approach, so nothing is wasted.
- **Weighting:** the 1/B² normalization in Wiegelmann-type functionals misbehaves in weak-field regions. Use Codex's fixed `S_F = |∇P| + ρg + |L|` from the reference state, as Codex recommends; that part of Codex's design is good.
- **Solenoidality:** consider representing B through a vector potential A (or by staggered faces). ∇·B = 0 then holds by construction, which removes one penalty term and one source of ill-conditioning. The data term then acts on `H(∇×A)`.

Codex's warning that "a penalty minimum is not an exact MHS solution" is true but secondary. With noisy B and a dynamic input, the meaningful target is **a residual small compared with the physical force scales, plus δB consistent with σ_B**, not machine-zero. Exactness matters for synthetic verification, which the augmented-Lagrangian loop can deliver.

### 2.6 The input is probably a dynamic snapshot; measure its own force residual first ("Gate 0")

`504 × 504` matches, as far as I recall, the horizontal grid of the public Bifrost enhanced-network run `en024048_hion` (24 Mm box, 48 km spacing, non-equilibrium hydrogen ionization). **Please confirm the source of the cube.** If it is Bifrost (or MURaM or MURaM-ChE):

- The chromosphere is shock-dominated; `ρv²` and `ρ Dv/Dt` are comparable to P. A static equilibrium at fixed T **does not exist** in large parts of the domain, whatever the solver.
- Bifrost's own `P` and `ρ` come from a non-LTE, non-equilibrium hydrogen EOS. Recomputing ρ and nₑ from Wittmann LTE at the same `(T, P)` will give different values, by orders of magnitude for nₑ in the upper chromosphere.
- B in Bifrost is natively **face-staggered**. If you can export the native staggered B, ∇·B is zero to roundoff for free. Ask whether the native files are available rather than a cell-centred interpolation.
- Bifrost is **horizontally periodic**. Periodic x, y boundaries are then the natural choice, which removes the hardest open question in Codex's §5.2 and allows FFT preconditioning horizontally.
- The Bifrost vertical grid is **nonuniform**, and FISHPACK `HW3CRT` (used by FIRTEZ) assumes uniform spacing. It takes only min/max and N−1 intervals. Feeding a nonuniform z into a FIRTEZ-style Poisson solve gives wrong derivatives silently. **That is a concrete candidate for earlier failures.**

**Add Gate 0:** compute the raw force residual `R = ∇P − ρg − J×B` of the *input snapshot itself*, using its native P and ρ, layer by layer. That residual is the floor any static model can be expected to reach. Demanding a normalized RMS of 10⁻⁶ (Codex §11.2) on such data is meaningless. Use tiered targets: machine or truncation level for analytic tests, and "fraction of the snapshot's own residual" for real data.

### 2.7 EOS envelope: the box contains transition-region and coronal temperatures

With z up to +3000 km, many columns in a typical simulation are at 10⁵–10⁶ K at the top. Codex mentions auditing "partition-function temperature coverage" but does not draw the conclusion:

- Wittmann/Kurucz partition functions and the dummy `HOTOP()` are not meant for 10⁵–10⁶ K. You need an **explicit high-T branch**: a fully ionized H/He closure, `ρ = P μ m_u /kT` with μ ≈ 0.6 and nₑ from full ionization, blended smoothly across about 2–5 × 10⁴ K, with a validity mask.
- At the cold end, chromospheric "cool pockets" reach about 2000 K in Bifrost. H₂ chemistry matters and must be covered and tested (FIRTEZ floors T at 2500 K in the HSE spline and clamps T to 2000–14000 K in `APPLY_THRESHOLDS`).
- **Tabulate the EOS early, not "optionally after direct calls pass".** Tabulate `ln ρ(ln T, ln P)` and `ln nₑ(ln T, ln P)` with bicubic Hermite or monotone interpolation, built from the validated direct solver. Codex treats this as a late optimization, but it is an enabler:
  - it is smooth, so you get exact and consistent `∂ln ρ/∂ln P`. Codex itself flags EOS noise (prec = 1e-5) as limiting derivatives;
  - it is thread-safe and allocation-free, and works on GPU;
  - it is differentiable by Julia AD;
  - it is about 100–1000× faster than an iterative Saha solve per call.

  Keep the direct solver as the oracle, and gate the table with a max-error map over the actual `(T, P)` envelope.

### 2.8 τ₅₀₀ top boundary: `τ_top = 0 → log τ = −Inf` will break downstream tools

Codex recommends a truncated-box `τ_top = 0` that preserves `−Inf` at the top node. That is mathematically honest, but every consumer of `logtau500` (FIRTEZ, inversion codes, plots, interpolation onto τ grids) will choke on it. Standard practice (RH `convertScales`, FIRTEZ `GET_TAU` with `TAU_MIN`) is a finite closure.

**Recommendation:** default `τ_top = χ₅₀₀(z_top) · H_ρ(z_top)`, which assumes exponential extinction above the box with the local density scale height. Label it `tau_top_closure = :scale_height` and allow `:zero` or a user value. In practice it barely matters: when z_top is in the TR or corona, χ is tiny and the top τ is negligible for anything at log τ > −8.

---

## 3. Moderate issues and suggestions

### 3.1 Vertical integration details
- Tsit5 plus tstops is fine but more than needed. On about 10 km spacing with `H ≥ 95 km`, an **exact-per-interval integrator** is simpler and robust: treat T as linear (or ln T as linear) in each interval, solve for `P_{k+1}` by Newton on the trapezoid or Simpson form with EOS calls, and estimate error by interval halving. It is second- or fourth-order, has no adaptive-step logic, and parallelizes trivially per column.
- **Interpolant choice matters at the TR**, where T can jump 10⁴ → 10⁵⁺ K in one cell. Interpolate in `ln T` or use a monotone (PCHIP) interpolant. **Never use cubic splines in T.** FIRTEZ uses `SPLINE_CUBIC_VAL` in single precision with a 2500 K floor, which overshoots at steep gradients; the floor is a symptom of that. Note that `∫dz/T` with T linear is analytic: `Δz · ln(T₂/T₁)/(T₂−T₁)`, which is a good ideal-gas cross-check.

### 3.2 Discretization
- Codex's mimetic staggered complex (§10.2) is the gold standard. It is also a large implementation burden on a nonuniform 3D grid with six boundary types. Pragmatic path: **(i)** start with collocated fourth-order central differences plus a ∇·B penalty, as the Wiegelmann-type codes do successfully; **(ii)** run Codex's checkerboard/null-mode tests; **(iii)** move to staggered B or a vector potential only if those tests fail. If native staggered input B exists (Bifrost), use a face-centred B from day one.
- Use Float64 everywhere (FIRTEZ MHS arrays are `REAL(4)`; Codex flags this correctly).

### 3.3 Parallelization: single node first, MPI last
- Column HSE: 254 k independent columns × 357 points. Julia `Threads.@threads` over columns on **one 256-core Olivia CPU node** should finish in minutes with a tabulated EOS. MPI adds nothing here except I/O complexity.
- MHS memory: Codex's own numbers (about 27–68 GiB for 40–100 cube-equivalents) fit comfortably in **one 768 GiB Olivia CPU node**. An L-BFGS optimizer with about 30 cube-equivalents of state is roughly 20–30 GiB.
- **GPU option:** Olivia also has GH200 nodes (4 GH200 per node, 96 GB HBM3 each). The force-residual objective and its gradient are pure local stencils plus a table lookup, which is the ideal GPU workload (KernelAbstractions.jl / CUDA.jl). One GH200 node could hold the whole problem.
- Recommendation: architect kernels to be decomposition-agnostic (operate on a local block with halos), but **ship v1 as threaded shared-memory (and optionally GPU)**. Add MPI halos only after the physics validates. This removes Codex's §13 (thread-level MPI safety, PETSc thread rules, collective failure propagation) from the critical path.

### 3.4 Inclined viewing (μ ≠ 1)
Codex's genuine 3D ray integration is correct, but add the two conventions people actually use:
- **Plane-parallel per column**, `τ_μ = τ_z/μ`. This is what RH 1.5D does; you need it for consistency with 1.5D synthesis.
- **Shear or rotate the cube** into the observer frame (as is done for slanted views in 3D synthesis), so that slanted rays become columns, and then reuse the vertical τ integrator.

Keep the equilibrium solve in the local frame regardless, as Codex says.

### 3.5 Outputs for the RH / FIRTEZ toolchain
Add `nH_total` (and LTE hydrogen populations or `n_p` if cheap), `Pe`, mean molecular weight, column mass `m(z)`, and `τ₅₀₀` (linear) to the outputs. RH atmosphere files need nH and nₑ. Having column mass also lets you validate against RH's formulation, where HSE is **algebraic** (`P = g m`, plus `B²/2μ₀` when `Stokes` is on; see `hydrostat.c`, where the RHS is `C2·cmass/T − C3·B²/T`). Note that RH's `Hydrostatic` is a per-depth Newton solve for `(n_H, n_e, n_p, n_H2, n_Htot)` given column mass and *fixed equilibrium constants from the current NLTE state*. As Codex says, it is not a height-grid integrator. In LTE it reduces to "EOS at `P = g m`".

### 3.6 FIRTEZ facts Codex missed or understated
- **Storage is bottom-first** (`K=1` is the bottom: `RK4_INTEGRATION_LOG_BOTTOM` loops `K=2..NZ` with `DZ>0`; `PZ0`/`PZ0MAX` sit at index 1). This is the opposite of the user's top-first convention, so it is a high-risk spot for sign and index bugs when porting.
- **The real reason FIRTEZ is "photospheric":** `LIMIT_MAGTERM` replaces the magnetic source above `LOCMT` (the top of the magnetic sensitivity region) with an exponentially decaying copy, `MAG_TERM(K) = MAG_TERM(LOCMT)·exp(LOCMT−K)`, per grid cell. Above the sensitivity of photospheric lines, FIRTEZ is effectively column HSE. Together with the `T ∈ [2000, 14000] K` clamp and `|B_i| ≤ 10⁴ G`, this is why it cannot be pushed into the chromosphere unmodified. Codex lists these items but doesn't connect them to the user's "logτ > −3" observation.
- **FIRTEZ's EOS takes `BETAE`**, an electron-density departure coefficient (NLTE nₑ). That is a ready-made hook for the "prescribed NLTE" future option Codex mentions. Codex is right that scaling nₑ alone is not a consistent closure; FIRTEZ's `GET_RHO` takes it as an input to the chemical-equilibrium solve, which is worth inspecting before designing the interface.
- The FIRTEZ "convergence" metric `CHI2N` is the mean |ΔZ_W| of the Wilson depression in km, with the loop also exiting at `MAXITER`. Codex is right that it is not a force criterion.

### 3.7 The document as a handoff to an implementing model
- It is about 760 lines, and much of it is caveats ("not accepted", "pending review", "do not invent"). An implementing model needs **defaults it may use, plus a list of things it must not decide**. As written, a model will either stall or invent defaults anyway.
- It lacks a concrete **discrete** specification for v1 (exact stencil, EOS table axes, ODE scheme, file format), and lacks a minimal test list with numeric pass thresholds for each milestone.
- Suggested fix: split it into (A) a 2–3 page **"v1 spec"** with concrete defaults and milestone tests, and (B) Codex's current text as the **rationale appendix**.

---

## 4. Additional failure hypotheses (to add to Codex §15)

| Hypothesis | Evidence to look for | Fix |
|---|---|---|
| Top-anchored P puts the mismatch in the high-β photosphere | Column-HSE P(z=0) spans ≫ 1 dex horizontally; MHS residual largest at the bottom | Bottom or photospheric anchoring (§2.1) |
| Fixed-T MHS requested where `ΔP_h ≫ B²/μ₀` | β/force-budget map (§2.3) | Allow T uncertainty, or restrict the MHS region |
| Parallel residual `b̂·R` treated as fixable by the Poisson update | `b̂·R` doesn't decrease while `R_⊥` does | Field-line HSE (§2.4) |
| Nonuniform z passed to a uniform-grid Poisson solver (HW3CRT) | Residual pattern correlates with dz variation | Metric-aware operators |
| Cubic-spline T overshoot at the TR, then T floors | Negative or floored T samples near steep gradients | ln T-linear or PCHIP |
| EOS outside validity (T > 3×10⁴ K, or below the H₂ range) | Inner EOS status, `HOTOP=0` region | High-T branch, table validity mask |
| Snapshot not static (shocks, flows) | Gate 0 residual of native snapshot is large | Lower the target; report the dynamic floor |
| Index-order flip when porting FIRTEZ (bottom-first) to top-first | Sign errors in `∂z`; P decreasing downward | Explicit order-tagged arrays; Gate B.6 |

---

## 5. Proposed revised build order

| Stage | Deliverable | Gate (pass criteria) |
|---|---|---|
| **0** | Input audit: units, order, grid, β map; **native-snapshot force residual** (if P and ρ are available) | Report produced; dynamic-floor residual known per layer |
| **1** | EOS: direct Wittmann adapter (status-returning) + high-T branch + `(ln T, ln P)` table with error map | Table vs direct: max \|Δln ρ\|, \|Δln nₑ\| ≤ 1e-4 over envelope; FD derivative agreement; Python/C++/Julia agreement |
| **2** | `column_hse` (anchor = top/bottom/height/τ) + τ₅₀₀ (finite top closure) + nₑ, ρ, nH outputs; threaded | Isothermal analytic to 1e-10; refinement order ≥ 2; top-first index tests; RH LTE column comparison at matched abundances; **this is already a usable product** |
| **3** | Diagnostics: horizontal force budget, `plane_parallel_hse` feasibility (Codex §3.2), MHS feasibility map (§2.3) | Maps reproduce hand-computed cases |
| **4** | `fieldline_hse` (trace B, integrate along lines, splat; β-blended with column HSE) | Vertical B reproduces column HSE exactly; uniform inclined B gives analytic shift; `b̂·R` small |
| **5** | MHS least-squares optimizer (L-BFGS, adjoint gradient, fixed scales), collocated stencils, small boxes (≤ 128³) | Codex Gate C.1–C.5 2.5D benchmarks; **Neukirch–Wiegelmann (2019) analytic 3D MHS** recovered from perturbed B; gradient verified by FD |
| **6** | Augmented-Lagrangian outer loop (exactness), optional staggered B or vector potential | Synthetic: normalized RMS force ≤ 1e-6, max ≤ 1e-4 under refinement |
| **7** | Full-size runs on one CPU or GH200 node; then MPI only if needed; HDF5 I/O, restart | Decomposition-invariance tests (Codex Gate E) |
| **8** | Inclined τ (per-column τ/μ, shear-cube, true 3D ray); prescribed-NLTE nₑ hook | Plane-parallel limit tests |

---

## 6. My suggested answers to Codex's open questions (Q1–Q12)

These are suggestions for the user to accept or override. Each one gives the implementing model a default so it doesn't stall.

| ID | Suggested default |
|---|---|
| Q1 T bounds | Absolute-plus-relative, `|ΔT| ≤ max(ΔT_abs, ε_rel·T)`, with ε_rel = 5 % in the photosphere and 15–20 % in the chromosphere (or from inversion uncertainties if available). Mainly relevant if option (a) in §2.3 is chosen. |
| Q2 profile selection | Accept Codex's weighted log-T projection (it's correct). The product is 1D; treat it as a diagnostic. |
| Q3 σ_B | Synthetic phase: declared σ_B = 5 % of \|B\| + 5 G floor per component, diagonal. Real data: from the inversion, or for simulation input pick a "permitted modification" level explicitly. |
| Q4 hard or soft | Soft (Gaussian χ²) in the optimizer; report the fraction of samples beyond 3σ. No hard bounds in v1. |
| Q5 magnetic BCs | If the input is a periodic simulation: periodic x, y. Bottom: B_z fixed to within σ, tangential components free within σ. Top: free within σ. Verify net flux. |
| Q6 frame/units | Local solar frame, km and G on input, converted to SI internally; heights absolute and common to all columns. |
| Q7 ranges | Measure from the actual cube in Gate 0 (T min/max per layer, dz pattern, B distribution). |
| Q8 τ_top | Finite scale-height closure (§2.8), total extinction (absorption + scattering) at 5000 Å. |
| Q9 composition | Asplund et al. (2009) or whatever the Wittmann code ships with, stated explicitly. If the input is Bifrost, match its abundances when comparing. |
| Q10 gravity/stresses | g = 274 m s⁻² constant; no turbulent or ram pressure in v1. Expect flows if the input is a simulation snapshot, which Gate 0 will quantify. |
| Q11 Olivia | One CPU node (256 cores, 768 GiB) for v1; benchmark a GH200 node for MHS. |
| Q12 old failures | Please supply logs. The hypotheses in §4 give a checklist. |

---

## 7. Questions for the user (only the ones that change what gets built)

1. **Where does the cube come from?** Bifrost (`en024048_hion`?), MURaM, or an inversion? This sets periodicity, staggering, nonuniform z, and whether a static solution can be expected at all.
2. **Pressure anchor:** is `Pgas_top = 0.1 Pa` a hard requirement, or can the default be photospheric/bottom anchoring with the top pressure as an output (§2.1)?
3. **By "3D HSE", did you mean column-by-column HSE on the 3D cube** (RH 1.5D style), or truly horizontal force balance (which is a 1D atmosphere)?
4. **For MHS in high-β layers where fixed T is infeasible:** may T change within an uncertainty (option a), should MHS be restricted to the feasible region (b), or should the residual simply be reported (c)?
5. **What will the outputs feed?** RH 1.5D synthesis, FIRTEZ inversion initialization, or something else? This sets which outputs and τ conventions matter most.

---

## 8. Suggested Codex ⇄ Claude workflow

- **Spec ownership:** fold this critique into a v1 spec, a short doc with concrete defaults, kept separate from the rationale. One model edits; the other reviews diffs only.
- **Independent oracles:** for each gate, have the model that did *not* write the kernel write the reference test (analytic solution, finite-difference gradient check, Python Wittmann oracle). This avoids a model grading its own work.
- **Small-first rule:** no run larger than 128³ until Stage 6 passes. Every PR must include the Gate-table row it satisfies, with numbers.

---

### References checked for this review
- Zhu & Wiegelmann (2018), *On the extrapolation of magneto-hydro-static equilibria on the Sun*, ApJ, [arXiv:1809.02168](https://ar5iv.arxiv.org/html/1809.02168)
- Zhu, Wiegelmann & Solanki (2020), MHS modelling of AR 11768 from SUNRISE/IMaX, [arXiv:2010.06174](https://arxiv.org/abs/2010.06174)
- *Toward a fast and consistent approach to model solar magnetic fields in multiple layers*, [arXiv:2109.12789](https://arxiv.org/pdf/2109.12789)
- Neukirch & Wiegelmann (2019), *Analytical 3D MHS equilibrium solutions … transition from non-force-free to force-free*, Sol. Phys., [arXiv:1911.11084](https://export.arxiv.org/abs/1911.11084)
- Wiegelmann & Neukirch (2006), *Computing nonlinear magnetohydrostatic equilibria via optimization*, A&A 457, 1053 (cited from memory; please verify)
- Olivia architecture (252 CPU nodes × 256 cores / 768 GiB; 76 GH200 nodes × 4 superchips): [Sigma2 docs](https://documentation.sigma2.no/hpc_machines/olivia/overview.html)
- Local source inspected: `RH/rhf1d/hydrostat.c`; `firtez-dz-mhs-master/src/{hydrostatic_tools.f90, mhs_tools.f90: MHS_SOLVER, LIMIT_MAGTERM, APPLY_THRESHOLDS, GET_DIVLORENTZPRE}`
