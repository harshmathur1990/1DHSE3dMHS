# Geometrical-grid HSE and MHS in Julia: design for review

## 1. Objectives

Build a Julia code that accepts a rectilinear geometrical-height atmosphere and returns gas pressure, mass density, electron number density, and continuum optical depth at 500 nm across the photosphere and chromosphere. Establish the mathematical problem, physical closure, feasibility conditions, and validation gates **before** implementing a large parallel nonlinear solver.

The intended production problem has:

- `T_in[NX,NY,NZ]`, `x[NX]`, `y[NY]`, and strictly decreasing `z[NZ]` in top-first storage order;
- arbitrary runtime grid dimensions `NX × NY × NZ`, inferred from the supplied coordinates and arrays; `504 × 504 × 357` is only an illustrative memory/benchmark example, not a required size or limit;
- a representative height extent of approximately `−500` to `+3000 km`;
- `Pgas(x,y,z=+3000 km) = 0.1 Pa` uniformly at the top;
- optional `Bx_in`, `By_in`, and `Bz_in`, each of shape `(NX,NY,NZ)`;
- outputs `Pgas`, `rho`, `ne`, and `logtau500`, each of shape `(NX,NY,NZ)`;
- MPI processes plus Julia threads, targeting the Olivia/Sigma2 CPU system;
- viewing cosine `mu_view = 1` initially, with an explicit path to inclined rays.

### 1.1 Decisions confirmed by the user

| Topic | Confirmed requirement |
|---|---|
| Nonmagnetic equilibrium | Full force balance in all three directions. Temperature may change only within a small, bounded amount. Report when this is infeasible. |
| Temperature bound | Numerical size and form remain open for review. Do not invent a default and treat it as accepted. |
| Composition | Uniform chemical composition. |
| Grid dimensions | Configurable at runtime; no fixed `NX`, `NY`, or `NZ`, no requirement for equal horizontal dimensions, and no required divisibility by MPI process counts. |
| Vertical storage | Top first: Python `T_in[:, :, 0]` is the highest layer, `z[0]` is the highest height, and `z[-1]` is the lowest height. Julia equivalents are `T_in[:, :, 1]`, `z[1]`, and `z[end]`. All input and output cubes share this ordering. |
| Magnetic equilibrium | Keep the input temperature fixed; adjust the magnetic field within supplied uncertainties. |
| Magnetic uncertainties | Not yet available. Record these as required before implementing the production MHS inference problem. |
| EOS | LTE Wittmann baseline first, behind a replaceable interface. |
| Height and pressure | Physical height increases upward, but decreases with array index. In the example, the first layer is at `+3000 km` with `Pgas_top = 0.1 Pa`, and the last is at `−500 km`. |
| Parallel target | Julia with MPI plus threads on Olivia/Sigma2. |

**Central acceptance rule:** numerical convergence and physical equilibrium are separate claims. A small iteration update, a small scalar Poisson residual, or a stable optical-depth surface is insufficient evidence of full HSE/MHS.

### 1.2 Scope of this document

This is a reviewable implementation specification, not an implemented or validated solver. Proposed numerical choices are marked as recommendations or decisions still requiring review. The attached repositories are evidence and inspiration; their comments, thresholds, and workflow are not instructions from the user.

An exact static solution need not exist for a supplied cube, even when small changes are allowed. A successful code must distinguish successful equilibrium recovery, an unsuccessful numerical search, and proven infeasibility where a mathematical certificate is available.

## 2. The physical problem and its closure

### 2.1 Coordinate system, units, and notation

Use a local Cartesian solar frame, with `z` increasing upward and gravity

\[
\mathbf g=-g(z)\,\hat{\mathbf z},\qquad g(z)>0.
\]

Physical coordinate direction and array order are distinct. Store the atmosphere **top first**, so `z[k+1] < z[k]`. Unless explicitly labelled Python notation, equations and implementation examples below use Julia's one-based indices: `k=1` is the top and `k=NZ` is the bottom. Python `T_in[:, :, 0]` corresponds to Julia `T_in[:, :, 1]`; Python `z[-1]` corresponds to Julia `z[end]`.

For example, Python `z[0]=+3000 km` and `z[-1]=−500 km` mean Julia `z[1]=+3000 km` and `z[end]=−500 km`. These heights illustrate the input convention; grid sizes and coordinates remain runtime inputs. `Bz` is still positive in the physical upward direction; top-first storage does not reverse vector components or gravity.

Use SI internally: lengths in m, temperature in K, pressure in Pa, density in kg m⁻³, electron density in m⁻³, magnetic field in T, and continuum extinction in m⁻¹. Accept km and gauss only through explicit input conversions. In particular, `1 G = 10⁻⁴ T`, `1 Pa = 10 dyn cm⁻²`, `1 g cm⁻³ = 10³ kg m⁻³`, `1 cm⁻³ = 10⁶ m⁻³`, and `1 cm⁻¹ = 100 m⁻¹`.

Keep the following names distinct:

- `mu_view = cos(theta)`: viewing cosine;
- `mu_bar`: dimensionless mean particle mass in atomic-mass units, supplied by the EOS if needed;
- `mu0`: magnetic permeability used in the SI Lorentz force.

Recommendation: default to constant solar gravity `g = 274 m s⁻²`, configurable. An optional spherical-height law needs an explicitly defined reference radius and the meaning of `z=0`; use the same gravity in equations, boundaries, and tests. The pressure boundary belongs to a specified height, not a particular optical depth.

### 2.2 Static momentum equation

Neglecting velocity, viscosity, radiation force, rotation, and turbulent stress, the governing equations are

\[
0=-\nabla P+\rho\mathbf g+\mathbf L,
\qquad
\mathbf L=\mathbf J\times\mathbf B,
\qquad
\mathbf J=\frac{1}{\mu_0}\nabla\times\mathbf B,
\qquad
\nabla\cdot\mathbf B=0.
\]

Define the force residual with a single sign convention throughout:

\[
\mathbf R=\nabla P+\rho g\hat{\mathbf z}-\mathbf L.
\]

Thus exact equilibrium requires

\[
\partial_xP=L_x,\quad
\partial_yP=L_y,\quad
\partial_zP=-\rho g+L_z.
\]

These equations contain magnetic pressure **and** tension. Adding `B²/(2 mu0)` to gas pressure in a vertical HSE equation does not give general 3D MHS. The standard MHS force equation is also stated in [Terradas and Neukirch, three-dimensional MHS models](https://doi.org/10.1051/0004-6361/202244687).

### 2.3 EOS closure and physical assumptions

The baseline closure is a local LTE map

\[
(\rho,n_e,P_e,\text{species},\text{status})
=\operatorname{EOS}(T,P;\text{abundances, atomic data}).
\]

`P` is the **total thermal gas pressure**, including electrons, not hydrogen pressure or electron pressure. Determine density and electrons from the same EOS state. Do not combine a fixed mean molecular weight with an independently computed Wittmann electron density.

The user supplies temperature rather than an energy equation. Full HSE adjusts temperature within bounds; MHS holds it fixed. Neither mode establishes radiative equilibrium, an energy balance, or dynamical stability. Use a single strongly coupled fluid with isotropic gas pressure as the initial approximation. Ion–neutral drift, anisotropic pressure, and dynamic support require different closures if later found necessary.

LTE is a controlled first approximation, not a claim of realistic chromospheric ionization. Time-dependent hydrogen ionization can differ substantially from LTE and instantaneous statistical equilibrium in chromospheric simulations. See [Leenaarts et al. (2007)](https://arxiv.org/abs/0709.3751). Future options must distinguish prescribed populations, statistical-equilibrium NLTE, and genuinely time-dependent non-equilibrium ionization. Temperature and pressure alone cannot specify the latter.

## 3. Full nonmagnetic HSE: feasibility comes before iteration

### 3.1 Why full HSE constrains horizontal temperature variations

For `B=0`, the horizontal equations require

\[
\partial_xP=\partial_yP=0\ \Longrightarrow\ P=P(z).
\]

With the same vertical gravity at every horizontal location,

\[
\rho(z)=-\frac{1}{g(z)}\frac{dP}{dz}.
\]

Therefore density is also horizontally uniform. On a validated, single-valued thermodynamic branch of a uniform-composition LTE EOS, fixed `P` and `rho` determine a horizontally uniform temperature `T_*(z)`.

This is a consequence of the specified force balance. It is not an optional smoothing choice. Small temperature changes are sufficient only if the cube is already sufficiently close to such a stratification. Changing pressure alone, changing the initial guess, or using a different elliptic solver cannot balance a horizontal pressure gradient without a horizontal supporting force.

The full-HSE output is still returned as 3D arrays for API consistency, but it represents a plane-parallel thermodynamic state. Arbitrary horizontal structure can survive only if the physical assumptions change, for example by adding magnetic support, other stresses, or horizontal composition variation. None of those is part of the agreed nonmagnetic baseline.

### 3.2 Explicit bounded-temperature feasibility test

Let the allowed change at each input sample be represented by positive bounds

\[
T^-_{ijk}\le T_{*,k}\le T^+_{ijk}.
\]

For example, a relative allowance `epsilon_T` gives `T± = T_in(1 ± epsilon_T)`, but `epsilon_T` is **pending review**. At each height define

\[
T^-_k=\max_{ij}T^-_{ijk},\qquad
T^+_k=\min_{ij}T^+_{ijk}.
\]

If `T⁻_k > T⁺_k` anywhere, no horizontally uniform temperature can satisfy all bounds there. Return `hse_infeasible_temperature_bounds` with the offending heights, conflicting extrema, and required changes. This is an analytic infeasibility certificate under the stated EOS branch and assumptions.

For a uniform symmetric relative allowance, the minimum allowance required by the horizontal spread at a height is

\[
\epsilon_{\min,k}
=\frac{T_{\max,k}-T_{\min,k}}{T_{\max,k}+T_{\min,k}}.
\]

Example: a layer spanning `4000–8000 K` requires at least a `33.3%` pointwise relative allowance; a `5%` allowance cannot work. A layer spanning `5900–6100 K` needs at least approximately `1.67%`. Compute this diagnostic without assuming a chosen bound.

If the intersections are nonempty, select a unique profile by an explicit objective. Recommendation, still for review:

\[
\min_{T_{*,k}\in[T^-_k,T^+_k]}
\sum_{ij}w_{ijk}
\left[\ln(T_{*,k}/T_{{\rm in},ijk})\right]^2.
\]

With no cross-height penalty, its solution is the weighted geometric mean at each height, clipped to the **feasible intersection**. Use horizontal dual-cell areas as baseline weights; if temperature uncertainties are supplied, include their log-temperature variances. A vertical smoothing penalty couples heights and changes this selection problem; it must be explicit and must not violate pointwise bounds.

Feasibility of these intersections does not override EOS validity or additional future boundary constraints. Check those separately. Until a temperature bound and profile-selection rule are approved, offer the diagnostics and a clearly marked candidate profile, not an accepted full-HSE solution.

### 3.3 Pressure solution after temperature selection

Set `u = ln(P/P_ref)` with a fixed `P_ref = 1 Pa`. Integrate downward from the actual top sample:

\[
\frac{du}{dz}
=-\frac{g(z)\rho(T_*(z),P_{\rm ref}e^u)}{P_{\rm ref}e^u},
\qquad
u(z_1)=\ln(0.1\,\mathrm{Pa}/P_{\rm ref}).
\]

Integrate in increasing array index, from `k=1` toward `k=NZ`. Because `dz=z[k+1]−z[k]<0` during downward integration, pressure rises downward. This scalar initial-value problem avoids a global 3D nonlinear solve. Evaluate the EOS at every integration stage, then recompute all output states at the requested sample heights. A positive pressure follows from the log variable, but overflow, excessive density, and EOS failure must still be detected.

Recommendation: adaptive embedded Runge–Kutta, such as `Tsit5`, with piecewise linear positive-temperature interpolation and explicit stops at input knots. Limit steps relative to the local pressure scale height and temperature-resolution scale; adaptive substeps cannot reconstruct an unresolved temperature transition. Start with log-pressure local error controls around `10⁻⁸` only after the EOS accuracy is improved to support them. An implicit method is a fallback if measurements demonstrate stiffness, not a substitute for resolving invalid EOS evaluations. See [OrdinaryDiffEq solver usage](https://docs.sciml.ai/DiffEqDocs/stable/api/ordinarydiffeq/usage/).

For a conservative cross-check, verify each interval using

\[
P_{k+1}-P_k
\simeq\int_{z_{k+1}}^{z_k}\rho(z)g(z)\,dz,
\qquad k=1,\ldots,NZ-1.
\]

Use quadrature from the continuous solution or a refinement study; a coarse-grid trapezoid is not expected to equal a high-order ODE solution to the ODE tolerance.

### 3.4 Independent-column HSE as a diagnostic tool

Implement `column_hse` as an explicitly named initializer/reference utility with the original `T_in` fixed. It solves the same vertical ODE independently in each `(i,j)` column and returns horizontal force residuals. It is useful for debugging, opacity checks, and MHS starting guesses. It must never be automatically substituted for requested full HSE or labelled as full 3D static balance.

## 4. MHS is a constrained inference problem when B can change

### 4.1 Why fixed T and fixed B can be incompatible

For a fixed magnetic field, pressure is one scalar unknown but momentum supplies three scalar equations, with density already tied to `T,P`. Necessary compatibility conditions include

\[
\nabla\times(\mathbf L-\rho g\hat{\mathbf z})=0,
\qquad
\partial_yL_x=\partial_xL_y,
\qquad
\mathbf B\cdot\nabla P=-\rho g B_z.
\]

On periodic or multiply connected domains, local curl conditions alone are insufficient; loop circulations also matter. Closed field lines constrain the integrated field-aligned pressure equation. The vertical equation requires `rho=(Lz−∂zP)/g>0`, so a candidate with `∂zP>=Lz` cannot give positive density. Check this together with the EOS, not as an afterthought.

The agreed production mode therefore adjusts B. It does not merely solve pressure for an immutable measured field. With noisy observations, some uncertainty sets may still contain no exact equilibrium.

### 4.2 Recommended mathematical formulation

Let `B_obs` denote the input magnetic cube. Unknowns are pressure `u=ln(P/P_ref)` and a solenoidal magnetic field `B`. Temperature is exactly `T_in`. Define an observation operator `H` that maps the internal magnetic representation to the user’s sampled component arrays.

Seek the closest admissible field by

\[
\begin{aligned}
\min_{u,\mathbf B}\quad &
\frac12[H\mathbf B-\mathbf B_{\rm obs}]^T
C_B^{-1}[H\mathbf B-\mathbf B_{\rm obs}]
+\lambda_B\mathcal S(\mathbf B-\mathbf B_{\rm obs})\\
\text{subject to}\quad &
\mathbf R(u,\mathbf B;T_{\rm in})=0,\quad
\nabla\cdot\mathbf B=0,\\
&P(z_1)=0.1\ \mathrm{Pa},\quad
\text{approved magnetic boundary constraints},\\
&\text{approved bounds on }H\mathbf B-\mathbf B_{\rm obs},\quad
\text{valid EOS states}.
\end{aligned}
\]

`C_B` is a componentwise uncertainty covariance or an approved approximation to it. With diagonal errors, the data term is a sum of squared component changes divided by `sigma_B_component²`. Apply covariance/whitening operators without constructing a dense production covariance. Zero uncertainty means a fixed datum, handled as a constraint rather than a division by zero. Confidence bounds and Gaussian fit weights are different concepts: the user has not yet specified whether bounds are hard, nor how many standard deviations are admissible.

`S` is optional, explicit regularization with stated physical length scales. Do not use it to hide force errors or impose a force-free chromosphere without approval. Positivity of P is represented by `u`; density and ne remain EOS outputs, not independent adjustable fields.

**Required before production MHS implementation:** magnetic uncertainty information, its statistical interpretation, and boundary conditions. Synthetic verification can proceed with artificial, explicitly declared uncertainties and known exact boundaries.

### 4.3 Solver recommendation

Use an infeasible-start constrained method: sequential quadratic programming or an augmented-Lagrangian method, with trust-region/line-search globalization and bound handling. The initial state need not satisfy all constraints. A practical staged route is:

1. Validate input, uncertainty, and boundary consistency; compute the initial Lorentz force and divergence.
2. Build pressure from `column_hse(T_in)` and use the supplied B as an initial observation state. Report all initial force components.
3. If useful, solve a declared weighted force/divergence least-squares restoration problem to approach feasibility while respecting data bounds.
4. Increase constraint enforcement and update multipliers; solve the constrained problem to the physical tolerances.
5. Validate unmodified full force residuals, magnetic divergence, pressure boundary, EOS consistency, and magnetic changes.

A penalty-only minimum is not an exact MHS solution. An approximate fit may be exported for diagnosis only with `approximate_only=true` and a failed physical-equilibrium acceptance flag.

Use matrix-free Jacobian products for the large problem, plus a sparse/block approximation for preconditioning. Supply transpose actions for optimization and validate them by dot-product tests. For small cases, assemble the Jacobian and compare against finite differences and a direct solve. Never form a dense Jacobian or dense `JᵀJ` at production scale.

The linearized coupling must include

\[
\delta\mathbf R
=\nabla(P\,\delta u)
+g\hat{\mathbf z}\,P\left(\frac{\partial\rho}{\partial P}\right)_T\delta u
-\frac{1}{\mu_0}\left[(\nabla\times\delta\mathbf B)\times\mathbf B
+(\nabla\times\mathbf B)\times\delta\mathbf B\right],
\qquad
\delta(\nabla\cdot\mathbf B)=\nabla\cdot\delta\mathbf B.
\]

For MHS, `delta T=0`. Derivatives must use the exact discretized residual, including boundaries and interpolation, rather than substituting continuous product rules that the discrete operators do not satisfy.

Preconditioning needs both pressure/gravity and magnetic/divergence blocks, with a Schur-complement approximation for constraints. Test multigrid or domain decomposition, vertical line relaxation, and semi-coarsening for anisotropic meshes. Unpreconditioned Krylov iteration and independent local solves are not a production strategy for a globally coupled problem.

Do not start magnetic continuation by claiming that `B=0` with arbitrary fixed `T_in` is a feasible full-HSE state. Section 3 proves why that is generally false. Any homotopy that temporarily changes T or exceeds uncertainty bounds is an initialization device and must reach the original target constraints before acceptance.

### 4.4 Julia solver infrastructure decision

Recommendation: Julia owns physics, EOS adapters, data, residual kernels, and diagnostics; evaluate PETSc through PETSc.jl for distributed vectors, Krylov methods, preconditioners, and optimization infrastructure. A small integration prototype must establish which constrained optimizer and matrix-free callbacks are available in the pinned version before selecting the production optimizer.

PETSc SNES provides nonlinear equation solvers; it does not by itself convert an overdetermined force system into the constrained inference problem above. PETSc TAO BRGN supports bounded regularized nonlinear least squares and can support restoration or diagnostic fits; it is not automatically an equality-constrained MHS solver. See [SNES documentation](https://petsc.org/release/manual/snes/), [TAO BRGN documentation](https://petsc.org/release/manualpages/Tao/TAOBRGN/), and [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl).

If an adequate distributed SQP interface is unavailable, implement the outer augmented-Lagrangian updates in Julia with documented inner optimization and PETSc-backed linear algebra. Pin all versions and benchmark this on small exact equilibria before committing to a production implementation. Package availability alone is not evidence that this coupled solver will converge.

### 4.5 Feasibility and failure language

An inconsistent fixed boundary flux can yield a proven failure, as can a violated hard data bound at an immovable boundary. In contrast, a nonlinear optimizer reaching its iteration limit does not prove that no MHS solution exists. Return `mhs_no_feasible_solution_found` or `solver_failed` with diagnostics unless there is a specific mathematical certificate. Multiple admissible equilibria or local minima may exist; use the data objective, restart experiments, and sensitivity studies to assess ambiguity.

## 5. Boundary conditions and magnetic admissibility

### 5.1 Pressure

Interpret `Pgas_top=0.1 Pa` as gas pressure at the **first** supplied geometrical plane, `z[1]` in Julia or `z[0]` in Python; this is `+3000 km` in the example. Impose `Pgas[:, :, 1]=Pgas_top` in Julia. Allow another positive scalar through configuration. A spatial top map is an API extension, not the agreed default.

For full HSE, one pressure value and the chosen temperature stratification suffice; bottom pressure follows by integration. Do not independently prescribe bottom pressure unless solving an additional constrained problem.

For MHS, pressure data and magnetic boundaries must be mutually compatible. A uniform pressure top implies zero tangential pressure gradient there, hence `Lx=Ly=0` at that plane for an exact smooth solution. A fully fixed observed top B whose Lorentz force violates this condition cannot be repaired by interior pressure changes alone.

### 5.2 Magnetic boundaries: still required

Specify each of the six faces: fixed normal flux, uncertain observed field, periodic pair, or a precisely defined extrapolation/matching model. State which tangential components may adjust and by how much. The supplied cube gives sample values at its edges, but does not by itself specify whether these are hard boundary values, uncertain observations, or inputs to an exterior model.

For periodic x/y, require explicit periods and no duplicated endpoint convention; verify temperature, magnetic observations, errors, and pressure constraints across the seam. Do not assume periodicity from array shape.

For fixed normal fields on a closed box, require

\[
\oint_{\partial\Omega}\mathbf B\cdot\mathbf n\,dS=0.
\]

This is necessary for a solenoidal interior. Any permitted flux correction must be an explicit uncertainty-weighted adjustment. A magnetic cleaning projection changes B and must be counted in the same data budget as all other changes.

For nonperiodic sides, select a real boundary model; “open” is not a mathematical specification for a static elliptic/inverse problem. Test boundary sensitivity by moving/cropping the domain or varying admissible boundary values. Do not impose incompatible values at edges and corners from different faces.

## 6. Why a pressure Poisson equation is useful but insufficient

From the vector momentum equation,

\[
\nabla^2P=-\partial_z(\rho g)+\nabla\cdot\mathbf L.
\]

For log pressure,

\[
\nabla u=-\frac{\rho g}{P}\hat{\mathbf z}+\frac{\mathbf L}{P},
\qquad
\nabla^2u=-\partial_z\left(\frac{\rho g}{P}\right)
+\nabla\cdot\left(\frac{\mathbf L}{P}\right).
\]

These are necessary scalar consequences of force balance. Solving them only constrains the divergence of a vector residual; a nonzero divergence-free residual can remain. Boundary/loop information and the vector equations still matter. This is the mathematical reason to validate all force components after a FIRTEZ-inspired scalar solve.

A scalar Poisson solve can serve as a diagnostic baseline or pressure preconditioner. With all-Neumann boundaries its compatibility condition and constant nullspace require explicit handling. A top Dirichlet anchor removes a pressure normalization ambiguity, but does not establish magnetic/thermodynamic compatibility or replace the missing magnetic boundaries.

The Poisson-like approach is motivated by [Borrero et al. (2019)](https://arxiv.org/abs/1910.14131) and [Borrero et al. (2021)](https://arxiv.org/abs/2101.04394). Our requirement for independent vector-residual acceptance is a design inference from the equations, not a claim that those papers prove this production problem is always solvable.

## 7. Findings from the supplied implementations

### 7.1 RH

Inspected [RH Hydrostatic](/Users/harshm/Documents/WorkRepo/RH/rhf1d/hydrostat.c) and `convertScales` in [multiatmos.c](/Users/harshm/Documents/WorkRepo/RH/rhf1d/multiatmos.c).

- `Hydrostatic` solves local coupled density/charge/hydrogen constraints using column mass and existing hydrogen populations. Its assumptions include quantities from the surrounding radiation/EOS calculation. It is not a self-contained LTE height-grid pressure ODE taking only T and a top pressure.
- It includes optional turbulent and magnetic contributions in its local pressure relation. Those are not general 3D Lorentz-force support.
- `convertScales` integrates density and continuum opacity to connect column mass, height, and reference optical depth. Its geometric branch sets a finite top optical-depth estimate using half the first interval.
- RH depth arrays are traversed from the upper atmosphere toward deeper layers in these routines. This matches the requested top-first ordering; still verify units, physical height signs, and zero-based C versus one-based Julia indices.

Reuse the conservation checks, pressure/particle accounting, and opacity-integration ideas. A comparison with RH must match populations, composition, gravity, pressure terms, top boundary, opacity conventions, and ordering before numerical differences can be interpreted.

### 7.2 FIRTEZ

Inspected [hydrostatic_tools.f90](/Users/harshm/Documents/WorkRepo/firtez-dz-mhs-master/src/hydrostatic_tools.f90), [mhs_tools.f90](/Users/harshm/Documents/WorkRepo/firtez-dz-mhs-master/src/mhs_tools.f90), and [mhs_struc.f90](/Users/harshm/Documents/WorkRepo/firtez-dz-mhs-master/src/mhs_struc.f90).

| Observed behavior | Design implication |
|---|---|
| `RK4_INTEGRATION_LOG` integrates log pressure downward and calls the EOS at intermediate stages. | Useful scalar-HSE pattern; replace hardcoded units, interpolation floors, and shared scratch state. |
| `MHS_SOLVER` iterates a Poisson equation for `ln P` using FISHPACK `HW3CRT`. | A useful comparison method, not the complete adjustable-B constrained formulation. |
| `GET_DIVLORENTZPRE` limits current estimates with a plasma-beta-dependent threshold before forming the magnetic source. | This changes the force being balanced; do not silently copy it into an exact physical residual. |
| `LIMIT_MAGTERM` exponentially extends/attenuates the magnetic source above its magnetic sensitivity region; pressure updates are bounded and underrelaxed. | Audit which physical terms are being suppressed; retain raw residual diagnostics. |
| `APPLY_THRESHOLDS` clamps T to `2000–14000 K` and B components to `±10⁴ G`. | These restrictions would alter our fixed-T magnetic input problem and are not inherited defaults. |
| `INTERPOL_INVDATA` remaps inversion quantities through optical depth during the MHS loop. | Our input T lives on a fixed geometrical grid. Do not introduce optical-depth remapping into the fixed-T MHS solve. |
| Lateral pressure boundaries derive from average/inversion models; the active solver requires lateral Dirichlet conditions. | Those boundary models do not follow from the new input contract. |
| Reported `CHI2N` tracks Wilson-depression changes; the shown loop also stops at its iteration cap. | This is not a sufficient full vector-force convergence criterion. |
| Arrays in the MHS structures are predominantly `REAL(4)`; geometry uses uniform spacings in the box representation. | Use Float64 and explicit nonuniform-grid operators in the new code. |
| `GET_TAU` integrates `opacity × rho` downward with a separately imposed `TAU_MIN`. | Optical-depth upper-boundary information is distinct from gas-pressure upper-boundary information. |
| `CHROMOSPHERE` and `BETAE` branches already exist. | The checkout is not accurately summarized by a universal hard cutoff at `logtau500 = −3`. |

The user’s prior photospheric-domain concern remains a validation requirement for this checkout. It should not be generalized to every FIRTEZ version: [Borrero et al. (2026), section 3.1](https://arxiv.org/html/2603.10543v1#S3.SS1) describes chromospheric Stokes inversions with NLTE treatments and an MHS pressure update. That is a broader inversion workflow and does not establish that an arbitrary fixed-T cube plus bounded B changes admits an exact static solution.

### 7.3 Supplied Wittmann EOS and opacity

Inspected [witt.py](/Users/harshm/Documents/WorkRepo/MHS/wittman-eos/witt.py), [witt_eos_cpp.cpp](/Users/harshm/Documents/WorkRepo/MHS/wittman-eos/witt_eos_cpp.cpp), and [pf_Kurucz.input](/Users/harshm/Documents/WorkRepo/MHS/wittman-eos/pf_Kurucz.input).

- Python exposes `pe_from_pg`, `rho_from_pg`, `rho_from_pe`, and `contOpacity`. Its documentation describes detailed hydrogen species and limited molecular chemistry, not a general chromospheric NLTE closure.
- `contOpacity` ultimately returns **extinction including scattering** as its first result; optional second output is scattering. `cop` sets `OPACITY=A+B` and `SCATTER=B`. Do not add scattering twice.
- The Python wavelength convention in `cop` is Å: use `5000 Å`, corresponding to `500 nm`. Its extinction is in cgs inverse-length units and needs conversion to m⁻¹.
- `HOTOP()` is a dummy returning zero. This is an explicit opacity-completeness issue to audit at the requested temperature range, not proof that the whole continuum calculation is unusable.
- Python `pe_from_pg` and the C++ counterpart cap iteration at 250 and return a value without an explicit nonconvergence status. An outer success return is not evidence that the inner EOS converged.
- Python’s constructor accepts `prec` but assigns a fixed `1e-5`; the C++ version has `PREC=1e-5`. EOS noise can therefore limit derivatives and outer convergence.
- Python abundance normalization modifies a class-level array and uses mutable scratch arrays. Create independent contexts and test repeated construction, custom compositions, and threading.
- C++ already exposes SI thermodynamic entry points, including `witt_thermodynamics_from_pgas`. It constructs an EOS object in the shown batch call. Rebuilding tables per scalar ODE-stage call would be costly: use persistent contexts or a native Julia port.
- The C++ continuum-state routines return background populations, not a complete final 500-nm extinction routine. A compiled opacity adapter or Julia port is still needed for production.

These are inspection findings, not runtime validation results. Audit partition-function temperature coverage, endpoint clamping, species coverage, conservation, and all positivity/flooring rules over the actual pressure–temperature envelope before trusting extrapolated chromospheric values.

## 8. EOS and opacity interfaces

Recommended logical interface; names are a proposed contract rather than claims about existing Julia functions:

```julia
eos_state!(state, eos_context, T_K, P_Pa)
# state: rho_kg_m3, ne_m3, Pe_Pa, species, drho_dP_T,
#        convergence_status, iterations, conservation_errors

continuum500!(opacity, opacity_context, state, T_K)
# opacity: chi_abs_m1, chi_scat_m1, chi_ext_m1, validity_status
```

Requirements:

1. Store immutable composition/partition data once per process; allocate scratch state per task or worker context. Do not rely on mutable global state or unsafe thread-ID indexing under task migration.
2. Return inner iteration status and physical residuals. Add finite-value, positivity, `0 < Pe < P`, species/charge conservation, and round-trip checks. Support diagnostic relative errors when ne is extremely small.
3. Tighten the EOS solve until its noise is comfortably below requested outer accuracy. Validate `drho_dP_T` against a sequence of finite-difference step sizes. A black-box FFI call cannot automatically be differentiated by Julia AD.
4. Prefer a native Julia production port if its results can be verified against both supplied implementations. A persistent C ABI backend is a valid first reference. Keep the Python implementation as a small-case oracle rather than a threaded hot loop.
5. Keep original constants for exact legacy comparisons, then make any constants update an explicit, separately validated change.
6. An EOS table is optional optimization only after direct calls pass. Tables need monotone/positive interpolation, derivative consistency, quantified errors, and explicit out-of-domain behavior.
7. A later prescribed-NLTE interface must update charge balance, pressure, density, species, and opacity consistently. Multiplying LTE ne by a departure factor while leaving the other quantities unchanged is not a general EOS closure.

## 9. Optical depth on the height grid

### 9.1 Definition and upper boundary

Define the requested default as **vertical total continuum extinction optical depth at 500 nm**, with scattering included:

\[
\tau_{500}(x,y,z)=\tau_{500,\rm top}(x,y)
+\int_z^{z_1}\chi_{500,\rm ext}(x,y,z')\,dz'.
\]

This convention needs review if the intended reference is absorption-only. If the opacity backend instead returns a mass coefficient `kappa` in m² kg⁻¹, convert once using `chi=kappa rho`. Never multiply an already volumetric extinction by rho.

`Pgas_top=0.1 Pa` does **not** determine `tau500_top`. Material above `+3000 km`, its opacity, and its thermal structure are not supplied. Required choice:

- **Truncated box:** set `tau500_top=0` and explicitly interpret tau as optical depth through the supplied box only. Then `logtau500(top) = −Inf` mathematically.
- **Exterior atmosphere:** supply `tau500_top` or an exterior profile to integrate; a positive top value gives finite logs everywhere.
- **Cell-center/half-cell convention:** only if the input samples are explicitly redefined as cell centers with an outer boundary above the first sample. Do not silently use this to make a top-node log finite.

Recommendation for the draft API: explicit `tau_top=0` with `tau_definition=:truncated_box`, preserving `−Inf` at the top. If finite log values are required at every grid point, the user must choose a positive physical/exterior top optical depth. A separate plotting floor may be recorded but must not alter scientific tau values.

### 9.2 Discrete integration

For top-first, nonuniform decreasing z and positive nodal extinction, set `tau[:, :, 1]=tau_top` in Julia and accumulate toward increasing index:

\[
\tau_{ij,k+1}=\tau_{ijk}
+\tfrac12(\chi_{ijk}+\chi_{ij,k+1})(z_k-z_{k+1}),
\quad k=1,\ldots,NZ-1.
\]

The geometrical thickness `z[k]−z[k+1]` is positive. Tau therefore increases, or remains equal, as array index increases downward. Return both tau and `log10(tau)`; retain the logarithm convention in metadata. Check monotonicity, nonnegativity, and refinement convergence, allowing equal adjacent tau where increments fall below floating-point resolution. Never repair scientific tau by forcing arbitrary monotonic log increments. Missing `tau=1` inside the box is a valid outcome, not a reason to shift the supplied heights.

Compute opacity and tau after a force solution has been accepted, and optionally for diagnostics. In the LTE baseline they do not feed back into the momentum solve. This avoids introducing an unnecessary opacity/height iteration.

### 9.3 Inclined viewing

Equilibrium is solved in the local solar frame and is independent of viewing angle. Let the outward ray direction be

\[
\mathbf n=(\sqrt{1-\mu_{\rm view}^2}\cos\phi,
\sqrt{1-\mu_{\rm view}^2}\sin\phi,\mu_{\rm view}).
\]

For a horizontally uniform plane-parallel atmosphere and a top exit, `tau_ray=tau_vertical/mu_view` if the exterior contribution obeys the same geometry. For an inhomogeneous cube, integrate `chi(x(s),y(s),z(s)) ds` along a genuine 3D path; division by mu alone is insufficient. Rays can exit a side before reaching the top. Specify side/exterior treatment and ray-entry optical-depth data. Reject `mu_view<=0` for the initial top-observer API; near-limb paths may require a curved-domain extension.

Inputs in an observer frame require a full rotation, viewing azimuth, coordinate mapping, and disambiguated vector field. Mu alone is insufficient. Do not rotate gravity in a local-height grid just because the observer is inclined.

## 10. Grid representation and discrete operators

### 10.1 Input contract

Treat `x,y,z` as **sample locations**, with the top pressure imposed at the first z sample. Require finite, strictly increasing x/y coordinates and strictly decreasing z coordinates; arrays must match the declared dimension order `(x,y,z)` and top-first vertical storage. Support nonuniform separable coordinates. A single rectilinear `z[NZ]` means all columns share the same absolute height levels; unknown per-column height offsets/Wilson depressions are outside this contract.

Infer `NX=length(x)`, `NY=length(y)`, and `NZ=length(z)` at runtime and validate all input shapes against them. Allocate arrays, build operators, partition work, and write output dimensions from these values. Do not hardcode the example dimensions or assume that dimensions are equal, powers of two, or divisible by thread/rank counts. Support uneven partitions and choose a process grid compatible with local stencil requirements; report resource or minimum-stencil limitations explicitly rather than imposing a fixed global grid size.

No automatic reversing, unit guessing, regridding, or padding of invalid data. Explicit adapters may perform these operations and must record them. Handle 1D/2D degenerate tests with dedicated operators; the 3D MHS stencil requires enough samples in every active direction.

Use signed coordinate differences for derivatives: for example, a vertical edge derivative is `(f[k+1]−f[k])/(z[k+1]−z[k])`, with a negative denominator in this storage convention. Use positive physical lengths/areas/volumes for integration and weights, and orient flux normals consistently with physical coordinates. If a library requires increasing interpolation knots, use an explicit internal adapter that reverses all related data consistently and maps results back; public inputs, outputs, and checkpoints remain top first.

### 10.2 Recommended staggered MHS representation

Use log pressure, T, and EOS state at input vertices, magnetic fluxes on primal cell faces, and currents/pressure differences on a compatible edge/dual representation. This preserves the actual top plane while supporting discrete magnetic flux conservation. `H` interpolates internal B to the original vertex observations; all uncertainty constraints apply **after H**, in the supplied observation space.

Build metric-aware incidence operators on the nonuniform mesh. A compatible complex should satisfy discrete `curl(grad)=0` and `div(curl)=0` to roundoff for the corresponding spaces. Obtain current using the metric-weighted adjoint curl appropriate to the staggered B space; it is not the same array operation as the node-to-edge gradient. Specify reconstructions for every cross-product component and boundary stencil.

The initial numerical prototype must document exact staggering, quadrature, interpolation, and boundary closure, then pass operator and manufactured-solution tests before this representation is frozen. Do not mix incompatible centered derivatives merely because all arrays have the same shape. If a simpler collocated prototype is used, treat it as a reference and explicitly test checkerboard/null modes before considering production use.

Compute pressure force as a difference of `P=P_ref exp(u)` at the chosen locations. Continuous `grad P=P grad u` is not an exact discrete identity with arbitrary averages. Define the gravity/source quadrature consistently, and prove refinement toward the continuum equilibrium. HSE truncation error must not be interpreted as a physical magnetic current.

Internal staggered arrays need not have `(NX,NY,NZ)` shapes; exported P/rho/ne and component observations do. Validate magnetic divergence on the native flux representation and also report a derivative diagnostic on exported B. Export/checkpoint the native representation so that resampling does not conceal divergence errors.

## 11. Convergence, scaling, and result classification

### 11.1 Scaling

Use physical normalization scales and metric/volume weights. For diagnostic cell/edge force residuals, a useful local force scale is

\[
S_F=\max(S_{F,\rm floor},|\nabla P|+\rho g+|\mathbf L|).
\]

Report `|R|/S_F` and absolute residuals in N m⁻³. For optimization, use documented **fixed** reference scales derived from the input/reference state, or update them only in an explicit outer strategy; otherwise the optimizer can improve a normalized objective by changing its denominator. Sample weights must account for nonuniform physical volumes and must not inadvertently let densely sampled layers dominate.

For magnetic divergence report absolute T m⁻¹, signed net boundary flux, and a normalized metric such as

\[
\epsilon_{\rm div}=\frac{\ell_{\rm local}|\nabla\cdot\mathbf B|}
{\max(B_{\rm floor},|\mathbf B|)}.
\]

Record the choice of local length, floor, and quadrature. Also report `beta=2 mu0 P/B²` as a regime diagnostic; low beta does not justify dropping pressure/gravity or clipping currents.

In low-beta regions, a force norm normalized by the magnetic force can conceal an error large compared with gas-pressure support. Also report residuals relative to thermal/gravity scales and per-component force budgets, using explicit floors. Judge chromospheric thermodynamic accuracy from those diagnostics as well as a total-force norm.

### 11.2 Acceptance checks

Always report componentwise, layerwise, and global maxima, volume-weighted RMS, and high percentiles for the **raw original equations**, including boundary-adjacent regions. Include locations of worst errors and EOS failures. Separate interior and boundary summaries; do not exclude the chromosphere to improve an aggregate norm.

Proposed initial verification targets, to be calibrated with mesh refinement and EOS accuracy:

| Quantity | Proposed acceptance approach |
|---|---|
| Full HSE temperature bounds | Every sample respects its approved bound, with only roundoff allowance. Empty intersections fail before solving. |
| Fixed MHS T | Bitwise unchanged input T, barring explicit I/O conversion. |
| Hard B bounds | Every observation-space component respects the approved interval. |
| Top pressure | Enforce directly; verify relative error near Float64 roundoff. |
| ODE HSE solution | Compare with analytic/small-case independent integration; target relative pressure error around `1e-6` after EOS tightening. |
| MHS force | Begin synthetic exact cases with normalized RMS target `1e-6` and maximum `1e-4`, then show refinement; these are review targets, not universal physical accuracy guarantees. |
| Solenoidality | Aim for near-roundoff compatible-operator identities; constrained-solve divergence needs a separately recorded target, initially normalized maximum `1e-8`. |
| EOS | Inner residual/error substantially below the outer budget; no hidden nonconvergence or unapproved extrapolation. |
| Optical depth | Correct sign, positive extinction, monotonic tau, correct upper boundary, and at least second-order trapezoidal convergence on smooth resolved tests. |
| Optimization | Constraint tolerance plus stationarity/projected-gradient and complementarity tests; a small update alone fails acceptance. |

Report the distinction between error with respect to the discrete equations and error with respect to a continuum reference. A discretely converged state can still be poorly resolved. Quantify changes in P/rho/ne/logtau and B under grid refinement, EOS-tolerance tightening, boundary variations, and uncertainty changes.

### 11.3 Structured statuses

Use statuses such as `success_full_hse`, `success_mhs`, `hse_infeasible_temperature_bounds`, `invalid_input`, `boundary_incompatible`, `eos_failure`, `opacity_failure`, `mhs_no_feasible_solution_found`, `solver_failed`, and `approximate_only`.

Allow `equilibrium_valid=true` with `optical_depth_valid=false` if the force solve succeeded but opacity failed; the complete requested output then remains incomplete. Preserve the last valid iterate and diagnostics on failure. Do not silently floor pressure, alter temperature, clip currents, or relax bounds to claim success.

## 12. Validation methodology and release gates

### Gate A: units, EOS, and opacity before spatial solvers

- Audit SI/cgs conversions against dimensional identities and independent sample values.
- Sweep the anticipated T/P range, including cold low-pressure chromospheric states and hot layers. The actual temperature envelope is still required from the user.
- Compare Python, C++ and Julia EOS values, conservation residuals, iteration status, monotonic density response, and inverse round trips. Agreement of ports alone is not independent physical validation.
- Compare continuum extinction and its absorption/scattering split with a matched RH background-opacity calculation at 500 nm. State omitted species/processes and table boundaries.
- Inject EOS failures and out-of-range states; confirm failures propagate rather than yielding plausible-looking output.

### Gate B: nonmagnetic equations and feasibility

1. **Analytic ideal-gas isothermal HSE:** with a test EOS of constant `mu_bar`, require `P(z)=Ptop exp[(ztop-z)/H]`, `H=kB T/(mu_bar m_u g)`. This isolates signs, units, integration, and boundary placement.
2. **Variable-T column HSE:** use smooth and sharply resolved profiles, nonuniform grids, and independent integration. Tightening ODE tolerance must cease to help once EOS/interpolation error dominates.
3. **Full-HSE bound tests:** replicated profiles, small variations with intersecting bounds, and empty intersections. Recover the analytic `epsilon_min` and identify failing heights.
4. **False-HSE test:** a horizontally structured T cube solved by independent columns must show horizontal force residuals and must fail full-HSE acceptance.
5. **Column mass:** for constant g, verify `P(z)-Ptop = g ∫_z^ztop rho dz` without invoking opacity.
6. **Top-first indexing:** verify the pressure boundary is at Julia index 1, `z[1]>z[end]`, and downward HSE pressure/tau increase with index. Test nonuniform decreasing z, derivative signs, paired internal reversals, and output ordering. Reject ascending input z under this API unless passed through an explicit documented adapter.

### Gate C: magnetic operators and physical equilibria

1. Zero B and uniform B must reduce to the appropriate full-HSE problem. With fixed horizontally varying T, a zero-field state is generally not a feasible MHS reference.
2. A known curl-free, solenoidal nonuniform B must produce zero Lorentz force to truncation error and approach HSE under refinement. This tests cancellation beyond a uniform field.
3. Verify compatible identities, adjointness, net flux, polynomial derivative accuracy on stretched grids, and absence of pressure/magnetic checkerboard modes.
4. **Nonzero-force vertical benchmark:** choose `B=(b(z),0,0)`, magnetic pressure `M=b²/(2 mu0)`, and a positive pressure profile. Set `rho=−(P'+M')/g>0` and recover T from the test EOS or an independently inverted validated Wittmann EOS. Verify `d(P+M)/dz=−rho g`.
5. **Horizontal-force benchmark:** choose `B=(0,b(x,z),0)`, `M=b²/(2 mu0)`, `P=Q(z)−M(x,z)>0`, and `rho=−Q'(z)/g>0`. Set `b(x,ztop)=0`, `Q(ztop)=0.1 Pa`, and obtain T from the EOS. This exact 2.5D equilibrium has both horizontal and vertical magnetic support and directly tests all active force terms. Use compatible periodic x data or exact nonperiodic boundaries.
6. Add a fully 3D independently verified equilibrium benchmark before release. Also use a fully 3D manufactured-source problem to test spatial order and parallel halos; a nonzero manufactured source is an operator test, not evidence of zero-source MHS equilibrium.
7. Perturb exact B with declared errors and recover an admissible equilibrium with fixed T. Compare magnetic change normalized by uncertainty, force residual, and recovered P/rho/ne. Uniqueness is not guaranteed, so compare objective and constraints as well as pointwise values.
8. Deliberately incompatible boundary flux, too-small error budgets, bad top tangential Lorentz force, and noisy unresolved currents must produce informative failure/qualification.
9. Freeze B in a diagnostic pressure-only Poisson solve and construct a case with a small scalar residual but a large vector residual. It must fail physical acceptance.

### Gate D: chromospheric and reference-code validation

Use a range of photosphere–chromosphere reference columns with explicitly matched EOS assumptions, not a single quiet-Sun profile. Test cold regions, steep thermal gradients, low beta, and the entire height range; do not stop validation at `logtau500=−3`.

Compare with RH HSE only after matching its population closure and pressure support. Compare with FIRTEZ first on its own intended configuration, then on a controlled case where clipping/remapping/BC differences are isolated. Do not require identical output from algorithms solving different equations.

An MHD snapshot is a realism/stress test, not automatically an exact MHS ground truth. Its momentum budget may contain inertia and other forces, and its EOS may be non-equilibrium. Where possible obtain velocity/time derivatives, pressure, density, ionization, and boundary data to estimate missing support. A residual consistent with those missing terms is different from a discretization failure.

### Gate E: parallel correctness and production acceptance

- Match serial, threaded, MPI, and hybrid outputs within documented reduction-order tolerances; require identical result classification.
- Test uneven subdomain sizes, narrow subdomains, all boundaries, nonuniform grids, restart on a different process grid, and collective failure propagation.
- Compare Jacobian-vector and transpose products, conservation integrals, and optical-depth prefix sums across decompositions.
- Establish repeatability from several initial guesses on synthetic feasible cases, and report failures without claiming a global existence result.
- Benchmark multiple configurable sizes and aspect ratios after smaller cases pass; `504×504×357` is one optional example. Record solver accuracy along with speed.

## 13. MPI plus threads on Olivia/Sigma2

### 13.1 Scale and memory

For any supplied grid, let `N=NX×NY×NZ`. One Float64 scalar cube requires `8N` bytes; four scalar outputs require `32N` bytes, and T plus three input B components plus those four outputs require `64N` bytes. Compute memory estimates from the actual dimensions before allocation, using overflow-safe size arithmetic.

As an illustration only, `504×504×357 = 90,683,712` samples. One Float64 scalar cube is `725,469,696 bytes`, approximately `0.676 GiB`. Four requested scalar outputs occupy approximately `2.70 GiB`; T plus input B plus those outputs is approximately `5.41 GiB` before uncertainties, native face/edge arrays, halos, EOS scratch, optimizer vectors, multipliers, Krylov bases, and preconditioners.

About 40 scalar-cube equivalents would be `27.0 GiB`, and 100 about `67.6 GiB`; actual staggered dimensions differ. A four-field primal state is about `2.70 GiB`, so retaining 30 such Krylov vectors already costs roughly `81 GiB` globally, before constraint variables. Budget peak memory explicitly rather than inferring it from output size.

Use matrix-free products, bounded Krylov restart sizes, reusable work arrays, and distributed I/O. Do not replicate production cubes on every rank or gather all optimization vectors on rank 0.

### 13.2 Distinct decomposition strategies

**Full nonmagnetic HSE:** parallelize horizontal feasibility extrema and weighted profile reductions. Reduce `O(NZ)` summaries across ranks, solve one scalar profile, and distribute/broadcast it. Thread local diagnostics/output expansion. A global 3D PDE solver provides no benefit for this mathematical problem.

**Column-HSE utility:** use a 2D horizontal process grid, keeping entire z columns local; distribute independent columns among Julia tasks. Vertical integration is sequential within each column, with variable cost handled by task-level scheduling. No halo exchange is needed for the ODE; horizontal residual diagnostics need neighbors.

**MHS:** use a distributed Cartesian 3D process grid with metric-aware halos. Full-height horizontal pencils are a useful first layout if they scale adequately, but the architecture must allow z decomposition. Use local blocks sized for cache and communication costs; avoid a forced one-cell vertical thickness. Preconditioners need coarse-grid communication across the whole domain.

**Optical depth:** with full columns local, integrate from the first/top sample toward increasing local/global z index. With z decomposition, compute each rank’s total slab extinction and use an ordered prefix scan from the physical top to add overlying contributions; construct communicator order explicitly rather than assuming MPI rank numbers encode height. Count shared intervals exactly once. Inclined-ray traversal needs separate distributed routing/interpolation and should be added after vertical tau is validated.

### 13.3 Thread ownership and MPI safety

Use MPI.jl with a requested thread level compatible with the actual calling pattern. Recommendation: `THREAD_FUNNELED` initially, with MPI calls only on the verified main thread. Julia tasks can migrate, so a communication phase must be deliberately kept on that thread; an arbitrary task that happens to make one MPI call at a time is not sufficient. Check the provided thread level and `MPI.Is_thread_main()` in debug builds. See [MPI.jl environment documentation](https://juliaparallel.org/MPI.jl/stable/reference/environment/).

Use phases of halo exchange, threaded interior kernels, completion, then boundary kernels. Synchronize access to send/receive buffers. Keep PETSc calls on the permitted thread unless its configured threading behavior explicitly permits more. Collective EOS/solver failures must be communicated so other ranks do not deadlock in later collectives.

One worker layer owns local parallelism: use Julia threads and single-threaded C++ EOS calls, rather than recursively launching C++ worker teams inside every Julia task. Set BLAS and OpenMP backends to one thread unless a measured solver phase intentionally owns them. Allocate EOS scratch by task/context and reuse it.

### 13.4 Olivia launch and tuning

The official Olivia CPU overview lists two 128-core AMD CPUs and 768 GiB physical memory per CPU node. Allocation limits and usable memory may differ by partition. Treat these as planning context and confirm the active allocation at deployment. See [Olivia architecture](https://documentation.sigma2.no/hpc_machines/olivia/overview.html) and [current job types](https://documentation.sigma2.no/jobs/job_types/olivia_job_types.html).

Benchmark several ranks/threads per node, for example `8×32`, `16×16`, and `32×8` on a 256-core CPU node. These are experiments, not a promised optimum. Start with NUMA-local placement and first-touch allocation, verify binding, and profile EOS cost versus bandwidth, halo exchange, global reductions, and preconditioning.

An illustrative Slurm resource sketch for review:

```bash
#SBATCH --account=<project-account>
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Launch with the command verified for the chosen Sigma2 MPI stack:
# srun [site-specific MPI flags] julia --project=. scripts/run_equilibrium.jl config.toml
```

This is not a ready-to-submit production job: account, Julia/MPI/PETSc modules, launcher flags, partition limits, and binding must be validated on Olivia. Ensure MPI.jl and PETSc use compatible MPI libraries; do not mix an arbitrary bundled MPI with a different system PETSc. Follow [Sigma2 job scripts](https://documentation.sigma2.no/jobs/job_scripts/olivia_job_scripts.html) and [PETSc.jl HPC configuration](https://juliaparallel.org/PETSc.jl/stable/man/hpc/).

Warm up/precompile before measuring. Measure one-node through multi-node strong scaling on fixed-size cases and weak scaling at fixed local volume. Report total wall time, nonlinear/linear iterations, EOS calls, communication fraction, peak RSS, and checkpoint time. Production MHS is likely limited by solver/preconditioner behavior before arithmetic throughput; measure rather than assume.

## 14. Data products and software organization

Recommended Julia package structure:

```text
src/
  Geometry.jl          # units, coordinates, metrics, frames
  EOS.jl               # closure interface and state/status types
  Wittmann.jl          # validated backend/port
  Opacity500.jl        # extinction and species adapters
  FullHSE.jl           # bounded feasibility, profile selection, ODE
  ColumnHSE.jl         # reference/initializer utility
  MagneticOps.jl       # staggered operators and observation map
  MHSResidual.jl       # force/divergence, derivatives, boundaries
  MHSOptimize.jl       # constraints, uncertainty objective, globalization
  Parallel.jl          # ownership, halos, reductions, tau scan
  Validation.jl       # raw residuals and acceptance
  IO.jl               # input, output, metadata, checkpoint/restart
```

Recommended logical API:

```julia
solve_full_hse(T_in, x, y, z;
    temperature_bounds, profile_objective,
    Pgas_top=0.1, eos, opacity, gravity, tau_top, parallel)

solve_mhs(T_in, Bx_in, By_in, Bz_in, x, y, z;
    magnetic_uncertainty, magnetic_bounds, boundaries,
    Pgas_top=0.1, eos, opacity, gravity, tau_top, parallel)
```

Do not put unapproved values behind silent defaults for `temperature_bounds`, `magnetic_uncertainty`, or magnetic boundaries. `T_out` and `B_out` are essential additional outputs: pressure/density alone would conceal the adjustments used to achieve equilibrium.

Return at least:

- `Pgas`, `rho`, `ne`, `tau500`, `logtau500`, `T_out`, and magnetic outputs where applicable;
- `delta_T`, `delta_B`, normalized changes, native magnetic fluxes, and uncertainty/bound checks;
- force residual components, divergence, boundary errors, EOS/opacity validity masks, and layer summaries;
- solver status, physical acceptance status, objective terms, constraint norms, iteration history, and worst-cell coordinates;
- units, frame, shape/dimension order, explicit `vertical_order=:top_first`, decreasing z coordinates, gravity, composition, EOS/opacity versions and hashes, tau convention, pressure boundary height, all bounds/weights, and software/MPI provenance.

Recommendation: portable HDF5 datasets with explicit dimension labels and units, initially using per-rank files plus a manifest or a validated parallel-HDF5 build. Restart stores the native distributed unknowns, geometry, multipliers, optimizer state as available, and configuration hashes. Coordinate-based repartitioning must support a different process grid. Preserve input arrays and original observational values.

## 15. Why previous attempts may have failed

No previous failure logs or implementation were supplied beyond the reference repositories. The following are hypotheses to test, not a diagnosis of the earlier runs.

| Possible cause | Distinguishing evidence | Correct response |
|---|---|---|
| Full HSE requested for a horizontally structured cube with tiny T allowance | Empty temperature-bound intersections | Report certified infeasibility; no iterative solver can fix it. |
| B treated as fixed while only pressure is varied | Persistent horizontal or rotational residual after scalar convergence | Use the agreed adjustable-B constrained problem and an actual error budget. |
| Inconsistent magnetic/pressure boundaries | Net flux error or top tangential force incompatible with constant P | Resolve boundary admissibility explicitly. |
| Scalar convergence mistaken for vector equilibrium | Small Poisson residual but large raw R | Fail physical acceptance. |
| Hidden current clipping or pressure floors | Modified-equation residual small, original residual large | Track modifications; remove from accepted physical equations. |
| EOS stopped internally or was too noisy | Inner residual plateau, unstable numerical derivatives | Expose status, improve inner solve, validate derivatives. |
| Unit, sign, or indexing error | Analytic scale-height test fails | Fix before nonlinear tuning. |
| Transition underresolved or B noisy | Residual/current changes strongly under refinement/smoothing | Refine data/model and propagate uncertainty; do not just increase iteration limits. |
| Optical-depth remapping changes fixed input T/B | Iteration no longer solves the declared geometrical-grid problem | Keep tau as an output in the LTE baseline. |
| Poor scaling or weak preconditioner | Inner Krylov count grows with grid size/aspect ratio | Improve blocks, multigrid, and physical scaling. |
| Snapshot is dynamically supported | Residual correlates with estimated inertial/missing forces | Qualify the static model; do not claim exact equilibrium exists. |
| Thread races or inconsistent halos | Different answers/status by decomposition | Fix ownership, operator closure, and collective error handling. |

For the next attempt, retain per-iteration raw force components, EOS status, bound activity, objective/constraint histories, line-search decisions, and representative vertical cuts. A single final “did not converge” message is insufficient to identify the failing layer.

## 16. Implementation sequence and stop conditions

1. **Review this design:** settle temperature bounds/profile selection, magnetic uncertainties and boundaries, tau upper boundary, units/frame, and physical validity envelope. Missing MHS inputs are not permission to invent them.
2. **EOS/opacity audit:** implement SI adapters, persistent contexts, inner status, conservation/derivative checks, and small-case reference comparisons.
3. **HSE reference and full-HSE feasibility:** implement the analytic bound test, diagnostic column ODE, selected full-HSE profile solve, and tau integration. Demonstrate both successful and certified-infeasible inputs.
4. **Discrete magnetic prototype:** freeze a tested staggering and observation map; pass operator, adjoint, boundary, and exact-equilibrium tests.
5. **Small constrained MHS prototype:** explicitly supplied synthetic uncertainty/boundaries, assembled Jacobian/direct reference, then matrix-free derivatives and globalization. Demonstrate feasible recovery and qualified failure.
6. **Distributed implementation:** MPI halos, hybrid threading, scalable preconditioner, distributed tau/I/O, and restart. Re-run the same physical acceptance suite.
7. **Chromospheric validation and Olivia scaling:** full height/temperature/low-beta range, uncertainty/boundary sensitivity, matched RH/FIRTEZ comparisons, then a range of runtime-selected production grid sizes and aspect ratios.
8. **Inclined-ray extension and improved EOS:** separate follow-on gates; do not let viewing geometry alter gravity in the equilibrium solver.

Do not proceed from small MHS cases to the target cube until physical residuals converge under refinement and the accepted result is insensitive to rank/thread decomposition within tolerance. Do not label the LTE baseline as a validated NLTE or dynamic chromospheric atmosphere.

## 17. Questions still requiring review or additional inputs

The objectives are now clear. These choices remain unresolved and are deliberately not treated as accepted defaults:

| ID | Required decision/input | Why it matters |
|---|---|---|
| Q1 | Pointwise temperature-change limits: relative, absolute, temperature-dependent, or uncertainty-based? | Defines whether full HSE is feasible. |
| Q2 | If full HSE is feasible, approve weighted log-temperature projection or another precise selection objective. | Bounds alone do not choose a unique T profile. |
| Q3 | Supply per-component magnetic uncertainty cubes/covariance or an explicitly approved error model, including correlations/systematics where known. | Defines the adjustable-B inference problem. |
| Q4 | Are B changes hard bounded, statistically weighted, or both? What confidence threshold is allowed? | Determines admissible equilibria and the meaning of failure. |
| Q5 | Specify all magnetic boundary faces, their uncertainties, and periodicity/exterior models. | Needed for flux consistency and a defined global problem. |
| Q6 | Confirm x/y/z/B units, local versus observer frame, azimuth/disambiguation, and common absolute height registration. | Derivatives and Lorentz forces depend directly on these. |
| Q7 | Supply actual T range, horizontal extents/spacings, nonuniform-grid patterns, typical B range/noise, and missing-data policy. | Sets EOS coverage, resolution, and memory/solver tests. |
| Q8 | Choose `tau500_top`/exterior atmosphere or accept truncated tau and `−Inf` at the top; confirm total-extinction convention. | Ptop alone cannot give absolute optical depth above the box. |
| Q9 | Approve composition/abundances and molecular coverage; provide a first representative cube and reference atmosphere if available. | Enables meaningful physics validation. |
| Q10 | Approve baseline gravity and omitted stress terms; identify whether shocks/flows are expected in input snapshots. | Determines whether static balance is an appropriate model. |
| Q11 | Provide Olivia project/software environment and the intended CPU-node allocation when implementation starts. | Needed to verify MPI/PETSc compatibility and launch/binding. |
| Q12 | Supply previous failure cases/logs/configurations and desired scientific tolerances/runtime targets. | Allows diagnosis and measurable acceptance beyond this provisional design. |

Already specified: grid dimensions are arbitrary runtime inputs; `504×504×357` is only an illustrative example. All cubes are top first: Python `T[:, :, 0]` is the top, `z[0]=+3000 km` and `z[-1]=−500 km` in the example; Julia uses first index 1 and final index `end`. The uniform gas pressure at that first/top plane is `0.1 Pa`, the production target is MPI plus threads on Olivia/Sigma2, and the initial closure is LTE Wittmann.

## 18. Provenance and implementation handoff

Prepared on 15 October 2026. Local source inspection and mathematical derivation were performed; neither reference code was executed as a validation experiment. Web references support specific physical/library/context claims, not a convergence guarantee for the proposed solver.

Primary source entry points for the implementing model:

- RH: `rhf1d/hydrostat.c::Hydrostatic`, `rhf1d/multiatmos.c::convertScales`.
- FIRTEZ: `src/hydrostatic_tools.f90::RK4_INTEGRATION_LOG`, `src/mhs_tools.f90::{MHS_SOLVER,GET_DIVLORENTZPRE,LIMIT_MAGTERM,INTERPOL_INVDATA,SET_XYBC,GET_TAU}`.
- Wittmann: `wittman-eos/witt.py::{witt,pe_from_pg,rho_from_pg,contOpacity,cop,HOTOP}` and SI C ABI entry points in `wittman-eos/witt_eos_cpp.cpp`.

SHA-256 of inspected key files:

```text
witt.py
4eeb83bb28c7b7251d9223d6dea5961d4d60be4891db38af1073eff15993771a
witt_eos_cpp.cpp
28f905a4fa41db79e0028e0fa70a27b30334d47448586a27381a0b572cff000e
pf_Kurucz.input
a7d15b25406b2736331253b6457d8ef4cfa3f3ef2275462120999121b8a734a8
RH/rhf1d/hydrostat.c
40aa13d8bbd4c342a0a08afea2f95ea39867b64f006a35a5dcd7b3a5700ba5c3
FIRTEZ/src/mhs_tools.f90
0e792cc707728bc16d3421a9ab8c023743928ad516c8f25f3d5467c9a8423e97
```

**Handoff requirement:** implement only the reviewed mathematical modes and explicit inputs. Preserve the user’s fixed-T MHS requirement, small bounded HSE temperature changes, and physical residual acceptance. When a required scientific input is absent, return an actionable missing-input diagnostic; when feasibility is unresolved, report that uncertainty accurately rather than changing the equations to manufacture convergence.
