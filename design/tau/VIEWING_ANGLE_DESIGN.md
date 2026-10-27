# Viewing angle (μ, azimuth) for the fixed-B modes: design for review

*Status: design for review, 26 Oct 2026. Branch `viewing-angle` (from `main` 5e901b5). Nothing here is implemented.
Implementation starts after the user approves this document; decisions are recorded in §10. History of the project:
[`../DESIGN_HISTORY.md`](../DESIGN_HISTORY.md).*

---

## 0. Request and the user's answers (26 Oct 2026)

The user asked for a viewing angle in the fixed-B modes, with **two geometries**:

- **plane-parallel**: each column vertical, with τ scaled by μ;
- **inclined**: true 3-D slanted rays.

It goes on a new branch, because `main` and `embedding-api` are in production for FFNOML.

| # | Question | Answer |
|---|---|---|
| 1 | Which two angles? | Inclination θ (μ = cos θ) and azimuth φ |
| 2 | One angle or a map? | **One angle per box** |
| 3 | Rays leaving the box | **Periodic** boundaries |
| 4 | Pixel grid and column meaning for τ inputs | Open: the user asked for a recommendation → §2 |
| 5 | Field frame | Ambiguity not resolved in the inversion; resolved later (minimum-energy method); representation open → §3 |
| 6 | Height-mode outputs | Not clear; explained and proposed in §5 |
| 7 | `full-mhs-z` | Stale, to be deprecated: **out of scope** |
| 8 | Where μ comes from | HSEMHS config; the embedding API takes it as an argument (FFNO3D will pass it) |
| 9 | Branch | `viewing-angle` from `main`; μ = 1 defaults keep FFNOML behaviour |
| 10 | Test data | None: build a round trip from Bifrost (§7) |
| 11 | Process | This document first, in `design/tau/` |

## 1. Geometry and notation

Local solar frame: x, y horizontal, z up, with the box's own axes.

- Direction to the observer: **n** = (sin θ cos φ, sin θ sin φ, cos θ), with μ = cos θ and 0 < μ ≤ 1.
- φ is measured from +x toward +y. At φ = 0 the observer is tilted toward +x.
- Slant vector: **t** = tan θ (cos φ, sin φ).

A ray reaching the observer is the straight line

    x = ξ + t_x z,   y = η + t_y z,

with (ξ, η) its **footprint**: where it crosses the reference height z = 0. Along the ray, ds = dz/μ, and the optical
depth seen by the observer is

    τ_LOS(point) = ∫ from the point up to the top along n of χ₅₀₀ ds.

**Sheared coordinates (ξ, η, z)** are the core of the design. A ray is a line of constant (ξ, η). Derivatives transform
as follows:

    ∂_x = ∂_ξ,   ∂_y = ∂_η,   ∂_z|_(x,y) = ∂_z|_ray − t_x ∂_x − t_y ∂_y.

On a periodic box, the map from (x, y) to (ξ, η) at fixed z is a translation, so **periodicity in x, y is periodicity
in ξ, η**. This is what the user chose in Q3.

- **Plane-parallel geometry** sets **t = 0** in every geometric term and keeps μ only in the optical depth: the RH 1.5-D
  convention, τ_LOS = τ_vert/μ along a vertical column.
- **Inclined geometry** keeps both.
- **μ = 1** gives t = 0 in both, i.e. today's code (§8.1).

## 2. Recommendation for Q4: what a τ-grid input means

**Recommended contract.** For μ < 1 a τ-grid input describes **one line of sight per pixel**:

- T and B are given on a common log₁₀ τ_LOS axis, which is what an inversion of off-centre Stokes profiles returns.
- The pixels lie on a **regular, deprojected footprint grid** (ξ_i, η_j) in the local solar frame, with x, y in
  metres as now.

The footprint at z = 0 is where the ray reaches the mean τ_LOS = 1 surface, because heights keep the gauge
mean z(τ_LOS = 1) = 0.

Why this, rather than the plane-of-sky grid:

1. **The solver needs a regular periodic grid in the local frame.** Its centred differences, distributed FFT and
   Fourier preconditioner all assume one. Sky pixels are foreshortened by μ along the tilt direction and live in a
   rotated frame.
2. **Easy to interpret.** Pixel (i, j) is "the atmosphere the observer sees at footprint (ξ_i, η_j)". Every node of that
   column sits at the known position x = ξ_i + t_x z, y = η_j + t_y z. With μ = 1 this is exactly today's input.
3. **The observer's job stays outside HSEMHS.** Converting sky pixels to a regular footprint grid is a fixed affine
   resampling for a small field of view: stretch by 1/μ along the tilt direction and rotate. Remapping to a heliographic
   or local frame is already a standard step in vector-magnetogram pipelines, as are the frame transformation and the
   disambiguation (§3). FFNO3D, or a small helper script later, does it once per observation.

Consequences for the user:

- **Pixel spacing.** dx, dy in the file are footprint spacings. For a sky grid with pixel size p, the footprint spacing
  is p along the axis perpendicular to the tilt and p/μ along the tilt.
- **Plane-parallel and inclined read the same file.** They differ only in physics: plane-parallel pretends each column
  is vertical at its footprint.

## 3. Recommendation for Q5: the magnetic field

**HSEMHS keeps taking B in the local solar frame**: B_x, B_y, B_z with z vertical and x, y the box axes, already
disambiguated. Every equation in the solvers (J = ∇×B/μ₀, the Lorentz force, β) needs it there.

What has to happen before HSEMHS, in the inversion pipeline (FFNO3D):

1. **Inversion** gives (B, γ, ψ) or (B_LOS, B_⊥, ψ) in the line-of-sight frame per τ_LOS level. ψ is ambiguous by
   180°.
2. **Disambiguation.** The minimum-energy method minimises |∇·B| plus the electric current
   (Metcalf 1994, Sol. Phys. 155, 235). It is the most-used algorithm and was ranked well in the method comparisons:
   Metcalf et al. 2006 ([workshop paper](https://www.cora.nwra.com/~graham/pubs/ambigworkshop2.pdf)) and
   [Leka et al. 2009](https://researchwith.njit.edu/en/publications/resolving-the-180-ambiguity-in-solar-vector-magnetic-field-data-e/)
   (Sol. Phys. 260, 83, with noise).
   Its vertical derivative comes from a potential or linear force-free extrapolation, and it slows down as the grid
   grows ([review](https://arxiv.org/pdf/1706.05769)).
   - **Open point for depth-resolved inversions:** the ambiguity has to be resolved consistently at every τ level, not
     only in one layer.
3. **Frame transformation** to the local frame using θ, φ and the orientation of the image axes (the standard
   magnetogram transformation, e.g. Gary & Hagyard 1990, Sol. Phys. 126, 21).

**Proposed HSEMHS support, phase 2 and optional:**

- A helper `los_to_local(B_LOS, B_⊥, ψ; μ, φ, image_angle)`, for the transformation only, with ψ already
  disambiguated.
- A *diagnostic* that flags pixels whose Lorentz force or ∇·B is large relative to their neighbours. A wrongly flipped
  pixel produces large currents, so these maps may point at residual ambiguity errors.

HSEMHS will not disambiguate.

## 4. τ modes (`column-hse-ltau`, `mhs-poisson-ltau`)

On the recommended contract (§2), each column is a ray at a fixed footprint. Level k of column (i, j) is the node
(ξ_i, η_j, τ_LOS,k) at the solved height z_kij. No interpolation of the input is ever needed: the grid already follows
the rays.

### 4.1 Heights and column HSE (both geometries)

Along the ray, dτ_LOS = χ ds = χ dz/μ, so:

    dz/d ln τ = −μ τ/χ,       (a := τ/χ becomes μ τ/χ in the height integral)

Hydrostatic along the ray (no Lorentz force, horizontal gradients neglected) gives dP/ds = −μ ρ g, so:

    d ln P/d ln τ = μ g τ ρ/(χ P),       top closure  P = μ g τ ρ/χ.

**`column-hse-ltau` is therefore identical for the two geometries**: same P and z on every ray. Only the horizontal
position of the nodes differs: footprint (plane-parallel) versus slanted.

Implementation: `hse_tau!` takes `g_eff = μ g` and scales `dz` by μ. `align_ptot!` and `gauge!` are unchanged; they act
on heights only.

**Plane-parallel is exactly the old code on a shifted axis.** Since τ_vert = μ τ_LOS, running today's code with
log₁₀ τ_vert = log₁₀ τ_LOS + log₁₀ μ gives the same P and z. This is the regression test of §8.

### 4.2 `mhs-poisson-ltau`: residuals and Jacobian

The current operators on the grid:

- level derivatives D_ξ, D_η (`sddx!`, `sddy!`, centred and periodic);
- column derivative D_ζ (`dzcol!`, along the ray per unit height);
- height slopes z_ξ = D_ξ z and z_η = D_η z.

The physical derivatives are:

    ∂_x = D_ξ − z_ξ D_ζ,    ∂_y = D_η − z_η D_ζ,
    ∂_z = D_ζ − t_x ∂_x − t_y ∂_y = (1 + t·∇z) D_ζ − t_x D_ξ − t_y D_η.

Today ∂_z = D_ζ. Only the inclined geometry adds the t terms.

**Force balance**, with u = ln P and e = 1/P: one horizontal pair at fixed height, plus the component **along the ray**.
These span 3-D for μ > 0, and the ray component integrates down the column like today's vertical row. Using
∂_ray = ∂_z + t·∇_h and the two balances −∇P − ρg ẑ + L = 0:

    ∂_ray u = −W_n,    W_n = g ρ/P − (L_z + t·L_h)/P.

The existing residuals keep their form, with **W → W_n** and **a → μ a**:

    R_x = −D_ξ u − W_n z_ξ + L_x e,        R_y = −D_η u − W_n z_η + L_y e,
    R_z,f = u_(k+1) − u_k − ½ Δln τ_k [ (W_n μ a)_k + (W_n μ a)_(k+1) ].

(Derivation: ∂_x u|_z = D_ξ u − z_ξ D_ζ u, and D_ζ u = ∂_ray u = −W_n. It is the same substitution the current code
makes with W.)

**The Lorentz force** comes from `lorentz_tau!`, with ∂_z replaced by the slanted form above.

**Jacobian changes**, all linear and local:

- (a) μ scales `a` and `aχ`. This is trivial.
- (b) W_n depends on L_h, which depends on the heights. This adds `t·δL_h e` terms to `dw` and to the coefficient
  arrays (`px`, `pXx`, …, `zζ`, `zX`, `zY`).
- (c) The slanted ∂_z depends on ∇z through the factor (1 + t·∇z). This adds `t_x D_ζB`-type coefficients to the X/Y
  (∂δz) couplings.

`Jop!` and `JTop!` keep their structure; only these coefficient arrays get slant terms. Verification uses the existing
adjoint test (exact) and finite differences (≤ 1e-6) at θ = 0°, 30° and 60°, and φ = 0° and 45°.

**Preconditioner.** Phase 1 keeps the layer-median operator A + |k|² B. μ enters through the medians of `aχ` and `W`.
The slant adds first-derivative couplings (a complex term i k·t per mode); these are omitted at first. CG iterations
will be measured at θ = 60°, and the term added only if CG degrades.

**Cut.** `logtau_cut` stays on the **input (LOS) axis**: the axis of the file, and the one the embedding API sees. At
μ = 0.5 a cut at log τ_LOS = −4 lies at log τ_vert ≈ −4.3, i.e. slightly higher. The log will report the median β at
the cut level, so the β ≳ 10 rule of the τ-cut study can be checked per run. Whether to cut on the vertical axis is a
decision for §10, after the experiments.

### 4.3 Outputs (τ modes)

- **Unchanged** arrays and layout, (τ, y, x) ≡ (k, η, ξ). `z` holds the node heights; `z_tau1` is on the footprint grid
  (by construction z(τ_LOS = 1) of each ray).
- **New attributes:** `view_mu`, `view_azimuth_deg`, `view_geometry`, `view_tx`, `view_ty`. Node positions follow from
  x = ξ + t_x z and are not stored, to save memory. A writer option can store `x_node`, `y_node` if wanted.
- **Plane-parallel only:** the attribute `logtau_vertical_offset = log10 μ`.

## 5. Height modes (`column-hse-z`, `mhs-poisson-z`): the meaning of Q6

In the height modes **the pressure does not depend on the viewing angle**. Hydrostatic or Poisson balance is solved on
the (x, y, z) grid, and μ only changes the **optical depth** computed afterwards. Q6 asked what to output for that
optical depth.

**Proposal:**

- `tau500` / `logtau500` on the native (x, y, z) grid, meaning **τ_LOS at each grid point**: the optical depth from that
  point to the observer along its own slanted ray. Plane-parallel: τ_vert/μ, as `tau.mu` does today.
- `z_tau1` on the **footprint grid** (ξ, η) = the same regular grid. It is the height where the ray with footprint
  (ξ, η) reaches τ_LOS = 1. Plane-parallel: the column at (x, y).
- **No resampling onto τ_LOS surfaces or onto a sky grid.** That is the forward model's or user's job. A height-grid
  cube with τ_LOS(x, y, z) is everything needed to cut it later.

**Implementation (inclined), shear and integrate:**

1. For each level k, shift χ horizontally by +t z_k (linear interpolation, periodic; positive, so τ stays monotonic).
   This gives χ on (ξ, η, z_k).
2. Integrate down each sheared column with ds = Δz/μ, using today's trapezoid and `exp_tail` top closure.
3. Read off z_tau1 on (ξ, η).
4. Shift τ back by −t z_k to (x, y, z_k).

MPI: the y shift is local in x slabs. The x shift needs columns from other ranks. Transpose the x slabs to y slabs (an
Alltoallv real-space transpose, re-using the pack logic of `Slab.transpose_fwd!`), shift in x and integrate there
(columns complete), then transpose back. Shifts can exceed the box (e.g. a 14 Mm top at 60°), so the shift is taken
modulo the period.

Cost: two transposes of one cube per output, which is small next to a Poisson solve.

## 6. Configuration and embedding API (Q8)

**Config.** A new section:

```toml
[view]
mu = 1.0                    # cos θ, 0 < mu ≤ 1
azimuth_deg = 0.0           # φ, from +x toward +y (ignored when mu = 1)
geometry = "plane_parallel" # or "inclined"
```

- It applies to all four fixed-B modes. `full-mhs-z` rejects mu ≠ 1.
- The existing `tau.mu` stays as a deprecated alias. Giving both with different values is an error.

**Embedding API.** `prepare_embedded(...; view = ViewOptions())`, with:

```julia
Base.@kwdef struct ViewOptions
    mu::Float64 = 1.0
    azimuth_deg::Float64 = 0.0
    geometry::Symbol = :plane_parallel     # :plane_parallel | :inclined
end
```

- The angle is fixed per workspace: one observation means one angle, and the slant enters the precomputed FFT
  preconditioner set-up.
- `ColumnZOptions.mu` stays for compatibility, and conflicts are rejected as in the config.
- The defaults reproduce today's results exactly, so FFNOML code runs unchanged if this branch is ever merged.
- The status gains `view` in `stats`.

Validation: 0 < μ ≤ 1, φ finite, geometry known. Warn when μ < 0.2: rays are then so slanted that 1.5-D inversions and
static balance are both questionable near the limb.

## 7. Test data: synthetic slanted cubes from Bifrost (Q10)

There are no off-centre inversions, so a round trip from Bifrost 385 (`IO/BIFROST_en024048_hion_0_504_0_504.nc`,
periodic) serves as test data.

1. **`make_slanted_tau.jl μ φ`**, for every footprint (ξ, η) on the Bifrost grid:
   - shear the height cube (§5) and compute χ₅₀₀ from Bifrost T and P with `cache/eos_table_fullz.h5`;
   - integrate τ_LOS along the rays;
   - interpolate T and the local-frame B onto the log τ_LOS axis of `IO/input-mhs-ltau.nc` (95 levels).

   It writes `input-ltau-mu{μ}-phi{φ}.nc` (same layout as the production input, with x, y = footprints) and a truth
   file with z, ln P and ln ρ per node. This extends `scripts/ltaucut/truth_on_tau.jl`.
2. **Runs:** μ ∈ {1, 0.866, 0.707, 0.5} and φ ∈ {0°, 45°}, with `column-hse-ltau` and `mhs-poisson-ltau`, in both
   geometries, at cut −4.
3. **Measures:** the τ-cut-study analysis (`scripts/ltaucut/analyse.py`), extended per μ:
   - z(τ_LOS = 1) error and correlation;
   - ln P error by τ band, below and above the cut;
   - the strong/quiet height difference (the slanted Wilson depression);
   - the force diagnostic with slanted derivatives;
   - steps, CG iterations and wall time.

   **Expected:** inclined ≈ the μ = 1 accuracy (27 km), with plane-parallel degrading as θ grows. The size of that gap
   is the scientific result.
4. **Height modes:** τ_LOS from the shear method against a brute-force ray tracer (trilinear sampling at small ds) on
   the Bifrost cube. Agreement is required in z(τ_LOS = 1) (target ≤ 1 km rms) and in log τ.

## 8. Tests

1. **Regression.** μ = 1 (either geometry) is **bitwise identical** to `main` in all four modes, standalone and
   embedded, on the 96² crops. The μ factors and slant branches are skipped at μ = 1.
2. **Plane-parallel equivalence.** τ modes at μ match today's code run on log τ + log₁₀ μ, to rounding.
3. **Analytic.**
   - Isothermal ideal gas: z(τ_LOS = 1) moves by H ln(1/μ) (or the exact exp_tail value).
   - Horizontally uniform atmosphere with uniform B: the inclined geometry gives the same P(z) as μ = 1, and the nodes
     shift by t z.
   - Sinusoidal χ: shear τ against the analytic ray integral.
4. **Jacobian.** Adjoint and finite differences at several angles (§4.2); adjoint of the transpose and shear.
5. **MPI.** 1 vs 4 ranks, with shifts crossing rank boundaries (θ = 60°, φ = 0°).
6. **Validation.** μ ∈ (0, 1], conflicting `tau.mu`/`[view]`, unknown geometry, `full-mhs-z` with μ ≠ 1.

## 9. Plan

| Phase | Work | Size |
|---|---|---|
| P1 | `ViewOptions`, config `[view]`, plumbing; μ in column HSE τ and the height integral (both τ modes, plane-parallel); tests 8.1–8.2 | small |
| P2 | Inclined geometry in `mhs-poisson-ltau`: W_n, slanted ∂_z in the Lorentz force, Jacobian coefficients; tests 8.3–8.4 | medium |
| P3 | Inclined τ_LOS for the height modes: real-space transpose, shear, integration, z_tau1 on footprints; ray-tracer check | medium |
| P4 | Synthetic slanted cubes and the experiments of §7 on the cluster; short report with figures | medium |
| P5 | `docs/physics` and `docs/technical` sections for the viewing angle; README; embedding-API notes for FFNO3D | small |
| later | `los_to_local` frame helper and the ambiguity diagnostic (§3) | small |

Each phase is committed and pushed on `viewing-angle`; the work waits for the user between phases where results need a
decision.

## 10. Decisions requested from the user

1. **Input contract (§2):** deprojected footprint grid, one LOS column per pixel on a common log τ_LOS axis.
2. **Field (§3):** local-frame, disambiguated B only; the frame helper and ambiguity diagnostic later and optional.
3. **Height-mode outputs (§5):** τ_LOS on the native grid and z(τ_LOS = 1) on footprints; no resampling.
4. **Cut axis (§4.2):** `logtau_cut` on the LOS axis for now, to be revisited after the §7 experiments.
5. **Angle per workspace (§6)** in the embedding API, not per solve.
6. **Go-ahead** for P1–P4 in that order.

| Date | Decision | By |
|---|---|---|
| 26 Oct 2026 | Inclination and azimuth; one angle per box; periodic; `full-mhs-z` out of scope; μ from config and the API; branch `viewing-angle`; design first | User |
| 26 Oct 2026 | Decisions 1–6 above accepted. Both τ modes offer `plane_parallel` and `inclined` (3-D) geometry, selected by `[view] geometry` in the config and by the `view` argument of the embedding API (FFNO3D). Go-ahead for P1–P4 | User |
