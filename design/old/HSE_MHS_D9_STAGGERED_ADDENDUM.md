# D9 addendum: staggered vector-potential discretization for `mhs` (M6)

*Status: design for review, 16 Oct 2026. It is triggered by the Gate C3(b) failure (`HSEMHS/reports/M4a_gate.md`, `HSEMHS/reports/M4a_c3_remedies.md`). It replaces spec §6.2 (operators) and the divergence term of §6.3 for the `mhs` mode only. `column_hse` is unchanged. Nothing here is implemented yet; implementation starts after user approval, recorded in spec §10.*

---

## 1. Why the collocated scheme is replaced

| Finding (M4a) | Cause | Consequence |
|---|---|---|
| Horizontal odd–even B checkerboards do not decay (b̃z keeps 30 %) | Centred 3-point first differences annihilate (−1)^i patterns. Such a B produces no current, no Lorentz force and almost no ∇·B. | Only the data and regularizer terms hold these modes; conditioning ~1e11. L-BFGS stalls (C1b ends `stalled`), which denies `success_mhs`. |
| Horizontal pressure checkerboard is felt only through λ_U = 1e-6 (curvature 5e-8) | The same null space of the centred ∇P | A near-free mode, stabilized only by the regularizer |
| Residual floor of exact equilibria, E3 (RMS \|R\|/S ≈ 1e-4 at Δz/H = 1/8) | NZ vertical equations (two one-sided rows) for NZ pressures with a free scale | The synthetic tolerance 1e-6 is unreachable; E3 had to relax Gate C |
| ∇·B only penalized (λ_D) | B is a free collocated field | One more stiff, ill-conditioned term |

**The remedy.** Use a staggered (Yee/MAC-type) mesh, with B represented as the discrete curl of a vector potential on cell edges, pressure at cell centres, and force balance evaluated on cell faces with compact two-point differences. That gives:

1. ∇·B = 0 exactly, as an identity of the discrete operators.
2. No odd–even null modes in ∇P or in curl B: compact differences see alternating patterns.
3. Per column, NZ pressures, NZ − 1 vertical equations and one global gauge, so exact discrete equilibria exist. The E3 floor disappears, and the spec's synthetic tolerance (1e-6 / 1e-4) becomes reachable again.

A face-staggered B without a vector potential (keeping λ_D) was considered and rejected: ∇·B stays a penalty, and the curl still has a null space unless the edge currents are compact.

## 2. Mesh and storage

Cell centres are the existing nodes (x_i, y_j, z_k). T, P = e^u, ρ and nₑ live there, so `column_hse`, the EOS table, the output nodes and the Bifrost input grid are unchanged.

| Location | Coordinates | Quantities | Local storage index |
|---|---|---|---|
| Cell centre | (i, j, k) | u, P, ρ, T | (i, j, k) |
| x-face | (i+½, j, k) | B_x, R_x | (i, j, k) |
| y-face | (i, j+½, k) | B_y, R_y | (i, j, k) |
| z-face | (i, j, k+½), k+½ between levels k and k+1 | B_z, R_z | (i, j, k), k = 0 … NZ |
| x-edge | (i, j+½, k+½) | A_x, J_x | (i, j, k), k = 0 … NZ |
| y-edge | (i+½, j, k+½) | A_y, J_y | (i, j, k), k = 0 … NZ |
| z-edge | (i+½, j+½, k) | A_z, J_z | (i, j, k) |

**Index convention.** A staggered index i means the position i + ½ ("right/top" face of cell i). In z, top-first storage keeps k = 1 at the top. z_{k+½} is the midpoint of z_k and z_{k+1} for 1 ≤ k ≤ NZ − 1. The boundary faces z_{½} and z_{NZ+½} sit half a cell outside, at z₁ + ½(z₁ − z₂) and z_NZ − ½(z_{NZ−1} − z_NZ).

**Grid metrics.**
- Δx_{i+½} = x_{i+1} − x_i (periodic wrap); cell widths Δx_i = ½(x_{i+1} − x_{i−1}).
- Same in y.
- Δz_{k+½} = z_k − z_{k+1} > 0 for 1 ≤ k ≤ NZ − 1; cell heights h_z,k as in doc §6.4.

z-face and edge arrays have NZ + 1 levels (k = 0 … NZ); the other arrays have NZ.

Halos: one cell in x and y, as now. Every operator below uses index offsets in {−1, 0, +1}, so `exchange_halos!` works unchanged on all staggered arrays.

## 3. Unknowns and gauge

```
x = (v,  A_x, A_y, A_z,  B̄_z)             v at centres, A on edges, one global scalar
u = v − v̄ + ln P_ref                        (spec D3 gauge, unchanged)
B = curl A + B̄_z ẑ                         (B̄_z: net vertical flux, which a periodic A cannot carry)
```

- A horizontally uniform B_x(z) or B_y(z) comes from horizontally uniform A_y(z) or A_x(z), which periodicity allows. Only the net B_z flux needs the extra scalar.
- The **gauge of A** (A → A + ∇χ) is a null space of Φ. Its directions are orthogonal to every gradient, so L-BFGS started from any A₀ never moves along them. No gauge penalty is needed. A diagnostic `max |∇·A|` is reported but not controlled.
- **Scaling (E4 analogue).** A is scaled by a fixed reference: A = A_ref ⊙ Ã with A_ref,e = σ̄ ℓ_e, where σ̄ is the local mean σ and ℓ_e the local edge-normal spacing, so that one unit of Ã changes B by about σ.

## 4. Operators (all compact two-point stencils)

**Curl, edges → faces** (exact discrete Stokes; div ∘ curl ≡ 0):

```
B_x(i+½,j,k)   = [A_z(i+½,j+½,k) − A_z(i+½,j−½,k)]/Δy_j   −  [A_y(i+½,j,k−½) − A_y(i+½,j,k+½)]/h_z,k
B_y(i,j+½,k)   = [A_x(i,j+½,k−½) − A_x(i,j+½,k+½)]/h_z,k  −  [A_z(i+½,j+½,k) − A_z(i−½,j+½,k)]/Δx_i
B_z(i,j,k+½)   = [A_y(i+½,j,k+½) − A_y(i−½,j,k+½)]/Δx_i   −  [A_x(i,j+½,k+½) − A_x(i,j−½,k+½)]/Δy_j  + B̄_z
```

Signs: "upward" is k → k − 1, so ∂/∂z at a cell centre is (f_{k−½} − f_{k+½})/h_z,k. Cell widths Δx_i, Δy_j, h_z,k are the face-to-face distances.

**Divergence, faces → centres:**

(div B)_ijk = (B_x,i+½ − B_x,i−½)/Δx_i + (B_y,j+½ − B_y,j−½)/Δy_j + (B_z,k−½ − B_z,k+½)/h_z,k

It is identically 0 for B = curl A + B̄_z ẑ. This is a unit test to rounding; λ_D is dropped.

**Current, faces → edges** (the dual curl), J = curl B/μ₀:

```
J_x(i,j+½,k+½) = [ (B_z(i,j+1,k+½) − B_z(i,j,k+½))/Δy_{j+½}  −  (B_y(i,j+½,k) − B_y(i,j+½,k+1))/Δz_{k+½} ] / μ₀
J_y(i+½,j,k+½) = [ (B_x(i+½,j,k) − B_x(i+½,j,k+1))/Δz_{k+½}  −  (B_z(i+1,j,k+½) − B_z(i,j,k+½))/Δx_{i+½} ] / μ₀
J_z(i+½,j+½,k) = [ (B_y(i+1,j+½,k) − B_y(i,j+½,k))/Δx_{i+½}  −  (B_x(i+½,j+1,k) − B_x(i+½,j,k))/Δy_{j+½} ] / μ₀
```

On the top and bottom boundary edges (k + ½ = ½ or NZ + ½), the z difference in J_x and J_y uses the one-sided value with the ghost B taken equal to the nearest interior B (zero normal gradient). These edge currents only enter the force at boundary faces, which carry no equation (below).

**Lorentz force on faces**, L = J × B, with each factor averaged to the face of the component:
- L_x at x-faces: J_y B_z − J_z B_y.
  - J_y: mean of the 2 y-edges (k ± ½) adjacent to the face.
  - J_z: mean of the 2 z-edges (j ± ½).
  - B_z: mean of the 4 z-faces (i, i + 1; k ± ½).
  - B_y: mean of the 4 y-faces (i, i + 1; j ± ½).
- L_y and L_z: by cyclic permutation.

All averages are arithmetic with equal weights. Their transposes are the same weights scattered back, written in gather form.

**Force residual on faces:**

```
R_x(i+½,j,k)   = (P_{i+1} − P_i)/Δx_{i+½}  − L_x
R_y(i,j+½,k)   = (P_{j+1} − P_j)/Δy_{j+½}  − L_y
R_z(i,j,k+½)   = (P_k − P_{k+1})/Δz_{k+½}  + g √(ρ_k ρ_{k+1})  − L_z      only for 1 ≤ k ≤ NZ−1
```

- The geometric mean ρ_{k+½} = √(ρ_k ρ_{k+1}) is exact for an isothermal exponential atmosphere at the face. With B = 0 the discrete vertical equations form a two-term recursion with an exact solution for any T(z), so C1 can reach machine precision.
- There are no equations on the top and bottom boundary faces. The force balance there is not imposed, matching the open vertical boundaries; the bottom pressure scale is fixed by the gauge.
- R_par and R_perp (diagnostics) are evaluated after averaging R and B to centres.

**Data operator, faces → centres** (spec §6.3 data term, still at the observed nodes):

(HB)_x,ijk = ½(B_x,i−½ + B_x,i+½), and the same for y and z.

Φ_B = (1/N_B) Σ_{c,n} ((HB)_c − Bobs_c)²/σ_c². Combined with exact ∇·B = 0, H has no null mode on the admissible space: the alternating mode that H annihilates has ∇·B ≠ 0 and is excluded. This is verified numerically by the C3 rerun.

## 5. Objective (replaces spec §6.3 for `mhs`)

```
Φ = λ_F/N_F Σ_faces w_f |R_f|²/S_f²                        (interior faces only)
  + 1/N_B Σ_{c,n} ((HB)_c,n − Bobs_c,n)²/σ_c,n²
  + λ_S/N Σ_n w_n Σ_a |h_a² ∂_aa ((HB) − Bobs)|²/σ²       (grid-scale regularizer on δB at centres)
  + λ_U/N Σ_n w_n Σ_{a∈x,y} (h_a² ∂_aa u)²                 (kept, default λ_U = 0; see below)
```

- **Weights:** w_f is the dual (face-control) volume over the mean. S_f = max(ρ₀g, 0.01·median_layer(ρ₀g)) at the face, the geometric mean of the two centre values, fixed for the whole solve as before.
- **λ_D is removed.**
- **λ_U defaults to 0.** Compact ∇P sees the pressure checkerboard. The term stays available, and the 1 % rule still applies.
- **E3 is withdrawn** for the staggered scheme. Synthetic gates return to the spec tolerances: RMS ≤ 1e-6, p99 ≤ 1e-4.

## 6. Adjoint

Write the forward pass as a chain of linear maps and pointwise nonlinearities:

```
Ã → A = A_ref Ã → B = C A + B̄_z ẑ → (J = C* B/μ₀) → averages M → L = (MJ) × (MB) → R = G P + g ρ̄ − L
v → u → P = e^u, ρ = ρ_EOS(T, P), ρ̄ = √(ρ_k ρ_{k+1})
```

Here C is the edge→face curl, C* the face→edge curl, M the averaging operators to faces, and G the compact gradient. Each linear map has an explicit gather-form transpose (Cᵀ, C*ᵀ, Mᵀ, Gᵀ, Hᵀ), implemented next to the forward map and unit-tested with ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ to 1e-13.

The gradient follows the same pattern as `MHSResidual.jl`:
- **Force term:** a = 2λ_F w R/(N S²) on faces. The u-gradient is Gᵀa·P, plus the vertical-face term g ρ̄ (a_z) distributed half to each neighbouring centre, multiplied by ρ_u.
- **B-gradient of the force term:** −(Mᵀ[(MB) × a]) through J, and −Mᵀ[a × (MJ)] directly. Then through C*ᵀ/μ₀, then Cᵀ, then A_ref.
- **Data term:** Cᵀ Hᵀ (2((HB) − Bobs)/(N_B σ²)) and the B̄_z component Σ of the z part.
- **Gauge chain on v:** unchanged.

The Gauss–Newton diagonal (Jacobi preconditioner) is assembled from the squared coefficients of the same maps, as `gn_diagonal!` does now.

## 7. Initialization

1. **u₀.** Integrate the staggered vertical recursion
   (P_k − P_{k+1})/Δz_{k+½} + g√(ρ_k ρ_{k+1}) = 0
   upward from the bottom anchor, solving for P_k with the EOS table by a safeguarded Newton step in ln P. This is an exact discrete equilibrium for B = 0 and differs from `column_hse` by O(Δz²). `column_hse` remains the user-facing product; this is only the MHS initializer.
2. **A₀, B̄_z₀.** Minimize the data term alone, a linear least-squares problem, with preconditioned CG. B̄_z₀ is the mean of Bobs_z. The result is the divergence-free field closest to Bobs in the σ-weighted norm, and it doubles as a diagnostic: ‖(HB₀ − Bobs)/σ‖ measures how far the observed field is from solenoidal on this mesh.
3. **Then B-first continuation** (Ã, B̄_z with v frozen), then the joint solve, as established in M4a.

## 8. Parallel layout

- No change in decomposition: 2D horizontal, full columns local.
- Each rank owns faces and edges with local index i ∈ 1:nx, meaning positions i + ½. The −½ neighbours come from the halo.
- Halo width 1 in x and y for every staggered array.
- Reductions, determinism rules, token-ring output and checkpoint are as before. The checkpoint stores Ã (3 edge fields), B̄_z and v.

## 9. Outputs

- **Node-centred B_out** = HB, as now, so consumers see the same layout.
- **Native staggered fields:** /output/Bx_face, By_face, Bz_face, and /diagnostics/A_x/y/z.
- **R at centres** (the average of the two faces) for the per-layer tables. Native face residuals go to /diagnostics/R_face_*.
- **`divB`** is reported at roundoff level as a check.

## 10. Tests and gates (re-run of Gate C, then M5)

| Test | Criterion |
|---|---|
| div ∘ curl | ≡ 0 to 1e-14·\|A\|/h² on random A, nonuniform z, 1–6 ranks |
| Transposes of C, C\*, M, H, G | ≤ 1e-13 |
| FD gradient, per field (v, Ã, B̄_z) and per term, ideal gas and Wittmann table | ≤ 1e-6 (expected ~1e-9) |
| Thread / rank invariance | bitwise / ≤ 1e-12 |
| C1a, C1b | residual RMS ≤ 1e-6, p99 ≤ 1e-4 (no floor); u − u_colhse = O(Δz²) |
| C2 potential field (analytic A) | L → 0 at order ≥ 1.9 |
| **C3** (checkerboards in u and in Ã and B) | both decay ≥ 10³; status `converged` |
| C4–C7 | as in implementation doc §14, M5 |

## 11. Effort and plan

| Step | Content | Estimate |
|---|---|---|
| M6a | `Staggered.jl` (metrics, C, C\*, M, H, G and transposes, tests) | 1 day |
| M6b | `MHSStaggered.jl` (objective, adjoint, GN diagonal, initialization, CG for A₀), gradient tests | 1–2 days |
| M6c | Gate C rerun on the staggered scheme, report | 0.5 day |
| M5 | Discrepancy principle, classification, C4–C7, on top of M6 | as planned |

The collocated implementation stays in the code as `scheme = :collocated` for comparison and is not used in production. The config key is `mhs.scheme = "staggered"` (default once Gate C passes).

## 12. Decisions requested

1. Approve replacing the collocated MHS discretization with the staggered vector-potential scheme above (spec D9 → recorded in §10).
2. Approve withdrawing E3 for the staggered scheme (synthetic tolerances back to 1e-6 / 1e-4) and dropping λ_D.
3. Confirm whether the Bifrost B in `IO/input-mhs-hse.nc` is cell-centred (interpolated) or native face-staggered. If native face values exist in the snapshot, the data term can compare on faces directly and skip H. (Bifrost itself is face-staggered; the FITS/NetCDF export may have been centred.)
