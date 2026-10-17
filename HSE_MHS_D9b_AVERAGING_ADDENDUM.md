# D9b addendum: data and Lorentz averaging operators, and the top pressure mode

*Status: **approved 16 Oct 2026** (spec §10); Φ_Q is subject to the M6d step-1 experiment. Originally a proposal, 16 Oct 2026. Nothing here is implemented beyond the verified options already in the code (`data_on_faces`, `gncg!`). It amends `HSE_MHS_D9_STAGGERED_ADDENDUM.md` §4–5 and §10. The evidence is in `HSEMHS/reports/M6c_gate.md` and `M6c_gn.md`; the experiments for §1 used commit `97904b1`.*

## 1. What fails, and why

### 1.1 The B checkerboard is a joint near-null mode of H and M

In C3 (16×16×33, uniform B = (20, 0, 50) G), an ε = 1e-5 checkerboard in Ã produces a δB with three properties:

| Quantity | Initial | After L-BFGS (centre data) | After L-BFGS (face data) |
|---|---|---|---|
| ‖δB‖ (relative to initial) | 1 | 0.63 | 0.0073 |
| h‖C\*δB‖/‖δB‖ (current content; ≈ 0 for a potential field) | 9.6 | 8.5 | 2.6 |
| ‖H δB‖/‖δB‖ (what the data term sees) | 1e-4 | 5e-3 | 0.50 |

The surviving error is **current-carrying** (not a potential field), and **almost invisible to H**. It also produces almost no linearized Lorentz force, which is why the force term does not remove it.

**Mechanism (Fourier symbol argument).** Every averaging operator in D9 is a symmetric two-point average onto a point midway between the samples:
- H averages each B component along its own normal direction.
- M averages J along the two directions named in addendum §4, and B over 2×2 stencils.

A symmetric average has symbol cos(κh/2), which is zero at the Nyquist wavenumber κh = π. Linearized about a uniform B₀, the face force is δL = M(δJ) × B₀ (J₀ = 0). So for any δB whose currents alternate along the averaging directions, M(δJ) = 0 exactly, and H(δB) = 0 for the normal-direction alternation of each component.

C\* and G are compact differences, with symbol 2 sin(κh/2)/h ≠ 0. They see the mode, but every path from δB to a residual passes through at least one symmetric average.

**Consequence for the redesign.** No re-averaging of M that stays symmetric about the face (wider stencils, Maxwell-stress form, evaluation at centres) can remove the null: each has the same zero at Nyquist. The fix must add an operator that senses the antisymmetric part of J or B, located where those quantities live, so that it is O(h²)-consistent for smooth fields. For H, the only symmetric-free option is to compare on the faces themselves.

### 1.2 The top pressure mode is an identifiability problem, made stiff by a sawtooth

After option (a), the C1b reference solve stalls with this error:
- δu = −4.8e-4, +3.3e-4, −2.2e-4, … in the top layers: a (−1)^k sawtooth decaying downward, horizontally uniform.
- A matching horizontally uniform δBx(z), whose current δJ_y = ∂_z δBx balances the sawtooth in the z-face equations.

This does not happen with B = 0 (converged to 2e-12). In u alone the sawtooth is an ordinary direction: its GN Rayleigh quotient is 2e-3, against 8e-5 for δP = const. The softness is in the coupled (δu, δB) pair:
- At the top, P ≈ 1e-3 Pa and β ≈ 1e-4.
- Balancing δP needs δBx ~ μ₀δP/B₀ ≈ 3e-10 T ≈ 1e-6 σ, which costs nothing in the data term.

Physically, pressure perturbations below δP_unobs ≈ B₀σ/μ₀ are not identifiable where β ≲ σ/|B|. At the C3 top this limit is ≈ 800 P_top.

## 2. Proposed changes

### 2.1 Data operator: compare on faces (replaces H in Φ_B)

Φ_B = (1/N_B) Σ_faces (B_f − (I Bobs)_f)² / σ_f², with N_B = NX·NY·(3NZ+1):
- I is the two-point centre→face average.
- At the boundary z-faces, I takes the nearest centre value.
- σ_f² = σ_a σ_b (product of the two adjacent centre σ).

- **No null mode:** the operator on B is the identity.
- **Bias:** O(h²∂²Bobs) from I. This is the same order as the existing truncation error, and far below σ for resolved data.
- **Already implemented and verified** as `data_on_faces` (FD 1.5e-10, GN = Hessian to 7e-13). Measured effect: B checkerboard decay ÷160 instead of ÷1.6.
- **If native face-staggered Bobs exist** (open question 3 of the D9 addendum), they are used directly, without I.
- **Unchanged:** the node-centred output B_out = HB, and the λ_S term on HB − Bobs.

### 2.2 Lorentz averaging: keep M, add a force-weighted current-Nyquist term Φ_Q

M stays as is: second-order, with exact transposes, and exact L = 0 for uniform and potential fields.

The new term Φ_Q gives the M-invisible currents force-like stiffness:

```
Φ_Q = λ_Q / N_F  Σ_edges w_e  Σ_{a ∈ avg(e)} ( B_ref · ¼ δ²_a J_e / S_e )²
```

- **avg(e)** is the set of directions along which M averages that edge current:
  - J_x: y and z
  - J_y: x and z
  - J_z: x and y
- **The z second difference** uses interior edge levels only (k = 1…NZ−1, excluding the boundary edges).
- **¼δ²_a** is the plain index second difference divided by 4. Its Nyquist symbol is 1 and its smooth symbol is O((κh)²/4).
- **B_ref, S_e** are layer weights at the edge level:
  - B_ref is the layer median of |Bobs|.
  - S_e is the geometric mean of the layer medians of S on the two adjacent layers.

  A Nyquist current δJ therefore costs (B_ref δJ/S)², the same as the force residual it would produce if M did not annihilate it.
- **w_e** is the edge control volume over the mean.

**Properties**
- Exactly zero for uniform B and for potential fields (J = 0), so C1b and C2 are unaffected.
- For smooth fields Φ_Q/Φ_F scales as λ_Q (κh)⁴/16. At the C2 resolution (κh ≈ 0.1) this is ≈ 6e-6 λ_Q, so the 1 % regularizer rule allows λ_Q up to O(100). Proposed default: λ_Q = 1.
- Linear in J at fixed weights, so its gradient, its GN seeds and its preconditioner diagonal are straightforward:
  - adjoint: δ²ᵀ = δ², then C\*ᵀ/μ₀
  - GN seeds: the same as for the gradient
  - preconditioner diagonal: exact squared stencil coefficients

**Why not the alternatives**
- Wider or one-sided averages in M: these break the symmetry of M and the exactness of the transposes, and introduce O(h) errors.
- A Maxwell-stress form: same Nyquist zero (§1.1).
- The B-normal regularizer λ_N already tried: it acts on B with data weight, so the mode stays force-soft (÷36 at λ_N = 1). Φ_Q acts on J with force weight.

### 2.3 Top pressure mode: vertical grid-scale term in Φ_U, plus an identifiability mask in the gates

1. **Extend Φ_U to z:** add λ_Uz (h_z² ∂²_z u)², using the existing collocated d2z with its transpose.
   - The sawtooth has |∂²_z u| ~ 4δu/h_z², so the term gives it a stiffness of order λ_Uz.
   - Hydrostatic profiles are nearly linear in z (∂²_z u = ∂_z(−1/H), small), so the bias is O(h_z⁴/(H L_T)²).
   - Proposed default: λ_Uz = 1e-4. The final value is set by the 1 % rule on the C1/Bifrost column tests.
   - λ_U for x and y stays 0.
2. **Identifiability diagnostic** (new output /diagnostics/dP_unobs): δP_unobs = |B|σ_B/μ₀ at centres.
   - The fraction of each layer with δP_unobs > P is reported, so pressure in low-β layers is flagged rather than trusted. M5's classification uses it.
   - In the C3(b) u check, the distance to the analytic solution is measured on nodes with δP_unobs ≤ 0.1 P. This is the only change to the gate. The B decay criterion (≥ 10³, status `converged`) is unchanged.

## 3. Solver

- L-BFGS with the GN-diagonal preconditioner stays the production solver.
- `gncg!` stays available, but not as the default: it was slower than L-BFGS and no more effective on these modes.
- The Ã preconditioner diagonal (per-layer Hutchinson) automatically includes Φ_Q.
- The B-first continuation stays off for the staggered scheme; M6c found it counterproductive.

## 4. Tests and gates (additions to D9 §10)

| Test | Criterion |
|---|---|
| FD gradient, Φ_Q and λ_Uz terms alone and combined; ideal gas and table | ≤ 1e-6 |
| GN product including Φ_Q: vs Hessian on quadratic terms; symmetry | ≤ 1e-9; ≤ 1e-12 |
| Rank and thread invariance with all terms on | bitwise / ≤ 1e-12 |
| Φ_Q = 0 for uniform and potential B (C1b, C2 fields) | ≤ 1e-13 relative |
| 1 % rule: Φ_Q, Φ_Uz ≤ 1 % of Φ at the C1a/C1b/C2 solutions and on Bifrost columns | yes |
| C1a, C1b | unchanged criteria, status `converged` |
| **C3 (b)**: B checkerboard (ε = 1e-5 and 1e-3) | decay ≥ 10³ against the analytic solution, status `converged` |
| **C3 (b)**: u checkerboard | decay ≥ 10³ on identifiable nodes (§2.3) |
| Rayleigh quotient of the GN Hessian along the post-solve error | ≥ 1e-2 × that of a smooth Ã direction (now 5e-8) |

## 5. Plan (M6d) and risk

| Step | Content | Estimate |
|---|---|---|
| 1 | **Gate experiment first:** Φ_Q prototype (forward and adjoint only) on C3, at λ_Q ∈ {0.1, 1, 10}, with face data. Go/no-go before the rest. | 0.5 day |
| 2 | Φ_Q in GN seeds and preconditioner; λ_Uz; dP_unobs diagnostic; `data_on_faces` default on; config keys `mhs.lambda_Q`, `mhs.lambda_Uz`, `mhs.data_on_faces` | 0.5 day |
| 3 | Tests (§4), M6d report, then M5 | 0.5 day |

**Risk.** §1.1 predicts that Φ_Q makes the mode force-stiff. The Hessian diagonal then matches it, and L-BFGS should resolve it.
- **If step 1 shows a decay below 10³**, the remaining softness is in the data/force stiffness ratio itself, not in a null. The fallback is a two-level preconditioner:
  - the existing A₀ CG solve (data-only quadratic)
  - applied as a coarse correction in the force-null subspace each outer iteration
  - or a block-Jacobi on (v, Ã)

  This would be a further short design note.

## 6. Decisions requested

1. **Data term on faces** (§2.1) as the default for the staggered scheme. B_out stays HB.
2. **Add Φ_Q** (§2.2) with λ_Q = 1 by default, subject to the step-1 experiment.
3. **Add λ_Uz and the identifiability mask** (§2.3), including the narrowing of the C3(b) u criterion to identifiable nodes.
4. **Proceed with M6d step 1 (experiment) and report** before steps 2–3.
