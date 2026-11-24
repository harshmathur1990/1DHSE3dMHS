# 03 — Poisson modes (Phases 4 and 5)

## Principle

Below the cut the solver returns x* ≈ argmin_x Φ(x; a), a = (ln T, B) and the start-dependent pieces s(a) (the fixed part
above the cut, the weights). At a stationary point

    G(x*, a) = ∇ₓΦ(x*, a) = 0,

so dx*/da = −H⁻¹ ∂G/∂a with H = ∂G/∂x (symmetric). For a cotangent x̄ of the solved unknowns:

    solve  H λ = x̄            (CG; H applied matrix-free)
    ā     += −(∂G/∂a)ᵀ λ       (one "mixed" product)

and the full outputs (ρ, nₑ, χ, τ, z over every level) pull back to x̄ and to direct ā terms through the output layer (01).

**Products.** With Φ = Rᵀ W R (W diagonal, frozen),

- G = 2 Jᵀ W R (J = ∂R/∂x, exact: A2);
- H v = 2 (Jᵀ W J v + Σ_i (W R)_i ∇²R_i v), computed as a central difference of G along v:
  H v ≈ [G(x + εv) − G(x − εv)]/(2ε), ε = 1e-6 ‖x‖/‖v‖ (Gauss–Newton option: 2 Jᵀ W J v only);
- (∂G/∂a)ᵀ λ = ∇ₐ ⟨λ, G⟩ and ⟨λ, G⟩ = dΦ/dε along λ, so
  (∂G/∂a)ᵀ λ ≈ [∇ₐΦ(x + ελ) − ∇ₐΦ(x − ελ)]/(2ε), with ∇ₐΦ = 2 (∂R/∂a)ᵀ W R.

So the new code needs, per Poisson problem: R and the exact Jᵀ(·) (from the existing evaluate/JTop with exact EOS partials),
and the transpose of ∂R/∂a (new: through the EOS pointwise and through the Lorentz force). Second derivatives never appear
explicitly. Cost: one CG solve of H (each iteration two G evaluations, about two residual evaluations and two Jᵀ products),
plus two ∇ₐΦ evaluations. Preconditioner: the solver's own (Fourier, block or hybrid) at x*.

## mhs-poisson-z (`PoissonZ.solve_pressure`)

- Unknowns x = u = ln P on the levels k ≥ k_c (z ≤ z_cut).
- R: R_x = L_x e^{−u} − D_x u, R_y likewise (centres), R_z,f = −(u_f − u_{f+1})/h_f − g (q_f + q_{f+1})/2 + (L_z,f e^{−u_f} + L_z,f+1 e^{−u_{f+1}})/2
  (faces), q = ρ/P. W: wc = w H̄², wf = h H̄_f² with H̄ the layer medians of P/(ρ g) in the start.
- ∂R/∂a: through q(X, u) (pointwise, `state_d`) and through L = J × B (`lorentz`, bilinear in B, derivative stencils
  sddx/sddy periodic and ddz on the height grid). Its transpose: the transposed stencils (sddx, sddy are antisymmetric; ddz
  has its own transpose at the ends) and the cross products.
- **Anchor (A5).** The solver damps changes of the anchor-cell mean of u (default: the bottom plane) per step; the adjoint
  solves the bordered system [H c; cᵀ 0][λ; μ] = [x̄; 0] with c the anchor indicator / count, i.e. the derivative with
  that mean held at its start value. The start value is ln P_ref for the default bottom anchor (constant); for column
  anchors `:top`/`:height` it is the column-HSE bottom pressure and carries a cotangent into the column adjoint (02).
- **Weights (A4).** H̄_l = median over the layer of 1/(g q) at the start state: λ-weighted ∂G/∂w_l is a scalar per layer
  (from the residuals already formed), passed to the median column of that layer → its (X, u_start) → column-z adjoint.
- **Continuation above z_cut.** For k = k_c − 1 … 1: u_k solves u_k − u_{k+1} + Δz g (q_k + q_{k+1})/2 = 0 (Newton to
  1e-12). Implicit function per level, a reverse sweep from the top down to k_c, giving ū_{k_c} and X̄.
- **Outputs.** P, ρ, nₑ, χ, τ, log τ through 01 at every level (`height_state!`).

## mhs-poisson-ltau and -logm (`PoissonTau.solve_tau`)

- Unknowns x = (c, ln P at k > k_c) per column (slot 1 = c, the height of the cut level; ln P at k_c is fixed).
- Fixed part s(a): ln P on k ≤ k_c and the relative heights zrel above the cut, from the column HSE of the start
  (`column_hse_tau`, 02). Its cotangent (from R's dependence on them and from the outputs above the cut) is pulled back
  by the column adjoint. A warm start does not change the stationary point and is not differentiated.
- R: the `evaluate!` residuals — R_x, R_y on the nodes (−∂_x u − W ∂_x z + L_x e^{−u}, W = g q − L_n e^{−u}), R_z per
  interval (Δu − Δℓ/2 (W a + W a)), the vertical balance R_v = R_ray − Δz t·R̄_h when inclined (D0); heights from the
  pressures by the trapezoid z_{k+1} = z_k − Δℓ/2 (a_k + a_{k+1}), a = μ τ/χ (mass axis: μ m/ρ); L from B on the τ
  surfaces at their heights (`lorentz_tau!`, curvilinear derivatives that depend on z).
- ∂R/∂a: through q, a (EOS at (X, u), pointwise), through L (bilinear in B; the stencils use the heights, which do not
  depend on a at fixed x except via a(X, u) in the trapezoid — included), and through the fixed part s.
- J exact: the solver's `aχ` (probe ±1e-4) replaced by the Hermite derivative in an "exact" evaluation used only by the
  adjoint (A2); everything else in `Jop!`/`JTop!` is already analytic.
- **Null direction (A5).** A uniform shift of all c leaves R unchanged (only differences and gradients of z enter). The
  output z is gauged, so x̄ has no component along it after the gauge adjoint; CG runs on the complement (the shift
  component is projected out of the right-hand side and of every iterate). Check: |⟨1_c, x̄⟩| ≤ 1e-12 ‖x̄‖.
- **Weights (A4).** wc, wf from the layer medians of H and Δz at the start (`tp_weights!`): per-layer scalars as above,
  to the median columns' start state → column adjoint.
- **Gauge** after the solve (01); mass axis through `logtau_mass`.

## Stationarity and accuracy

The implicit-function derivative is that of the exact stationary point. At the returned x* the error is first order in
‖G(x*)‖. The solver's own stop (`gn_tol` = 1e-3 relative predicted decrease) can be loose for this; options:

- report it (`adj.status`, `gn_decrement`, `grad_norm`), as A6;
- optionally polish: a few extra Newton steps on G before the adjoint (`polish = n`), only inside the adjoint, the returned
  fields unchanged — to be decided after Phase 4 measures how much it matters.

The CG tolerance of the adjoint solve (default 1e-8 relative) and the ε of the products are options of `adjoint_embedded!`.
