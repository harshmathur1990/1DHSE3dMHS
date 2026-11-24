# 01 — EOS derivatives and the output layer (Phase 1)

Notation: X = ln T, Y = ln P = u, per node (k, j, i) of the slab, vertical index 1 = top. A bar is a cotangent
(f̄ = ∂ℓ/∂f for the caller's scalar ℓ).

## EOS

`state(tab, X, Y)` returns (ln ρ, ln nₑ, ln χ₅₀₀) from the bicubic Hermite cell (p, q) (`eos/Table.jl`). Add

```julia
state_d(tab, X, Y) -> (r, n, c, r_X, r_Y, n_X, n_Y, c_X, c_Y, flags)
```

with the X and Y derivatives of the same interpolant (`hbasis`/`hderiv` in both directions, divided by hX, hY). Exact for
the interpolant and continuous across cells (C¹ Hermite). Unit test: against central differences of `state` inside cells,
1e-9 relative.

Pointwise pull-back for every node:

    ρ = e^r, nₑ = e^n, χ = e^c, P = e^Y
    Ȳ += P̄ P + ρ̄ ρ r_Y + n̄ₑ nₑ n_Y + χ̄ χ c_Y
    X̄ += ρ̄ ρ r_X + n̄ₑ nₑ n_X + χ̄ χ c_X

The mass axis uses ρ where the τ axis uses χ (`q_chi(...; mass = true)`): the same code with c → r.

## τ on the height grid (`tau_column!`)

τ_1 = t₀/μ, τ_{k+1} = τ_k + ½ (χ_k + χ_{k+1})(z_k − z_{k+1})/μ, with t₀ from the frozen top branch:

- `:exp_tail`: t₀ = χ₁ H, H = (z₁ − z₂)/ln(χ₂/χ₁) → ∂t₀/∂χ₁ = H + H/ln(χ₂/χ₁), ∂t₀/∂χ₂ = −χ₁ H/(χ₂ ln(χ₂/χ₁));
- `:half_cell`: t₀ = ½ χ₁ (z₁ − z₂); `:zero`: t₀ = 0.

Reverse: accumulate τ̄ from the bottom (a running sum), distribute to χ_k, χ_{k+1}. z is the native axis here (no z̄ on
height modes). `logtau500` cotangents become τ̄ += l̄/(τ ln 10).

## Inclined τ on the height grid (`ViewTau.jl`)

χ is sheared to the footprints level by level (periodic bilinear interpolation of the global plane), integrated per
footprint column as above, and τ̃ is sheared back. Both shears are linear in their input; their adjoints are the transposed
bilinear scatters, which need the same per-level plane exchange (`allgather_plane`) in reverse: each rank scatters its
contributions into a global plane, the planes are summed (`Allreduce`) and each rank keeps its slab. One plane at a time,
as the forward.

## z(τ = 1) (`z_tau1`, `interp_lt`)

Linear in log τ between the bracketing levels (frozen bracket): differentiate the interpolation formula in z_k, z_{k+1},
τ_k, τ_{k+1}. Used by the gauge (τ and mass axes) and returned as `z_tau1` (its cotangent accepted for completeness).

## Gauge (τ and mass axes)

z ← z − s, s = mean over all columns of z(τ = 1). Reverse: s̄ = −Σ z̄ (global sum), then each column's z(τ = 1) receives
s̄/(NY NX) and passes it to its bracket (heights, and on the mass axis the log τ of the two levels).

## log τ on the mass axis (`logtau_mass`)

f_k = exp(c_k − r_k + ln m_k), τ₁ = f₁, τ_k = τ_{k−1} + ½ Δ ln m (f_{k−1} + f_k), output log10 τ. Reverse: as `tau_column!`,
then f̄_k → (c̄, r̄) at (X_k, Y_k) through `state_d`.

## Tests (Phase 1)

- `state_d` against central differences (random points in cells, at cell edges, LTE and NLTE tables).
- Each output adjoint against central differences of its forward function, dot-product form
  ⟨ȳ, J δx⟩ vs ⟨Jᵀ ȳ, δx⟩, 1e-8 relative; inclined τ at μ = 0.8 with azimuths 0, 30, 90°, on 1 and 3 ranks.
