# Adjoint of the embedded equilibrium (all six modes)

Branch `equilibrium-adjoint`, started 24 Nov 2026. **All phases implemented and tested (24 Nov 2026)**: results and
deviations from this plan in `HSEMHS/reports/adjoint_phase1-3.md` and `adjoint_phase4-6.md`; usage in the top-level README
("Adjoint of the embedded equilibrium").

## Why

FFNOInversion inverts T (and B) through HSEMHS. Its gradient needs the cotangent of the equilibrium outputs
(P, ρ, nₑ, χ₅₀₀, z, τ) pulled back to the inputs (ln T, B). Today it gets that by central finite differences: two full
`solve_embedded!` calls per control coordinate (`FFNOInversion.jl/src/ProductionGradients.jl`, `_force_regularization_vjp!`).
The first inversion (en024048 385, 6 + 4 nodes on 8 × 8 maps) pays 1280 global solves per gradient. A per-pixel inversion
of a 252² cube would pay ~10⁶. The goal is a vector–Jacobian product (VJP) whose cost does not depend on the number of
control parameters: a few solves' worth per gradient, for every mode.

## Scope

All modes of the embedded API (`src/Embedded.jl`, `EMBEDDED_MODES`):

| Mode | Forward map | Adjoint (this plan) |
| --- | --- | --- |
| `column_hse_z` | RK4 per column from an anchor, τ by trapezoid of χ | discrete adjoint, [02](02-column-modes.md) |
| `column_hse_ltau` | Heun per column from the τ-hydrostatic top closure, z by trapezoid, `align = "ptot"` shift, gauge | discrete adjoint, [02](02-column-modes.md) |
| `column_hse_logm` | P = μ g m, z by trapezoid of 1/ρ, τ from m, align, gauge | discrete adjoint, [02](02-column-modes.md) |
| `mhs_poisson_z` | column HSE start, Gauss–Newton on Φ(u) below z_cut, hydrostatic continuation above | implicit function theorem, [03](03-poisson-modes.md) |
| `mhs_poisson_ltau` | column HSE start, Gauss–Newton on Φ(u, c) below the τ cut, gauge | implicit function theorem, [03](03-poisson-modes.md) |
| `mhs_poisson_logm` | as `mhs_poisson_ltau` on the mass axis | implicit function theorem, [03](03-poisson-modes.md) |

Each with every `ViewOptions` (μ = 1; μ < 1 plane-parallel; μ < 1 inclined), `align = "ptot"` and `"none"`, zero and non-zero
B, LTE and NLTE tables (both are an `EOSTable`), 1 to N MPI ranks in x slabs.

Out of scope: the given-nₑ path (`solve_embedded!(...; lnne)`), which FFNOInversion uses only in forward runs (user decision,
3 Nov 2026: inversions always use the EOS); the standalone driver modes that are not embedded (MHSOptimize, MHSJointGN,
MHSStaggered, MHSAlternate). `adjoint_embedded!` refuses a workspace whose last solve had `lnne`.

## API (target)

```julia
res = solve_embedded!(ws, lnT, Bx, By, Bz; record = true)           # as now; record = true keeps what the adjoint needs
bar = EquilibriumCotangent(ws)                                        # zero cotangents on this rank's slab, (vertical, y, x)
bar.Pgas .= …; bar.rho .= …; bar.ne .= …; bar.chi500 .= …; bar.z .= …   # any subset; also tau500, logtau500
adj = adjoint_embedded!(ws, bar)                                      # collective
adj.lnT, adj.Bx, adj.By, adj.Bz                                       # cotangents of the inputs, same slab layout
adj.status                                                            # :exact, :stationary_approx, … (see below)
```

- Must follow a valid `solve_embedded!` on the same workspace with `record = true`; the record (the "tape") holds the
  frozen discrete choices and the solved state. A mismatch (other inputs, an invalid solve, a later solve) is refused.
- Collective, with the same two-stage error agreement as `solve_embedded!` (`agree`).
- Cotangents are with respect to ln T (the caller multiplies by 1/T) and the cell-centred B [T].
- Output cotangents not given are zero. `Pgas` and `lnP` may both be given; they are combined (P̄ P + lnP̄).
- `adj.status` reports how exact the result is: `:exact` (column modes: the exact derivative of the implemented map with
  its discrete choices frozen), `:stationary` (Poisson modes: implicit-function derivative at a state that met the
  stationarity tolerance of the adjoint), or `:not_stationary` (the returned state was far from stationary; the derivative
  is that of the stationary point the solver was approaching, and its error is bounded by the reported `gn_decrement`).
- A tangent (JVP) is not part of the API: validation compares against finite differences of `solve_embedded!` itself.

## Decisions

These are proposals for the user to confirm or change before the code is written.

- **A1. Frozen discrete choices.** Adaptive substep counts (RK4 doubling), EOS table cells, the exp-tail/half-cell
  branch of the τ top closure, the alignment level index, interpolation brackets (z(τ = 1), gauge) and order statistics
  (medians) are recorded in the forward solve and held fixed. Away from these switching points the map is smooth and
  this is its exact derivative; at a switch the finite difference itself is not defined. *Implemented (Phase 2) more
  strictly than first proposed:* the Heun step sizes are not frozen; the tape records which bound set each step (the
  interval end or the slope limit 0.05/|f₁|) and how often it was quartered, and the adjoint differentiates the step
  length through x and f₁, so the τ- and mass-axis adjoints agree with finite differences of the solver itself to ~1e-9
  (frozen lengths gave 6e-4). The implicit top closure is differentiated by the implicit function theorem (the forward's
  damped iteration stops at 1e-10).
- **A2. EOS derivatives.** Derivatives of ln ρ, ln nₑ, ln χ₅₀₀ in ln T and ln P are those of the bicubic Hermite
  interpolant (exact for the table, continuous across cells), not of the Wittmann model behind it. The Poisson solver's
  own Jacobian uses a ±1e-4 probe for ∂ ln χ/∂ ln P (`EPS_PROBE`); the adjoint uses the exact interpolant derivative. The
  solver's Jacobian only steers its iterations, so this does not change the solution.
- **A3. Poisson modes by the implicit function theorem** at the returned state, with the full Hessian of Φ (Gauss–Newton
  part plus the second-order part Σ W R ∇²R), not the Gauss–Newton approximation: with B fixed the force balance is a
  least-squares problem whose residual does not vanish at the solution (J × B is not a gradient), so Gauss–Newton alone
  is not the derivative (HSEMHS-integration plan §8). Second derivatives as central differences of the exact first
  derivative (Hessian–vector products), step chosen per vector; a Gauss–Newton-only option is kept for comparison.
- **A4. What the Poisson solve depends on.** The weights of Φ (layer medians of H and Δz in the start state) are part of
  the objective and depend on T through the start; their derivative (through the median column of each layer) is
  included. The warm start (`start = :previous`), the Levenberg–Marquardt damping, `ereg`, the anchor penalty weight and
  the preconditioner only steer the iteration and are not differentiated.
- **A5. Null and near-null directions.** On the τ and mass axes a uniform shift of all heights does not change Φ and is
  removed by the gauge, so the adjoint solve works in the complement of that direction. On the height grid a uniform
  shift of ln P changes Φ only through ionisation; the solver holds the anchor-cell mean of ln P (soft, per step). The
  adjoint treats that mean as an equality constraint (bordered system), with its value from the column-HSE start. The
  validation measures how well this matches the solver.
- **A6. Non-stationary returns.** An adjoint is computed for any valid solve; `adj.status` and the solve's
  `gn_decrement`, `grad_norm` say how far from stationary it was. FFNOInversion rejects non-converged trials, so it asks for
  gradients only at accepted states.

## Phases

Each phase ends with its tests passing on hyades (1, 2 and 3 ranks), a commit, and a short report
`reports/adjoint_phase<N>.md` (what was tested, numbers, deviations from this plan).

| Phase | Content | Doc |
| --- | --- | --- |
| 0 | this plan, agreed with the user | — |
| 1 | EOS derivatives; adjoints of the output layer: state (ρ, nₑ, χ), `tau_column!`, inclined τ (`ViewTau`), `logtau_mass`, `z_tau1`, gauge | [01](01-eos-and-outputs.md) |
| 2 | `column_hse_ltau`, `column_hse_logm`: tape of `hse_tau!`, top closure, `align_ptot!` (median included), gauge | [02](02-column-modes.md) |
| 3 | `column_hse_z`: tape of the RK4 integrator, the three anchors, P_ref at the anchor node, τ | [02](02-column-modes.md) |
| 4 | `mhs_poisson_z`: exact gradient of Φ, its transpose in (ln T, B), Hessian–vector products, bordered CG solve, continuation above z_cut, median weights | [03](03-poisson-modes.md) |
| 5 | `mhs_poisson_ltau`, `mhs_poisson_logm`: the same on (ln P, c), fixed part above the cut from the column adjoint, inclined and vertical-balance residuals, null direction | [03](03-poisson-modes.md) |
| 6 | `adjoint_embedded!` and `EquilibriumCotangent` in the API, README, MPI tests, cost benchmarks | [04](04-validation.md) |
| 7 | FFNOInversion: use the VJP in place of the finite-difference loop (its own plan in FFNOML, `FFNOInversion.jl/docs/equilibrium-adjoint/`) | — |

Phases 1–3 make the inversions that FFNOInversion runs today (column HSE on all three axes) scalable; 4–5 cover the MHS
modes, which FFNOInversion does not run yet (HSEMHS-integration Phase 4).

## Files (planned)

- `src/eos/Table.jl`: `state_d(tab, X, Y)` (values and both partial derivatives of the three fields).
- `src/Adjoint.jl` (new): `EquilibriumCotangent`, `adjoint_embedded!`, the tape types, dispatch per mode.
- `src/AdjointColumn.jl` (new): `hse_tau!` and RK4 tapes and their reverse sweeps; align, gauge, τ, z(τ = 1) adjoints.
- `src/AdjointPoisson.jl` (new): exact gradients of the two Poisson objectives, their input transposes, Hessian–vector
  products, the adjoint CG solves.
- `src/Embedded.jl`: `record` keyword, the tape kept on the workspace.
- `test/test_adjoint.jl` (+ `test/mpi/`): the checks of [04](04-validation.md).
