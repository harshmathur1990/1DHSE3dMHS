# 02 — Column modes (Phases 2 and 3)

Every column is independent except for the alignment reference (a global median) and the gauge (a global mean), so the
adjoint is a reverse sweep per column, threaded like the forward, plus two global reductions. Cost target: ≤ 3 × the
forward column solve.

## column-hse-ltau (`PoissonTau.hse_tau!`, `column_hse_tau`)

Forward, per column (axis ℓ = ln τ, μ, g):

1. **Top closure.** u₁ solves u₁ = ln(μ g τ₁ ρ(X₁, u₁)/χ(X₁, u₁)) (fixed-point iteration to 1e-10). Implicit function:
   F(u₁, X₁) = u₁ − ln(μ g τ₁) − r(X₁, u₁) + c(X₁, u₁) = 0 →
   ū-contribution: X̄₁ += ū₁ (r_X − c_X)/(1 − r_Y + c_Y).
2. **Heun substeps** inside each interval [ℓ_k, ℓ_{k+1}], X linear in ℓ:
   f(X, ℓ, u) = μ g e^ℓ ρ/(χ P), g_z(X, ℓ, u) = −μ e^ℓ/χ,
   u' = u + h/2 (f(X₀, ℓ₀, u) + f(X₁, ℓ₀ + h, u + h f(X₀, ℓ₀, u))), z' = z + h/2 (g_z(X₀, ℓ₀, u) + g_z(X₁, ℓ₀ + h, u + h f₀)).
   The tape records, per substep, (k, ℓ₀, h, the interpolation fractions s₀, s₁ of X). Reverse: standard adjoint of the
   two-stage scheme; each stage's partials from `state_d`: f_X = f (r_X − c_X), f_u = f (r_Y − c_Y − 1),
   ∂g_z/∂X = −g_z c_X, ∂g_z/∂u = −g_z c_Y; X̄ of a substep goes to X_k, X_{k+1} with weights (1 − s), s.
3. **z(τ = 1) = 0 per column** (`z .-= interp_lt(z, lt, 0)`): a per-column shift, adjoint as in 01.
4. **align = "ptot"** (`align_ptot!`): ref = global median over columns of ln(P_nt + B²_nt/2μ₀) (bottom level); each column
   is shifted by z_r, found by linear interpolation of z in ln P_tot at ref, or by hydrostatic extrapolation below the
   column. Frozen: the median column, the bracket level, which branch. Derivatives: z_r in (z, ln P_tot of the two
   bracket levels, ref), ln P_tot in (u, B); ref in (u_nt, B_nt) of the median column only (one global reduction to
   collect ref̄ on its owner). `align = "none"` skips it.
5. **Gauge**: as 01.
6. **Outputs**: ρ, nₑ, χ at (X, u) (01); log τ is the axis (no cotangent path).

Failed columns (EOS envelope) make the solve invalid; no adjoint is asked for.

## column-hse-logm

`hse_tau!(...; mass = true)`: u_k = ln(μ g m_k) exactly, so ∂u/∂X = 0. Heights: g_z = −μ m/ρ (the same Heun tape, with ρ in
place of χ). Gauge on log τ from `logtau_mass` (01), so the gauge's bracket carries X̄ and ū through τ as well. Alignment as
above. Outputs ρ, nₑ, χ pointwise; `logtau500` from `logtau_mass`.

## column-hse-z (`ColumnHSE.hse_column!`, `column_state!`)

1. **RK4** with n substeps per interval (n frozen: with `adapt = true` the returned value is the 2n-substep solution of the
   last doubling), X linear in z on each interval: u' = u + h/6 (k₁ + 2k₂ + 2k₃ + k₄), k = −g e^{r(X, u) − u}. Reverse:
   the RK4 adjoint (four stages, partials k_X = k r_X, k_u = k (r_Y − 1)). The co-integrated column mass m is not an
   embedded output and is not differentiated.
2. **Anchors**: `:bottom` (u_NZ = ln P_ref, integrate up), `:top` (u₁ = ln P_ref, down), `:height` (both directions from
   z_ref, the first partial interval from z_ref). P_ref is a constant: no cotangent on it. The anchor node returns
   P = P_ref exactly (`column_state!`), so its P̄ has no path back.
3. **Outputs**: ρ, nₑ, χ pointwise; τ and log τ (01), plain or inclined; z is the axis.

## Threading and MPI

Reverse sweeps are threaded over the same column chunks as the forward. Global pieces: the alignment median (owner of
the median column receives ref̄ by `Allreduce`), the gauge mean (`Allreduce`), the inclined-τ plane scatters (01).
Results are independent of the thread count; across rank counts they differ only by the order of global sums.

## Tape size

Heun: a few substeps per level (the fraction s and h per substep, ~16 bytes each): ≲ 1 kB per column for 100 levels,
~60 MB for 252². RK4: n per interval, one Int16 per interval. Stored on the workspace, replaced by every recorded solve.
