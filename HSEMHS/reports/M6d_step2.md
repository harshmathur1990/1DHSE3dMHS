# M6d step 2: Φ_Q in production, collapse safeguard, λ_Uz, dP_unobs (17 Oct 2026)

**Follows** `reports/M6d_step1.md`. The user chose option (a) (pressure-collapse safeguard in L-BFGS), λ_Q = 1, and step 2.

**Code:**
- `src/MHSStaggered.jl`: `phiQ_seeds!`, λ_Uz, `dP_unobs!`, `unidentifiable_fraction`, the A₀ fix and new defaults.
- `src/MHSGaussNewton.jl`: Φ_Q and λ_Uz seeds in `gn_hvp!`.
- `src/MHSOptimize.jl`: `collapse_du`, `gauge_shift`, `project_lower!` and `step_to_lower`.
- `src/Config.jl`: the new keys.

**Tests:** `test/test_stag_gradient.jl`, `test/test_gateC_staggered.jl`, `test/test_units_config.jl`, `test/scripts/sgrad_ranks.jl`.

## 1. What changed

| Item | Change |
|---|---|
| Φ_Q | Forward, adjoint (step 1), and now **GN seeds**: Φ_Q is linear in J, so `gn_hvp!` applies the same seeds to δJ (`phiQ_seeds!`, shared with `objective!`). The per-layer Hutchinson Ã preconditioner includes Φ_Q automatically. |
| **A₀ (`init_A!`)** | **Now excludes Φ_Q and λ_Uz** (and λ_F, λ_U as before): a data + λ_S quadratic, as D9b §5 describes. With Φ_Q included, its (B_ref/S)² weights made the CG system so ill-conditioned that C1b (8×8×65) started 3σ off in B. The plain solve then collapsed (u → −2000 without the safeguard) or stalled at the bound with it. |
| **L-BFGS collapse safeguard** | `LBFGSOptions.collapse_du = 3.0` (Inf turns it off). It enforces u ≥ u_start − collapse_du **on u = ln P, not on the gauge variable v**. A first version bounded v: L-BFGS then raised the anchor-plane mean of v by about 47, which lowers u by the same amount everywhere above it, so u still collapsed. **Each iteration:** remove the gauge component of d[1] (changes neither Φ nor ⟨d, ∇Φ⟩); zero d on nodes at the bound; cap α at the bound. If the projected direction is no longer a descent direction, fall back to projected preconditioned steepest descent and drop the memory; if that also fails, return `stalled` ("no descent above the u bound"). `gauge_shift(pb, f) = anchor_mean(pb, f)` is the hook (both problem types have it). |
| λ_Uz | λ_Uz (h_z² ∂²_z u)² added to Φ_U, through the objective, adjoint, GN seeds and exact squared-stencil diagonal. **Default 0**; see §3. |
| dP_unobs | `dP_unobs!(out, pb, w)` gives δP_unobs = \|HB\| σ_B/μ₀ at centres, with σ_B = √(σx²+σy²+σz²) as in screening (E10). `unidentifiable_fraction` gives the per-layer fraction with δP_unobs > P. Meant for `/diagnostics/dP_unobs`; the `mhs` run-mode writer (M7) does not exist yet. |
| Defaults | `SMHSProblem` and `sc1_problem`: `data_on_faces = true`, `lambda_Q = 1.0`, `lambda_Uz = 0.0`. Config `[mhs]`: `lambda_Q = 1.0`, `lambda_Uz = 0.0`, `data_on_faces = true`, `collapse_du = 3.0`. |

## 2. Verification

| Check | Result | Criterion |
|---|---|---|
| FD, Φ_Q alone (all fields; Ã), ideal gas / table | 5e-15, 7e-15 / 2e-14, 3e-14 | ≤ 1e-6 |
| FD, λ_Uz alone, ideal gas / table | 7.9e-13 / 8.7e-13 | ≤ 1e-6 |
| FD, all terms + Φ_Q + λ_Uz (all fields; Ã), ideal gas / table | 8e-12, 2.4e-11 / 3.2e-10, 2.9e-11 | ≤ 1e-6 |
| GN = Hessian on the quadratic terms, with Φ_Q and λ_Uz | 3.8e-12 | ≤ 1e-9 |
| GN symmetry and positivity, linearized force seeds, production terms | pass | ≤ 1e-12 |
| Rank/thread invariance, production terms (step 1 scratch; now in `sgrad_ranks.jl`) | bitwise across threads; ≤ 3e-15 across ranks | bitwise / ≤ 1e-12 |
| Φ_Q = 0 for uniform B | Φ_Q/Φ ≤ 3e-3 at the C1b solution, with Φ ≈ 1e-18 (round-off level) | — |

## 3. Gates (D9b defaults)

| Gate | Result | Pass |
|---|---|---|
| C1a (NZ 33 / 65 / 129) | `converged`; at NZ = 33 and 65, \|u − u_s\| = 3.1e-8 / 1.1e-8 and RMS = 4.5e-9 / 2.1e-9 (from the scratch runs) | yes |
| C1b (8×8×65) | `converged` in 3245 iterations at gtol 1e-7; RMS = p99 = 1.6e-7; \|u − u_s\| = 1.8e-7; δB = 9e-8 σ | yes |
| C2 | unchanged (operators untouched); interior order still `@test_broken` | as before |
| **C3(b) u checkerboard**, identifiable nodes | δu 1e-3 → 4.1e-11 (÷2.5e7) | **yes** (was `@test_broken`) |
| **C3(b) Ã checkerboard ε = 1e-5** | δB 6.0e-8 → 2.7e-11 T (÷2.2e3), `converged`, 531 iterations | **yes** (was `@test_broken`) |
| **C3(b) v + Ã checkerboard ε = 1e-3**, B-first | δB 6.0e-6 → 1.4e-12 T (÷4.2e6); Φ = 2.3e-18 vs reference 1.2e-19; ends `stalled` at the round-off floor | **yes, with the criterion below** |

The Rayleigh-quotient criterion of D9b §4 is moot where the decay passes: the remaining error is at round-off.

**Identifiability.** At the C3 reference only 27 % of the nodes have δP_unobs ≤ 0.1 P, and the top layers are 100 % unidentifiable. This is consistent with D9b §1.2 (limit ≈ 800 P_top).

**Test changes:**
- C1 solves use gtol 1e-7. At 1e-6, C1b converges with RMS = 1.6e-6, because Φ_Q stiffens the problem.
- The C3 rows are no longer `@test_broken`.

## 4. Findings that deviate from D9b (decisions requested)

1. **λ_Uz default 0 instead of 1e-4.**
   - With λ_Uz = 1e-4, C1a fails its gate: \|u − u_s\| = 2e-5 and RMS = 6.8e-6 at NZ = 33. The tanh T transition gives the hydrostatic u a real ∂²_z u, so the bias is not the O(h_z⁴/(H L_T)²) the addendum estimated.
   - λ_Uz = 1e-6 passes (2.2e-7) but is too small to stiffen anything.
   - With face data and Φ_Q, the top additive-pressure mode λ_Uz was meant for **does not occur**: every C3 reference converges to \|u − u_s\| ≤ 3e-9.
   - So the term stays implemented and tested, but off. A Bifrost column test may revisit this.
2. **The safeguard alone does not fix the ε = 1e-3 row. B-first continuation does.**
   - With a plain joint solve, the initial Φ is dominated by Φ_Q (4.5e5 vs Φ_F = 2.9). In the first steps L-BFGS moves Ã a lot (δB up to 1.6e-3 T) and drags u by 1–2 at every depth.
   - With the safeguard, the solve stalls at the u bound after about 45 iterations (B ÷0.007). Without it, u collapses (δu = 28–50).
   - Relaxing Ã first with v frozen (`lbfgs_continuation!`, 2000 iterations), then solving jointly, gives ÷4e6–2.7e7 with δu ≤ 2e-9.
   - **Production note:** on C1b, B-first now reaches the accuracy criteria (RMS 2e-7) but ends at `max_iter`. M6c's "B-first is counterproductive" was at least partly the A₀ ill-conditioning above.
   - So the C1 tests use a plain solve and the large C3 checkerboard uses B-first. **A solver policy for real data is open.** Options: always B-first; or plain with automatic B-first when Φ_Q/Φ_F at the start exceeds a threshold.
3. **C3(b) ε = 1e-3 status.**
   - The B-first solve ends at the round-off floor, where the line search fails twice (`stalled`). Tightening gtol to 1e-13 does not change this.
   - The test therefore checks decay ≥ 10³ and Φ ≤ 10·max(Φ_ref, 1e-16), instead of `status == converged`.
   - The ε = 1e-5 row and C1a/C1b still require `converged`.

## 4a. Resolution (17 Oct 2026, user)

All three deviations are accepted (spec §10). The solver policy is now implemented:
- `mhs_solve!(pb, w, x; opts, bfirst_ratio = 10, b_first_iter = 2000)` runs plain L-BFGS, unless the starting Φ_Q/Φ_F exceeds `bfirst_ratio`; then it runs B-first continuation. Config keys: `mhs.bfirst_ratio`, `mhs.b_first_iter`.
- The ratio does not depend on the amplitude in the linear regime: C1b starts give 1.3e-3 – 0.59, and every C3 Ã checkerboard gives 1.5e5. So the threshold separates "start dominated by M-invisible currents" from ordinary starts, but not small from large perturbations.
- Gate C under the policy:
  - The C1 starts and the u checkerboard take the plain path.
  - Both Ã rows take B-first: ε = 1e-5 gives ÷2.3e5 (plain gave ÷2.2e3); ε = 1e-3 gives ÷4.2e6.
  - Both rows end at the round-off floor, so the floor criterion of §4.3 applies to both B rows.

## 5. Remaining for step 3

- The 1 % rule for Φ_Q on C2 (needs a C2 solve) and on Bifrost columns.
- A solver-policy decision (§4.2).
- Committing; then M5.
