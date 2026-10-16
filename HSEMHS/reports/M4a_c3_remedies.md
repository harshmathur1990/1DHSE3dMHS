# Gate C3: cheap remedies on the collocated scheme (16 Oct 2026)

**Setup.** C1b equilibrium (16×16×33, ideal gas, uniform B₀ = (20, 0, 50) G, synthetic σ). We add ε = 1e-3 times the (−1)^{i+j+k} checkerboard to u and to each b̃ component, then re-solve. "Mode" is the remaining checkerboard amplitude divided by the initial amplitude; C3(b) requires ≤ 1e-3. Scripts are reproduced in this file's tables; the code is `lbfgs_continuation!` and `LBFGSOptions.frozen` in `src/MHSOptimize.jl`.

## 1. λ_U sweep, with and without B-first continuation

Continuation means: relax b̃ with v frozen (≤ 500 iterations), then solve jointly.

| λ_U | Continuation | u mode | b̃x mode | Φ_final / Φ_eq | (Φ_S+Φ_U)/Φ at the end |
|---|---|---|---|---|---|
| 1e-6 (default) | no | **6.3e2** (grows) | 2.5e-3 | 720 (trap) | 2e-3 |
| 1e-6 | yes | **6.9e-4** | 2.4e-3 | 1.0015 | 2e-4 |
| 1e-4 | no / yes | 6.3e2 / 4.2e-4 | 2.6e-3 / 2.4e-3 | 860 / 1.0015 | 0.17 / 2e-4 |
| 1e-2 | no / yes | 1.1 / 1.2e-4 | 2.4e-3 / 2.4e-3 | 720 / 1.0015 | 5e-4 / 2e-4 |
| 1 | no / yes | 4.0e-2 / 2.4e-5 | 2.8e-3 / 2.4e-3 | 717 / 1.0015 | 1e-5 / 2e-4 |

- **B-first continuation removes the collapsed-pressure trap** at every λ_U, and the pressure checkerboard then decays by 1.5×10³ (default λ_U) to 4×10⁴.
- **Raising λ_U is not needed** for this, and on its own (without continuation) it does not avoid the trap.
- C1 is horizontally uniform, so it cannot test the 1 % rule for Φ_U; at λ_U = 1e-6 it holds trivially. The rule must be re-checked on a structured case (C5/C6, M5).

## 2. The b̃ checkerboard is a null mode of the collocated curl

| Test | b̃x mode | b̃y mode | b̃z mode |
|---|---|---|---|
| ε = 1e-3, gtol 1e-6 | 2.4e-3 | 2.4e-3 | 0.31 |
| ε = 1e-2 (scales with ε, so not a tolerance floor) | 2.4e-3 | 2.4e-3 | 0.30 |
| gtol 1e-9 | 2.4e-3 | 2.4e-3 | 0.30 |

The horizontal odd–even part of δB_z gives D_x δB_z = D_y δB_z = 0 with centred differences: no current, no Lorentz force, and ∇·B only through the small z non-uniformity. Only the data term (curvature 2/N_B ≈ 8e-5) and the δB regularizer restore it. The stiff B modes have curvature ~6e6, so the conditioning is ~1e11, which a diagonal preconditioner cannot remove. Here is the D9 concern made concrete.

## 3. λ_S sweep (with continuation)

| λ_S | b̃z mode | Φ_final / Φ_eq | regularizer share at the end |
|---|---|---|---|
| 1e-3 (default) | 0.31 | 1.0014 | 1.8e-4 |
| 1e-1 | 0.20 | 1.013 | **1.2e-2** |
| 1 | 0.13 | 1.07 | **6.4e-2** |
| 10 | 0.063 | 1.27 | **0.21** |

Raising λ_S helps very little, breaks the 1 % rule from λ_S = 0.1, and degrades the solution. **Rejected.**

## 4. C1a / C1b with continuation

The accuracy is unchanged (C1b NZ = 33/65/129: RMS/floor 0.92/0.57/0.33; \|Δu\| orders 1.76, 1.93). But **C1b now ends `stalled`** after 2–4.5 thousand iterations instead of `converged`, because the slow B checkerboard keeps the gradient from falling.

## Conclusion

- **Keep:** B-first continuation. It fixes the pressure trap, and it will be the default solve path in M5 (cheap when starting from b̃ = 0).
- **Not useful:** raising λ_U or λ_S.
- **Still failing:** C3(b) for B. The horizontal odd–even B modes are null modes of the collocated curl and are restored only by the data and regularizer terms, at curvatures 10¹¹ below the stiff modes. This is the failure D9 anticipates. It also shows up as `stalled` instead of `converged` on C1b, which would deny `success_mhs` on real data.

**Recommendation:** invoke D9 and design M6. A vector potential on a staggered (Yee-type) grid makes ∇·B = 0 exact and removes the curl null space. Write a design addendum first.
