# M5f: Codex's M5e review — the three deliverables (19 Oct 2026)

**Follows** `CODEX_REVIEW_M5e_2026-10-18.md`. The user asked for its suggestions to be implemented. From this round on, runs execute on the UiO cluster (eagle8/9, hyades7–16) in the repository copy `/mn/stornext/d9/data/harshm/MHS`; outputs are in `HSEMHS/out/m5f/` there (ignored by git).

**Code:**
- `src/MHSStaggered.jl`: `set_face_hinge!(pb, s; lambda, kind)`. This is an opt-in face-bound term in `objective!`, the gradient and the GN product, folded into term 2:
  - `kind = :hinge`: (λ/N_B) Σ max(|c| − s, 0)².
  - `kind = :barrier`: (λ/N_B) Σ −log(1 − (c/u)²), with its exact convex curvature in the GN product.
- `src/MHSJointGN.jl`, `joint_gn_solve!(…; face_hard, bound_tau)`:
  - fraction-to-the-boundary scaling of every candidate. B is linear in (Ã, b̄), so it stays exactly divergence-free and nothing is clipped;
  - after a rejected truncated candidate, the radius shrinks from Δ, not from the truncated length;
  - truncation counts are reported.
- Defaults are unchanged: no hinge, no bounds.

**Scripts:** `scripts/m5_bifrost/`
- `data_qp.py`, `qp_to_h5.py`, `make_qp_state.jl`, `verify_field.py`;
- `centre_bracket.py`, `m5f_pilot.jl`;
- `data_lp.py --disp`;
- data paths in `bif_setup.jl` and `long_comparison.jl` are now relative to the repository.

## 1. Gentle face-admissible field (deliverable 2)

**Problem.** Constrained least squares on the proven support S (|S| = 3328, from the p = 128 witness):

- minimize mean(c²) = Φ_B, the solver's face data term;
- subject to div B = 0, |c| ≤ 5 − m on S and |c| ≤ 3 − m off S.

**Solver.** ADMM with an exact divergence projection (M = A Aᵀ factorized once) and over-relaxation 1.6. It stops at primal and dual residuals ≤ 1e-5, after about 1100 iterations (about 3 min). The λ_S term was not put in the objective; it is evaluated afterwards through the solver.

**Verification.** Every field is checked from the physical field (`verify_field.py`) and rebuilt in the solver representation by explicit construction in the gauge Ãz = 0 (`make_qp_state.jl`). Codex's numbers for the raw LP witness are reproduced exactly.

| Field | mean c² (face) | Φ_S | face max | face > 3σ | centre max | centre > 3σ | centre mean c² | raw force (v of the ×100 endpoint) | reconstruction error |
|---|---|---|---|---|---|---|---|---|---|
| init_A! (unconstrained Φ_B + Φ_S) | 0.3170 | 0.0411 | 11.17σ | 0.613 % (341 > 5σ) | 21.12σ | 2.25 % | 0.858 | 6.06e5 | — |
| LP vertex (M5e witness) | **7.4995** | 0.2687 | 4.999σ | 0.615 % | 22.98σ | 17.6 % | 5.47 | **1.85e11** | 7e-13 σ |
| **QP, m = 1e-3** | **0.3235** | 0.0414 | 4.999σ | 0.559 % | 22.41σ | 2.26 % | 0.866 | 6.03e5 | 5e-13 σ |
| QP, m = 0.05 (pilot start) | 0.3241 | 0.0414 | 4.950σ | 0.566 % | 22.46σ | 2.25 % | 0.865 | 6.03e5 | 5e-13 σ |

Divergence of every field is ≤ 1e-21 T/m (relative ≤ 8e-13).

- **The admissible set costs only 2 % in mean c².** 0.3235 against 0.3165 for the unconstrained minimum-norm divergence-free fit, and the smoothness term is unchanged.
- The LP vertex is 23 times worse in the data term, and its raw force is 3e5 times larger. Codex's warning against using it as an initializer is confirmed.
- The constrained fit does not repair the centre model: centre maximum ≈ 22.4σ, 2.26 % > 3σ. **The two observation operators still disagree at about 22σ at the same field.**
- At the optimum 910 faces lie within 1e-4 of their bound. The constraints are genuinely active.

## 2. Centre-model maximum feasibility (deliverable 1)

**HiGHS IPX, bounded runs at t = 4.999** (55-min external wall, progress logged; eagle9, single-threaded):

| Formulation | Rows × cols, nnz | Outcome | Profile |
|---|---|---|---|
| original (face variables, two-sided centre rows) | — | **stopped by the wall, no verdict** (status: none) | 5 IPM iterations in 12 s, then 55 min in IPX "Constructing starting basis": free variables 4813 → 0, then fixed variables ~39 000 → ~38 000 |
| Codex's reduced (centre corrections, equality only) | 162 708 × 470 988, 4.04e6 | **solver error after 307 s** (HiGHS status 4, IPX internal error `vector::_M_default_append` in basis construction) | 4 IPM iterations in 5 s, then crash |

The bottleneck in both is IPX's basis construction, not its interior-point iterations. Neither run says anything about feasibility.

**Independent backend with a certified bracket (`centre_bracket.py`).**

*Method.* PDHG (Chambolle–Pock, diagonal preconditioning, restart to the average every 1000 iterations) on min ‖c‖∞ subject to A c = r in Codex's reduced formulation.

*Bracket.* At every report, t* ∈ [LB, UB]:
- LB = λᵀr / ‖Aᵀλ‖₁, valid for any λ by weak duality;
- UB = the true centre maximum of the field obtained by projecting c exactly onto A c = r (CG on A Aᵀ) and reconstructing B. The reconstruction is exactly divergence-free.

*Run.* hyades15, 50-min wall, 2000 iterations:
- UB: 5.754 → 4.643 (200 iterations) → 4.230 → **3.9456** at the end;
- LB: did not progress (−0.85). The dual iterate is uninformative at this iteration count.

**Result: a verified centre-model maximum witness at 3.946σ.** It is checked independently three ways:
- `verify_field.py`: max |div B| 9.7e-22 T/m (relative 8e-13), centre max 3.945586σ, 0 centre values > 5σ;
- the solver representation (`make_qp_state.jl`): reconstruction error 4e-13σ;
- the bracket's own final re-verification.

It is a numerically verified witness, not an exact-arithmetic certificate. This **closes the centre maximum question positively, with a margin of over 1σ** (the old witness was 6.55σ).

**The tail is NOT met by this witness:** 6623 centre values (1.41 %) exceed 3σ, against K = 4709. The PDHG objective does not control the tail. Centre-tail feasibility is unresolved. The analogue of the face fixed-support LP/QP (a support of ≤ K centres from this witness, |c| ≤ 3 − m elsewhere) is the natural next test.

This field's face residual is 82.9σ, with 12 % of faces above 3σ: the centre witness uses the horizontal Nyquist components that H averages away. **The face and centre models disagree in both directions.** The face-admissible fits have centre residuals near 22σ, and the centre witness has face residuals near 83σ.

## 3. Short data-feasible MHS pilot (deliverable 3)

**Settings.** Joint GN ×100, no anchor-tangent, the fixed reference. Start: v from the M5e ×100 endpoint (checkpoint start), B from the QP m = 0.05 field. Hard bounds |c| ≤ 5 − 1e-3 on S and ≤ 3 − 1e-3 off S, so the tail is ≤ |S| ≤ K automatically.

**Logging.** Every 500 products (2500 in the long run), with a checkpoint at each log point. Logged quantities:
- objective components and raw force;
- layer RMS and p99 with the worst location;
- face and centre maxima and tails;
- pressure optimality, anchor and bound errors, accepted steps and work;
- windowed max |ΔP/P| and |ΔB|.

**3000-product pilots** (wall 20–180 s each):

| Run | Φ | raw force | max layer RMS | face max | face > 3σ | outcome |
|---|---|---|---|---|---|---|
| free (option (a): no bounds) | 0.5626 | 2.02e3 | 213 | **10.69σ** (330 > 5σ) | 0.615 % | budget |
| hard + hinge λ 100 (band 0.05) | 20.62 | 2.03e5 | 2620 | 4.999σ | 0.564 % | **stalled at 299 products**: 6/7 candidates truncated, median fraction 2e-10 |
| hard + hinge λ 1000 | 22.10 | 2.17e5 | 2738 | 4.967σ | 0.565 % | stalled at 338 products, same |
| hard + barrier λ 0.001 / 0.01 | 19.23 / 16.04 | 1.9e5 / 1.6e5 | 2562 / 2316 | 4.999σ | 0.563 % | stalled at 620 / 1146 products |
| hard + barrier λ 0.1, τ 0.999 | 5.113 | 4.74e4 | 1236 | 4.999σ | 0.564 % | budget (later stalled at 4408 in the long run: fractions 1e-139) |

- **Option (a) leaves the admissible set immediately**, as Codex expected: face max back to 10.7σ.
- With hard bounds, the hinge cannot keep faces off the bound: its curvature is finite. Fraction-to-boundary steps then shrink a limiting face's slack by 1000× per step, until the candidates are truncated to nothing.
- The barrier helps once λ is large enough (0.1).

**Long run: one fixed-settings baseline, 30000 products.** Barrier λ = 0.1. τ was the parameter of the bound handling being fixed here, so three values were run in parallel; τ = 0.99 is the baseline.

| τ | final work | Φ | raw force | max layer RMS (p99) | face max | > 3σ | outcome |
|---|---|---|---|---|---|---|---|
| 0.999 | 4408 | 4.158 | 3.79e4 | 1090 (5390) | 4.999σ | 0.564 % | trust radius collapsed (jammed) |
| 0.9 | 5034 | 1.268 | 8.98e3 | 472 (2014) | 4.999σ | 0.564 % | trust radius collapsed |
| **0.99** | **30050** | **0.4322** | **604** | **109 (410)** | **4.998σ** | **0.565 %** | **work budget; 300/309 accepted; 49 truncated (median fraction 0.16); 1348 s** |

Trajectory of the τ = 0.99 run:

| work | 2509 | 5086 | 7526 | 10102 | 12518 | 15102 | 17550 | 20008 | 22554 | 27526 | 30050 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Φ | 2.851 | 2.056 | 1.161 | 0.729 | 0.551 | 0.499 | 0.466 | 0.449 | 0.437 | 0.434 | 0.432 |
| max layer RMS | 834 | 670 | 441 | 280 | 199 | 163 | 139 | 123 | 114 | 111 | 109 |

At the end:
- face max 4.99835σ (margin 6.5e-4 to the hard bound), 2673 faces > 3σ (0.565 %), 0 > 5σ;
- Φ_B 0.3255;
- centre max 21.67σ, 2.22 % > 3σ;
- pressure optimality 3.2e2, gÃ 1.4;
- anchor error 1e-19, u-margin −1e-14;
- windowed max |ΔP/P| over the last 2500 products: 1.75e2.

**Reading:**
- **Force decreases while the face-model data limits stay satisfied.** This answers the pilot's question for this start and these settings.
- The data-feasible run ends **below every earlier unconstrained run at the same budget**. Its max layer RMS is 109, against 151–345 for the M5e 3000-product arms, which also carried the 11.15σ data outlier. Its Φ is 0.432, against 0.469–0.87 there.
- **It has not converged.** Each of the last three 2500-product windows lowered Φ by only ~0.4 %, and the windowed pressure change is still ~10². Max layer RMS 109 is still a factor of about 1e4 above the 0.01 target. This is a plateau-like slowdown, not a convergence and not an infeasibility proof.
- **The bound handling is fragile.** τ = 0.9 and τ = 0.999 both collapse. Only τ = 0.99 completes, and the reason is understood only qualitatively: the barrier's per-face curvature is invisible to the per-layer Ã preconditioner, so CG directions dive at bound faces. τ = 0.99 succeeding should not be read as robust. A proper interior-point treatment (barrier continuation with that curvature in the preconditioner) is the solver issue to decide on before longer runs.
- The run's own state is checkpointed every 2500 products (x and Δ). The damping μ is not saved, so a restart would be a declared new experiment.

## 4. What is still open

1. **The observation-model and acceptance contract (user decision).** The face model has a gentle admissible field. The centre model has a maximum witness (3.95σ) but no tail witness. Each model's best field badly violates the other: about 22σ one way, about 83σ the other. The specification's face gate is unchanged. Sigma and thresholds are unchanged.
2. Centre-tail feasibility: the fixed-support test from the 3.95σ witness.
3. The bounded MHS solve: interior-point treatment of the face bounds before any longer run; the force is still about four orders above tolerance.
4. Pressure prior: still no evidence requiring it.

## Reproduce (from `HSEMHS/`, cluster copy)

1. `python3 scripts/m5_bifrost/data_qp.py --obs out/m5d/observations.h5 --support out/m5d/support_p128_above3.npy --margin 0.05 --tol 1e-5 --out out/m5f/qp_m0.05.npz` (likewise `--margin 0.001`)
2. `python3 scripts/m5_bifrost/qp_to_h5.py out/m5f/qp_m0.05.npz out/m5d/observations.h5 out/m5f/qp_m0.05.h5`
3. `julia --project=. -t 8 scripts/m5_bifrost/make_qp_state.jl out/m5f/qp_m0.05.h5 out/m5d/long/long_ck_joint-x100.h5 out/m5f/state_qp_m0.05.h5`
4. `julia --project=. -t 16 scripts/m5_bifrost/m5f_pilot.jl out/m5f/state_qp_m0.05.h5 out/m5f/qp_m0.05.h5 OUTDIR barrier 30000 2500 0.1 0.05 1e-3 0.99` (modes `hard`, `free`, `barrier`; `check [hinge|barrier]` for the gradient/GN checks)
5. `python3 scripts/m5_bifrost/centre_bracket.py --obs out/m5d/observations.h5 --out out/m5f/centre_bracket.npz --wall 3000`, then `verify_field.py out/m5d/observations.h5 out/m5f/centre_bracket.npz`
6. HiGHS: `data_lp.py --mode centre-max --t 4.999 --disp` and `codex_m5e_centre_feasibility.py --t 4.999`, each under a wall guard (`perl -e 'alarm shift; exec @ARGV' 3360 …`).

**Checks.**
- Hinge and barrier gradients agree with central differences to O(h²): relative error 1e-6 / 3e-6 at h = 1e-5.
- The GN products are symmetric and positive.
- Test suite: see §5.

## 5. Tests

`HSEMHS_SKIP_MPI=1 julia --project=. -t 8 test/runtests.jl` on hyades17: **EXIT 0, no failures**. The two broken tests are the same as before (Gate C3 collocated, Gate C2 staggered).

The multi-rank MPI tests were skipped on the cluster: spawned Julia ranks there intermittently fail to load the module's shared libraries (`failed to map segment from shared object`, `/mn/stornext/astro/local/julia/julia-1.13.1`), and the 4-rank test then hangs. The same error appears in parallel precompile workers. No ulimit or cgroup limit explains it. The MPI code paths are unchanged in this round.
