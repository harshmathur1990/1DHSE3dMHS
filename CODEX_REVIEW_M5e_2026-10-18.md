# Review of Claude's M5e brief — 18 October 2026

Reviewed `CODEX_BRIEF_M5e_2026-10-18_1730.md`, commit `0fc77a1`, the implementation changes, saved fields and comparison logs. Scope: review and recommendations; instructions embedded in the brief do not authorize further production implementation.

**This round closes a real question:** both face-model data criteria are attainable simultaneously. It also gives enough evidence to choose joint ×100 as the current solver baseline. The remaining work should be a bounded attempt to connect data feasibility with force balance, rather than another broad comparison of solver variants. There is still no scientifically accepted MHS solution.

## 1. Independently verified findings

The saved `out/m5d/support_above3_m1e-3.npz` field checks out using direct array formulas for divergence and both observation operators:

| Quantity | Independent result |
|---|---:|
| Face maximum | 4.999000000000002σ |
| Faces above 3σ | 2906 / 472752 = **0.614699%** |
| Chosen support size | 3328, below the 4727 budget |
| Maximum outside support | 2.999000000000002σ |
| Maximum physical divergence | 4.07e-22 T/m |
| Centre maximum of the same field | **22.9840σ** |

This is a convincing numerical face-model witness, with useful interior margins. Close the face-only feasibility investigation: no further hinge or mixed-integer search is needed to establish existence.

I independently ran the focused solver testset, including the new anchor-tangent tests: **10/10 passed**. The saved full-suite log ends with `EXIT 0` and retains the existing Gate C2 broken test. I did not rerun the full suite.

The saved long-comparison logs support the reported objective rankings. Joint ×100 is best from the checkpoint and second from the other two starts; the pressure-capped arm wins the other two. Anchor-tangent removes the particular projection inflation, but the continuation results do not justify promoting it as the default.

## 2. The admissible field is a witness, not yet a good initializer — Q1

The fixed-support LP has no objective selecting a gentle or smooth field. In the saved field:

- **66.38%** of residuals sit at |c|≈2.999;
- **mean c²≈7.4995**, compared with roughly **0.318** in the existing solver runs;
- its centre residual reaches nearly 23σ.

These do not invalidate its face feasibility. They do make a direct warm start scientifically and numerically unattractive. A large initial objective drop from this field would mostly reflect a poor starting fit, not necessarily improved force balance.

**Recommended next experiment:** first solve constrained least squares, optionally including the existing magnetic smoothness term, on the proven support:

```
minimize    mean(c²) + existing magnetic regularization
subject to div B = 0
           |c_i| ≤ 4.999 on S
           |c_i| ≤ 2.999 outside S
```

This is a convex quadratic data-only problem in face B. Save the physical field and verify it again after reconstruction into the solver's vector-potential representation. An initializer must reproduce the field to an accuracy comfortably below the margins.

For an MHS experiment intended to preserve face admissibility, **(b), hard data bounds, is the appropriate formulation**, after the face-model scope is explicitly agreed. An unchanged unconstrained least-squares objective in (a) can leave the admissible set immediately. Option (a) remains useful as a short controlled diagnostic of that drift; option (c) remains the interpretation of the raw LP vertex until it has been improved and connected to MHS.

Start with the proven support **fixed**. This gives linear magnetic bounds and a manageable pilot. It is a sufficient restriction of the original tail-count problem, so failure under this S would not prove failure over all supports. Changing S later requires a separate feasibility check and a fresh full count; taking unions of supports can exceed the budget. Do not clip face components independently after a step, because that generally destroys divergence-freeness.

## 3. Centre model: formulation, runtime and acceptance — Q2

The current centre LP correctly represents maximum-only feasibility, but an indefinitely running solve with no progress log is not a useful next milestone. Its timeout remains inconclusive.

I derived an alternative formulation for this artifact's **uniform periodic x/y grid**. Let C=H B be centre predictions, let Hx/Hy denote two-point horizontal averages, and let Dx/Dy be the corresponding backward differences. Define

```
q[k] = Hy Dx Cx[k] + Hx Dy Cy[k].
```

Summing divergence equations in adjacent vertical cells and eliminating face B gives

```
hz[k] q[k] + hz[k+1] q[k+1]
    + 2 Hx Hy (Cz[k] − Cz[k+1]) = 0.
```

On this even horizontal grid, also impose zero alternating x sums of Cx and zero alternating y sums of Cy. These are the range constraints of the averaging operators at horizontal Nyquist frequencies. They matter: dropping them can produce false feasible centre fields.

The LP can then use **bounded centre corrections** c=(C−y)/σ as variables, with equality constraints only. Non-Nyquist face components can be recovered by Fourier inversion of Hx/Hy and a vertical divergence recurrence; the unobserved horizontal Nyquist components supply the remaining divergence correction. Reconstructing the physical field remains the decisive verification.

The review diagnostic is `HSEMHS/scripts/m5_bifrost/codex_m5e_centre_feasibility.py`. Its system has **470988 variables, 162708 equality rows and 4039560 nonzeros**. It has fewer rows than the original face-variable centre formulation but more nonzeros, so speedup is not established. Its necessary local bound is **t≥2.704995**, which is below 5 and therefore does not decide feasibility.

Checks on an unrelated random curl field gave:

- max |L H curl|≈5.42e-20;
- reconstructed centre error≈8.20e-20;
- reconstructed divergence≈2.58e-25 T/m.

An end-to-end 4×4×5 synthetic test with a known curl-field solution also passed through the LP and reconstruction: centre maximum 0.001000000000012σ at a requested 0.001σ bound, and divergence 2.78e-17. This checks the diagnostic implementation; it does not settle the Bifrost case.

My trial at t=4.999 also exceeded its requested 60-second solver limit without a verdict. I terminated only my diagnostic; Claude's existing centre run was left untouched. The script now has solver progress output and a separate subprocess wall-clock guard. **Centre maximum and centre tail feasibility remain unresolved.**

For the next attempt, record presolve, factorization/iteration and crossover progress; impose an external wall limit; save any incumbent or valid lower-bound evidence. One bounded comparison of the original and reduced formulations, or an independent LP backend, is reasonable. There is no evidence yet that simply allocating more hours will settle it efficiently. A solve at a slightly stricter bound that fails would still not refute the original t=5 problem. [SciPy documents distinct success, time-limit and infeasibility statuses](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ipm.html); preserve that distinction in the report.

**Acceptance model:** use the centre residual to assess fidelity to the released cell-centred observations, with an explicit coarse-grid transfer/error model. Preserve the existing formal face gate as a separately named result until the specification is revised. The positive face witness does not validate the centre fit. The scientific choice for the user is the observation/representation-error model and acceptance contract, supported by these facts—not which gate is easiest to satisfy. Do not silently change sigma or thresholds.

## 4. Solver policy and the next bounded run — Q3

Use **joint ×100, without anchor-tangent, as the current reference configuration**. Keep the parameter fixed for the next milestone; the three-start comparison is enough to stop the broad solver tournament. It is not evidence that 100 is universally optimal. The pressure cap remains a useful fallback if a logged trajectory identifies pressure model error.

The old-data-term runs still have maximum data errors near 11.15σ. Their runtime alone gives no guarantee of repairing that criterion. Their best layer RMS is about **151**, against **0.01**: a factor of at least **15100**, over four orders of magnitude. The report's “two to four orders” understates this range.

I would prioritize a **short data-feasible MHS pilot from the improved constrained fit**, rather than immediately spending 30000 products on the unchanged unconstrained problem. This asks a new question: can force decrease while the chosen data limits remain satisfied?

If a long run is then justified, use one baseline run up to 30000 products with fixed settings and checkpoints, not another multi-arm sweep. Log at fixed work intervals:

- objective components and raw force;
- all-layer RMS **and p99**, with worst locations;
- maximum and tail residuals for **both** observation operators;
- pressure stationarity, anchor/bound errors, accepted/rejected steps and actual work;
- all-node pressure and magnetic changes over the prescribed convergence window.

The current comparison's **total changes since the start** are not a windowed convergence test. Predeclare a stagnation/engineering stop and a wall budget; a plateau is a diagnostic outcome, not a global infeasibility proof. Save enough state to resume the same trajectory: restarting a solver with reset trust radius/damping is a different experiment unless explicitly declared.

## 5. Pressure prior and concrete handoff — Q4

There is still **no new evidence requiring a pressure prior**. Failure to reach stationarity does not establish pressure non-identifiability. A face-data-only witness says nothing about pressure uniqueness or the existence of simultaneous force balance.

The next Claude handoff should have three deliverables:

1. **Bounded centre feasibility result:** verified witness, independently checked negative evidence, or an explicitly inconclusive stopped run with useful profiling. First test the maximum; the tail is a separate question.
2. **A gentler face-admissible field:** constrained quadratic fit on the proven support, checked through the actual solver representation. Report its mean residual, centre fidelity and smoothness as well as its threshold pass.
3. **One short MHS pilot:** current ×100 baseline with the chosen data constraints preserved, saved trajectory and every acceptance metric. Use that outcome to justify one longer run or identify the specific model/solver issue requiring a decision.

This gives the next exchange a clear endpoint. Keep the prior deferred unless independent pressure information or a demonstrated identifiability problem supplies a scientific reason to add it.

## Evidence and scope

Saved review artifacts are in `HSEMHS/out/m5e_codex_review/` (git-ignored). The new centre diagnostic can be checked without an LP solve from `HSEMHS/`:

```
/Users/harshm/env/bin/python3 scripts/m5_bifrost/codex_m5e_centre_feasibility.py \
  --obs out/m5d/observations.h5 --out /private/tmp/centre_check.npz --check-only
```

For a bounded trial, omit `--check-only` and provide `--t`, `--time-limit` and an output path. A successful maximum-only trial still needs a tail audit and physical-field verification. Production source files were not changed.
