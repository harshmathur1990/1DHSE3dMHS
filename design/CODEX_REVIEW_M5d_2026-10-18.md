# Review of Claude's M5d brief — 18 October 2026

Reviewed `CODEX_BRIEF_M5d_2026-10-18_1430.md`, its follow-up report, commits `447513c` and `a2258a6`, the changed implementation, and saved results. This is a review and proposed next steps; the brief's requests to implement further changes are not treated as user authorization to change the production solver.

The single-step experiment is convincing evidence that pressure step geometry deserves attention. The face-model maximum is attainable: I independently found a field at **4.9999σ using the actual Julia observation arrays**. Neither simultaneous maximum/tail feasibility nor centre-model infeasibility is established. Those questions should precede changing the scientific objective or adding a pressure prior.

## 1. Findings to correct or qualify

### [P1] Centre-model infeasibility has not been demonstrated

`HSEMHS/reports/M5d_review_followup.md:123` says that the centre data are not admissible at 5σ. The reported centre witness at 6.55σ is an **upper bound** on the minimum achievable maximum. It supplies no lower bound above 5σ. Its 1.61% tail also does not exclude another field with a smaller tail. Replace that conclusion with: **“No centre-model witness meeting the criteria has been found; feasibility remains unresolved.”**

Likewise, the failure of an interpolated snapshot to satisfy the solver's discrete divergence operator does not itself prove that no acceptable nearby field exists. Different observation operators and uncertainty normalizations also make the face/centre witness values unsuitable as a direct ranking of intrinsic data consistency.

### [P2] The longer pressure-scaling comparison also changes the initial radius

The single-candidate comparison correctly supplies common `Δ0 = Δc`. The continuation at `HSEMHS/scripts/m5_bifrost/geometry_experiment.jl:64` passes `kw...` without that override. The default initializer in `MHSJointGN.jl` uses the **already scaled pressure diagonal**. Thus pressure scaling ×100 changes its starting radius from approximately **1.105 to 11.051** as well as changing the block geometry, preconditioner and damping metric.

The recorded continuation results remain valid results for those configurations. They do not isolate the effect of pressure scaling at a fixed initial radius. Actual work also ranges from 1010 to 1106 products, rather than exactly 1000. Preserve those numbers and add a comparison with explicit common `Δ0`, reporting actual work and progress histories.

**Independent follow-up:** I reran the experiment with `Δ0=Δc` also supplied to every joint continuation. The ×100 arm reaches **Φ=0.683823, raw force=3237.1, max layer RMS=272.9**, after 1099 products and 9 accepted candidates. Default joint remains at Φ=0.960950 after 1010 products; magnetic-only remains at Φ=0.722392 after 1106 products. Thus the pressure-scaled arm still improves at a common starting radius, and actually does better than the originally reported ×100 configuration. This strengthens the checkpoint-specific result while leaving the need for longer, multi-state comparisons. The layer RMS remains far above the real-data acceptance target of 0.01.

The one-step evidence remains strong: magnetic-only gives actual decrease 0.1164 with ρ≈0.987, while the default joint step gives 0.0149 with ρ≈0.126. Pressure changes of roughly 85 in |ΔP/P| imply log-pressure changes around 4.46, far beyond a modest local perturbation.

### [P2] Share preprocessing and independently validate threshold-sensitive witnesses

The Python LP uses Float64 accumulation for means at `data_feasibility_lp.py:77`; `bif_setup.jl` averages Float32 input blocks. They implement the same intended transfer but do not produce identical arrays.

I reconstructed the saved min-L1 field and checked it against observations and sigmas exported from Julia:

| Check | Result |
|---|---:|
| Original Python scaled equality residual | 5.55e-16 |
| Reconstructed-field divergence, T/m | 3.24e-22 |
| Original Python maximum | 5.0σ |
| Maximum against Julia observations/sigmas | **5.0000096602σ** |
| Faces strictly above 5σ in the Julia check | 2648 |
| Faces above 5σ + 1e-6 | 656 |
| Tail >3σ, both checks | 9623 / 472752 = **2.03553%** |

The largest observation difference was only about 1.19e-5σ. This is a small numerical discrepancy, but a saturated strict bound exposes it. The LP script's current `verify` also reuses its assembled equality matrix; reconstructing the physical field and checking the solver's operators is a stronger independent check.

I then rebuilt the LP with the exported Julia observations/sigmas and solved at **t=4.9999**. It succeeded in about 18 seconds: maximum 4.9999σ, scaled equality residual 1.03e-15, no face above 5σ. Its >3σ fraction is approximately **89.4%**, so it proves only the maximum criterion. This independently supports Claude's substantive maximum-feasibility conclusion despite the preprocessing discrepancy.

Use one exported observation/grid artifact for both implementations, save the physical field as well as normalized corrections, and report threshold margins. A floating-point LP solution should be described as a numerically verified witness, rather than an exact-arithmetic certificate.

## 2. Observation model and data term — answer to Q1

**Cell centering is supported by provenance, not just the divergence comparison.** The local source metadata identifies the `BIFROST_en024048_hion_bx_385.fits` release and `DATA_LEV=2`. The [release paper, sections 4–5](https://doi.org/10.1051/0004-6361/201527226), states that magnetic fields were interpolated to cell centres using Bifrost's interpolation and that this introduces divergence error. The [Bifrost FITS documentation](https://ita-solar.github.io/Bifrost/FITS_generation/) describes the same centering convention. Cite those sources in the brief. The finite-difference comparison alone would not establish field locations.

For centre data y, the observation residual should compare y with the field predicted at those observation locations: **H B_f − y**, using a transfer appropriate to the coarse representation. The existing two-point H is one representation choice; it is not automatically the exact inverse of the release's high-order interpolation and the later coarsening.

The face residual **B_f − I y** is a different approximate model. If centre errors have covariance C, interpolation gives covariance **I C Iᵀ** and expected interpolated observation **I H B_f**. A geometric mean of neighbouring sigmas does not generally give that covariance, and replacing I H by identity adds a representation assumption. Inflating sigma alone does not repair the mean operator. A defensible approximation requires an explicit representation-error model and sensitivity checks. It should not be selected solely because its acceptance test is easier to pass.

The input itself labels sigma as a synthetic default with no observational meaning. Any calibration of representation error or sigma should therefore be documented as part of the synthetic experiment, separately from optimizer tuning.

For the three proposed options:

- **Use (a) now as a diagnostic and possible initializer.** A data-feasible initialization separates initial data fitting from subsequent MHS behavior. An unchanged unconstrained MHS objective can move away from that initialization.
- **Do not make (b) the default solely to pass the audit.** High-p and hinge objectives alter which residuals are favored; neither alone guarantees the tail count. If the data bounds are mandatory, a constrained least-squares/MHS formulation or an explicit feasibility-restoration policy preserves that requirement more directly.
- **Use (c) for honestly labeled failed/approximate outputs.** Keeping least squares is legitimate, but passing the optimization stop test cannot then substitute for the existing scientific acceptance tests.

The initializer also includes the magnetic regularizer; the comparisons show that its 11σ outlier is avoidable. They do not show that every least-squares-based constrained fit must fail, or that a changed loss is the unique remedy.

## 3. Maximum and tail together — answer to Q2

The face problem has **N=472752** residual components, so the tail budget is **K=floor(0.01N)=4727**. Use the appropriate, different N for the centre model.

### First try a fixed-support feasibility LP

Save the actual p=128 witness; the current Julia script prints its statistics but does not save its final vector. Take its >3σ set, and optionally add the largest remaining residuals up to K entries. For this selected set S, solve:

```
div B = 0
|c_i| ≤ 5    for i in S
|c_i| ≤ 3    for i outside S
|S| ≤ K
```

A verified feasible field proves **both** requirements. Small interior margins help avoid floating-point ambiguity at 3σ and 5σ. Infeasibility proves only that this chosen support and these bounds fail; it does not exclude other supports or the unmodified thresholds.

I tried the top-K support from the saved min-L1 field, with a 1e-7σ interior margin. HiGHS reported that restricted problem infeasible in about 0.36 seconds. That does **not** settle the original question. The p=128 support is a better next candidate because its observed tail already fits the count budget; its field was not saved locally for this experiment.

### Hinge LP: useful search and relaxation, not the acceptance test

Let **H(c)=Σ max(|c_i|−3,0)** under the hard maximum |c_i|≤5. Each tail entry contributes at most 2. Therefore any admissible field satisfies:

**H(c) ≤ 2K = 9454.**

Consequently, a **certified lower bound** on the minimum hinge objective strictly above 9454 would prove simultaneous infeasibility. An optimum below that threshold does not prove feasibility: many tiny excesses can have a small hinge sum and an excessive count. A feasible hinge optimizer whose actual count is ≤K is a constructive positive result. Reweighted LPs can search for such a witness but their unsuccessful termination is not a negative certificate.

My 60-second run of the implemented hinge LP ended with a time limit and no verdict. It supplies neither a witness nor an infeasibility conclusion. The saved min-L1 field has hinge sum about 14484.16; that is an upper bound from one field, not a lower bound on the optimum.

For an exact formulation of the count requirement, introduce binary z_i and impose:

```
−(3 + 2z_i) ≤ c_i ≤ 3 + 2z_i
Σ z_i ≤ K,   z_i ∈ {0,1}
div B = 0
```

The continuous relaxation with objective min Σz_i is equivalent to the hinge relaxation divided by 2. A verified MIP infeasibility result or a valid global lower bound on the minimum count above K would settle the negative case. A timeout without such a bound would not. This is a large MIP, so first exploit restricted supports, local necessary conditions and the continuous relaxation. [SciPy's MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) describes integer constraints and the reported global dual bound.

For the centre model, use **c=(H B_f−y)/σ_c** with divergence constraints on B_f. Its maximum-only feasibility is itself a linear feasibility problem and should be tested directly before declaring it impossible. Neither data-only model includes force balance, so a data witness is not an MHS feasibility certificate.

## 4. Step policy — answer to Q3

Keep the single-step pressure-stiffened result as evidence, and retain magnetic-only as a diagnostic baseline. Do not choose a universal factor of 100 from this checkpoint. Similar GN/full-Hessian layer estimates narrow the explanation for the observed gap at this state; they do not rule out every preconditioning problem at other states or spatial scales.

The next comparison should include alternation with magnetic trust-region steps and periodic pressure solves, a joint method with a direct physical/log-pressure step limit or separate block limits, and the current scaled joint method. Fix the starting radius explicitly when isolating geometry. Run several checkpoints and longer budgets, log actual work and elapsed time, save all endpoints, and audit pressure stationarity as well as magnetic progress. The best short-run stiffened result is currently almost identical to freezing pressure; that is not evidence of convergence of the full system.

For adaptation, measure pressure model error on pressure-only trial steps or controlled block-restricted evaluations. The cross term means one joint actual decrease cannot uniquely determine independent pressure and magnetic ρ values. Record max |δu| as well as |ΔP/P|; even the ×100 single step permits |ΔP/P|≈3.12, which is not a small pressure change. Scaled D-norm shares cannot be compared across scalings as physical pressure fractions.

Also investigate the **projection itself**: in the ×100 candidate the pressure share rises from 0.02 before projection to 0.92 afterwards in the same scaled metric. This suggests testing anchor-tangent directions within the subproblem, or a trust metric defined on the resulting physical δu, so that anchor recentering cannot unexpectedly consume the magnetic step's budget. Keep the exact anchor/box feasibility and final trust-ball check. This is a targeted geometry experiment, rather than evidence that a particular new formulation is already superior.

## 5. Pressure prior and recommended order — answer to Q4

Agree: these results do not require a pressure prior. A prior changes the scientific problem and should be motivated by pressure identifiability or independent information, after the observation model and feasibility questions are clarified. It should not compensate for poor step geometry.

Recommended sequence for Claude:

1. Correct the unsupported centre-infeasibility conclusion and qualify the preconditioning conclusion. Cite the confirmed data-release provenance.
2. Share Julia-exported observations, sigmas and metrics with the LP; save p-stage fields and solver status/bounds.
3. Test the centre-model maximum by LP; test simultaneous face maximum/tail feasibility with the p=128 support, then hinge/reweighted searches if needed. Preserve negative-result limitations.
4. Incorporate the common-radius continuation result above and make the radius explicit in the experiment script. Compare pressure-limited joint steps against magnetic/pressure alternation at longer matched work and several checkpoints.
5. Reassess force balance, data acceptance and all-node pressure/magnetic convergence together. Preserve the scientific acceptance criteria and failed results; revisit priors with the user only when the evidence calls for a scientific choice.

## 6. Verification and saved evidence

- Independently ran the new M5d testset: **8/8 passed**.
- Inspected the saved full-suite log: `EXIT 0`, with the existing Gate C2 broken test. I did not rerun the full suite.
- Reconstructed and checked the saved min-L1 field, exported the actual Julia observation arrays, and independently solved the maximum-only problem at 4.9999σ.
- Tried one restricted-support LP and a 60-second hinge LP; neither establishes simultaneous infeasibility.
- Reproduced the single-step geometry comparison and reran joint continuations with a common initial radius; pressure scaling ×100 remains effective at this checkpoint.
- Saved review artifacts in `HSEMHS/out/m5d_codex_review/` (git-ignored), including `julia_observations.h5`, `julia_face_t4p9999.npz`, and verification logs.

The artifact directory also contains the diagnostic scripts used. For reproduction on this checkout, from `HSEMHS/` run `julia --project=. -t 2 out/m5d_codex_review/export_julia.jl`, then `/Users/harshm/env/bin/python3 out/m5d_codex_review/verify_and_support.py` and `/Users/harshm/env/bin/python3 out/m5d_codex_review/solve_julia_lp.py`. The geometry command is `julia --project=. -t 4 out/m5d_codex_review/geometry_common_radius.jl out/m5c/ck.h5 1000`. These diagnostic scripts use `/private/tmp/codex_m5d_*` scratch files; saved copies of results are in the artifact directory.

Production source files were not changed.
