# Round 8: release-candidate assessment — 2026-10-29

Reviewed HEAD `b1bfca3`, round-8 brief, diff since `a60d959`, qualification/final diagnostics, CLI status changes, reference helpers/tests, README and cap-probe implementation. No tests, scientific jobs or cluster access performed. Verification counts and timing results are Claude's reported evidence. No scientific source edited. Merge and scope acceptance remain the user's decisions.

## Assessment

**Yes: this is a reviewable release candidate for the implemented angle support and opt-in acceleration, with explicit limitations.** It is not completion of qualified full-cube MHS convergence or the requested deployment runtime. The brief makes that distinction correctly. Keep Fourier as default and switch opt-in; do not reopen multilevel or trust-region development for this candidate.

Column HSE has dedicated angle/API checks, full-size valid results and reported MPI coverage. MHS has derivative/control/API/MPI verification and measured budget-limited acceleration. Those are useful deliverables independently of Bifrost truth-height agreement.

## Final findings and closeout actions

### 1. CLI exit codes: keep honest numerical status, document the compatibility change

Exit 0 for convergence, 2 for a valid max_iter/stalled result and 3 for invalid output is a defensible and clearer contract. Do not revert to exit 0 for every valid atmosphere merely to make shell scripts succeed. A valid atmosphere and a converged solve are different outcomes.

However, this changes caller behaviour. Audit the repository's batch/delivery scripts for `set -e`, `&&`, pipeline status and retry logic. Scripts intentionally retaining budget-limited output should explicitly handle exit 2 and inspect the saved status. Invalid results should remain a failure. Add a concise migration note and shell example, rather than another compatibility option unless an actual caller requires it.

The new tests invoke `H.run_config` in process; that checks its return value, not the executable process's final exit code. Add one small subprocess smoke test through `scripts/run.jl` for 0/2 and invalid-input handling, and confirm resource cleanup/output status. This is a narrow interface check, not a new scientific test campaign.

### 2. Invalid reruns can leave an older output at the requested path

`check_output` permits an existing file with overwrite=true. If the new solve is invalid, `invalid_result` returns without writing; an old successful file can therefore remain at that path. A caller may mistake it for the failed run's result.

Make the behaviour explicit and test it. Prefer unique run IDs/output paths in examples, and a clear failure record/log identifying that the existing file was not replaced. If changing write lifecycle, use a temporary output and atomic publication on success, with a policy for an existing destination. Do not silently delete prior results. This is artifact provenance, not a reason to change numerical status.

### 3. Same-machine references: acceptable for parity, not independent solver verification

Using the standalone driver in process is reasonable for testing embedding-vs-CLI equivalence when budget-limited iterative trajectories differ across CPUs/BLAS settings. Name these **interface parity tests**, not portable numerical golden references. The independent derivative, direct-system and known-equilibrium checks remain necessary and are already reported.

`ref_file` currently discards the standalone return code. Require an allowed code plus a present, valid output with the expected status before using it as the parity reference. Compare stopping reason/validity as well as arrays, and clean temporary config/output files. Preserve provenance for historical stored results.

The reported relative differences 0.86/1.97 concern a sensitive unconverged fixture and a relative metric; they do not demonstrate differences of that magnitude in a converged physical atmosphere. Record absolute/scaled field differences and status alongside those numbers if retained in user-facing documentation. No need to investigate hardware sensitivity further to close this candidate unless the converged controls fail.

### 4. Qualification threshold and decrement accuracy are separate statements

`gn_decrement_approx=false` after a qualified interval means the threshold was decided, not necessarily that the returned scalar `q.pred` is accurately known. For an ordinary final PCG solve, meeting its residual tolerance likewise is not a rigorous decrement-error certificate. Keep the interval/outcome as authoritative and describe the scalar as a computed estimate; rename/document the flag precisely rather than presenting it as exactness.

The early rms_atol/roundoff path returns gradient/decrement zero without computing them. Mark those as not evaluated or explicitly conventionally zero, and preserve the acceptance reason. This is a diagnostic-label fix; do not change the configured stopping rule or start another qualification study.

### 5. Cap experiment: reasonable scope decision, narrower numerical conclusion

The probe shows that tested larger **radial** steps increase the objective at the saved states. That supports declining the trust-region trial for this release. It does not prove the capped step is optimal, that the GN model itself disallows it, or that a different trust-region direction could not improve progress. Replace “already as long as the model allows” with the measured observation. Record the correction that lnP, not the height cap, binds. No further experiment is requested here.

## Answers to Claude

1. **Reviewable candidate?** Yes, after the small interface/provenance checks above and a fixed commit/configuration/test record. It can be offered for user review as angle support plus faster valid budget-limited MHS output. Do not label the original convergence/runtime requirement achieved.
2. **Reference policy?** Yes for same-environment interface parity, backed by independent correctness/control tests; not as the sole solver regression oracle.
3. **Exit 2?** Keep it, document the breaking behaviour and update intentional budget-limited callers. Do not equate process success, physical validity and numerical convergence.

## Final handoff

- Complete the small exit-code, existing-output and parity-reference checks; correct diagnostic labels and the cap interpretation.
- Freeze the release candidate commit, supported settings, Julia/Manifest requirements and reported test commands/results. Run only checks affected by these final fixes; do not repeat the full benchmark campaign without a new concern.
- Present one final change summary and requirements table to the user: column HSE delivered; MHS correctness coverage and opt-in speedup delivered; qualified Bifrost convergence not met; eight-node runtime unmeasured.
- Prepare the merge/release proposal without merging or deploying on the user's behalf.

No new algorithm research is a prerequisite for this candidate. The unresolved convergence/runtime work must remain explicit future scope unless the user chooses to continue it. A limited release and completion of the original request are different decisions, and the current evidence supports the former.
