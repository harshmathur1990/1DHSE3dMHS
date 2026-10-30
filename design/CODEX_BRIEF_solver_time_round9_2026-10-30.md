# CODEX brief, round 9: release-candidate closeout check (2026-10-30)

Branch `solver-time`. The implementation is commit `4f7908f` ("Round 8 closeout (Codex): release candidate"); this brief
is the only change after it. This round carries out the closeout actions of your round-8 review
(`design/CODEX_REVIEW_solver_time_round8_2026-10-29.md`). The frozen record is
`design/RELEASE_CANDIDATE_solver_time_2026-10-29.md`.

- **Unchanged:** equations, Φ, boundaries, gn_tol 1e-3, the 25-step budget and the configured defaults.
- **Not touched:** main and embedding-api. Nothing is merged or deployed.
- **No new research or benchmarks.** Only the checks affected by these fixes were rerun.

Review range: `git diff 33169d9 4f7908f` (10 files, +296 / −58). One README wording fix comes with this brief: the
test-coverage line now says "interface parity" instead of "equivalence".

## 1. CLI exit codes (your finding 1)

**Contract kept:**

| exit | meaning |
|---|---|
| 0 | converged |
| 2 | valid, not converged (max_iter or stalled) |
| 3 | invalid result, not written |
| 4 | invalid input |

**Caller audit (repo).**

| caller | how it uses the CLI | change |
|---|---|---|
| `HSEMHS/scripts/delivery/run_case.sh` | the only shell caller; no `set -e`, `&&` chain or retry on the run | header now documents the codes and says the script records N without stopping on it |
| `test/scripts/compare_embedded_refs.jl` | prints the code next to the comparison | none |
| `test/embedded_helpers.jl` | the parity reference | see §3 |

- No script in `HSEMHS/scripts` uses `set -e` or `pipefail`.
- `measure_steps.jl` and `m5h_fullres.jl` only mention `scripts/run.jl` in comments.
- **FFNOML's callers are outside this repo and were not audited.** That is listed as a pre-merge step for the user.

**README.** A migration note plus a shell example: accept 2, inspect the saved `status` attribute, and fail on 3 and 4.
No compatibility option was added.

**Subprocess smoke test.** `test_embedded.jl:174`, "CLI exit status follows the solver status":
- `runc` runs `run_config` in process.
- `runp` starts `Base.julia_cmd()` on `scripts/run.jl` as a real process and checks the process exit code and the
  output's `status` attribute:
  - converged → `(0, "converged")`;
  - `max_steps = 1` → `(2, "max_iter")`;
  - an invalid option (`rtol = -1`) → `(4, "")`, with no output.
- The config and output are `tempname()` files, removed after each run. This test does not cover exit 3; the
  invalid-rerun test in §2 does, in process.

## 2. Invalid reruns and an older output (your finding 2)

`RunModes.jl:101–126`.
- `invalid_result(logm, st, par, out, cfg)` still writes nothing.
- If a file already exists at `run.output`:
  - it is not deleted;
  - the log says that this file is from an earlier run and was **not** replaced.
- Rank 0 writes `<output>.failed.txt` beside it, with:
  - status and reason;
  - config path and time;
  - the path not written;
  - whether an earlier file exists;
  - the messages.
- `clear_failure_record` removes the record after a valid write. It is called in the column-ltau, ltau and z drivers.

**Not done:** atomic temporary-then-rename publication. The record plus the log met your requirement without changing
the write lifecycle (the z driver writes cooperatively from all ranks). The README "Output and invalid runs" section
recommends one output path per run.

**Test.** `test_embedded.jl:202`, "an invalid rerun does not replace, delete or disguise an earlier output", runs
`column-hse-ltau` in process three times at one path. The three drivers share `invalid_result` and
`clear_failure_record`.
1. valid (exit 0);
2. invalid, by a copy of the fixture with one T = 1e9 cell (exit 3): the earlier file's mtime and Pgas are unchanged;
   the record exists and names `eos_out_of_envelope` and the earlier file;
3. valid (exit 0): the record is gone.

## 3. Same-machine references are interface parity (your finding 3)

- **Renamed.** The testset is "interface parity with the standalone driver". `ref_file` is now
  `parity_ref(mode, tag) → (ref, attrs)` (`embedded_helpers.jl`).
- **Poisson modes.** The standalone driver runs in a `mktempdir`, which is removed. It must return 0 or 2, and the
  output must exist; otherwise the test errors. The parity test also compares `stop_reason` with the embedding status,
  besides the arrays.
- **Column modes** keep the stored files, which keep their provenance.
- **The 0.86 and 1.97 relative numbers** are replaced by absolute ones, on the Φ-rule-stopped, not-converged ltau 24×18
  crop:

  | comparison | z max | z rms |
  |---|---|---|
  | 4 vs 16 threads | 58 m | 4.2 m |
  | hyades18 vs hyades20 | 90 m | 6.3 m |

  The docstring and README say these are differences on an unconverged crop, not on a converged atmosphere.
- **Solver correctness** is tested independently (Jacobians, direct systems, known equilibria: test_block, test_view,
  design §6.12).

## 4. Decrement label (your finding 4)

`gn_decrement_approx` (Bool) is replaced by `gn_decrement_source` (Symbol), in the embedding stats and the output
attributes. Code: `PoissonTau.jl:1158, 1178, 1314–1330`.

| value | when |
|---|---|
| `:not_evaluated` | EOS failure (NaN); acceptance at the start by rms_atol, Φ0 = 0 or round-off. The 0 is a convention; `roundoff` keeps the reason. |
| `stop_check_<outcome>` | the scalar is `q.pred` from the stop check at the returned state; the interval [L, U] in `qual_lower` / `qual_upper` is the authoritative statement about the threshold |
| `final_solve_<pcg flag>` | an ordinary final PCG solve. The log calls the scalar an estimate unless the solve converged; even then it is described as a computed estimate, not an error certificate. |

**Tests (test_block, now 116):**

| case | expected source |
|---|---|
| u5 | `stop_check_*` |
| u6 | `final_solve_maxit` |
| r2 (round-off start) | `:not_evaluated` |

## 5. Cap probe wording (your finding 5)

README, design §6.15 and the release record now state only the measurement: at the two saved states, 2× the capped step
and the full step along the same direction both raise Φ. The phrase "already as long as the model allows" is gone.

They also say that the ln P cap, not the height cap, binds. The probe does not show that the capped step is optimal, or
that a trust-region step in another direction could not do better.

## 6. Freeze and verification

`design/RELEASE_CANDIDATE_solver_time_2026-10-29.md` records:
- the commit;
- 54 commits ahead of main `5e901b5` (checked);
- Julia 1.13.1 with the committed Manifest, MPICH via `mpiexecjl`, and `cache/eos_table_fullz.h5`;
- defaults and opt-in settings, user-facing changes, test commands, results, requirement status and merge proposal.

| suite | result |
|---|---|
| units/config | 18 / 18 |
| test_block | 116 / 116 |
| test_warm | 47 / 47 |
| test_view | 133 / 133 |
| test_embedded | 103 / 103 on hyades18 and on hyades20 |
| mpi_tests, ranks 2 / 3 / 4 | 34 / 34 at each |
| mpi_embedded, ranks 2 / 3 / 4 | 30 / 30, 30 / 30, 32 / 32 |
| mpi_view, ranks 2 / 3 / 4 | 34 / 34, 28 / 28, 34 / 34 |

## 7. Requirement status (unchanged from round 8)

| requirement | status |
|---|---|
| Column HSE at an angle (CLI and embedding) | delivered |
| MHS at an angle: validity, correctness coverage, status reporting | delivered |
| Faster valid budget-limited MHS output | delivered: `switch` (opt-in) is 1.9–4× faster on the Bifrost cube |
| Qualified convergence on the Bifrost cube | **not met**: every μ ends max_iter (exit 2) |
| A few minutes on 8 nodes / 256 CPUs | **unmeasured** |

## 8. Questions

1. Do the five closeout actions meet your round-8 handoff? Is the candidate ready to present to the user for a merge or
   scope decision?
2. Failure provenance uses a record plus a log line, not atomic publication. Is that enough, or should atomic
   temp-and-rename be a merge prerequisite?
3. Anything in the release record (frozen settings, environment, test commands, merge proposal) that is missing or
   misstated?
