# HSEMHS design history (15–26 October 2026)

This file condenses the 27 design, brief, review and response documents written between 15 and 24 October 2026. The
originals are unchanged in [`old/`](old/), and every section below names the ones it draws on. Code, `docs/` and the
root `README.md` describe the current state; this file records **how and why** the project got there. New design work
lives in topic folders (e.g. [`tau/`](tau/) for the viewing angle).

Roles throughout: the **user** decides; **Claude** implements and writes briefs; **Codex** reviews independently
(`CODEX_BRIEF_*` → `CODEX_REVIEW_*` / `REVIEW_*`, with Claude's assessment in `CLAUDE_RESPONSE_*`).

---

## 1. Timeline at a glance

| Date | Phase | Outcome | Original documents (`old/`) |
|---|---|---|---|
| 15 Oct | Design v1 (Codex) and critique (Claude) | Problem framed; three main risks identified | `HSE_MHS_DESIGN_REVIEW.md`, `HSE_MHS_DESIGN_CRITIQUE.md` |
| 16 Oct | Reply, rulings, binding spec, implementation plan | Two modes (`column_hse`, `mhs`); bottom anchor; staged gates | `review-response.md`, `REVIEW_RESPONSE_CRITIQUE.md`, `HSE_MHS_FINAL_SPEC.md`, `HSE_MHS_IMPLEMENTATION.md` |
| 16 Oct | M0–M4a: EOS, column HSE, collocated MHS | Gate C3(b) fails (checkerboard null modes) | spec §10; reports `M0`–`M4a` |
| 16–17 Oct | D9 / D9b: staggered vector-potential MHS, Φ_Q | Synthetic gates pass | `HSE_MHS_D9_STAGGERED_ADDENDUM.md`, `HSE_MHS_D9b_AVERAGING_ADDENDUM.md` |
| 17–21 Oct | M5a–M5g: full MHS on Bifrost, six Codex rounds | Never converges; data admissibility settled; no accepted MHS state | `CODEX_BRIEF_M5*.md`, `CODEX_REVIEW_M5*.md`, `CLAUDE_RESPONSE_CODEX_M5.md`, `REVIEW_M5f/M5g_*.md` |
| 21–23 Oct | M5h / M5i: full-resolution runs | 24 h runs, still unconverged; stopped by the user | `REVIEW_M5g_2026-10-21.md` |
| 23–24 Oct | Fixed-B Poisson pressure steps (z, then τ) | Minutes per cube; usable inside inversions | `CODEX_BRIEF_modes_2026-10-24.md` |
| 24 Oct | Five run modes merged into `main`; README and docs | Φ-only stopping policy | `CODEX_BRIEF_modes_2026-10-24.md`, `CLAUDE_BRIEF/REPORT_README_DOCS_2026-10-24.md` |
| 24–26 Oct | Embedding API (FFNOML), merged to `main` | Production use by FFNOML | git history; `HSEMHS/reports/embedding_api_gate.md` |
| 26 Oct | τ-cut study (branch `ltau-cut-study`) | Keep `logtau_cut = −4` (β ≳ 10) | `HSEMHS/reports/ltaucut/` |

---

## 2. Problem and founding decisions (15–16 Oct)

**Goal.** Given T (and optionally B) on a 3-D grid, either a geometric-height grid or, later, an optical-depth grid,
compute P, ρ, nₑ and τ₅₀₀ (on τ grids: geometric heights) with Wittmann LTE EOS/opacity, MPI plus threads, top-first
storage, SI units.

**What the design exchange established** (Codex design → Claude critique → Codex reply → Claude rulings):

- **Full 3-D HSE with B = 0 forces P, ρ and T to be horizontally uniform** (Codex). A structured cube therefore needs
  the Lorentz force, so the plane-parallel "3-D HSE" mode was dropped (**D0/D1, user**). Two modes were built:
  `column_hse` (each column independently) and `mhs`.
- **Fixed T and fixed B over-determine P** (three equations, one unknown). Either B must move within σ_B (full MHS) or
  the force balance is only satisfied in a least-squares sense (what the Poisson modes later do).
- **The top pressure anchor was the largest single error source** (Claude, verified on Bifrost 385). With a uniform
  0.1 Pa at +3 Mm, the error at z = 0 is +2.5 dex (median) and +5.7 dex (p99); with a bottom anchor it is
  +0.01 / +0.11 dex. Hence **D2**: `column_hse` is bottom-anchored by default, and **D3**: `mhs` uses a single scalar
  gauge (the mean ln P over the anchor plane), not a pressure map.
- **Codex corrected the critique in eleven places**, all accepted. Examples: field-line HSE in arclength; the
  loop-closure problem with footpoint anchors; "screening", not "feasibility", maps; τ-top closure with H_χ, not H_ρ;
  L-BFGS memory arithmetic; centred differences annihilate odd–even modes.
- Other binding decisions: T fixed bitwise (D4); one EOS state supplies ρ, nₑ, Pₑ (D5); τ₅₀₀ is total extinction with a
  top-first trapezoid and an `exp_tail` top closure (D6, with erratum E1 on the sign of H_χ); **viewing angle only
  `per_column` τ/μ in v1, with 3-D rays deferred to M8 (D7)**; Float64/SI with units read from metadata (D8); L-BFGS
  least squares, no PETSc (D10); threads first, MPI at M7 (D11).
- Gate 0 (input audit) on Bifrost 385 recorded the facts that shaped everything later:
  - β crosses 1 near z ≈ 1 Mm;
  - at 1.5–2 Mm, 30–37 % of cells have ρv² > P, so a static fixed-T model cannot be faithful there;
  - 66 % of the top columns are above 3×10⁴ K, so a hot EOS branch was needed.

`HSE_MHS_IMPLEMENTATION.md` turned the spec into layouts, stencils, schemas and tests (errata E1–E3, later copied into
the spec's decision log).

## 3. EOS, opacity and column HSE (M0–M3, 16 Oct)

- Wittmann C++ adapter with tightened convergence, a bicubic table, and a Saha hot branch. Gate 1 passed.
- **`FIX_GASC_H2`** (user): Wittmann's `gasc()` omits a factor pₑ in the H₂/H₂⁺ constants. Fixed by default; it
  changes χ₅₀₀ by up to 1.36 dex below 3500 K.
- **Opacity**: Wittmann/ATLAS 500 nm opacity kept. It agrees with RH to 4.3 % after two RH artifacts were removed
  (hydrogen bound-free continua with n ≥ 5 skipped; n(H⁻) computed with total instead of neutral H).
- Column HSE: RK4 in ln P with co-integrated column mass (E2); Gate 2 passed. Screening maps (M3).
- Later (22 Oct): `cache/eos_table_fullz.h5` (T up to 1.781 MK, log P from −13 to 15) replaced the cropped table.

## 4. Full MHS: from collocated to staggered (M4a–M6d, 16–17 Oct)

- **Collocated scheme (D9 v1) failed Gate C3(b).** Centred differences leave odd–even modes in B and P with no force
  response; conditioning was ~1e11, and the exact discrete equilibria had a residual floor (E3).
- **D9, staggered vector potential** (user-approved): P at centres, B = curl A on faces (∇·B = 0 exactly), J on edges,
  compact two-point force balance on faces. E3 was withdrawn.
- **D9b**: the remaining checkerboard was a joint near-null of the data operator and the averaging operators, because
  every symmetric average vanishes at Nyquist. Fixes: data compared on faces; a current-Nyquist penalty Φ_Q;
  horizontal force rows on interior layers only. Also, the top pressure is **not identifiable** where
  δP_unobs = |B|σ/μ₀ > P.
- **M6d** (accepted with deviations): λ_Uz = 0; plain L-BFGS, with B-first continuation when the start is
  Lorentz-dominated; a collapse safeguard; C3(b) judged by decay ≥ 10³ to the round-off floor. Synthetic gates passed.

## 5. Full MHS on real data: six review rounds (M5a–M5g, 17–21 Oct)

Setting: Bifrost coarsened to 42×42×89; unknowns u = ln P plus a vector potential; objective
Φ = λ_F Φ_F + Φ_B + Φ_S + λ_F λ_Q Φ_Q.

| Round | Main question | What was learned |
|---|---|---|
| M5 / M5a | Why no convergence on Bifrost? | Start Φ_F is dominated by the Lorentz force in low-β layers (\|R\|/S up to 1e4). δP_unobs > P in the top ~48 of 89 layers. Joint L-BFGS never moves B; B-first moves B but the joint stage stalls; pressure lurches by large factors. |
| Codex M5 review | Formulation or numerics? | Fix the safeguards and stopping first: bound/anchor projection, zero-work "converged", the inexact anchor diagonal. Do not lower the acceptance standard because P is unidentifiable. |
| M5b (+2 follow-ups) | Pressure block and frozen-B experiment | A trust-region pressure block, a separate trial workspace and fixed base model. Neither branch converges, and earlier apparent agreement was not evidence of a common optimum. `stalled_at_floor` label defined (18 Oct, four conditions; never `converged`). |
| M5c | Joint GN vs alternation | Audits and work counters (`MHSAudit.jl`), joint GN (`MHSJointGN.jl`). Alternation stays the baseline; the priority is magnetic **data admissibility**. |
| M5d / M5e | Can any divergence-free field meet the data limits (max 5σ, ≤ 1 % > 3σ)? | **Face model: yes**: a verified witness at 4.999σ with 0.615 % > 3σ. Centre model: unresolved. Joint ×100 is the most consistent solver arm; none converges. |
| M5f / M5g | Data-feasible MHS pilot | A centre-model witness at 3.946σ (tail 1.41 %, still unresolved). A barrier-based pilot keeps face admissibility while lowering Φ 60.6 → 0.43, but does not converge. M5g made the curvature probes barrier-safe, the bounded steps face-feasible and the brackets "numerically verified". |

**Net result:** a well-tested staggered MHS machinery, with feasibility questions settled for the face data model,
but **no accepted MHS equilibrium on real data**. The obstruction is physical as much as numerical: low-β layers where
the given B is not force-free and P is unidentifiable.

## 6. Full-resolution runs and the policy change (M5h–M5i, 21–23 Oct)

- **M5h** (Codex review handoff `REVIEW_M5g`): native 504×504×177 run, monitoring the height of τ₅₀₀ = 1
  (target: max change < 1 mm per iteration). It ran 24 h on 4 nodes and ended at the wall cap without converging.
- **M5i** (branch `m5i-phi-notrust`, user): σ_B a uniform 5 G (B may change, penalised); no trust radius; stop on
  |ΔΦ|/Φ < 1e-6. The user's reasoning: real observations give no usable σ_B, and Φ tracks the MHS equation directly.
  It was killed at accepted iteration 112 (Φ 7.9e3).
- **Consequence:** a 24 h solve cannot sit inside an inversion loop, so attention moved to fast fixed-B pressure
  corrections.

## 7. Fixed-B Poisson pressure steps (23–24 Oct)

- **`poisson-step` (z grid), FIRTEZ-like.** Weighted least squares in u = ln P with B fixed; Gauss–Newton with an
  FFT/tridiagonal preconditioner.
  - Solving in P collapses P.
  - Solving in u inflates P by 1–3 dex wherever the solved region reaches low β, where Bifrost B is not force-free.
  - Remedy: solve only below `z_cut` (default 1 Mm) and use column HSE above. A full cube then takes ~4 min.
- **`poisson-tau`, then `poisson-tau-direct` (τ grid)**, for inversion output.
  - The fixed-point version lifted strong-field columns, so it was replaced by **one Gauss–Newton/LM solve**.
  - Unknowns: ln P below `logtau_cut` plus one height per column.
  - Heights come from the opacity integral; the Jacobian includes the Lorentz–height coupling.
  - Preconditioner: per Fourier mode, the diagonalised layer-median operator.
- Start state: column HSE in τ, with top closure P = gτρ/χ and alignment of P + B²/2μ₀. There is no external pressure
  reference.

## 8. Five run modes, stopping policy and documentation (24 Oct)

- `main` fast-forwarded to `main-modes`. There is one entry point, `scripts/run.jl cfg.toml`, with MPI plus threads in
  every mode: `column-hse-z`, `column-hse-ltau`, `mhs-poisson-z`, `mhs-poisson-ltau`, and `full-mhs-z` (the M5i
  driver).
- `Slab.jl` provides x-slab MPI: periodic halos, exact distributed quantiles, a distributed 2-D FFT and cooperative
  HDF5 writes (no parallel HDF5).
- **Stopping policy (user): Φ only.** The Poisson modes stop when ΔΦ/Φ < `rtol` on `n_consec` = 3 consecutive steps.
  The ltau height shortcut and `poisson.dz_tol_m` were removed (813a3af). Heights are diagnostics, not stop criteria.
  Exit status ≠ convergence ≠ physical equilibrium.
- Root README and ten mode PDFs (`docs/physics`, `docs/technical`) were audited against the code (Codex brief, Claude
  report). Open code issues from that audit are listed in `old/CLAUDE_REPORT_README_DOCS_2026-10-24.md`.

## 9. Embedding API and production (24–26 Oct)

- `src/Embedded.jl`: an in-memory API for the four fixed-B modes, for FFNOInversion. The caller owns MPI and threads;
  agreement is collective; physics failures come back as status, never as a crash.
- Merged to `main` (PR #1). Later fixes: the τ-grid column HSE has no cap on the rise of ln P, and Heun steps are
  limited by the slope at both ends.
- **`main` and `embedding-api` are in production for FFNOML; experiments go on new branches.**

## 10. τ-cut study (26 Oct, branch `ltau-cut-study`)

`mhs-poisson-ltau` with cuts at −4, −5, −6 and none, against the Bifrost truth mapped to τ:

| run | error rms of z(τ=1) |
|---|---|
| column HSE | 85 km |
| cut −4 | 27 km |
| cut −5 | 40 km |
| cut −6 | 1137 km |
| no cut | 20 Mm |

With cut −4 the Wilson depression is −78 km, against −74 km in Bifrost. Below β ≈ 1 the given B is not force-free, and
the pressure absorbs the Lorentz force. **Keep −4: cut where the median β ≳ 10.** Report:
`HSEMHS/reports/ltaucut/ltau_cut_study.pdf`.

---

## 11. Decisions still in force

| Topic | Decision | Source |
|---|---|---|
| Modes | Four fixed-B modes are production; `full-mhs-z` is stale and will be deprecated (user, 26 Oct) | §8, §9 |
| Pressure reference | z modes: bottom anchor (`column_hse`), scalar gauge (MHS); τ modes: τ itself (top closure), heights gauged by mean z(τ=1) = 0 | §2, §7 |
| Stopping | Φ only (ΔΦ/Φ < rtol on n_consec steps) | §8 |
| Solved region | z: `z_cut` 1 Mm; τ: `logtau_cut` −4 | §7, §10 |
| EOS | `cache/eos_table_fullz.h5`, `FIX_GASC_H2` on | §3 |
| I/O | No parallel HDF5; cooperative writes | §8 |
| Branches | `main`, `embedding-api` are production (FFNOML); experiments on new branches | §9 |
| Viewing angle | v1: plane-parallel τ/μ in the height modes only. Planned: μ and azimuth in all fixed-B modes, plane-parallel and inclined geometry ([`tau/`](tau/)) | D7, user 26 Oct |

## 12. Index of the original documents (`old/`)

| File | Author | Date | Role |
|---|---|---|---|
| `HSE_MHS_DESIGN_REVIEW.md` | Codex | 15 Oct | Design v1 |
| `HSE_MHS_DESIGN_CRITIQUE.md` | Claude | 15 Oct | Critique of v1 |
| `review-response.md` | Codex | 16 Oct | Annotated reply to the critique |
| `REVIEW_RESPONSE_CRITIQUE.md` | Claude | 16 Oct | Final rulings |
| `HSE_MHS_FINAL_SPEC.md` | Claude | 16–18 Oct | Binding spec and decision log (§10) |
| `HSE_MHS_IMPLEMENTATION.md` | Claude | 16 Oct | Implementation document |
| `HSE_MHS_D9_STAGGERED_ADDENDUM.md` | Claude | 16 Oct | Staggered vector-potential scheme |
| `HSE_MHS_D9b_AVERAGING_ADDENDUM.md` | Claude | 16 Oct | Face data, Φ_Q, identifiability |
| `CODEX_BRIEF_M5.md` | Claude | 17 Oct | M5 brief, with follow-ups §8–9 |
| `CODEX_REVIEW_M5.md`, `_FOLLOWUP.md`, `_SECOND_FOLLOWUP.md` | Codex | 17 Oct | M5 reviews |
| `CLAUDE_RESPONSE_CODEX_M5.md` | Claude | 17 Oct | Assessment of the M5 review |
| `CODEX_BRIEF_M5c/d/e/f/g_*.md` | Claude | 18–21 Oct | Round briefs |
| `CODEX_REVIEW_M5c/d/e/f_*.md` | Codex | 18–19 Oct | Round reviews |
| `REVIEW_M5f_2026-10-21.md` | Codex | 21 Oct | Consolidated M5f review |
| `REVIEW_M5g_2026-10-21.md` | Codex | 21 Oct | M5h full-input task handoff |
| `CODEX_BRIEF_modes_2026-10-24.md` | Claude | 24 Oct | Five modes merged |
| `CLAUDE_BRIEF_README_DOCS_2026-10-24.md` | Codex | 24 Oct | README/docs handoff |
| `CLAUDE_REPORT_README_DOCS_2026-10-24.md` | Claude | 24 Oct | README/docs completion report |
