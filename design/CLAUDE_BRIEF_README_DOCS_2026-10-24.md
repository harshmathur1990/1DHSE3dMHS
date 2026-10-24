# Claude handoff: root README and code-matched scientific documentation

Date: 24 October 2026. Inspected HEAD: `cb90a4f`. Supervisor: Codex. Implementer: Claude.

## User request and current instruction

Write a complete root `README.md` explaining the context, aims, methods, implementation and practical operation, including MPI and threads. Audit and update the PDFs in `docs/` through their TeX sources against the corresponding code. Codex has inspected the repository and prepared this brief; it has not edited the README, TeX, scientific source or existing user work, and has not launched scientific runs or tests.

**Latest user clarification overrides the earlier height-based handoff:** stopping is now based on **relative objective change, `ΔΦ/Φ`, only**. Do not present change in height of `tau500 = 1` as the current stopping requirement. Heights can remain useful output/diagnostics. Historical or optional height-based paths must be clearly distinguished from the chosen stopping policy.

Use the current code as evidence, but do not silently endorse inconsistencies between the code and this instruction. Report any necessary implementation follow-up separately. Scientific recommendations and acceptance decisions remain the user's.

## Deliverables

1. A substantive root `README.md`, usable by a reader unfamiliar with this conversation or the cluster.
2. Focused corrections/additions to the ten mode documents in `docs/{physics,technical}/`, followed by regenerated PDFs. Keep physics explanations accessible and put execution, configuration, verification and performance details in the technical documents.
3. A short completion report listing changed documentation, discrepancies resolved, commands checked, PDF build results, and unresolved code issues. Give links and concrete evidence; do not claim new numerical validation from old reports.

Preserve the untracked files already present at review time: `HSEMHS/reports/figs/`, `HSEMHS/reports/poisson_physics.pdf`, `HSEMHS/reports/poisson_physics.tex`, and `HSEMHS/scripts/m5_bifrost/ztop_logtau.py`. Inspect them if useful, but do not overwrite or remove them as part of this task.

## README content and structure

### Purpose, scope and present status

Explain that HSEMHS constructs thermodynamic atmospheres from supplied temperature and magnetic fields on geometric-height or optical-depth grids, using column hydrostatic equilibrium and progressively more coupled magnetohydrostatic approximations. State the SI convention, gravity direction, periodic horizontal boundaries, top-first vertical storage, and the significance of optical depth at 500 nm.

Lead with the five supported modes and the difference between holding B fixed and adjusting B. Do not equate an implemented mode, a successful process exit, an optimization stop and a scientifically acceptable equilibrium. Summarize present limitations from the current evidence, rather than declaring the project scientifically complete because implementation is nearly finished.

| Mode | What is fixed / solved | Principal implementation | MPI ownership |
|---|---|---|---|
| `column-hse-z` | T fixed; independent HSE columns on z | `RunColumnHSE.jl`, `ColumnHSE.jl` | 2-D x/y blocks, complete z columns |
| `column-hse-ltau` | T and B on tau fixed; HSE pressure and opacity-derived heights, with optional total-pressure alignment | `RunModes.jl`, `PoissonTau.column_hse_tau` | x slabs |
| `mhs-poisson-z` | T and B fixed on z; weighted least-squares pressure correction below z cut, HSE above | `RunModes.jl`, `PoissonZ.jl` | x slabs |
| `mhs-poisson-ltau` | T and B fixed on tau; coupled pressure and column geometry below tau cut | `RunModes.jl`, `PoissonTau.solve_tau` | x slabs |
| `full-mhs-z` | T fixed; pressure and divergence-free face B adjusted jointly | `scripts/run.jl`, `m5h_fullres.jl`, staggered MHS modules | 2-D x/y blocks |

Explain `column_hse` as the existing alias and `gate0`, `eos_table`, `screening` as auxiliary modes. The schema also lists legacy `mhs`, but `run_config` has no matching implementation branch; do not advertise it as an executable sixth solver. Explain why `full-mhs-z` must use the script entry point rather than calling `run_config` directly.

### Methods and assumptions

Give a compact equation-level overview, linking the detailed PDFs:

- Force balance `-grad P + J cross B - rho g zhat = 0`, `J = curl B / mu0`, and magnetic divergence treatment by mode.
- Height-grid HSE: anchored integration of ln P with RK4/substep refinement and a tabulated EOS. Contrast its per-column pressure anchor with the full-MHS anchor on the area-weighted mean of ln P.
- Tau-grid HSE: top closure, pressure/height integration through opacity, pressure-rise caps and total-pressure alignment assumptions; mean-height gauge versus an externally measured absolute height.
- Pressure-only Poisson-type solvers: residual normalization, fixed start-state weights, solved cut region, matrix-free Gauss–Newton/PCG, line search or LM, and FFT-based preconditioners. These minimize a residual; they do not guarantee exact equilibrium or a divergence-free transformed input field.
- Full MHS: centre thermodynamics, edge vector potential/current, face B, `B = A_ref curl Atilde + bbar zhat`, discrete divergence identity, magnetic fit/smoothness/current terms, analytic adjoints and GN products. Explain current line-search defaults versus the retained trust-region option. The default full driver uses no hard face bounds/barrier; M5g bounded experiments are separate evidence.
- Wittmann EOS/opacity table construction, interpolation and derivatives, envelope checks, status flags, fallback/hot branches, molecular-pressure correction, and known incomplete-opacity flags. Keep exact choices grounded in current configuration and table provenance. RH is an optional cross-check, not a prerequisite for every normal run.
- Clarify that sigma is a specified magnetic deviation penalty, currently uniform 5 G in the full-mode example, not automatically an observational uncertainty or a hard acceptance bound. Keep face and centre observation operators distinct.

### Installation, data and setup

Provide an actual setup chain, with expected outputs and prerequisites at each stage:

1. Julia environment, C++17 compiler for `deps/witt/build.jl`, partition-function data, MPI runtime, and optional Python/postprocessing and TeX dependencies. `Manifest.toml` records Julia **1.13.1**; `Project.toml` declares compatibility from **1.10**. Distinguish the recorded/tested environment from declared compatibility. Do not promise every supported Julia version was tested.
2. Instantiate the `HSEMHS/` environment, compile the Wittmann library for the local platform, and configure/install the MPI.jl launcher consistently with its MPI library. Give executable commands, not a reference to someone's shell history.
3. Explain input acquisition/preparation and the ignored `IO` directory: this checkout's IO is a cluster-specific symlink, not data supplied by Git. Distinguish `input-mhs-hse.nc` (177-level crop), `input-mhs-hse-fullz.nc` (496-level production input in current full-mode example), and `input-mhs-ltau.nc` (95-level example); these are dataset examples, not interchangeable names. Describe preparation scripts and their hard-coded assumptions before recommending them.
4. Explain the gate0/envelope -> full-z EOS build -> run dependency. Link `config/gate0_bifrost385_fullz.toml` and `config/eos_table_bifrost385_fullz.toml`; identify which source cube the audit needs. Cache files and output evidence are ignored and may not exist after a clone. Do not claim `build_if_missing` automatically builds the table without verifying an active code path.
5. Document per-mode input datasets, dimensions, units/attributes, geometry assumptions and output layouts. Include separate input and output examples: native input is h5py `(x,y,z)` or `(x,y,tau)`, whereas many generated outputs are h5py `(z,y,x)` or `(tau,y,x)`. Check every mode and dataset rather than assuming one universal layout.

### Reproducible run instructions: serial, threads, MPI and hybrid

Use one stated working directory, preferably:

```sh
cd HSEMHS
julia --project=. -t 8 scripts/run.jl config/modes/column-hse-z.toml
```

Then provide serial (`-t 1`), threads-only, MPI-only and hybrid examples for the five mode configs, with actual executable commands. Explain that relative paths resolve from the **process working directory**, not automatically relative to the TOML. From the root, `--project=.` selects the wrong environment; use the documented `cd HSEMHS` convention consistently. `[mpiexec -n R]` is illustrative notation, not a command to paste literally. Prefer the launcher installed for MPI.jl; explain when plain `mpiexec` is safe.

Document:

- R ranks, T Julia threads **per rank**, ranks per node, total core allocation and avoidance of oversubscription. Include placement/binding guidance appropriate to the actual launcher, with a generic portable example separate from a tested cluster-specific one.
- `THREAD_FUNNELED`, MPI on the main task outside threaded loops, complete vertical columns, minimum owned horizontal extents (at least two), and x-slab rank-count restrictions. No vertical MPI decomposition.
- Which `[parallel]` keys are honored by which path: column-hse-z passes px/py; slab modes force px=R, py=1; full driver calls `bif_problem` which currently auto-selects its process grid. Do not imply px/py tune all five modes.
- Distributed FFT transpose and main performance costs at a useful level; exact global order statistics versus floating-point reduction effects. Avoid promises of bitwise identical nonlinear solver trajectories across ranks/threads.
- Julia threads, BLAS threads and FFTW threads as distinct controls. `run_mhs_poisson_ltau` sets BLAS to `min(32, Threads.nthreads())` for the solve then resets to 1. Other paths and FFT defaults must be checked; do not give a global environment setting and claim it governs all kernels.
- Serial/cooperative HDF5 output: ranks take turns writing hyperslabs, no parallel-HDF5 dependency, with possible I/O and filesystem costs. Memory estimates must include full-MHS setup's replicated raw input arrays per rank (`bif_setup.jl`), not just distributed solver fields.
- Site-specific node-local Julia, IPoIB/libfabric, and launcher workarounds as a reproducibility appendix. `~/.julia/bin/mhs_rank.sh` and `ssh_bash_launcher` are external files missing from the repository: supply their required setup/content or label the example nonportable and dependent on an existing site installation. No implication that a fresh clone includes them.

### Configuration, stopping, output and restart

Explain current effective defaults and a compact configuration example for each mode; link the schema and checked-in configs. The full-mode wrapper forwards only a subset of generic settings to an older driver. Explicitly list ignored/not-forwarded keys instead of suggesting they work. Include how input attributes supply gravity and reference pressure, per-mode overwrite behavior, and the directory-versus-file meaning of `run.output`.

**Stopping is relative objective change only, per the latest instruction.** State the exact numerator/denominator and accepted-step/consecutive-step rule for each current path. Full mode currently computes `(phi_prev - phi_new)/phi_new` and compares its absolute value to `phi_rtol` on three accepted steps; Poisson modes compute the decrease divided by `phi_prev`, default `rtol=1e-4`, `n_consec=3`. Do not casually call these identical definitions. Explain separate PCG tolerances and work/step/wall limits.

The tau Poisson code still contains a height-based early shortcut (`nsmall >= 1 && max|delta c| < dz_tol`, default 100 m). That conflicts with the user's current phi-only policy. Identify how the documented current run disables it (the present comparison is strict `<`, so `dz_tol_m = 0` disables this branch), or propose a minimal policy correction separately if needed. Do not silently publish the existing mixed rule as the chosen policy. The retained full-mode `stop="dz"` is historical/optional, not the current run setting.

Explain history/candidate logs, groups/datasets and units, stop reasons, status/exit codes as they actually work, checkpoints, STOP file, native-versus-smoke resolution, safe restart and rank-change limitations. Clarify that a small ΔPhi/Phi or exit 0 alone does not certify acceptable force residuals, stationarity or data fidelity. Give a short post-run interpretation checklist and how to find the saved progress.

Full restart currently checks dimensions and recomputed tau1 heights but does not establish identical scientific settings solely by that check. Document changes in input/EOS/sigma/solver policy as changed experiments unless equivalence is explicitly verified. Existing output paths can be overwritten/appended by the full driver despite generic `overwrite=false`; explain safe use of fresh run directories.

### Tests, evidence, navigation

Give the actual test command, Python oracle dependencies and `HSEMHS_PYTHON` override, MPI subprocess behavior on 2/4/6 ranks, and how `HSEMHS_SKIP_MPI=1` reduces coverage. Do not call a skip full MPI validation. Identify current automated coverage versus historical manual comparison logs, especially for recently ported slab/Poisson modes.

Link the ten PDFs and their TeX sources in a five-mode table. Explain `bash docs/build.sh`, its `pdflatex`/package requirements and generated files. Add a short repository map distinguishing active library/driver/config, reports, ignored outputs/cache/IO, legacy experimental scripts and historical `design/` material. Record numerical performance with dated commit/input/grid/rank/thread/stop information and artifact links. Keep claims of scientific convergence separate from timings and smoke completion. Only state licensing/citation/attribution supported by repository evidence; flag missing policy rather than inventing a license for vendored code.

## Specific documentation findings to resolve

These come from code inspection and text extraction of all ten existing PDFs (26 pages), not from newly executed numerical validation. Review both the TeX and regenerated PDFs after making the changes.

| Priority / affected documents | Evidence and concrete action |
|---|---|
| High: all solver technical documents and README | `RunModes.jl` returns `converged` after both Poisson solvers without inspecting termination; those solvers can break for no decrease or reach their step cap. `scripts/run.jl` also returns 0 after the full driver returns. State that process success is not numerical convergence; document how to inspect stop reason and flag structured termination/status propagation as a separate code follow-up. Do not imply exit 0 proves the Phi criterion was met. |
| High: `technical/mhs-poisson-ltau.tex`, stopping explanations | Current `PoissonTau.solve_tau` includes the 100 m offset shortcut. Current user instruction is Phi-only. Correct the documented chosen configuration and clearly distinguish current instruction, available legacy switches and actual source behavior. `max|delta c|` is the cut-level offset change; it is **not** the actual maximum change in tau1 height when pressure-dependent layer thicknesses also change. |
| High: `physics/full-mhs-z.tex` sections 2/4/11 | The table says prescribed mean pressure, but `MHSStaggered.anchor_mean` and the state map constrain the area-weighted mean of **ln P**. Call this a geometric-mean pressure anchor; it is not arithmetic mean P. Replace categorical statements that every moving snapshot necessarily has no fixed-B equilibrium, or that changing B guarantees one. The curl compatibility relation motivates an obstruction but must be checked for a particular field. |
| High: `physics/column-hse-ltau.tex` sections 5/8/9/10 and its technical companion | No externally supplied P_ref does not mean no boundary assumption: top closure `P_top = g tau_top/kappa_top` and the height-alignment prescription supply modeling choices. The pressure-rise cap can interrupt HSE integration. In `hse_tau!`, a capped interval also breaks before integrating its remaining geometric thickness. Describe capped results honestly, expose counts/location/monotonic-height implications, and flag investigation of affected intervals as a numerical follow-up. Remove unqualified claims that every column is exactly HSE or that the pressure is guaranteed to return to the HSE value after a few levels. |
| Medium: `physics/column-hse-ltau.tex` total-pressure alignment | Equal total pressure at one chosen height neglects magnetic tension and is an alignment approximation, not a general consequence of static MHD. `align_ptot!` interpolates log total pressure and extrapolates it with a gas-pressure scale height; label that approximation. Wilson depression is an expected tendency under stated conditions, not guaranteed by every arbitrary input/closure. |
| Medium: `physics/mhs-poisson-z.tex` sections 4/5/7, technical anchor description | The domain is horizontally periodic with finite vertical boundaries; Helmholtz uniqueness/orthogonality needs boundary/gauge assumptions. The implemented normalized, weighted nonlinear least squares is not simply an exact unweighted Helmholtz projection. The bottom-plane anchor is a rank-one penalty on the step, not the full solver's exact reparameterization; subsequent elementwise clipping can also alter its mean. Qualify claims of exact removal of the gradient part and exact preservation of the pressure level. |
| Medium: both tau-grid physics/technical pairs | Moving the geometry while keeping input B values fixed does not impose discrete div B=0. Explain which magnetic constraints are absent, and why the full-mode curl identity cannot be transferred to tau-grid modes. Describe fixed layers above the cut, the global height gauge, and what is and is not determined by force balance. |
| Medium: `technical/column-hse-ltau.tex` input contract | `read_tau_input` converts coordinate units but directly reads T/B values and hard-codes primary T/B dataset names. Its comment that SI applies unless a dataset has units is too broad if it suggests B in G or non-SI T is converted. Document actual support, aliases/layout branch and unsupported generic `[input]` mappings. `issorted(lt)` is non-strict and dx/dy use only first spacing; do not claim strict-increase/uniform-grid validation that the source does not perform. State the numerical requirement separately and flag missing validation. |
| Medium: `physics/full-mhs-z.tex` objective; technical normalizations | The explanatory brackets call all terms volume-weighted averages, whereas the face data term is a count-normalized sum and other terms have different N/N_F/N_B and geometric weights. Label the equation schematic or give term-specific definitions. State the pressure sign convention and omitted boundary force equations. Do not portray finite weighted least squares as an exact equilibrium guarantee. |
| Medium: `technical/mhs-poisson-ltau.tex` LM update | Text says damping decreases after a full **scaled** step, but source checks `alpha == 1.0`, not `alpha == sc`. When whole-step scaling is active, these differ. Describe the actual rule, and flag whether it was intended as a separate code question rather than modifying the algorithm in a documentation task. |
| Medium: `technical/full-mhs-z.tex`, root run instructions | Clarify direct-driver defaults (5 percent sigma, trust region, dz stop) versus wrapper/config defaults (uniform 5 G, no trust region, Phi stop). The module-level ENV constants are read at include time. Paths, generic settings and overwrite/restart behavior differ by entry point. Full driver assumes the dataset-specific 504x504 native horizontal grid. |
| Medium: all technical verification sections | Scope reproducibility statements: exact quantiles are not exact global sums or identical nonlinear trajectories. The reported tau-Poisson crop differences reach 150 m and large runs have capped PCG residuals. Label measured ranges and unresolved weakly constrained columns. Supply source logs for benchmark tables; reported results are not fresh confirmation. |
| Medium: `physics/column-hse-z.tex` opacity discussion | Clarify that tau_top need not be zero under exp_tail/half_cell closure. Statements about spectral lines forming at a fixed optical depth and visible-surface interpretation need the normal contribution-function/formation-range qualification; retain an accessible explanation without claiming every spectral diagnostic samples one universal tau500 surface. |

The existing technical documents already explain many algorithms and FFT details. Extend and correct them instead of copying the historical design documents wholesale. Add a common assumptions/limitations and convergence/status reference where it prevents conflicting definitions across ten PDFs.

## Implementation and validation instructions

Proceed with the documentation work and locally reproducible documentation/command checks. This handoff does **not** request new production solves, broad solver tuning, changes to the observation contract, or unreviewed numerical-method changes.

- Trace each claimed config key through `Config.jl` -> `scripts/run.jl`/`Driver.jl` -> the selected driver. Keep default, example override and historical experiment separate.
- Check commands from the stated working directory and validate minimal examples where resources are available. Document missing IO/cache/site tools as prerequisites, not algorithm failures. Do not start full-sized runs merely to validate a README command.
- Build all ten PDFs from the edited TeX with `docs/build.sh`; inspect page layout, tables, equations, links and warnings as well as build exit status. Check PDF/TeX consistency and fix broken paths to `../preamble.tex` or newly linked resources.
- Keep code-level discrepancies in a short separate follow-up list if documentation cannot resolve them. User decisions govern whether to change scientific implementation.
- In the completion report, state exactly what was executed and what evidence was reused. Leave a concrete, reviewable documentation result.

## Audit scope

Inspected the five configs and mode README, entry point/schema/dispatch, HSE and Poisson driver/solver portions, full-MHS setup/driver, parallel/slab infrastructure, EOS build path, test launcher and all ten TeX/PDF documents. PDFs were read using local PDFKit text extraction (3/3/3/3/4 pages for the physics set and 2 pages per technical document). No new scientific run or test was performed. This is a documentation-supervision brief, not a complete correctness certification of the recently merged Poisson implementations.
