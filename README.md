# HSEMHS: hydrostatic and magnetohydrostatic solar atmospheres

HSEMHS builds thermodynamic atmospheres (gas pressure, density, electron density, optical depth at 500 nm and, on
optical-depth grids, geometric heights) from a supplied **temperature** and **magnetic field**. The input can be on a
geometric-height grid (a simulation snapshot such as Bifrost) or on an optical-depth grid (the output of a Stokes
inversion). The physics goes from independent hydrostatic columns to progressively more coupled magnetohydrostatic (MHS)
approximations. The Julia package lives in `HSEMHS/`; this file is the entry point.

**Status (24 Oct 2026).** All five modes below are implemented, run with MPI + threads and are documented. That is not
the same as scientifically converged results: see [Stopping, status and what "finished" means](#stopping-status-and-what-finished-means)
and [Known limitations](#known-limitations). Of the three iterative modes, only `mhs-poisson-ltau` has met its
stopping rule on the full production cube (under a superseded rule); `mhs-poisson-z` reached its step limit while still
decreasing, and `full-mhs-z` has not reached its stopping rule in any production run.

## Conventions

* SI units internally and in all outputs (Pa, kg m⁻³, K, T, m). Gravity points along −ẑ, `g = 274 m s⁻²` for the Sun.
* `z` grows upward, but **vertical arrays are stored top level first** (z strictly decreasing; τ increasing).
* Periodic boundaries in `x` and `y`; finite top and bottom.
* Optical depth `τ₅₀₀` is the extinction (absorption + scattering) at 500 nm integrated downward from the top; the level
  `τ₅₀₀ = 1` is the photosphere seen in the continuum. Spectral lines form over ranges of optical depth (their contribution
  functions), so `τ₅₀₀ = 1` is a reference surface, not the formation height of every diagnostic.

## The five modes

The basic equation is the static force balance

    −∇P + J×B − ρ g ẑ = 0,   J = ∇×B / μ₀,   ρ = ρ(T, P) from the equation of state.

The modes differ in what is held fixed and what is solved:

| `run.mode` | Fixed | Solved | Principal code | MPI ownership |
|---|---|---|---|---|
| `column-hse-z` | T (on z) | P in each column independently (dP/dz = −ρg), anchored at one level | `src/RunColumnHSE.jl`, `src/ColumnHSE.jl` | 2-D x/y blocks, complete z columns |
| `column-hse-ltau` | T, B (on τ) | P(τ) and heights z(τ) per column from the opacity; optional total-pressure alignment | `src/RunModes.jl`, `src/PoissonTau.jl` (`column_hse_tau`) | x slabs |
| `mhs-poisson-z` | T, B (on z) | P by weighted least squares of the force residual below `z_cut`; HSE above | `src/RunModes.jl`, `src/PoissonZ.jl` | x slabs |
| `mhs-poisson-ltau` | T, B (on τ) | P(τ) and the column geometry jointly below the τ cut; τ-HSE above | `src/RunModes.jl`, `src/PoissonTau.jl` (`solve_tau`) | x slabs |
| `full-mhs-z` | T (on z) | P and a divergence-free face field B jointly (staggered vector potential) | `scripts/run.jl` → `scripts/m5_bifrost/m5h_fullres.jl`, `src/MHSStaggered.jl`, `src/MHSJointGN.jl` | 2-D x/y blocks |

The `mhs-poisson` modes keep **B fixed**: they can reduce, but in general not remove, the force imbalance (a fixed field
cannot hold up every horizontal density contrast). Only `full-mhs-z` **adjusts B**. Moving the τ surfaces in the τ-grid
modes while keeping the B values on them does not enforce a discrete ∇·B = 0; only `full-mhs-z` has ∇·B = 0 exactly
(B = curl A on a staggered grid).

Other `run.mode` values: `column_hse` (alias of `column-hse-z`), and the auxiliary modes `gate0` (input audit and EOS
envelope), `eos_table` (builds the EOS/opacity table) and `screening`. The schema also lists a legacy `mhs`, which has no
implementation in `run_config`; it is not a sixth solver.

## Methods in brief

Detailed derivations: the physics PDFs in [`docs/physics/`](docs/physics); implementation, configuration, verification
and timings: [`docs/technical/`](docs/technical).

* **Height-grid HSE** (`column-hse-z`): `d ln P/dz = −1/H(T,P)` per column, RK4 with substep doubling (n vs 2n substeps,
  tolerance 1e-6 in ln P, up to 64), `ln T` linear in z between levels; one pressure `P_ref` prescribed at the anchor level
  (bottom by default) of **every column**.
* **τ-grid HSE** (`column-hse-ltau`): `d ln P/d ln τ = gτρ/(χP)`, `dz/d ln τ = −τ/χ`, Heun substeps; top closure
  `P_top = g τ_top/κ_top` (κ = χ/ρ constant above the top level). With P = g m this is d ln m/d ln τ = τ/(κm) and
  dz = −dm/ρ, so heights strictly decrease; the rise of ln P is not limited (below a hot-over-cool interface the cool
  layer's low opacity genuinely needs a large column mass per unit τ), and a column whose pressure would leave the EOS
  table is reported as failed rather than altered. (Until `embedding-api` the rise was capped at `tau_grid.cap_per_dex`,
  which left capped intervals with zero thickness.) Columns are
  aligned by equal `P + B²/2μ₀` at one height (an approximation: magnetic tension neglected), then a global gauge sets
  the mean height of τ = 1 to zero. No externally supplied reference pressure is needed, but the top closure and the
  alignment are modelling choices, and absolute heights are relative to the gauge, not measured.
* **Pressure-only Poisson-type solvers** (`mhs-poisson-z`, `mhs-poisson-ltau`): minimise a weighted sum of squares of
  the relative force residual `R = F/P` (weights from the start state, fixed), in the region below the cut; matrix-free
  Gauss–Newton with PCG, a backtracking line search (z) or Levenberg–Marquardt with whole-step scaling (τ), and
  preconditioners built per horizontal Fourier mode (a distributed FFT). They minimise a residual; they do not guarantee
  an exact equilibrium.
* **Full MHS** (`full-mhs-z`): P, ρ, T at cell centres; vector potential and currents on edges; face field
  `B = A_ref curl Ã + b̄ ẑ` (∇·B = 0 to round-off by construction). Φ = force term + data term (face B against the input
  field, in units of σ) + smoothness of the field change + a grid-scale current penalty; analytic adjoints and
  Gauss–Newton Hessian products. The pressure gauge is the **area-weighted mean of ln P** over the bottom plane (a
  geometric-mean pressure, not an arithmetic mean). Default step control: backtracking line search; a trust-region variant
  is retained. No hard face bounds or barrier in the default driver (those were separate M5g experiments).
  σ is a specified penalty scale on the field change (uniform 5 G in the mode config), not an observational uncertainty
  and not a hard acceptance bound.
* **Equation of state and opacity:** Wittmann's LTE EOS (C++, `wittman-eos/`, wrapped through `HSEMHS/deps/witt`) with
  solar abundances, H₂, an H/He Saha branch above a switch temperature, and the 5000 Å continuum extinction, tabulated in
  (ln T, ln P) with bicubic Hermite interpolation and derivatives (`src/eos/`). The table records its envelope; calls
  outside it return an error status. Build provenance is in `config/eos_table_bifrost385_fullz.toml`.

## Installation

Recorded environment: Julia **1.13.1** (`HSEMHS/Manifest.toml`); `Project.toml` declares compatibility from 1.10, but
only 1.13.1 has been used. Prerequisites: a C++17 compiler (for the EOS library), an MPI runtime compatible with MPI.jl
(the default MPICH_jll is used here), Python 3 with `numpy` and `h5py` for the test oracles and post-processing, and a
TeX installation (`pdflatex` with `extarticle`, `tikz`, `tabularx`, `lmodern`, `titlesec`) for the documents.

```sh
cd HSEMHS
julia --project=. -e 'using Pkg; Pkg.instantiate()'            # Julia dependencies
julia --project=. deps/witt/build.jl                           # compiles deps/witt/libwitt.{so,dylib} for this platform
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'      # installs ~/.julia/bin/mpiexecjl matching MPI.jl's MPI
julia --project=. -e 'using HSEMHS'                            # precompiles; must finish without error
```

`deps/witt/build.jl` patches the verbatim Wittmann source into `witt_shim.cpp` by counted replacements and compiles it;
the partition-function file is `wittman-eos/pf_Kurucz.input` (`eos.pf_file`, path relative to `HSEMHS/`).

## Data and the EOS table

`IO/`, `HSEMHS/cache/` and `HSEMHS/out/` are **not in git** (`.gitignore`). In the original checkout `IO` is a symlink to
a cluster directory; after a fresh clone you must supply the input cubes and build the table.

| File (in `IO/`) | Grid | Used by | Origin |
|---|---|---|---|
| `BIFROST_en024048_hion_0_504_0_504.nc` | 504 × 504 × 496, −2.44 … 14.37 Mm | `gate0` | Bifrost en024048_hion snapshot 385 |
| `input-mhs-hse.nc` | 504 × 504 × 177, −0.49 … 2.98 Mm | `column-hse-z`, `mhs-poisson-z` | `make_input_mhs_hse.py` (default crop) |
| `input-mhs-hse-fullz.nc` | 504 × 504 × 496 | `full-mhs-z` | `make_input_mhs_hse.py` over all levels (`crop_source_levels` 0–495) |
| `input-mhs-ltau.nc` | 504 × 504 × 95, log τ −8 … 1.4 | τ modes | supplied by the user (no script in the repository) |

`make_input_mhs_hse.py` (repository root) has hard-coded default paths (`--src`, `--out`) from the original machine and
a default crop `--zmin -0.5e6 --zmax 3e6`; it writes `x, y, z` (m), `temperature` (K), `B_x, B_y, B_z` (T) with units and
dimension scales, and the attributes `P_ref_Pa`, `gravity_m_s2`, `period_x_m`, `period_y_m`.

EOS table chain (the table `cache/eos_table_fullz.h5` is referenced by every mode config):

```sh
cd HSEMHS
julia --project=. -t 32 scripts/run.jl config/gate0_bifrost385_fullz.toml       # audit of the raw Bifrost cube → T, P envelope
julia --project=. -t 32 scripts/run.jl config/eos_table_bifrost385_fullz.toml    # builds cache/eos_table_fullz.h5 from that envelope
```

`eos.build_if_missing` exists in the schema but no code path builds the table automatically.

### Input contracts per mode

* **Height grid** (`column-hse-z`, `mhs-poisson-z`): `x`, `y`, `z` datasets (z strictly decreasing, x and y increasing),
  `temperature`, `B_x`, `B_y`, `B_z` with `units` attributes (m/km/cm/Mm; K; T/G) and `DIMENSION_LIST` dimension scales
  (netCDF4 style); h5py shape (x, y, z). Periodicity from the global attributes `periodic_x/_y` or `period_x_m/_y_m`.
  `P_ref = "from_input"` reads `P_ref_Pa`; `physics.g = "from_input"` reads `gravity_m_s2`.
* **τ grid** (τ modes): `log10tau500` (n_τ, increasing, top first), `x` (NX), `y` (NY), `temperature`, `B_x`, `B_y`,
  `B_z`, h5py shape (x, y, τ). Units: `x`, `y` honour a `units` attribute; **T and B are read as K and T without
  conversion**, and their dataset names are fixed (`input.temperature`, `input.bx` … are not used by these modes).
  The code checks that log τ is non-decreasing and takes dx, dy from the first spacing; it assumes but does not verify
  strictly increasing, uniform coordinates. The file has no attributes, so set a numeric `physics.g`.
* **Full MHS** (`full-mhs-z`): the height-grid contract plus the attributes `P_ref_Pa`, `period_x_m`, `period_y_m`,
  `gravity_m_s2`; `full_mhs.run = "run"` requires the 504 × 504 horizontal grid of the production cube.

## Running

Work from `HSEMHS/` (relative paths in the configs resolve from the **process working directory**, not from the TOML
file; from the repository root `--project=.` would select the wrong environment):

```sh
cd HSEMHS
julia --project=. -t 1  scripts/run.jl config/modes/column-hse-z.toml                     # serial
julia --project=. -t 32 scripts/run.jl config/modes/mhs-poisson-ltau.toml                 # threads only
~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. -t 1  scripts/run.jl config/modes/mhs-poisson-z.toml    # MPI only
~/.julia/bin/mpiexecjl --project=. -n 4 julia --project=. -t 16 scripts/run.jl config/modes/column-hse-ltau.toml  # hybrid
```

Configs for all five modes: [`HSEMHS/config/modes/`](HSEMHS/config/modes) (`column-hse-z.toml`, `column-hse-ltau.toml`,
`mhs-poisson-z.toml`, `mhs-poisson-ltau.toml`, `full-mhs-z.toml`). `mpiexecjl` uses the MPI library MPI.jl was built
with; a system `mpiexec` is safe only if MPI.jl is configured for that same MPI (`MPIPreferences`).

**Ranks and threads.** `-n R` ranks, `-t T` Julia threads **per rank**; keep R × T at or below the cores you have on each
node (R/nodes ranks per node). MPI runs as `THREAD_FUNNELED`: all MPI calls are on the main task, outside threaded
loops. Columns are never split (no vertical decomposition); every rank owns at least two columns per decomposed axis; the
slab modes need NX/R ≥ 2.

| Mode | Decomposition | `[parallel] px, py` |
|---|---|---|
| `column-hse-z` | 2-D blocks | honoured (0 = `MPI.Dims_create`) |
| `column-hse-ltau`, `mhs-poisson-z`, `mhs-poisson-ltau` | x slabs, px = R, py = 1 (forced) | ignored |
| `full-mhs-z` | 2-D blocks chosen by `bif_problem` | ignored |

**Other thread controls.** BLAS: one thread by default (set in `HSEMHS.__init__`); `mhs-poisson-ltau` raises it to
`min(32, T)` for its solve and resets it to 1. FFTW: single-threaded plans, applied from Julia threads (FFTW's own
threading crashes under Julia 1.13). No global environment variable governs all kernels.

**Distributed operations (slab modes).** x-halo planes via `MPI.Sendrecv!`; global sums and dot products by `Allreduce`
(summation order differs with the rank count, so results agree to rounding, and iterative solvers can amplify that);
layer medians and quantiles are exact global order statistics (bisection on order-preserving integer keys, no gathering);
the FFT preconditioners use rfft along y, an `Alltoallv` transpose to y-wavenumber slabs, and an FFT along x.

**Output** is written without parallel HDF5: rank 0 creates each file, then ranks write their hyperslabs in turn. This
costs time on large outputs (the compressed column-HSE output takes minutes). Memory: the slab modes hold only their slab;
`full-mhs-z` setup (`bif_setup.jl`) reads the **complete** raw input arrays on every rank before distributing them.

**Multi-node on hyades (site-specific, not portable).** Used for the timings below; depends on files outside the
repository (`~/.julia/bin/mhs_rank.sh`, `~/.julia/bin/ssh_bash_launcher`, a node-local Julia in `/tmp/julia-1.13.1`):

```sh
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
printf "hyades4:2\nhyades5:2\nhyades15:2\nhyades21:2\n" > hosts
mpiexecjl --project=. -launcher ssh -launcher-exec $HOME/.julia/bin/ssh_bash_launcher -f hosts -n 8 \
    mhs_rank.sh julia --project=. -t 32 scripts/run.jl config/modes/mhs-poisson-ltau.toml
```

`mhs_rank.sh` sets `FI_PROVIDER=tcp` and `FI_TCP_IFACE` to the node's IPoIB (`ib*`) interface (the default libfabric
choice fails in `MPI_Init`) and puts the node-local Julia first on `PATH` (concurrent start-up from the shared filesystem
failed with "failed to map segment"). `ssh_bash_launcher` runs the remote command through bash because the login shell is
zsh. Only nodes with an `ib*` interface work.

## Configuration

Every key and its default is declared in [`HSEMHS/src/Config.jl`](HSEMHS/src/Config.jl) (`CONFIG_SCHEMA`); unknown
sections or keys are rejected. Keys used per mode:

| Mode | Honoured | Notes |
|---|---|---|
| `column-hse-z` | `[input]` (names, `crop_z_m`), `physics.g`, `[eos]`, `[column_hse]`, `[tau]`, `[parallel]`, `output.compression`, `run.overwrite` | |
| `mhs-poisson-z` | as `column-hse-z` except `[parallel]` and `output.compression`; plus `poisson.{z_cut_m, max_steps, rtol, n_consec, cg_tol, cg_maxit, anchor_w}` | output uncompressed |
| `column-hse-ltau` | `input.{file, x, y}`, `physics.g`, `eos.{table_file, pf_file}`, `tau_grid.align`, `run.overwrite` | other `[input]` keys ignored |
| `mhs-poisson-ltau` | as `column-hse-ltau`, plus `poisson.{logtau_cut, max_steps, rtol, n_consec, cg_tol, cg_maxit, dc_cap_m, lm_mu0}` | |
| `full-mhs-z` | `input.file`, `eos.table_file`, `run.output` (a **directory**), `full_mhs.*` | everything else is not forwarded: `physics.g`, `[column_hse]`, `[mhs]`, `[parallel]`, `[tau]`, `[output]`, `run.overwrite`, input dataset names; P_ref and g come from the file attributes |

`full-mhs-z` runs the older M5 driver: `scripts/run.jl` validates the config, exports `MHS_INPUT`, `MHS_EOS_TABLE`,
`MHS_SIGMA_REL`, `MHS_TRUST`, `MHS_STOP`, `MHS_PHI_RTOL` (read as constants when the driver is included), includes
`m5h_fullres.jl` and calls its `main`. That is why this mode cannot go through `run_config`. Run directly,
`m5h_fullres.jl` has different defaults (5 % relative σ, trust region, height stop: the M5h configuration); the mode
config defaults are the M5i choice (uniform 5 G, line search, Φ stop).

## Stopping, status and what "finished" means

**Stopping policy: relative change of the objective Φ only.** Heights (z(τ=1), column offsets) are computed and logged as
diagnostics but do not stop any mode in its documented configuration.

| Mode | Rule | Limits |
|---|---|---|
| `column-hse-z`, `column-hse-ltau` | none (direct integration) | — |
| `mhs-poisson-z` | (Φ_prev − Φ_new)/Φ_prev < `rtol` (1e-4) on `n_consec` (3) consecutive accepted steps | `max_steps` (100); stops early if no decrease is found along a step |
| `mhs-poisson-ltau` | same definition and defaults | `max_steps` (100); stops if no damping level gives a decrease |
| `full-mhs-z` | \|Φ_prev − Φ_new\|/Φ_new < `phi_rtol` (1e-6) on 3 consecutive accepted steps | wall time, `STOP` file, line-search failure (30 halvings); `stop = "dz"` (height) is a retained historical option |

Each linear solve has its own PCG tolerance and iteration cap; reaching the cap does not stop the outer iteration.

**Exit status is not numerical convergence.** `run_config` exits 0 for `converged`, 2 `stalled`/`max_iter`, 3 EOS
failure, 4 invalid input; but the two Poisson drivers currently return `converged` whatever their termination, and
`scripts/run.jl` returns 0 after the full driver returns. Read the log: Poisson runs print `converged: …` when the rule was
met, `no decrease …` when they stopped otherwise, and nothing when `max_steps` ran out; the full driver logs
`solver returned: <status> (<reason>)`. Even a met Φ rule does not certify small force residuals, stationarity or data
fidelity. After a run check: the stop line; the residual per band/layer in the log or output; for `full-mhs-z` the data
term and the largest field deviations in `history.csv`; and z(τ=1) and the pressure change for plausibility.

**full-mhs-z run directory:** `history.csv` (one row per accepted step: Φ and its terms, z(τ=1) change, residual and
data statistics, step data), `candidates.csv` (every trial), `init.h5`, checkpoints `atmosphere_A.h5`/`atmosphere_B.h5`
(alternating, every `ckpt_hours`), `atmosphere_final.h5`, `ztau1/`. Creating a file `STOP` in the directory stops it at
the next candidate (after a save). `full_mhs.run = "smoke"` runs every code path on a 42 × 42 grid (every second level).
**Restart** (`full_mhs.restart = <file>`) works with another rank count; it checks the grid and the recomputed z(τ=1) map,
not that input, EOS, σ or solver settings are the same: a changed setting is a new experiment. The driver writes into an
existing directory regardless of `run.overwrite`; use a fresh directory per run.

## Outputs

| Mode | Groups and datasets | Array order |
|---|---|---|
| `column-hse-z` | `/input` (copies), `/output/{Pgas, rho, ne, Pe, nH_total, mu_bar, colmass, colmass_trap, chi500, tau500, logtau500}`, `/diagnostics/{R_x, R_y, R_z, eos_status, column_status, tau_top_method, rk_max_substeps, z_tau500_1, layer tables, z}`, `/meta` | h5py (x, y, z), as the input |
| `mhs-poisson-z` | `/atmosphere/{Pgas, rho, T, tau500, Bx_centre, By_centre, Bz_centre, Pgas_column_hse, z_tau1_m}`, `/coords/{x_m, y_m, z_m}` | h5py (z, y, x) |
| `column-hse-ltau`, `mhs-poisson-ltau` | `/tau_grid/{logtau500, z_m, Pgas, rho, T, Bx, By, Bz, z_tau1_m}`, `/coords/{x_m, y_m}` | h5py (τ, y, x) |
| `full-mhs-z` | checkpoints and final file: `/atmosphere/{Pgas, T, rho, Bx_xface, By_yface, Bz_zface, Bx_centre, By_centre, Bz_centre, tau500, z_tau1_m, z_tau1_flag}`, `/solver/{x1…x5, ulo}` (restart state), run attributes | h5py (z, y, x) |

## Tests

```sh
cd HSEMHS
HSEMHS_PYTHON=$(which python3) julia --project=. -t 16 test/runtests.jl
```

The Python oracle (`oracles/python/`, needs `numpy`, `h5py`) generates fixtures; `HSEMHS_PYTHON` selects the
interpreter. The suite covers units/config, operators, parallel infrastructure, the reader and writer, the EOS (Gates 1–2,
including bitwise thread/rank invariance of column HSE), staggered MHS gradients and invariance, Gate C synthetic tests
and the M5 solver tests, and runs `test/mpi/mpi_tests.jl` on 2, 4 and 6 ranks via `MPI.mpiexec`.
`HSEMHS_SKIP_MPI=1` skips the multi-rank part (reduced coverage, not MPI validation). The slab infrastructure and the two
Poisson modes have **no automated tests yet**; their checks were manual comparisons (1 vs 4 ranks, against the original
scripts), recorded in the technical PDFs.

## Measured performance

Not fresh validation: timings of single runs, with their code state. Logs are on hyades in
`/mn/stornext/d9/data/harshm/MHS_modes/HSEMHS/out/modes/` (`<mode>.log`, `run_all.sh`) and
`…/out/modes_test/`. Multi-node runs used 4 nodes (hyades4, 5, 15, 21), 8 ranks × 32 threads.

| Date / commit | Mode | Input | Layout | Wall | Termination |
|---|---|---|---|---|---|
| 24 Oct / 13666b7 | `column-hse-ltau` | input-mhs-ltau.nc (spacing from the then `input.dx_m`) | 8 × 32 | 95 s | direct |
| 24 Oct / 13666b7 | `mhs-poisson-ltau` | same | 8 × 32 | 8.3 min | 17 steps, previous single-step rule |
| 24 Oct / 13666b7 | `column-hse-z` | input-mhs-hse.nc | 8 × 32 | 5.3 min (13 s compute, rest compressed output) | direct |
| 24 Oct / 13666b7 | `mhs-poisson-z` | input-mhs-hse.nc, z_cut 1 Mm | 8 × 32 | 4.7 min | step limit (then 60), ΔΦ/Φ 1.8e-4 per step |
| 24 Oct / b96bb4e | `mhs-poisson-ltau` (script) | input-mhs-ltau.nc | 1 × 96 | 14 min | 17 steps |
| 24 Oct / 0dad96a | `full-mhs-z` smoke | input-mhs-hse-fullz.nc (42 × 42 × 249) | 4 × 8, one node | 97 s | 3 accepted (smoke) |
| 22–23 Oct / c96a922 | M5i (`full-mhs-z` settings) | input-mhs-hse-fullz.nc | 24 × 16 | stopped on request | 112 accepted, Φ 7.88e3 |

The current stopping rule (Φ-only, 3 consecutive) has been run only on 96² crops so far.

## Known limitations

* `mhs-poisson-z` converges slowly (Gauss–Newton on a large residual that a fixed field cannot remove); low-β layers
  inflate the pressure, hence `z_cut`.
* `mhs-poisson-ltau`: the PCG reaches its iteration cap in every step on the full cube; a few strong-field column heights
  are weakly determined (runs agree to ~1 m median, 12–24 m at the 99th percentile, up to 150 m on a 96² crop).
* `full-mhs-z`: no production run has met its stopping rule; a run on the full cube takes more than a day on 4 nodes.

**Code follow-ups (not changed by the documentation work):** propagate the Poisson solvers' termination (converged / no
decrease / step limit) into the exit status, and the full driver's status into `scripts/run.jl`; in `solve_tau` decide whether the damping should also be lowered after a
full *scaled* step (`α == sc`, currently only `α == 1`); validate strictly increasing, uniform x, y and log τ in the τ
reader; automated tests for the slab layer and the Poisson modes.

## Documentation

| Mode | Physics (B.Sc.-level derivations) | Implementation |
|---|---|---|
| `column-hse-z` | [PDF](docs/physics/column-hse-z.pdf) · [TeX](docs/physics/column-hse-z.tex) | [PDF](docs/technical/column-hse-z.pdf) · [TeX](docs/technical/column-hse-z.tex) |
| `column-hse-ltau` | [PDF](docs/physics/column-hse-ltau.pdf) · [TeX](docs/physics/column-hse-ltau.tex) | [PDF](docs/technical/column-hse-ltau.pdf) · [TeX](docs/technical/column-hse-ltau.tex) |
| `mhs-poisson-z` | [PDF](docs/physics/mhs-poisson-z.pdf) · [TeX](docs/physics/mhs-poisson-z.tex) | [PDF](docs/technical/mhs-poisson-z.pdf) · [TeX](docs/technical/mhs-poisson-z.tex) |
| `mhs-poisson-ltau` | [PDF](docs/physics/mhs-poisson-ltau.pdf) · [TeX](docs/physics/mhs-poisson-ltau.tex) | [PDF](docs/technical/mhs-poisson-ltau.pdf) · [TeX](docs/technical/mhs-poisson-ltau.tex) |
| `full-mhs-z` | [PDF](docs/physics/full-mhs-z.pdf) · [TeX](docs/physics/full-mhs-z.tex) | [PDF](docs/technical/full-mhs-z.pdf) · [TeX](docs/technical/full-mhs-z.tex) |

Rebuild with `bash docs/build.sh` (pdflatex, twice per document; removes `.aux/.log/.out`).

## Repository map

* `HSEMHS/src/` – the package (modes, EOS, operators, parallel layer, solvers); `HSEMHS/scripts/run.jl` – entry point.
* `HSEMHS/config/` – mode configs (`modes/`), gate0 and EOS-table configs.
* `HSEMHS/test/`, `HSEMHS/oracles/` – test suite and Python oracles.
* `HSEMHS/scripts/m5_bifrost/` – the full-MHS driver (`m5h_fullres.jl`, `bif_setup.jl`) and M5-era experiment and
  analysis scripts; `HSEMHS/scripts/poisson/` – the original Poisson scripts (history; the modes supersede them).
* `HSEMHS/reports/` – gate reports and earlier write-ups; `docs/` – the current mode documents.
* `design/` – specification, design reviews and the Codex brief/review history (historical; the code and `docs/` are
  current).
* `wittman-eos/` – Wittmann EOS source and partition functions; `HSEMHS/deps/witt/` – the shim and build script.
* Not in git: `IO/`, `HSEMHS/cache/`, `HSEMHS/out/`, built `libwitt`.

## Licence and attribution

HSEMHS is released under the [MIT License](LICENSE). The licence covers the code written for this project; the
Wittmann EOS code in `wittman-eos/` is third-party, keeps its own terms (not recorded in the repository), and is not
relicensed by it. The Poisson approach follows the FIRTEZ-dz inversion code
(Borrero et al.); Bifrost data: simulation en024048_hion.
