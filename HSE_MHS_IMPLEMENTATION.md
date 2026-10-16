# HSEMHS.jl: implementation document (MPI + threads, Julia)

*Status: implementation handoff, 16 Oct 2026. Companion to `HSE_MHS_FINAL_SPEC.md` (the "spec"). The spec decides **what** is built; this file decides **how**, down to array layouts, stencil coefficients, call signatures, loop structure, file schemas and test thresholds. An engineer or AI agent should be able to implement the full code from this file plus the spec without reading the four rationale documents.*

*Precedence: spec §1 decisions > this document > the rationale documents. Where this document had to resolve an ambiguity or found an error in the spec, it says so in §1 (errata, E-numbers). Those items must be copied into spec §10 by the user before M7.*

---

## Contents

0. How to use this document
1. Errata and implementation choices against the spec
2. Toolchain, repository layout, dependencies
3. Conventions: indexing, layout, units, constants, determinism
4. Core types
5. Parallel infrastructure (MPI + threads)
6. Grid metrics and discrete operators
7. EOS and opacity (M1)
8. Input reader and Gate 0 audit (M0)
9. `column_hse` (M2)
10. τ₅₀₀
11. Output writer and HDF5 schema
12. Force-budget screening and column-HSE residual report (M3)
13. `mhs` (M4a, M5)
14. Synthetic benchmarks (Gate C)
15. Test suite: every gate, file by file
16. Configuration file and command-line entry points
17. Memory preflight, performance, Olivia launch
18. Milestone execution plan with definitions of done
19. Decisions still owed by the user

---

## 0. How to use this document

**Rules for the implementer.**

1. Implement milestones strictly in the order of §18. Do not start a milestone until the previous gate passes and its numbers are recorded in `reports/`.
2. Every formula here is normative. If a formula looks wrong, stop and raise it; do not "fix" it silently. (Two spec formulas were wrong; see E1 and E2.)
3. No run larger than 128³ before M5 passes (spec §9).
4. Never use `@fastmath`, `muladd`, `@simd` on a reduction, `@turbo`, or BLAS in any code path that produces an output array or an objective value. They change bit patterns between thread counts and machines. `@inbounds` is allowed after the bounds-checked version passes its tests.
5. Every public function validates its inputs and returns a status rather than throwing in a hot loop. Throw only at API boundaries (`invalid_input`).
6. The Python Wittmann code (`wittman-eos/witt.py`) and the C++ code (`wittman-eos/witt_eos_cpp.cpp`) are reference implementations. Never edit the originals; copy and patch (§7.1).
7. When this document says "global", it means over all MPI ranks. When it says "local", it means the owned part of one rank's block.

**Glossary.**

| Symbol | Meaning |
|---|---|
| `NX, NY, NZ` | Global grid sizes (runtime) |
| `nx, ny` | Local owned sizes on a rank (z is never decomposed, so local `nz = NZ`) |
| `u` | `ln(P / 1 Pa)` |
| `v` | Unconstrained pressure variable of the MHS optimizer; `u = v − ⟨v⟩_anchor + ln P_ref` |
| `b̃_c` | Scaled field perturbation, `B_c = Bobs_c + σ_c · b̃_c` (E4) |
| `Lρ, Lne, Lχ` | Tabulated `ln ρ`, `ln nₑ`, `ln χ₅₀₀` as functions of `(ln T, ln P)` |
| `ρ_u` | `∂ ln ρ / ∂ ln P` at fixed T (from the table interpolant) |
| `D_a` | First-derivative operator along axis `a ∈ {x, y, z}` (§6.2) |
| `D2_a` | Second-derivative operator along axis `a` (§6.3) |
| `w_n` | Volume weight, cell volume / global mean cell volume |
| `S_n` | Fixed force scale (spec §6.3) |

---

## 1. Errata and implementation choices against the spec

Items E1–E3 are corrections; the spec is wrong or contradicts its own gate. E4–E12 are choices the spec left open. Each states what to implement.

| ID | Spec location | Issue | What to implement |
|---|---|---|---|
| **E1** | §5.4 `:exp_tail` | `H_χ = (z_1 − z_2)/ln(χ_1/χ_2)` is negative whenever χ decreases upward (χ₁ < χ₂), so the `0 < H_χ` test would reject every normal column. The intended quantity is `H_χ = −(d ln χ/dz)⁻¹`. | `H_χ = (z_1 − z_2) / ln(χ_2/χ_1)`. Accept `:exp_tail` iff `χ_1 > 0`, `χ_2 > χ_1`, and `0 < H_χ < H_max` (default 5000 km). |
| **E2** | §5.3 vs Gate 2 | "Column mass is the trapezoid of ρ" cannot meet the Gate 2 mass identity `|P − P_top − g m|/P ≤ 1e-6`: trapezoid error is about `Δz²/(12 H²)` ≈ 3e-3 at 20 km spacing. | Integrate `m` together with `u` in the same RK4 stages (`dm/dz = −ρ`, §9.3). Then the identity holds to RK4 accuracy. The trapezoid value is still computed and stored as the diagnostic `colmass_trap`. |
| **E3** | §6.7 vs Gate C1 | With collocated 3-point stencils and one-sided z-face rows, each column has `NZ` force equations for a pressure profile whose continuum solution has a free scale. The discrete system is therefore over-determined by one equation per column, and the least-squares minimum has a nonzero truncation-level residual floor. Measured for an isothermal ideal-gas column on a uniform grid (smallest singular vector of the scaled 1D system, checked during the writing of this document): RMS `|R|/S` ≈ 8.9e-4, 1.2e-4 and 1.4e-5 at Δz/H = 1/4, 1/8 and 1/16 (about third order), with the maximum about 7× the RMS. Bifrost has Δz/H ≈ 1/8 in the photosphere. The synthetic `tol_rms = 1e-6` is therefore unreachable on practical grids even for exact equilibria. | Keep the discretization (D9 is binding). For synthetic gates, compute the **truncation floor** `R_floor` = per-layer RMS of `|R|/S` of the exact analytic solution sampled on the grid. Gate C passes if the solver's per-layer RMS ≤ `max(1e-6, 1.5·R_floor)`, p99 ≤ `max(1e-4, 1.5·R_floor_p99)`, and both converge at order ≥ 1.8 under 2× refinement. `equilibrium_tol_met` in production keeps the spec definition. **User to confirm (§19).** |
| E4 | §6.1 | Unknowns `u, B` differ in magnitude by ~10⁵, which makes L-BFGS badly scaled. | Optimize over `x = (v, b̃x, b̃y, b̃z)` with `B_c = Bobs_c + σ_c b̃_c`. This is a fixed diagonal change of variables; it changes neither the objective nor its minimizer. |
| E5 | §6.5/§6.7 | Mapping of stop reasons to `solver_status`. | Gradient test → `converged`; Φ-stagnation test or two consecutive line-search failures → `stalled`; iteration cap → `max_iter`. |
| E6 | D5, §5.3 | Outputs from the table or the direct solver? | All outputs (`rho`, `ne`, χ₅₀₀) come from the **same table** used in the solve, so that the solve and the outputs share one EOS state. `Pe = nₑ k T`, `mu_bar = ρ k T/(P m_u)`, `nH_total = ρ / m_per_H`. The direct solver is used only for table construction and validation. |
| E7 | §4.6 | Opacity is evaluated where? | χ₅₀₀ is tabulated as a third table field `Lχ(ln T, ln P)` from the Julia port of `contOpacity` (§7.7), with the same accuracy gate as `ln nₑ`. |
| E8 | M7 restart | Saving the L-BFGS history costs 80 cubes per checkpoint. | A checkpoint stores `x`, the continuation state and the iteration counter only. A restart rebuilds the L-BFGS history from empty. |
| E9 | §6.3 `h_n` | "Local mean spacing" is undefined. | `h_n = (h_x,i + h_y,j + h_z,k)/3`, with `h_a` the dual-cell width of §6.4. |
| E10 | §3 item 5 | σ_B in the screening scale is a scalar; σ is per component. | `σ_B,n = sqrt(σ_x,n² + σ_y,n² + σ_z,n²)`. |
| E11 | D11 | MPI wired in at M7. | The user asked for MPI + threads now. The MPI layer (§5) is built at M0 and is the only parallel layer; every run, including tests, goes through it with one or more ranks. M7 adds the multi-rank validation (decomposition invariance). This is compatible with D11, since kernels are block-local from day one. |
| E12 | §6.5 line search | "strong-Wolfe (Hager–Zhang)". | In-house L-BFGS with the strong-Wolfe line search of Nocedal & Wright, Algorithms 3.5 and 3.6 (§13.8). This is allowed by the spec ("in-house is fine") and works directly on distributed vectors. |

---

## 2. Toolchain, repository layout, dependencies

### 2.1 Toolchain

- Julia ≥ 1.10 (LTS). The development machine has 1.12.3. Set `[compat] julia = "1.10"`.
- C++17 compiler (clang++ on macOS, g++ ≥ 9 on Linux) for the Wittmann shim.
- MPI: on macOS use the default `MPICH_jll` that MPI.jl installs; on Olivia use the system MPI through `MPIPreferences.use_system_binary()` (§17.3).
- Python 3 with numpy and h5py (already at `/Users/harshm/env/bin/python3`) for the oracles only.

### 2.2 Dependencies (`Project.toml`)

```toml
name = "HSEMHS"
uuid = "<generate with UUIDs.uuid4()>"
version = "0.1.0"

[deps]
MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
SHA = "ea8e919c-243c-51af-8825-aaa63cd721ce"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcabc9567"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[extras]
Test = "8dfed614-e22f-5e08-85e1-65c5234f0b40"
MPIPreferences = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"

[compat]
MPI = "0.20"
HDF5 = "0.17"
OffsetArrays = "1"
julia = "1.10"
```

No Optim.jl, no PETSc, no AD packages in `[deps]`. ForwardDiff may be added to a separate `test/Project.toml` for gradient cross-checks only.

### 2.3 Layout

```text
HSEMHS/
  Project.toml
  deps/
    witt/
      witt_eos_cpp.cpp          # verbatim copy of wittman-eos/witt_eos_cpp.cpp (provenance)
      witt_shim.cpp             # patched copy + new C API (§7.1)
      build.jl                  # compiles libwitt.{dylib,so}
  src/
    HSEMHS.jl                   # module, includes, exports
    Constants.jl                # SI constants (§3.4)
    Status.jl                   # status enums and bit masks (§4.5)
    Config.jl                   # TOML parsing + validation (§16)
    Parallel.jl                 # MPI context, decomposition, halos, reductions, error propagation (§5)
    Fields.jl                   # haloed local arrays (§4.2)
    Geometry.jl                 # Grid, axis metrics, stencil coefficients (§6)
    Operators.jl                # D, Dᵀ, D2, D2ᵀ kernels (§6)
    eos/
      WittmannDirect.jl         # ccall wrapper (§7.2)
      SahaHot.jl                # H+He(+metals) hot branch (§7.4)
      Envelope.jl               # envelope + T_sw selection (§7.3, §7.5)
      Opacity500.jl             # Julia port of contOpacity at 5000 Å (§7.7)
      Table.jl                  # bicubic Hermite table: build, refine, evaluate, save/load (§7.6)
      TestEOS.jl                # analytic ideal-gas EOSs for gates (§7.9)
    IOInput.jl                  # NetCDF/HDF5 reader, units, crop, validation (§8.1)
    IOOutput.jl                 # HDF5 writer, metadata, checkpoint (§11)
    Gate0.jl                    # audit (§8.2)
    ColumnHSE.jl                # column_hse (§9)
    Tau.jl                      # τ₅₀₀ (§10)
    Screening.jl                # force-budget screening + column residual report (§12)
    MHSResidual.jl              # forward kernel: R, Φ terms (§13.4)
    MHSAdjoint.jl               # adjoint kernel: ∇Φ (§13.5)
    MHSOptimize.jl              # L-BFGS, line search, continuation, classification (§13.7–13.10)
    Diagnostics.jl              # layer statistics, distributed quantiles (§5.6)
    Synthetic.jl                # Gate C analytic equilibria (§14)
    Driver.jl                   # run(config) entry (§16)
  scripts/
    run.jl                      # julia --project scripts/run.jl config.toml
    build_eos_table.jl
    gate0_audit.jl
    export_rh_atmos.jl
    olivia_job.sh
  config/
    bifrost385_column_hse.toml
    bifrost385_mhs.toml
    synthetic_*.toml
  test/
    runtests.jl
    (one file per gate, §15)
  oracles/
    python/                     # oracle scripts written by the non-implementing model (spec §9)
  reports/                      # gate reports (markdown + HDF5), committed
  cache/                        # eos_table.h5 etc. (not committed)
```

`src/HSEMHS.jl` includes files in the order listed above (each depends only on earlier ones).

---

## 3. Conventions

### 3.1 Vertical order and indices

- Storage is **top-first** everywhere: `z[1]` is the highest level, `z[NZ]` the lowest, `z[k+1] < z[k]`.
- `B_z > 0` means upward. Top-first storage never flips vector components or the sign of g.
- Gravity is `−g ẑ` with `g > 0` (default 274.0 m s⁻²; the input file carries `gravity_m_s2`).
- `k` always denotes the z index, `i` the x index, `j` the y index.

### 3.2 In-memory layout

- Julia arrays of physical fields are `(x, y, z)` with x fastest (column-major). Local blocks have a one-cell halo in x and y and none in z (§4.2).
- On disk (HDF5), datasets are written in the Python/C convention `(x, y, z)` C-order, which Julia's HDF5.jl sees as `(z, y, x)`. The reader permutes with `permutedims(A, (3, 2, 1))`; the writer writes `permutedims(A, (3, 2, 1))`. The `dims` attribute on every dataset is the string array `["x","y","z"]` (C order), as in `IO/input-mhs-hse.nc`.

### 3.3 Units

SI internally (D8). Units are read from the `units` attribute of each dataset (bytes or string, strip whitespace). Accepted strings and the factor that converts to SI:

| Quantity | Accepted unit strings → factor |
|---|---|
| length | `m`→1, `km`→1e3, `cm`→1e-2, `Mm`→1e6 |
| temperature | `K`→1 |
| magnetic field | `T`→1, `G`, `gauss`→1e-4 |
| pressure | `Pa`→1, `dyn/cm2`, `dyn cm-2`, `dyn / cm2`→0.1 |
| density | `kg / m3`, `kg/m3`, `kg m-3`→1, `g/cm3`, `g cm-3`→1e3 |
| number density | `1 / m3`, `m-3`, `1/m3`→1, `cm-3`, `1 / cm3`→1e6 |
| velocity | `m / s`, `m/s`→1, `cm/s`→1e-2, `km/s`→1e3 |

Any other string, or a missing attribute, is `invalid_input` with the dataset name in the message. Never guess.

### 3.4 Constants (`src/Constants.jl`)

```julia
const K_B   = 1.380649e-23          # J/K   (SI 2019, exact)
const M_U   = 1.66053906660e-27     # kg
const M_E   = 9.1093837015e-31      # kg
const H_PL  = 6.62607015e-34        # J s   (exact)
const E_CH  = 1.602176634e-19       # C     (exact)
const MU0   = 1.25663706212e-6      # N A^-2 (CODATA 2018)
const SIGMA_T = 6.6524587321e-29    # m^2 Thomson
const ONE_GAUSS = 1.0e-4            # T
```

The C++ Wittmann code uses its own CGS constants (CODATA 2010, e.g. `BK = 1.3806488e-16`). Do **not** change them; derived Julia quantities that must agree with the C++ state (`Pe = nₑ k T`, `mu_bar`) use `K_B_WITT = 1.3806488e-23` and `M_U_WITT = 1.660538921e-27`, exported from `Constants.jl`. Record both constant sets in `/meta`.

### 3.5 Determinism rules

1. Column-HSE outputs must be bitwise identical for any thread count and any MPI decomposition (columns are independent; no reductions enter the values).
2. MHS objective values and gradients must be bitwise identical for any thread count at a fixed MPI decomposition. Achieve this with **fixed-order partial sums** (§5.5): each `(j, k)` row of a local block is summed sequentially over `i` into a slot `partial[j, k]`; the slots are then summed sequentially in `k`-major, `j`-minor order on the main thread. Never use `Threads.Atomic` accumulation or `sum` over a threaded generator.
3. Across different MPI decompositions, values agree to reduction-order tolerance: the relative difference in Φ ≤ 1e-12, and the classification flags are identical.

---

## 4. Core types

### 4.1 Grid (`Geometry.jl`)

```julia
struct Axis
    s::Vector{Float64}          # global coordinates, length N (x,y increasing; z decreasing)
    periodic::Bool
    period::Float64             # L for periodic axes, NaN otherwise
    # first-derivative coefficients, global, row n uses nodes idx[n,1:3]
    d1_idx::Matrix{Int}         # N×3 global node indices (periodic wrap applied)
    d1_c::Matrix{Float64}       # N×3
    d2_c::Matrix{Float64}       # N×3 second-derivative coefficients, same nodes
    h::Vector{Float64}          # dual-cell width (§6.4), >0
end

struct Grid
    x::Axis; y::Axis; z::Axis
    NX::Int; NY::Int; NZ::Int
    wmean::Float64              # global mean cell volume
end
```

Precompute everything on all ranks (the axes are 1D and tiny).

### 4.2 Local fields (`Fields.jl`)

```julia
const HALO = 1
const Field = OffsetArray{Float64,3,Array{Float64,3}}

newfield(b::Block) = OffsetArray(zeros(b.nx + 2HALO, b.ny + 2HALO, b.nz),
                                 0:b.nx+1, 0:b.ny+1, 1:b.nz)
```

- Owned indices: `i ∈ 1:nx`, `j ∈ 1:ny`, `k ∈ 1:NZ`. Halo indices: `i ∈ {0, nx+1}`, `j ∈ {0, ny+1}`.
- Global index of local `i` is `b.i0 + i − 1` (likewise `j`).
- All 3D work arrays are `Field`s, allocated once per run and reused. No allocation inside kernels (check with `@allocated` in tests: 0 bytes after warm-up).
- 2D per-column arrays (masks, anchor-plane data) are plain `Matrix` of size `(nx, ny)`.
- Integer masks are `OffsetArray{UInt8,3}` with the same index ranges.

### 4.3 Block and decomposition (`Parallel.jl`)

```julia
struct Block
    nx::Int; ny::Int; nz::Int    # owned sizes; nz == NZ
    i0::Int; j0::Int             # global index of owned (1,1)
end

struct ParCtx
    comm::MPI.Comm               # Cartesian communicator
    rank::Int; nranks::Int
    px::Int; py::Int             # process grid
    cx::Int; cy::Int             # my coordinates (0-based)
    nbr_xm::Int; nbr_xp::Int; nbr_ym::Int; nbr_yp::Int   # neighbour ranks
    block::Block
    nthreads::Int
end
```

### 4.4 Run context and result

```julia
struct RunCtx
    par::ParCtx
    grid::Grid
    cfg::Config                  # parsed TOML (§16)
    eos::EOSTable                # loaded table (§7.6), or an analytic test EOS
    log::IO                      # rank-0 log stream; other ranks write to devnull unless debug
end

struct Result
    mode::Symbol                         # :column_hse | :mhs
    fields::Dict{String,Field}           # local blocks, names as in §11
    masks::Dict{String,Any}              # UInt8 local arrays
    layer_tables::Dict{String,Matrix{Float64}}   # NZ×ncols, global (identical on all ranks)
    flags::Dict{String,Any}              # solver_status, equilibrium_tol_met, ...
    meta::Dict{String,Any}
end
```

### 4.5 Status codes (`Status.jl`)

```julia
@enum SolverStatus::UInt8 converged=0 stalled=1 max_iter=2 eos_failure=3 invalid_input=4

# per-node EOS bit mask (UInt8)
const EOS_OK              = 0x00
const EOS_HOT_BRANCH      = 0x01   # T ≥ T_sw·1.1, Saha branch
const EOS_BLEND           = 0x02   # T in the blend band
const EOS_OUT_OF_ENVELOPE = 0x04
const OPACITY_INCOMPLETE  = 0x08   # T ≥ 3.0e4 K (HOTOP placeholder, LUKEOP cut-off)
const EOS_DIRECT_FAILED   = 0x10   # table node built from a non-converged direct call (must never appear)

# per-column column_hse mask (UInt8)
const COL_OK              = 0x00
const COL_RK_REFINED      = 0x01   # some interval needed n > 2 n_sub
const COL_RK_TOL_NOT_MET  = 0x02   # error > tol even at n = 64
const COL_EOS_FAIL        = 0x04

# per-column tau_top_method (UInt8)
const TAU_EXP_TAIL = 0x00; const TAU_HALF_CELL = 0x01; const TAU_ZERO = 0x02
```

---

## 5. Parallel infrastructure (`Parallel.jl`)

### 5.1 Initialization

```julia
function init_parallel(cfg)::ParCtx
    if !MPI.Initialized()
        provided = MPI.Init(threadlevel = :funneled)
        provided < MPI.THREAD_FUNNELED && error("MPI does not provide THREAD_FUNNELED")   # MPI.jl 0.20 returns the provided ThreadLevel
    end
    world = MPI.COMM_WORLD
    nr = MPI.Comm_size(world)
    px, py = cfg.parallel.px, cfg.parallel.py           # 0 = auto
    dims = MPI.Dims_create(nr, [px, py])                # fills zeros
    comm = MPI.Cart_create(world, dims; periodic = [true, true], reorder = false)
    cx, cy = MPI.Cart_coords(comm)
    nbr_xm, nbr_xp = MPI.Cart_shift(comm, 0, 1)
    nbr_ym, nbr_yp = MPI.Cart_shift(comm, 1, 1)
    ...
end
```

- `periodic = [true, true]` always (v1 supports periodic x and y only; any other `bc` value is `invalid_input` with the message "non-periodic lateral boundaries are not implemented in v1").
- Rank 0 asserts `MPI.Is_thread_main()` at every MPI call site in debug builds (`HSEMHS_DEBUG=1`).
- **All MPI calls are made from the main task, outside any `Threads.@threads` region.** Kernels never call MPI.
- BLAS: `LinearAlgebra.BLAS.set_num_threads(1)` at start-up. Set `OMP_NUM_THREADS=1` in the job script.

### 5.2 Block distribution

For an axis of global length `N` split over `p` ranks, rank coordinate `c` (0-based) owns

```julia
base, rem = divrem(N, p)
n_c  = base + (c < rem ? 1 : 0)
i0_c = c*base + min(c, rem) + 1
```

Reject at start-up (collectively, `invalid_input`) if any `n_c < 2`.

### 5.3 Halo exchange

`exchange_halos!(par, fields::Vector{Field})` updates `i ∈ {0, nx+1}` for `j ∈ 1:ny` and `j ∈ {0, ny+1}` for `i ∈ 1:nx`. Corners are not needed: every stencil is axis-aligned (no mixed derivatives anywhere in this code).

Algorithm (all on the main task):

1. Pack. For each field `f`, x-faces: `sendL[:, :, m] = f[1, 1:ny, :]` and `sendR = f[nx, 1:ny, :]`. Pack all fields into one contiguous buffer per direction (size `nfields·ny·NZ`). The packing loop may be threaded over `k` (it writes disjoint slices).
2. Post `MPI.Irecv!` from `nbr_xm` into `recvL` and from `nbr_xp` into `recvR`; post `MPI.Isend` of `sendR` to `nbr_xp` (tag 1) and `sendL` to `nbr_xm` (tag 2). `MPI.Waitall`.
3. Unpack into `f[0, 1:ny, :]` (from `recvL`) and `f[nx+1, 1:ny, :]` (from `recvR`).
4. Repeat for y with `f[1:nx, 1, :]`, `f[1:nx, ny, :]`.
5. Self-neighbour shortcut: if `nbr_xm == rank` (px = 1), copy directly (`f[0,j,k] = f[nx,j,k]`, `f[nx+1,j,k] = f[1,j,k]`) and skip MPI.

Buffers are allocated once per distinct `nfields` and kept in a cache inside `ParCtx`.

Test (`test/test_halo.jl`): fill `f[i,j,k] = gi + 1000 gj + 1e6 k` with global indices; after exchange, every halo cell equals the formula evaluated at the periodically wrapped global index. Run on 1, 2, 4 and 6 ranks (`mpiexec -n`).

### 5.4 Threaded loops

- Column work (`column_hse`, τ): `Threads.@threads :dynamic for c in 1:(nx*ny)`; `(i, j) = divrem(c−1, nx) .+ ...`. Each column is independent; scheduling does not affect results.
- Stencil kernels: `Threads.@threads :static for jk in 1:(ny*NZ)`; `j = (jk−1) % ny + 1`, `k = (jk−1) ÷ ny + 1`; the inner loop over `i in 1:nx` is sequential. A `(j, k)` row is the unit of work and of partial sums. With `ny·NZ` ≥ 10⁴ rows this load-balances well on 256 threads.

### 5.5 Deterministic reductions

```julia
struct RowPartials
    p::Matrix{Float64}            # (ny, NZ), one slot per row, per term
end
# kernel: p[j,k] = Σ_{i=1..nx} term(i,j,k)   (sequential in i, written once)
# local total: s = 0.0; for k in 1:NZ, j in 1:ny; s += p[j,k]; end   (main task)
# global total: MPI.Allreduce(s, +, comm)
```

When several terms are reduced together (Φ_F, Φ_D, Φ_B, Φ_S, Φ_U, and counters), use one `(nterms, ny, NZ)` array and one `Allreduce` of an `nterms` vector.

Dot products for L-BFGS (`dot4(a, b)` over the four state fields) use the same scheme.

### 5.6 Distributed per-layer quantiles (`Diagnostics.jl`)

Needed for `median_layer(ρ₀g)` (§13.3), Gate 0 percentiles, and the per-layer p99 in the classification. These must be exact and decomposition-independent.

```julia
function layer_quantiles(par, f::Field, ps::Vector{Float64}; mask=nothing, stride=1)
    # for each k: gather owned values (optionally masked / subsampled on GLOBAL indices
    # i ≡ 1 mod stride, j ≡ 1 mod stride) to rank 0 with MPI.Gatherv!, then
    # sort!, Statistics.quantile(v, ps; sorted=true)  (type-7, same as numpy default),
    # NaNs excluded and counted. Broadcast the NZ×length(ps) table.
end
```

Gathering one layer at a time keeps rank-0 memory at `NX·NY` doubles. The table is identical on all ranks after `MPI.Bcast!`.

### 5.7 Collective error propagation

Every phase ends with `status = collective_status(par, local_status)`, which is `MPI.Allreduce(UInt8(local_status), max, comm)` (the enum order is severity order: `converged < stalled < max_iter < eos_failure < invalid_input`). Exceptions inside a phase are caught on the rank, converted to a status plus a message, and the message is gathered to rank 0 before all ranks return the same status. No rank may enter a collective that another rank skips: kernels never return early, they set the local status and finish the loop.

### 5.8 Logging

Rank 0 logs to stdout and `<output>.log`. Each log line: ISO time, elapsed seconds, phase, message. Other ranks log only in debug mode, to `<output>.rankNNNN.log`.

---

## 6. Grid metrics and discrete operators (`Geometry.jl`, `Operators.jl`)

All operators are 1D along one axis, applied along that axis of a 3D field. Coefficients are computed from **signed coordinates**, so decreasing z and nonuniform spacing need no special cases (D9).

### 6.1 Coordinates of neighbours

For axis `s` with `N` nodes:

- Interior node `n` (`2 ≤ n ≤ N−1`), or any node of a periodic axis: neighbours `n−1`, `n+1`. For a periodic axis, wrap the index and shift the coordinate: the left neighbour of node 1 has coordinate `s[N] − L`, and the right neighbour of node `N` has coordinate `s[1] + L`.
- Period: `L = period_x_m` from the input attributes if present; otherwise `L = N·Δ` where `Δ = s[2] − s[1]`, and the axis must be uniform to 1e-9 relative (else `invalid_input`). Check for a duplicated endpoint: if `|s[N] + Δ − s[1] − L| > 1e-6 Δ` or `|s[N] − s[1] − L| < 1e-6 Δ`, reject.
- Non-periodic axis (z only in v1): row 1 uses nodes (1, 2, 3); row N uses (N, N−1, N−2).

### 6.2 First derivative `D_a` (quadratic Lagrange, signed coordinates)

Interior or periodic row `n`, with `h1 = s_n − s_{n−1}` and `h2 = s_{n+1} − s_n` (both signed; both negative on z):

```
c_{−1} = −h2 / (h1 (h1 + h2))
c_0    = (h2 − h1) / (h1 h2)
c_{+1} =  h1 / (h2 (h1 + h2))
(D f)_n = c_{−1} f_{n−1} + c_0 f_n + c_{+1} f_{n+1}
```

One-sided row at the first node (nodes 1, 2, 3), with `d1 = s_2 − s_1`, `d2 = s_3 − s_1`:

```
c_1 = −(d1 + d2) / (d1 d2)
c_2 =   d2 / (d1 (d2 − d1))
c_3 =  −d1 / (d2 (d2 − d1))
```

One-sided row at the last node (nodes N, N−1, N−2): the same formulas with `d1 = s_{N−1} − s_N`, `d2 = s_{N−2} − s_N`, coefficients applied to `(f_N, f_{N−1}, f_{N−2})`.

Properties to unit-test (`test/test_operators.jl`): row sums are 0 to 1e-14·max|c|; the operator is exact for `f = 1, s, s²` to 1e-12 relative on random nonuniform grids (both orderings); second-order convergence on `sin` for interior rows; second order for boundary rows; reproduces `numpy.gradient(f, s, edge_order=2)` exactly to 1e-13 (Python oracle).

### 6.3 Second derivative `D2_a`

The second derivative of the quadratic through the same three nodes `(a, b, c)` used by the row (for interior rows `(n−1, n, n+1)`; for boundary rows `(1, 2, 3)` or `(N, N−1, N−2)`):

```
D2 f = 2 [ f_a / ((s_a − s_b)(s_a − s_c)) + f_b / ((s_b − s_a)(s_b − s_c)) + f_c / ((s_c − s_a)(s_c − s_b)) ]
```

Store the three coefficients in `d2_c` in the same node order as `d1_idx`.

### 6.4 Dual-cell widths, local spacing, weights

- `h_a,n = |s_{n+1} − s_{n−1}| / 2` for interior or periodic rows (with wrapped coordinates); `h_a,1 = |s_2 − s_1| / 2`, `h_a,N = |s_N − s_{N−1}| / 2` on non-periodic ends.
- Cell volume `V_ijk = h_x,i h_y,j h_z,k`. `wmean = (Σ_i h_x,i)(Σ_j h_y,j)(Σ_k h_z,k) / (NX NY NZ)`. `w_ijk = V_ijk / wmean`. `w` is separable, so store only the three 1D arrays and compute `w` on the fly.
- Local mean spacing for the divergence term: `h_n = (h_x,i + h_y,j + h_z,k) / 3` (E9).
- For the regularizers, `h_a` at the row node is `h_a,n` above.

### 6.5 Applying operators to local fields

The kernel signature, for axis x:

```julia
@inline function dx(f::Field, ax::AxisLocal, i, j, k)
    c = ax.d1_c_loc            # (nx, 3), rows for owned local i, periodic coefficients
    return c[i,1]*f[i-1,j,k] + c[i,2]*f[i,j,k] + c[i,3]*f[i+1,j,k]
end
```

Because x and y are periodic in v1 and halos hold the wrapped neighbours, every owned row uses local offsets −1, 0, +1. Precompute `AxisLocal` per rank by slicing the global coefficient matrices with the owned global indices.

For z (never decomposed), `dz(f, az, i, j, k)` reads `az.idx[k, 1:3]` and `az.c[k, 1:3]`; for `k = 1` the nodes are `(1, 2, 3)`, for `k = NZ` they are `(NZ, NZ−1, NZ−2)`.

### 6.6 Transposed operators `D_aᵀ` and `D2_aᵀ` (for adjoints)

The adjoint kernel uses **gather** form so that every output node is written by exactly one thread:

```
(Dᵀ q)_m = Σ_{rows n that reference node m} D[n, m] q_n
```

Precompute, per axis, a transposed stencil `tr_idx[m, 1:4]`, `tr_c[m, 1:4]` (unused slots have coefficient 0 and index m). For periodic x and y every node is referenced by rows `m−1, m, m+1` only:

```
(Dxᵀ q)_i = c[i+1, 1] q_{i+1} + c[i, 2] q_i + c[i−1, 3] q_{i−1}
```

This needs `q` in the halo, so **every adjoint source array is halo-exchanged before the transpose** (§13.5). Note that `c[i+1, 1]` and `c[i−1, 3]` are coefficients of the neighbouring rows; for owned `i = 1` and `i = nx` these belong to halo rows, so `AxisLocal` stores coefficient rows for `i ∈ 0:nx+1`.

For z, build the transposed stencil by looping over all rows `n` and their three referenced nodes and appending `(n, coefficient)` to node `m`'s list. Nodes 2 and 3 (and NZ−1, NZ−2) are referenced by up to 4 rows. Test: for random `f`, `q`, `⟨D f, q⟩ = ⟨f, Dᵀ q⟩` to 1e-13 relative, for each axis and for `D2`, on 1 and 4 ranks.

### 6.7 Vector operators used later

```
div B   = Dx Bx + Dy By + Dz Bz
curl B  = (Dy Bz − Dz By,  Dz Bx − Dx Bz,  Dx By − Dy Bx)
J       = curl B / μ0
L       = J × B = (Jy Bz − Jz By,  Jz Bx − Jx Bz,  Jx By − Jy Bx)
∇P      = (Dx P, Dy P, Dz P)        # P differenced directly, never P·∇u
```

---

## 7. EOS and opacity (M1)

### 7.1 C++ shim (`deps/witt/witt_shim.cpp`)

Copy `wittman-eos/witt_eos_cpp.cpp` to `deps/witt/witt_eos_cpp.cpp` unchanged (provenance; record its SHA-256 in `/meta`). Create `witt_shim.cpp` as a second copy with exactly these changes:

1. Replace the global `constexpr double PREC = 1.0e-5;` with members of `WittEOS`:
   ```cpp
   double prec_ = 1.0e-10;   // outer fixed-point tolerance (spec §4.1)
   int    maxit_ = 5000;     // outer iteration cap (was 250)
   int    maxit_inner_ = 50; // f5 loop cap (was 5)
   ```
   and a constructor `WittEOS(const char* pf_path, double prec, int maxit, const double* log_abund_or_null)`. If `log_abund_or_null` is non-null it holds 99 `log10` fractions that replace `ABUND_RAW` before normalization.
2. In `pe_pg` (both copies of the f5 loop, the ones with `while ((diff > 1.0e-5) && (it < 5))`), use `prec_` and `maxit_inner_`.
3. Replace `pe_from_pg` with a version that returns a struct:
   ```cpp
   struct PeSolve { double pe; int it; bool converged; double dif; };
   PeSolve pe_from_pg_status(double t, double pg) const {
       double dif = 1.1, pe = init_pe_from_pg(t, pg), ope = pe; int it = 0;
       while ((std::abs(dif) > prec_) && (it < maxit_)) {
           pe = (ope + pe) * 0.5; ope = pe;
           pe = pe_pg(t, pe, pg).first;
           dif = 2.0 * std::abs(pe - ope) / (pe + ope);
           ++it;
       }
       return {pe, it, std::abs(dif) <= prec_ && std::isfinite(pe) && pe > 0.0, dif};
   }
   ```
   Keep the old `pe_from_pg` as a wrapper returning `.pe` so that other methods compile unchanged.
4. Add the C API:
   ```cpp
   extern "C" {
   struct WittState { double rho, ne, pe, mu_bar, nhtot, closure_residual, dif;
                      int32_t iterations; int32_t converged; };       // 7*8 + 2*4 = 64 bytes
   void*  witt_ctx_new(const char* pf_path, double prec, int maxit, const double* log_abund_or_null);
   void   witt_ctx_free(void* ctx);
   int    witt_ctx_state(void* ctx, double T, double P_pa, WittState* out);
   int    witt_ctx_partials(void* ctx, double T, double P_pa, double pe_cgs,
                            double* xna_cm3, double* xne_cm3, double* n17);
   double witt_ctx_mass_per_H_kg(void* ctx);
   void   witt_ctx_abundances(void* ctx, double* abund99, double* amass99);
   }
   ```
   `witt_ctx_state` (all SI out, cgs inside):
   ```cpp
   pg = P_pa*10; s = pe_from_pg_status(T, pg);
   rho_cgs = rho_from_pe(T, s.pe);
   out->rho = rho_cgs*1e3; out->pe = s.pe*0.1; out->ne = s.pe/(BK*T)*1e6;
   out->mu_bar = rho_cgs*BK*T/(pg*AMU);
   out->nhtot = out->rho / mass_per_H_kg;          // mass_per_H_kg = avw/ABUND[0]*1e-3 (avw already *AMU, in g)
   out->closure_residual = std::abs(pg_from_pe(T, s.pe) - pg)/pg;
   out->dif = s.dif; out->iterations = s.it;
   out->converged = s.converged && out->closure_residual <= 1e-8;
   return out->converged ? 0 : 1;   // 2 on exception (catch(...))
   ```
   `witt_ctx_partials` calls `background_partials(T, pg, pe_cgs, n17)` and returns `xna = (pg − pe)/(BK T)`, `xne = pe/(BK T)` in cm⁻³, using the `pe` from a previous `witt_ctx_state` call so that opacity and density share one state.
5. Do not touch the physics anywhere else.

Build (`deps/build.jl`):

```julia
ext = Sys.isapple() ? "dylib" : "so"
cxx = get(ENV, "CXX", Sys.isapple() ? "clang++" : "g++")
run(`$cxx -O2 -std=c++17 -fPIC -shared -fno-fast-math -ffp-contract=off -o libwitt.$ext witt_shim.cpp`)
```

`-ffp-contract=off` prevents FMA contraction, so results do not depend on the compiler's choice. Record the compiler version string in `/meta`.

Thread safety: after construction all `WittEOS` methods are `const` and use only stack scratch, so concurrent calls on one context are safe. Still, following the spec, allocate one context per thread-chunk: `ctxs = [witt_ctx_new(...) for _ in 1:Threads.nthreads()]` and use `ctxs[chunk_id]` with `@threads :static`.

### 7.2 Julia wrapper (`eos/WittmannDirect.jl`)

```julia
struct WittState
    rho::Float64; ne::Float64; pe::Float64; mu_bar::Float64; nhtot::Float64
    closure_residual::Float64; dif::Float64
    iterations::Int32; converged::Int32
end
@assert sizeof(WittState) == 64

function witt_state(ctx::Ptr{Cvoid}, T::Float64, P::Float64)
    r = Ref{WittState}()
    code = ccall((:witt_ctx_state, LIBWITT), Cint,
                 (Ptr{Cvoid}, Float64, Float64, Ref{WittState}), ctx, T, P, r)
    return r[], code
end
```

`LIBWITT` is resolved at module init from the `HSEMHS_LIBWITT` environment variable or `deps/witt/libwitt.<ext>`; fail with a clear message if missing. A non-converged call (`code ≠ 0`) is a value-less error state: callers never use its fields except for diagnostics.

### 7.3 Envelope (`eos/Envelope.jl`)

- `T_lo = 0.9 · T_min`, `T_hi = 1.1 · T_max`, with `T_min`, `T_max` the global extrema of the (cropped) input temperature.
- `P_lo`, `P_hi`: the union of the Gate 0 proxy anchor runs (§8.2, both anchors), widened by 1 dex each side. For Bifrost 385 expect about `[10^-5, 10^12] Pa`. The envelope is stored with the table. After every `column_hse` run, check all nodes against it; if any node lies outside, the result is `eos_failure` with the message "rebuild the table with logP_range = [..]" giving the needed range. Never extend silently.

### 7.4 Hot branch: H + He + metals Saha (`eos/SahaHot.jl`)

Used only if the direct solver fails the high-T test of §7.5.

Inputs: `T`, `P` (SI); the abundance fractions `A[1:99]` and atomic masses from `witt_ctx_abundances` (so the composition is identical). Use the C++ CGS constants converted to SI (`K_B_WITT`, etc.) for consistency with the direct branch.

```
Φ(T)   = (2π m_e k T / h²)^{3/2}
S_H    = Φ exp(−13.598434 eV / kT)                 # (2 U_HII/U_HI) = 2·1/2 = 1
S_He1  = 4 Φ exp(−24.587389 eV / kT)               # 2·U_HeII/U_HeI = 2·2/1
S_He2  =   Φ exp(−54.417765 eV / kT)               # 2·U_HeIII/U_HeII = 2·1/2
metals Z ∈ {C, N, O, Ne, Mg, Si, S, Fe}: stage ladder r = 1..4 with
S_Z,r  = 2 Φ exp(−χ_Z,r / kT)                      # partition-function ratios set to 1
χ (eV): C 11.2603 24.3833 47.8878 64.4939;  N 14.5341 29.6013 47.4453 77.4735
        O 13.6181 35.1211 54.9355 77.4135;  Ne 21.5646 40.9633 63.4233 97.1900
        Mg 7.6462 15.0353 80.1437 109.2655; Si 8.1517 16.3458 33.4930 45.1418
        S 10.3600 23.3379 34.8600 47.2220;  Fe 7.9025 16.1992 30.6510 54.9100
other elements: neutral (no electrons), counted as particles and mass.
```

For a given `nₑ`, stage fractions of an element with ladder `S_1..S_r` are `f_0 = 1 / (1 + Σ_q Π_{p≤q} S_p/nₑ)`, `f_q = f_0 Π_{p≤q} S_p/nₑ`; mean charge `z̄ = Σ q f_q`.

Nuclei: `n_nuc = P/(kT) − nₑ`; `n_X = A_X · n_nuc` (A normalized to Σ A = 1, as in the C++ constructor). Charge equation:

```
F(nₑ) = nₑ − Σ_X n_X z̄_X(nₑ) = 0,     nₑ ∈ (0, P/(kT))
```

Solve with Brent's method on `ln nₑ` in `[ln(1e-30·P/kT), ln(P/kT · (1 − 1e-12))]` to relative tolerance 1e-14 (bracket is guaranteed: F < 0 at the lower end, F > 0 at the upper end). Then `ρ = n_H · m_per_H` (from `witt_ctx_mass_per_H_kg`), `Pe = nₑ k T`.

### 7.5 Coverage test and `T_sw` selection

1. Sample the envelope on a 400 × 200 grid uniform in `(log₁₀ T, log₁₀ P)`. Call the direct solver everywhere. Record `converged`, `ln ρ`, `ln nₑ`, `mu_bar`.
2. Fully ionized limit from the composition: `nₑ,full = Σ_X Z_X n_X` with `n_nuc + nₑ,full = P/kT`, `μ̄_full = (Σ_X A_X m_X)/(m_u (1 + Σ_X A_X Z_X))`.
3. **Use the direct solver everywhere** iff all samples converged, and for `T ≥ 2×10⁵ K`, `|ln nₑ − ln nₑ,full| ≤ 0.02` and `|μ̄/μ̄_full − 1| ≤ 0.02`, and `ln ρ`, `ln nₑ` are monotone in `ln T` at fixed P above 3×10⁴ K. Record `hot_branch = "none"`.
4. Otherwise, find `T_sw`: the smallest tabulated `T ∈ [1.5×10⁴, 2×10⁵] K` such that, for all `P` samples and for all `T' ∈ [T_sw, 1.1 T_sw]` (5 points), the direct solver converges and `|Δ ln ρ| ≤ 1e-3` and `|Δ ln nₑ| ≤ 1e-3` against the hot branch. If none exists: **stop**, write the disagreement maps to `reports/gate1_tsw.h5`, and report to the user (spec: "reject"). Record `hot_branch = "saha_HHeZ"`, `T_sw`.
5. Blend for `ln T ∈ [ln T_sw, ln(1.1 T_sw)]`: `t = (ln T − ln T_sw)/ln 1.1`, `w = t²(3 − 2t)`, `f = (1 − w) f_direct + w f_hot` for each of `ln ρ`, `ln nₑ`, `ln χ`. Above the band use the hot branch only (and never call the direct solver there).
6. Cold end: confirm convergence down to `T_lo` at all `P` (H₂ active). Non-convergence at the cold end is a Gate 1 failure; report the region.

### 7.6 Table (`eos/Table.jl`)

**Fields.** `F1 = ln ρ`, `F2 = ln nₑ`, `F3 = ln χ₅₀₀` (χ in m⁻¹), on axes `X = ln T`, `Y = ln P`. For each field and node store `f, f_X, f_Y, f_XY` (4 arrays of size `nX × nY`).

**Initial axes.** Uniform in log₁₀ with Δlog₁₀T = 0.002 and Δlog₁₀P = 0.02 over the envelope, endpoints included, converted to natural logs.

**Node values and derivatives.** For node `(X, Y)`, evaluate the blended function `F(X, Y)` (direct, blend or hot as per §7.5) at the 9 points `(X + aδ, Y + bδ)`, `a, b ∈ {−1, 0, 1}`, `δ = 1e-3`:

```
f_X  = (F(+,0) − F(−,0)) / 2δ
f_Y  = (F(0,+) − F(0,−)) / 2δ
f_XY = (F(+,+) − F(+,−) − F(−,+) + F(−,−)) / 4δ²
```

If any of the 9 calls fails to converge, set `EOS_DIRECT_FAILED` for that node; any such node fails Gate 1 (it is never silently used).

Threading: `@threads :dynamic` over `Y` rows; one context per chunk.

**Evaluation.** Find the cell `[X_p, X_{p+1}] × [Y_q, Y_{q+1}]` with an `O(1)` lookup: keep for each axis a bucket array over a uniform super-grid of spacing equal to the smallest interval, mapping a bucket to the first interval it overlaps, then step forward at most a few intervals. Let `hX = X_{p+1} − X_p`, `t = (X − X_p)/hX`, and likewise `hY`, `s`. With the Hermite basis

```
H0(t) = 2t³ − 3t² + 1     H1(t) = −2t³ + 3t²
K0(t) = t³ − 2t² + t      K1(t) = t³ − t²
H0' = 6t² − 6t, H1' = −6t² + 6t, K0' = 3t² − 4t + 1, K1' = 3t² − 2t
```

the interpolant is

```
F(t,s) = Σ_{a,b∈{0,1}} [ f_ab Ha(t)Hb(s) + hX fX_ab Ka(t)Hb(s) + hY fY_ab Ha(t)Kb(s) + hX hY fXY_ab Ka(t)Kb(s) ]
∂F/∂Y  = (1/hY) Σ_{a,b} [ f_ab Ha(t)Hb'(s) + hX fX_ab Ka(t)Hb'(s) + hY fY_ab Ha(t)Kb'(s) + hX hY fXY_ab Ka(t)Kb'(s) ]
```

where `f_ab` is the node value at `(X_{p+a}, Y_{q+b})`. `ρ_u = ∂F1/∂Y` is `∂ ln ρ/∂ ln P|_T`.

API (allocation-free, `@inline`):

```julia
lnrho(tab, X, Y)           -> (value, status::UInt8)
lnrho_dY(tab, X, Y)        -> (value, dvalue_dY, status)
state(tab, X, Y)           -> (lnrho, lnne, lnchi, status)
```

Outside the axes: return `NaN` values and `EOS_OUT_OF_ENVELOPE`. Status also carries `EOS_HOT_BRANCH`/`EOS_BLEND` from `X` and `OPACITY_INCOMPLETE` if `T ≥ 3.0×10⁴ K`.

**Refinement.** Repeat up to 5 times:

1. For every cell evaluate the interpolant at the X-edge midpoint `(t, s) = (½, 0)`, the Y-edge midpoint `(0, ½)` and the centre `(½, ½)`, and compare with direct evaluation (or the blended function).
2. Thresholds (half the Gate 1 tolerances): `|Δ ln ρ| > 5e-6`, `|Δ ln nₑ| > 5e-5`, `|Δ ln χ| > 5e-5`. An X-edge failure marks X-interval `p`; a Y-edge failure marks Y-interval `q`; a centre failure marks both.
3. Bisect all marked intervals (insert midpoints in the global axis vectors), recompute only new nodes, rebuild buckets.

Then run the **withheld validation**: 10⁶ points from a fixed-seed (`Random.Xoshiro(20261016)`) uniform sample in `(X, Y)` over the envelope, excluding points within 1e-9 of nodes. Record max and p99.9 of `|Δ ln ρ|`, `|Δ ln nₑ|`, `|Δ ln χ|` in `reports/gate1_table.md` plus error maps in HDF5.

**File.** `cache/eos_table.h5`: datasets `X`, `Y`, `F{1,2,3}/{f,fX,fY,fXY}`, attributes `T_sw`, `hot_branch`, `prec`, `maxit`, `abundances` (99), `amass` (99), `mass_per_H_kg`, `pf_sha256`, `shim_sha256`, `source_sha256`, `envelope`, `version = "1"`, and `table_sha256` (SHA-256 over the concatenated raw bytes of all datasets in a fixed order). At load, verify `table_sha256`, and verify `pf_sha256` against the configured pf file; mismatch → `invalid_input`.

Expected size for Bifrost: ~1500 × 900 nodes before refinement, 3 fields × 4 arrays × 8 B ≈ 130 MB. Building it needs ~1.2×10⁷ direct calls; on 8 threads expect tens of minutes. Build once.

### 7.7 Opacity port (`eos/Opacity500.jl`)

Port `contOpacity` and every function it calls from `wittman-eos/witt.py` (lines 767–1363: `SEATON, COULFF, COULX, HOP, HRAYOP, H2PLOP, HMINOP, HE1OP, HE2OP, HEMIOP, HERAOP, Mg1OP, C1OP, Al1OP, Si1OP, Fe1OP, COOLOP, N1OP, O1OP, Mg2OP, Si2OP, Ca2OP, LUKEOP, HOTOP, ELECOP, H2RAOP, cop`) to Julia, one function per Python function, same names in lower case, scalar arguments, one wavelength.

Translation rules:

- Keep CGS units and every numeric literal verbatim (copy the module-level arrays such as `E1` and `FREQSI2` exactly, as `const` tuples).
- Python `log` → `log`, `exp` → `exp`, `**` → `^` (use `x^2` → `x*x` only where Python wrote `x*x`; otherwise keep `^`). Python `int(...)` truncation → `trunc(Int, ...)`; Python floor division `//` → `fld`; Python negative indices → explicit `end − m`. 0-based array indices → +1.
- `HOTOP()` returns 0.0 (placeholder; hence `OPACITY_INCOMPLETE` above 3×10⁴ K).
- Wavelength `5000.0` Å, `FREQ = 2.997925e18 / 5000.0`.
- `cop` returns `(A + B, B)`, i.e. extinction including scattering, and scattering alone. **Use only `A + B`; never add scattering again.** Convert once: `χ [m⁻¹] = 100 · (A + B) [cm⁻¹]`.
- Inputs: `H1, H2, HMIN, HE1, HE2, HE3, C1, AL1, SI1, SI2, CA1, CA2, MG1, MG2, FE1, N1, O1` = `n17[1:17]` from `witt_ctx_partials` (already divided by partition functions, as Python's `divide_by_u=True`; the C++ `ion_partials` divides by `u`, and `n[0]` carries the 0.5 factor), `XNA = xna`, `XNE = xne` (cm⁻³), `TK = k T` (erg), `TKEV = TK / EV`, `HKT = HH/TK`, `TLOG = log(T)`.
- In the hot branch, build `n17` from the Saha fractions: `H1 = n_HI/2`, `H2 = n_HII`, `HMIN = 0`, `HE1 = n_HeI/1`, `HE2 = n_HeII/2`, `HE3 = n_HeIII/1`, all metal entries 0 (cm⁻³). Their contribution is negligible against electron scattering at these temperatures, and `OPACITY_INCOMPLETE` is set anyway.

Oracle test (`oracles/python/opacity_oracle.py`, written by the non-implementing model): 1000 random `(T, P)` points in the envelope; Python computes `Pe` with `witt(prec=1e-10).pe_from_pg`, then calls `cop` directly with `getBackgroundPartials(..., divide_by_u=True)` (call `cop` at module level if `contOpacity`'s `this.cop` reference does not resolve). Julia must match `A + B` to 1e-12 relative with the same `Pe`.

### 7.8 EOS interface used by the solvers

```julia
abstract type AbstractEOS end
struct EOSTable <: AbstractEOS ... end           # §7.6
struct IdealGasIsothermal <: AbstractEOS; mu::Float64; end   # tests
struct IdealGasMu <: AbstractEOS; mu::Float64; end           # ρ = P μ m_u/(kT)
# all implement:
lnrho_dY(eos, X, Y) -> (lnρ, ∂lnρ/∂lnP, status)
state(eos, X, Y)    -> (lnρ, lnne, lnχ, status)
mass_per_H(eos)     -> Float64
```

The test EOSs return `ρ_u = 1`, `lnne = lnρ − ln(μ m_u)` (fully ionized hydrogen convention, tests only) and `lnχ = lnρ + ln(κ₀)` with a constant `κ₀ = 1e-2 m² kg⁻¹` (tests only).

### 7.9 Gate 1 tests

| Test | File | Threshold |
|---|---|---|
| Python ↔ C++ shim ↔ Julia wrapper, direct solver, 500 points, `prec = 1e-10` in both | `test/test_gate1_direct.jl` + `oracles/python/witt_oracle.py` | ≤ 1e-8 relative in ρ, nₑ |
| Table vs direct, 10⁶ withheld points | `test/test_gate1_table.jl` | max `|Δ ln ρ|` ≤ 1e-5, max `|Δ ln nₑ|` ≤ 1e-4, max `|Δ ln χ|` ≤ 1e-4 |
| `∂ ln ρ/∂ ln P` vs a step sweep of the direct solver (`δ ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4}`, central differences; take the step with the smallest difference from the next-smaller step), 10⁴ points | same | ≤ 1e-3 absolute |
| No node with `EOS_DIRECT_FAILED`; no unflagged non-converged call | same | 0 |
| Opacity port vs Python | `test/test_opacity.jl` | ≤ 1e-12 relative |
| Opacity vs RH background opacity at 10 photospheric points, matched EOS | manual, `reports/gate1_rh_opacity.md` | ≤ 5 %; report deviations |

---

## 8. Input reader and Gate 0 audit (M0)

### 8.1 Reader (`IOInput.jl`)

Two input files exist:

| File | Content | Julia HDF5.jl shape |
|---|---|---|
| `IO/input-mhs-hse.nc` (production input, made by `make_input_mhs_hse.py`) | `x[504]`, `y[504]`, `z[177]` (m, top-first, −0.494 to +2.981 Mm), `temperature`, `B_x`, `B_y`, `B_z` float32, global attributes `P_ref_Pa = 103509.296875`, `gravity_m_s2 = 274.0`, `periodic_x = periodic_y = 1`, `period_x_m = period_y_m = 23999976.0`, `vertical_order = "top_first"` | `(177, 504, 504)` |
| `IO/BIFROST_en024048_hion_0_504_0_504.nc` (full snapshot, Gate 0 only) | `x`, `y`, `z[1,496]` (m, top-first), cubes `(1, 504, 504, 496)` float32: `temperature`, `gas_pressure`, `density`, `electron_density`, `B_x/y/z`, `velocity_x/y/z`, ... | `(496, 504, 504, 1)`; `z` is `(496, 1)` |

Both are NetCDF4, i.e. HDF5; read them with HDF5.jl. Dimension names come from the `DIMENSION_LIST` attribute; identify axes by the dimension-scale names `x`, `y`, `z`/`depth` exactly as `make_input_mhs_hse.py` does (do not assume the order). Implementation:

```julia
function read_cube(file, name, par, crop_k::UnitRange)   # returns Field (Float64, local block)
    ds = file[name]
    # map Julia dims (reversed C order) to (x, y, z) and drop a length-1 time dim
    # read only the hyperslab (gi0:gi1, gj0:gj1, crop_k) in the permuted order
    # convert Float32 → Float64 with the unit factor from attrs["units"]
end
```

Each rank reads only its own hyperslab plus nothing else (halos are filled by `exchange_halos!`). The file is opened read-only by every rank independently (`h5open(path, "r")`; no MPI-IO driver needed).

Validation (all collective, `invalid_input` on failure, with a precise message):

1. `z` strictly decreasing; `x`, `y` strictly increasing. A 2D `z` must have all rows equal to 1e-6 relative before reduction to `z[NZ]`.
2. Shapes of T and B agree with `(NX, NY, NZ)`.
3. `T > 0` and finite everywhere; B finite everywhere.
4. Periodic axes: uniform spacing to 1e-9 relative; no duplicated endpoint (§6.1).
5. Units recognized (§3.3).
6. `vertical_order` attribute, if present, equals `"top_first"`.

Crop (`crop_z_m = [zmin, zmax]`): keep levels with `zmin − 1 m ≤ z ≤ zmax + 1 m`, preserving order (same rule as the Python script). Resampling (D13) is an explicit adapter `resample_z(...)`: ln T linear in z and B components linear in z, onto a new strictly decreasing grid; it is applied only if `input.resample = "uniform_dz"` and recorded in `/meta/preprocessing`.

### 8.2 Gate 0 audit (`Gate0.jl`, `scripts/gate0_audit.jl`)

Runs on the full Bifrost snapshot cropped to `[−0.5, 3.0] Mm` (177 levels). It needs no EOS (proxy only), so it runs at M0.

Fields read: `temperature, gas_pressure, density, B_x, B_y, B_z, velocity_x, velocity_y, velocity_z` (convert to Float64 SI).

**Statistics convention.** Percentiles use `Statistics.quantile` type 7 (same as numpy's default). "4×4 horizontal subsampling" means statistics over global nodes with `i ≡ 1 (mod 4)` and `j ≡ 1 (mod 4)` (126 × 126 columns); derivatives are always computed on the full-resolution grid first. Target heights `{3.0, 2.0, 1.5, 1.0, 0.5, 0.0, −0.5}` Mm map to the nearest native level (record the actual `z`).

**Items:**

1. *Metadata report*: units of every field, frame (`local Cartesian, z up`), dimension order as read, `periodic_x/y` (from the FITS header card in the `fits_header` attribute or the input attributes), z-uniqueness check result, `NX, NY, NZ`, `Δx`, min/max Δz in the window.
2. *Per layer*: min/median/max of T, |B|, and β = 2μ₀P/|B|² (with `|B|` floored at 1 G), using P from both proxy anchor runs (item 3) and the native P. Also the fraction of nodes with `T > 3×10⁴ K` and the median of `ρ|v|²/P` (native ρ, P).
3. *Anchor sensitivity (proxy)*: with `r = ρ/P` frozen from Bifrost, integrate `du/dz = −g r` by the trapezoid rule on native levels:
   - top anchor: `u_1 = ln 0.1`, `u_{k+1} = u_k + ½ g (r_k + r_{k+1})(z_k − z_{k+1})` (downward);
   - bottom anchor: `u_NZ = ln P_bifrost(i, j, NZ)` **per column**, `u_k = u_{k+1} − ½ g (r_k + r_{k+1})(z_k − z_{k+1})` (upward).
   Report per target height the p1/p50/p99 of `log₁₀(P_hse/P_bifrost)`, the implied opposite-boundary pressure distribution, and the τ = 1 height map (only after M2, with the real opacity table; at M0 write "pending M2").
4. *Native residual* with native P, ρ, B and the operators of §6: per layer p50/p90/p99 of `|R_c| / (ρg)` for each component and of `|R| / (|∇P| + ρg + |L|)`; plus the **vertical-only** residual `|D_z ln P + ρg/P| / (ρg/P)` (D_z applied to `ln P`, which reproduces `numpy.gradient(lnP, z, edge_order=2)` used in the reference analysis). Label "baseline".
5. *Force-budget screening (proxy)*: per layer, `|∇_h P|` with `P` from the bottom-anchored proxy run, `|L_h(B_obs)| = sqrt(L_x² + L_y²)`, the mismatch `M = |∇_h P − L_h|` (vector difference), and `|δL| = |B| σ_B/(μ₀ ℓ)` with `ℓ = 2Δx` and `σ_B` from E10 and the synthetic σ model. Report p50 of each and the fraction of nodes with `|δL| ≥ M`. Label "screening". Re-run in M3 with `P_colHSE` from the real EOS.

Outputs: `reports/gate0_<run>.h5` (all per-layer tables as `NZ × ncol` datasets with column-name attributes) and `reports/gate0_<run>.md` (the tables of spec §3 at the seven target heights).

**Pass criterion (M0):** the markdown tables reproduce the reference values in spec §3 to within 0.1 dex (log quantities) or 10 % relative (ratios and fractions). A test (`test/test_gate0.jl`, skipped if the Bifrost file is absent) hard-codes the spec table and checks it.

---

## 9. `column_hse` (M2) (`ColumnHSE.jl`)

### 9.1 API

```julia
column_hse(ctx::RunCtx, T::Field; anchor::Symbol = :bottom, P_ref::Float64,
           z_ref::Float64 = NaN, g::Float64 = ctx.cfg.physics.g,
           n_sub::Int = 4, adapt::Bool = true, rk_tol::Float64 = 1e-6, n_sub_max::Int = 64,
           tau_top::Symbol = :exp_tail, H_max::Float64 = 5.0e6, mu::Float64 = 1.0) -> Result
```

`P_ref` has no default. The config may set `P_ref = "from_input"` to take the `P_ref_Pa` attribute of the input file (the user made that file); the source is recorded in `/meta/anchor`. Any other missing `P_ref` is `invalid_input`. If `anchor = :top` and `P_ref` is unset, the value 0.1 Pa is used (spec D2) and recorded.

A serial convenience method `column_hse(T::Array{Float64,3}, x, y, z; eos, kwargs...)` builds a single-rank `RunCtx` internally; it is what the unit tests use.

### 9.2 Per-column procedure

For each owned column `(i, j)` (threaded, §5.4):

1. Copy `X_k = ln T[i, j, k]` into a contiguous buffer `Xc[1:NZ]` (task-local, preallocated per chunk).
2. Integrate (§9.3) to get `u[1:NZ]` and `mraw[1:NZ]`, plus per-column flags and the max substep count used.
3. Node outputs from the table at `(X_k, u_k)`: `lnρ, lnnₑ, lnχ, status_k`.
4. Set `Pgas[k] = exp(u[k])`, **except** that at the anchor node `Pgas[anchor_k] = P_ref` exactly (`exp(log(P_ref))` is not always bitwise `P_ref`). For `:height` there is no anchor node.
5. `colmass[k] = mraw[k] − mraw[1]` (so `colmass[1] = 0` and it increases downward). Also `colmass_trap[k]` from the trapezoid of the output ρ, top-down, `colmass_trap[1] = 0`.
6. `rho = exp(lnρ)`, `ne = exp(lnnₑ)`, `Pe = ne K_B_WITT T`, `mu_bar = rho K_B_WITT T/(Pgas M_U_WITT)`, `nH_total = rho / mass_per_H`, `chi500 = exp(lnχ)`.
7. τ₅₀₀ (§10).
8. Write back to the `Field`s (strided writes are fine; this phase is EOS-bound).

If any table call in the column returns `EOS_OUT_OF_ENVELOPE`, stop integrating that column, set `COL_EOS_FAIL`, fill the column's outputs with `NaN`, and continue with the next column. After the loop, a collective count of failed columns decides `solver_status = eos_failure` (any failure) and the rebuild message of §7.3.

### 9.3 Integrator (RK4, ln T linear, co-integrated column mass)

The ODE system in `z`, with `X(z)` linear between nodes:

```
du/dz = f_u(z, u) = −g exp(Lρ(X(z), u) − u)
dm/dz = f_m(z, u) = −exp(Lρ(X(z), u))
X(z)  = X_a + (X_b − X_a)(z − z_a)/(z_b − z_a)       on the interval [z_a, z_b]
```

One interval step from node `a` to node `b` with `n` equal substeps (`h = (z_b − z_a)/n`, signed: positive when integrating upward, i.e. toward smaller k):

```julia
function rk4_interval(eos, Xa, Xb, za, zb, ua, ma, n, g)
    h = (zb - za) / n; u = ua; m = ma
    for s in 0:n-1
        z0 = za + s*h                        # compute z0 this way (not by accumulation) for reproducibility
        X0 = lerp(Xa, Xb, (z0 - za)/(zb - za)); Xh = lerp(Xa, Xb, (z0 + h/2 - za)/(zb - za))
        X1 = lerp(Xa, Xb, (z0 + h - za)/(zb - za))
        r1 = exp(Lρ(X0, u));               k1u = -g*r1/exp(u);            k1m = -r1
        u2 = u + h/2*k1u; r2 = exp(Lρ(Xh, u2)); k2u = -g*r2/exp(u2);   k2m = -r2
        u3 = u + h/2*k2u; r3 = exp(Lρ(Xh, u3)); k3u = -g*r3/exp(u3);   k3m = -r3
        u4 = u + h*k3u;   r4 = exp(Lρ(X1, u4)); k4u = -g*r4/exp(u4);   k4m = -r4
        u += h/6*(k1u + 2k2u + 2k3u + k4u)
        m += h/6*(k1m + 2k2m + 2k3m + k4m)
    end
    return u, m, status
end
```

(`exp(Lρ − u)` may be used instead of `exp(Lρ)/exp(u)` as long as one form is used everywhere; compute `r` once per stage.) `lerp(a, b, t) = a + (b − a)t`. At the last substep force `X1 = Xb` exactly.

Adaptive control per interval (when `adapt = true`):

```julia
n = n_sub
uc, mc = rk4_interval(..., n); uf, mf = rk4_interval(..., 2n)
while abs(uf - uc) > rk_tol && 2n < n_sub_max
    n *= 2; uc, mc = uf, mf; uf, mf = rk4_interval(..., 2n)
end
accept (uf, mf); if n > n_sub: flag COL_RK_REFINED; if abs(uf-uc) > rk_tol: flag COL_RK_TOL_NOT_MET
```

With `adapt = false` use exactly `n_sub` substeps (Gate 2 order tests).

Directions:

- `:bottom`: `u[NZ] = log(P_ref)`, `m[NZ] = 0`; for `k = NZ−1:-1:1` step from node `k+1` to node `k`.
- `:top`: `u[1] = log(P_ref)`, `m[1] = 0`; for `k = 2:NZ` step from `k−1` to `k`.
- `:height`: find `kr` with `z[kr+1] ≤ z_ref ≤ z[kr]` (error if outside the grid); `X_ref = lerp` in z. Step from `z_ref` up to node `kr` and down to node `kr+1` (one partial interval each, same adaptive rule, `X` linear over the original interval), then continue upward and downward as above. `m = 0` at `z_ref`.

### 9.4 Diagnostics

- Implied opposite-boundary pressure: the per-column `Pgas` at the plane opposite the anchor, with per-layer quantiles.
- Horizontal force residual of the column solution (B = 0): after `exchange_halos!(Pgas)`, `R = (Dx P, Dy P, Dz P + ρ g)`, stored as `R_colhse` (3 components) and normalized by `ρg`; per-layer p50/p99 of `|R_h|/ρg` and `|R_z|/ρg`. Label `vertical_hse_only`.
- RK statistics: per-column max substeps used; global count of `COL_RK_REFINED` and `COL_RK_TOL_NOT_MET` columns.

### 9.5 Gate 2 tests (`test/test_gate2.jl`)

| Test | Setup | Threshold |
|---|---|---|
| Isothermal ideal gas | `IdealGasIsothermal(μ = 1.3)`, T = 6000 K (H ≈ 140 km), 4×3×251 grid, z from +2 Mm to −0.5 Mm nonuniform (uniform 10 km spacing with random ±30 % jitter, fixed seed), both anchors | `|P/P_exact − 1| ≤ 1e-12` with `P_exact = P_ref exp(−(z − z_ref)/H)`, `H = kT/(μ m_u g)`. RK4 is not exact for exponentials (local error ≈ `(h/H)⁵/120`), so run with `adapt = false`, `n_sub = 64`; the expected global error is ≈ 2e-13. |
| Linear-T ideal gas | `IdealGasMu(μ = 1.3)`, T linear in z between 4000 and 8000 K | matches `ln P(z) = ln P_ref − (μ m_u g/k) Δz ln(T₂/T₁)/(T₂ − T₁)` per interval to 1e-8; fourth-order slope (≥ 3.9) of the error vs `n_sub ∈ {1, 2, 4, 8}` |
| Top-first indexing | any | `P` and `τ` strictly increase with k; `Pgas[:, :, anchor_k] == P_ref` bitwise |
| Mass identity | Wittmann table, Bifrost column sample (32 columns) | `|P(z) − P_top − g·colmass(z)|/P(z) ≤ 1e-6` |
| τ convergence | smooth analytic χ(z) = χ₀ e^{−z/Hχ}, grid halving ×4 | order ≥ 1.95 |
| Thread invariance | Bifrost 64×64 crop, 1 vs 8 threads, 1 vs 4 ranks | all outputs bitwise identical (`==` on the arrays, NaN-aware) |
| Anchor report | full Bifrost 385 with the table, both anchors | report only, `reports/gate2_anchor.md` |
| RH comparison | 3 columns, matched EOS and `m_top` | `|ΔP|/P ≤ 1e-3` (manual, `scripts/export_rh_atmos.jl`) |

---

## 10. τ₅₀₀ (`Tau.jl`)

Per column, with `χ_k` from the table (m⁻¹) and native z:

```
τ_1 = τ_top
τ_{k+1} = τ_k + ½ (χ_k + χ_{k+1}) (z_k − z_{k+1})          k = 1 .. NZ−1
```

Top closure (`tau_top`):

- `:exp_tail` (default): if `χ_1 > 0`, `χ_2 > χ_1` and `H_χ = (z_1 − z_2)/ln(χ_2/χ_1)` satisfies `0 < H_χ < H_max` (default 5.0e6 m), then `τ_top = χ_1 H_χ`, method `TAU_EXP_TAIL`. Otherwise `τ_top = ½ χ_1 (z_1 − z_2)`, method `TAU_HALF_CELL` (E1).
- `:half_cell`: always the half-cell value.
- `:zero`: `τ_top = 0`, `logtau500[1] = −Inf`, method `TAU_ZERO`.

Viewing angle `μ ≠ 1`: `tau500 = τ/μ` (label `per_column` in `/meta/tau/convention`). `logtau500 = log10(tau500)`.

Store `tau500`, `logtau500`, and the per-column `tau_top_method`. Report the per-method column counts in `/meta`.

---

## 11. Output writer and HDF5 schema (`IOOutput.jl`)

### 11.1 Writing protocol (no parallel HDF5 required)

1. Rank 0 creates the file, all groups and datasets at their global shapes (Julia dims `(NZ, NY, NX)`, `Float64`, chunk `(min(16,NZ), min(64,NY), min(64,NX))`, deflate level 4 with shuffle), writes `/input` (copied untouched from the source datasets: original dtype and units), `/meta`, all attributes, and closes the file.
2. Token ring: rank `r` waits for `MPI.Recv` from `r−1` (rank 0 starts), opens the file `"r+"`, writes its hyperslab `[1:NZ, j0:j0+ny−1, i0:i0+nx−1]` of every distributed dataset (the owned part only, permuted to `(z, y, x)`), closes the file, and sends the token to `r+1`. The last rank sends to rank 0; then `MPI.Barrier`.
3. If `HDF5.has_parallel()` and `cfg.output.parallel_hdf5 = true`, the same layout may be written collectively instead; the two paths must produce identical datasets (test on 4 ranks).
4. Never overwrite an existing file unless `run.overwrite = true`.

### 11.2 Schema

```
/input        temperature, B_x, B_y, B_z, x, y, z      (copies, original dtype and units)
/output       Pgas [Pa], rho [kg m-3], ne [m-3], Pe [Pa], nH_total [m-3], mu_bar [1],
              colmass [kg m-2] (+ attr m_top), colmass_trap [kg m-2], chi500 [m-1],
              tau500 [1], logtau500 [1]
              mhs only: Bx_out, By_out, Bz_out [T], dB_over_sigma_x/_y/_z [1]
/diagnostics  R_x, R_y, R_z [N m-3], R_over_S_x/_y/_z [1], R_par [N m-3], R_perp [N m-3] (magnitude),
              divB [T m-1], beta [1], eos_status [uint8 bitmask], tau_top_method [uint8, (x,y)],
              column_status [uint8, (x,y)], rk_max_substeps [int16, (x,y)],
              layer tables: layer_<name> [NZ × ncol] with attr "columns"
/meta         attributes and small datasets (below)
```

Every dataset has attributes `units` (string, SI; `"1"` for dimensionless), `dims` (C-order names, e.g. `["x","y","z"]`), and `long_name`.

`/meta` contents: `vertical_order = "top_first"`; `units_system = "SI"`; `mode`; `label` (`vertical_hse_only` or `mhs_fit`); `anchor` (`type`, `plane` or `z_ref`, `value`, `source`); `gauge` (MHS: `"mean u over anchor plane"`); `g`; `eos` (`hot_branch`, `T_sw`, `prec`, `maxit`, `table_sha256`, `pf_sha256`, `shim_sha256`, `source_sha256`, compiler); `composition` (99 abundances, 99 masses, `mass_per_H_kg`, source string); `constants` (both sets of §3.4); `tau` (`convention`, `top_method`, `H_max`, `mu`); `lambda` (all λ values, the continuation history table, selected λ_F); `sigma` (model string and parameters); `tolerances`; `solver_history` (per-iteration Φ, Φ terms, ‖g‖, step length, line-search evaluations, per λ stage); `flags` (all §13.10 flags); `git_hash` (`git rev-parse HEAD` if available, else `"unavailable"`); `package_versions` (from `Pkg.dependencies()`); `julia_version`; `nthreads`; `mpi` (`nranks`, `px`, `py`, MPI library version string); `config_toml` (verbatim text of the config file); `created` (ISO time); `preprocessing` (crop, resampling).

### 11.3 Checkpoint and restart (MHS)

Every `checkpoint_every` iterations (default 200) and at each λ stage boundary, write `<output>.ckpt.h5` (to a temporary name, then rename atomically on rank 0 after the token ring finishes):

- `/state/v`, `/state/bx`, `/state/by`, `/state/bz` (global layout, owned parts written by the token ring);
- attributes: `lambda_stage_index`, `lambda_F`, `iteration`, `Phi`, the continuation table so far, the bisection bracket, `config_sha256`, `table_sha256`.

Restart (`mhs.restart_from = path`): verify `config_sha256` (excluding the `restart_from` key itself) and `table_sha256`; read `x` with **any** process grid (each rank reads its own hyperslab of the global datasets); resume the λ schedule; L-BFGS history starts empty (E8). Test: interrupt a 32³ run at iteration 50, restart on a different process grid, and finish; the final classification must match the uninterrupted run and Φ must agree to 1e-6 relative.

---

## 12. Force-budget screening and column-HSE residual report (M3) (`Screening.jl`)

Inputs: `P_colHSE` from `column_hse` (bottom anchor by default), `Bobs`, σ model.

Per node (after halo exchange of `P`, `B`):

```
gradhP  = (Dx P, Dy P)
Lh      = (Lx, Ly) of Bobs (§6.7)
M       = |gradhP − Lh|
dL      = |Bobs| σ_B / (μ0 ℓ),  ℓ = 2 Δx,  σ_B = sqrt(σx² + σy² + σz²)   (E10)
ok      = dL ≥ M
```

Per layer: p50/p90/p99 of `|gradhP|`, `|Lh|`, `M`, `dL` and the fraction `ok`. Write to `/diagnostics/layer_screening` and `reports/screening_<run>.md`. Also store the 3D `ok` mask as `UInt8` if `screening.write_maps = true`. Label everything "screening" (never "feasibility").

Hand-case test (`test/test_screening.jl`): (a) uniform P and B = 0 gives `M = 0` and `ok = 1` wherever `dL ≥ 0`; (b) `P = P₀(1 + ε cos(kx))`, B = 0 gives `M = P₀ ε k |sin(kx)|` to second-order accuracy; (c) `B = (0, b cos(kx), 0)` gives `L_x = (b²k/2μ₀) sin(2kx)`, checked to second order.

---

## 13. `mhs` (M4a, M5)

### 13.1 API

```julia
mhs(ctx::RunCtx, T::Field, Bobs::NTuple{3,Field}; sigma::NTuple{3,Field},
    anchor_plane::Symbol = :bottom, P_ref::Float64, g = ctx.cfg.physics.g,
    lambda_D = 1.0, lambda_S = 1e-3, lambda_U = 1e-6,
    lambda_F = :discrepancy,             # or a Float64 to skip the selection
    init = :column_hse, data_kind = :real,  # :synthetic | :real (selects tolerance defaults)
    tol_rms = nothing, tol_p99 = nothing,
    lbfgs_m = 10, max_iter = 5000, gtol_rel = 1e-6, ftol_rel = 1e-10, ftol_window = 20) -> Result
```

`sigma` is either supplied as three datasets in the input file (real data) or built from the synthetic model `σ_c,n = sqrt((0.05 |Bobs_n|)² + (5e-4 T)²)` (|Bobs_n| is the field magnitude, the same σ for all three components; 5e-4 T = 5 G), selected by `mhs.sigma_B = "synthetic"`. `P_ref` has no default except via `"from_input"` (as in §9.1); for `anchor_plane = :top` the default is 0.1 Pa.

### 13.2 Variables and fixed data

State vector (four local `Field`s, owned part significant):

```
x = (v, b̃x, b̃y, b̃z)
u   = v − v̄ + ln P_ref,      v̄ = Σ_{(i,j)} α_ij v[i, j, k_a]     (global sum; k_a = NZ for :bottom, 1 for :top)
α_ij = h_x,i h_y,j / (Σ_i h_x,i Σ_j h_y,j)                           (Σ α = 1)
B_c = Bobs_c + σ_c b̃_c
```

Fixed arrays (computed once, halos exchanged once where marked H):

| Array | Definition |
|---|---|
| `X = ln T` | owned |
| `Bobs_c` (H), `σ_c` (H) | input; σ floored at 1e-12 T to avoid division by zero |
| `invBs2` | `1 / max(|Bobs|, 1e-4 T)²` |
| `invS2` | `1 / S²`, `S = max(ρ₀ g, 0.01 · median_k(ρ₀ g))`, ρ₀ from the initializer, median over layer k (exact, §5.6); fixed for the whole solve |
| `hx, hy, hz` (1D), `w` separable | §6.4 |
| `hmean` | `(h_x,i + h_y,j + h_z,k)/3`, computed on the fly |
| constants | `N = NX NY NZ` (global), `N_B = 3N` |

### 13.3 Initialization

1. `u₀` from `column_hse` with `anchor = anchor_plane` and the same `P_ref` (`:top` → column anchor `:top`). Then `v₀ = u₀` (the gauge is already satisfied: `u₀ = ln P_ref` on the anchor plane, so `v̄₀ = ln P_ref`).
2. `b̃ = 0` (so `B₀ = Bobs`).
3. `ρ₀ = ρ_EOS(T, P₀)`; compute `S` and `invS2`.
4. If `init = :fieldline_hse` (M4b, optional) use that `u₀` instead, followed by a gauge shift `v₀ = u₀ − ū₀ + ln P_ref`.

### 13.4 Forward evaluation `Φ(x)` (`MHSResidual.jl`)

Objective (spec §6.3), written in terms of row-local quantities:

```
Φ   = Φ_F + Φ_D + Φ_B + Φ_S + Φ_U
Φ_F = (λ_F/N)  Σ_n w_n |R_n|² invS2_n
Φ_D = (λ_D/N)  Σ_n w_n hmean_n² (divB_n)² invBs2_n
Φ_B = (1/N_B)  Σ_{c,n} b̃_{c,n}²                             (= Σ ((B − Bobs)/σ)²/N_B)
Φ_S = (λ_S/N)  Σ_n w_n Σ_{a∈x,y,z} Σ_c (h_a,n² (D2_a (σ_c b̃_c))_n)² / σ_c,n²
Φ_U = (λ_U/N)  Σ_n w_n Σ_{a∈x,y} (h_a,n² (D2_a u)_n)²
```

`D2_a(σ b̃)` is `D2_a(B − Bobs)`, i.e. the regularizer acts on δB.

Algorithm (each numbered step is a phase; MPI only between phases, on the main task):

1. `v̄ = global_sum(α ⊙ v[:, :, k_a])` (deterministic, §5.5).
2. Pointwise, owned: `u = v − v̄ + ln P_ref`; `B_c = Bobs_c + σ_c b̃_c`; `δB_c = σ_c b̃_c`.
3. `exchange_halos!([u, Bx, By, Bz, δBx, δBy, δBz])`.
4. Pointwise, owned + halo: `P = exp(u)`.
5. Pointwise, owned: `(lnρ, ρ_u, st) = lnrho_dY(eos, X, u)`; `ρ = exp(lnρ)`. If any `st & EOS_OUT_OF_ENVELOPE`, set the local flag `oob = true` (do not exit the loop).
6. Stencil kernel over rows `(j, k)` (threaded), for each owned `i`:
   ```
   gP   = (dx(P), dy(P), dz(P))
   J    = ((dy(Bz) − dz(By)), (dz(Bx) − dx(Bz)), (dx(By) − dy(Bx))) / μ0
   L    = J × B
   R    = gP + (0, 0, ρ g) − L
   dv   = dx(Bx) + dy(By) + dz(Bz)
   eF   = w |R|² invS2;   eD = w hmean² dv² invBs2;   eB = Σ_c b̃_c²
   eS   = w Σ_a Σ_c (h_a² d2_a(δB_c))² / σ_c²;   eU = w Σ_{a∈x,y} (h_a² d2_a(u))²
   partial[term, j, k] += e_term                       (sequential in i)
   # adjoint sources (stored, owned):
   aR_c   = 2 λ_F w invS2 R_c / N                      c = x, y, z
   aD     = 2 λ_D w hmean² invBs2 dv / N
   qS_a,c = 2 λ_S w h_a⁴ d2_a(δB_c) / (N σ_c²)         a = x, y, z; c = x, y, z
   qU_a   = 2 λ_U w h_a⁴ d2_a(u) / N                   a = x, y
   ```
7. Reduce partials: local sums in fixed order, then one `MPI.Allreduce` of `[ΦF', ΦD', ΦB', ΦS', ΦU', oob]` (oob with `max`). Then `Φ_F = λ_F/N · ΣeF`, etc. If `oob` is set globally, return `Φ = +Inf` (the line search treats this as a failed trial; §13.8).

The `aR`, `aD`, `qS`, `qU` arrays are the only extra storage the adjoint needs.

### 13.5 Gradient `∇Φ` (`MHSAdjoint.jl`)

Run immediately after a forward evaluation at the same `x` (the line search always needs both).

1. `exchange_halos!([aR_x, aR_y, aR_z, aD, qS_xx, qS_xy, ..., qS_zz, qU_x, qU_y])` (15 fields). `B` halos from the forward pass are still valid.
2. Stencil gather kernel over rows `(j, k)`, for each owned `i` (all `Dᵀ` in gather form, §6.6):
   ```
   # u-gradient
   g_u = P ⊙ (Dxᵀ aR_x + Dyᵀ aR_y + Dzᵀ aR_z)  +  aR_z g ρ ρ_u  +  D2xᵀ qU_x + D2yᵀ qU_y
   # B-gradient (force term)
   c_m   = B_m × aR_m         evaluated on the fly at every referenced neighbour m
   J     = curl B / μ0 at this node
   g_Bx  = −(aR × J)_x − (1/μ0) (Dzᵀ c_y − Dyᵀ c_z)
   g_By  = −(aR × J)_y − (1/μ0) (Dxᵀ c_z − Dzᵀ c_x)
   g_Bz  = −(aR × J)_z − (1/μ0) (Dyᵀ c_x − Dxᵀ c_y)
   # divergence term
   g_Bc += D_cᵀ aD            (Dxᵀ aD for Bx, Dyᵀ aD for By, Dzᵀ aD for Bz)
   # regularizer on δB (δB_c = σ_c b̃_c, so ∂δB_c/∂b̃_c = σ_c)
   g_δBc = Σ_a D2_aᵀ qS_a,c
   # chain to the optimizer variables
   g_b̃c = σ_c (g_Bc + g_δBc) + 2 b̃_c / N_B
   ```
   Here `D2ᵀ` is the transpose of `D2` in gather form. Write `g_u` to a work field and `g_b̃c` to the output gradient fields.
3. Gauge chain: `G = global_sum(g_u)` (deterministic). Then `g_v = g_u − α ⊙ G` on the anchor plane `k_a`, and `g_v = g_u` elsewhere.

Derivation check (keep in the code as a comment): with `δR = D δP + ẑ g δρ − δL`, `δP = P δu`, `δρ = ρ ρ_u δu`, `δL = δJ × B + J × δB`, `a·(δJ × B) = δJ·(B × a)` and `a·(J × δB) = δB·(a × J)`, the gradient of `Σ a·R` is exactly the expression above.

### 13.6 Gradient verification (gate for M4a)

`test/test_gradient.jl`, on 16³ boxes (periodic, nonuniform z), with (a) the ideal-gas test EOS and (b) the Wittmann table; `Bobs` a smooth random field (sum of 6 Fourier modes plus a uniform part), `T` horizontally structured, `x` random around the initializer (`v += 0.05·randn`, `b̃ += 0.5·randn`), fixed seeds:

1. Transpose tests for every `Dᵀ`, `D2ᵀ`: `|⟨D f, q⟩ − ⟨f, Dᵀ q⟩| ≤ 1e-13 · |⟨D f, q⟩|`.
2. Directional derivative: random unit direction `p` (each field normalized to unit global RMS); `fd(ε) = (Φ(x + εp) − Φ(x − εp)) / 2ε` for `ε ∈ {1e-2, 1e-3, …, 1e-8}`; require `min_ε |fd(ε) − ⟨∇Φ, p⟩| / |⟨∇Φ, p⟩| ≤ 1e-6`. Repeat with `p` restricted to each of the 4 fields, and with each Φ term alone (set the other λ's to 0) so that a failure points to its term.
3. Decomposition: the same test on 1, 2 and 4 ranks; ∇Φ bitwise identical across thread counts, within 1e-12 relative across rank counts.

### 13.7 L-BFGS (`MHSOptimize.jl`)

Two-loop recursion (Nocedal & Wright Alg. 7.4) on 4-field vectors, with `dot4` the deterministic global dot product over owned nodes (§5.5).

```julia
function lbfgs!(st, x, f, g)               # st holds S[1:m], Y[1:m], rho[1:m], ring index
    k = 0; f0 = f; g0n = sqrt(dot4(g, g)); fhist = ring buffer of the last ftol_window+1 values of f (plain Vector, no extra package)
    while true
        gn = sqrt(dot4(g, g))
        gn ≤ gtol_rel * g0n && return :converged
        k ≥ max_iter && return :max_iter
        # direction
        q = copy!(work_q, g)
        for i in newest→oldest: a_i = rho_i dot4(S_i, q); q -= a_i Y_i
        γ = isempty ? 1/gn : dot4(S_new, Y_new)/dot4(Y_new, Y_new)
        r = γ q
        for i in oldest→newest: β = rho_i dot4(Y_i, r); r += (a_i − β) S_i
        d = −r
        if dot4(d, g) ≥ −1e-12 * gn * sqrt(dot4(d, d))      # not a descent direction
            reset history; d = −g / gn
        end
        α0 = isempty(history) ? min(1.0, 1/gn) : 1.0
        (α, fnew, gnew, ok) = strong_wolfe(x, f, g, d, α0)
        if !ok
            if history was empty: nfail += 1; nfail ≥ 2 && return :stalled
            reset history; continue
        end
        s = α d; y = gnew − g
        sy = dot4(s, y)
        if sy > 1e-10 * sqrt(dot4(s, s) * dot4(y, y))       # curvature check
            push!(history, (s, y, 1/sy))                   # drop the oldest beyond m
        end
        x += s; f, g = fnew, gnew; k += 1
        push!(fhist, f)
        if length(fhist) == ftol_window + 1 && abs(fhist[1] − f) ≤ ftol_rel * abs(f)
            return :stalled
        end
        record (k, f, terms, gn, α, nevals) in history log; checkpoint if due
    end
end
```

All vector updates are threaded pointwise loops over owned nodes (no reductions inside). The history stores `S_i`, `Y_i` as 4-field vectors (`2m · 4 = 80` cubes for `m = 10`).

### 13.8 Strong-Wolfe line search (Nocedal & Wright Alg. 3.5 and 3.6)

`c1 = 1e-4`, `c2 = 0.9`, `α_max = 1e6`, at most 20 function evaluations per search (each one is a forward + adjoint).

```
φ(α) = Φ(x + α d),  φ'(α) = ⟨∇Φ(x + α d), d⟩
α_prev = 0; α = α0; i = 1
loop:
    evaluate φ(α), φ'(α)            (Inf if any node is out of the EOS envelope)
    if φ(α) = Inf or φ(α) > φ(0) + c1 α φ'(0) or (i > 1 and φ(α) ≥ φ(α_prev)):
        return zoom(α_prev, α)
    if |φ'(α)| ≤ −c2 φ'(0): return α
    if φ'(α) ≥ 0: return zoom(α, α_prev)
    α_prev = α; α = min(2α, α_max); i += 1
zoom(lo, hi):
    repeat:
        if φ(hi) finite: α = cubic minimizer from (lo, φ(lo), φ'(lo)) and (hi, φ(hi), φ'(hi))
        else:           α = lo + 0.1 (hi − lo)       (back off strongly from an Inf trial)
        safeguard: if α ∉ [min(lo,hi) + 0.1|hi−lo|, max(lo,hi) − 0.1|hi−lo|], α = (lo + hi)/2
        evaluate; if φ(α) > φ(0) + c1 α φ'(0) or φ(α) ≥ φ(lo): hi = α
        else: if |φ'(α)| ≤ −c2 φ'(0): return α
              if φ'(α)(hi − lo) ≥ 0: hi = lo
              lo = α
        if |hi − lo| < 1e-14 · max(1, |lo|) or evals exhausted: fail (return the best α with sufficient decrease if any, flagged ok=false otherwise)
```

Cubic minimizer: `d1 = φ'(lo) + φ'(hi) − 3 (φ(lo) − φ(hi))/(lo − hi)`, `d2 = sign(hi − lo) sqrt(d1² − φ'(lo) φ'(hi))`, `α = hi − (hi − lo)(φ'(hi) + d2 − d1)/(φ'(hi) − φ'(lo) + 2 d2)`; if `d1² − φ'(lo)φ'(hi) < 0`, use bisection.

Keep the trial point's `x`, `Φ`, `∇Φ` in the `x_trial`, `g_trial` buffers so that the accepted step needs no re-evaluation.

### 13.9 λ_F selection by the discrepancy principle

```
grid    = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
thresh  = 1 + 3 sqrt(2 / N_B)
χ²(λ)  = Φ_B at the L-BFGS solution for λ_F = λ  (= Σ ((B − Bobs)/σ)² / N_B)
```

1. Continuation: for `λ` in `grid` (increasing), warm-start from the previous solution (the first from the initializer), run L-BFGS, record `(λ, status, Φ terms, χ², iterations)`. Keep the solution of the last λ with `χ² ≤ thresh` in a 4-field buffer `x_ok`.
2. Let `λ_ok` be the largest grid value with `χ² ≤ thresh`. If it is `1e4`, select it and set the flag `lambda_F_at_upper_bound`. If no grid value passes, select `1e-4` and set `lambda_F_at_lower_bound`.
3. Otherwise bisect in `log λ` between `λ_ok` and the next grid value `λ_bad`: `λ_mid = sqrt(λ_ok λ_bad)`, warm-started from `x_ok`; update the bracket; stop when `λ_bad / λ_ok ≤ 1.5`. Select `λ_ok` and its solution.
4. The reported result is the selected solution, with the full continuation table in `/meta/lambda`.

Each stage writes a checkpoint (§11.3). For debugging, `lambda_F = <number>` runs a single stage.

The L-BFGS stopping test `gtol_rel` refers to `‖∇Φ₀‖` **at the start of each stage**.

### 13.10 Result classification and diagnostics

After the selected solution is final:

1. Recompute the forward pass; store `R`, `R/S` (components), `divB`, `B_out = B`, `dB/σ = b̃`, `Pgas = exp(u)`, and node outputs from the table (`rho`, `ne`, `Pe`, `nH_total`, `mu_bar`, `chi500`), `beta = 2μ₀ P / max(|B|, 1e-4)²`.
2. `b̂ = B / max(|B|, 1e-4 T)`; `R_par = b̂·R`; `R_perp = |R − R_par b̂|`.
3. `colmass`: trapezoid of the output ρ from the top (no HSE identity holds in MHS); τ₅₀₀ as in §10.
4. Per-layer tables (exact quantiles, §5.6): RMS and p50/p99 of `|R|/S`, of `|R_par|/S` and `R_perp/S`, of `|divB| h/|Bobs|`, of `|b̃|`, and the face layers `k = 1, NZ` listed separately.
5. Flags:
   ```
   solver_status        = converged | stalled | max_iter | eos_failure | invalid_input   (E5)
   tol_rms, tol_p99     = synthetic: 1e-6, 1e-4;  real: 1e-2, 1e-1  (overridable; recorded)
   equilibrium_tol_met  = ∀k: RMS_k(|R|/S) ≤ tol_rms  ∧  p99_k(|R|/S) ≤ tol_p99
   frac3                = #(|b̃_c,n| > 3) / N_B;   maxsig = max |b̃_c,n|
   data_admissible      = frac3 ≤ 0.01  ∧  maxsig ≤ 5
   regularizer_dominant = (Φ_S + Φ_U) > 0.01 Φ
   success_mhs          = equilibrium_tol_met ∧ data_admissible ∧ solver_status == converged
   approximate_only     = !success_mhs
   ```
   All flags are always written, including `lambda_F_at_upper_bound`, `lambda_F_at_lower_bound` and `regularizer_dominant`. For synthetic gates the E3 floor-relative test is evaluated by the test harness, not by this classification.

### 13.11 Memory (local, per rank, in field-equivalents of `(nx+2)(ny+2)NZ` doubles)

| Group | Fields |
|---|---|
| fixed: `X`, `Bobs`×3, `σ`×3, `invS2`, `invBs2` | 9 |
| state `x`, gradient `g`, `x_trial`, `g_trial`, direction `d`, `work_q` | 24 |
| L-BFGS history (m = 10) | 80 |
| forward work: `u`, `P`, `ρ`, `ρ_u`, `B`×3, `δB`×3 | 10 |
| adjoint sources: `aR`×3, `aD`, `qS`×9, `qU`×2 | 15 |
| `g_u` | 1 |
| `x_ok` (continuation) | 4 |
| **total** | **143** |

For the Bifrost crop (`N = 504·504·177 = 4.50×10⁷`) that is `143 × 8 × 4.5×10⁷ ≈ 51.5 GB` in total over all ranks, plus the EOS table (~0.15 GB per rank, replicated) and MPI buffers. It fits one 768 GiB Olivia node.

---

## 14. Synthetic benchmarks (Gate C) (`Synthetic.jl`)

Every benchmark is a function returning `(x, y, z, T, Bexact, Pexact, ρexact, eos)` on a requested grid, plus `Bobs = Bexact + noise` with `noise_c,n = σ_c,n · ξ`, `ξ ~ N(0,1)` from `Random.Xoshiro(seed)` drawn in **global index order** (so that it is decomposition-independent: generate it on each rank by skipping to the global linear index, or draw the full global array on every rank for boxes ≤ 128³). Use `IdealGasMu(μ = 1.3)` unless stated, `g = 274`. Grids: periodic x, y with `Lx = Ly = 4 Mm` unless stated; z from +2 Mm down to 0 with nonuniform spacing `z_k = z_top − (Z) · ζ_k`, `ζ_k = (k−1)/(NZ−1) + 0.1 sin(π (k−1)/(NZ−1))`. Refinement studies use `(NX, NY, NZ) ∈ {(16,16,17), (32,32,33), (64,64,65), (128,128,129)}`.

**Floor.** For each benchmark, compute `R_floor` (E3): evaluate the discrete residual `R` of the exact solution (`P = Pexact`, `B = Bexact`, `ρ = ρ_EOS(T, Pexact)`) with the §6 operators and take per-layer RMS and p99 of `|R|/S`.

| ID | Construction | Pass criteria |
|---|---|---|
| C1a | Horizontally uniform `T(z) = 6000 + 3000 tanh((z − 1.2 Mm)/0.2 Mm)` K, `Bobs = 0`, σ synthetic (5 G floor), no noise | `u_mhs − u_colhse` converges at order ≥ 1.8; per-layer RMS `|R|/S` ≤ `max(1e-6, 1.5 R_floor)`; `B_out` within `1e-3 σ` of 0 |
| C1b | Same T, `Bobs = (20 G, 0, 50 G)` uniform | same; discrete `L` is zero to rounding (row sums are 0) |
| C2 | Potential field `B = ∇φ`, `φ = (B₀/k) cos(k_x x) cos(k_y y) e^{k(z − z_top)}`, `k_x = 2π/Lx`, `k_y = 4π/Ly`, `k = sqrt(k_x² + k_y²)`, `B₀ = 0.01 T` (operator test, no optimizer) | `max|L| / (B₀² k/μ₀)` and `max|divB|/(B₀ k)` decrease at order ≥ 1.9 |
| C3 | C1b equilibrium plus `ε(−1)^{i+j+k}` added to `u` (ε = 1e-3) and `ε σ_c (−1)^{i+j+k}` added to each `B_c` (even `NX`, `NY`) | (a) `Φ(perturbed) − Φ(eq) ≥ 0.5 · (regularizer terms evaluated on the perturbation)` and > 0; (b) after L-BFGS at fixed λ_F = 1, the checkerboard amplitude `|⟨u, (−1)^{i+j+k}⟩|/N` and its B analogue fall by ≥ 10³. Failure → D9 fallback (staggered B / vector potential), and report |
| C4 | Vertical benchmark `B = (b(z), 0, 0)`: `P = P₀ e^{−(z−z_b)/H₀}`, `b = b₀ e^{−(z−z_b)/(2H_b)}`, `ρ = (P/H₀ + b²/(2μ₀ H_b))/g`, `T = P μ m_u/(ρ k)`; `P₀ = 1e4 Pa`, `H₀ = 200 km`, `H_b = 300 km`, `b₀ = 0.05 T`, `z_b` = bottom | (a) no noise: floor-relative tolerance and order ≥ 1.8; (b) with noise: discrepancy λ_F selected, `data_admissible = true`, RMS of `(B_out − Bexact)/σ` ≤ 1 |
| C5 | 2.5D `B = (0, b(x, z), 0)`, `b = b₀ (1 + ½ cos(2π x/Lx)) e^{−(z − z_b)/(2H)}`, `Q = Q₀ e^{−(z − z_b)/H}`, `P = Q − b²/(2μ₀)`, `ρ = Q/(H g)`, `T = P μ m_u/(ρ k)`, `H = 200 km`, `Q₀ = 1e4 Pa`, `b₀` chosen so that `max b²/(2μ₀) = 0.25 Q` (T then varies horizontally between 0.75 and 1 times `μ m_u g H/k`) | as C4 (a) and (b) |
| C6 | Neukirch & Wiegelmann (2019, Sol. Phys., DOI 10.1007/s11207-019-1561-0, arXiv:1911.11084) 3D MHS family: current `J = αB + f(z) ∇B_z × ẑ` (μ₀ absorbed as in the paper) with `f(z) = a (1 − b tanh((z − z₀)/Δz))`; pressure and density from the paper's expressions; pick a member with positive P and ρ everywhere (raise the background `p₀(z)` if needed) and set `T = P μ m_u/(ρ k)`. **Transcribe the formulas from the paper; do not reconstruct them from memory.** Before using it, verify the transcription: the continuum residual `|∇P + ρgẑ − J×B|`, evaluated with 6th-order central differences on a 256³ grid, must be ≤ 1e-6 of `ρg`. | Recover from `Bobs` with noise at σ: floor-relative tolerance, `data_admissible`, `success_mhs` with E3 tolerances; residual order ≥ 1.8 under refinement |
| C7a | Incompatible flux: C5 field plus a localized monopole-like bump in `B_z` (`δB_z = 50 σ exp(−r²/(2·(3Δx)²))` at mid-height), σ synthetic | `approximate_only = true`; `data_admissible = false` or `equilibrium_tol_met = false`; `divB` diagnostics peak at the bump |
| C7b | Too-small σ: C5 with `Bobs = Bexact · (1 + 0.1 sin(3 k_x x))` and σ divided by 100 | `approximate_only = true`; `lambda_F_at_lower_bound` or `equilibrium_tol_met = false` reported consistently |
| C7c | Horizontally structured `T(x, z)` with `B = 0`, σ = 1e-3 G floor (essentially fixed B): no static solution exists | `approximate_only = true`; per-layer `|R_h|/S` large and reported; `solver_status ∈ {converged, stalled, max_iter}` with no crash |

---

## 15. Test suite, file by file

All tests run with `julia --project -t 4 test/runtests.jl`; the MPI ones additionally with `mpiexec -n {2,4} julia --project -t 2 test/mpi/<file>.jl` (wrap with `MPI.mpiexec()` from MPI.jl). Tests that need the Bifrost file or the Wittmann library skip with a visible message if the file is missing; CI on the development machine must have both.

| File | Gate | Content |
|---|---|---|
| `test_units.jl` | M0 | unit table, rejects unknown units |
| `test_reader.jl` | M0 | reads `IO/input-mhs-hse.nc` 8×8×177 hyperslabs on 1 and 4 ranks, compares with h5py dumps; validation rejections (non-monotone z, duplicated endpoint, T ≤ 0) |
| `test_halo.jl`, `mpi/test_halo.jl` | M0 | §5.3 |
| `test_reductions.jl` | M0 | deterministic sums: bitwise equal on 1/2/8 threads; layer quantiles equal numpy's on a fixed array |
| `test_operators.jl` | M0/M4a | §6.2, §6.3, §6.6 |
| `test_gate0.jl` | Gate 0 | spec §3 table reproduction |
| `test_gate1_direct.jl` | Gate 1 | §7.9 |
| `test_gate1_table.jl` | Gate 1 | §7.9 |
| `test_opacity.jl` | Gate 1 | §7.7 |
| `test_gate2.jl` | Gate 2 | §9.5 |
| `test_tau.jl` | Gate 2 | E1 closure: an exponential χ gives `τ_top = χ₁H` exactly; a χ increasing upward falls back to half-cell and sets the mask |
| `test_screening.jl` | M3 | §12 hand cases |
| `test_gradient.jl` | M4a | §13.6 |
| `test_gateC.jl` | Gate C | C1–C7 at 16³ and 32³ (fast); the refinement studies to 128³ run with `HSEMHS_LONG=1` |
| `mpi/test_decomposition.jl` | M7 | column_hse bitwise identical on 1, 2, 4, 6 ranks; MHS (C5, 32³) on 1, 2, 4 ranks: Φ within 1e-12 relative at each of the first 20 iterations, identical final flags, outputs within 1e-8 relative |
| `mpi/test_restart.jl` | M7 | §11.3 |
| `test_allocations.jl` | all | 0 bytes allocated per forward/adjoint call after warm-up |

Every PR lists the rows it passes and the measured numbers (spec §9).

---

## 16. Configuration and entry points

### 16.1 Command line

```bash
julia --project=. -t auto scripts/run.jl config/bifrost385_column_hse.toml
mpiexec -n 16 julia --project=. -t 16 scripts/run.jl config/bifrost385_mhs.toml
julia --project=. -t auto scripts/build_eos_table.jl config/eos.toml
julia --project=. -t auto scripts/gate0_audit.jl config/gate0.toml
```

`scripts/run.jl` does: parse config → `init_parallel` → read input → memory preflight (§17.1) → load or build the EOS table → dispatch on `run.mode` → write output → `MPI.Finalize()`. Exit code: 0 for `converged`/`success`, 2 for `approximate_only`, 3 for `eos_failure`, 4 for `invalid_input`.

### 16.2 TOML schema (all keys; defaults shown; `REQUIRED` means no default)

```toml
[run]
mode = REQUIRED              # "column_hse" | "mhs" | "gate0" | "eos_table" | "screening"
output = REQUIRED            # path of the result .h5
overwrite = false
debug = false

[input]
file = REQUIRED
temperature = "temperature"
bx = "B_x"
by = "B_y"
bz = "B_z"
x = "x"
y = "y"
z = "z"
sigma_x = ""                 # dataset names for real-data σ (empty = use mhs.sigma_B model)
sigma_y = ""
sigma_z = ""
crop_z_m = []                # [zmin, zmax] or empty
resample = "none"            # "none" | "uniform_dz"
resample_dz_m = 0.0

[physics]
g = "from_input"             # number, or "from_input" (attribute gravity_m_s2); else REQUIRED

[eos]
lib = "deps/witt/libwitt"    # extension added automatically
pf_file = "wittman-eos/pf_Kurucz.input"
table_file = "cache/eos_table.h5"
build_if_missing = false     # building takes long; must be requested explicitly
prec = 1e-10
maxit = 5000
abundances = "wittmann_default"   # or a path to a text file of 99 log10 fractions
dlog10T = 0.002
dlog10P = 0.02
logP_range = []              # [lo, hi] in log10 Pa; empty = from Gate 0 report (§7.3)
refine_max_levels = 5

[column_hse]
anchor = "bottom"            # "bottom" | "top" | "height"
P_ref = REQUIRED             # number (Pa) or "from_input"; for "top" defaults to 0.1
z_ref_m = 0.0                # only for anchor = "height"
n_sub = 4
adapt = true
rk_tol = 1e-6
n_sub_max = 64

[tau]
top = "exp_tail"             # "exp_tail" | "half_cell" | "zero"
H_max_m = 5.0e6
mu = 1.0

[mhs]
anchor_plane = "bottom"      # "bottom" | "top"
P_ref = REQUIRED             # number or "from_input"; for "top" defaults to 0.1
data_kind = "real"           # "real" | "synthetic"  (tolerance defaults)
sigma_B = "synthetic"        # "synthetic" | "input"
sigma_rel = 0.05
sigma_floor_T = 5.0e-4
lambda_D = 1.0
lambda_S = 1.0e-3
lambda_U = 1.0e-6
lambda_F = "discrepancy"     # or a number
lambda_F_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
bisect_ratio = 1.5
lbfgs_m = 10
max_iter = 5000
gtol_rel = 1.0e-6
ftol_rel = 1.0e-10
ftol_window = 20
ls_c1 = 1.0e-4
ls_c2 = 0.9
ls_max_evals = 20
tol_rms = "auto"             # "auto" (from data_kind) or a number
tol_p99 = "auto"
init = "column_hse"          # "column_hse" | "fieldline_hse"
checkpoint_every = 200
restart_from = ""

[screening]
ell_factor = 2.0             # ℓ = ell_factor · Δx
write_maps = false

[parallel]
px = 0                       # 0 = MPI.Dims_create
py = 0

[output]
parallel_hdf5 = false
compression = 4
```

`Config.jl` parses into nested immutable structs, rejects unknown keys (catches typos), resolves `"from_input"` after the input is opened, and stores the verbatim TOML text for `/meta/config_toml`. A config hash (`SHA-256` of the canonicalized parsed config, excluding `run.output`, `run.overwrite` and `mhs.restart_from`) is used for restart checks.

### 16.3 Provided configs

- `config/bifrost385_column_hse.toml`: input `IO/input-mhs-hse.nc`, `anchor = "bottom"`, `P_ref = "from_input"` (103 509.3 Pa at z = −0.494 Mm, the median Bifrost P there; this is the user's file, and the user must still confirm the value in spec §10).
- `config/bifrost385_column_hse_top.toml`: `anchor = "top"`, `P_ref = 0.1` (anchor-sensitivity run).
- `config/bifrost385_mhs.toml`: `anchor_plane = "bottom"`, `P_ref = "from_input"`, `data_kind = "real"`, `sigma_B = "synthetic"` (flagged in `/meta/sigma` as having no observational meaning until the user supplies σ).
- `config/synthetic_C1a.toml` … `synthetic_C7c.toml`.

---

## 17. Memory preflight, performance, Olivia

### 17.1 Preflight

Before allocating, each rank computes `bytes = 8 · (nx+2)(ny+2)·NZ · nfields + table_bytes + buffers`, with `nfields` from §13.11 (MHS: 143) or for `column_hse`: T, Pgas, rho, ne, Pe, nH_total, mu_bar, colmass, colmass_trap, chi500, tau500, logtau500, R×3 = 15 (+ 2D masks). Ranks on the same node (`MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)`) sum their bytes; if the node total exceeds `0.85 · Sys.total_memory()`, all ranks stop with `invalid_input` and print the required memory and a suggested `ntasks-per-node`. Use `Int128` arithmetic for the products.

### 17.2 Performance notes

- Table lookups dominate `column_hse`; one Bifrost run is about `4.5×10⁷ nodes × (4 stages × 2–3 runs × n_sub 4–8)` ≈ 2–4×10⁹ evaluations. At ~50 ns each, that is minutes on a 256-core node.
- The MHS forward + adjoint touches ~40 fields per node; it is memory-bandwidth bound. Expect ~0.3–1 s per evaluation for the Bifrost crop on one Olivia node. Budget: up to 13 λ stages × ≤ 5000 iterations × ~1.2 evaluations.
- Optional optimization (only after Gate C passes, and only if it keeps bitwise results): since T is fixed, precompute per node the table X-interval index and the X-basis values `H0, H1, K0, K1` once; then each lookup only computes the Y-part.
- First-touch: allocate and zero each `Field` inside a threaded loop with the same `(j, k)` partition the kernels use, so pages land on the NUMA node of the thread that uses them.
- Pin Julia threads (`JULIA_EXCLUSIVE=1`, or `ThreadPinning.jl` in the job script only) and bind MPI ranks to NUMA domains.

### 17.3 Olivia job script (`scripts/olivia_job.sh`; values to be validated on the machine)

```bash
#!/bin/bash
#SBATCH --account=<project>
#SBATCH --job-name=hsemhs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00

module load <julia module> <MPI module verified for MPI.jl>
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_EXCLUSIVE=1
# one-time: julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
srun --cpu-bind=cores julia --project=. scripts/run.jl config/bifrost385_mhs.toml
```

Benchmark `16×16`, `32×8` and `8×32` (ranks × threads) and record the wall time per objective evaluation, peak RSS, communication fraction and EOS-lookup share in `reports/m7_scaling.md`.

---

## 18. Milestone execution plan

Each milestone ends with its gate report committed under `reports/` and the numbers quoted in the pull request.

| M | Build | Definition of done |
|---|---|---|
| **M0** | `Constants`, `Status`, `Config`, `Parallel` (init, decomposition, halos, reductions, quantiles, errors), `Fields`, `Geometry`, `Operators` (D, D2 and their transposes), `IOInput`, `IOOutput` (token ring, schema skeleton), `Gate0` | `test_units`, `test_reader`, `test_halo` (1–6 ranks), `test_reductions`, `test_operators` pass; `reports/gate0_bifrost385.md` reproduces spec §3 within 0.1 dex |
| **M1** | `deps/witt` shim and build, `WittmannDirect`, `SahaHot`, `Envelope`, `Opacity500`, `Table`, `TestEOS`; `scripts/build_eos_table.jl` | Gate 1 rows of §7.9 pass; `cache/eos_table.h5` built for the Bifrost envelope; `reports/gate1_*.md` |
| **M2** | `ColumnHSE`, `Tau`, full output schema; first Bifrost products (both anchors) | Gate 2 rows of §9.5 pass (RH comparison may follow as a manual report); `out/bifrost385_colhse_bottom.h5` and `_top.h5`; Gate 0 item 3 τ = 1 maps filled in |
| **M3** | `Screening`; column-HSE residual report | `test_screening` passes; `reports/screening_bifrost385.md` |
| **M4a** | `MHSResidual`, `MHSAdjoint`, objective assembly at fixed λ | `test_gradient` passes; Gate C1, C2, C3 pass at 16³/32³ (with the E3 floor rule) |
| M4b (optional) | `fieldline_hse` | not specified here; only if requested |
| **M5** | `MHSOptimize` (L-BFGS, line search, discrepancy principle, classification, checkpoint) | Gate C4–C7 pass, refinement studies to 128³ |
| M6 (conditional) | augmented Lagrangian or staggered B / vector potential | only if Gate C fails; write a design addendum first |
| **M7** | multi-rank validation, restart, Olivia runs | `mpi/test_decomposition`, `mpi/test_restart` pass; full Bifrost MHS run with per-layer residual and data tables in `reports/m7_bifrost385_mhs.md`; scaling report |
| M8 | 3D inclined rays, NLTE hook | separate design addendum |

**Order inside a milestone:** write the oracle/test first (by the other model, spec §9), then the code, then run on the smallest case, then scale up.

---

## 19. Decisions still owed by the user

1. **E3**: accept the floor-relative synthetic tolerance (§1, §14) or specify another rule. Without it, Gate C1 cannot pass by construction.
2. **E1, E2**: confirm the corrected `H_χ` formula and the RK4 co-integrated column mass (both are corrections of evident errors, but they change spec text).
3. Bottom `P_ref` for Bifrost-like cubes: the input file proposes 103 509.3 Pa at z = −0.494 Mm (spec §10 suggests 1.0×10⁵ Pa at −0.5 Mm).
4. Real-data σ_B and the real-data tolerances (`tol_rms = 1e-2`, `tol_p99 = 1e-1` are placeholders).
5. Downstream consumer (RH 1.5D, FIRTEZ, other) and whether to resample to 357 levels (D13).
6. Composition: keep the Wittmann default abundance vector (recorded in `/meta/composition`) unless overridden.

---

## 20. Implementation notes recorded during M0 (16 Oct 2026)

- Package lives in `HSEMHS/` (next to `IO/` and `wittman-eos/`); config paths are relative to `HSEMHS/`, so the default `eos.pf_file` is `../wittman-eos/pf_Kurucz.input`.
- Tests run with `julia --project=. -t 4 test/runtests.jl` (not `Pkg.test()`, which fails to resolve the `Test` stdlib with Homebrew Julia 1.13). The multi-rank part launches `test/mpi/mpi_tests.jl` on 2, 4 and 6 ranks through `MPI.mpiexec()`; set `HSEMHS_SKIP_MPI=1` to skip it.
- `MPIPreferences` is a regular dependency (needed by the Olivia setup step), not an extra.
- The z transposed stencil width is `W = max rows referencing a node`: 4 for `NZ ≥ 6`, 5 for `NZ = 5` (§6.6 assumed 4).
- New config section `[gate0]` (keys: `crop_z_m`, `stride`, `target_heights_m`, `P_top`, `sigma_rel`, `sigma_floor_T`, `ell_factor`, dataset names `pressure`, `density`, `vx`, `vy`, `vz`).
- `physics.g = "from_input"` falls back to the Bifrost FITS cards `GRAV`, `U_L`, `U_T` when the file has no `gravity_m_s2` attribute.
- Spec §3 inconsistency: median ρv²/P at −0.5 Mm is 0.0315 (independently recomputed with numpy on the same levels), not 0.04. All other §3 cells reproduce within tolerance.

## 21. Implementation notes recorded during M1 (16 Oct 2026)

**EOS defect found in the supplied Wittmann code (C++ and Python identical).** `gasc()` (P from pe, used for ρ and for the background partials, hence the H⁻/H opacity) uses H₂/H₂⁺ constants `g4 = 10^cmol0`, `g5 = 10^cmol1`, while `pe_pg()` (pe from P) uses `g4 = pe·10^cmol0`, `g5 = pe·10^cmol1`. Writing out the H, H⁺, H⁻, H₂, H₂⁺ balance (`f5 = f1² p_H'/K`, `p_H' = pe/fe`) reproduces `pe_pg`'s quadratic coefficients exactly with `c = pe/K`, so `pe_pg` is right and `gasc` is missing the factor `pe`. Consequence in the original code (measured on Bifrost 385, `reports/gasc_h2_check.md`): nₑ is unaffected (pe comes from `pe_pg`); ρ changes by ≤ 0.001 dex on the cube's own (T, P) but is wrong at high P (×2 at 3000 K, 1e8 Pa; ×185 at 5000 K, 1e8 Pa, reached only by top-anchored runs); χ₅₀₀ is too low by up to 1.36 dex below 3000 K and ~0.1 dex at 3000–3500 K (H and H⁻ partials), affecting 2.3 % of the cube's cells by more than 0.1 dex; the pressure closure is off by up to 44 %. The shim build applies `FIX_GASC_H2` (default on; `HSEMHS_FIX_GASC_H2=0` builds the original), which aligns `gasc` with `pe_pg`. The closure residual then drops to ≤ 1e-8 wherever the solver converges and is part of the convergence test. The Python oracle applies the same substitution. **User approval pending** (it changes the supplied EOS).

**Hot branch as built.** Instead of a separate H+He Saha model, the hot branch (`SahaHot.jl`) solves exactly the Wittmann equations (same partition functions, donors, molecules and H⁻) with a bracketed root in ln pe, no 1e±15 clamps, and cancellation-free quadratic roots. The direct solver fails where n(H⁺)/n(H) exceeds its clamp (T ≳ 15 kK at low P) and at P ≳ 1e9 Pa. Where both converge, the branches agree to ≤ 3e-9. T_sw follows §7.5, with the "all P" condition taken over the P samples where the direct solver converges. Below T_sw, nodes where the direct solver fails use the hot branch and are flagged `EOS_ROBUST_FALLBACK (0x20)`.

**Non-smooth EOS model.** U(T) is linearly interpolated on a 0.01-dex grid (201 points, 1e3–1e5 K), so ρ, nₑ and χ have slope jumps at those temperatures. The opacity code jumps at 7730 K (H⁻ treatment), 12 kK (COOLOP) and 30 kK (LUKEOP), and has kinks at the Mg I/Si I/Si II table switches and at COULFF's integer switches. The table therefore places a node pair (b − 1e-9, b) in ln T at every such breakpoint and uses one-sided X-derivative stencils that never cross a breakpoint (`model_breaks`, `xstencil` in `Table.jl`). Without this, Gate 1 cannot pass for nₑ and χ.

**Other.** The `Bucket` bin width ignores the 1e-9 breakpoint pairs. `DelimitedFiles` is a dependency (oracle I/O in tests). Above 1e5 K the partition functions are held constant (the Wittmann table ends there), in both branches.

## 22. Implementation notes recorded during M2 (16 Oct 2026)

- **E13**: Gate 2's analytic test `∫dz/T = Δz ln(T₂/T₁)/(T₂−T₁)` is for T linear in z, while §5.2 interpolates ln T linearly. The test uses `Δz (1/T₁ − 1/T₂)/ln(T₂/T₁)`.
- Substep doubling stops when 2n would reach `n_sub_max`, so with defaults the finest run uses 64 substeps.
- A top-anchored Bifrost run exceeds the table envelope in 5 % of columns (P > 3×10¹⁴ Pa at depth). Following §7.3 this is reported as `eos_failure`; the anchor report excludes those columns and counts them.
- `run.mode = "column_hse"` is implemented in `RunColumnHSE.jl`; anchor-sensitivity report: `scripts/gate2_anchor_report.jl`.

## 23. Implementation notes recorded during M4a (16 Oct 2026)

- The MHS objective and adjoint follow §13.4–13.5 exactly; verification is in `HSEMHS/reports/M4a_gate.md`.
- **L-BFGS additions beyond §13.7:**
  - Jacobi preconditioning: the initial inverse Hessian is γD⁻¹, with D the Gauss–Newton diagonal (`gn_diagonal!`), exact except for an upper bound on the Lorentz coupling.
  - A per-iteration step cap: |Δ ln P| ≤ 0.5 and |Δb̃| ≤ 5. Without preconditioning, C1b does not converge in 3000 iterations; with it, C1b converges in 331.
- Gate C2 is gated on the 128 → 256 order (the coarser grids are pre-asymptotic).
- Gate C3 (b) fails; analysis and options are in `HSEMHS/reports/M4a_gate.md`. **User decision pending.**
