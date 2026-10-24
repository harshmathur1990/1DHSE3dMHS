# Run modes

One entry point for every mode, MPI + threads in all of them:

    [mpiexec -n R] julia --project=. -t T scripts/run.jl config/modes/<mode>.toml      (from HSEMHS/)

| `run.mode` | grid | what it computes | MPI decomposition |
|---|---|---|---|
| `column-hse-z` | height | hydrostatic columns, anchor P_ref (`column_hse` is an alias) | 2-D (x, y) blocks |
| `column-hse-ltau` | log10 τ500 | hydrostatic columns in τ: P(τ), z(τ), ρ(τ); aligned on P + B²/2μ₀ | x slabs |
| `mhs-poisson-z` | height | column HSE, then the Poisson-type pressure step below `poisson.z_cut_m` (T, B fixed) | x slabs |
| `mhs-poisson-ltau` | log10 τ500 | column HSE in τ, then one Gauss–Newton solve on the τ grid for ln P(τ) and the column heights | x slabs |
| `full-mhs-z` | height | full staggered MHS joint GN solve of P and B (the M5 driver; defaults = the M5i run) | 2-D (x, y) blocks |

Results do not depend on the rank count beyond the rounding of sums: quantiles and medians are exact global order
statistics, the Poisson preconditioners use a distributed FFT (rfft in y, all-to-all transpose, FFT in x). Output is
written cooperatively (no parallel HDF5): rank 0 creates each file, the ranks then write their slabs in turn.

## Outputs

* `column-hse-z`: the column HSE file (`/output`, `/diagnostics`, `/input`).
* `mhs-poisson-z`: MHS checkpoint layout, `/atmosphere/{Pgas, rho, T, tau500, B*_centre, Pgas_column_hse, z_tau1_m}`,
  `/coords/{x_m, y_m, z_m}`; Julia (x, y, z), h5py (z, y, x).
* `column-hse-ltau`, `mhs-poisson-ltau`: `/tau_grid/{logtau500, z_m, Pgas, rho, T, Bx, By, Bz, z_tau1_m}`,
  `/coords/{x_m, y_m}`; Julia (x, y, τ), h5py (τ, y, x); z gauged to mean z(τ = 1) = 0.
* `full-mhs-z`: `run.output` is a directory: `history.csv`, `candidates.csv`, `init.h5`, checkpoints
  `atmosphere_A/B.h5`, `atmosphere_final.h5`, `ztau1/`; restart with `full_mhs.restart`.

## τ-grid inputs

`IO/input-mhs-ltau.nc` layout: `log10tau500` (nt, increasing, top level first), `temperature`, `B_x`, `B_y`, `B_z` as
h5py (x, y, τ) in SI, no coordinates or attributes — set `input.dx_m` (and `input.dy_m` if different) and a numeric
`physics.g`.

## Multi-node on hyades

Nodes with an IPoIB `ib*` interface only (not hyades2/3/12/16/19), at most 4, nobody else on them. Each rank goes
through `~/.julia/bin/mhs_rank.sh` (libfabric TCP on the `ib*` interface, node-local `/tmp/julia-1.13.1`):

    export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
    printf "hyades4:2\nhyades5:2\nhyades15:2\nhyades21:2\n" > hosts
    mpiexecjl --project=. -launcher ssh -launcher-exec $HOME/.julia/bin/ssh_bash_launcher -f hosts -n 8 \
        mhs_rank.sh julia --project=. -t 32 scripts/run.jl config/modes/mhs-poisson-ltau.toml
