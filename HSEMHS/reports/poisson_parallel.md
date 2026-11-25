# Parallel efficiency of the Poisson modes — 25 Nov 2026

Branch `equilibrium-adjoint`. Goal (user, 25 Nov 2026): mhs-poisson-ltau, -logm and -z get faster when cores are added,
up to 8 Olivia nodes × 256 cores. Harness: `test/scripts/poisson_scaling.jl` (MODE, STEPS, POLISH, ADJ, PROFILE),
multi-node launcher `test/scripts/poisson_scaling_nodes.sh` (one rank per NUMA domain through `numa_bind.sh`).

## What limited it

- **Threads had little to do.** Nearly every threaded loop ran over the rank's own x columns (`for i in 1:nx`): 252
  on one rank, 32 on 8 ranks, 8 on 32 ranks, against 64–256 threads.
- **Serial work in the CG operator.** The broadcasts of `Aop!`/`Jop!`/`JTop!` (`@. out += …`, `jx .*= pr.wc`), the
  dense products of the Fourier preconditioner, the per-layer copies and sorts of `gmedians`, and the pack/unpack loops
  of the FFT transposes ran on one core.
- **Allocation.** Every operator product allocated ~10 arrays of 30–50 MB; the residual reverse pass (`resid_vjp`,
  used twice per Hessian–vector product by the polish and the adjoint) ~60. On many ranks the uncoordinated GC pauses
  add up: each rank waits at the next halo exchange for whichever rank is collecting.
- **Memory bandwidth per NUMA domain.** A hyades node has 8 NUMA domains (Zen 2: 16 cores, Zen 3: 8 cores each), each
  with its own memory controller. A single rank's arrays are all in one domain (first touch by the main thread), so
  its threads share one controller: 1 rank × 16 threads in one domain is as fast as 64 threads unpinned.

## Changes

- `Slab.@cols i j nx ny [setup] body`: threaded loop over all (j, i) columns in chunks (8 per thread), the body a
  closure through a function barrier (`foreach_col`; the first version, an inline `@threads` loop in the macro
  expansion, boxed variables and ran 50× slower). About 40 loops in PoissonTau, PoissonZ, their adjoints, Embedded,
  Adjoint and ViewTau use it. Loops that accumulated per x column (`sc`, `sf`, `refbar`) now accumulate per column:
  under `@cols` the columns of one x run on different threads.
- `tmap!`, `tsum`, `tdot`: threaded element-wise update and sum in fixed chunks (the same value on any thread count);
  `gdot`/`gnorm` use them.
- `Aop!`, `Jop!`, `JTop!`: threaded updates in the same order of operations, scratch slots on `TP` (`scratch`).
  `evaluate!`, `lorentz_tau!`, `resid_vjp`, `lorentz_tau_vjp`: named buffers on `TP` (`buf`, `zbuf`); `lorentz_tau_vjp`
  forms the curvilinear derivatives, J and the cotangents in one pointwise pass.
- Fourier preconditioner (τ, mass): Q·X and Qᵀ·Y in column chunks across threads. Height-grid preconditioner: every
  (kx, ky) mode is a work item (was: local ky only).
- `SlabFFT`: transforms per (plane, chunk of `FFT_ROWS` = 4 vertical rows), so a rank with few planes still uses its
  threads; the chunking depends only on the grid, not on the thread count. Threaded pack/unpack of the transposes.
- `gmedians`: the layer sets in one threaded pass over the columns; `gkth` sorts the sets in parallel. Its bisection
  counts stay serial: a threaded loop in each of its ≤ 66 rounds doubled the allocation of a column-hse-ltau solve at
  32 threads (4.9 → 10.3 volumes, over the 7.2 budget of `test_embedded.jl`); serial, 5.0.

Results are unchanged: the adjoint tests (every mode against finite differences, 2 and 3 ranks against one) pass, and
the work counters of the solves (CG iterations, operator products) are identical before and after.

## Timings, 252² (the adjoint benchmark's cases)

Short ltau solve (3 Gauss–Newton steps, no polish), hyades2 (Zen 2, 8 NUMA domains of 16 cores):

| Layout | solve |
| --- | --- |
| before, 1 rank × 64 threads | 23.8 s |
| 1 rank × 16 threads in one NUMA domain | 15.9 s |
| 2 ranks × 16, one domain each | 9.8 s |
| 4 ranks × 16 | 5.8 s |
| 8 ranks × 16 (whole node) | 3.8 s |

Full runs (polish 3, the adjoint benchmark's settings), one Zen 3 node, 8 ranks × 8 threads (one per NUMA domain):

| Mode | before (1 × 64): solve / adjoint | now: solve / adjoint |
| --- | --- | --- |
| mhs-poisson-ltau | 631 s / 555 s | 67 s / 52 s |
| mhs-poisson-logm | — | 168 s / 62 s |
| mhs-poisson-z | 281 s / 9.5 s | 12.8 s / 2.2 s |

The τ and mass-axis solves themselves take 2–4 s; the rest is the polish (3 Newton steps, exact Hessian–vector
products by central differences of `resid_vjp`, CG to 1e-10), now memory-bandwidth bound with no dominant kernel.

## Across nodes

MPI.jl's bundled MPICH has no InfiniBand (verbs) provider and runs over TCP, also on the 200 Gb/s IPoIB interface
`ibp65s0`: 151 µs one-way latency, and no gain across nodes (short ltau solve: 1 node 3.2 s, 2 nodes 2.7 s, 6 nodes
3.1 s). The DOCA OpenMPI RPM on the nodes has no UCX component either. `test/scripts/poisson_scaling_ompi.sh` runs over
InfiniBand: OpenMPI 5.0.8 built against the nodes' UCX 1.22 (`~/packages/openmpi-5.0.8-ucx`, `--with-ucx=/usr
--without-hcoll`, built with plain gcc 13: loadgcc.sh's `CC=mpicc` links the hcoll OpenMPI) and a Julia environment
whose MPI.jl uses it (MPIPreferences system binary; HDF5.jl on the system's serial libhdf5, since HDF5_jll's OpenMPI
build loads a second libmpi). Ping-pong hyades10–11: 1.2 µs one way, 24 GB/s at 8 MB.

mhs-poisson-ltau at 252², 25 Gauss–Newton steps + polish 3 (the full runs above), 8 ranks × 8 threads per Zen 3 node
(hyades7, 9, 10, 12–16), one rank per NUMA domain:

| Nodes (ranks) | solve | adjoint | solve + adjoint | speed-up |
| --- | --- | --- | --- | --- |
| 1 (8) | 63.5 s | 51.0 s | 114 s | 1 |
| 2 (16) | 35.6 s | 31.2 s | 67 s | 1.7 |
| 4 (32) | 28.1 s | 20.0 s | 48 s | 2.4 |
| 8 (64) | 23.6 s | 18.1 s | 42 s | 2.7 |

The work is the same on every layout (76 CG iterations, 6 evaluations). The polish and the adjoint scale; the
Gauss–Newton steps do not: their time grows from 2.6 s (1 node) to 4.0 s (8 nodes), and the short 3-step solve is
fastest on 2 nodes (1: 2.99 s, 2: 2.44 s, 4: 2.84 s, 8: 4.90 s). A profile of rank 0 on 8 nodes has its CG almost
entirely in MPI: the Alltoallv of the Fourier preconditioner's transposes (~13 ms each) and the scalar Allreduces of
the CG (~5 ms each), far more than the latency, so either the collectives at 64 ranks or waiting on slower ranks;
not yet separated.

## Layout for Olivia

One rank per NUMA domain: on the GH200 nodes one rank per Grace CPU (4 per node, 32 for 8 nodes), its threads on that
CPU's cores and its memory local. 252 x columns over 32 ranks is 7–8 columns per rank; the threading is over all
(y, x) columns, the transforms over (plane, row chunk), so 64–72 threads per rank have work.

## Open

- The Gauss–Newton steps beyond 2 nodes: time the transposes and reductions alone at 16–64 ranks against the
  per-rank compute (collective cost or load imbalance), then fewer or overlapped transposes.
- Fewer messages per CG iteration (batch the halo exchanges of one operator product).
- The polish dominates the τ and mass-axis cost: an inexact-Newton CG tolerance instead of 1e-10 per step would cut
  its Hessian–vector products (a numerical choice; not changed).
