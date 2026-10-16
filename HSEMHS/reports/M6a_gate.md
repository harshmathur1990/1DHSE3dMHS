# M6a gate report: staggered-mesh operators (16 Oct 2026)

Code: `src/Staggered.jl`. Tests: `test/test_staggered.jl`, `staggered_checks` in `test/helpers.jl`, and `test/mpi/mpi_tests.jl`. Design: `HSE_MHS_D9_STAGGERED_ADDENDUM.md` §2–4.

| Check (addendum §10) | Result | Pass |
|---|---|---|
| div ∘ curl ≡ 0 (random A, B̄_z = 0.37 T, jittered nonuniform z), relative to \|B\|/h | 1.4e-16 | yes |
| Transposes, \|⟨Ax, y⟩ − ⟨x, Aᵀy⟩\|/(‖Ax‖‖y‖): C (with B̄_z) / C\* / H / G | 1.3e-16 / 3.8e-18 / 1.8e-18 / 5.7e-18 | yes |
| Same checks on 2, 4, 6 ranks, every process-grid shape | all within thresholds | yes |
| Curl of an analytic A on the smooth nonuniform grid, max error on all faces including the boundary faces | 3.0e-7 → 7.6e-8 → 1.9e-8 (orders 1.97, 1.99) | yes |
| Dual curl (J) of an analytic B, interior edges | 8.4e-13 → 2.4e-13 → 6.2e-14 (orders 1.83, 1.94) | yes (finest pair) |
| Full suite (M0–M6a) | all pass; collocated C3(b) for B marked `@test_broken` (the D9 trigger) | yes |

**Infrastructure changes.**
- `exchange_halos!` handles any z range per field (k = 0:NZ for z-faces and x/y-edges).
- `newfield0` allocates z-staggered arrays.
- `sdot` and `sdot_ones` are deterministic reductions over arbitrary z ranges.

**Bug found and fixed during M6a.** A first `@rows` macro, which expanded `Threads.@threads` inside macro-generated code, silently skipped some (j, k) rows. It was replaced by a plain higher-order `forrows(fun, kr, par)`; `@rows` is now only a fully escaped wrapper around it. The transposes and the div∘curl identity caught the bug.

**Next (M6b).** Objective and adjoint on this mesh (Lorentz-force face averages M and their transposes), Gauss–Newton diagonal, staggered vertical initializer, CG for A₀, then the gradient tests.
