# M0 gate report (16 Oct 2026)

Julia 1.13.1, MPI.jl 0.20 (MPICH 5.0.2), HDF5.jl 0.17, macOS (14 cores).

| Check | Result |
|---|---|
| Units / config (`test_units_config.jl`) | 16/16 pass |
| Operators: exactness on 1, s, s² (both z orders), order ≥ 1.8 interior and boundary, periodic order ≥ 1.95, rejections | 24/24 pass |
| `numpy.gradient(f, z, edge_order=2)` oracle | max diff ≤ 1e-13 relative |
| 3D transposes ⟨Df,q⟩ = ⟨f,Dᵀq⟩ for Dx, Dy, Dz, D2x, D2y, D2z | ≤ 1e-13 on 1, 2, 4, 6 ranks, all process-grid shapes |
| Lorentz force of B = (0, b cos kx, 0), uniform B gives L = 0 | order ≥ 1.9; uniform L < 1e-12 relative |
| Halo exchange | exact on 1, 2, 4, 6 ranks |
| Deterministic sums / exact layer quantiles (= Statistics.quantile type 7) | bitwise on 1 rank; quantiles identical on all rank counts |
| Collective error propagation (one rank fails) | all ranks report invalid_input |
| Reader (fixtures, Bifrost-like 4D layout, crop, rejections, production input hyperslab) | 17/17 pass, 2–6 ranks |
| Token-ring HDF5 writer and schema | pass on 1–6 ranks |
| Gate 0 reproduction of spec §3 | 39/39 pass (one spec cell corrected: ρv²/P at −0.5 Mm = 0.0315, not 0.04) |
| Gate 0 decomposition invariance (1 rank vs 2 ranks) | all per-layer tables bitwise identical |

Gate 0 full run: 504×504×177, 2 ranks × 6 threads, 13 s. Report: `reports/gate0_bifrost385.md` / `.h5`.
