# M3: force-budget screening and column-HSE residual

- Pressure: `out/bifrost385_colhse_bottom.h5` (column_hse, anchor bottom, Wittmann table). B: `../IO/input-mhs-hse.nc`.
- ℓ = 95.2 km; σ per component = sqrt((0.05 |B|)² + (0.0005 T)²); σ_B = √3 σ. **Screening only, not a feasibility certificate.**

## Force budget per layer (medians, N m⁻³)

| z [Mm] | \|∇ₕP\| | \|Lₕ(B)\| | mismatch M | \|δL\| | fraction \|δL\| ≥ M | M/\|δL\| p50 / p90 / p99 | median β |
|---|---|---|---|---|---|---|---|
| 2.98 | 4.61e-09 | 1.38e-08 | 2.07e-08 | 8.7e-06 | 1.000 | 0.00274 / 0.0155 / 0.089 | 0.0016 |
| 2.50 | 7.57e-09 | 1.61e-08 | 2.56e-08 | 9.6e-06 | 1.000 | 0.00307 / 0.016 / 0.0837 | 0.00204 |
| 2.00 | 2.15e-08 | 4.01e-08 | 6.72e-08 | 1.07e-05 | 1.000 | 0.00683 / 0.0292 / 0.125 | 0.00448 |
| 1.50 | 1.57e-07 | 2.1e-07 | 3.72e-07 | 1.18e-05 | 0.998 | 0.0307 / 0.124 / 0.481 | 0.0303 |
| 1.00 | 4.89e-06 | 2.43e-06 | 6.16e-06 | 1.47e-05 | 0.846 | 0.389 / 1.26 / 3.62 | 1.24 |
| 0.50 | 0.000338 | 1.7e-05 | 0.00035 | 2.08e-05 | 0.031 | 15.4 / 126 / 768 | 96.8 |
| -0.00 | 0.0118 | 8.59e-06 | 0.0118 | 9.51e-06 | 0.001 | 857 / 8.08e+03 / 4.31e+04 | 2.59e+04 |
| -0.49 | 0 | 5.76e-05 | 5.76e-05 | 2.5e-05 | 0.354 | 2.01 / 11 / 26.5 | 2.42e+04 |

## Column-HSE residual with B = 0 (label `vertical_hse_only`)

| z [Mm] | \|R_h\|/ρg p50 / p90 / p99 | \|R_z\|/ρg p50 / p90 / p99 | \|R_h\|/(\|∇ₕP\|+ρg) p50 / p99 |
|---|---|---|---|
| 2.98 | 39.5 / 248 / 671 | 9.22e-04 / 1.08e-02 / 8.26e-02 | 0.975 / 0.999 |
| 2.50 | 6.31 / 153 / 475 | 1.56e-03 / 1.92e-02 / 9.72e-02 | 0.863 / 0.998 |
| 2.00 | 2.23 / 17.9 / 231 | 3.70e-03 / 2.61e-02 / 1.03e-01 | 0.69 / 0.996 |
| 1.50 | 0.753 / 2.4 / 6.06 | 5.93e-03 / 2.09e-02 / 7.51e-02 | 0.43 / 0.858 |
| 1.00 | 0.261 / 0.627 / 1.33 | 6.66e-03 / 1.51e-02 / 3.37e-02 | 0.207 / 0.571 |
| 0.50 | 0.124 / 0.388 / 0.646 | 4.88e-03 / 1.00e-02 / 3.22e-02 | 0.11 / 0.393 |
| -0.00 | 0.136 / 0.49 / 0.833 | 3.79e-03 / 2.61e-02 / 4.87e-02 | 0.12 / 0.455 |
| -0.49 | 0 / 7.57e-16 / 1.51e-15 | 5.55e-04 / 1.56e-03 / 7.38e-03 | 0 / 1.51e-15 |

R_z is the discrete vertical residual of the RK4 solution (truncation level); R_h is the unbalanced horizontal pressure gradient that MHS must supply through Lₕ.

Full per-level tables: `/diagnostics/layer_screening`, `/diagnostics/layer_colhse_residual`.
