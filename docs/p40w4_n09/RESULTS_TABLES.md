# P40W4-N09 numeric tables (generated from `verification.json`)

## 1. Configuration

| key | value |
| --- | --- |
| `conductivity` | 2.500000e+02 |
| `contact_columns` | [0.45, 0.75] |
| `damping` | 8.000000e+00 |
| `degree` | 1 |
| `density` | 8.000000e+03 |
| `drag_coefficient` | 5.000000e-01 |
| `fluid_diffusivity` | 2.500000e-02 |
| `fluid_nx` | 40 |
| `fluid_ny` | 4 |
| `fluid_temperature` | 1.000000e-02 |
| `height` | 5.000000e-02 |
| `length` | 2.000000e-01 |
| `nx` | 6 |
| `ny` | 16 |
| `particle_density` | 2.000000e+03 |
| `particle_radius` | 1.000000e-02 |
| `particle_safety` | 5.000000e-02 |
| `particle_steps_per_exchange` | 16 |
| `particle_velocity` | 2.000000e-02 |
| `poisson` | 3.000000e-01 |
| `solid_safety` | 4.000000e-01 |
| `start_gap` | 1.000000e-04 |
| `stiffness` | 5.000000e+03 |
| `t_end` | 4.000000e-02 |
| `tau` | 1.250000e+00 |
| `thickness` | 5.000000e-02 |
| `volumetric_heat_capacity` | 1.000000e+04 |
| `young` | 1.000000e+07 |

## 2. Rates and step counts

| quantity | value |
| --- | --- |
| `dt_fluid` | 2.500000e-04 |
| `dt_macro` | 2.500000e-04 |
| `dt_particles` | 1.562500e-05 |
| `dt_solid` | 3.125000e-05 |
| `ratio_fluid_over_particles` | 1.600000e+01 |
| `ratio_fluid_over_solid` | 8.000000e+00 |
| `ratio_solid_over_particles` | 2.000000e+00 |
| `macro_steps` | 160 |
| `n_panel_steps` | 4 |
| `n_solid_steps` | 8 |
| `n_dem_steps` | 16 |
| `events` | 4640 |

## 3. Degree scan

| p | ledger residual | abs. imbalance [J] | flux L2 (settled) | thermal mismatch | drag mismatch |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.697752e-04 | 2.345055e-05 | 1.862246e-02 | 7.484070e-06 | 1.346275e-05 |
| 2 | 4.950133e-04 | 1.733168e-05 | 1.494628e-01 | -1.025357e-03 | 2.176696e-05 |
| 3 | 1.395596e+06 | 4.886341e+04 | 2.184334e+00 | 1.306713e+00 | 2.696961e-05 |

## 4. Energy book (production degree)

| component | initial [J] | final [J] |
| --- | --- | --- |
| kinetic | 1.256637e-05 | 9.291378e-06 |
| potential | 0.000000e+00 | 0.000000e+00 |
| internal | 3.500000e-02 | 3.501648e-02 |
| dissipated | 0.000000e+00 | 1.025044e-05 |

* `initial_scale` = 3.501257e-02
* `residual` = 6.697752e-04
* `residual_settled` = 6.611278e-04
* `residual_thermal` = 4.707172e-04
* `residual_mechanical` = 5.550885e-01
* `flux_fluid_peak` = 4.282362e-01
* `contact_max_overlap` = 4.484056e-05
* `contact_impulse` = None
* `dissipated` = 1.025044e-05
* `drag_source_energy` = 1.643056e-05
* `drag_delivered_energy` = 1.643078e-05
* `max_panel_temperature` = 5.294673e-03
* `max_temperature_fluid` = 5.618442e-03
* `min_particle_velocity` = 1.719749e-02

## 5. Negative control (coupling switched off)

| field | secular drift | bounded oscillation |
| --- | --- | --- |
| fluid_thermal | 1.332268e-14 | 1.734723e-14 |
| panel_thermal | 1.386060e-14 | 1.711186e-14 |
| plate_elastic | 2.363674e-05 | 6.305973e-03 |
| particles_kinetic | 0.000000e+00 | 0.000000e+00 |
| **max drift** | 2.363674e-05 | 6.305973e-03 |

## 6. Exchange-operator order study

| p | h [s] | relative quadrature error | observed order |
| --- | --- | --- | --- |
| 1 | 2.500000e-04 | 4.599167e-05 | 1.996 |
| 1 | 1.250000e-04 | 1.154167e-05 | 1.996 |
| 1 | 6.250000e-05 | 2.890898e-06 | 1.996 |
| 2 | 2.500000e-04 | 1.078824e-05 | 2.032 |
| 2 | 1.250000e-04 | 2.620340e-06 | 2.032 |
| 2 | 6.250000e-05 | 6.453884e-07 | 2.032 |
| 3 | 2.500000e-04 | 1.036877e-05 | 2.011 |
| 3 | 1.250000e-04 | 2.566931e-06 | 2.011 |
| 3 | 6.250000e-05 | 6.386505e-07 | 2.011 |

gains: {"p1_over_p2": 4.263, "p2_over_p3": 1.04}

## 7. Gate verdicts

| criterion | value | threshold | cmp | verdict |
| --- | --- | --- | --- | --- |
| `contact_reciprocity` | 0.000000e+00 | 1.000000e-12 | below | **PASS** |
| `drag_transfer_balance` | 1.346275e-05 | 5.000000e-02 | below | **PASS** |
| `flux_continuity` | 1.862246e-02 | 2.000000e-02 | below | **PASS** |
| `interpolation_order_gain` | 1.353045e+00 | 4.000000e+00 | above | **FAIL** |
| `negative_control_conservation` | 2.363674e-05 | 1.000000e-06 | below | **FAIL** |
| `reconstruction_operator_order` | 4.263131e+00 | 4.000000e+00 | above | **PASS** |
| `total_energy_residual` | 6.697752e-04 | 1.000000e-03 | below | **PASS** |
| `vacuity_contact_engaged` | 4.484056e-05 | 0.000000e+00 | above | **PASS** |
| `vacuity_drag_engaged` | 1.643056e-05 | 0.000000e+00 | above | **PASS** |
| `vacuity_thermal_engaged` | 4.282362e-01 | 0.000000e+00 | above | **PASS** |

verdict = **FAIL**, counts = {'FAIL': 2, 'PASS': 8, 'WITHHELD': 0}, criteria digest = `34a58662782dcc37fae906d2a3af6d5464eb262c1b28faf5c4579b7b56b0cbdb`, git rev = `170acaa5fb2b4e25c329135bd828d90bbbd5eb7c`
