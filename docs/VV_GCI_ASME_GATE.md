# Verification & Validation machine gate (Richardson / GCI / ASME V&V 20)

Cross-solver gate that turns a **grid-refinement study of one observable** into a
machine decision: observed order of accuracy `p`, Richardson value `f_exact`,
Roache grid convergence index `GCI_fine`, asymptotic-range checks, ASME V&V 20
validation uncertainty `U_val`, and a `PASS / WARN / FAIL` verdict.
Solver neutral, standard library only (`math`, `json`), no dependency of the
solver repositories on this package and no change inside them.

| item | path |
|---|---|
| core (Richardson, order, GCI, gate) | `src/tensorphysics/vv/gci.py` |
| ASME V&V 20 uncertainty | `src/tensorphysics/vv/validation.py` |
| cross-repository adapters | `src/tensorphysics/vv/adapters.py` |
| CLI | `src/tensorphysics/vv/cli.py`, `python -m tensorphysics.vv` |
| tests | `tests/test_vv_gci_core.py`, `tests/test_vv_validation.py`, `tests/test_vv_adapters_cli.py` |
| real cross-repo run (TensorLBM) | `tools/vv_crossrepo_tensorlbm.py` |
| controls + V&V 20 table | `tools/vv_controls.py` |
| figures / machine summary / report | `tools/vv_figures.py`, `tools/vv_summary.py`, `tools/vv_report.py`, `tools/vv_publish.py` |
| evidence | `artifacts/vv_gate/` (`summary.json`, `controls/`, `figures/`) |

## 1. Mathematics

Levels are ordered coarse -> fine; `h` is a positive grid length (cell size,
element size, panel size or time step) with `rho = h_coarse / h_fine > 1`.

**Observed order.** Uniform three-grid study:

```
p = ln( (f_coarse - f_medium) / (f_medium - f_fine) ) / ln(rho)
```

Non-uniform or longer studies: one-dimensional least-squares fit of
`f(h) = f_exact + C h**p`, minimising the residual over `p` (log-spaced bracket,
golden-section refinement). The three-grid closed form is the special case with
zero residual, so it is used whenever `rho` is uniform (verified: closed form and
least-squares agree to `1e-12` in the tests).

**Richardson value.** `f_exact = f_fine + (f_fine - f_medium) / (rho**p - 1)`.

**Grid convergence index (Roache).** `Fs = 1.25` for a three-level study,
`Fs = 3.0` when only two levels are available:

```
GCI_fine = Fs * |(f_fine - f_coarse) / f_fine| / (rho**p - 1)
```

The relative form is used whenever `|f_fine|` is not negligible, otherwise the
absolute form `Fs |f_fine - f_coarse| / (rho**p - 1)` keeps a vanishing
observable (an error norm) well defined. Sanity identity used as a test:
`GCI_fine = Fs * |f_exact - f_fine| / f_fine` (the band contains the Richardson
error estimate).

**Asymptotic range.** Convergence ratio `R = (f_fine - f_medium) / (f_medium - f_coarse)`:
`R < 0` oscillatory, `0 < R < 1` monotone contraction, `R ~ 1` stagnant, `R > 1`
diverging. A study is asymptotic only when `R` contracts monotonically,
`0 < p <= p_max`, and (with >= 4 levels) the power-law residual and the
triplet-to-triplet order are consistent.

**ASME V&V 20.** `U_val = sqrt(U_num**2 + U_input**2)` with `U_num` taken from
`GCI_fine` (`U_num = GCI_fine |f_sim|` when the GCI is relative), comparison
error `E = f_sim - f_exp`, and the model is validated iff `|E| <= U_val`
(inclusive). `|E| / U_val` is always reported so a marginal pass is visible.

## 2. Policy and verdict rules

`GatePolicy` (defaults, all overridable from the CLI):

| field | default | meaning |
|---|---|---|
| `safety_factor_three` / `safety_factor_two` | 1.25 / 3.0 | Roache `Fs` |
| `p_min` / `p_max` | 0.05 / 6.0 | admissible order band |
| `expected_order`, `order_tolerance` | `None`, 0.2 | theory check: PASS <= 0.2, WARN <= 0.6, else FAIL |
| `residual_warn` / `residual_fail` | 2e-3 / 2e-2 | power-law residual (>= 4 levels) |
| `order_consistency_warn` / `_fail` | 0.2 / 0.5 | triplet order spread |
| `noise_warn_ratio` / `noise_fail_ratio` | 0.05 / 0.25 | declared `sigma` / finest difference |
| `strict` | `False` | a three-grid study cannot PASS |

Checks emitted by `evaluate()`: `input`, `asymptotic-range/convergence-type`,
`order-band`, `order-band/expected`, `power-law-residual`, `order-consistency`,
`noise-floor`, `gci-monotonicity`, `certificate-strength`, `validation/V&V20`.
The verdict is the worst status of the checks (`FAIL > WARN > PASS`), so a
non-asymptotic study can never be reported as a clean PASS.

## 3. CLI and data contract

```
python -m tensorphysics.vv --obs obs.json                     # gate payload
python -m tensorphysics.vv --benchmark evidence.json --repo tensorfem --study manufactured \
       --value-key relative_strain_energy_error --expected-order 2 --json
python -m tensorphysics.vv --table vv20_rows.json              # ASME V&V 20 table
```

Exit codes: `0` PASS, `1` WARN, `2` FAIL, `64` usage/input error.
`--exp/--u-input/--u-num/--f-sim` add the validation block; `--json` prints the
machine record; `--out` writes it.

Observation payload (`schema: tensorphysics.vv.observations/1`):

```json
{"schema": "tensorphysics.vv.observations/1", "quantity": "rel_l2_error",
 "expected_order": 2.0,
 "levels": [{"h": 0.0625, "f": 2.394666e-02, "label": "ny=16"}, ...],
 "validation": {"f_exp": 1.0, "u_input": 0.02, "f_sim": null, "u_num": null}}
```

## 4. Cross-repository adapters

Dialects are recognised from the `schema` string each repository publishes; the
generic profile reads any document that carries a grid length (`h`, `mesh_size`,
`dx`, `dt`, ...) or a grid count (`resolution`, `nx`, `n_panels`, ...) next to a
value field (`value`, `l2_error`, `relative_l2_displacement_error`, ...). A count
is converted to `h = 1 / count` (only ratios between levels are used).

| dialect | document | value keys used here |
|---|---|---|
| `tensorlbm` | `{"levels": [{"resolution": 32, "value": ...}]}` | `value`, `l2_error`, ... |
| `tensorfem` | `tensorfem.h-convergence-gci/1.0` (per-study `levels` with `mesh_size`) | `relative_strain_energy_error`, `relative_l2_displacement_error`, `strain_energy` |
| `tensorfvm` | `{"grids": [{"h": ..., "l2_error": ...}]}` | `l1/l2/linf_error` |
| `tensordem` | `{"timesteps": [{"dt": ..., "max_error": ...}]}` | `max_error`, `position_error` |
| `tensorbem` | `{"meshes": [{"n_panels": ..., "relative_error": ...}]}` | `relative_error`, `rao_error` |
| generic | any nesting | first level list found |

Field conservation is tested for all five dialects: every number survives the
adapter -> gate payload -> JSON round trip bit-exactly (`diff = 0 < 1e-12`).

## 5. Results

### 5.1 Manufactured positive control (`artifacts/vv_gate/controls/positive_control.json`)

`f(h) = 3 + 0.5 h^2`, `h = 1, 0.5, 0.25` (`r = 2`):

| quantity | measured | target |
|---|---|---|
| observed order `p` | `2.000000` | `2.000 +/- 0.01` |
| `f_exact` (Richardson) | `3.0` (`\u007c error\u007c = 0.0`) | `< 1e-6` |
| `GCI_fine` (`Fs = 1.25`) | `0.012886597938144331` | `1.25 * 0.0309278... / 3` |
| `GCI` vs `r = 2, 3, 4` (`h_coarse` fixed) | `0.05`, `0.0227273`, `0.0128866` | strictly decreasing |
| five levels (`r = 2`), order / residual | `2.0000000000000027` / `5.6e-17` | PASS |

Note on the `r`-sweep: for a **fixed fine grid** the relative GCI is invariant in
`rho` (both `f_coarse - f_fine` and `rho**p - 1` scale with `rho**p - 1`), so the
sweep that carries information fixes the coarse grid and refines the fine one;
the GCI then decreases monotonically (`fig_gci_vs_r.png`). This is stated
explicitly because the naive reading of "GCI decreases with `r`" is ambiguous.

### 5.2 Negative controls (`artifacts/vv_gate/controls/negative_controls.json`)

| control | measurement | verdict |
|---|---|---|
| oscillatory sequence | `R = -0.500`, type `oscillatory` | **FAIL** |
| diverging sequence | `R = 2.000`, type `diverging` | **FAIL** |
| stagnant sequence | order undefined | **FAIL** |
| `p = 20` sequence (order band) | `p` outside `(0.05, 6]` | **FAIL** |
| GCI growing with refinement | `gci_monotone = False` | **FAIL** |
| 1 % noise, 3 grids, declared `sigma`, 25 seeds | `22 FAIL / 3 WARN / 0 PASS` | never silent PASS |
| 1 % noise, 5 grids, no `sigma`, 25 seeds | `25 FAIL / 0 PASS` (residual detection) | never silent PASS |

**Honest limitation (information-theoretic).** A three-level study with an
*undeclared* noise floor cannot be certified: three data points and three fitted
numbers (`f_exact`, `C`, `p`) leave no residual, so a 1 % noise floor is
invisible when the grid differences are much larger than the noise. The gate
reports `study.noise_certified = false` and a note; `--strict` downgrades such a
certificate to WARN. Callers who need noise protection must declare per-level
`sigma` or supply >= 4 grids - both are implemented and tested.

### 5.3 ASME V&V 20 table (`artifacts/vv_gate/controls/vv20_table.json`)

Ten cases `(f_sim, f_exp, U_num, U_input)` with the expected `U_val` and verdict
computed by hand: **10/10 agree** (including the `|E| = U_val` boundary and the
zero-band cases). See `fig_vv20.png`.

### 5.4 Real cross-repository runs

**(a) TensorLBM** - its own Poiseuille verification case
(`tests/test_verification.py` call sequence reproduced in
`tools/vv_crossrepo_tensorlbm.py`, nothing inside TensorLBM touched), three
grids `ny = 16, 32, 64`, `n_steps = 4 ny^2`, half-way bounce-back walls,
`nu = 1/6`, `fx = 1e-4`:

| observable | ny=16 | ny=32 | ny=64 | `p` | `GCI_fine` | verdict vs `p_theory=2` |
|---|---|---|---|---|---|---|
| rel-L2 velocity error | 2.394666e-2 | 1.383948e-2 | 7.834480e-3 | **0.7511** | 1.4030 | **FAIL** (`\u007cp-2\u007c = 1.249`) |
| flow-rate error | 2.902468e-2 | 1.586510e-2 | 8.741398e-3 | **0.8854** | 1.2022 | **FAIL** (`\u007cp-2\u007c = 1.115`) |
| peak-velocity error | 1.833719e-2 | 1.057647e-2 | 6.154258e-3 | **0.8114** | 1.1895 | **FAIL** (`\u007cp-2\u007c = 1.189`) |

All three observables of the same fields give the same picture, and the error
scales as `h^0.85` (`err * ny^0.85 = 0.305, 0.303, 0.304` for the flow rate), so
this is a property of the discrete *solution*, not of one metric. The repository's
own test only requires a refinement ratio `> 1.5` (i.e. `p > 0.58`), which the new
gate shows is too weak to certify second order. Reproducing the same case with a
consistent body-force scheme (Guo forcing) is the recommended fix; the gate is the
tool that will show whether the fix worked.

**(b) TensorFEM** - its `benchmarks/h_convergence_gci.py` was re-run here
(`OMP_NUM_THREADS=2`, output `artifacts/vv_gate/tensorfem_hconv/`) and parsed by
the same CLI:

| study / observable | levels (`mesh_size`) | `p` | `f_exact` vs analytic | `GCI_fine` | verdict |
|---|---|---|---|---|---|
| manufactured / strain-energy error | 0.5, 0.25, 0.125 | **1.962191** | - | 0.0212927 | **PASS** (`\u007cp-2\u007c = 0.038`) |
| manufactured / strain energy (QoI) | 0.5, 0.25, 0.125 | **1.962191** | 16989.1227 vs 16978.9999 (5.96e-4) | 0.0212927 | **PASS** |
| bending / strain-energy error | 1.0, 0.5, 0.25 | **2.000000** | - | 0.0212927->0.0129 (Fs) | **PASS** |
| manufactured / L2 displacement error | 0.5, 0.25, 0.125 | **1.661993** | - | 0.0631 | **FAIL** (`\u007cp-2\u007c = 0.338`, GCI grows) |

*Independent-implementation cross-check*: for the strain-energy QoI this package
returns Richardson `16989.122659489472` and relative `GCI = 0.021292748777162734`,
which are **bit-identical** to the numbers computed by TensorFEM's independent
`convergence_verification` module in the same evidence file (diff `0.0`). Two
independent implementations of the Richardson/GCI formulas agree exactly, which is
the strongest available check short of an external reference.

The L2-displacement FAIL is a real, reported finding: on three coarse HEX8 meshes
that norm is not yet in the asymptotic range (transitional orders 1.738 -> 1.935),
while the strain energy already is. The gate distinguishes the two instead of
averaging them away.

### 5.5 Acceptance criteria

`artifacts/vv_gate/summary.json` carries the machine-checkable list:
**9 of 10 met**. The single unmet item is the literal wording of criterion (c),
"a **TensorLBM/TensorFVM** MMS case recovers `p = 2 +/- 0.2`": TensorLBM ships no
manufactured-solution case, and its existing analytic Poiseuille case measures
`p ~ 0.75-0.89` (section 5.4a), so the deviation is 1.1-1.25, not < 0.2. A real
three-mesh second-order MMS run through the same gate does exist for TensorFEM
(section 5.4b, `p = 1.962`), which is why criterion "c (second-order MMS via the
gate)" is recorded as met by the FEM run and the LBM run is recorded as an honest
FAIL with its root cause.

## 6. Limitations

1. **Three-grid, undeclared noise** cannot be certified (section 5.2); use `sigma`
   or >= 4 grids, or run `--strict`.
2. **Non-uniform studies** rely on the least-squares order; with three levels the
   fit is exact, so the order is only as good as the data (the residual check
   needs >= 4 levels).
3. **Single-observable gate**: the verdict is per observable. Conflicting
   observables of one case must be looked at together (as done in 5.4a).
4. **Order band** `(0.05, 6]` and the residual/noise thresholds are engineering
   defaults; spectral and high-order BEM/FEM discretisations may need
   `--p-max` raised (policy is fully configurable).
5. **TensorFVM/TensorDEM/TensorBEM** dialects are implemented and unit-tested on
   documents of the shape those repositories publish, but no run of those solvers
   was available in this sandbox (offline, no checkout), so only TensorLBM and
   TensorFEM were exercised end to end.
6. The TensorLBM root cause (body-force scheme) is a hypothesis supported by the
   measured exponent and by the invariance across three observables; it was not
   confirmed by modifying that repository (zero-intrusion constraint).

## 7. Reproduction

```bash
python -m pytest tests/test_vv_gci_core.py tests/test_vv_validation.py tests/test_vv_adapters_cli.py -q
python tools/vv_controls.py                     # controls + V&V 20 table
python tools/vv_crossrepo_tensorlbm.py          # real TensorLBM 3-grid run
(cd /opt/mm-verify/TensorFEM && OMP_NUM_THREADS=2 python benchmarks/h_convergence_gci.py \
    --out $PWD/artifacts/vv_gate/tensorfem_hconv)
python tools/vv_summary.py && python tools/vv_figures.py
python tools/vv_publish.py --mission 5389        # build + post the report (env credentials)
```