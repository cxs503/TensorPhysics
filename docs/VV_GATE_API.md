# V&V gate - API reference

Complete reference for the verification & validation machine gate shipped as
`tensorphysics.vv`. Solver neutral: the package imports the standard library
only (`math`, `json`, `subprocess` is not used at runtime), so the solver
repositories never take a dependency on this package and this package never
imports a solver.

| module | contents |
|---|---|
| `tensorphysics.vv.gci` | Richardson extrapolation, observed order, GCI, asymptotic checks, PASS/WARN/FAIL gate |
| `tensorphysics.vv.validation` | ASME V&V 20 validation uncertainty and the `\|E\| <= U_val` test (single case + table driver) |
| `tensorphysics.vv.adapters` | benchmark-document normalisation for TensorLBM / TensorFVM / TensorFEM / TensorDEM / TensorBEM |
| `tensorphysics.vv.cli` | command line front end (`python -m tensorphysics.vv`, `python -m tensorphysics.vv.gci`) |

Everything below is exported from `tensorphysics.vv` unless it is marked
module-private.

```python
from tensorphysics.vv import (
    GatePolicy, GateVerdict, GridConvergenceStudy, Observation, ValidationResult,
    evaluate, evaluate_validation_table, grid_convergence_study,
    observed_order, richardson_extrapolated, validate, validation_uncertainty,
)
```

---

## 1. Data model (`tensorphysics.vv.gci`)

### `Observation`

```python
@dataclass(frozen=True)
class Observation:
    h: float                  # > 0, finite: grid length (cell/element/panel size or dt)
    f: float                  # finite: the observable at this level
    label: str = ""           # free text, echoed in reports
    sigma: float | None = None  # >= 0: 1-sigma uncertainty of f (round-off/noise)
```

* `Observation(h=0.0, f=1.0)` -> `ValueError`; `f=nan` -> `ValueError`;
  `sigma=-1` -> `ValueError`.
* `as_dict() -> dict` - serialises `h`, `f` and the optional `label`/`sigma`.
* `Observation.from_mapping(payload) -> Observation` - tolerant reader used by
  the CLI for `tensorphysics.vv.observations/1` files. It accepts the aliases
  `h | mesh_size | dx | dt | spacing | cell_size` and
  `f | value | metric | observable | error | l2_error`; a payload without one of
  each raises `ValueError`.

### `GatePolicy`

All thresholds are data; nothing is hidden in the code. Defaults are the
documented ASME V&V 20 / Roache / ITTC set.

| field | default | meaning |
|---|---|---|
| `safety_factor_three` | `1.25` | `Fs` for a study with >= 3 levels |
| `safety_factor_two` | `3.0` | `Fs` for a two-level study |
| `p_min`, `p_max` | `0.05`, `6.0` | admissible band for the observed order |
| `assumed_order` | `None` | order to use when it cannot be observed (2 grids) |
| `expected_order` | `None` | theoretical order; when set, `\|p - p_theory\|` is gated |
| `order_tolerance` | `0.2` | WARN above this deviation, FAIL above `2.5x` |
| `residual_warn`, `residual_fail` | `2e-3`, `2e-2` | power-law fit residual (>= 4 levels) |
| `order_consistency_warn`, `order_consistency_fail` | `0.2`, `0.5` | triplet-to-triplet order spread |
| `noise_warn_ratio`, `noise_fail_ratio` | `0.05`, `0.25` | declared `sigma` / finest difference |
| `require_monotone` | `True` | non-monotone refinement is not acceptable |
| `strict` | `False` | additionally emit the certificate-strength WARN |
| `pass_requires_noise_certificate` | `True` | **conservative default**: an uncertified noise floor caps the verdict at WARN |
| `gci_absolute` | `False` | always report the absolute GCI band (use for error norms) |
| `gci_increase_tolerance` | `0.05` | slack on the GCI monotonicity test |

`policy.merged(**overrides)` returns a copy with the non-`None` overrides
applied; `policy.as_dict()` serialises every field.

### `GridConvergenceStudy`

Frozen result of the numerical study. Fields: `levels` (ordered coarse -> fine),
`order_p`, `order_source` (`closed-form` / `least-squares` / `assumed` /
`undefined`), `f_extrapolated`, `gci_fine`, `safety_factor`, `gci_is_relative`,
`convergence_ratio`, `convergence_type` (`monotone` / `oscillatory` /
`diverging` / `stagnant` / `not-applicable` / `undefined`), `refinement_ratios`,
`uniform_refinement`, `fit_residual_rel`, `order_triplets`, `order_spread`,
`noise_ratio`, `noise_certified`, `gci_pairs`, `gci_monotone`,
`extrapolation_gain`, `notes`.

Properties: `n_levels`, `f_fine`, `asymptotic` (`monotone` **and** an order
exists). `as_dict()` is the JSON form used by the CLI.

### `GateVerdict`

Frozen result of the gate: `status` (`PASS`/`WARN`/`FAIL`), `quantity`, `study`,
`checks` (tuple of `{"status", "check", "detail", **metrics}`),
`validation` (`ValidationResult | None`), `policy`. Properties: `reasons` (the
details of every WARN/FAIL check), `as_dict()`.

---

## 2. Formulas (`tensorphysics.vv.gci`)

### `observed_order(levels, *, p_min=0.05, p_max=6.0) -> tuple[float|None, str, float|None]`

Returns `(p, source, residual)`.

* uniform three-level study, closed form:
  `p = ln((f1 - f2) / (f2 - f3)) / ln(r)` where `1, 2, 3` are coarse, medium, fine;
* any other study (non-uniform ratios, >= 4 levels): one-dimensional
  least-squares fit of `f(h) = f_exact + C h^p`, bracketed in `(p_min, p_max]`,
  golden-section refinement; `residual` is the relative RMS misfit;
* fewer than three levels: `(None, "undefined", None)`.

### `richardson_extrapolated(levels, p) -> float`

`f_exact = f_fine + (f_fine - f_medium) / (r**p - 1)`, `r = h_medium / h_fine`.
Raises `ValueError` for fewer than two distinct levels or `r**p == 1`.

### `gci(coarse, fine, p, *, safety_factor=1.25, force_absolute=False) -> tuple[float, bool]`

`GCI = Fs * |f_fine - f_coarse| / scale / (r**p - 1)`, `r = h_coarse / h_fine`.
`scale = |f_fine|` (relative band, `is_relative=True`) unless `|f_fine| <=
1e-300` or `force_absolute=True` (absolute band of the observable's own units).
Raises `ValueError` when `r**p <= 1`.

### `refinement_ratio(coarse, fine) -> float`

`h_coarse / h_fine`; raises `ValueError` unless `> 0`.

### `sweep_refinement_ratio(model, *, h_coarse, p, ratios=(2.0, 3.0, 4.0), safety_factor=3.0) -> tuple[tuple[float, float], ...]`

Two-grid GCI of a fixed coarse grid while the fine grid is refined; the machine
form of "GCI decreases with refinement" (a fixed fine grid has an invariant
relative GCI, so the sweep must be anchored on the coarse grid).

### `grid_convergence_study(levels, *, policy=None, assumed_order=None) -> GridConvergenceStudy`

Assembles the whole study: ordering, refinement ratios, order, Richardson value,
GCI per pair and for the finest pair, convergence ratio/type, power-law
residual, triplet order spread, noise ratio, noise certificate, monotonicity of
the GCI, notes.

### `evaluate(levels, *, quantity="observable", policy=None, f_exp=None, f_sim=None, u_num=None, u_input=None) -> GateVerdict`

Runs the study and the checks below, then fuses them with "worst wins"
(`FAIL > WARN > PASS`). When `f_exp` is given the ASME V&V 20 comparison is
appended as one more check.

| check | FAIL when | WARN when |
|---|---|---|
| `input` | < 2 levels, non-positive `h`, duplicated `h` | - |
| `asymptotic-range/convergence-type` | oscillatory / diverging / stagnant | - |
| `order-band` | `p` outside `(p_min, p_max]` or undefined | - |
| `order-band/expected` | `\|p - p_theory\| > 2.5 * order_tolerance` | `> order_tolerance` |
| `power-law-residual` | `> residual_fail` | `> residual_warn` |
| `order-consistency` | spread `> order_consistency_fail` | `> order_consistency_warn` |
| `noise-floor` | declared `sigma/diff > noise_fail_ratio` | `> noise_warn_ratio` |
| `gci-monotonicity` | GCI grows with refinement | - |
| `noise-certificate` | - | 3 levels, no `sigma` and `pass_requires_noise_certificate` |
| `certificate-strength` | - | `strict=True` and no noise certificate |
| `validation/V&V20` | `\|E\| > U_val` | - |

---

## 3. ASME V&V 20 (`tensorphysics.vv.validation`)

### `validation_uncertainty(u_num, u_input=0.0) -> float`

`U_val = sqrt(U_num**2 + U_input**2)`. Raises `ValueError` for negative or
non-finite inputs.

### `validate(f_sim, f_exp, *, u_num, u_input=0.0, label="") -> ValidationResult`

`ValidationResult` fields: `f_sim`, `f_exp`, `u_num`, `u_input`, `u_val`,
`error` (`f_sim - f_exp`), `relative_error` (`error / f_exp`, `None` for
`f_exp == 0`), `ratio` (`|error| / u_val`), `passed`, `label`; property
`status`. `as_dict()` serialises all of them.

The acceptance test is `|E| <= U_val + BOUNDARY_SLACK * max(U_val, |E|)` with
`BOUNDARY_SLACK = 1e-12`: the boundary `|E| = U_val` is a legitimate PASS under
ASME V&V 20, and floating point rounding of both sides must not flip it. The
reported `ratio` is never slacked, so a marginal case is still visible.
`u_val == 0` degenerates to "only an exact match passes".

### `evaluate_validation_table(rows, *, keys=("f_sim","f_exp","u_num","u_input")) -> list[ValidationResult]`

`rows` are mappings (field names, optional `label`) or sequences in `keys` order.
Missing fields raise `ValueError` with the row index.

---

## 4. Adapters (`tensorphysics.vv.adapters`)

### `AdapterSeries`

`repo`, `quantity`, `levels` (ordered coarse -> fine), `source_path`, `study`,
`metadata`, `raw_levels` (the untouched native level mappings). Methods:
`as_dict()`, `to_gate_payload(expected_order=None, validation=None)`.

### `extract_series(payload, *, repo="auto", quantity=None, study=None, value_key=None, source_path="") -> AdapterSeries`

Normalises one benchmark document.

* `repo="auto"` detects the dialect from the document's `schema`/`repo`/`solver`
  field, falling back to the generic profile; an explicit `repo=` forces a
  dialect and an unknown one raises `AdapterError`.
* grid length comes from a length key (`h`, `mesh_size`, `dx`, `dt`,
  `cell_size`, `panel_size`, ...) or from a count key (`resolution`, `nx`,
  `n_panels`, ...; then `h = 1/count`);
* the observable comes from `value_key` (must exist on every level) or the first
  known value key of the dialect;
* `sigma`/`std`/`uncertainty`/`stderr` are carried through;
* `study=` selects one sub-study of an evidence document; without it a
  manufactured/MMS study is preferred, then alphabetical order;
* no guessing: a level without a grid measure, or a document without any level
  list, raises `AdapterError`.

`REPO_PROFILES` holds the registered dialects (`tensorphysics`, `generic`,
`tensorlbm`, `tensorfvm`, `tensorfem`, `tensordem`, `tensorbem`); each entry
lists its length keys, count keys, value keys and level containers.

### `detect_repo(payload) -> str`, `load_payload(path) -> Any`

Dialect detection and JSON reading (`AdapterError` on invalid JSON).

### `to_gate_payload(series, *, expected_order=None, validation=None) -> dict` / `write_gate_payload(series, path, **kwargs) -> dict`

Serialise an adapted series as a `tensorphysics.vv.observations/1` document
(the CLI input format), optionally with a validation block; the `write_` variant
also saves it.

---

## 5. Command line

```
python -m tensorphysics.vv          # same as python -m tensorphysics.vv.gci
python -m tensorphysics.vv.gci --obs obs.json --format json
```

Sources (mutually exclusive, one required):

| flag | meaning |
|---|---|
| `--obs PATH` | gate observation document (`tensorphysics.vv.observations/1`), `-` for stdin |
| `--benchmark PATH` | native benchmark document of a solver repository (adapter path) |
| `--table PATH` / `--vv20-table PATH` | ASME V&V 20 table `{"rows": [...]}` |

Output formats (`--format`, default `text`; `--json` is an alias of `--format json`):

| format | output |
|---|---|
| `text` | human report: study numbers, notes, per-check lines, verdict |
| `json` | the full `GateVerdict` record (identical to `verdict.as_dict()`) |
| `table` | fixed-width ASCII tables: study metrics, gate checks, V&V 20 block |
| `plot` | PNG figure of the convergence study (`--plot-out`, default `vv_gate.png`) plus the text report |

For `--table` input, `--format plot` draws `\|E\|/U_val` per row against the
acceptance line (default `vv20_table.png`).

Study / policy flags: `--repo`, `--study`, `--quantity`, `--value-key`,
`--expected-order`, `--order-tolerance`, `--assumed-order`, `--p-min`, `--p-max`,
`--strict`, `--allow-undeclared-noise`, `--gci-absolute`, `--exp`, `--u-input`,
`--u-num`, `--f-sim`, `--out`.

Exit codes: `0` PASS, `1` WARN, `2` FAIL, `64` usage or input error (including a
missing matplotlib for `--format plot`).

---

## 6. JSON contracts

### `tensorphysics.vv.observations/1` (CLI input)

```json
{
  "schema": "tensorphysics.vv.observations/1",
  "quantity": "relative_strain_energy_error",
  "expected_order": 2.0,
  "levels": [{"h": 0.5, "f": 0.2539, "label": "n=2", "sigma": 1e-9}],
  "validation": {"f_exp": 0.0, "u_input": 0.0, "u_num": 0.021, "f_sim": 1.0},
  "source": {"repo": "tensorfem", "path": "...", "study": "manufactured"}
}
```

Only `levels` is mandatory; each level needs `h`+`f` (or an alias).

### Verdict (CLI output, `--format json`)

```json
{
  "status": "WARN",
  "quantity": "Cd",
  "study": {"n_levels": 3, "order_p": 2.0, "f_extrapolated": 3.0,
            "gci_fine": 0.0129, "gci_is_relative": true, "noise_certified": false,
            "levels": [...], "notes": [...]},
  "checks": [{"status": "WARN", "check": "noise-certificate", "detail": "..."}],
  "validation": null,
  "policy": {"pass_requires_noise_certificate": true, "...": "..."},
  "adapter": {"repo": "tensorlbm", "study": "", "levels": [...]}
}
```

### `tensorphysics.vv.vv20-table/1` (CLI output for `--table` input)

```json
{
  "schema": "tensorphysics.vv.vv20-table/1",
  "source": "table.json",
  "rows": [{"f_sim": 1.0, "f_exp": 1.02, "u_val": 0.0224, "ratio": 0.894,
            "status": "PASS", "...": "..."}],
  "n_pass": 1,
  "n_fail": 1
}
```

---

## 7. Worked example

```python
from tensorphysics.vv import GatePolicy, Observation, evaluate

levels = [Observation(h=h, f=0.5 * h ** 2) for h in (1.0, 0.5, 0.25)]
verdict = evaluate(levels, quantity="l2_error",
                   policy=GatePolicy(expected_order=2.0, gci_absolute=True))
print(verdict.status, verdict.study.order_p, verdict.study.gci_fine)
# WARN 2.0 0.006666666666666667   (WARN: three levels, no declared sigma)
```

The WARN is the honest answer for a three-grid study whose noise floor is not
certified; declare per-level `sigma`, supply a fourth grid, or pass
`pass_requires_noise_certificate=False` when the solver is deterministic and the
noise floor is known to be irrelevant.

See `docs/VV_GCI_ASME_GATE.md` for the acceptance evidence,
`docs/VV_GATE_MIGRATION_GUIDE.md` for the migration recipe, and
`artifacts/vv_gate_n07/acceptance.json` for the machine-checkable record.