# Unified plate-buckling + structural-reliability framework

`tensorphysics.plate_buckling_reliability` couples the deterministic
arc-length buckling solver (`tensorphysics.plate_buckling`, node W3-N03) with
the Nataf/FORM/SORM reliability core (`tensorphysics.structural`, node W3-N04)
into one assessment chain: one panel in, `sigma_ultimate` + `beta` + `P_f`
(+ the thickness that meets a target index) out.

```python
from tensorphysics.plate_buckling_reliability import (
    PanelSpec, UnifiedBucklingReliability)

model = UnifiedBucklingReliability(PanelSpec(a=3.0, b=1.0, t_mm=19.49, nx=8, ny=4))
result = model.assess(samples=1_000_000, target_beta=4.0)
result["sigma_ultimate_MPa"], result["beta"], result["pf"]
result["target_thickness"]["t_required_mm"]
```

## What is inside

| module | role |
|---|---|
| `panel.PanelSpec` | geometry/material/mesh/load specification; pre-analysis gate (a/b, beta, b/t, mesh aspect) |
| `deterministic.DeterministicAssessment` | linear buckling (4 BCs), Riks/Crisfield arc-length limit point (cached), imperfection sweep, IACS UR I style ULS check |
| `surrogate.UltimateStrengthSurrogate` | `sigma_u/sigma_y = c0 + c1/beta + c2/beta^2` fitted to the FE limit points, residual reported |
| `limit_states.PanelLimitStates` | `g_ult` (FE capacity vs ice/global membrane demand), `g_yield`, diagnostic `g_elastic` / `g_deflection` |
| `reliability.ProbabilisticAssessment` | FORM (HL-RF), Breitung SORM, Ditlevsen bounds, crude + importance-sampling Monte Carlo, design-point sensitivity vs finite differences, MVFOSM reference, target-beta thickness inversion |
| `reliability.GenericLimitStates` | adapter so any other limit-state model (e.g. the W3-N04 ice belt) runs through the same machinery |
| `unified.UnifiedBucklingReliability` | the single public interface |
| `api.handle_calculation` | framework-neutral handler of `POST /calculations/plate-buckling-reliability` (+ stdlib server, optional FastAPI router) |

## Evidence

* verification record: `artifacts/w4n03_plate_buckling_reliability/verification.json`
* FE calibration (6 arc-length traces + probes): `artifacts/w4n03_plate_buckling_reliability/calibration.json`
  and the packaged copy `src/tensorphysics/plate_buckling_reliability/data/fem_calibration.json`
* report with the eight machine-checked criteria and figures: `docs/reports/w4n03/plate_buckling_reliability_report.md`
* reproduction:

```bash
python scripts/run_w4n03_plate_buckling_reliability.py calibrate   # 6 FE traces -> response surface
python scripts/run_w4n03_plate_buckling_reliability.py verify      # gates a-h -> verification.json
python scripts/run_w4n03_plate_buckling_reliability.py figures     # 8 figures
python scripts/run_w4n03_plate_buckling_reliability.py render      # report.md from the artifacts
python -m pytest tests/test_plate_buckling_reliability.py tests/test_plate_buckling_reliability_api.py -q
```

## HTTP endpoint

```bash
python -m tensorphysics.plate_buckling_reliability.api --port 8011
curl -s -X POST http://127.0.0.1:8011/calculations/plate-buckling-reliability \
  -H 'Content-Type: application/json' \
  -d '{"panel": {"t_mm": 13.6, "sigma_y_MPa": 313}, "options": {"target_beta": 4.0}}'
```

Response fields: `sigma_ultimate_MPa`, `beta`, `pf`, `beta_sorm`, `pf_sorm`,
`beta_system`, `pf_system`, `design_point`, `target_thickness`, plus the full
`deterministic` and `probabilistic` blocks and the surrogate provenance
(coefficients, fit residual, calibration window).

## Known limits

* the capacity inside the limit state is an FE-calibrated response surface, not
  a per-sample FE solve; its residual and its out-of-window error are reported;
* the FE calibration window is `beta` in [1, 3] (mesh 8x4, imperfection
  "average"); the design case needs `beta` up to 3.6 at the net thickness and
  the surrogate is then conservative (measured at -9.6 % against one FE trace
  at that thickness);
* Paik et al. (2008) tabulated ultimate strengths are not available offline, so
  the von Karman/Faulkner curve is used as the primary reference;
* `g_elastic` and `g_deflection` are implemented for diagnosis but are not part
  of the default series system (plate buckling is an onset event with a large
  post-buckling reserve, and the uniform-pressure deflection mode does not match
  the patch-like ice load).