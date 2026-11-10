# Machine-generated V&V reports (`tensorphysics.vv.report`)

Status: landed with mission **m8633** (re-open of TensorPhysics #161 follow-up work).
Module: `src/tensorphysics/vv/report.py` · Tests: `tests/test_vv_report_m8633.py` ·
Example: `examples/vv_report_from_verdict.py` → `artifacts/vv_report_m8633/`.

## 1. The problem this closes

`docs/vv_report_template.md` fixes the six-element structure of a V&V report
(case introduction → method → results → figures → pitfalls → error judgement),
but a template is a *form*. Filling it by hand has three failure modes that the
platform cannot detect afterwards:

1. **transcription drift** — a number is copied from the solver output into the
   report by hand and differs from what the gate actually computed;
2. **silent blanks** — a section with no data (no reference value, no parameter
   table) looks exactly like a section with a nice number, so the report reads
   as complete when it is not;
3. **unprovable provenance** — nothing ties the rendered report to the artefacts
   it came from.

`build_report` removes all three by rendering the template **from the gate
verdict**:

```python
from tensorphysics.vv.gci import Observation, evaluate
from tensorphysics.vv.report import build_report

levels = [Observation(h=h, f=2.0 + 3.0 * h**1.8) for h in (0.4, 0.2, 0.1)]
verdict = evaluate(levels, quantity="Cd", f_exp=2.01, u_input=0.02)
report = build_report(
    verdict,
    title="Cylinder drag: grid convergence and validation",
    metadata={
        "case_name": "manufactured sequence f(h) = 2 + 3 h^1.8",
        "reference_source": "manufactured solution",
        "parameters": [{"name": "f_inf", "value": 2.0, "unit": "-", "uncertainty": 0.0}],
    },
)
print(report.status)          # worst(verdict.status, withheld required sections)
print(report.markdown)        # the six-element report, numbers already injected
print(report.coverage)        # which required sections are filled / withheld
print(report.provenance)      # sha256 of the verdict payload and of the body
```

## 2. Contract

| Property | How it is enforced |
|---|---|
| single source of truth | the only input is a `GateVerdict`; the renderer computes **no** numerics. `section.data` exposes the exact numbers each section was rendered from, so a test (or a reader) compares `section.data[k] == verdict.<field>` without parsing prose. |
| no silent blanks | a section the machine cannot fill becomes `status="WITHHELD"` with a `reason`; the report `status` is `worst(verdict.status, WITHHELD for every withheld **required** section)` under the gate ordering `PASS < WARN < WITHHELD < UNRESOLVED < FAIL`. |
| no hand-filling | the template placeholders (`[value]`, `[Describe …]`, `[Yes/No]`, `[PASS/FAIL/UNRESOLVED]`, …) are scanned in the title, the metadata and the rendered markdown; finding one raises `ValueError` instead of producing a "generated" report with a typed-in number. |
| provenance | `provenance["source_digest"]` = sha256 of the canonicalised `verdict.as_dict()`; `provenance["body_digest"]` = sha256 of the rendered body; both are printed in the report's provenance block. |
| deterministic | no timestamps, sorted-key JSON, `%.6g` number formatting: the same verdict renders byte-identically. |
| explicit error judgement | §6 prints the absolute relative error against the reference and answers the acceptance question in words (`error criterion MET / NOT MET`) against the 3 % bound (`error_acceptance`, overridable). |

Required sections: `case`, `grid-convergence`, `asme-vv20`, `error-judgement`,
`conclusion`. Optional sections (never counted as withheld): `cross-solver`,
`resolvability` — when they are not requested the report says so **and states
the resulting limitation** (e.g. without a declared feature scale the gate
cannot certify that the reported physics is resolved).

## 3. Section → gate field map

| Report section | Injected from |
|---|---|
| 1. Case introduction | `metadata["parameters"]`, `metadata["reference_source"]`, `verdict.quantity` |
| 2.1 Grid convergence | `verdict.study`: `levels` (`h`, `f`, `sigma`), `refinement_ratios`, `order_p`/`order_source`, `f_extrapolated`, `f_fine`, `gci_fine`, `gci_pairs`, `convergence_ratio`, `convergence_type`, `asymptotic`, `fit_residual_rel`, `noise_ratio`, plus the `asymptotic-range/*`, `order-*`, `power-law-residual`, `noise-floor` checks |
| 2.2 ASME V&V 20 | `verdict.validation` (`u_num`, `u_input`, `u_val`, `error`, `ratio`, `passed`) and `verdict.uncertainty` ledger (`u_ref`, `u_val_total`, `reference_ratio`, `dominant_term`, `discriminative`) |
| 2.3 Cross-solver | `metadata["cross_solver"]` (consensus ledger, per-solver samples) |
| 5. Resolvability | `verdict.resolvability` (`R = delta/dx`) |
| 6. Error judgement | `verdict.validation.f_sim` / `f_exp` vs `error_acceptance` |
| 7. Conclusion | `verdict.checks`, `verdict.reasons`, withheld-section ledger |

## 4. Limits (stated, not hidden)

* The parameter table, the reference citation and the cross-solver ledger are
  **context**, not numerics: the machine cannot know them, so they are passed in
  through `metadata` and are never invented. A report without them is
  `WITHHELD`, which is the honest outcome.
* The report renders **text**, not figures. Figures referenced by §4 of the
  template must be produced by the caller and attached to the report payload
  (see `examples/vv_report_from_verdict.py`, which emits the log-log
  error-vs-`h` figure next to the markdown).
* The renderer never re-runs the gate: if the verdict is stale, the report is
  stale, and the only defence is the provenance digest — re-render from the
  current verdict rather than editing the markdown.