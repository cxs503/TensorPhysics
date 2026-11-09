# Criterion Pre-Registration Mechanism

## Overview

This module implements a criterion pre-registration and drift audit mechanism for campaign nodes, providing three properties for acceptance criteria:

1. **Pre-freeze (a)**: Criteria are extracted from mission goals and frozen with sha256 hash before execution
2. **Post-audit (b)**: After execution, report criteria are re-extracted and compared against frozen versions
3. **Drift detection (c)**: Changes between frozen and current criteria are classified and recorded

**Fail-closed**: Unparseable criteria → `unregistered`, never `pass`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mission Goal Text                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              criterion_manifest.py                               │
│  extract_criteria_from_goal() → CriterionManifest[]             │
│  freeze_manifest() → sha256 + status='registered'               │
│  verify_frozen_hash() → bool                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ frozen manifests
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Campaign Execution                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ final report text
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    drift.py                                      │
│  extract_criteria_from_goal(report) → current criteria          │
│  classify_drift(frozen, current) → DriftResult                  │
│  Classifications: unchanged|tightened|loosened|scope_changed    │
│                 |undefined|unregistered|tampered                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ drift events
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ledger.py                                     │
│  CriterionLedger (append-only, JSON-lines persistence)          │
│  record_drift_event() → CriterionEvent                          │
│  Fail-closed: loosened/scope_changed require reason field       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    kpi.py                                        │
│  compute_kpi() → KPIView                                        │
│  Metrics: registered_rate, unregistered_rate, loosened_rate     │
└─────────────────────────────────────────────────────────────────┘
```

## Canonical JSON Schema

Each criterion is represented as canonical JSON with six structural fields:

```json
{
  "comparator": "le",
  "denominator": "ref_value",
  "exclusion": "edge_case",
  "reference_path": "/data/baseline",
  "threshold": 5.0,
  "unit": "pct"
}
```

**Zero domain semantics**: The manifest contains only structural fields. Domain content (physical quantity names, baseline sources) is injected via API/config, not hardcoded.

## Drift Classification

| Classification | Meaning | Requires Reason |
|----------------|---------|-----------------|
| `unchanged` | No difference between frozen and current | No |
| `tightened` | Threshold became stricter | No |
| `loosened` | Threshold became looser | **Yes** |
| `scope_changed` | Denominator/exclusion/reference/unit changed | **Yes** |
| `undefined` | Cannot determine drift direction | No |
| `unregistered` | Criterion could not be parsed | No |
| `tampered` | Frozen hash does not match content | No |

## Usage

```python
from tensorphysics.campaign.criterion_registry import (
    extract_criteria_from_goal,
    freeze_manifest,
    classify_drift,
    CriterionLedger,
    record_drift_event,
    compute_kpi,
)

# 1. Extract and freeze criteria before execution
goal_text = "C1: error <= 5 pct of reference (ref: /data/baseline)"
criteria = extract_criteria_from_goal(goal_text)
frozen = [freeze_manifest(c, "agent_id") for c in criteria]

# 2. After execution, extract criteria from report
report_text = "..."  # final report
current = extract_criteria_from_goal(report_text)

# 3. Audit drift
ledger = CriterionLedger(storage_path="ledger.jsonl")
for f in frozen:
    c = next((x for x in current if x.criterion_id == f.criterion_id), None)
    if c:
        result = classify_drift(f, c)
        record_drift_event(
            ledger,
            mission_id="mission_001",
            criterion_id=f.criterion_id,
            classification=result.classification,
            reason=result.reason,
        )

# 4. Compute KPIs
kpi = compute_kpi(ledger)
print(f"registered_rate: {kpi.registered_rate:.3f}")
print(f"unregistered_rate: {kpi.unregistered_rate:.3f}")
print(f"loosened_rate: {kpi.loosened_rate:.3f}")
```

## Verification

See `tests/test_criterion_preregistration.py` for comprehensive tests covering:

- **C1**: 24 synthetic test cases (12 modified, 12 unchanged): recall ≥ 0.92, precision ≥ 0.95
- **C2**: Field-level F1 ≥ 0.95 on 12 synthetic criteria
- **C3**: Fail-closed negative controls (empty goal, tampered hash, etc.)
- **C4**: Real replay of 100 synthetic goals
- **C5**: Zero domain semantics scan

## Files

- `manifest.py`: Criterion extraction and freezing
- `drift.py`: Drift classification
- `ledger.py`: Event ledger with persistence
- `kpi.py`: KPI computation
- `__init__.py`: Public API exports
