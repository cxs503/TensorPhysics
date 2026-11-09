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

## Durable freeze store (cross-process pre-registration)

The in-memory cycle above only proves pre-registration *within one process*.
A campaign node's freeze happens at mission creation (dispatcher) while the
audit happens when the delivery report lands (pr-gate, possibly another host and
day), so the frozen manifests need a durable anchor. `store.py` adds one:

```python
from tensorphysics.campaign.criterion_registry import (
    freeze_to_store, load_bundle, load_registrations, audit_from_store,
    CriterionLedger,
)

# process 1 — at mission creation, before any work starts
bundle = freeze_to_store("runs/freeze_store", "m9001", goal_text,
                         frozen_by="agent/m8695",
                         frozen_at="2026-11-09T00:00:00+00:00")
print(bundle.bundle_sha256, bundle.registered_ids)

# process 2 — at delivery, days later, no shared memory
audit = audit_from_store("runs/freeze_store", "m9001", report_text,
                         ledger=CriterionLedger(storage_path=Path("runs/criterion_ledger.jsonl")))
print(audit.failed_closed, audit.kpi.loosened_rate)
```

On-disk layout — one canonical JSON file per mission
(`<store>/<mission_id>.json`, atomic `tmp` + `os.replace`, mission ids are
validated as path-safe `[A-Za-z0-9][A-Za-z0-9._-]*`):

```json
{"schema": "criterion_freeze_store/v1", "mission_id": "m9001",
 "frozen_by": "agent/m8695", "frozen_at": "2026-11-09T00:00:00+00:00",
 "criteria": [{"criterion_id": "criterion_001", "fields": {...},
               "status": "registered", "sha256": "..."}],
 "bundle_sha256": "..."}
```

Two digest layers, both fail-closed:

| layer | scope | detection |
|---|---|---|
| per-criterion `sha256` | six canonical fields | threshold edited after freeze → criterion loads as status `tampered` (never `registered`), audit classification `tampered`, `failed_closed=True` |
| `bundle_sha256` | whole payload incl. metadata | file hand-edited / truncated / unknown `schema` → `StoreCorruptError` |

Guarantees:

* **Fail-closed**: a goal without a numeric level writes **nothing** (no file
  exists, so nothing can later be audited as "registered"), and a mission with
  no bundle audits as `unregistered`, never `unchanged`.
* **Idempotent**: re-freezing the identical criterion set (compared by
  `(criterion_id, field sha256)` pairs, so timestamps are irrelevant) returns
  the stored bundle and does not rewrite the file.
* **No silent re-registration**: a *different* criterion set for an already
  frozen mission raises `PreregistrationConflict`; the original level stays
  frozen (a post-hoc loosening cannot be laundered into a fresh registration).
* **Tamper visibility**: `audit_delivery` now propagates a non-registered
  status such as `tampered` into the ledger event instead of flattening it to
  `unregistered`, so tampering is countable in the KPI view.

`store.py` carries only mission ids, actor ids, timestamps and the six
structural fields — zero domain semantics (guarded by
`tests/test_criterion_freeze_store.py::test_store_has_zero_domain_semantics`).

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
- `registration.py`: fail-closed `register_goal` / `audit_delivery` cycle
- `store.py`: durable cross-process freeze store (two digest layers)
- `bench.py`: synthetic-corpus drift benchmark
- `__init__.py`: Public API exports

## Drift-audit artifacts (reproducibility contract)

The numbers quoted by the maturity tests are not typed by hand: they live in two
committed JSON artifacts, and the tests refuse to run without them.

| artifact | sha256 |
|---|---|
| `artifacts/criterion_drift_audit/bench_v2_hardened.json` | `ce61f66f939d0dd5f54dad53ca18a81b76180c2b2d9334c0a70f687975aecfbc` |
| `artifacts/criterion_drift_audit/bench_v1_legacy.json` | `98703c5869e3325e1a3ec0224ac2430c8e2b88a168632b51dbc60849fe15305e` |

Both are pinned to the same corpus fingerprint
`corpus_sha256 = 01d420d1ad918cc3cbc5cdccdc51a4aa9942639b73be680e48fd3486ea180d26`.

| record | extraction P | extraction R | extraction F1 | held-out F1 | drift P | drift R | all gates |
|---|---|---|---|---|---|---|---|
| v1_legacy | 0.8462 | 0.7674 | 0.8049 | 0.7523 | 1.0000 | 1.0000 | no |
| v2_hardened | 1.0000 | 1.0000 | 1.0000 | 0.9735 | 1.0000 | 1.0000 | yes |

The `v1_legacy` row is a measurement of the **pre-hardening** extractor, not a
straw man written to look bad: `benchmarks/criterion_drift_audit/legacy_manifest_v1.py`
is a byte-identical copy of the extractor as introduced by commit `15326d3`
(m7397), superseded by `41d0e52` (comparator-anchored extraction). Its whole-file
sha256 is pinned in `generate.py` (`82fe953188c730f0…`) and checked before every
use — an edited baseline raises `LegacySnapshotTampered` instead of quietly
producing a friendlier delta.

### Regenerating

```bash
python benchmarks/criterion_drift_audit/generate.py
```

Generation is a pure function of the corpus (no clock, no randomness, no
network): re-running on an unchanged tree reproduces both files byte for byte.
The script fails closed if the hardened record misses a gate, if the baseline is
no longer worse, or if the two records disagree on the corpus fingerprint.

Guarded by `tests/test_criterion_drift_artifacts.py` (baseline provenance,
bit-for-bit reproduction, corpus pinning, direction of the delta).
