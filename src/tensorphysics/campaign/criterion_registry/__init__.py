"""Criterion pre-registration and drift audit mechanism.

Provides three properties for campaign node acceptance criteria:
(a) pre-freeze (before execution), (b) post-audit (after execution),
(c) drift detection (threshold changes between freeze and audit).
The freeze is anchored durably on disk by ``store.py`` so that freeze and
audit may happen in different processes.

Fail-closed: unparseable criteria → 'unregistered', never 'pass'.

Zero domain semantics: manifest contains only structural fields
(threshold/unit/denominator/exclusion/reference_path/comparator).
Domain content injected via API/config, not hardcoded.
"""
from .manifest import (
    CriterionManifest,
    CriterionField,
    extract_criteria_from_goal,
    freeze_manifest,
    verify_frozen_hash,
)
from .drift import (
    DriftClassifier,
    DriftResult,
    audit_drift,
    classify_drift,
)
from .ledger import (
    CriterionEvent,
    CriterionLedger,
    record_drift_event,
    query_kpi,
)
from .kpi import (
    KPIView,
    compute_kpi,
)
from .registration import (
    Registration,
    DeliveryAudit,
    register_goal,
    audit_delivery,
)
from .store import (
    FrozenBundle,
    PreregistrationConflict,
    StoreCorruptError,
    STORE_SCHEMA,
    audit_from_store,
    bundle_path,
    freeze_to_store,
    load_bundle,
    load_registrations,
)

__all__ = [
    "CriterionManifest",
    "CriterionField",
    "extract_criteria_from_goal",
    "freeze_manifest",
    "verify_frozen_hash",
    "DriftClassifier",
    "DriftResult",
    "audit_drift",
    "classify_drift",
    "CriterionEvent",
    "CriterionLedger",
    "record_drift_event",
    "query_kpi",
    "KPIView",
    "compute_kpi",
    "Registration",
    "DeliveryAudit",
    "register_goal",
    "audit_delivery",
    "FrozenBundle",
    "PreregistrationConflict",
    "StoreCorruptError",
    "STORE_SCHEMA",
    "audit_from_store",
    "bundle_path",
    "freeze_to_store",
    "load_bundle",
    "load_registrations",
]
