"""Criterion pre-registration: freeze acceptance criteria, audit drift, expose KPI.

Campaign nodes are only as trustworthy as their *pre-declared* acceptance
criteria.  This package gives every node three properties:

(a) **pre-freeze** -- :func:`freeze` extracts a structured criterion block from
    a mission goal, canonicalises it and derives ``manifest_sha256``;
    :func:`attach_to_metadata` publishes it into mission metadata (and refuses
    an unregistered manifest, fail-closed).
(b) **post-hoc auditability** -- the canonical bytes travel with the digest, so
    :func:`verify_manifest` can refute a producer's digest
    (``registered`` / ``unregistered`` / ``tampered``).
(c) **drift decidability** -- :func:`audit` pairs frozen and receipt criteria and
    classifies each movement as ``unchanged`` / ``tightened`` / ``loosened`` /
    ``scope_changed`` / ``undefined``; only ``unchanged`` + ``tightened`` pass.
    :meth:`DriftReport.events` emits event-stream records and :func:`kpi_view`
    aggregates the audits into a scorecard.

Modules
-------
``criterion_manifest``  extraction, canonical JSON, sha256 freeze/verify
``drift_audit``         field-level diff, drift classes, verdict, events,
                        KPI scorecard aggregation
``scan``                zero-domain-semantics scan (vocabulary externalised)
``synth``               synthetic corpora with generator-side ground truth

Zero domain semantics: the package contains no engineering-domain vocabulary;
metric names and scopes are data (checked by :mod:`...criterion_prereg.scan`).
"""
from .criterion_manifest import (
    EXTRACTOR_VERSION,
    REGISTERED,
    SCHEMA,
    TAMPERED,
    UNREGISTERED,
    Criterion,
    Extraction,
    Freeze,
    ManifestError,
    attach_to_metadata,
    canonical_json,
    criteria_block,
    criteria_of,
    extract_criteria,
    freeze,
    iter_field_triples,
    manifest_from_metadata,
    normalise_text,
    semantic_digest,
    sha256_obj,
    sha256_text,
    verify_manifest,
)
from .drift_audit import (
    EVENT_SCHEMA,
    CriterionDrift,
    DriftClass,
    DriftReport,
    Verdict,
    audit,
    audit_goals,
    compare_criterion,
    kpi_view,
)
from .scan import ScanResult, load_vocabulary, scan_paths, scan_text

__all__ = [
    "SCHEMA",
    "EVENT_SCHEMA",
    "EXTRACTOR_VERSION",
    "REGISTERED",
    "UNREGISTERED",
    "TAMPERED",
    "Criterion",
    "Extraction",
    "Freeze",
    "ManifestError",
    "DriftClass",
    "Verdict",
    "CriterionDrift",
    "DriftReport",
    "ScanResult",
    "canonical_json",
    "sha256_text",
    "sha256_obj",
    "criteria_block",
    "extract_criteria",
    "freeze",
    "verify_manifest",
    "attach_to_metadata",
    "manifest_from_metadata",
    "criteria_of",
    "iter_field_triples",
    "normalise_text",
    "semantic_digest",
    "compare_criterion",
    "audit",
    "audit_goals",
    "kpi_view",
    "load_vocabulary",
    "scan_paths",
    "scan_text",
]