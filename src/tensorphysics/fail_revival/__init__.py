"""Honest-FAIL revival registry and blocker-graph scheduler.

An honest FAIL is a *debt*, not a dead end: it names a criterion that was not
met and an obstruction that keeps it from being re-measured.  This package turns
that prose into a machine-checkable loop:

``extract``
    campaign node acceptance artifacts (verbatim platform snapshots) are parsed
    into :class:`~.model.FailRecord` objects: mission, verbatim criterion row +
    sha256, measured / reference / allowed / error fields where the table
    carries them, and an attribution label with its basis.  The report digest is
    re-verified before extraction (fail-closed on a tampered artifact).
``blockers``
    every FAIL declares one blocker (capability slug / upstream mission /
    numerical-caliber change); the declarations plus the campaign DAG form a
    directed graph whose resolvability is an annotation, not a guess.
``scheduler``
    a blocker that reads ``done`` emits at most one revival task, keyed by
    ``sha256(criterion_digest | input_digest)`` -- so a re-tick cannot duplicate
    work and the task body carries no editable threshold.
``gate``
    revival requests are refused unless they present the registered criterion
    digest and an input digest; any extra field (threshold, verdict, metric) is a
    classified rejection.
``adjudicate``
    three-way verdict ``PASS`` / ``STILL_FAIL`` / ``NOT_COMPARABLE``; a
    non-comparable outcome must list the changes (inputs, caliber, code version)
    and stays a member of the FAIL family.
``audit`` / ``lineage``
    a verdict whose criterion digest is not the registered one is refused and
    marked ``OVERCLAIM`` in the audit ledger; every accepted verdict writes one
    lineage edge (old verdict -> new verdict, blocker, criterion digest,
    timestamp) that a single SQL query answers.

Zero domain semantics: all lexical data (column keywords, verdict markers,
attribution rules) lives in ``layout.json``, and the externalised vocabulary of
``campaign.criterion_prereg`` is used to prove the code stays domain-neutral.
"""
from __future__ import annotations

from . import acceptance
from .adjudicate import (ADJUDICATION_SCHEMA, RevivalOutcome, adjudicate, family_status,
                         parse_number, parse_tolerance)
from .audit import OK, OVERCLAIM, WhitewashRefused, audit_adjudication, audited_adjudicate
from .blockers import (BlockerGraph, BlockerState, blocker_done, blocker_of, build_graph,
                       load_declarations)
from .extract import (ArtifactExtraction, Extraction, classify_cell, criterion_block,
                      cross_check, extract_artifact, extract_directory, iter_tables,
                      load_artifacts, load_layout, scan_fail_marks)
from .gate import GateDecision, request_from_payload, zero_rewrite_gate
from .lineage import connect, edge_of_task, lineage_of, lineage_of_criterion, record_edge
from .model import (BLOCKER_KINDS, FailRecord, RevivalRequest, RevivalTask, VERDICTS, Blocker,
                    canonical_json, sha256_text, task_key)
from .scheduler import EmittedTasks, TickReport, audit_triggers, baseline_input_digest, tick

__all__ = [
    "ADJUDICATION_SCHEMA",
    "ArtifactExtraction",
    "BLOCKER_KINDS",
    "Blocker",
    "BlockerGraph",
    "BlockerState",
    "EmittedTasks",
    "Extraction",
    "FailRecord",
    "GateDecision",
    "OK",
    "OVERCLAIM",
    "RevivalOutcome",
    "RevivalRequest",
    "RevivalTask",
    "TickReport",
    "VERDICTS",
    "WhitewashRefused",
    "acceptance",
    "adjudicate",
    "audit_adjudication",
    "audited_adjudicate",
    "audit_triggers",
    "baseline_input_digest",
    "blocker_done",
    "blocker_of",
    "build_graph",
    "canonical_json",
    "classify_cell",
    "connect",
    "criterion_block",
    "cross_check",
    "edge_of_task",
    "extract_artifact",
    "extract_directory",
    "family_status",
    "iter_tables",
    "lineage_of",
    "lineage_of_criterion",
    "load_artifacts",
    "load_declarations",
    "load_layout",
    "parse_number",
    "parse_tolerance",
    "record_edge",
    "request_from_payload",
    "scan_fail_marks",
    "sha256_text",
    "task_key",
    "tick",
    "zero_rewrite_gate",
]