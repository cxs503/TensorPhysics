"""Machine-checked applicability envelopes for registered capabilities.

The mechanism turns "the model's validity range lives in a human-written report
paragraph" into a reproducible artifact:

* **stage 1 -- capability interface.**  A capability is registered as data
  (:mod:`~tensorphysics.applicability.capability`): its parameter axes with the
  span to explore, an evaluator operator, the acceptance criterion, the failure
  signals and -- optionally -- the human-written declaration it must not exceed.
  Axis names, spans and tolerances are injected by configuration; this package
  contains no capability or domain vocabulary.
* **stage 2 -- probing operator.**  :func:`~tensorphysics.applicability.probe.probe_capability`
  walks each axis with a geometric gradient scan and marks every sample with
  three criteria: (a) acceptance criterion against the reference/anchor,
  (b) monotonicity / sign law, (c) numerical pathology (non-finite, declared
  failure flags, convergence-order collapse).
* **stage 3 -- boundary refinement.**  Every in/out neighbour pair is bisected
  down to ``1e-3`` of the axis span and reported with a confidence
  (:mod:`~tensorphysics.applicability.refine`).
* **stage 4 -- out-of-domain verdict.**  A result carries
  ``domain_status in {IN, OUT, UNKNOWN}``; ``OUT`` forces the grading field to
  ``OUT_OF_DOMAIN`` and a ``REAL`` write is refused
  (:func:`~tensorphysics.applicability.envelope.assign_grade`).
* **stage 2b/3b -- joint (multi-axis) probe.**  A capability can fail where two
  parameters are unfavourable *together* while every 1-D scan stays inside; the
  joint probe scans the axis pairs, records conservative out-of-domain cells
  around each witnessed failure and bisects the witness ray, so the offline gate
  cannot certify such a point (:mod:`~tensorphysics.applicability.interaction`).
* **stage 5 -- consumer side.**  The written artifact can be reloaded in another
  process (:func:`~tensorphysics.applicability.artifact.load_envelopes`), its
  suite digest re-verified, and any later point gated without re-running the
  probe or importing the evaluators
  (:func:`~tensorphysics.applicability.artifact.gate`).

Quick start::

    from tensorphysics.applicability import CapabilityRegistry, probe_registry, assign_grade

    registry = CapabilityRegistry.from_config_file(".../configs/default_capabilities.v1.json")
    suite = probe_registry(registry)
    envelope = suite.envelopes()[registry.ids()[0]]
    verdict = envelope.classify({"<axis>": 42.0}, spec=registry.get(registry.ids()[0]))
    record = assign_grade(verdict, "REAL")     # raises when the point is out of domain
"""
from __future__ import annotations

from .artifact import (ARTIFACT_SCHEMA, ArtifactError, gate, gate_queries, load_envelopes,
                       suite_digest)
from .axis import AxisConfigError, AxisSpec
from .capability import (ApplicabilityConfigError, CapabilityRegistry, CapabilitySpec, Criterion,
                         DeclaredAxis, EvaluatorSpec, FailurePolicy, InteractionConfig,
                         import_target)
from .envelope import (ApplicabilityEnvelope, AxisCompat, AxisEnvelope, DomainCompat,
                       DomainStatus, EnvelopeFormatError, GradeRecord, GradeWriteError,
                       InteractionBox, InteractionRegion, Interval, OUT_OF_DOMAIN, Verdict,
                       assign_grade, declaration_compat)
from .evaluation import (CRITERION, ERROR, LAW, PASS, PATHOLOGY, Evaluation, evaluate_point,
                         law_violation)
from .probe import EnvelopeSuite, ProbeReport, ScanRecord, probe_capability, probe_axis, probe_registry
from .interaction import INTERACTION_CASES, InteractionCase, run_interaction_selfcheck, \
    scan_interactions, scan_pair, synthetic_interaction_registry
from .refine import Boundary, bisect_boundary

__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactError",
    "AxisConfigError",
    "AxisSpec",
    "ApplicabilityConfigError",
    "ApplicabilityEnvelope",
    "AxisCompat",
    "AxisEnvelope",
    "Boundary",
    "CapabilityRegistry",
    "CapabilitySpec",
    "Criterion",
    "CRITERION",
    "DeclaredAxis",
    "DomainCompat",
    "DomainStatus",
    "EnvelopeFormatError",
    "ERROR",
    "EvaluatorSpec",
    "Evaluation",
    "InteractionBox",
    "InteractionConfig",
    "InteractionRegion",
    "InteractionCase",
    "INTERACTION_CASES",
    "run_interaction_selfcheck",
    "scan_interactions",
    "scan_pair",
    "synthetic_interaction_registry",
    "EnvelopeSuite",
    "FailurePolicy",
    "GradeRecord",
    "GradeWriteError",
    "Interval",
    "LAW",
    "OUT_OF_DOMAIN",
    "PASS",
    "PATHOLOGY",
    "ProbeReport",
    "ScanRecord",
    "Verdict",
    "assign_grade",
    "bisect_boundary",
    "declaration_compat",
    "evaluate_point",
    "gate",
    "gate_queries",
    "import_target",
    "load_envelopes",
    "law_violation",
    "probe_axis",
    "probe_capability",
    "probe_registry",
    "suite_digest",
]