"""P35-G1 campaign integration acceptance gate (cross-domain, ten nodes).

Public surface::

    from tensorphysics.campaign.p35_gate import run_gate, write_outputs, spec

    snapshot = build_snapshot(client, plan_id=spec.PLAN_ID)
    scans = {node.ext_id: scan_node_repository(root, node) for node in spec.NODES}
    readiness = run_gate(snapshot, scans=scans)
    write_outputs(readiness, "artifacts/p35_gate")

``readiness`` is the ``campaign_readiness.v1`` payload the dispatcher consumes:
``overall_status`` / ``dispatcher_ready`` / seven criteria / ten node rows /
regression records / blocking list / provenance.  It is validated against the
registered schema before it is written (:func:`run_gate`), and a payload that
cannot be validated is never reported as criterion (f) PASS.
"""
from __future__ import annotations

from . import collect, criteria, gate, readiness_schema, regression, spec
from .collect import (PlatformClient, build_snapshot, load_snapshot, parse_repo_roots,
                      scan_node_repository, write_snapshot)
from .gate import (GENERATOR, LIMITATIONS, READINESS_JSON, READINESS_MD, READINESS_SCHEMA,
                   build_readiness, render_markdown, run_gate, validate_readiness,
                   write_outputs)
from .regression import ProbeResult, evaluate_regression, run_probe

__all__ = [
    "GENERATOR", "LIMITATIONS", "PlatformClient", "ProbeResult", "READINESS_JSON",
    "READINESS_MD", "READINESS_SCHEMA", "build_readiness", "build_snapshot", "collect",
    "criteria", "evaluate_regression", "gate", "load_snapshot", "parse_repo_roots",
    "readiness_schema", "regression", "render_markdown", "run_gate", "run_probe", "scan_node_repository",
    "spec", "validate_readiness", "write_outputs", "write_snapshot",
]