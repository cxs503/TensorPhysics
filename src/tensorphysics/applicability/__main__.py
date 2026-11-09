"""Command line entry point: probe registered capabilities, write an envelope artifact.

Examples::

    python -m tensorphysics.applicability --out artifacts/applicability/envelopes.json
    python -m tensorphysics.applicability --figures artifacts/applicability
    python -m tensorphysics.applicability --query '{"<capability id>": {"<axis>": 42.0}}'

The artifact is the machine-readable form of what used to be a paragraph in a
report: per capability, per axis, the searched span, the in-domain intervals,
the refined boundaries with confidence, and the comparison against the
human-written declaration.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Mapping

from .capability import CapabilityRegistry
from .envelope import DomainStatus, assign_grade
from .probe import probe_registry
from .refine import DEFAULT_REL_TOL

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent / "configs" / "default_capabilities.v1.json"


def _parse_queries(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    text = pathlib.Path(raw).read_text(encoding="utf-8") if pathlib.Path(raw).is_file() else raw
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise SystemExit("--query must be a JSON object {capability_id: {axis: value}}")
    return [{"label": str(k), "capability_id": str(k),
             "point": {str(a): float(v) for a, v in dict(point).items()}}
            for k, point in payload.items()]


def _summary_lines(suite, registry) -> list[str]:
    lines: list[str] = []
    for capability_id in registry.ids():
        report = suite.reports[capability_id]
        envelope = report.envelope
        for name, axis_envelope in envelope.axes.items():
            intervals = "; ".join(
                f"[{iv.low:.6g}, {iv.high:.6g}]{'*' if iv.low_open else ''}"
                f"{'*' if iv.high_open else ''}" for iv in axis_envelope.intervals) or "<empty>"
            bounds = ", ".join(
                f"{b.side}={b.value:.6g} (conf {b.confidence_label}, {b.source})"
                for b in axis_envelope.boundaries) or "none"
            compat = ""
            if report.compat is not None:
                compat = f"  declaration: {report.compat.verdict}"
            lines.append(f"{capability_id} :: {name} in {intervals} "
                         f"[{axis_envelope.samples_in}/{axis_envelope.samples_total} in] "
                         f"boundaries: {bounds}{compat}")
    lines.append("('*' marks an open interval end: no failure was found inside the searched span)")
    return lines


def build_artifact(config_path: pathlib.Path, *, rel_tol: float = DEFAULT_REL_TOL,
                   queries: Any = None, refine: bool = True) -> tuple[dict[str, Any], Any, Any]:
    """Probe the configuration and return ``(artifact, suite, registry)``.

    ``queries`` accepts either ``{capability_id: {axis: value}}`` or a list of
    ``{"label": ..., "capability_id": ..., "point": {axis: value}}`` records, so a
    report can ask several questions about the same capability.
    """
    if queries is None:
        query_list: list[dict[str, Any]] = []
    elif isinstance(queries, Mapping):
        query_list = [{"label": str(k), "capability_id": str(k),
                       "point": {str(a): float(v) for a, v in dict(point).items()}}
                      for k, point in queries.items()]
    else:
        query_list = [dict(q) for q in queries]
    registry = CapabilityRegistry.from_config_file(config_path)
    suite = probe_registry(registry, rel_tol=rel_tol, refine=refine)
    artifact = suite.to_dict()
    artifact["config_path"] = str(config_path)
    artifact["config_digest"] = _config_digest(config_path)
    verdicts: list[dict[str, Any]] = []
    for query in query_list:
        capability_id = str(query["capability_id"])
        point = {str(a): float(v) for a, v in dict(query["point"]).items()}
        spec = registry.get(capability_id)
        envelope = suite.envelopes()[capability_id]
        verdict = envelope.classify(point, spec=spec)
        record = assign_grade(verdict, "REAL", strict=False)
        verdicts.append({"label": str(query.get("label", capability_id)),
                         "verdict": verdict.to_dict(), "grade": record.to_dict()})
    artifact["queries"] = verdicts
    out_of_domain = sum(1 for v in verdicts if v["verdict"]["domain_status"] == DomainStatus.OUT.value)
    artifact["query_summary"] = {"total": len(verdicts), "out_of_domain": out_of_domain}
    return artifact, suite, registry


def _config_digest(path: pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tensorphysics.applicability",
                                     description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="capability configuration JSON (default: shipped real capabilities)")
    parser.add_argument("--out", default=None, help="write the envelope artifact JSON here")
    parser.add_argument("--figures", default=None, help="directory for matplotlib figures")
    parser.add_argument("--query", default=None,
                        help="JSON object (inline or a path) {capability_id: {axis: value}}")
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL,
                        help="bisection resolution as a fraction of the axis span (default 1e-3)")
    parser.add_argument("--no-refine", action="store_true", help="skip stage-3 bisection")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    artifact, suite, registry = build_artifact(pathlib.Path(args.config), rel_tol=args.rel_tol,
                                               queries=_parse_queries(args.query),
                                               refine=not args.no_refine)
    if not args.quiet:
        for line in _summary_lines(suite, registry):
            print(line)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"artifact: {out} (suite digest {artifact['suite_digest']})")
    if args.figures:
        from .figures import make_figures

        paths = make_figures(suite, registry, pathlib.Path(args.figures))
        if not args.quiet:
            for path in paths:
                print(f"figure: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())