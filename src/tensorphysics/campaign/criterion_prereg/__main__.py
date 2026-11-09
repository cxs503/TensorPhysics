"""CLI for the criterion pre-registration gate.

``python -m tensorphysics.campaign.criterion_prereg freeze --goal-file g.txt``
``python -m tensorphysics.campaign.criterion_prereg audit  --frozen f.json --receipt-file r.txt``
``python -m tensorphysics.campaign.criterion_prereg scan``
``python -m tensorphysics.campaign.criterion_prereg synth-eval``

Exit codes are fail-closed: ``2`` on ``unregistered``/``tampered``, ``1`` on a
``FAIL`` drift verdict, ``0`` only when the audit passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .criterion_manifest import freeze, verify_manifest
from .drift_audit import audit, kpi_view
from .scan import scan_paths
from .synth import baseline_text_hash, detection_metrics, field_scores, gen_drift_cases, gen_extraction_corpus


def _read(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def _load_manifest(path: str) -> dict:
    """Accept either a bare manifest or a ``freeze`` wrapper (``{"state", "manifest"}``)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("manifest"), dict):
        return payload["manifest"]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="criterion_prereg")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze", help="freeze criteria from a goal text")
    p_freeze.add_argument("--goal-file", required=True)
    p_freeze.add_argument("--mission-id", default=None)
    p_freeze.add_argument("--node-id", default=None)
    p_freeze.add_argument("--frozen-by", default="")
    p_freeze.add_argument("--out", default=None)

    p_audit = sub.add_parser("audit", help="audit a receipt against a frozen manifest")
    p_audit.add_argument("--frozen", required=True)
    p_audit.add_argument("--receipt-file", default=None)
    p_audit.add_argument("--out", default=None)
    p_audit.add_argument("--kpi-out", default=None)

    p_scan = sub.add_parser("scan", help="zero-domain-semantics scan")
    p_scan.add_argument("paths", nargs="*")

    p_synth = sub.add_parser("synth-eval", help="run the synthetic acceptance corpora")
    p_synth.add_argument("--out", default=None)
    p_synth.add_argument("--seed", type=int, default=20261109)

    p_verify = sub.add_parser("verify", help="re-derive the digest of a frozen manifest")
    p_verify.add_argument("--frozen", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        result = freeze(
            _read(args.goal_file),
            mission_id=args.mission_id,
            node_id=args.node_id,
            frozen_by=args.frozen_by,
        )
        payload = result.to_dict()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return 0 if result.registered else 2

    if args.cmd == "audit":
        manifest = _load_manifest(args.frozen)
        report = audit(manifest, receipt_goal=_read(args.receipt_file) if args.receipt_file else None)
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        if args.kpi_out:
            Path(args.kpi_out).write_text(json.dumps(kpi_view([report]), ensure_ascii=False, indent=2), encoding="utf-8")
        print(text)
        if report.state != "registered":
            return 2
        return 0 if not report.failed() else 1

    if args.cmd == "scan":
        result = scan_paths(args.paths or None)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.clean and result.vocabulary_loaded else 1

    if args.cmd == "verify":
        manifest = _load_manifest(args.frozen)
        state = verify_manifest(manifest)
        print(json.dumps({"state": state, "manifest_sha256": manifest.get("manifest_sha256", "")}, indent=2))
        return 0 if state == "registered" else 2

    if args.cmd == "synth-eval":
        ext = gen_extraction_corpus(20, seed=args.seed)
        field = field_scores(ext)
        cases = gen_drift_cases(seed=args.seed)
        preds, base_preds = [], []
        for case in cases:
            fz = freeze(case.frozen_goal, frozen_at="2026-11-09T00:00:00+00:00")
            report = audit(fz.manifest, receipt_goal=case.receipt_goal)
            klass = report.dominant_class
            preds.append((report.failed(), klass))
            base_preds.append((baseline_text_hash(case.frozen_goal, case.receipt_goal), "drift" if baseline_text_hash(case.frozen_goal, case.receipt_goal) else "unchanged"))
        payload = {
            "extraction_field_f1": field["f1"],
            "extraction": {k: v for k, v in field.items() if k != "error_cases"},
            "drift": detection_metrics(preds, cases),
            "baseline_text_hash": detection_metrics(base_preds, cases),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=float)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())