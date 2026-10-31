#!/usr/bin/env python3
"""Publish the ice-induced vibration campaign report to the TensorPlatform.

Reads the evidence bundle in ``artifacts/ice_vibration/`` (report.json,
metrics.json, pytest_summary.json, docs/ICE_VIBRATION_LOCKIN.md and the figures)
and posts a single ``/tensor/reports`` payload.  Nothing is typed by hand: the
summary, the criteria table and the figure captions are all built from the
artifacts.

Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``) and are never printed.

Usage:
    python3 scenarios/ice_vibration_lockin/post_report.py [--mission-id 788] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKTREE = Path(__file__).resolve().parents[2]
ARTIFACTS = WORKTREE / "artifacts" / "ice_vibration"
DOC = WORKTREE / "docs" / "ICE_VIBRATION_LOCKIN.md"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def api(method: str, path: str, payload: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None, timeout: int = 120) -> Any:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return exc.code, body
    except Exception as exc:  # noqa: BLE001
        return -1, repr(exc)


def login() -> str:
    status, payload = api(
        "POST",
        "/auth/login",
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    )
    if status != 200 or not isinstance(payload, dict) or "access_token" not in payload:
        raise SystemExit(f"login failed: {status} {payload}")
    return payload["access_token"]


def criteria_table(criteria: Dict[str, Any]) -> str:
    lines = ["| criterion | value | threshold | result |", "| --- | ---: | ---: | --- |"]
    for entry in criteria["entries"]:
        value = entry["value"]
        value_text = f"{value:.6g}" if isinstance(value, (int, float)) else str(value)
        lines.append(
            f"| `{entry['criterion']}` | {value_text} | {entry['threshold']} | "
            f"{'PASS' if entry['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def build_payload(mission_id: int) -> Dict[str, Any]:
    report = load_json(ARTIFACTS / "report.json")
    metrics = load_json(ARTIFACTS / "metrics.json")
    pytest_summary = load_json(ARTIFACTS / "pytest_summary.json")
    doc = DOC.read_text()
    headline = metrics["headline"]
    criteria = report["criteria"]

    summary = (
        "Ice-induced vibration framework added to TensorPhysics "
        "(src/tensorphysics/ice_vibration/): SDOF/MDOF ice-structure coupling, the 5% "
        f"lock-in criterion, self-excited negative-damping lock-in, Duffing harmonic balance "
        f"and two-mode hull-girder whiplash. Amplification at +-5% detuning "
        f"{headline['amplification_at_5pct_detuning_min']:.2f} (>3); Duffing error "
        f"{headline['duffing_max_relative_error']:.2%} (<10%); whiplash peak vs analytic "
        f"{headline['whiplash_midship_moment_relative_error']:.1e} (<8%); self-excited "
        f"lock-in deviation {headline['self_excited_frequency_deviation_operational']:.2%} "
        f"(<5%, U>=0.2 m/s); pytest {pytest_summary['passed']} passed; criteria "
        f"{criteria['n_passed']}/{criteria['n_criteria']}."
    )
    if len(summary) > 500:
        summary = summary[:497] + "..."

    markdown = "\n".join([
        doc,
        "",
        "---",
        "",
        "## Appendix A. Machine-checked criteria (from artifacts/ice_vibration/report.json)",
        "",
        criteria_table(criteria),
        "",
        f"`all_passed = {criteria['all_passed']}` — "
        f"{criteria['n_passed']}/{criteria['n_criteria']} criteria satisfied.",
        "",
        "## Appendix B. pytest",
        "",
        f"`{pytest_summary['summary_tail']}` for "
        f"`{pytest_summary['command']}` (run {pytest_summary['run_utc']}).",
        "",
        "## Appendix C. Headline numbers (from artifacts/ice_vibration/metrics.json)",
        "",
        "```json",
        json.dumps(headline, indent=2),
        "```",
        "",
        "## Appendix D. Artifact inventory",
        "",
        "\n".join(f"- `{path.relative_to(WORKTREE)}`"
                  for path in sorted(ARTIFACTS.rglob("*")) if path.is_file()
                  and path.suffix in {".json", ".csv", ".png"}),
    ])

    figures: List[Dict[str, str]] = []
    for figure in metrics["figures"]:
        path = WORKTREE / figure["path"]
        caption = figure["caption"]
        if len(caption) > 300:
            caption = caption[:297] + "..."
        figures.append({
            "name": figure["name"],
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        })

    return {
        "title": "W3-N01 Ice-induced vibration: frequency lock-in and transient whiplash response",
        "summary": summary,
        "scenario_key": "ice_vibration_lockin_w3n01",
        "mission_id": mission_id,
        "tags": [
            "ice-induced-vibration", "frequency-lock-in", "whiplash", "hull-girder",
            "duffing-oscillator", "modal-superposition", "TensorPhysics", "W3-N01", "pytest",
        ],
        "markdown": markdown,
        "figures": figures,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-id", type=int, default=788)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = build_payload(args.mission_id)
    size_kb = len(json.dumps(payload)) / 1024.0
    print(f"payload: {size_kb:.0f} KiB, {len(payload['figures'])} figures, "
          f"markdown {len(payload['markdown'])/1024:.1f} KiB")
    if args.dry_run:
        Path(WORKTREE / "artifacts" / "ice_vibration" / "_report_payload.json").write_text(
            json.dumps({k: v for k, v in payload.items() if k != "figures"}, indent=2)
        )
        return 0
    token = login()
    status, response = api("POST", "/tensor/reports", payload, token=token)
    print("POST /tensor/reports ->", status)
    print(json.dumps(response, indent=2)[:1500] if isinstance(response, dict) else response)
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())