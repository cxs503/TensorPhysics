#!/usr/bin/env python3
"""Publish the ``p223w3-n06`` progress report to the platform report API.

Everything is read from the worktree artifacts, so the published report cannot
drift from the gate record it summarises:

    artifacts/p223w3_n06/verification.json   (study + verdicts + sha256)
    docs/P223W3_N06_COUPLING_TIME_EXCHANGE.md (paper-level narrative)
    artifacts/p223w3_n06/fig*.png             (embedded as base64)

Credentials come from the environment (``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS``); nothing is written to disk except the request body.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p223w3_n06"
DOC = ROOT / "docs" / "P223W3_N06_COUPLING_TIME_EXCHANGE.md"

FIGURES = [
    ("fig1_operator_order.png",
     "Log-log exchange defect versus exchange step for the six operator arms on the "
     "coupling manufactured solution. Slopes: lag 1.00, CN midpoint 2.00, causal "
     "predictor-corrector 1.96 (provisional) / 2.00 (corrected), end-stamped window "
     "average 0.98, de-stamped window average 2.01."),
    ("fig2_staircase_repair.png",
     "Exchange-cadence ladders on the VIV / weld / settling chains: lag + end stamp, "
     "CN + end stamp (the reported practice, flat plateau), CN + de-stamp (the repair) "
     "and the every-sub-window arm (exact)."),
    ("fig3_attribution_loss.png",
     "Exchanged quantity never booked versus sub-cycling ratio for tail-only booking, "
     "against the analytic 1-1/r and the reported 87-94 % loss band: 87.4 % at r=8 and "
     "93.6 % at r=16."),
    ("fig4_chain_trajectories.png",
     "Receiver deviation from the solve_ivp oracle at r=8 for the three chains, broken "
     "versus repaired exchange (symlog vertical axis, normalised by the oracle span)."),
]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not user or not password:
        print("TENSOR_AGENT_USER / TENSOR_AGENT_PASS are not set", file=sys.stderr)
        return 2

    token = _post(f"{base}/auth/login",
                  {"username": user, "password": password}).get("access_token")
    if not token:
        print("login did not return an access token", file=sys.stderr)
        return 2

    payload = json.loads((ART / "verification.json").read_text())
    record_digest = dict(line.split()[::-1] for line in
                         (ART / "SHA256SUMS").read_text().splitlines())["verification.json"]
    gate = payload["gate"]
    study = payload["study"]
    tail = {r["ratio"]: r["unattributed_share"] for r in study["attribution"]["tail_sweep"]}
    crit = {c["key"]: c for c in gate["criteria"]}
    mms = study["mms"]

    summary = (
        f"coupling exchange algebra (lag/centered/predictor-corrector, one interface) + "
        f"interval-complete attribution ledger; MMS orders {mms['lag']['observed_order']:.2f}/"
        f"{mms['centered']['observed_order']:.2f}/"
        f"{mms['predictor_corrector']['observed_order']:.2f}; "
        f"tail-only booking loses {100*tail[8]:.1f}% / {100*tail[16]:.1f}% at r=8/16 "
        f"(reported 87-94%); end-stamped window exchange plateaus "
        f"(order {study['chains']['weld']['arms']['centered_endstamp']['observed_order']:.2f}, "
        f"weld state plateau {crit['weld_plateau_matches_reported']['measured']['plateau_state_error']:.2e} "
        f"vs reported 1.2e-3); CN + de-stamping cuts the defect "
        f"{min(v['defect_gain'] for v in crit['repair_removes_plateau']['measured'].values()):.0f}x; "
        f"gate {gate['passed']}/{gate['total']} {gate['verdict'].upper()}."
    )

    markdown = (DOC.read_text()
                + "\n\n---\n\n### Gate table (machine verdicts)\n\n"
                + "\n".join(f"* `{c['key']}`: **{c['verdict']}** — required {c['required']}"
                            for c in gate["criteria"])
                + f"\n\n**Overall: {gate['passed']}/{gate['total']} "
                  f"{gate['verdict'].upper()}**; record "
                  f"`artifacts/p223w3_n06/verification.json` "
                  f"(sha256 `{record_digest}` from `artifacts/p223w3_n06/SHA256SUMS`).\n")

    figures = []
    for name, caption in FIGURES:
        path = ART / name
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(path.read_bytes()).decode()})

    body = {
        "title": "p223w3-n06 · Coupling time-exchange operator algebra, second order, "
                 "and the interval-complete conservation ledger",
        "summary": summary,
        "scenario_key": "coupling_time_exchange",
        "mission_id": 9722,
        "tags": ["TensorPhysics", "coupling", "temporal-order", "conservation-ledger",
                 "MMS", "p223w3-n06", "m9722"],
        "markdown": markdown,
        "figures": figures,
    }
    try:
        resp = _post(f"{base}/tensor/reports", body, token)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        print(f"report POST failed: {exc.code} {exc.read()[:400]!r}", file=sys.stderr)
        return 1
    print(json.dumps(resp)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())