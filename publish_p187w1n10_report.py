"""Publish the P187-W1-N10 external-validity report to the Tensor platform API.

Reads the committed audit artefact + report and posts them (figures base64-embedded)
to ``$TENSOR_API_BASE/tensor/reports``.  Credentials come from the environment; no
secret is printed or stored.
"""

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "failure_taxonomy_external"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode()[:600])
        raise


token = None
for path in ("/auth/login",):
    try:
        data = post(path, {"username": USER, "password": PASS})
    except Exception as exc:  # noqa: BLE001
        print("login attempt failed:", path, type(exc).__name__)
        continue
    token = data.get("access_token") or data.get("token") or (data.get("data") or {}).get("access_token")
    if token:
        break
if not token:
    raise SystemExit("could not obtain a token")

d = json.loads((ART / "audit.json").read_text(encoding="utf-8"))
markdown = (ROOT / "docs" / "FAILURE_TAXONOMY_EXTERNAL_VALIDITY_P187W1N10.md").read_text(encoding="utf-8")

figures = []
for name, caption in (
    (
        "fig1_transfer_recall",
        "Per-class recall of the rule annotator: frozen corpus vs document-disjoint held-out corpus "
        "(24 records, 17 unseen documents); chance line at 1/6.",
    ),
    (
        "fig2_risk_coverage",
        "Risk-coverage of the score-margin abstention gate. Floor chosen on the in-corpus curve, applied "
        "unchanged to the held-out curve.",
    ),
    (
        "fig3_leak_arms",
        "Five accuracy arms isolating the document leak: record-level CV vs document-grouped CV on the same "
        "frozen corpus.",
    ),
):
    png = ART / f"{name}.png"
    figures.append(
        {
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(png.read_bytes()).decode("ascii"),
            "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        }
    )

tr = d["transfer"]["rule_zero_refit"]
leak = d["leak"]
sel = d["selective"]
report = {
    "title": (
        "Failure-taxonomy external validity: document-disjoint transfer, document leak and margin-gated "
        "rerouting (P187-W1-N10)"
    ),
    "summary": (
        "In-corpus bars pass (kappa 0.750, agreement 0.792) but the document-disjoint protocol does not: on 24 records "
        f"from 17 unseen documents transfer is {tr['accuracy']:.3f} accuracy / kappa {tr['kappa']:.3f} (bars 0.70/0.60 -> "
        f"honest FAIL). Record-level CV overstates the supervised arm by {abs(leak['accuracy_gap_grouped_minus_record_level']):.3f}. "
        "Loss sits on BASELINE_ABSENT -> DATA_INSUFFICIENCY; decoy/permuted controls collapse off-corpus. Gate: route 4 stable "
        f"classes, escalate 2; margin gate cuts held-out risk {sel['heldout_ungated']['risk']:.3f}->{sel['heldout_gated']['risk']:.3f} "
        f"at coverage {sel['heldout_gated']['coverage']:.3f}."
    ),
    "scenario_key": "P187-W1-N10-failure-taxonomy-external-validity",
    "mission_id": 9812,
    "tags": [
        "failure-taxonomy",
        "reroute-decision-table",
        "external-validity",
        "document-disjoint-holdout",
        "cohen-kappa",
        "selective-classification",
        "zero-domain-semantics",
        "honest-fail",
    ],
    "markdown": markdown,
    "figures": figures,
}

resp = post("/tensor/reports", report, token)
print("report created:", json.dumps(resp, ensure_ascii=False)[:500])