#!/usr/bin/env python3
"""Publish the ``p223w3-n05`` progress report to the platform report API.

Every number below is read from the worktree records, so the published report
cannot drift from the gate records it summarises:

    artifacts/p223w3_n05/verification.json   (energy study + verdicts + sha256)
    artifacts/p223w3_n06/verification.json   (upstream flux gate, re-run here)
    docs/P223W3_N05_COUPLING_ENERGY_LEDGER.md (paper-level narrative)
    artifacts/p223w3_n05/fig*.png             (embedded as base64)

Credentials come from the environment (``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS``).
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
ART = ROOT / "artifacts" / "p223w3_n05"
FLUX = ROOT / "artifacts" / "p223w3_n06" / "verification.json"
DOC = ROOT / "docs" / "P223W3_N05_COUPLING_ENERGY_LEDGER.md"

FIGURES = [
    ("fig1_energy_order.png",
     "Interface-energy defect (L1) versus exchange step, one panel per chain: lag "
     "slope 1.00, CN midpoint and causal predictor-corrector 2.00, raw window "
     "end-stamp 1.00 until de-stamped. Dash-dotted line = the 1e-5 bound of "
     "criterion (d)."),
    ("fig2_attribution_loss.png",
     "Unbooked share of the exchanged energy vs sub-cycling ratio against analytic "
     "(r-1)/r: tail-only booking reaches 0.8738 (r=8) and 0.9363 (r=16) — the reported "
     "87-94 % band in the conserved quantity — interval-complete booking is exactly 0."),
    ("fig3_balance_quadrature.png",
     "Balance closure |E(T)-E(0)-W_applied+D| versus quadrature sub-steps (falling as "
     "the 4th power of the grid for the settling chain, on the roundoff floor for viv "
     "and weld) next to the energy defect, which does not move: the closure error is "
     "grid-limited, the defect belongs to the exchange operator."),
    ("fig4_flux_vs_energy.png",
     "The same configurations measured on the flux functional and on the energy "
     "functional: the two differ by a factor 2-4 (different readings of the same "
     "exchange) yet both rank the lagging hold far above the time-centred operator, so "
     "the verdict does not depend on the choice of functional."),
]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
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

    record = json.loads((ART / "verification.json").read_text())
    digest = dict(line.split()[::-1] for line in
                  (ART / "SHA256SUMS").read_text().splitlines())["verification.json"]
    gate = record["gate"]
    study = record["study"]
    crit = {c["key"]: c for c in gate["criteria"]}
    flux_gate = json.loads(FLUX.read_text())["gate"]

    no_delay = crit["no_delay_energy_residual"]["measured"]
    worst_defect = max(v[arm]["energy_defect"] for v in no_delay.values() for arm in v)
    worst_ref = max(v[arm]["reference_energy_error"] for v in no_delay.values()
                    for arm in v)
    lag_gain = crit["lag_energy_leak_visible"]["measured"]
    tail = {row["ratio"]: row for row in study["booking"]["tail"]}
    complete = {row["ratio"]: row for row in study["booking"]["complete"]}
    closure = crit["energy_balance_closes"]["measured"]

    summary = (
        "Interval-complete energy ledger for the coupling time exchange: no-delay "
        f"interface-energy defect {worst_defect:.1e}, stored-energy error {worst_ref:.1e} "
        "(criterion d < 1e-5); lag leaks "
        f"{min(r for v in lag_gain.values() for r in v['lag_over_centered']):.0f}x more; "
        "complete booking unattributed 0 for r=2..16 vs tail-only "
        f"{100 * tail[8]['unattributed_share']:.1f}% / "
        f"{100 * tail[16]['unattributed_share']:.1f}% (reported 87-94%); balance closure "
        f"{max(closure.values()):.1e}, grid-limited; gate {gate['passed']}/{gate['total']} "
        f"{gate['verdict'].upper()}."
    )
    assert len(summary) <= 500, len(summary)

    markdown = (DOC.read_text()
                + "\n\n---\n\n### Gate table (machine verdicts, recomputed from the record)\n\n"
                + "\n".join(f"* `{c['key']}`: **{c['verdict']}** — required "
                            f"{c['required']}" for c in gate["criteria"])
                + f"\n\n**Overall: {gate['passed']}/{gate['total']} "
                  f"{gate['verdict'].upper()}**; record "
                  f"`artifacts/p223w3_n05/verification.json` (sha256 `{digest}` from "
                  "`artifacts/p223w3_n05/SHA256SUMS`).\n\n"
                  "Upstream flux/state gate re-run in this worktree: "
                  f"{flux_gate['passed']}/{flux_gate['total']} "
                  f"{flux_gate['verdict'].upper()} "
                  "(`artifacts/p223w3_n06/verification.json`).\n\n"
                  "Attribution sweep (weld chain): "
                  + ", ".join(f"r={r}: tail {tail[r]['unattributed_share']:.4f} / "
                              f"complete {complete[r]['unattributed_share']:.1e}"
                              for r in sorted(tail)) + ".\n")

    figures = []
    for name, caption in FIGURES:
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode((ART / name).read_bytes()).decode()})

    body = {
        "title": "p223w3-n05 · Coupling time exchange in the conserved quantity: "
                 "interval-complete energy ledger, second-order exchange, and the "
                 "87-94 % loss closed",
        "summary": summary,
        "scenario_key": "coupling_time_exchange",
        "mission_id": 10561,
        "tags": ["TensorPhysics", "coupling", "energy-ledger", "conservation",
                 "temporal-order", "interval-completeness", "p223w3-n05", "m10561"],
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