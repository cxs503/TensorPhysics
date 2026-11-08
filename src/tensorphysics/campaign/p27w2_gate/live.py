"""Thin I/O layer for the P27-W2 integration gate.

The core gate is pure; this module fetches live campaign state from the
TensorPlatform API and writes artifacts into the worktree.  It is
deliberately small and side-effectful so the pure core stays testable.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from . import contracts as C
from . import dag as D
from . import evidence as E
from . import gate as G
from . import maturity as M

DEFAULT_API_BASE = "http://127.0.0.1:8010/api/v1"
P27W2_PLAN_ID = 153
INTEGRATION_NODE_ID = 970
INTEGRATION_MISSION_ID = 6665


def login(api_base: str) -> str:
    body = json.dumps({
        "username": os.environ.get("TENSOR_AGENT_USER", ""),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    }).encode()
    req = urllib.request.Request(
        api_base + "/auth/login", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["access_token"]


def _get(api_base: str, token: str, path: str) -> Any:
    req = urllib.request.Request(
        api_base + path, headers={"Authorization": "Bearer " + token}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def fetch_plan_nodes(api_base: str, token: str,
                     plan_id: int = P27W2_PLAN_ID) -> list[dict]:
    data = _get(api_base, token, f"/tensor/campaigns/plans/{plan_id}/nodes")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("nodes", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def fetch_mission(api_base: str, token: str, mission_id: int) -> dict | None:
    try:
        data = _get(api_base, token, f"/tensor/missions/{mission_id}")
    except Exception:
        return None
    return data.get("mission", data) if isinstance(data, dict) else None


def collect_live_state(api_base: str, token: str,
                       plan_id: int = P27W2_PLAN_ID) -> dict:
    """Fetch plan nodes + per-node mission state; returns a plain dict."""
    nodes = fetch_plan_nodes(api_base, token, plan_id)
    missions: dict[int, dict | None] = {}
    for n in nodes:
        mid = n.get("mission_id")
        if mid and mid not in missions:
            missions[mid] = fetch_mission(api_base, token, int(mid))
    return {"plan_id": plan_id, "nodes": nodes, "missions": missions}


def build_ledger_from_live(state: dict,
                           commits: dict[int, str | None] | None = None) -> E.EvidenceLedger:
    """Build the evidence ledger from live state.

    ``commits`` optionally maps node id -> branch commit hash (discovered
    by the caller from the mission branches).
    """
    commits = commits or {}
    entries = []
    for n in state["nodes"]:
        mid = n.get("mission_id")
        mission = state["missions"].get(mid)
        entry = E.entry_from_mission(n, mission, commit=commits.get(int(n["id"])))
        entries.append(entry)
    return E.build_evidence_ledger(entries)


def build_maturity_from_live(state: dict,
                             artifacts: dict[int, dict | None] | None = None) -> M.MaturityReport:
    """Build the maturity report from live mission results.

    ``artifacts`` optionally overrides per-node parsed result artifacts
    (e.g. loaded from the node worktrees).
    """
    artifacts = dict(artifacts or {})
    for n in state["nodes"]:
        nid = int(n["id"])
        if nid not in M.P27W2_MATURITY_NODES:
            continue
        if nid in artifacts:
            continue
        mission = state["missions"].get(n.get("mission_id"))
        result = (mission or {}).get("result") or {}
        if result:
            artifacts[nid] = result
    return M.build_maturity_report(artifacts)


def run_live_gate(
    api_base: str = DEFAULT_API_BASE,
    token: str | None = None,
    plan_id: int = P27W2_PLAN_ID,
    contract_payloads: dict[str, dict] | None = None,
    capabilities: list[dict] | None = None,
    test_summary: dict | None = None,
    commits: dict[int, str | None] | None = None,
    maturity_artifacts: dict[int, dict | None] | None = None,
) -> tuple[G.GateVerdict, dict]:
    """Fetch live state, run the gate, return (verdict, state)."""
    if token is None:
        token = login(api_base)
    state = collect_live_state(api_base, token, plan_id)
    dag = D.build_dag_from_plan(state["nodes"])
    ledger = build_ledger_from_live(state, commits)
    maturity = build_maturity_from_live(state, maturity_artifacts)
    contract_results = C.validate_all(contract_payloads or {})
    verdict = G.run_gate(
        dag=dag,
        integration_node_id=INTEGRATION_NODE_ID,
        contract_results=contract_results,
        maturity_report=maturity,
        ledger=ledger,
        capabilities=capabilities or [],
        test_summary=test_summary or {},
        plan_id=plan_id,
    )
    return verdict, state


def write_all_outputs(
    out_dir: str | Path,
    verdict: G.GateVerdict,
    state: dict,
    ledger: E.EvidenceLedger,
    maturity: M.MaturityReport,
    contract_results: list[C.ContractResult],
    capabilities: list[dict],
    test_summary: dict,
) -> dict[str, str]:
    """Persist every gate artifact; returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def dump(name: str, payload: Any) -> None:
        p = out / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        paths[name] = str(p)

    dump("campaign_nodes.json", state.get("nodes", []))
    dump("evidence_ledger.json", ledger.as_dict())
    dump("maturity_report.json", maturity.as_dict())
    dump("contract_results.json", [r.as_dict() for r in contract_results])
    dump("capability_ledger.json", {"capabilities": capabilities})
    dump("pytest_summary.json", test_summary)
    paths.update(G.write_outputs(out, verdict, extra={
        "evidence_ledger": ledger.as_dict(),
        "maturity_report": maturity.as_dict(),
        "contracts": [r.as_dict() for r in contract_results],
        "capabilities": capabilities,
        "pytest": test_summary,
    }))
    return paths
