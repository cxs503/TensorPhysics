"""Lineage inheritance adjudication (P84-W2 -> P84-W2C).

A "wave 2C" campaign is a *lineage* event: ten nodes are re-seeded from the
previous wave, five of them as maturity (成熟化) promotions.  The gate has to
answer, per node, whether the node inherits a verified parent capability or
seeds something new -- otherwise a re-seeded node could claim credit for work
that never landed, and a genuinely new node could hide inside the lineage.

The adjudication uses platform evidence (parent plan node title + status) and
falls back to the spec-declared parent title when the API is unavailable, in
which case the verdict is explicitly ``inherited_unverified``: inheritance is
plausible but unproven.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def index_parent_nodes(parent_nodes: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Map normalized parent node name -> parent node record."""
    out: dict[str, Mapping[str, Any]] = {}
    for value in (parent_nodes or {}).values():
        if isinstance(value, Mapping) and value.get("name"):
            out[_norm(value["name"])] = value
    return out


def adjudicate(node: Mapping[str, Any],
               parent_index: Mapping[str, Mapping[str, Any]],
               declared_parent_title: str) -> dict[str, Any]:
    candidates = [declared_parent_title] if declared_parent_title else []
    candidates.append(node.get("title", ""))
    parent = None
    for title in candidates:
        key = _norm(title)
        if not key:
            continue
        parent = parent_index.get(key)
        if parent is None:
            # tolerate the "N01: ..." prefix differences by comparing the suffix
            for norm_name, record in parent_index.items():
                if norm_name.endswith(key) or key.endswith(norm_name):
                    parent = record
                    break
        if parent is not None:
            break
    if parent is None:
        return {
            "parent_node": None,
            "parent_mission_id": None,
            "parent_status": None,
            "verdict": "seeded_new",
            "note": "父计划中无同名节点：本节点为新种子",
        }
    status = str(parent.get("status") or "unknown")
    verdict = "inherited_verified" if status == "done" else "inherited_unverified"
    return {
        "parent_node": parent.get("name"),
        "parent_mission_id": parent.get("mission_id"),
        "parent_status": status,
        "verdict": verdict,
        "note": ("父节点已完成，继承链可验证" if verdict == "inherited_verified"
                 else f"父节点状态 {status}，继承链未经父节点验收背书"),
    }


def adjudicate_all(nodes: Sequence[Mapping[str, Any]],
                   parent_nodes: Mapping[str, Any],
                   declared_titles: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_index = index_parent_nodes(parent_nodes)
    offline = not parent_index
    results: list[dict[str, Any]] = []
    counts = {"inherited_verified": 0, "inherited_unverified": 0, "seeded_new": 0}
    for node in nodes:
        lineage = adjudicate(node, parent_index, declared_titles.get(node["node_key"], ""))
        if offline:
            if declared_titles.get(node["node_key"]):
                lineage["parent_node"] = declared_titles.get(node["node_key"])
                base = "父计划同名节点声明存在但平台证据缺失"
                lineage["verdict"] = "inherited_unverified"
                lineage["note"] = base + "；继承未经父节点验收背书"
            lineage["note"] = (lineage.get("note", "") +
                               "；父计划证据缺失（offline），继承未经平台背书").strip("；")
            if lineage["verdict"] == "inherited_verified":
                lineage["verdict"] = "inherited_unverified"
        counts[lineage["verdict"]] += 1
        results.append(lineage)
    summary = {
        "parent_plan_id": 0,
        "inherited_verified": counts["inherited_verified"],
        "inherited_unverified": counts["inherited_unverified"],
        "seeded_new": counts["seeded_new"],
        "note": "父计划同名节点 done → inherited_verified；否则 inherited_unverified",
    }
    return results, summary