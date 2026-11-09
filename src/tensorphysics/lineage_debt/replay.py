"""Replay: campaign snapshot -> debt ledger -> coverage matrix -> seed verdict.

The replay is a pure function of the input snapshot: no network, no clock, no
hash-order dependence.  Given the same bundle twice it emits the same bytes
(tested), which is what makes the ledger usable as a campaign gate rather than
as a narrative artefact.

Bundle layout (``artifacts/lineage_debt/inputs/lineage_debt_bundle.json``)::

    {
      "schema": "lineage_debt_input.v1",
      "fetched_utc": "...", "source": {...}, "provenance": {...},
      "plans": {"26": {...}, "173": {...}, "388": {...}},
      "nodes": {"26": [...], "173": [...], "388": [...]},
      "reports": {"8117": {...}, ...},
      "downstream_plan": 388
    }

Per plan the ledger contains:

* **own debts** -- extracted from the reports of the plan's own nodes plus the
  plan's terminal (``blocked``/``failed``) nodes;
* **inherited debts** -- the same extraction run over every ancestor plan
  reached through ``metadata.parent_plan`` (that is the "lineage inheritance
  contract" the campaign only used to *describe* in prose);
* **coverage matrix** -- judged against the descendant plan named by
  ``downstream_plan`` (falling back to ``metadata.autonomy.new_plan`` and then
  to the plan itself);
* **seed verdict** -- fail-closed gate over the open debts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import VocabConfig, load_config
from .coverage import build_matrix
from .criteria import parse_node_criteria
from .extract import (DEFAULT_TERMINAL_STATUSES, extract_node_terminal_debts,
                      extract_report_observations, observations_to_debts)
from .gate import apply_autonomy_patch, seeding_check
from .model import LEDGER_SCHEMA, canonical_json, dedupe_debts, sha256_text, summarize

__all__ = ["BUNDLE_SCHEMA", "LedgerBundle", "load_bundle", "plan_chain", "plan_debts",
           "build_ledger", "write_ledger"]

BUNDLE_SCHEMA = "lineage_debt_input.v1"


@dataclass(frozen=True)
class LedgerBundle:
    raw: Mapping[str, Any]
    plans: Mapping[int, Mapping[str, Any]]
    nodes: Mapping[int, list]
    reports: Mapping[int, Mapping[str, Any]]
    downstream_plan: int | None
    sha256: str


def _int_keys(d: Mapping[Any, Any] | None) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def load_bundle(source: Mapping[str, Any] | str | Path) -> LedgerBundle:
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        with Path(source).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    schema = str(raw.get("schema") or "")
    if schema != BUNDLE_SCHEMA:
        raise ValueError(f"unexpected bundle schema: {schema!r}")
    downstream = raw.get("downstream_plan")
    return LedgerBundle(
        raw=raw,
        plans=_int_keys(raw.get("plans")),
        nodes=_int_keys(raw.get("nodes")),
        reports=_int_keys(raw.get("reports")),
        downstream_plan=None if downstream is None else int(downstream),
        sha256=sha256_text(canonical_json(raw)),
    )


def plan_chain(bundle: LedgerBundle, plan_id: int) -> list[int]:
    """Ancestor chain of ``plan_id`` (root first), cycle-safe and bounded."""
    chain: list[int] = [plan_id]
    seen = {plan_id}
    cur = bundle.plans.get(plan_id) or {}
    meta = cur.get("metadata") or {}
    parent = meta.get("parent_plan")
    guard = 0
    while parent is not None and int(parent) in bundle.plans and int(parent) not in seen and guard < 64:
        chain.append(int(parent))
        seen.add(int(parent))
        parent = ((bundle.plans.get(int(parent)) or {}).get("metadata") or {}).get("parent_plan")
        guard += 1
    return list(reversed(chain))


def _reports_of(node: Mapping[str, Any], bundle: LedgerBundle) -> list[Mapping[str, Any]]:
    mission = node.get("mission_id")
    if mission is None:
        return []
    hits = [r for r in bundle.reports.values() if r.get("mission_id") == mission]
    hits.sort(key=lambda r: (str(r.get("created_at") or ""), int(r.get("id") or 0)))
    return hits


def plan_debts(bundle: LedgerBundle, plan_id: int, cfg: VocabConfig,
               terminal_statuses: tuple[str, ...] = DEFAULT_TERMINAL_STATUSES,
               collect_unresolved: list | None = None) -> list:
    """Debts of one plan: report error rows + terminal nodes (deduped)."""
    chain = plan_chain(bundle, plan_id)
    nodes = bundle.nodes.get(plan_id) or []
    debts = []
    for node in nodes:
        for report in _reports_of(node, bundle):
            obs = extract_report_observations(
                report, cfg, plan_id=plan_id, node_id=node.get("id"), mission_id=node.get("mission_id"))
            if obs:
                debts.extend(observations_to_debts(obs, cfg, plan_chain=chain, node=node,
                                                   collect_unresolved=collect_unresolved))
    debts.extend(extract_node_terminal_debts(nodes, cfg, plan_chain=chain,
                                             terminal_statuses=terminal_statuses))
    return dedupe_debts(debts)


def _criteria_of(bundle: LedgerBundle, plan_id: int, cfg: VocabConfig) -> list:
    out = []
    for node in bundle.nodes.get(plan_id) or []:
        out.extend(parse_node_criteria(node, cfg))
    out.sort(key=lambda c: (int(c.node_id or 0), str(c.criterion_id)))
    return out


def build_ledger(bundle: LedgerBundle, plan_id: int, cfg: VocabConfig | None = None,
                 deferrals: Mapping[str, str] | None = None) -> dict:
    cfg = cfg or load_config()
    chain = plan_chain(bundle, plan_id)
    own = plan_debts(bundle, plan_id, cfg)
    inherited = []
    for ancestor in chain[:-1]:
        inherited.extend(plan_debts(bundle, ancestor, cfg))
    debts = dedupe_debts(own + inherited)

    target = bundle.downstream_plan
    target_source = "bundle"
    if target is None:
        meta = (bundle.plans.get(plan_id) or {}).get("metadata") or {}
        auto = meta.get("autonomy") or {}
        cand = auto.get("new_plan")
        if cand is not None and int(cand) in bundle.plans:
            target, target_source = int(cand), "plan_metadata"
        else:
            target, target_source = plan_id, "self"
    if target not in bundle.plans:
        target, target_source = plan_id, "self"

    criteria = _criteria_of(bundle, target, cfg)
    matrix = build_matrix(debts, criteria, cfg, deferrals)
    verdict = seeding_check(plan_id, debts, matrix, cfg)
    meta = (bundle.plans.get(plan_id) or {}).get("metadata") or {}
    patched_meta = apply_autonomy_patch(meta, verdict)

    ledger = {
        "schema": LEDGER_SCHEMA,
        "plan_id": plan_id,
        "plan_name": (bundle.plans.get(plan_id) or {}).get("name"),
        "lineage": {
            "chain": chain,
            "own_debts": len(own),
            "inherited_debts": len(inherited),
            "deduped": len(debts),
        },
        "coverage_target_plan": target,
        "coverage_target_source": target_source,
        "coverage_target_criteria": len(criteria),
        "provenance": {
            "bundle_sha256": bundle.sha256,
            "bundle_schema": BUNDLE_SCHEMA,
            "fetched_utc": bundle.raw.get("fetched_utc"),
            "source": bundle.raw.get("source"),
            "input_counts": {
                "plans": len(bundle.plans),
                "nodes": {str(pid): len(v) for pid, v in bundle.nodes.items()},
                "reports": len(bundle.reports),
            },
            "config": cfg.as_dict(),
        },
        "debts": [d.as_dict() for d in debts],
        "coverage_matrix": matrix,
        "seed_gate": verdict.as_dict(),
        "plan_metadata_after": patched_meta,
        "summary": summarize(debts),
    }
    return ledger


def write_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canonical_json(ledger), encoding="utf-8")
    return p