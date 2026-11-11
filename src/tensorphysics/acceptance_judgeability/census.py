"""Whole-corpus census: class distribution, KS distribution, hollow-PASS ledger.

Everything here is a pure function of a **frozen corpus file** (produced by
``scripts/harvest_acceptance_criteria.py``) plus the two policies, so the census
can be re-run offline and its digest compared -- the "recomputable" half of the
acceptance criterion. No number in the report is typed by hand: the report
generator reads this module's ``as_dict()``.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .classify import Judgeability, Subtype, classify_all
from .support import (
    DEFAULT_POLICY,
    UNIFORM_WEIGHTS,
    Adjudication,
    Evidence,
    KsRow,
    PassPolicy,
    TOLERANCE_WEIGHTS,
    WeightPolicy,
    adjudicate,
    gate_flag,
    node_ks,
    upgrade_paths,
)


# --------------------------------------------------------------------------- io
def load_corpus(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load one or more frozen JSONL corpus files, de-duplicated by ``node_id``."""
    nodes: dict[int, dict[str, Any]] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                nodes[int(row["node_id"])] = row
    return sorted(nodes.values(), key=lambda r: int(r["node_id"]))


def corpus_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ks_table_digest(rows: Sequence[KsRow]) -> str:
    """Stable digest of the (node_id, n_clauses, ks) table: the recomputability check."""
    payload = [
        [r.node_id, r.n_clauses, None if r.ks is None else round(r.ks, 9), r.status]
        for r in sorted(rows, key=lambda r: r.node_id)
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- result
@dataclass(frozen=True)
class CensusResult:
    """Aggregate census + per-node scorecard + hollow-PASS ledger."""

    rows: tuple[KsRow, ...]
    policy: PassPolicy
    weights: WeightPolicy
    corpus_paths: tuple[str, ...] = ()
    corpus_sha256: str = ""
    n_nodes: int = 0
    n_nodes_with_criteria: int = 0
    n_criteria: int = 0
    class_counts: Mapping[str, int] = field(default_factory=dict)
    subtype_counts: Mapping[str, int] = field(default_factory=dict)
    class_weight_share: Mapping[str, float] = field(default_factory=dict)
    hollow: tuple[KsRow, ...] = ()
    upgrade_ledger: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()

    # -- derived statistics -------------------------------------------------
    @property
    def ks_table_sha256(self) -> str:
        return ks_table_digest(self.rows)

    @property
    def done_rows(self) -> list[KsRow]:
        return [r for r in self.rows if r.status.lower() == "done"]

    @property
    def ks_defined(self) -> list[float]:
        return [r.ks for r in self.rows if r.ks is not None]

    @property
    def judgeable_share(self) -> float | None:
        """Share of *clauses* (uniform weights) that are judgeable."""
        if not self.n_criteria:
            return None
        return self.class_counts.get(Judgeability.JUDGEABLE.value, 0) / self.n_criteria

    @property
    def hollow_ratio(self) -> float | None:
        if not self.done_rows:
            return None
        return len(self.hollow) / len(self.done_rows)

    def severity_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows:
            out[row.severity] = out.get(row.severity, 0) + 1
        return out

    def by_repo(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            key = row.repo or "(unspecified)"
            rec = out.setdefault(key, {"nodes": 0, "criteria": 0, "hollow": 0, "ks": []})
            rec["nodes"] += 1
            rec["criteria"] += row.n_clauses
            rec["hollow"] += int(row.hollow_pass)
            if row.ks is not None:
                rec["ks"].append(row.ks)
        for rec in out.values():
            ks = rec["ks"]
            rec["ks_median"] = round(statistics.median(ks), 4) if ks else None
            rec["hollow_ratio"] = round(rec["hollow"] / rec["nodes"], 4) if rec["nodes"] else None
            rec.pop("ks")
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["nodes"]))

    def ks_histogram(self, bins: int = 10) -> list[dict[str, Any]]:
        edges = [i / bins for i in range(bins + 1)]
        counts = [0] * bins
        for value in self.ks_defined:
            idx = min(bins - 1, max(0, int(value * bins)))
            counts[idx] += 1
        undefined = sum(1 for r in self.rows if r.ks is None)
        return [
            {"lo": edges[i], "hi": edges[i + 1], "count": counts[i]} for i in range(bins)
        ] + [{"lo": None, "hi": None, "count": undefined, "label": "KS undefined (no clause)"}]

    def as_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        ks = self.ks_defined
        payload: dict[str, Any] = {
            "policy": self.policy.as_dict(),
            "weights": self.weights.as_dict(),
            "corpus": {
                "paths": list(self.corpus_paths),
                "sha256": self.corpus_sha256,
                "n_nodes": self.n_nodes,
                "n_nodes_with_criteria": self.n_nodes_with_criteria,
                "n_criteria": self.n_criteria,
                "extraction_rate": (
                    self.n_nodes_with_criteria / self.n_nodes if self.n_nodes else None
                ),
            },
            "classes": {
                "counts": dict(self.class_counts),
                "share": {
                    k: (v / self.n_criteria if self.n_criteria else None)
                    for k, v in self.class_counts.items()
                },
                "subtypes": dict(self.subtype_counts),
                "weight_share": dict(self.class_weight_share),
            },
            "ks": {
                "defined_nodes": len(ks),
                "undefined_nodes": len(self.rows) - len(ks),
                "mean": round(statistics.fmean(ks), 4) if ks else None,
                "median": round(statistics.median(ks), 4) if ks else None,
                "p10": round(statistics.quantiles(ks, n=10)[0], 4) if len(ks) > 10 else None,
                "min": round(min(ks), 4) if ks else None,
                "max": round(max(ks), 4) if ks else None,
                "histogram": self.ks_histogram(),
                "table_sha256": self.ks_table_sha256,
                "judgeable_clause_share": (
                    None if self.judgeable_share is None else round(self.judgeable_share, 4)
                ),
            },
            "pass_dependency": {
                "done_nodes": len(self.done_rows),
                "hollow_pass_nodes": len(self.hollow),
                "hollow_ratio": None if self.hollow_ratio is None else round(self.hollow_ratio, 4),
                "severity": self.severity_counts(),
                "by_repo": self.by_repo(),
            },
            "upgrade_ledger_size": len(self.upgrade_ledger),
            "notes": list(self.notes),
        }
        if include_rows:
            payload["rows"] = [r.as_dict() for r in self.rows]
            payload["hollow_ledger"] = [r.as_dict() for r in self.hollow]
            payload["upgrade_ledger"] = list(self.upgrade_ledger)
        return payload


# --------------------------------------------------------------------------- census
def _count_subtypes(classified: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for clause in classified:
        out[clause.subtype.value] = out.get(clause.subtype.value, 0) + 1
    return out


def run_census(
    nodes: Sequence[Mapping[str, Any]],
    *,
    policy: PassPolicy = DEFAULT_POLICY,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
    corpus_paths: Sequence[str] = (),
    corpus_sha256: str = "",
    upgrade_limit: int | None = None,
) -> CensusResult:
    """Score every node, then assemble the aggregate view and the hollow ledger."""
    rows: list[KsRow] = []
    class_counts = {k.value: 0 for k in Judgeability}
    subtype_counts: dict[str, int] = {}
    weight_total_all = 0.0
    weight_by_class: dict[str, float] = {k.value: 0.0 for k in Judgeability}
    n_criteria = 0

    for node in nodes:
        row = node_ks(node, weights=weights, policy=policy)
        rows.append(row)
        classified = classify_all(node.get("criteria") or [])
        n_criteria += len(classified)
        for clause in classified:
            class_counts[clause.judgeability.value] += 1
            w = weights.weight(clause)
            weight_total_all += w
            weight_by_class[clause.judgeability.value] += w
        for key, value in _count_subtypes(classified).items():
            subtype_counts[key] = subtype_counts.get(key, 0) + value

    rows.sort(key=lambda r: r.node_id)
    hollow = sorted((r for r in rows if r.hollow_pass), key=lambda r: ((r.ks if r.ks is not None else -1.0), r.node_id))

    ledger: list[dict[str, Any]] = []
    by_id = {int(n["node_id"]): n for n in nodes}
    for idx, row in enumerate(hollow):
        if upgrade_limit is not None and idx >= upgrade_limit:
            break
        node = by_id[row.node_id]
        adj = adjudicate(node.get("criteria") or [], row.evidence, policy=policy, weights=weights)
        ledger.append(
            {
                "node_id": row.node_id,
                "plan_id": row.plan_id,
                "ext_id": row.ext_id,
                "repo": row.repo,
                "ks": row.ks,
                "severity": row.severity,
                "counts": dict(row.counts),
                "adjudication": adj.as_dict(),
                "upgrades": upgrade_paths(node.get("criteria") or []),
            }
        )

    notes = (
        "KS=weight(JUDGEABLE & measured)/weight(all clauses); None means no clause at all",
        "evidence proxy: node.status=done and consecutive_fails=0 (validate with require_report mode)",
        "RECORD_ONLY and WITHHELD_OK never enter the numerator by construction",
    )
    return CensusResult(
        rows=tuple(rows),
        policy=policy,
        weights=weights,
        corpus_paths=tuple(str(p) for p in corpus_paths),
        corpus_sha256=corpus_sha256,
        n_nodes=len(rows),
        n_nodes_with_criteria=sum(1 for r in rows if r.n_clauses),
        n_criteria=n_criteria,
        class_counts=class_counts,
        subtype_counts=subtype_counts,
        class_weight_share={
            k: (round(v / weight_total_all, 6) if weight_total_all else 0.0)
            for k, v in weight_by_class.items()
        },
        hollow=tuple(hollow),
        upgrade_ledger=tuple(ledger),
        notes=notes,
    )


def adjudicate_node(node: Mapping[str, Any], *, policy: PassPolicy = DEFAULT_POLICY) -> Adjudication:
    """Convenience: adjudicate one corpus node row (used by the gate adapter/tests)."""
    evidence = Evidence.from_node(node)
    return adjudicate(node.get("criteria") or [], evidence, policy=policy)