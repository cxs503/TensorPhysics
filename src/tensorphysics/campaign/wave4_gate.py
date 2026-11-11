"""Wave-4 (IN40, plan 84) closure gate: whole-plan acceptance aggregation.

The wave-3 gate (:mod:`tensorphysics.campaign.wave3_gate`) answers "did each node
of one wave satisfy its own machine criteria, and may the capability be
inherited?".  A *plan-level* closure gate has to answer one question more, and it
is the one campaigns actually get wrong:

    may the plan be closed, and may the next wave be proposed from it?

That question is contractual, so it is decided by a number and not by prose::

    P = |{n : verdict(n) = ACCEPT}| / 10
    plan -> completed  iff  P >= 0.80  and  no critical failure

Critical failure is the closed set of three conditions in
:attr:`Wave4Node.is_critical_failure` (measured miss, repeated attempt collapse,
or a node the plan itself declared critical).  Everything else -- still running,
no submission, evidence withheld -- keeps the plan ``active``.  There is no code
path in this module that turns absence of evidence into a pass, because a gate
that rounds "not delivered" up to "delivered" destroys the value of every later
lineage claim.

Design notes
------------
* Node semantics (criteria, verdict roll-up, lineage derivation, evidence grades,
  error-row schema) are **imported** from :mod:`wave3_gate`, not duplicated: one
  source of truth for what ``ACCEPT`` and ``MATURE`` mean across waves.
* Collection and evaluation are separated (``scripts/run_in40_g40_wave4_gate.py``):
  the evaluation is a pure function of a committed JSON snapshot, so the closing
  report is reproducible and cannot drift with the live database.
* The report body is hashed (``report_digest``); the final verdict and the plan
  decision are re-derived from the node records by
  :func:`verify_report_integrity`, so honest FAIL is not overridable.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import Any, Iterable, Mapping, Sequence

from .wave3_gate import (
    CRITERION_HEADING,
    Criterion,
    GateCheck,
    GateError,
    GateIntegrityError,
    Grade,
    Lineage,
    NodeRecord,
    PeerReview,
    RootCause,
    RootCauseKind,
    Verdict,
    _error_row as _shared_error_row,  # one error-row schema for all wave gates
)

PLAN_ID = 84
"""Campaign plan id of plan 40 (``p39_wave2_aggregated``)."""

GATE_EXT_ID = "IN40-G40"
"""External id of this closure-gate node."""

GATE_EXT_ID_PREFIX = "IN40-N"
"""Prefix shared by the ten task nodes the gate must cover (IN40-N01..N10)."""

GATE_NODES = 10
"""Number of task nodes the wave-4 closure gate must cover."""

PASS_THRESHOLD = 0.80
"""Plan-level pass rate required to flip the plan to ``completed``."""

CRITICAL_CONSECUTIVE_FAILS = 2
"""Consecutive node failures that make a node a *critical* failure of the plan."""

_PARA_ID = re.compile(r"[（(](\d{1,2})[)）]")


# --------------------------------------------------------------------------- parsing
def _criterion_text(item: Any) -> str:
    """Deterministic text of one declared criterion (str or structured mapping).

    Structured criteria (``{"metric": ..., "threshold": ...}``) are rendered as
    ``key=value; ...`` in declaration order, so the text stays stable for hashing
    and remains readable by the judgeability classifier; no key is interpreted.
    """
    if isinstance(item, Mapping):
        parts: list[str] = []
        for key, value in item.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                value = "[" + ", ".join(str(v) for v in value) + "]"
            parts.append(f"{key}={value}")
        return " ".join("; ".join(parts).split())
    return " ".join(str(item).split())


def parse_acceptance_criteria(metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Machine criteria of one node as ``[(cid, text), ...]``.

    Three sources are tried in order, and the order is the contract:

    1. ``metadata["acceptance_criteria"]`` -- the campaign table's own list, the
       authoritative form for wave-4 nodes;
    2. ``C1 …; C2 …`` numbering inside the ``机器可算验收判据`` section of the goal
       (wave-3 convention, parsed by the shared :func:`parse_criteria`);
    3. ``(1) … (2) …`` parenthesised numbering in the same section -- the form
       wave-4 goals actually use, which the wave-3 parser cannot see because it
       only looks for ``C<k>``.

    Returning ``[]`` is a legitimate answer *and* a defect the gate reports: a
    node with no parseable criterion cannot be adjudicated, so it never counts as
    passed.
    """
    declared = metadata.get("acceptance_criteria") or []
    if isinstance(declared, (str, bytes)):
        # Some producers store the list JSON-encoded in a single string.  Iterating
        # that string would yield one bogus criterion per character (found by the
        # P106W3A2-N08 judgeability census: 145 nodes, ~28k fragment clauses), so the
        # payload is decoded first and only then enumerated.
        try:
            decoded = json.loads(declared)
        except (TypeError, ValueError):
            decoded = [declared]
        declared = decoded if isinstance(decoded, list) else [decoded]
    elif isinstance(declared, Mapping):
        declared = [declared]
    if declared:
        out: list[tuple[str, str]] = []
        for idx, text in enumerate(declared, start=1):
            clean = _criterion_text(text)
            if clean:
                out.append((f"C{idx}", clean))
        if out:
            return out

    from .wave3_gate import parse_criteria

    goal = str(metadata.get("goal") or "")
    parsed = parse_criteria(goal)
    if parsed:
        return parsed

    if CRITERION_HEADING in goal:
        body = goal.split(CRITERION_HEADING, 1)[1].lstrip("：: ")
        body = re.split(r"纪律[：:]", body, maxsplit=1)[0]
        body = body.split("【", 1)[0]  # cut trailing bracket blocks (pytest/artifacts)
        hits = list(_PARA_ID.finditer(body))
        if hits:
            found: list[tuple[str, str]] = []
            seen: set[str] = set()
            for idx, hit in enumerate(hits):
                cid = "C" + hit.group(1)
                if cid in seen:
                    continue
                seen.add(cid)
                end = hits[idx + 1].start() if idx + 1 < len(hits) else len(body)
                text = " ".join(body[hit.end():end].strip().strip("；;。，,").split())
                if text:
                    found.append((cid, text))
            if found:
                return found
    return []


def physics_goal(metadata: Mapping[str, Any]) -> str:
    """The ``【物理目标】`` statement of a node goal (physics, not process)."""
    goal = str(metadata.get("goal") or "")
    if "【物理目标】" not in goal:
        return " ".join(goal.split())[:200]
    body = goal.split("【物理目标】", 1)[1]
    for marker in ("【实现路径】", "【机器可算验收判据】"):
        body = body.split(marker, 1)[0]
    return " ".join(body.split())[:400]


# --------------------------------------------------------------------------- node
@dataclass(frozen=True)
class Wave4Node:
    """One wave-4 node: the shared :class:`NodeRecord` plus plan-level ledger state."""

    record: NodeRecord
    node_status: str = "unknown"
    reproduction_verdict: str | None = None
    consecutive_fails: int = 0
    report_count: int = 0
    declared_critical: bool = False
    goal: str = ""

    # -- passthrough of the shared node semantics --------------------------
    @property
    def node_id(self) -> int:
        return self.record.node_id

    @property
    def ext_id(self) -> str:
        return self.record.ext_id

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def repo(self) -> str:
        return self.record.repo

    @property
    def mission_id(self) -> int | None:
        return self.record.mission_id

    @property
    def mission_status(self) -> str:
        return self.record.mission_status

    @property
    def criteria(self) -> tuple[Criterion, ...]:
        return self.record.criteria

    @property
    def verdict(self) -> Verdict:
        return self.record.verdict

    @property
    def measured_criteria(self) -> int:
        return self.record.measured_criteria

    @property
    def lineage(self) -> Lineage:
        return self.record.lineage

    @property
    def root_cause(self) -> RootCause | None:
        return self.record.root_cause

    @property
    def rationale(self) -> str:
        return self.record.rationale

    # -- plan-level structure ---------------------------------------------
    @property
    def prerequisites_ok(self) -> bool:
        """Prerequisite verdict: node ``done`` *and* ``reproduction_verdict=pass``."""
        return self.node_status == "done" and (self.reproduction_verdict or "") == "pass"

    @property
    def is_critical_failure(self) -> bool:
        """Closed definition of a critical failure (blocks plan closure)."""
        if self.verdict is Verdict.FAIL:
            return True  # a criterion was measured and missed: physical/numerical defect
        if self.consecutive_fails >= CRITICAL_CONSECUTIVE_FAILS:
            return True  # repeated attempt collapse
        if self.declared_critical and self.verdict is not Verdict.ACCEPT:
            return True  # the plan said this node is critical and it is not accepted
        return False

    def as_dict(self) -> dict[str, Any]:
        body = self.record.as_dict()
        body.update(
            {
                "node_status": self.node_status,
                "reproduction_verdict": self.reproduction_verdict,
                "consecutive_fails": self.consecutive_fails,
                "report_count": self.report_count,
                "declared_critical": self.declared_critical,
                "prerequisites_ok": self.prerequisites_ok,
                "is_critical_failure": self.is_critical_failure,
                "physics_goal": self.goal,
            }
        )
        return body


def _default_root_cause(entry: Mapping[str, Any], status: str, report_count: int) -> RootCause:
    """Root cause for a node that delivered no acceptance evidence."""
    if status == "failed":
        return RootCause(
            kind=RootCauseKind.EVIDENCE_GAP,
            statement="the node mission terminated failed before publishing acceptance evidence",
            evidence=f"tensor_dev_missions.status=failed; linked reports={report_count}",
            confidence="high",
            actions=("re-run the node under a reduced scope and publish an acceptance JSON",),
        )
    return RootCause(
        kind=RootCauseKind.EVIDENCE_GAP,
        statement="node delivered no acceptance submission before the gate cutoff",
        evidence=(
            f"campaign_nodes.status={entry.get('status')!r}; "
            f"mission status={status!r}; linked reports={report_count}"
        ),
        confidence="high",
        actions=("let the node finish and publish its acceptance artifact, then re-run the gate",),
    )


def _default_rationale(record: NodeRecord, entry: Mapping[str, Any]) -> str:
    if record.verdict is Verdict.ACCEPT:
        return "all machine criteria measured and passed against external references"
    if str(entry.get("mission_status") or "") in ("failed", "stopped", "withdrawn"):
        return "attempt collapsed before acceptance; capability stays out of the next baseline"
    return "acceptance open at gate cutoff; capability is not inheritable yet"


def node_from_ledger(
    entry: Mapping[str, Any], *, submission: Mapping[str, Any] | None = None
) -> Wave4Node:
    """Build a :class:`Wave4Node` from one ``campaign_nodes ⊗ mission`` row.

    ``entry`` keys: ``node_id``, ``name``, ``status``, ``mission_id``,
    ``mission_status``, ``reproduction_verdict``, ``consecutive_fails``,
    ``report_count``, ``metadata``.  ``submission`` is the node's own acceptance
    JSON when one was delivered; an absent submission leaves every criterion
    ``WITHHELD`` -- the gate never upgrades silence into a pass.
    """
    meta = dict(entry.get("metadata") or {})
    ext_id = str(meta.get("ext_id") or f"node-{entry.get('node_id')}")
    parsed = parse_acceptance_criteria(meta)
    submitted = {str(k).upper(): v for k, v in ((submission or {}).get("criteria") or {}).items()}

    criteria: list[Criterion] = []
    for cid, text in parsed:
        item = submitted.get(cid.upper())
        if item is None:
            criteria.append(
                Criterion(
                    cid=cid,
                    text=text,
                    verdict=Verdict.WITHHELD,
                    evidence="no acceptance submission at gate cutoff",
                )
            )
            continue
        criteria.append(
            Criterion(
                cid=cid,
                text=text,
                verdict=Verdict(item["verdict"]),
                measured=item.get("measured"),
                reference=item.get("reference"),
                unit=item.get("unit", ""),
                evidence=str(item.get("evidence", "")),
                note=str(item.get("note", "")),
            )
        )

    status = str(entry.get("mission_status") or "unknown")
    report_count = int(entry.get("report_count") or 0)
    grade = Grade(str((submission or {}).get("evidence_grade", Grade.WITHHELD.value)))
    record = NodeRecord(
        node_id=int(entry["node_id"]),
        ext_id=ext_id,
        name=str(entry.get("name") or ext_id),
        repo=str(meta.get("repo") or "unknown"),
        mission_id=entry.get("mission_id"),
        mission_status=status,
        is_maturity="成熟化" in str(meta.get("goal") or ""),
        criteria=tuple(criteria),
        delivered_artifacts=tuple((submission or {}).get("artifacts") or ()),
        root_cause=None,
        rationale=str((submission or {}).get("rationale") or ""),
        evidence_grade=grade,
    )
    if record.verdict is not Verdict.ACCEPT and record.root_cause is None:
        record = replace(record, root_cause=_default_root_cause(entry, status, report_count))
    if not record.rationale:
        record = replace(record, rationale=_default_rationale(record, entry))
    rv = entry.get("reproduction_verdict")
    return Wave4Node(
        record=record,
        node_status=str(entry.get("status") or "unknown"),
        reproduction_verdict=None if rv in (None, "") else str(rv),
        consecutive_fails=int(entry.get("consecutive_fails") or 0),
        report_count=report_count,
        declared_critical=bool(meta.get("critical", False)),
        goal=physics_goal(meta),
    )


# --------------------------------------------------------------------------- plan
class PlanDecision(str, Enum):
    """Plan-level outcome of the closure gate."""

    COMPLETED = "completed"
    RETAINED_ACTIVE = "retained_active"


@dataclass(frozen=True)
class PlanAcceptance:
    """The ``>=80% and no critical failure`` rule, evaluated and explained."""

    total: int
    passed: tuple[str, ...] = ()
    open: tuple[str, ...] = ()
    critical: tuple[str, ...] = ()
    threshold: float = PASS_THRESHOLD

    @property
    def pass_count(self) -> int:
        return len(self.passed)

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.total == 0 else self.pass_count / self.total

    @property
    def decision(self) -> PlanDecision:
        if self.total == 0:
            return PlanDecision.RETAINED_ACTIVE
        if self.critical:
            return PlanDecision.RETAINED_ACTIVE
        return (
            PlanDecision.COMPLETED
            if self.pass_rate >= self.threshold
            else PlanDecision.RETAINED_ACTIVE
        )

    @property
    def completed(self) -> bool:
        return self.decision is PlanDecision.COMPLETED

    @property
    def next_wave_triggered(self) -> bool:
        return self.completed

    @property
    def reason(self) -> str:
        rate = f"{self.pass_count}/{self.total} = {self.pass_rate:.1%}"
        if self.total == 0:
            return "no nodes in the evidence set: the gate cannot decide, plan stays active"
        if self.critical:
            return (
                f"pass rate {rate} >= {self.threshold:.0%} is not decisive: critical failure(s) "
                f"in {', '.join(self.critical)} keep the plan active"
            )
        if self.completed:
            return (
                f"pass rate {rate} >= {self.threshold:.0%} and no critical failure: "
                "plan completed, next-wave proposal triggered"
            )
        return f"pass rate {rate} < {self.threshold:.0%}: plan stays active, nodes must iterate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total,
            "passed_nodes": list(self.passed),
            "open_nodes": list(self.open),
            "critical_failures": list(self.critical),
            "pass_count": self.pass_count,
            "pass_rate": round(self.pass_rate, 6),
            "threshold": self.threshold,
            "plan_decision": self.decision.value,
            "next_wave_triggered": self.next_wave_triggered,
            "reason": self.reason,
        }


def decide_plan(nodes: Sequence[Wave4Node], *, threshold: float = PASS_THRESHOLD) -> PlanAcceptance:
    """Apply the plan-completion rule to node verdicts (pure function)."""
    return PlanAcceptance(
        total=len(nodes),
        passed=tuple(n.ext_id for n in nodes if n.verdict is Verdict.ACCEPT),
        open=tuple(n.ext_id for n in nodes if n.verdict is not Verdict.ACCEPT),
        critical=tuple(n.ext_id for n in nodes if n.is_critical_failure),
        threshold=threshold,
    )


# --------------------------------------------------------------------------- guide
@dataclass(frozen=True)
class GuideGap:
    """One ``tensor_guide_tracker`` row reinterpreted as a capability gap."""

    metric_key: str
    metric_value: float
    target_value: float | None
    notes: str = ""

    @property
    def progress(self) -> float:
        if self.target_value in (None, 0):
            return 1.0 if self.metric_value >= (self.target_value or 0) else 0.0
        return max(0.0, min(1.0, self.metric_value / self.target_value))

    @property
    def gap(self) -> float:
        if self.target_value is None:
            return 0.0
        return max(0.0, self.target_value - self.metric_value)

    @property
    def unmet(self) -> bool:
        return self.progress < 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "metric_value": self.metric_value,
            "target_value": self.target_value,
            "progress": round(self.progress, 6),
            "gap": round(self.gap, 6),
            "unmet": self.unmet,
            "notes": self.notes,
        }


def guide_gaps_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[GuideGap, ...]:
    """Turn raw ``tensor_guide_tracker`` rows into typed gaps, worst-first."""
    gaps = [
        GuideGap(
            metric_key=str(r["metric_key"]),
            metric_value=float(r.get("metric_value") or 0.0),
            target_value=None if r.get("target_value") is None else float(r["target_value"]),
            notes=str(r.get("notes") or ""),
        )
        for r in rows
    ]
    return tuple(sorted(gaps, key=lambda g: (g.progress, g.metric_key)))


# --------------------------------------------------------------------------- report
@dataclass(frozen=True)
class Wave4Report:
    """Whole-plan closure-gate report: matrix, decision, lineage, gaps, audit chain."""

    plan_id: int
    plan_name: str
    gate_ext_id: str
    gate_mission_id: int
    cutoff: str
    nodes: tuple[Wave4Node, ...]
    guide_gaps: tuple[GuideGap, ...] = ()
    limitations: tuple[str, ...] = ()
    next_wave: tuple[str, ...] = ()
    audit_chain: tuple[dict[str, Any], ...] = ()
    peer_review: PeerReview | None = None
    figures: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    threshold: float = PASS_THRESHOLD
    publish_status: str = "not_published"
    plan_update_applied: bool = False

    # -- derived structure -------------------------------------------------
    @cached_property
    def acceptance(self) -> PlanAcceptance:
        return decide_plan(self.nodes, threshold=self.threshold)

    @property
    def pass_rate(self) -> float:
        return self.acceptance.pass_rate

    @property
    def matrix(self) -> list[dict[str, Any]]:
        """10 nodes x criteria acceptance matrix with the key error numbers."""
        return [
            {
                "ext_id": n.ext_id,
                "node_id": n.node_id,
                "repo": n.repo,
                "mission_id": n.mission_id,
                "node_status": n.node_status,
                "reproduction_verdict": n.reproduction_verdict,
                "prerequisites_ok": n.prerequisites_ok,
                "node_verdict": n.verdict.value,
                "criteria_total": len(n.criteria),
                "criteria_accepted": sum(1 for c in n.criteria if c.verdict is Verdict.ACCEPT),
                "criteria_measured": n.measured_criteria,
                "evidence_grade": n.record.evidence_grade.value,
                "lineage": n.lineage.value,
                "is_critical_failure": n.is_critical_failure,
                "rows": [c.as_dict() for c in n.criteria],
            }
            for n in self.nodes
        ]

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return len(self.nodes), sum(len(n.criteria) for n in self.nodes)

    @property
    def error_table(self) -> list[dict[str, Any]]:
        return [_shared_error_row(n.record, ()) for n in self.nodes]

    @property
    def lineage_table(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {k.value: [] for k in Lineage}
        for n in self.nodes:
            buckets[n.lineage.value].append(n.ext_id)
        return buckets

    @property
    def prerequisites(self) -> dict[str, Any]:
        done_pass = [n.ext_id for n in self.nodes if n.prerequisites_ok]
        return {
            "requirement": "campaign_nodes.status == done and reproduction_verdict == pass",
            "satisfied": done_pass,
            "count": f"{len(done_pass)}/{len(self.nodes)}",
            "all_satisfied": len(done_pass) == len(self.nodes) and bool(self.nodes),
        }

    @property
    def final_verdict(self) -> str:
        """Gate outcome: document defects outrank open nodes, open nodes outrank praise."""
        checks = self.checks
        document_checks = [c for c in checks if c.cid in ("G2", "G4", "G5", "G6")]
        if not all(c.passed for c in document_checks):
            return "BLOCKED_CRITERIA"
        if self.prerequisites["all_satisfied"] and self.acceptance.completed:
            return "ACCEPTED_PLAN_COMPLETED"
        return "NOT_ACCEPTED_NODES_OPEN"

    @cached_property
    def checks(self) -> list[GateCheck]:
        return self._evaluate_checks()

    def _evaluate_checks(self) -> list[GateCheck]:
        acc = self.acceptance
        prereq = self.prerequisites

        g1 = GateCheck(
            "G1",
            "every prerequisite node is status=done with reproduction_verdict=pass",
            bool(prereq["all_satisfied"]),
            str(prereq["count"]),
            "satisfied: " + (", ".join(prereq["satisfied"]) or "none"),
        )

        nodes_covered, criteria_total = self.matrix_shape
        covered = [n for n in self.nodes if n.criteria]
        g2 = GateCheck(
            "G2",
            f"acceptance matrix covers all {GATE_NODES} wave-4 nodes with >=1 criterion each",
            nodes_covered == GATE_NODES and len(covered) == GATE_NODES,
            f"{len(covered)}/{GATE_NODES} nodes, {criteria_total} criteria",
            "every cell carries an explicit verdict; withheld is a verdict, not a pass",
        )

        expected = (
            PlanDecision.COMPLETED
            if (acc.total > 0 and acc.pass_rate >= self.threshold and not acc.critical)
            else PlanDecision.RETAINED_ACTIVE
        )
        g3 = GateCheck(
            "G3",
            f"pass-rate rule applied: completed iff rate >= {self.threshold:.0%} and no critical failure",
            acc.decision is expected,
            f"{acc.pass_count}/{acc.total} = {acc.pass_rate:.1%}, critical={len(acc.critical)}",
            acc.reason,
        )

        open_nodes = [n for n in self.nodes if n.verdict is not Verdict.ACCEPT]
        unattributed = [
            n.ext_id
            for n in open_nodes
            if n.root_cause is None or not any(n.ext_id in line for line in self.limitations)
        ]
        g4 = GateCheck(
            "G4",
            "closing report carries a limitations section naming every open node and its cause",
            not unattributed and len(self.limitations) > 0,
            f"{len(open_nodes) - len(unattributed)}/{len(open_nodes)} open nodes attributed, "
            f"{len(self.limitations)} limitation entries",
            "unattributed: " + (", ".join(unattributed) if unattributed else "none"),
        )

        unmet = [g for g in self.guide_gaps if g.unmet]
        g5 = GateCheck(
            "G5",
            "guide_tracker gaps enumerated as next-wave proposal raw material",
            len(self.guide_gaps) > 0,
            f"{len(unmet)}/{len(self.guide_gaps)} metrics below target, {len(self.next_wave)} next-wave seeds",
            "each gap carries metric_key, value, target and progress",
        )

        pr = self.peer_review
        g6 = GateCheck(
            "G6",
            "report body digest computed and (when reviewed) signed by a non-participant",
            bool(pr is None or (pr.independent and pr.reviewed_digest == self.body_digest)),
            f"digest={self.body_digest[:12]}…, reviewer={'none' if pr is None else pr.reviewer_agent}",
            "peer review pending" if pr is None else f"verdict={pr.verdict}",
        )
        return [g1, g2, g3, g4, g5, g6]

    def core(self) -> dict[str, Any]:
        """Evidence-determined body (no self-checks, no signature)."""
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "gate_ext_id": self.gate_ext_id,
            "gate_mission_id": self.gate_mission_id,
            "cutoff": self.cutoff,
            "threshold": self.threshold,
            "nodes": [n.as_dict() for n in self.nodes],
            "matrix": self.matrix,
            "matrix_shape": {"nodes": self.matrix_shape[0], "criteria": self.matrix_shape[1]},
            "prerequisites": self.prerequisites,
            "plan_acceptance": self.acceptance.as_dict(),
            "error_table": self.error_table,
            "lineage": self.lineage_table,
            "guide_gaps": [g.as_dict() for g in self.guide_gaps],
            "limitations": list(self.limitations),
            "next_wave": list(self.next_wave),
        }

    @cached_property
    def body_digest(self) -> str:
        blob = json.dumps(self.core(), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_json(self, *, stamped: bool = True) -> dict[str, Any]:
        body = self.core()
        body["checks"] = [c.as_dict() for c in self.checks]
        body["final_verdict"] = self.final_verdict
        if stamped:
            body["audit_chain"] = list(self.audit_chain)
            body["report_digest"] = self.body_digest
            body["publish_status"] = self.publish_status
            body["plan_update_applied"] = self.plan_update_applied
            body["peer_review"] = None if self.peer_review is None else self.peer_review.as_dict()
            body["figures"] = [dict(x) for x in self.figures]
            body["artifacts"] = [dict(x) for x in self.artifacts]
            body["reproduce"] = [
                "python scripts/run_in40_g40_wave4_gate.py collect",
                "python scripts/run_in40_g40_wave4_gate.py evaluate",
                "python scripts/run_in40_g40_wave4_gate.py publish",
            ]
        return body

    def dumps(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=indent, default=str)

    def proposal_trigger(self) -> dict[str, Any]:
        """Machine-readable hand-off to the next proposal wave."""
        gaps = [g.metric_key for g in self.guide_gaps if g.unmet]
        if not self.acceptance.next_wave_triggered:
            return {
                "triggered": False,
                "reason": self.acceptance.reason,
                "blocked_by": list(self.acceptance.open),
                "carry_over_gaps": gaps,
                "seeds": [],
            }
        return {
            "triggered": True,
            "reason": self.acceptance.reason,
            "inherited_mature": self.lineage_table.get(Lineage.MATURE.value, []),
            "carry_over_gaps": gaps,
            "seeds": list(self.next_wave),
        }

    def render_markdown(self) -> str:
        from ._wave4_render import render_closing_report

        return render_closing_report(self)


# --------------------------------------------------------------------------- builders
def build_report(
    *,
    plan_id: int,
    plan_name: str,
    gate_ext_id: str,
    gate_mission_id: int,
    cutoff: str,
    nodes: Iterable[Wave4Node],
    guide_gaps: Sequence[GuideGap] = (),
    limitations: Sequence[str] = (),
    next_wave: Sequence[str] = (),
    audit_chain: Sequence[Mapping[str, Any]] = (),
    peer_review: PeerReview | None = None,
    figures: Sequence[Mapping[str, Any]] = (),
    artifacts: Sequence[Mapping[str, Any]] = (),
    threshold: float = PASS_THRESHOLD,
    publish_status: str = "not_published",
    plan_update_applied: bool = False,
) -> Wave4Report:
    """Assemble a report, enforcing the structural invariants up front."""
    node_tuple = tuple(nodes)
    if len(node_tuple) != GATE_NODES:
        raise GateError(f"wave-4 gate must cover exactly {GATE_NODES} nodes, got {len(node_tuple)}")
    if len({n.node_id for n in node_tuple}) != GATE_NODES:
        raise GateError("duplicate node id in gate evidence")
    for n in node_tuple:
        if not n.criteria:
            raise GateError(f"{n.ext_id}: no parseable machine criterion, node is unadjudicable")
        if n.verdict is not Verdict.ACCEPT and n.root_cause is None:
            raise GateError(f"{n.ext_id}: non-accepted node without root cause")
        if not n.rationale.strip():
            raise GateError(f"{n.ext_id}: lineage rationale missing")
    return Wave4Report(
        plan_id=plan_id,
        plan_name=plan_name,
        gate_ext_id=gate_ext_id,
        gate_mission_id=gate_mission_id,
        cutoff=cutoff,
        nodes=node_tuple,
        guide_gaps=tuple(guide_gaps),
        limitations=tuple(limitations),
        next_wave=tuple(next_wave),
        audit_chain=tuple(dict(a) for a in audit_chain),
        peer_review=peer_review,
        figures=tuple(dict(x) for x in figures),
        artifacts=tuple(dict(x) for x in artifacts),
        threshold=threshold,
        publish_status=publish_status,
        plan_update_applied=plan_update_applied,
    )


def verify_report_integrity(report: Wave4Report, recorded: Mapping[str, Any]) -> None:
    """Refuse a recorded outcome rosier than the evidence.

    Checks the three numbers a reader would trust: the gate verdict, the plan
    decision, and the pass rate.  ``recorded`` is the JSON actually written or
    published.  Honest FAIL is not overridable.
    """
    claimed_verdict = recorded.get("final_verdict", report.final_verdict)
    if claimed_verdict != report.final_verdict:
        raise GateIntegrityError(
            f"recorded final verdict {claimed_verdict!r} contradicts evidence-derived "
            f"{report.final_verdict!r}"
        )
    acc = recorded.get("plan_acceptance") or {}
    claimed_decision = acc.get("plan_decision", report.acceptance.decision.value)
    if claimed_decision != report.acceptance.decision.value:
        raise GateIntegrityError(
            f"recorded plan decision {claimed_decision!r} contradicts evidence-derived "
            f"{report.acceptance.decision.value!r}"
        )
    claimed_rate = acc.get("pass_rate")
    if claimed_rate is not None and abs(float(claimed_rate) - report.pass_rate) > 1e-9:
        raise GateIntegrityError(
            f"recorded pass rate {claimed_rate!r} contradicts evidence-derived "
            f"{report.pass_rate!r}"
        )


def plan_update_sql(report: Wave4Report) -> str:
    """The exact plan-status statement the decision implies (never auto-applied).

    The gate *computes* the update; applying it is a separate, deliberate act, so
    that a report can be audited before it mutates campaign state.  When the plan
    is not completed the statement is a no-op::

        UPDATE campaign_plans SET status = status WHERE id = 84
    """
    if report.acceptance.completed:
        return (
            "UPDATE campaign_plans SET status = 'completed', "
            f"updated_at = now() WHERE id = {report.plan_id} AND status <> 'completed'"
        )
    return f"UPDATE campaign_plans SET status = status WHERE id = {report.plan_id}"


def write_report(report: Wave4Report, *, json_path: str, md_path: str) -> dict[str, Any]:
    """Persist the machine report (JSON) and the paper-grade report (Markdown)."""
    payload = report.to_json()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.render_markdown())
    with open(json_path, encoding="utf-8") as fh:
        reread = json.load(fh)
    verify_report_integrity(report, reread)
    return payload