"""Wave-3 (IN19) closure gate: acceptance aggregation and lineage inheritance.

A campaign *gate* has to answer three questions with machine-checkable evidence:

1.  **Acceptance** -- for every planned node, did it satisfy its own declared
    machine criteria?  A node that submitted nothing is *not* a pass: it is
    ``WITHHELD`` (evidence insufficient).  Substituting an optimistic verdict for
    missing evidence is the single most expensive failure mode of a research
    campaign, so this module has no code path that can produce ``ACCEPT`` from a
    node whose criteria were never evaluated.
2.  **Root cause** -- for every node that is not ``ACCEPT``, which of the four
    admissible defect classes explains the gap: model-form error, input
    uncertainty, implementation bug, or an evidence gap (nothing was delivered)?
    The taxonomy is closed on purpose; ``"needs more work"`` is not a class.
3.  **Lineage inheritance** -- may the node's capability enter the next campaign
    baseline (``MATURE``), must it be iterated (``ITERATE``), or should it be
    shelved (``SHELVE``)?  Inheritance is *derived*, never asserted: a node can
    only inherit when its verdict is ``ACCEPT`` **and** at least one criterion was
    measured against an external reference.

The gate report additionally carries the five campaign criteria (C1 matrix
coverage, C2 root-cause completeness, C3 lineage explicitness, C4 error table,
C5 peer review) as first-class, numerically evaluated checks, plus an error table
that lists *every* node's key quantity against its baseline -- with an explicit
``WITHHELD`` row rather than an omitted row when no measurement exists.

Determinism: the report body is hashed (``report_digest``); a peer reviewer signs
the *digest* of the artifact they actually read, so a reviewed report can never
silently become a different one.  :func:`verify_verdict_integrity` recomputes the
final verdict from the node records and refuses a recorded verdict that is
rosier than the recomputation (honest FAIL is not overridable).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from functools import cached_property
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

GATE_NODES = 9
"""Number of task nodes the wave-3 closure gate must cover (IN19-T1..T9)."""


class GateError(ValueError):
    """The evidence set or a request to the gate is inconsistent."""


class GateIntegrityError(GateError):
    """A recorded verdict contradicts the evidence (attempted override)."""


class Verdict(str, Enum):
    """Acceptance verdict of one criterion or one node."""

    ACCEPT = "accept"
    PARTIAL = "partial"
    FAIL = "fail"
    WITHHELD = "withheld"

    @property
    def decisive(self) -> bool:
        return self in (Verdict.ACCEPT, Verdict.FAIL)


class Grade(str, Enum):
    """Platform evidence grading of the underlying capability."""

    REAL = "REAL"
    MOCK = "MOCK"
    STUB = "STUB"
    WITHHELD = "WITHHELD"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class RootCauseKind(str, Enum):
    """Closed taxonomy of why a node did not pass."""

    MODEL_FORM = "model_form_error"
    INPUT_UNCERTAINTY = "input_uncertainty"
    IMPLEMENTATION_BUG = "implementation_bug"
    EVIDENCE_GAP = "evidence_gap"


class Lineage(str, Enum):
    """Inheritance decision for the next campaign baseline."""

    MATURE = "mature"
    ITERATE = "iterate"
    SHELVE = "shelve"


CRITERION_HEADING = "机器可算验收判据"
_CRIT_ID = re.compile(r"(?<![A-Za-z0-9_])C(\d{1,2})(?![0-9.])")


def parse_criteria(goal: str) -> list[tuple[str, str]]:
    """Extract ``[(cid, text), ...]`` from a mission goal.

    The campaign convention writes machine criteria after a ``机器可算验收判据``
    heading as ``C1 ...；C2 ...`` -- sometimes with a colon after the id, often
    without.  Parsing them (instead of hand-copying) is what makes matrix coverage
    (criterion C1) a *machine* check.
    """
    if not goal or CRITERION_HEADING not in goal:
        return []
    body = goal.split(CRITERION_HEADING, 1)[1].lstrip("：: ")
    body = re.split(r"纪律[：:]", body, maxsplit=1)[0]
    hits = list(_CRIT_ID.finditer(body))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for idx, hit in enumerate(hits):
        cid = "C" + hit.group(1)
        if cid in seen:
            continue
        seen.add(cid)
        end = hits[idx + 1].start() if idx + 1 < len(hits) else len(body)
        text = body[hit.end():end].lstrip("：:、 ")
        text = " ".join(text.strip().strip("；;。，,").split())
        if text:
            out.append((cid, text))
    return out


@dataclass(frozen=True)
class Criterion:
    """One machine criterion of one node, evaluated against evidence."""

    cid: str
    text: str
    verdict: Verdict = Verdict.WITHHELD
    measured: float | None = None
    reference: float | None = None
    unit: str = ""
    evidence: str = ""
    note: str = ""

    @property
    def rel_error_pct(self) -> float | None:
        if self.measured is None or self.reference in (None, 0):
            return None
        return 100.0 * (self.measured - self.reference) / self.reference

    def as_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "text": self.text,
            "verdict": self.verdict.value,
            "measured": self.measured,
            "reference": self.reference,
            "unit": self.unit,
            "rel_error_pct": None if self.rel_error_pct is None else round(self.rel_error_pct, 4),
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass(frozen=True)
class RootCause:
    """Why a node did not pass, in the closed four-class taxonomy."""

    kind: RootCauseKind
    statement: str
    evidence: str
    confidence: str = "medium"  # low | medium | high
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.statement.strip() or not self.evidence.strip():
            raise GateError("a root cause needs both a statement and evidence")
        if self.confidence not in ("low", "medium", "high"):
            raise GateError(f"unknown confidence {self.confidence!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "statement": self.statement,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class GateProbe:
    """An independent measurement of *current* platform state taken by the gate.

    Probes never count as a node deliverable; they exist so that the error table
    has a measured number instead of a blank when a node under test is reachable,
    and so that a claimed maturity can be falsified from outside the node.
    """

    probe_id: str
    node_id: int
    endpoint: str
    quantity: str
    measured: float
    reference: float
    unit: str
    reference_source: str
    reference_tolerance_pct: float
    threshold_pct: float
    note: str = ""

    @property
    def rel_error_pct(self) -> float:
        return 100.0 * (self.measured - self.reference) / self.reference

    @property
    def within_threshold(self) -> bool:
        return abs(self.rel_error_pct) <= self.threshold_pct

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "quantity": self.quantity,
            "measured": self.measured,
            "reference": self.reference,
            "unit": self.unit,
            "rel_error_pct": round(self.rel_error_pct, 4),
            "threshold_pct": self.threshold_pct,
            "within_threshold": self.within_threshold,
            "reference_source": self.reference_source,
            "reference_tolerance_pct": self.reference_tolerance_pct,
            "note": self.note,
        }


@dataclass(frozen=True)
class NodeRecord:
    """All gate-visible evidence about one campaign node."""

    node_id: int
    ext_id: str
    name: str
    repo: str
    mission_id: int | None
    mission_status: str
    is_maturity: bool
    criteria: tuple[Criterion, ...] = ()
    delivered_artifacts: tuple[str, ...] = ()
    root_cause: RootCause | None = None
    rationale: str = ""
    evidence_grade: Grade = Grade.WITHHELD
    #: Optional judgeability flag (P106W3A2-N08).  ``None`` keeps the historical
    #: behaviour.  When supplied by
    #: :func:`tensorphysics.acceptance_judgeability.gate.judgeability_flag` it is a
    #: plain dict with ``supports_pass``; an ``ACCEPT`` whose criteria cannot carry
    #: a PASS is then downgraded to ``WITHHELD`` instead of being reported.
    judgeability: dict | None = None

    @property
    def verdict(self) -> Verdict:
        """Roll-up of the node's criteria, guarded by the judgeability flag.

        * every criterion ``ACCEPT`` -> ``ACCEPT``
        * any criterion ``FAIL``    -> ``FAIL`` (a measured miss is decisive)
        * nothing evaluated        -> ``WITHHELD``
        * otherwise                -> ``PARTIAL``

        When a ``judgeability`` flag is attached and reports ``supports_pass=False``
        (a hollow PASS: no judgeable clause, or KS below policy), an ``ACCEPT`` is
        downgraded to ``WITHHELD`` -- "the report exists" is not a pass condition.
        A ``FAIL`` is never softened here: honest FAIL is not overridable.
        """
        rolled = self._verdict_rollup
        if rolled is Verdict.ACCEPT and self.judgeability is not None:
            if not bool(self.judgeability.get("supports_pass", True)):
                return Verdict.WITHHELD
        return rolled

    @property
    def _verdict_rollup(self) -> Verdict:
        if not self.criteria:
            return Verdict.WITHHELD
        verdicts = [c.verdict for c in self.criteria]
        if all(v is Verdict.ACCEPT for v in verdicts):
            return Verdict.ACCEPT
        if any(v is Verdict.FAIL for v in verdicts):
            return Verdict.FAIL
        if all(v is Verdict.WITHHELD for v in verdicts):
            return Verdict.WITHHELD
        return Verdict.PARTIAL

    @property
    def judgeability_supported(self) -> bool | None:
        """``None`` when no judgeability flag was attached (historical record)."""
        if self.judgeability is None:
            return None
        return bool(self.judgeability.get("supports_pass", True))

    @property
    def measured_criteria(self) -> int:
        return sum(1 for c in self.criteria if c.measured is not None)

    @property
    def lineage(self) -> Lineage:
        return decide_lineage(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ext_id": self.ext_id,
            "name": self.name,
            "repo": self.repo,
            "mission_id": self.mission_id,
            "mission_status": self.mission_status,
            "is_maturity": self.is_maturity,
            "verdict": self.verdict.value,
            "judgeability": self.judgeability,
            "criteria_total": len(self.criteria),
            "criteria_measured": self.measured_criteria,
            "evidence_grade": self.evidence_grade.value,
            "criteria": [c.as_dict() for c in self.criteria],
            "root_cause": None if self.root_cause is None else self.root_cause.as_dict(),
            "lineage": self.lineage.value,
            "rationale": self.rationale,
            "delivered_artifacts": list(self.delivered_artifacts),
        }


def decide_lineage(record: NodeRecord) -> Lineage:
    """Derive the inheritance decision from the node record (never asserted).

    ``MATURE`` needs an ``ACCEPT`` verdict *and* a criterion measured against an
    external reference: an acceptance that only checked internal consistency is
    not inheritable physics.  ``SHELVE`` is reserved for nodes that are both
    unmeasured and explicitly abandoned; a node still in flight is ``ITERATE``.
    """
    verdict = record.verdict
    if verdict is Verdict.ACCEPT:
        if record.measured_criteria == 0:
            raise GateError(
                f"{record.ext_id}: ACCEPT without any measured criterion cannot be inherited"
            )
        return Lineage.MATURE
    if record.evidence_grade is Grade.NOT_SUPPORTED and record.mission_status in (
        "failed",
        "stopped",
        "withdrawn",
    ):
        return Lineage.SHELVE
    return Lineage.ITERATE


@dataclass(frozen=True)
class PeerReview:
    """An independent reviewer's signed assessment of the gate report."""

    reviewer_agent: str
    reviewer_affiliation: str
    independent: bool
    reviewed_digest: str
    verdict: str  # accept | accept_with_minor | reject
    findings: tuple[str, ...] = ()
    requested_changes: tuple[str, ...] = ()
    signed_at: str = ""
    response: str = ""  # author's response, when the review is recorded as prior

    ACCEPTING = ("accept", "accept_with_minor")

    def __post_init__(self) -> None:
        if self.verdict not in ("accept", "accept_with_minor", "reject"):
            raise GateError(f"unknown peer-review verdict {self.verdict!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "reviewer_agent": self.reviewer_agent,
            "reviewer_affiliation": self.reviewer_affiliation,
            "independent": self.independent,
            "reviewed_digest": self.reviewed_digest,
            "verdict": self.verdict,
            "findings": list(self.findings),
            "requested_changes": list(self.requested_changes),
            "signed_at": self.signed_at,
            "response": self.response,
        }


@dataclass(frozen=True)
class GateCheck:
    """One of the campaign's own machine criteria (C1..C5)."""

    cid: str
    requirement: str
    passed: bool
    number: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "requirement": self.requirement,
            "passed": self.passed,
            "number": self.number,
            "detail": self.detail,
        }


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _error_row(record: NodeRecord, probes: Sequence[GateProbe]) -> dict[str, Any]:
    """One row of the error judgement table (always present, never omitted)."""
    node_probes = [p for p in probes if p.node_id == record.node_id]
    measured = [c for c in record.criteria if c.measured is not None]
    key_quantity = ""
    reference_source = ""
    threshold_pct: float | None = None
    attained: float | None = None
    reference: float | None = None
    rel_error: float | None = None
    basis = ""
    if node_probes:
        p = max(node_probes, key=lambda q: abs(q.rel_error_pct))
        key_quantity, reference_source = p.quantity, p.reference_source
        threshold_pct, attained, reference = p.threshold_pct, p.measured, p.reference
        rel_error, basis = p.rel_error_pct, "gate_probe"
    elif measured:
        c = measured[0]
        key_quantity, reference_source = c.text[:60], c.evidence or "declared reference"
        attained, reference, rel_error, basis = c.measured, c.reference, c.rel_error_pct, "node_criterion"
    else:
        key_quantity = record.criteria[0].text[:60] if record.criteria else "not declared"
        reference_source = "declared in node goal"
        basis = "withheld"
    return {
        "ext_id": record.ext_id,
        "node_id": record.node_id,
        "key_quantity": key_quantity,
        "baseline_source": reference_source,
        "threshold_pct": threshold_pct,
        "attained": attained,
        "baseline": reference,
        "rel_error_pct": None if rel_error is None else round(rel_error, 3),
        "within_threshold": None if rel_error is None else abs(rel_error) <= (threshold_pct or math.inf),
        "basis": basis,
        "verdict": record.verdict.value,
        "withheld_reason": (
            "" if basis != "withheld" else "no acceptance evidence delivered at gate cutoff"
        ),
    }


@dataclass(frozen=True)
class GateReport:
    """The complete, hashable wave-3 closure-gate report."""

    plan_id: int
    plan_name: str
    gate_ext_id: str
    gate_mission_id: int
    cutoff: str
    nodes: tuple[NodeRecord, ...]
    probes: tuple[GateProbe, ...] = ()
    peer_review: PeerReview | None = None
    prior_reviews: tuple[PeerReview, ...] = ()
    limitations: tuple[str, ...] = ()
    next_wave: tuple[str, ...] = ()
    audit_chain: tuple[dict[str, Any], ...] = ()

    # -- derived structure -------------------------------------------------
    @cached_property
    def error_table(self) -> list[dict[str, Any]]:
        return [_error_row(n, self.probes) for n in self.nodes]

    @property
    def matrix(self) -> list[dict[str, Any]]:
        return [
            {
                "ext_id": n.ext_id,
                "node_id": n.node_id,
                "node_verdict": n.verdict.value,
                "rows": [c.as_dict() for c in n.criteria],
            }
            for n in self.nodes
        ]

    @property
    def lineage_table(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {k.value: [] for k in Lineage}
        for n in self.nodes:
            buckets[n.lineage.value].append(n.ext_id)
        return buckets

    @property
    def final_verdict(self) -> str:
        """Gate outcome; ``ACCEPTED`` demands passing checks *and* all nodes."""
        checks = self.checks
        if all(c.passed for c in checks) and all(n.verdict is Verdict.ACCEPT for n in self.nodes):
            return "ACCEPTED"
        if all(c.passed for c in checks):
            return "NOT_ACCEPTED_NODES_OPEN"
        return "BLOCKED_CRITERIA"

    @cached_property
    def checks(self) -> list[GateCheck]:
        return self._evaluate_checks()

    def _evaluate_checks(self) -> list[GateCheck]:
        nodes = self.nodes
        covered = [n for n in nodes if n.criteria]
        c1 = GateCheck(
            "C1",
            "acceptance matrix covers all 9 wave-3 nodes",
            len(nodes) == GATE_NODES and len(covered) == GATE_NODES,
            f"{len(covered)}/{GATE_NODES}",
            "every node carries at least one parsed machine criterion with explicit verdict",
        )

        non_accept = [n for n in nodes if n.verdict is not Verdict.ACCEPT]
        with_cause = [n for n in non_accept if n.root_cause is not None]
        c2 = GateCheck(
            "C2",
            "every non-accepted node has a root-cause analysis in the closed taxonomy",
            len(with_cause) == len(non_accept),
            f"{len(with_cause)}/{len(non_accept)}",
            "classes: " + ", ".join(k.value for k in RootCauseKind),
        )

        explicit = [n for n in nodes if isinstance(n.lineage, Lineage) and n.rationale.strip()]
        mature_violations = [
            n.ext_id for n in nodes if n.lineage is Lineage.MATURE and n.verdict is not Verdict.ACCEPT
        ]
        c3 = GateCheck(
            "C3",
            "lineage inheritance explicit for every node and derived (never asserted)",
            len(explicit) == len(nodes) and not mature_violations,
            f"{len(explicit)}/{len(nodes)}",
            "MATURE/ITERATE/SHELVE assigned per node; no MATURE without ACCEPT",
        )

        rows = self.error_table
        resolved = [r for r in rows if r["basis"] != "withheld"]
        c4 = GateCheck(
            "C4",
            "summary report carries an itemised error judgement table (key quantity vs baseline)",
            len(rows) == len(nodes) and len(nodes) > 0,
            f"{len(rows)}/{len(nodes)} rows ({len(resolved)} measured, {len(rows) - len(resolved)} withheld)",
            "withheld rows are stated, not omitted",
        )

        body_digest = self.body_digest
        pr = self.peer_review
        c5 = GateCheck(
            "C5",
            "report passed peer review by at least one non-participant",
            bool(
                pr
                and pr.independent
                and pr.verdict in PeerReview.ACCEPTING
                and pr.reviewed_digest == body_digest
            ),
            "1/1 reviewer" if pr and pr.independent and pr.reviewed_digest == body_digest else "0/1 reviewer",
            "independent reviewer must sign the digest of the body reviewed"
            if not pr
            else f"reviewer={pr.reviewer_agent}, verdict={pr.verdict}, digest_match={pr.reviewed_digest == body_digest}",
        )
        return [c1, c2, c3, c4, c5]

    # -- determinism -------------------------------------------------------
    @cached_property
    def body_digest(self) -> str:
        """Digest of everything a reviewer is being asked to sign.

        Deliberately excludes the self-checks and the peer-review block: C5 checks
        the reviewer's signature against this digest, so the digest must not depend
        on them (otherwise the check would be circular).
        """
        return _digest(self.core())

    def core(self) -> dict[str, Any]:
        """Evidence-determined body of the report (no self-checks, no signature)."""
        return {
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "gate_ext_id": self.gate_ext_id,
            "gate_mission_id": self.gate_mission_id,
            "cutoff": self.cutoff,
            "nodes": [n.as_dict() for n in self.nodes],
            "probes": [p.as_dict() for p in self.probes],
            "error_table": self.error_table,
            "lineage": self.lineage_table,
            "limitations": list(self.limitations),
            "next_wave": list(self.next_wave),
        }

    def to_json(self, *, stamped: bool = True) -> dict[str, Any]:
        body = self.core()
        body["checks"] = [c.as_dict() for c in self.checks]
        body["final_verdict"] = self.final_verdict
        if stamped:
            body["audit_chain"] = list(self.audit_chain)
            body["report_digest"] = self.body_digest
            body["peer_review"] = None if self.peer_review is None else self.peer_review.as_dict()
            body["prior_reviews"] = [r.as_dict() for r in self.prior_reviews]
        return body

    def dumps(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=indent, default=str)

    # -- rendering ---------------------------------------------------------
    def render_markdown(self) -> str:
        from ._wave3_render import render_report

        return render_report(self)


def verify_verdict_integrity(report: GateReport, recorded: Mapping[str, Any]) -> None:
    """Refuse a recorded verdict that is more optimistic than the evidence.

    ``recorded`` is the JSON actually written to disk / published.  If it claims
    ``ACCEPTED`` while the node records recompute to something weaker, the gate
    raises :class:`GateIntegrityError`: honest FAIL is not overridable.
    """
    recomputed = report.final_verdict
    claimed = recorded.get("final_verdict", recomputed)
    if claimed == recomputed:
        return
    raise GateIntegrityError(
        f"recorded final verdict {claimed!r} contradicts evidence-derived {recomputed!r}"
    )


def build_report(
    *,
    plan_id: int,
    plan_name: str,
    gate_ext_id: str,
    gate_mission_id: int,
    cutoff: str,
    nodes: Iterable[NodeRecord],
    probes: Sequence[GateProbe] = (),
    peer_review: PeerReview | None = None,
    prior_reviews: Sequence[PeerReview] = (),
    limitations: Sequence[str] = (),
    next_wave: Sequence[str] = (),
    audit_chain: Sequence[Mapping[str, Any]] = (),
) -> GateReport:
    """Assemble a report, enforcing the structural invariants up front."""
    node_tuple = tuple(nodes)
    if len(node_tuple) != GATE_NODES:
        raise GateError(f"gate must cover exactly {GATE_NODES} nodes, got {len(node_tuple)}")
    if len({n.node_id for n in node_tuple}) != GATE_NODES:
        raise GateError("duplicate node id in gate evidence")
    for n in node_tuple:
        if n.verdict is not Verdict.ACCEPT and n.root_cause is None:
            raise GateError(f"{n.ext_id}: non-accepted node without root cause")
        if not n.rationale.strip():
            raise GateError(f"{n.ext_id}: lineage rationale missing")
    return GateReport(
        plan_id=plan_id,
        plan_name=plan_name,
        gate_ext_id=gate_ext_id,
        gate_mission_id=gate_mission_id,
        cutoff=cutoff,
        nodes=node_tuple,
        probes=tuple(probes),
        peer_review=peer_review,
        prior_reviews=tuple(prior_reviews),
        limitations=tuple(limitations),
        next_wave=tuple(next_wave),
        audit_chain=tuple(dict(a) for a in audit_chain),
    )


def node_record_from_ledger(entry: Mapping[str, Any], *, probes: Sequence[GateProbe] = ()) -> NodeRecord:
    """Build a :class:`NodeRecord` from the campaign-node ledger dump.

    ``entry`` is one row of ``campaign_nodes`` joined with its mission
    (``node_id``, ``name``, ``mission_id``, ``mission_status``, ``metadata``).
    A node with no acceptance submission gets ``WITHHELD`` criteria: the gate
    never upgrades absence of evidence into a pass.
    """
    meta = entry.get("metadata") or {}
    ext_id = str(meta.get("ext_id") or f"node-{entry['node_id']}")
    goal = str(meta.get("goal") or "")
    parsed = parse_criteria(goal)
    submission = entry.get("submission") or {}
    submitted = {str(k).upper(): v for k, v in (submission.get("criteria") or {}).items()}

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

    grade = Grade(str(submission.get("evidence_grade", Grade.WITHHELD.value)))
    delivered = tuple(submission.get("artifacts") or ())
    record = NodeRecord(
        node_id=int(entry["node_id"]),
        ext_id=ext_id,
        name=str(entry["name"]),
        repo=str(meta.get("repo") or "unknown"),
        mission_id=entry.get("mission_id"),
        mission_status=str(entry.get("mission_status") or "unknown"),
        is_maturity=bool(meta.get("is_maturity", False)),
        criteria=tuple(criteria),
        delivered_artifacts=delivered,
        evidence_grade=grade,
        rationale=str(submission.get("rationale") or ""),
    )
    if record.verdict is not Verdict.ACCEPT:
        cause = submission.get("root_cause")
        if cause:
            record = replace(
                record,
                root_cause=RootCause(
                    kind=RootCauseKind(cause["kind"]),
                    statement=cause["statement"],
                    evidence=cause["evidence"],
                    confidence=cause.get("confidence", "medium"),
                    actions=tuple(cause.get("actions") or ()),
                ),
            )
        else:
            record = replace(record, root_cause=_default_root_cause(record, probes))
    if not record.rationale:
        record = replace(record, rationale=_default_rationale(record, probes))
    return record


def _default_root_cause(record: NodeRecord, probes: Sequence[GateProbe]) -> RootCause:
    """Root cause for a node that delivered nothing: the evidence gap itself."""
    node_probes = [p for p in probes if p.node_id == record.node_id]
    if node_probes:
        worst = max(node_probes, key=lambda p: abs(p.rel_error_pct))
        return RootCause(
            kind=RootCauseKind.EVIDENCE_GAP,
            statement=(
                f"no acceptance submission; gate probe on {worst.endpoint} measures "
                f"{worst.rel_error_pct:+.2f}% on {worst.quantity}, so the node boundary "
                "(fix not yet evidenced) is still open"
            ),
            evidence=(
                f"mission {record.mission_id} status={record.mission_status}; "
                f"probe {worst.probe_id} measured {worst.measured:g} vs reference "
                f"{worst.reference:g} {worst.unit} ({worst.reference_source})"
            ),
            confidence="high",
            actions=("deliver the node's own acceptance run before the gate can adjudicate",),
        )
    return RootCause(
        kind=RootCauseKind.EVIDENCE_GAP,
        statement="node delivered no acceptance evidence before the gate cutoff",
        evidence=(
            f"mission {record.mission_id} status={record.mission_status}; "
            f"no linked report, no acceptance JSON, worktree clean at cutoff"
        ),
        confidence="high",
        actions=("rerun the node, publish an acceptance artifact, then re-run the gate",),
    )


def _default_rationale(record: NodeRecord, probes: Sequence[GateProbe]) -> str:
    if record.verdict is Verdict.ACCEPT:
        return "all machine criteria measured and passed against external references"
    if record.evidence_grade is Grade.NOT_SUPPORTED:
        return "capability absent at gate cutoff; cannot inherit, must be rebuilt then re-gated"
    return "acceptance open at gate cutoff; capability stays out of the next baseline"


def write_report(report: GateReport, *, json_path: str, md_path: str) -> dict[str, Any]:
    """Persist the machine report (JSON) and the paper-grade summary (Markdown)."""
    payload = report.to_json()
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.render_markdown())
    verify_verdict_integrity(report, payload)
    return payload