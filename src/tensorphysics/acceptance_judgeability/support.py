"""Support ratio (KS), evidence model, PASS adjudication and upgrade paths.

``KS`` (*judgeability support*) is defined before any number is looked at::

    KS(node) = sum of weights of (JUDGEABLE and measured) clauses
               --------------------------------------------------
               sum of weights of all clauses of the node

Three consequences are built in, not bolted on:

* ``RECORD_ONLY`` and ``WITHHELD_OK`` clauses only ever appear in the
  denominator, so prose can never raise support;
* a ``JUDGEABLE`` clause contributes nothing until there is *evidence* that it was
  actually measured (:class:`Evidence`), so "wrote a good criterion" is not the
  same as "measured it";
* a node with no parseable clause has ``KS = None`` (undefined), never ``0.0``:
  "no criterion" and "criterion that cannot fail" must not be confused when
  thresholds are applied.

Evidence is deliberately a node-level fact here (``done`` + no repeat failure),
because the platform stores measurement results in per-node reports, not per
clause. :meth:`Evidence.from_node` therefore records *how* that node-level fact
was obtained, and the census can be re-run in a stricter ``require_report`` mode
for a sensitivity check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .classify import ClassifiedClause, Judgeability, Subtype, classify_all

DEFAULT_KS_MIN = 0.5
PASS_VERDICTS = ("ACCEPT", "PASS", "pass", "accepted", "accept")


# --------------------------------------------------------------------------- clauses
@dataclass(frozen=True)
class Clause:
    """A raw acceptance clause (``cid`` + ``text``)."""

    cid: str
    text: str

    @classmethod
    def from_any(cls, item: Any) -> "Clause":
        if isinstance(item, Clause):
            return item
        if isinstance(item, Mapping):
            return cls(str(item.get("cid", "")), str(item.get("text", "")))
        cid, text = item
        return cls(str(cid), str(text))

    def as_dict(self) -> dict[str, Any]:
        return {"cid": self.cid, "text": self.text}


def _as_classified(clauses: Iterable[Any]) -> list[ClassifiedClause]:
    first = list(clauses)
    if not first:
        return []
    if isinstance(first[0], ClassifiedClause):
        return list(first)  # type: ignore[arg-type]
    return classify_all([Clause.from_any(c) for c in first])


# --------------------------------------------------------------------------- evidence
@dataclass(frozen=True)
class Evidence:
    """Whether the node's criteria were actually measured against something."""

    measured: bool
    kind: str
    detail: str = ""

    @classmethod
    def from_node(
        cls, node: Mapping[str, Any], *, require_report: bool = False, min_reports: int = 1
    ) -> "Evidence":
        status = str(node.get("status") or "").lower()
        fails = int(node.get("consecutive_fails") or 0)
        reports = int(((node.get("evidence") or {}).get("n_reports")) or 0)
        done = status == "done" and fails == 0
        if not done:
            return Evidence(False, "NODE_NOT_DONE", f"status={status or 'unknown'} fails={fails}")
        if require_report and reports < min_reports:
            return Evidence(False, "REPORT_MISSING", f"status=done reports={reports}")
        return Evidence(
            True,
            "NODE_DONE_ACCEPTED" if not require_report else "REPORT_PUBLISHED",
            f"status=done fails=0 reports={reports}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"measured": self.measured, "kind": self.kind, "detail": self.detail}


# --------------------------------------------------------------------------- weights
@dataclass(frozen=True)
class WeightPolicy:
    """Clause weighting. Uniform by default: KS is then a plain share of clauses."""

    base: float = 1.0
    by_class: Mapping[str, float] = field(default_factory=dict)
    name: str = "uniform"

    def weight(self, clause: ClassifiedClause) -> float:
        return float(self.by_class.get(clause.judgeability.value, self.base))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "base": self.base, "by_class": dict(self.by_class)}


UNIFORM_WEIGHTS = WeightPolicy()
#: sensitivity variant: a hard (judgeable) clause counts double against prose
TOLERANCE_WEIGHTS = WeightPolicy(
    name="tolerance_weighted",
    by_class={"JUDGEABLE": 2.0, "THRESHOLD_ONLY": 1.0, "RECORD_ONLY": 1.0, "WITHHELD_OK": 1.0},
)


# --------------------------------------------------------------------------- policy
@dataclass(frozen=True)
class PassPolicy:
    """What it takes for a node verdict to be *supported* by its criteria."""

    ks_min: float = DEFAULT_KS_MIN
    allow_self_baseline: bool = True
    require_measured: bool = True
    #: sensitivity switch: count a *threshold on an error/deviation metric* as
    #: judgeable even when the reference is not named (an "implicit pair": the
    #: clause says ``error < x``, which presupposes something compared against).
    #: Off by default -- a PASS must be re-checkable from the clause alone.
    count_implicit_pair_as_judgeable: bool = False
    #: mechanism invariant -- kept for documentation/testing, never set to True
    record_only_can_support: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ks_min": self.ks_min,
            "allow_self_baseline": self.allow_self_baseline,
            "require_measured": self.require_measured,
            "record_only_can_support": self.record_only_can_support,
            "count_implicit_pair_as_judgeable": self.count_implicit_pair_as_judgeable,
        }


DEFAULT_POLICY = PassPolicy()
STRICT_POLICY = PassPolicy(ks_min=0.5, allow_self_baseline=False)
#: optimistic bound: unnamed error/deviation thresholds get the benefit of the doubt
IMPLICIT_POLICY = PassPolicy(ks_min=0.5, count_implicit_pair_as_judgeable=True)


# --------------------------------------------------------------------------- support
@dataclass(frozen=True)
class KsRow:
    """One node's judgeability scorecard."""

    node_id: int
    plan_id: int
    ext_id: str
    repo: str
    status: str
    n_clauses: int
    counts: Mapping[str, int]
    weight_total: float
    weight_judgeable: float
    weight_support: float
    ks: float | None
    evidence: Evidence
    ks_min: float = DEFAULT_KS_MIN
    name: str = ""
    plan_name: str = ""

    @property
    def hollow_pass(self) -> bool:
        """A node the campaign counts as PASS whose criteria cannot carry a PASS."""
        return self.status.lower() == "done" and (self.ks is None or self.ks < self.ks_min)

    @property
    def severity(self) -> str:
        if self.ks is None:
            return "NO_CRITERIA"
        if self.ks <= 0.0:
            return "TOTAL"      # nothing judgeable at all
        if self.ks < 0.25:
            return "SEVERE"
        if self.ks < self.ks_min:
            return "PARTIAL"
        return "SUPPORTED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "plan_id": self.plan_id,
            "plan_name": self.plan_name,
            "ext_id": self.ext_id,
            "name": self.name,
            "repo": self.repo,
            "status": self.status,
            "n_clauses": self.n_clauses,
            "counts": dict(self.counts),
            "weight_total": round(self.weight_total, 6),
            "weight_judgeable": round(self.weight_judgeable, 6),
            "weight_support": round(self.weight_support, 6),
            "ks": None if self.ks is None else round(self.ks, 6),
            "severity": self.severity,
            "hollow_pass": self.hollow_pass,
            "evidence": self.evidence.as_dict(),
        }


def judgeability_support(
    clauses: Iterable[Any],
    evidence: Evidence,
    *,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
    policy: PassPolicy = DEFAULT_POLICY,
) -> tuple[float | None, dict[str, Any]]:
    """Return ``(KS, components)`` for one node.

    ``KS`` is ``None`` when the node declares no clause at all (undefined, not 0).
    """
    classified = _as_classified(clauses)
    comp: dict[str, Any] = {
        "n_clauses": len(classified),
        "counts": {k.value: 0 for k in Judgeability},
        "weight_total": 0.0,
        "weight_judgeable": 0.0,
        "weight_support": 0.0,
        "subtypes": {},
    }
    if not classified:
        return None, comp

    for clause in classified:
        w = weights.weight(clause)
        comp["counts"][clause.judgeability.value] += 1
        comp["subtypes"][clause.subtype.value] = comp["subtypes"].get(clause.subtype.value, 0) + 1
        comp["weight_total"] += w
        counts_as_judgeable = clause.judgeability is Judgeability.JUDGEABLE or (
            policy.count_implicit_pair_as_judgeable
            and clause.judgeability is Judgeability.THRESHOLD_ONLY
            and clause.features.has_tolerance
        )
        if counts_as_judgeable:
            if clause.subtype is Subtype.SELF_REFERENCE and not policy.allow_self_baseline:
                continue  # strict policy: a self-referential target is not a baseline
            comp["weight_judgeable"] += w
            if evidence.measured or not policy.require_measured:
                comp["weight_support"] += w

    total = comp["weight_total"]
    if total <= 0:  # pragma: no cover - guarded by base>0
        return None, comp
    return comp["weight_support"] / total, comp


def node_ks(
    node: Mapping[str, Any],
    *,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
    policy: PassPolicy = DEFAULT_POLICY,
    require_report: bool = False,
) -> KsRow:
    """Score one corpus node row (``{"criteria": [...], "status": ...}``)."""
    evidence = Evidence.from_node(node, require_report=require_report)
    ks, comp = judgeability_support(
        node.get("criteria") or [], evidence, weights=weights, policy=policy
    )
    return KsRow(
        node_id=int(node.get("node_id") or 0),
        plan_id=int(node.get("plan_id") or 0),
        ext_id=str(node.get("ext_id") or ""),
        repo=str(node.get("repo") or ""),
        status=str(node.get("status") or "unknown"),
        n_clauses=comp["n_clauses"],
        counts=comp["counts"],
        weight_total=comp["weight_total"],
        weight_judgeable=comp["weight_judgeable"],
        weight_support=comp["weight_support"],
        ks=ks,
        evidence=evidence,
        ks_min=policy.ks_min,
        name=str(node.get("name") or ""),
        plan_name=str(node.get("plan_name") or ""),
    )


# --------------------------------------------------------------------------- adjudication
@dataclass(frozen=True)
class Adjudication:
    """The gate-level decision for one node."""

    outcome: str                    # SUPPORTED | HOLLOW_PASS | NO_CRITERIA | UNMEASURED
    supports_pass: bool
    ks: float | None
    blocking: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    policy: PassPolicy = DEFAULT_POLICY

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "supports_pass": self.supports_pass,
            "ks": None if self.ks is None else round(self.ks, 6),
            "blocking": list(self.blocking),
            "reasons": list(self.reasons),
            "policy": self.policy.as_dict(),
        }


def adjudicate(
    clauses: Iterable[Any],
    evidence: Evidence,
    *,
    policy: PassPolicy = DEFAULT_POLICY,
    weights: WeightPolicy = UNIFORM_WEIGHTS,
) -> Adjudication:
    """Decide whether these clauses can carry a PASS (mechanism, not advice).

    Order of refusal matters and is part of the contract:

    1. no clause at all                  -> ``NO_CRITERIA`` (cannot be judged)
    2. no ``JUDGEABLE`` clause at all    -> ``HOLLOW_PASS`` (RECORD_ONLY alone)
    3. ``KS < policy.ks_min``            -> ``HOLLOW_PASS``
    4. no evidence that it was measured  -> ``UNMEASURED``
    5. otherwise                         -> ``SUPPORTED``
    """
    classified = _as_classified(clauses)
    if not classified:
        return Adjudication(
            "NO_CRITERIA", False, None, blocking=("NO_CRITERIA",), reasons=(), policy=policy
        )

    ks, comp = judgeability_support(classified, evidence, weights=weights, policy=policy)
    counts = comp["counts"]
    judgeable = counts[Judgeability.JUDGEABLE.value]
    blocking: list[str] = []
    reasons: list[str] = []

    implicit_counts = policy.count_implicit_pair_as_judgeable and comp["weight_judgeable"] > 0
    if not policy.record_only_can_support and judgeable == 0 and not implicit_counts:
        blocking.append("RECORD_ONLY_ONLY")
        reasons.append(
            "no JUDGEABLE clause: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        )
        return Adjudication("HOLLOW_PASS", False, ks, tuple(blocking), tuple(reasons), policy)

    if comp["weight_judgeable"] <= 0.0:
        blocking.append("SELF_BASELINE_DISALLOWED")
        reasons.append("all judgeable clauses rely on a self-referential target (strict policy)")
        return Adjudication("HOLLOW_PASS", False, ks, tuple(blocking), tuple(reasons), policy)

    if policy.require_measured and not evidence.measured and comp["weight_judgeable"] > 0:
        # the criteria are fine, they were simply never measured: say so instead of
        # blaming the threshold, otherwise the two very different defects look alike
        blocking.append("NOT_MEASURED")
        reasons.append(f"evidence={evidence.kind} ({evidence.detail})")
        return Adjudication("UNMEASURED", False, ks, tuple(blocking), tuple(reasons), policy)

    if ks is None or ks < policy.ks_min:
        blocking.append("KS_BELOW_THRESHOLD")
        reasons.append(f"KS={None if ks is None else round(ks, 4)} < ks_min={policy.ks_min}")
        return Adjudication("HOLLOW_PASS", False, ks, tuple(blocking), tuple(reasons), policy)

    if policy.require_measured and not evidence.measured:
        blocking.append("NOT_MEASURED")
        reasons.append(f"evidence={evidence.kind} ({evidence.detail})")
        return Adjudication("UNMEASURED", False, ks, tuple(blocking), tuple(reasons), policy)

    reasons.append(f"KS={round(ks, 4)} >= {policy.ks_min}; judgeable clauses={judgeable}")
    return Adjudication("SUPPORTED", True, ks, (), tuple(reasons), policy)


def gate_flag(adjudication: Adjudication) -> dict[str, Any]:
    """The plain-dict flag a gate record can carry without importing this package."""
    d = adjudication.as_dict()
    d["gate"] = "acceptance_judgeability"
    return d


def enforce_verdict(verdict: str, adjudication: Adjudication) -> str:
    """Downgrade a PASS to ``WITHHELD`` when its criteria cannot carry it.

    Only pass-like verdicts are touched: a FAIL is already honest and must not be
    softened by this guard (evidence discipline: honest FAIL is not overridable).
    """
    if str(verdict).upper() in {v.upper() for v in PASS_VERDICTS}:
        return str(verdict) if adjudication.supports_pass else "WITHHELD"
    return str(verdict)


# --------------------------------------------------------------------------- upgrades
_ACTIONS: Mapping[tuple[str, str], str] = {
    (Judgeability.THRESHOLD_ONLY.value, Subtype.NO_BASELINE.value): (
        "补基准：写明对照物 y_ref（解析解/文献/实测参考值）与相对容差 τ；当前只有阈值，"
        "达标与否无法指向任何外部真值"
    ),
    (Judgeability.THRESHOLD_ONLY.value, Subtype.NO_TOLERANCE.value): (
        "补容差：把趋势/极限关系改写为数值判据 |Δ|/scale ≤ τ，并给出自变量网格"
    ),
    (Judgeability.RECORD_ONLY.value, Subtype.MONOTONIC.value): (
        "单调性判据可量化：给出自变量网格、采样点与斜率下界（或参考区间），升级为数值判据"
    ),
    (Judgeability.RECORD_ONLY.value, Subtype.QUALITATIVE.value): (
        "把定性要求改写为可判量：指定被测量 y、参考 y_ref 与容差 τ；若确实无法量化，"
        "改为 WITHHELD_OK 并写明缺哪个数据"
    ),
    (Judgeability.RECORD_ONLY.value, Subtype.UNCLASSIFIED.value): (
        "条款无法解析：拆分为可判子句，或显式标注 WITHHELD_OK（缺数据来源）"
    ),
    (Judgeability.WITHHELD_OK.value, Subtype.DECLARED_MISSING.value): (
        "已显式声明缺数据：保留原样但不计入 PASS 支撑；一旦数据可得，补 y_ref 与 τ 转入 JUDGEABLE"
    ),
    (Judgeability.JUDGEABLE.value, Subtype.EXTERNAL_REFERENCE.value): (
        "已可判但未实测：发布报告数值表 (y, y_ref, τ, error/τ) 并挂到节点证据上"
    ),
    (Judgeability.JUDGEABLE.value, Subtype.SELF_REFERENCE.value): (
        "自参照残差：可判但只证明自洽；补一个外部锚（解析解/文献/实测）才能计入成熟度"
    ),
}

_MISSING: Mapping[str, tuple[str, ...]] = {
    Subtype.NO_BASELINE.value: ("reference",),
    Subtype.NO_TOLERANCE.value: ("tolerance",),
    Subtype.MONOTONIC.value: ("tolerance", "reference"),
    Subtype.QUALITATIVE.value: ("quantity", "reference", "tolerance"),
    Subtype.UNCLASSIFIED.value: ("quantity", "reference", "tolerance"),
    Subtype.DECLARED_MISSING.value: ("data",),
    Subtype.SELF_REFERENCE.value: ("external_reference",),
    Subtype.EXTERNAL_REFERENCE.value: ("measurement",),
}


def upgrade_paths(clauses: Iterable[Any]) -> list[dict[str, Any]]:
    """One actionable upgrade suggestion per clause that cannot carry a PASS."""
    out: list[dict[str, Any]] = []
    for clause in _as_classified(clauses):
        if clause.contributes_to_support and clause.subtype is Subtype.EXTERNAL_REFERENCE:
            continue  # already the strongest form
        key = (clause.judgeability.value, clause.subtype.value)
        out.append(
            {
                "cid": clause.cid,
                "text": clause.text,
                "judgeability": clause.judgeability.value,
                "subtype": clause.subtype.value,
                "missing": list(_MISSING.get(clause.subtype.value, ())),
                "action": _ACTIONS.get(key, "补量/补基准/补容差三件套"),
                "needs_review": clause.needs_review,
            }
        )
    return out