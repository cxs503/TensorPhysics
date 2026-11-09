"""Pre-registration of campaign-node acceptance criteria (pre-freeze).

A campaign node's acceptance criteria are published *before* the work starts.
This module turns the free-text "acceptance criteria" block of a mission goal
into a structured, canonical record whose ``sha256`` is written into the mission
metadata.  Everything downstream (drift audit, KPI view, event stream) is
computed against this frozen record, so "the criteria moved after the fact" is
a machine-detectable fact rather than a human judgement.

Three properties are provided here:

(a) **pre-freeze** -- :func:`freeze` produces a canonical manifest plus its
    ``manifest_sha256``; :func:`attach_to_metadata` is the only sanctioned way
    to publish it into mission metadata (it refuses unregistered manifests).
(b) **post-hoc auditable** -- the canonical bytes are stored alongside the
    digest, so a re-canonicalisation reproduces (or refutes) the digest without
    trusting the producer (:func:`canonical_bytes`, :func:`verify_manifest`).
(c) **drift decidable** -- every criterion carries a stable identity key so a
    later receipt can be matched field-by-field (see
    :mod:`~tensorphysics.campaign.criterion_prereg.drift_audit`).

Failure direction is fail-closed: an empty/unparseable goal yields the state
``unregistered`` and *no* digest is published; a digest that does not match its
own canonical bytes yields ``tampered``.

Zero domain semantics: metric names, scopes and operators are data extracted
from the goal text.  The only built-in vocabulary is process vocabulary
("acceptance criteria" block headers, connector words and generic separators);
it is loaded from :mod:`criterion_prereg.domain_vocabulary`-free constants
below -- no engineering-domain term appears in this module (see
:mod:`~tensorphysics.campaign.criterion_prereg.scan`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "criterion-prereg/1"
EXTRACTOR_VERSION = "1.0.0"

#: ``state`` values of a freeze/verify operation.
REGISTERED = "registered"
UNREGISTERED = "unregistered"
TAMPERED = "tampered"

#: Goal block headers that introduce the acceptance-criteria block.  Generic
#: process vocabulary only (English + Chinese process terms); matches are
#: case-insensitive, and the *last* header wins (mission goals append the
#: criteria table after the narrative).
CRITERIA_HEADERS: tuple[str, ...] = (
    "acceptance criteria",
    "acceptance-criteria",
    "machine acceptance",
    "验收判据",
    "验收标准",
)

#: Clause separators.  A comma splits unless it sits *between* two digits, so
#: thousands separators ("1,200") stay intact while "...=1, (d) ..." still splits.
_CLAUSE_SPLIT = re.compile(r"[;；\n。、]|(?<!\d)[,，]|[,，](?!\d)")

#: Optional clause label: ``(a)``, ``[b]``, ``c)``, ``1.``  (full-width too).
#: Full-width glyphs folded to ASCII before parsing, so the same requirement
#: spelled with "＞＝" / "％" / "（a）" extracts identically to its ASCII form.
_FULLWIDTH = str.maketrans(
    {
        "＞": ">",
        "＜": "<",
        "＝": "=",
        "％": "%",
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "\u3000": " ",
    }
)


def normalise_text(text: str) -> str:
    """Fold full-width punctuation/glyphs so parsing is surface-independent."""
    return (text or "").translate(_FULLWIDTH)


_LABEL_RE = re.compile(r"^\s*[\(\[（]\s*([A-Za-z]|\d{1,2})\s*[\)\]）]\s*|^\s*([A-Za-z]|\d{1,2})\s*[\.\)）、]\s*")

#: Punctuation/whitespace tolerated in front of a clause (block header residue).
_CLAUSE_PREFIX = " \t\u3000:：-—,，、。.；;"

#: Numeric comparator: ``>= 0.92``, ``≥0.92``, ``= 0``, ``<=5%``.
_OP_MAP = {
    ">=": ">=",
    "≥": ">=",
    "=>": ">=",
    "<=": "<=",
    "≤": "<=",
    "=<": "<=",
    ">": ">",
    "＞": ">",
    "<": "<",
    "＜": "<",
    "==": "==",
    "=": "==",
    "＝": "==",
}
_NUM_RE = re.compile(
    r"(?P<op>>=|≥|=>|<=|≤|=<|＞|＜|>|<|==|=|＝)\s*"
    r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>%|％)?"
)

#: Behavioural/mapping criterion: ``input -> expected_state``.
_ARROW = "->"
_ARROW_RE = re.compile(r"(?P<lhs>[^\-]+?)\s*(?:->|→|-->)\s*(?P<rhs>[A-Za-z_][A-Za-z0-9_]*)")

#: Evidence-size qualifier inside the scope ("24 cases", "24例").
_COUNT_RE = re.compile(r"(?P<n>\+?\d+)\s*(?:cases?|items?|samples?|例|个|条)", re.IGNORECASE)

_ASCII_METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.\-]*)\s*$")
_CJK_METRIC_RE = re.compile(r"([\u4e00-\u9fff]+)\s*$")

#: Connector words stripped from the tail of a scope descriptor.
_CONNECTORS = ("不低于", "不小于", "不高于", "不大于", "至少", "至多", "达到", "要求", "应为", "should be", "must be", "is", "at least", "at most")

#: Max length of a Chinese metric token (documented heuristic, see module doc).
_CJK_METRIC_MAX = 4


class ManifestError(ValueError):
    """The manifest or its inputs violate the pre-registration contract."""


# --------------------------------------------------------------------------- #
# canonical serialisation
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, UTF-8 safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Digest of the canonical form of ``obj``."""
    return sha256_text(canonical_json(obj))


# --------------------------------------------------------------------------- #
# structured criterion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criterion:
    """One structured acceptance criterion extracted from a goal text."""

    label: str
    kind: str  # "threshold" | "mapping"
    metric: str
    op: str  # ">=", "<=", ">", "<", "==", "->"
    value: str  # canonical numeric string, or symbolic target for "mapping"
    unit: str = ""
    count: int | None = None
    scope: str = ""
    text: str = ""

    @property
    def identity(self) -> str:
        """Stable key used to pair a frozen criterion with a receipt one.

        Deliberately *independent of the value/scope* so that a value change or
        a scope change is still paired (and therefore classifiable) rather than
        being reported as a removal plus an addition.
        """
        return f"{self.kind}:{self.metric.lower()}"

    @property
    def cid(self) -> str:
        return sha256_text(canonical_json([self.identity, self.label]))[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "label": self.label,
            "kind": self.kind,
            "metric": self.metric,
            "op": self.op,
            "value": self.value,
            "unit": self.unit,
            "count": self.count,
            "scope": self.scope,
            "text": self.text,
        }

    def numeric(self) -> Decimal | None:
        if self.kind != "threshold":
            return None
        try:
            return Decimal(self.value)
        except InvalidOperation:  # pragma: no cover - value is validated at parse
            return None


@dataclass
class Extraction:
    """Result of parsing a goal text."""

    criteria: list[Criterion] = field(default_factory=list)
    block: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.criteria),
            "criteria": [c.to_dict() for c in self.criteria],
            "warnings": list(self.warnings),
        }


def _normalise_value(num: str, unit: str) -> tuple[str, str]:
    """Canonical numeric text (``0.920`` -> ``0.92``) and unit (``％`` -> ``%``)."""
    try:
        dec = Decimal(num)
    except InvalidOperation as exc:  # pragma: no cover - guarded by regex
        raise ManifestError(f"unparseable number {num!r}") from exc
    text = format(dec.normalize(), "f")
    if text in ("-0", ""):
        text = "0"
    return text, ("%" if unit in ("%", "％") else "")


def _strip_connectors(text: str) -> str:
    out = text.strip(" :：-\u3000\t")
    changed = True
    while changed:
        changed = False
        for conn in _CONNECTORS:
            if out.lower().endswith(conn.lower()):
                out = out[: -len(conn)].strip(" :：-\u3000\t")
                changed = True
    return out


def _split_metric(prefix: str) -> tuple[str, str, int | None]:
    """Split ``prefix`` into ``(metric, scope, count)``.

    Contract (documented grammar): ``scope`` is the descriptor text in front of
    the metric, a trailing ASCII identifier run is the metric, and a trailing
    Chinese run truncated to :data:`_CJK_METRIC_MAX` characters is the metric
    when no ASCII identifier is present.  An evidence-size qualifier
    ("24 cases" / "24例") is lifted out of the scope into ``count``.
    """
    text = _strip_connectors(prefix)
    count: int | None = None
    m_count = None
    for m_count in _COUNT_RE.finditer(text):
        pass
    if m_count is not None:
        raw = m_count.group("n").lstrip("+")
        try:
            count = int(raw)
        except ValueError:  # pragma: no cover
            count = None
        text = (text[: m_count.start()] + " " + text[m_count.end():]).strip()

    metric = ""
    scope = text.strip()
    m_ascii = _ASCII_METRIC_RE.search(text)
    if m_ascii:
        metric = m_ascii.group(1)
        scope = text[: m_ascii.start()].strip()
    else:
        m_cjk = _CJK_METRIC_RE.search(text)
        if m_cjk:
            run = m_cjk.group(1)
            metric = run[-_CJK_METRIC_MAX:]
            scope = (text[: m_cjk.start()] + run[:-_CJK_METRIC_MAX]).strip()
    scope = _strip_connectors(scope)
    return metric, scope, count


def _label_of(clause: str) -> tuple[str, str]:
    clause = clause.lstrip(_CLAUSE_PREFIX)
    m = _LABEL_RE.match(clause)
    if not m:
        return "", clause
    label = (m.group(1) or m.group(2) or "").strip()
    return label.lower(), clause[m.end():]


def _parse_clause(clause: str) -> tuple[Criterion | None, str | None]:
    """Parse one clause into a :class:`Criterion` plus an optional warning."""
    label, body = _label_of(clause)
    body_stripped = body.strip()
    if not body_stripped:
        return None, None

    m_thr = _NUM_RE.search(body)
    if m_thr:
        if len(_NUM_RE.findall(body)) > 1:
            return None, f"ambiguous_clause:{body_stripped[:60]}"
        op = _OP_MAP[m_thr.group("op")]
        value, unit = _normalise_value(m_thr.group("num"), m_thr.group("unit") or "")
        metric, scope, count = _split_metric(body[: m_thr.start()])
        if not metric:
            return None, f"missing_metric:{body_stripped[:60]}"
        return (
            Criterion(
                label=label,
                kind="threshold",
                metric=metric,
                op=op,
                value=value,
                unit=unit,
                count=count,
                scope=scope,
                text=body_stripped,
            ),
            None,
        )

    m_arrow = _ARROW_RE.search(body)
    if m_arrow:
        lhs = m_arrow.group("lhs")
        rhs = m_arrow.group("rhs")
        metric, scope, count = _split_metric(lhs)
        if not metric:
            return None, f"missing_metric:{body_stripped[:60]}"
        return (
            Criterion(
                label=label,
                kind="mapping",
                metric=metric,
                op=_ARROW,
                value=rhs,
                unit="",
                count=count,
                scope=scope,
                text=body_stripped,
            ),
            None,
        )
    return None, f"unparseable_clause:{body_stripped[:60]}"


def criteria_block(goal: str, headers: Sequence[str] = CRITERIA_HEADERS) -> str:
    """Return the acceptance-criteria block of ``goal`` (whole goal if absent)."""
    if not goal or not goal.strip():
        return ""
    lowered = goal.lower()
    position = -1
    for header in headers:
        idx = lowered.rfind(header.lower())
        if idx >= 0:
            position = max(position, idx + len(header))
    if position < 0:
        return goal
    return goal[position:]


def extract_criteria(goal: str, *, headers: Sequence[str] = CRITERIA_HEADERS) -> Extraction:
    """Extract structured criteria from a goal text (fail-closed on empty input)."""
    block = normalise_text(criteria_block(goal, headers))
    if not block.strip():
        return Extraction(criteria=[], block="", warnings=["empty_goal"])
    criteria: list[Criterion] = []
    warnings: list[str] = []
    for raw in _CLAUSE_SPLIT.split(block):
        clause = raw.strip()
        if not clause:
            continue
        crit, warn = _parse_clause(clause)
        if crit is not None:
            criteria.append(crit)
        elif warn:
            warnings.append(warn)
    return Extraction(criteria=criteria, block=block.strip(), warnings=warnings)


# --------------------------------------------------------------------------- #
# freeze / verify
# --------------------------------------------------------------------------- #
@dataclass
class Freeze:
    """Outcome of a pre-registration freeze."""

    state: str
    manifest: dict[str, Any]

    @property
    def registered(self) -> bool:
        return self.state == REGISTERED

    @property
    def sha256(self) -> str:
        return self.manifest.get("manifest_sha256", "")

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "manifest": self.manifest}


#: Keys excluded from the digest: the digest itself and the derived ``state``
#: label (so a consumer can restamp the state without invalidating the digest).
_NOT_DIGESTIBLE = ("manifest_sha256", "state")


def _digestible(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k not in _NOT_DIGESTIBLE}


def _source_digest() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - source not readable (zip/frozen import)
        return ""


def freeze(
    goal: str,
    *,
    mission_id: str | int | None = None,
    node_id: str | None = None,
    frozen_by: str = "",
    frozen_at: str | None = None,
    headers: Sequence[str] = CRITERIA_HEADERS,
    metadata: Mapping[str, Any] | None = None,
) -> Freeze:
    """Freeze the acceptance criteria of ``goal`` into a digestible manifest.

    Fail-closed: no criteria extracted -> ``state = unregistered`` and an empty
    digest; the caller must not publish it (:func:`attach_to_metadata` refuses).
    """
    extraction = extract_criteria(goal, headers=headers)
    stamp = frozen_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "extractor": {
            "version": EXTRACTOR_VERSION,
            "source_sha256": _source_digest(),
        },
        "mission_id": mission_id,
        "node_id": node_id,
        "frozen_by": frozen_by,
        "frozen_at": stamp,
        "goal_sha256": sha256_text(goal or ""),
        "n_criteria": len(extraction.criteria),
        "criteria": [c.to_dict() for c in extraction.criteria],
        "warnings": list(extraction.warnings),
        "metadata_keys": sorted((metadata or {}).keys()),
    }
    if not extraction.criteria:
        manifest["manifest_sha256"] = ""
        manifest["state"] = UNREGISTERED
        return Freeze(state=UNREGISTERED, manifest=manifest)
    manifest["manifest_sha256"] = sha256_obj(_digestible(manifest))
    manifest["state"] = REGISTERED
    return Freeze(state=REGISTERED, manifest=manifest)


def semantic_digest(manifest: Mapping[str, Any]) -> str:
    """Digest of the *semantics* of a frozen record (raw clause text excluded).

    The manifest digest (:func:`freeze`) is bitwise: it freezes the exact bytes
    including the raw clause text.  This second digest makes reformatting
    ("0.920" vs "0.92", "≥" vs ">=") invisible while still catching any change
    to a metric, operator, value, scope, count or label -- so an audit can
    separate *presentation* drift from *requirement* drift.
    """
    semantic = [
        {k: v for k, v in rec.items() if k not in ("text", "cid")}
        for rec in (manifest.get("criteria") or [])
    ]
    return sha256_obj(semantic)


def verify_manifest(manifest: Mapping[str, Any], *, goal: str | None = None) -> str:
    """Re-derive the digest of a frozen manifest and return its ``state``.

    ``tampered`` means the recorded digest does not match its own canonical
    bytes (or the goal text no longer hashes to the frozen ``goal_sha256``).
    ``unregistered`` means the record carries no digest / no criteria.
    """
    if not manifest:
        return UNREGISTERED
    digest = manifest.get("manifest_sha256") or ""
    if not digest or not manifest.get("criteria"):
        return UNREGISTERED
    if sha256_obj(_digestible(manifest)) != digest:
        return TAMPERED
    if goal is not None and manifest.get("goal_sha256") not in (None, "") and sha256_text(goal) != manifest.get("goal_sha256"):
        return TAMPERED
    return REGISTERED


def attach_to_metadata(metadata: Mapping[str, Any] | None, freeze_result: Freeze) -> dict[str, Any]:
    """Publish a *registered* manifest into mission metadata (fail-closed)."""
    if not freeze_result.registered:
        raise ManifestError("refusing to publish an unregistered criterion manifest")
    payload = {
        "schema": SCHEMA,
        "manifest_sha256": freeze_result.sha256,
        "n_criteria": freeze_result.manifest["n_criteria"],
        "frozen_at": freeze_result.manifest["frozen_at"],
        "criterion_ids": [c["cid"] for c in freeze_result.manifest["criteria"]],
    }
    out = dict(metadata or {})
    out["criterion_manifest"] = payload
    return out


def manifest_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = (metadata or {}).get("criterion_manifest")
    return dict(payload) if isinstance(payload, Mapping) else None


def criteria_of(manifest: Mapping[str, Any]) -> list[Criterion]:
    """Rebuild :class:`Criterion` objects from a manifest's stored records."""
    out: list[Criterion] = []
    for rec in manifest.get("criteria") or []:
        out.append(
            Criterion(
                label=rec.get("label", ""),
                kind=rec.get("kind", "threshold"),
                metric=rec.get("metric", ""),
                op=rec.get("op", "=="),
                value=rec.get("value", ""),
                unit=rec.get("unit", ""),
                count=rec.get("count"),
                scope=rec.get("scope", ""),
                text=rec.get("text", ""),
            )
        )
    return out


def iter_field_triples(criteria: Iterable[Criterion], fields: Sequence[str] = (
    "label", "kind", "metric", "op", "value", "unit", "count", "scope",
)) -> list[tuple[int, str, str]]:
    """Flatten criteria into ``(index, field, value)`` triples (for F1 scoring)."""
    out: list[tuple[int, str, str]] = []
    for idx, crit in enumerate(criteria):
        for name in fields:
            raw = getattr(crit, name)
            out.append((idx, name, "" if raw is None else str(raw)))
    return out


__all__ = [
    "SCHEMA",
    "EXTRACTOR_VERSION",
    "REGISTERED",
    "UNREGISTERED",
    "TAMPERED",
    "CRITERIA_HEADERS",
    "Criterion",
    "Extraction",
    "Freeze",
    "ManifestError",
    "canonical_json",
    "sha256_text",
    "sha256_obj",
    "criteria_block",
    "extract_criteria",
    "freeze",
    "verify_manifest",
    "attach_to_metadata",
    "manifest_from_metadata",
    "criteria_of",
    "iter_field_triples",
    "normalise_text",
    "semantic_digest",
]