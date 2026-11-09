"""Synthetic corpora with generator-side ground truth for the pre-registration gate.

Two corpora are generated here so that the acceptance numbers of the gate are
measured against *independent* labels (the generator knows what it wrote; the
extractor/auditor must recover it):

``extraction corpus``
    Goal texts plus the structured field values that were used to render them
    (surface variation: clause labels, separators, comparator spellings,
    full-width glyphs, number formatting, evidence-size qualifiers).  Used to
    score field-level precision/recall/F1 of
    :func:`~tensorphysics.campaign.criterion_prereg.criterion_manifest.extract_criteria`.

``drift corpus`` (24 cases)
    A frozen goal plus a receipt goal obtained by applying exactly one labelled
    mutation (``unchanged_reformatted``, ``tightened_raise``, ``loosened_lower``,
    ``scope_count``, ``scope_desc``, ``criterion_dropped``, ``criterion_added``,
    ``op_family``, ``exact_value``, ``mapping_target``).  Used to score drift
    *detection* (recall/precision against "is this a drift?") and per-class
    classification.

Baselines are provided too (:func:`baseline_text_hash`) so the detector can be
compared with the obvious "digest the goal text" approach: reformatting and
legitimate tightening make that baseline cry wolf, which is exactly what the
structured audit is supposed to avoid.

Zero domain semantics: scopes and metric names below are *generic software
measurement* vocabulary (synthetic), never engineering-domain terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import random
from typing import Any, Iterable, Sequence

from .criterion_manifest import Criterion, extract_criteria, iter_field_triples

# --------------------------------------------------------------------------- #
# extraction corpus
# --------------------------------------------------------------------------- #
SCOPES = ["合成检测集", "回归用例集", "字段级", "边界样本", "负控样本", "golden 集", "全量用例"]
ASCII_METRICS = ["recall", "precision", "f1", "accuracy", "miss_rate", "drift_rate"]
#: Chinese metric names are rendered *delimited* (space) because undelimited
#: Chinese metric boundaries are genuinely ambiguous -- that limitation is
#: measured separately by ``UNDELIMITED_SHARE``.
CJK_METRICS = ["扫描命中", "错误条数"]
OP_RENDER = {
    ">=": ["≥", ">="],
    "<=": ["≤", "<="],
    ">": [">", "＞"],
    "<": ["<", "＜"],
    "==": ["=", "=="],
}
COUNT_UNITS = ["例", "cases", "条", "items"]
LABEL_STYLES = ["({})", "[{}]", "{}）", "{}."]
SEPARATORS = ["；", ";", "。", "\n", "、", "，"]


@dataclass
class ExtractedCase:
    case_id: str
    text: str
    gold: list[Criterion]

    def gold_triples(self) -> list[tuple[int, str, str]]:
        return iter_field_triples(self.gold)


def _render_value(value: Decimal, rng: random.Random) -> tuple[str, str]:
    """Render a number in one of several equivalent surface forms."""
    style = rng.choice(["plain", "padded", "percent", "trim"])
    if style == "percent" and value < 1:
        text = format(value * 100, "f").rstrip("0").rstrip(".")
        return (text or "0"), "%"
    if style == "padded":
        return f"{value:.3f}", ""
    return format(value.normalize(), "f"), ""


def _render_criterion(crit: Criterion, rng: random.Random, label_style: str | None, delimiter: str = "") -> str:
    label = ""
    if crit.label and label_style:
        label = label_style.format(crit.label.upper() if rng.random() < 0.5 else crit.label) + " "
    op = rng.choice(OP_RENDER.get(crit.op, [crit.op]))
    value_text = crit.value + crit.unit
    count_text = f"{crit.count}{rng.choice(COUNT_UNITS)}" if crit.count is not None else ""
    metric = crit.metric + delimiter
    return f"{label}{crit.scope}{count_text}{metric}{op}{value_text}"


def _random_criterion(rng: random.Random, label: str, *, undelimited_cjk: bool = False) -> tuple[Criterion, str]:
    """Return a criterion plus the delimiter the renderer must use."""
    scope = rng.choice(SCOPES)
    if rng.random() < 0.25:
        metric = rng.choice(CJK_METRICS)
        delimiter = "" if undelimited_cjk else " "
        kind, op = "threshold", rng.choice([">=", "<=", "==", ">", "<"])
        value = Decimal(rng.choice(["0.92", "0.95", "0.97", "0", "1", "0.5"]))
        count = rng.choice([None, 12, 24, 48, 100])
    else:
        metric = rng.choice(ASCII_METRICS)
        delimiter = ""
        kind, op = "threshold", rng.choice([">=", "<=", "==", ">", "<"])
        value = Decimal(rng.choice(["0.92", "0.95", "0.90", "0.99", "1", "0"]))
        count = rng.choice([None, 12, 24, 48])
    unit = ""
    crit = Criterion(
        label=label,
        kind=kind,
        metric=metric,
        op=op,
        value=format(value.normalize(), "f"),
        unit=unit,
        count=count,
        scope=scope,
    )
    return crit, delimiter


def gen_extraction_corpus(n_cases: int = 20, *, seed: int = 20261109, undelimited_cjk: bool = False) -> list[ExtractedCase]:
    """Generate ``n_cases`` goal texts with generator-side ground truth."""
    rng = random.Random(seed)
    cases: list[ExtractedCase] = []
    for ci in range(n_cases):
        n_crit = rng.randint(2, 4)
        labels = [chr(ord("a") + i) for i in range(n_crit)]
        label_style = rng.choice(LABEL_STYLES + [None])
        pieces: list[str] = []
        gold: list[Criterion] = []
        for crit_label in labels:
            crit, delim = _random_criterion(rng, crit_label, undelimited_cjk=undelimited_cjk)
            unit_value, unit = _render_value(Decimal(crit.value), rng)
            crit = Criterion(
                label=crit.label,
                kind=crit.kind,
                metric=crit.metric,
                op=crit.op,
                value=format(Decimal(unit_value).normalize(), "f"),
                unit=unit,
                count=crit.count,
                scope=crit.scope,
            )
            pieces.append(_render_criterion(crit, rng, label_style, delim))
            gold.append(crit if label_style else Criterion("", crit.kind, crit.metric, crit.op, crit.value, crit.unit, crit.count, crit.scope))
        body = ""
        for i, piece in enumerate(pieces):
            body += piece
            if i < len(pieces) - 1:
                body += rng.choice(SEPARATORS)
                if rng.random() < 0.3:
                    body += " "
        text = "机器验收判据：" + body + "。"
        cases.append(ExtractedCase(case_id=f"ext-{ci:02d}", text=text, gold=gold))
    return cases


def field_scores(
    cases: Iterable[ExtractedCase],
    *,
    fields: Sequence[str] = ("label", "kind", "metric", "op", "value", "unit", "count", "scope"),
) -> dict[str, Any]:
    """Field-level precision/recall/F1 of the extractor against generator truth."""
    tp = fp = fn = 0
    errors: list[dict[str, Any]] = []
    for case in cases:
        pred = extract_criteria(case.text).criteria
        gold_triples = iter_field_triples(case.gold, fields)
        pred_triples = iter_field_triples(pred, fields)
        gold_set, pred_set = set(gold_triples), set(pred_triples)
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        if gold_set != pred_set:
            errors.append(
                {
                    "case_id": case.case_id,
                    "text": case.text,
                    "missing": sorted(map(str, gold_set - pred_set))[:8],
                    "extra": sorted(map(str, pred_set - gold_set))[:8],
                }
            )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_cases": len(list(cases)) if not isinstance(cases, list) else len(cases),
        "n_fields": len(fields),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_cases": errors,
    }


# --------------------------------------------------------------------------- #
# drift corpus
# --------------------------------------------------------------------------- #
MUTATIONS = (
    "unchanged_reformatted",
    "tightened_raise",
    "loosened_lower",
    "scope_count",
    "scope_desc",
    "criterion_dropped",
    "criterion_added",
    "op_family",
    "exact_value",
    "mapping_target",
)

#: Mutation -> expected drift class (the *dominant* class of the change).
EXPECTED_CLASS = {
    "unchanged_reformatted": "unchanged",
    "tightened_raise": "tightened",
    "loosened_lower": "loosened",
    "scope_count": "scope_changed",
    "scope_desc": "scope_changed",
    "criterion_dropped": "loosened",
    "criterion_added": "tightened",
    "op_family": "undefined",
    "exact_value": "undefined",
    "mapping_target": "undefined",
}

#: Mutations that constitute drift (must be flagged) -- fail-closed classes.
DRIFT_MUTATIONS = {"loosened_lower", "scope_count", "scope_desc", "criterion_dropped", "op_family", "exact_value", "mapping_target"}


@dataclass
class DriftCase:
    case_id: str
    frozen_goal: str
    receipt_goal: str
    mutation: str
    expected_class: str
    is_drift: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mutation": self.mutation,
            "expected_class": self.expected_class,
            "is_drift": self.is_drift,
            "frozen_goal": self.frozen_goal,
            "receipt_goal": self.receipt_goal,
        }


def _base_criteria(rng: random.Random, labels: Sequence[str]) -> list[Criterion]:
    """Frozen base: ``>=`` thresholds plus one mapping criterion (last)."""
    out: list[Criterion] = []
    for label in labels:
        scope = rng.choice(SCOPES)
        metric = rng.choice(["recall", "precision", "f1", "accuracy"])
        out.append(
            Criterion(
                label=label,
                kind="threshold",
                metric=metric,
                op=">=",
                value=rng.choice(["0.92", "0.95", "0.90"]),
                unit="",
                count=rng.choice([24, 24, 48]),
                scope=scope,
            )
        )
    out.append(Criterion(label="z", kind="mapping", metric="empty_goal", op="->", value="unregistered", unit="", count=None, scope="负控"))
    return out


def _threshold_indices(criteria: Sequence[Criterion]) -> list[int]:
    return [i for i, c in enumerate(criteria) if c.kind == "threshold" and c.op in (">=", "<=")]


def _with_value(c: Criterion, value: str) -> Criterion:
    return Criterion(c.label, c.kind, c.metric, c.op, value, c.unit, c.count, c.scope)


def _render_frozen(criteria: Sequence[Criterion], rng: random.Random) -> str:
    style = rng.choice(LABEL_STYLES)
    parts = [_render_criterion(c, rng, style, " " if c.kind == "threshold" and not c.metric.isascii() else "") for c in criteria]
    return "机器验收判据：" + "；".join(parts) + "。"


def _render_receipt(criteria: Sequence[Criterion], rng: random.Random, *, reformat: bool) -> str:
    if reformat:
        style = rng.choice([s for s in LABEL_STYLES if s]) if rng.random() < 0.5 else None
        parts = []
        for c in criteria:
            rendered = _render_criterion(c, rng, style, " " if not c.metric.isascii() else "")
            # equivalent surface forms: padded numbers, extra spaces
            rendered = rendered.replace("≥", ">=").replace("；", ";")
            parts.append(" " + rendered + " ")
        return "机器验收判据:" + "；".join(parts)
    style = rng.choice(LABEL_STYLES)
    return "机器验收判据：" + "；".join(
        _render_criterion(c, rng, style, " " if not c.metric.isascii() else "") for c in criteria
    ) + "。"


def _mutate(criteria: list[Criterion], mutation: str, rng: random.Random) -> tuple[list[Criterion], str]:
    """Apply one labelled mutation; returns (receipt criteria, receipt text)."""
    crit = list(criteria)
    thresholds = _threshold_indices(crit)
    if mutation == "unchanged_reformatted":
        return crit, _render_receipt(crit, rng, reformat=True)
    if mutation == "tightened_raise":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = _with_value(c, "0.85" if c.op == "<=" else "0.97")
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "loosened_lower":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = _with_value(c, "0.97" if c.op == "<=" else "0.85")
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "scope_count":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = Criterion(c.label, c.kind, c.metric, c.op, c.value, c.unit, 12 if c.count != 12 else 6, c.scope)
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "scope_desc":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = Criterion(c.label, c.kind, c.metric, c.op, c.value, c.unit, c.count, "缩小范围样本")
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "criterion_dropped":
        idx = thresholds[rng.randrange(len(thresholds))]
        crit.pop(idx)
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "criterion_added":
        crit.append(Criterion("y", "threshold", "boundary_cases", ">=", "0.98", "", None, "补充样本"))
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "op_family":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = Criterion(c.label, c.kind, c.metric, "<=", c.value, c.unit, c.count, c.scope)
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "exact_value":
        idx = thresholds[rng.randrange(len(thresholds))]
        c = crit[idx]
        crit[idx] = Criterion(c.label, c.kind, c.metric, "==", "0.93", c.unit, c.count, c.scope)
        return crit, _render_receipt(crit, rng, reformat=False)
    if mutation == "mapping_target":
        idx = next(i for i, c in enumerate(crit) if c.kind == "mapping")
        c = crit[idx]
        crit[idx] = Criterion(c.label, "mapping", c.metric, "->", "accepted", "", None, c.scope)
        return crit, _render_receipt(crit, rng, reformat=False)
    raise ValueError(f"unknown mutation {mutation!r}")


def gen_drift_cases(*, seed: int = 20261109) -> list[DriftCase]:
    """Generate the 24-case drift corpus (13 drift positives / 11 negatives)."""
    rng = random.Random(seed)
    plan = (
        ["unchanged_reformatted"] * 6
        + ["tightened_raise"] * 3
        + ["criterion_added"] * 2  # 11 negatives
        + ["loosened_lower"] * 3
        + ["criterion_dropped"] * 2
        + ["scope_count"] * 2
        + ["scope_desc"] * 2
        + ["op_family"] * 2
        + ["exact_value"] * 1
        + ["mapping_target"] * 1  # 13 positives
    )
    rng.shuffle(plan)
    cases: list[DriftCase] = []
    for i, mutation in enumerate(plan):
        labels = [chr(ord("a") + k) for k in range(rng.randint(2, 3))]
        base = _base_criteria(rng, labels)
        frozen_goal = _render_frozen(base, rng)
        mutated, receipt_text = _mutate(list(base), mutation, rng)
        cases.append(
            DriftCase(
                case_id=f"drift-{i:02d}",
                frozen_goal=frozen_goal,
                receipt_goal=receipt_text,
                mutation=mutation,
                expected_class=EXPECTED_CLASS[mutation],
                is_drift=mutation in DRIFT_MUTATIONS,
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# baseline + scoring
# --------------------------------------------------------------------------- #
def baseline_text_hash(frozen_goal: str, receipt_goal: str) -> bool:
    """Naive baseline: flag drift iff the goal *text* digest changed.

    Included to show what the structured audit buys: any reformatting, extra
    space or legitimate tightening trips this detector.
    """
    return hashlib.sha256(frozen_goal.encode()).hexdigest() != hashlib.sha256(receipt_goal.encode()).hexdigest()


def detection_metrics(predictions: Sequence[tuple[bool, str]], cases: Sequence[DriftCase]) -> dict[str, Any]:
    """Detection recall/precision plus per-class confusion vs generator truth."""
    tp = fp = fn = tn = 0
    confusion: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    for (is_drift_pred, klass_pred), case in zip(predictions, cases):
        if case.is_drift and is_drift_pred:
            tp += 1
        elif case.is_drift and not is_drift_pred:
            fn += 1
        elif not case.is_drift and is_drift_pred:
            fp += 1
        else:
            tn += 1
        row = confusion.setdefault(case.expected_class, {})
        row[klass_pred] = row.get(klass_pred, 0) + 1
        if klass_pred != case.expected_class or is_drift_pred != case.is_drift:
            mismatches.append(
                {"case_id": case.case_id, "mutation": case.mutation, "expected": case.expected_class, "predicted": klass_pred}
            )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_cases": len(cases),
        "n_positive": sum(1 for c in cases if c.is_drift),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_accuracy": (len(cases) - len(mismatches)) / len(cases) if cases else 0.0,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
        "mismatches": mismatches,
    }


__all__ = [
    "SCOPES",
    "MUTATIONS",
    "EXPECTED_CLASS",
    "DRIFT_MUTATIONS",
    "ExtractedCase",
    "DriftCase",
    "gen_extraction_corpus",
    "field_scores",
    "gen_drift_cases",
    "baseline_text_hash",
    "detection_metrics",
]