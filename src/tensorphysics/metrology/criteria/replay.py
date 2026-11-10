"""Replay re-judge: re-run historical criteria through the adaptive family.

For every harvested record the re-judge produces, in one row:

* ``baseline_*``  -- the current ``relative error < tau`` criterion applied to the same
  numbers (including its naive division, i.e. ``inf`` when ``y_ref == 0``);
* ``degeneracy``  -- the three-type classification (see :mod:`.degeneracy`);
* ``adaptive_*``  -- the adaptive verdict and the *decisive alternative value*, so the
  conclusion is recomputable by a third party;
* ``flip``        -- whether the verdict changed between baseline and adaptive family;
* ``downgrade``   -- whether a previously *decided* verdict became ``WITHHELD`` because the
  evidence needed for a bounded metric is not in the artifact.

The output is a machine-readable "criterion degeneracy list" plus KPI numbers; nothing is
written back into the source artifacts.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .adaptive import deviation_fraction, direction, judge
from .degeneracy import DEGENERACY_TYPES, Degeneracy, analyze
from .harvest import HarvestStats, build_corpus
from .model import FAIL, LOG_RATIO, PASS, RELATIVE, WITHHELD, CriterionRecord, Verdict, finite, sign

__all__ = [
    "ReplayRow",
    "ReplayResult",
    "baseline_verdict",
    "replay",
    "replay_repository",
    "case_studies",
    "summarize",
    "DEGRADED_MIN_FRACTION",
    "CORPUS_MIN_RECORDS",
]

#: acceptance floors of P266W2-N09
DEGRADED_MIN_FRACTION = 0.05
CORPUS_MIN_RECORDS = 200


def baseline_verdict(record: CriterionRecord) -> Verdict:
    """The current criterion: naive ``relative error < tau`` (no degeneracy awareness).

    ``y_ref == 0`` with ``y != 0`` produces ``+inf`` -> ``FAIL`` exactly as a naive
    ``abs(y - y_ref) / abs(y_ref) < tol`` implementation would behave; ``y_ref == 0`` and
    ``y == 0`` produces ``0/0`` -> ``NaN`` -> the comparison is False, i.e. ``FAIL``
    unless the artifact recorded a numeric error.  This reproduces the pre-registered
    baseline honestly, including its pathologies.
    """
    tol = finite(record.parsed.tolerance)
    value = record.derived_relative_error()
    note = ""
    if value is None:
        if record.reported_error is not None:
            value = record.reported_error
            note = "recorded relative error only"
        elif record.parsed.tolerance_kind == "absolute":
            v = finite(record.y) if record.y is not None else None
            value = v - finite(record.y_ref) if (record.y is not None and record.y_ref is not None) else None
            value = abs(value) if value is not None else None
            note = "absolute criterion evaluated as recorded"
        else:
            return Verdict(WITHHELD, RELATIVE, None, tol, None, ("no numeric evidence in artifact",))
    if tol is None:
        return Verdict(WITHHELD, RELATIVE, value, None, None, ("no parseable tolerance",))
    if value is None:
        return Verdict(WITHHELD, RELATIVE, None, tol, None, ("no numeric evidence in artifact",))
    if isinstance(value, float) and math.isnan(value):
        return Verdict(FAIL, RELATIVE, value, tol, None, ("NaN relative error (0/0) -> comparison False",))
    status = PASS if value < tol else FAIL
    reasons = ("y_ref at zero: relative error unbounded (inf)" if value == math.inf else "",) if value == math.inf else ()
    return Verdict(status, record.parsed.metric or RELATIVE, value, tol, direction(record.y, record.y_ref), reasons + ((note,) if note else ()))


def _alternative(verdict: Verdict) -> dict[str, Any]:
    return {
        "metric": verdict.metric,
        "value": verdict.value,
        "threshold": verdict.threshold,
        "status": verdict.status,
        "formula": next((e.formula for e in verdict.evaluations if e.metric == verdict.metric), ""),
    }


def _alt_direction(record: CriterionRecord, verdict: Verdict) -> int | None:
    """Direction implied by the decisive alternative metric (log ratio is sign-carrying)."""
    y, r = finite(record.y), finite(record.y_ref)
    if y is None or r is None:
        return None
    if verdict.metric == LOG_RATIO and y != 0 and r != 0 and (y > 0) == (r > 0):
        return sign(abs(y) - abs(r)) * (sign(r) or 1)
    d = direction(y, r)
    return d


@dataclass
class ReplayRow:
    record: CriterionRecord
    degeneracy: Degeneracy
    baseline: Verdict
    adaptive: Verdict
    deviation: float | None

    # -- derived properties ------------------------------------------------------------
    @property
    def degraded(self) -> bool:
        return self.degeneracy.flagged

    @property
    def flip(self) -> bool:
        if not (self.baseline.decided and self.adaptive.decided):
            return False
        return self.baseline.status != self.adaptive.status

    @property
    def downgrade(self) -> bool:
        return self.baseline.decided and self.adaptive.status == WITHHELD

    @property
    def upgrade(self) -> bool:
        """Baseline could not decide, the adaptive family can (evidence recovered)."""
        return (not self.baseline.decided) and self.adaptive.decided

    @property
    def verdict_change(self) -> bool:
        return self.flip or self.downgrade or self.upgrade

    @property
    def direction_consistent(self) -> bool:
        alt = _alt_direction(self.record, self.adaptive)
        if alt is None or self.adaptive.direction is None:
            return False
        return alt == self.adaptive.direction

    @property
    def has_numeric_alternative(self) -> bool:
        return self.adaptive.value is not None and self.adaptive.status != WITHHELD or self.adaptive.value is not None

    def as_dict(self, compact: bool = False) -> dict[str, Any]:
        payload = {
            "rid": self.record.rid,
            "source": self.record.source,
            "measurand": self.record.measurand,
            "criterion": self.record.criterion,
            "metric": self.record.parsed.metric,
            "tolerance": self.record.parsed.tolerance,
            "tolerance_kind": self.record.parsed.tolerance_kind,
            "unit": self.record.unit,
            "y": self.record.y,
            "y_ref": self.record.y_ref,
            "reported_error": self.record.reported_error,
            "scale": self.degeneracy.scale,
            "noise": self.degeneracy.noise,
            "old_verdict": self.record.old_verdict,
            "degeneracy": self.degeneracy.as_dict(),
            "baseline": self.baseline.as_dict(),
            "adaptive": self.adaptive.as_dict(),
            "alternative": _alternative(self.adaptive),
            "deviation_fraction": self.deviation,
            "flip": self.flip,
            "downgrade": self.downgrade,
            "upgrade": self.upgrade,
            "verdict_change": self.verdict_change,
            "direction_consistent": self.direction_consistent,
            "degraded": self.degraded,
        }
        if compact:
            for key in ("baseline", "adaptive"):
                payload[key] = {k: v for k, v in payload[key].items() if k != "evaluations"}
        return payload


@dataclass
class ReplayResult:
    rows: list[ReplayRow]
    stats: HarvestStats | None = None

    # -- KPIs --------------------------------------------------------------------------
    def kpi(self) -> dict[str, Any]:
        n = len(self.rows)
        degraded = [r for r in self.rows if r.degraded]
        provable = [r for r in self.rows if r.degeneracy.provable]
        undecidable = [r for r in self.rows if r.degeneracy.undecidable and not r.degeneracy.provable]
        flips = [r for r in self.rows if r.flip]
        downgrades = [r for r in self.rows if r.downgrade]
        upgrades = [r for r in self.rows if r.upgrade]
        with_alt = [r for r in degraded if r.adaptive.value is not None]
        by_class = {k: sum(1 for r in self.rows if k in r.degeneracy.classes) for k in DEGENERACY_TYPES}
        by_class_undecidable = {k: sum(1 for r in self.rows if k in r.degeneracy.undecidable) for k in DEGENERACY_TYPES}
        baseline_pass = sum(1 for r in self.rows if r.baseline.status == PASS)
        adaptive_pass = sum(1 for r in self.rows if r.adaptive.status == PASS)
        baseline_fail = sum(1 for r in self.rows if r.baseline.status == FAIL)
        adaptive_fail = sum(1 for r in self.rows if r.adaptive.status == FAIL)
        withheld = sum(1 for r in self.rows if r.adaptive.status == WITHHELD)
        true_positive_rule = sum(1 for r in provable if r.deviation is not None and r.deviation > 0.5)
        return {
            "n_records": n,
            "n_degraded": len(degraded),
            "degraded_fraction": (len(degraded) / n) if n else 0.0,
            "n_degraded_provable": len(provable),
            "n_degraded_undecidable_only": len(undecidable),
            "degenerate_by_class": by_class,
            "undecidable_by_class": by_class_undecidable,
            "n_degraded_with_numeric_alternative": len(with_alt),
            "alternative_coverage": (len(with_alt) / len(degraded)) if degraded else 0.0,
            "n_flips": len(flips),
            "n_downgrades_to_withheld": len(downgrades),
            "n_upgrades": len(upgrades),
            "n_verdict_changes": len(flips) + len(downgrades) + len(upgrades),
            "n_with_outcome": sum(1 for r in self.rows if r.record.reported_error is not None or r.record.has_pair()),
            "n_criterion_only": sum(1 for r in self.rows if r.record.reported_error is None and not r.record.has_pair()),
            "baseline": {"PASS": baseline_pass, "FAIL": baseline_fail,
                         "WITHHELD": n - baseline_pass - baseline_fail},
            "adaptive": {"PASS": adaptive_pass, "FAIL": adaptive_fail, "WITHHELD": withheld},
            "n_provable_deviations_above_50pct": true_positive_rule,
            "n_direction_consistent": sum(1 for r in self.rows if r.direction_consistent),
            "n_direction_checked": sum(1 for r in self.rows if r.adaptive.direction is not None and r.adaptive.status != WITHHELD),
            "corpus_min_records": CORPUS_MIN_RECORDS,
            "degraded_min_fraction": DEGRADED_MIN_FRACTION,
            "meets_corpus_size": n >= CORPUS_MIN_RECORDS,
            "meets_degraded_fraction": (len(degraded) / n if n else 0.0) >= DEGRADED_MIN_FRACTION,
        }

    def degraded_rows(self) -> list[ReplayRow]:
        return [r for r in self.rows if r.degraded]

    def as_dict(self, compact: bool = False) -> dict[str, Any]:
        return {
            "kpi": self.kpi(),
            "harvest": self.stats.as_dict() if self.stats else {},
            "rows": [r.as_dict(compact=compact) for r in self.rows],
        }


def replay(records: Sequence[CriterionRecord]) -> ReplayResult:
    """Re-judge a corpus with baseline and adaptive families."""
    classified = analyze(records)
    fams = tuple({d.family for _, d in classified if d.family})
    # multi-metric families were resolved inside analyze(); reconstruct for judge()
    from .degeneracy import multi_metric_families as _mmf
    multi = tuple(_mmf(records))
    rows: list[ReplayRow] = []
    fam_scales: dict[str, float] = {}
    buckets: dict[str, list[float]] = {}
    for rec in records:
        r = finite(rec.y_ref)
        if r is not None and r != 0.0:
            buckets.setdefault(rec.family_key, []).append(abs(r))
    for fam, vals in buckets.items():
        vals.sort()
        n = len(vals)
        fam_scales[fam] = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    for rec, deg in classified:
        adaptive = judge(rec, deg, family_scale=fam_scales.get(rec.family_key), multi_metric_families=multi)
        rows.append(
            ReplayRow(
                record=rec,
                degeneracy=deg,
                baseline=baseline_verdict(rec),
                adaptive=adaptive,
                deviation=deviation_fraction(rec.y, rec.y_ref, scale=deg.scale),
            )
        )
    return ReplayResult(rows=rows)


def replay_repository(root: str, *, subdirs: Iterable[str] | None = None) -> ReplayResult:
    records, stats = build_corpus(root, subdirs=subdirs)
    result = replay(records)
    result.stats = stats
    return result


# --------------------------------------------------------------------------------------
# case studies (acceptance item c)
# --------------------------------------------------------------------------------------
def _case(origin: str, row: ReplayRow, note: str) -> dict[str, Any]:
    rec = row.record
    payload = row.as_dict()
    payload.update({
        "origin": origin,
        "note": note,
        "deviation_gt_50pct": bool(row.deviation is not None and row.deviation > 0.5),
        "must_be_explicit_fail": bool(row.deviation is not None and row.deviation > 0.5),
        "explicit_fail_satisfied": (row.adaptive.status == FAIL) if (row.deviation or 0) > 0.5 else True,
        "direction_consistent": row.direction_consistent,
    })
    return payload


def case_studies(result: ReplayResult, extra_rows: Sequence[tuple[str, ReplayRow]] = (),
                 family_merges: Sequence[Mapping[str, Any]] = (), n: int = 4) -> dict[str, Any]:
    """Four typical degeneration cases with recomputable alternative numbers.

    Roles (deterministic selection, one record each):

    ``corpus_metric_misparse``
        a real artifact whose criterion is a *ratio* written with vertical bars
        (``|a - b| / b < 2%``); the recorded deviation is reproduced under the adaptive
        family and, if it exceeds 50 % of the engineering scale, it must be an explicit
        FAIL;
    ``sign_flip``
        a T2 control: opposite signs make the relative criterion structurally unbounded;
    ``multi_metric_merge``
        a T3 control where one metric passes and another fails -- the family merge must
        return a single FAIL instead of a "partial pass";
    ``legit_zero_reference``
        a zero reference that must still PASS, proving the family does not blanket-fail
        near-zero references.
    """
    pool: list[tuple[str, ReplayRow]] = [("corpus", r) for r in result.rows] + list(extra_rows)
    used: set[str] = set()
    cases: list[dict[str, Any]] = []
    merges = {m.get("family"): m for m in family_merges}

    def _one(role: str, rows: Iterable[ReplayRow], note: str) -> None:
        for row in rows:
            if row.record.rid not in used:
                used.add(row.record.rid)
                origin = "corpus" if row in result.rows else "controls"
                payload = _case(origin, row, note)
                payload["role"] = role
                fam = row.degeneracy.family
                if fam in merges:
                    payload["family_merge"] = {
                        "members": merges[fam]["members"],
                        "merged_status": merges[fam]["merged_status"],
                        "partial_pass_risk": merges[fam]["partial_pass_risk"],
                    }
                cases.append(payload)
                return

    ratio_rows = [r for _o, r in pool
                  if r.record.has_pair() and r.record.parsed.metric == RELATIVE]
    ratio_rows.sort(key=lambda r: -(r.deviation or 0.0))
    _one("corpus_metric_misparse", ratio_rows,
         "ratio criterion written with vertical bars: the string parser must keep it relative, "
         "otherwise a " + "large deviation is silently compared against an absolute band")

    sign_rows = [r for _o, r in pool
                 if r.degeneracy.has("T2_sign_flip") and (r.record.y_ref or 0) != 0.0]
    _one("sign_flip", sign_rows,
         "T2: opposite signs saturate the relative metric at 100 %, the adaptive family keeps the sign and FAILs")

    multi_rows = [r for _o, r in pool
                  if r.degeneracy.has("T3_multi_metric_no_priority")
                  and r.degeneracy.family in merges]
    multi_rows.sort(key=lambda r: (r.adaptive.status != "PASS",
                                   -(merges[r.degeneracy.family]["merged_status"] == "FAIL")))
    _one("multi_metric_merge", multi_rows,
         "T3: one metric passes and another fails; the family merge returns a single FAIL (no partial pass)")

    zero_rows = [r for _o, r in pool if r.degeneracy.has("T1_zero_reference") and r.adaptive.status == "PASS"]
    zero_rows.sort(key=lambda r: -(r.record.scale or 0.0))
    _one("legit_zero_reference", zero_rows,
         "legitimate zero reference: the deviation stays inside the engineering band -> PASS (no false alarm)")

    for _o, row in pool:
        if len(cases) >= n:
            break
        if row.record.rid not in used and row.degeneracy.flagged:
            _one("additional", [row], "additional degraded record")

    cases = cases[:n]
    return {
        "n_cases": len(cases),
        "roles": [c["role"] for c in cases],
        "n_direction_consistent": sum(1 for c in cases if c["direction_consistent"]),
        "n_deviation_gt_50pct": sum(1 for c in cases if c["deviation_gt_50pct"]),
        "n_explicit_fail_satisfied": sum(1 for c in cases if c["explicit_fail_satisfied"]),
        "cases": cases,
    }


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` so artifacts are strict JSON.

    ``inf`` is a legitimate *value* of a degenerate relative error but not a legitimate
    JSON number; the verdict it produced is preserved in ``status``/``reasons``.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def summarize(root: str, *, out_path: str | None = None) -> dict[str, Any]:
    """Full re-judge of the repository, optionally persisted as a JSON artifact."""
    result = replay_repository(root)
    from .controls import evaluate_controls  # local import keeps module import light

    controls = evaluate_controls()
    extra = controls.pop("_rows", [])
    cs = case_studies(result, extra_rows=extra, family_merges=controls.get("family_merges", ()))
    payload = {
        "schema": "tensorphysics.metrology.criteria.replay/1",
        "root": os.path.abspath(root),
        "kpi": result.kpi(),
        "harvest": result.stats.as_dict() if result.stats else {},
        "controls": controls,
        "case_studies": cs,
        "degeneracy_list": [r.as_dict() for r in result.degraded_rows()],
        "rows": [r.as_dict(compact=True) for r in result.rows],
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(json_safe(payload), fh, ensure_ascii=False, indent=2, allow_nan=False)
    return payload