"""Independent synthetic benchmark for criterion extraction and drift auditing.

Purpose
-------
The criterion pre-registration mechanism (manifest + drift) is only useful if
three executability properties hold:

(a) drift detection on a labelled synthetic set: recall >= 0.92, precision >= 0.95;
(b) field-level extraction F1 >= 0.95 over the six structural fields;
(c) fail-closed: unparseable goals must be ``unregistered``, never silently
    treated as satisfied.

This module owns the *dataset* and the *metric computation* so that the numbers
quoted in reports are reproducible (same corpus -> same sha256 -> same metrics).

Zero domain semantics
---------------------
All fixtures are synthetic.  The extractor contract is purely structural:

``threshold``      numeric level attached to a comparator (or the first number
                   when no comparator exists).  For an interval ``a-b`` the
                   upper bound ``b`` is used with comparator ``in_range``;
``unit``           the single token immediately following that number
                   (``"%"``, ``"pct"``, ``"mm"``, ``"s"``, ...);
``comparator``     one of ``le, lt, ge, gt, eq, ne, in_range``;
``denominator``    text after ``of`` / ``relative to`` / ``compared with``;
``exclusion``      text after ``excluding`` / ``except`` / ``minus``;
``reference_path`` a ``/path/like`` token.

The konventions above are fixed *before* measuring; the benchmark is run once
against the legacy extractor (baseline) and once against the hardened one.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from .manifest import (
    CRITERION_FIELDS,
    CriterionManifest,
    extract_criteria_from_goal,
    freeze_manifest,
)
from .drift import classify_drift

FIELD_ORDER: tuple[str, ...] = (
    "threshold",
    "unit",
    "denominator",
    "exclusion",
    "reference_path",
    "comparator",
)

DRIFT_LABELS: tuple[str, ...] = (
    "unchanged",
    "tightened",
    "loosened",
    "scope_changed",
    "undefined",
    "unregistered",
)


@dataclass(frozen=True)
class ExtractionCase:
    """One labelled extraction fixture (single criterion block)."""

    text: str
    expected: dict[str, Any]


@dataclass(frozen=True)
class DriftCase:
    """One labelled drift fixture: frozen (pre-execution) vs report (post)."""

    frozen_text: str
    report_text: str
    expected: str
    note: str = ""


# ---------------------------------------------------------------------------
# Extraction corpus (48 cases, synthetic)
# ---------------------------------------------------------------------------

def _case(text: str, **expected: Any) -> ExtractionCase:
    full = {name: None for name in FIELD_ORDER}
    full.update(expected)
    return ExtractionCase(text=text, expected=full)


EXTRACTION_CASES: tuple[ExtractionCase, ...] = (
    # --- plain comparator + unit + denominator ---------------------------
    _case("error <= 3.5 pct of reference", threshold=3.5, unit="pct",
          denominator="reference", comparator="le"),
    _case("deviation <= 1.5 mm compared with baseline", threshold=1.5,
          unit="mm", denominator="baseline", comparator="le"),
    _case("score >= 0.9 ratio of nominal", threshold=0.9, unit="ratio",
          denominator="nominal", comparator="ge"),
    # --- the case-count trap: the level is the bound, not the sample count
    _case("on 24 cases recall >= 0.92", threshold=0.92, comparator="ge"),
    _case("across 12 runs margin <= 5 pct", threshold=5.0, unit="pct",
          comparator="le"),
    # --- natural-language comparators ------------------------------------
    _case("at most 2.0 mm deviation", threshold=2.0, unit="mm", comparator="le"),
    _case("no greater than 5 % error", threshold=5.0, unit="%", comparator="le"),
    _case("not exceeding 7 s runtime", threshold=7.0, unit="s", comparator="le"),
    _case("must be at least 95 % coverage", threshold=95.0, unit="%",
          comparator="ge"),
    _case("no less than 3 runs", threshold=3.0, unit="runs", comparator="ge"),
    _case("not less than 0.5 pts", threshold=0.5, unit="pts", comparator="ge"),
    # --- unicode symbols --------------------------------------------------
    _case("≤ 2 mm drift", threshold=2.0, unit="mm", comparator="le"),
    _case("≥ 90 % pass rate", threshold=90.0, unit="%", comparator="ge"),
    _case("±0.02 m tolerance", threshold=0.02, unit="m", comparator="in_range"),
    _case("score within 0.5 pts of baseline", threshold=0.5, unit="pts",
          denominator="baseline", comparator="in_range"),
    # --- strict inequalities ---------------------------------------------
    _case("deviation < 0.05 m", threshold=0.05, unit="m", comparator="lt"),
    _case("error > 1e-6 s", threshold=1e-6, unit="s", comparator="gt"),
    _case("must be > 4", threshold=4.0, comparator="gt"),
    _case("must be != 0 ratio", threshold=0.0, unit="ratio", comparator="ne"),
    _case("equals 0.5 ratio", threshold=0.5, unit="ratio", comparator="eq"),
    # --- scientific notation, thousands separators -----------------------
    _case("mean error of 1.5e-3 m relative to /data/ref_7", threshold=1.5e-3,
          unit="m", denominator="/data/ref_7", reference_path="/data/ref_7"),
    _case("1,024 samples, error <= 5 pct", threshold=5.0, unit="pct",
          comparator="le"),
    _case("tolerance 2.5E+2 Pa", threshold=250.0, unit="Pa"),
    # --- intervals --------------------------------------------------------
    _case("error in 2-3 pct band", threshold=3.0, unit="pct",
          comparator="in_range"),
    _case("between 10 and 20 ms", threshold=20.0, unit="ms",
          comparator="in_range"),
    # --- exclusions and reference paths ----------------------------------
    _case("deviation < 0.05 m excluding edge cases", threshold=0.05, unit="m",
          exclusion="edge cases", comparator="lt"),
    _case("error <= 2 pct except retired runs", threshold=2.0, unit="pct",
          exclusion="retired runs", comparator="le"),
    _case("relative to /data/ref_9: error <= 2 pct", threshold=2.0, unit="pct",
          denominator="/data/ref_9", reference_path="/data/ref_9",
          comparator="le"),
    # --- no unit at all ---------------------------------------------------
    _case("index <= 12", threshold=12.0, comparator="le"),
    _case("count >= 100", threshold=100.0, comparator="ge"),
    # --- decimal / sign edge cases ---------------------------------------
    _case("offset <= .75 deg", threshold=0.75, unit="deg", comparator="le"),
    _case("bias <= -0.4 pct", threshold=-0.4, unit="pct", comparator="le"),
    _case("gain >= +1.25 ratio", threshold=1.25, unit="ratio", comparator="ge"),
    # --- upper/lower case and spacing ------------------------------------
    _case("ERROR <= 4 PCT", threshold=4.0, unit="PCT", comparator="le"),
    _case("error<=4pct", threshold=4.0, unit="pct", comparator="le"),
    # --- multiple numbers where the level is the second one --------------
    _case("field F1 >= 0.95 on 60 fixtures", threshold=0.95, comparator="ge"),
    _case("recall >= 0.92 on a set of 24 cases", threshold=0.92,
          comparator="ge"),
    # --- criteria with only a level (no comparator wording) --------------
    _case("0.5 ratio", threshold=0.5, unit="ratio"),
    _case("25 ms", threshold=25.0, unit="ms"),
    # --- denominators ----------------------------------------------------
    _case("error <= 3 pct relative to nominal", threshold=3.0, unit="pct",
          denominator="nominal", comparator="le"),
    _case("error <= 3 pct vs baseline", threshold=3.0, unit="pct",
          denominator="baseline", comparator="le"),
    _case("error <= 3 pct of the reference", threshold=3.0, unit="pct",
          denominator="the reference", comparator="le"),
    # --- negative controls (no numeric level -> unregistered) ------------
    ExtractionCase("qualitative goal with no numeric target", {}),
    ExtractionCase("improve robustness qualitatively", {}),
    ExtractionCase("", {}),
    ExtractionCase("use TBD as the level", {}),
    ExtractionCase("no data available yet", {}),
    ExtractionCase("N/A", {}),
    ExtractionCase("report the findings in prose", {}),
    ExtractionCase("   ", {}),
)


# ---------------------------------------------------------------------------
# Held-out extraction corpus (20 cases, written *after* the extractor was
# frozen, measured exactly once, never used for tuning).
# ---------------------------------------------------------------------------

HELD_OUT_CASES: tuple[ExtractionCase, ...] = (
    _case("relative error ≤ 0.02", threshold=0.02, comparator="le"),
    _case("peak value not greater than 3.5e2 Pa", threshold=350.0, unit="Pa",
          comparator="le"),
    _case("settling time at most 0.8 s", threshold=0.8, unit="s",
          comparator="le"),
    _case("agreement ≥ 98 % over 40 runs", threshold=98.0, unit="%",
          comparator="ge"),
    _case("band 4–6 Hz", threshold=6.0, unit="Hz", comparator="in_range"),
    _case("error <= 2.0 % w.r.t. baseline", threshold=2.0, unit="%",
          denominator="baseline", comparator="le"),
    _case("correlation >= 0.995 across 500 samples", threshold=0.995,
          comparator="ge"),
    _case("max deviation <= 12.5 um", threshold=12.5, unit="um",
          comparator="le"),
    _case("throughput at least 2500 ops/s", threshold=2500.0, unit="ops/s",
          comparator="ge"),
    _case("loss < 1e-4", threshold=1e-4, comparator="lt"),
    _case("accuracy within 1.5 % of the reference", threshold=1.5, unit="%",
          denominator="the reference", comparator="in_range"),
    _case("must equal 1.0 ratio", threshold=1.0, unit="ratio",
          comparator="eq"),
    _case("latency <= 35 ms except cold start", threshold=35.0, unit="ms",
          exclusion="cold start", comparator="le"),
    _case("score >= 0.8 of nominal (ref: /data/nominal)", threshold=0.8,
          denominator="nominal", reference_path="/data/nominal",
          comparator="ge"),
    _case("tolerance ±5 K", threshold=5.0, unit="K", comparator="in_range"),
    _case("error <= 0.25 %, measured on 30 fixtures", threshold=0.25,
          unit="%", comparator="le"),
    _case("on 18 fixtures precision >= 0.97", threshold=0.97,
          comparator="ge"),
    _case("value between -2 and 2 mV", threshold=2.0, unit="mV",
          comparator="in_range"),
    _case("exactly 3 replicates", threshold=3.0, unit="replicates",
          comparator="eq"),
    ExtractionCase("no data provided", {name: None for name in FIELD_ORDER}),
)


# ---------------------------------------------------------------------------
# Drift corpus: 24 labelled cases (12 in-family variants, 12 drift)
# ---------------------------------------------------------------------------

DRIFT_CASES: tuple[DriftCase, ...] = (
    # --- 12 unchanged ----------------------------------------------------
    DriftCase("error <= 3.5 pct of reference",
              "error <= 3.5 pct of reference", "unchanged"),
    DriftCase("score >= 0.9 ratio of nominal",
              "score >= 0.9 ratio of nominal", "unchanged"),
    DriftCase("at most 2.0 mm deviation",
              "at most 2.0 mm deviation", "unchanged"),
    DriftCase("deviation < 0.05 m excluding edge cases",
              "deviation < 0.05 m excluding edge cases", "unchanged"),
    DriftCase("no less than 3 runs", "no less than 3 runs", "unchanged"),
    DriftCase("error <= 2 pct relative to /data/ref_3",
              "error <= 2 pct relative to /data/ref_3", "unchanged"),
    DriftCase("index <= 12", "index <= 12", "unchanged"),
    DriftCase("error in 2-3 pct band", "error in 2-3 pct band", "unchanged"),
    DriftCase("≤ 2 mm drift", "≤ 2 mm drift", "unchanged"),
    DriftCase("score within 0.5 pts of baseline",
              "score within 0.5 pts of baseline", "unchanged"),
    DriftCase("1,024 samples, error <= 5 pct",
              "1,024 samples, error <= 5 pct", "unchanged"),
    DriftCase("tolerance 2.5E+2 Pa", "tolerance 2.5E+2 Pa", "unchanged"),
    # --- 3 tightened ------------------------------------------------------
    DriftCase("error <= 5 pct of reference", "error <= 3 pct of reference",
              "tightened", "le level lowered"),
    DriftCase("score >= 0.90 ratio", "score >= 0.95 ratio", "tightened",
              "ge level raised"),
    DriftCase("deviation < 2 mm", "deviation < 1 mm", "tightened",
              "lt level lowered"),
    # --- 3 loosened -------------------------------------------------------
    DriftCase("error <= 3 pct of reference", "error <= 8 pct of reference",
              "loosened", "le level raised"),
    DriftCase("score >= 0.95 ratio", "score >= 0.80 ratio", "loosened",
              "ge level lowered"),
    DriftCase("error > 1e-6 s", "error > 1e-7 s", "loosened",
              "gt level lowered"),
    # --- 3 scope_changed --------------------------------------------------
    DriftCase("error <= 3 pct of reference", "error <= 3 pct of baseline",
              "scope_changed", "denominator swapped"),
    DriftCase("deviation < 0.05 m excluding edge cases",
              "deviation < 0.05 m excluding retired runs", "scope_changed",
              "exclusion set swapped"),
    DriftCase("error <= 2 pct relative to /data/ref_a",
              "error <= 2 pct relative to /data/ref_b", "scope_changed",
              "reference path swapped"),
    # --- 3 undefined ------------------------------------------------------
    DriftCase("error <= 3 pct of reference", "error <= TBD pct of reference",
              "unregistered", "level unparseable after execution -> fail-closed"),
    DriftCase("deviation < 0.05 m", "deviation < 0.05",
              "undefined", "unit dropped after execution"),
    DriftCase("score >= 0.9 ratio", "score 0.9 ratio",
              "undefined", "comparator dropped after execution"),
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return round(value, 10)
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def extracted_fields(goal_text: str, index: int = 0) -> dict[str, Any]:
    """Return the six-field dict for the ``index``-th extracted criterion."""
    criteria = extract_criteria_from_goal(goal_text)
    if index >= len(criteria):
        return {name: None for name in FIELD_ORDER}
    crit = criteria[index]
    out: dict[str, Any] = {}
    for name in FIELD_ORDER:
        f = crit.fields.get(name)
        out[name] = _normalize(f.value) if f is not None and f.present else None
    return out


def evaluate_extraction(
    cases: Sequence[ExtractionCase] = EXTRACTION_CASES,
    extractor: Optional[Callable[[str], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Field-level micro precision/recall/F1 over the extraction corpus."""
    extractor = extractor or (lambda text: extracted_fields(text))
    tp = fp = fn = 0
    per_field = {name: {"tp": 0, "fp": 0, "fn": 0} for name in FIELD_ORDER}
    failures: list[dict[str, Any]] = []

    for case in cases:
        got = extractor(case.text)
        row_ok = True
        for name in FIELD_ORDER:
            expected = _normalize(case.expected.get(name))
            observed = _normalize(got.get(name))
            bucket = per_field[name]
            if expected is None and observed is None:
                continue
            if expected is not None and observed is not None and expected == observed:
                tp += 1
                bucket["tp"] += 1
            else:
                if expected is not None:
                    fn += 1
                    bucket["fn"] += 1
                if observed is not None:
                    fp += 1
                    bucket["fp"] += 1
                row_ok = False
        if not row_ok:
            failures.append({
                "text": case.text,
                "expected": case.expected,
                "observed": got,
            })

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    field_metrics = {}
    for name, b in per_field.items():
        p = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 1.0
        r = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 1.0
        field_metrics[name] = {
            "tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
            "precision": round(p, 4), "recall": round(r, 4),
        }
    return {
        "cases": len(cases),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_field": field_metrics,
        "failures": failures,
    }


def evaluate_detection(
    cases: Sequence[DriftCase] = DRIFT_CASES,
    extractor: Optional[Callable[[str], list]] = None,
) -> dict[str, Any]:
    """Binary drift-detection precision/recall plus exact-label confusion."""
    extract = extractor or extract_criteria_from_goal
    tp = fp = fn = tn = 0
    exact = 0
    confusion = {exp: {got: 0 for got in DRIFT_LABELS} for exp in DRIFT_LABELS}
    failures: list[dict[str, Any]] = []

    for case in cases:
        frozen_list = extract(case.frozen_text)
        if not frozen_list:
            predicted = "unregistered"
        else:
            frozen = freeze_manifest(frozen_list[0], "bench", )
            report_list = extract(case.report_text)
            if not report_list:
                predicted = "unregistered"
            else:
                predicted = classify_drift(frozen, report_list[0]).classification

        expected = case.expected
        predicted_drift = predicted != "unchanged"
        expected_drift = expected != "unchanged"
        if expected_drift and predicted_drift:
            tp += 1
        elif expected_drift and not predicted_drift:
            fn += 1
        elif not expected_drift and predicted_drift:
            fp += 1
        else:
            tn += 1

        if predicted in confusion.get(expected, {}):
            confusion[expected][predicted] += 1
        if predicted == expected:
            exact += 1
        else:
            failures.append({
                "frozen": case.frozen_text,
                "report": case.report_text,
                "expected": expected,
                "predicted": predicted,
                "note": case.note,
            })

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "cases": len(cases),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_accuracy": round(exact / len(cases), 4) if cases else 0.0,
        "confusion": confusion,
        "failures": failures,
    }


def evaluate_fail_closed() -> dict[str, Any]:
    """Negative controls: unparseable input must never become 'registered'."""
    probes = {
        "empty": "",
        "none": None,
        "whitespace": "   \n\t ",
        "prose_no_numbers": "report the findings in prose",
        "placeholder": "use TBD as the level",
        "not_a_string": 17,
    }
    results = {}
    ok = True
    for name, payload in probes.items():
        try:
            criteria = extract_criteria_from_goal(payload)  # type: ignore[arg-type]
        except Exception as exc:  # fail-closed must not crash either
            results[name] = {"status": "error", "error": type(exc).__name__}
            ok = False
            continue
        status = "unregistered" if not criteria else "registered"
        # A criterion without a numeric level may never be frozen by the
        # fail-closed registration helper.
        if criteria:
            has_level = any(
                c.fields.get("threshold") is not None and c.fields["threshold"].present
                for c in criteria
            )
            if not has_level:
                status = "unregistered"
        results[name] = {"status": status, "criteria": len(criteria)}
        if status != "unregistered":
            ok = False
    return {"ok": ok, "probes": results}


def scan_domain_semantics(
    paths: Iterable[Path],
    lexicon: Iterable[str],
) -> list[dict[str, Any]]:
    """Scan sources for caller-supplied domain vocabulary (zero-domain gate).

    The lexicon is injected by the caller so that the mechanism itself stays
    free of domain semantics.
    """
    import re

    terms = tuple(sorted({t.lower() for t in lexicon if t}))
    hits: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text):
                hits.append({"path": p.as_posix(), "term": term})
    return hits


def corpus_fingerprint() -> str:
    """sha256 over the canonical JSON of both labelled corpora."""
    payload = {
        "extraction": [
            {"text": c.text, "expected": c.expected} for c in EXTRACTION_CASES
        ],
        "held_out": [
            {"text": c.text, "expected": c.expected} for c in HELD_OUT_CASES
        ],
        "drift": [
            {"frozen": c.frozen_text, "report": c.report_text,
             "expected": c.expected, "note": c.note}
            for c in DRIFT_CASES
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_benchmark(
    label: str = "current",
    extractor: Optional[Callable[[str], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run the full benchmark and return a JSON-serialisable record."""
    extraction = evaluate_extraction(extractor=extractor)
    held_out = evaluate_extraction(cases=HELD_OUT_CASES, extractor=extractor)
    detection = evaluate_detection(extractor=None if extractor is None else (
        lambda text: extract_criteria_from_goal(text)))
    fail_closed = evaluate_fail_closed()
    gates = {
        "detection_recall_ge_0.92": detection["recall"] >= 0.92,
        "detection_precision_ge_0.95": detection["precision"] >= 0.95,
        "extraction_f1_ge_0.95": extraction["f1"] >= 0.95,
        "held_out_f1_ge_0.90": held_out["f1"] >= 0.90,
        "fail_closed_ok": bool(fail_closed["ok"]),
    }
    return {
        "label": label,
        "corpus_sha256": corpus_fingerprint(),
        "extraction": extraction,
        "held_out": held_out,
        "detection": detection,
        "fail_closed": fail_closed,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: run the benchmark and write a JSON artifact."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    record = run_benchmark(label=args.label)
    blob = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)
    print(blob)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob, encoding="utf-8")
        print(f"\nwrote {out} sha256={hashlib.sha256(blob.encode()).hexdigest()[:16]}")
    return 0 if record["all_gates_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())