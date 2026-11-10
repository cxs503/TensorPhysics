#!/usr/bin/env python3
"""P266W1-N09 — criterion metrology: re-adjudicate the four case studies.

Four historical cases are re-judged from their *own published numbers* (no re-typing: the
numbers are extracted from the report tables / mission headline at run time):

=================  ===========================================================
mission            criterion of record
=================  ===========================================================
7241 (ship)        ζ_eq = 0.0175 vs experiment 0.04-0.06, tolerance 10 %
7243 (ship)        STAWAVE-2 vs ITTC 5.99 % @3 %; STAWAVE-3 no dataset; stall 3.83 %;
                   spectral long-wave closure 8.40 % "within the 10 % band"
7245 (ship)        emergency stop 1352 m vs IMO 20L = 3600 m; K_T error 6.2 %
7248 (TensorBEM)   pseudo-resonance: pole positions, out-of-band A 4.804 %
=================  ===========================================================

For each row we report the baseline verdict, the adaptive verdict, the *numeric value of the
replacement criterion* (error / tau under the adaptive metric) and whether the verdict
direction agrees with the deviation the report itself states (``direction_consistent``).

Output: ``artifacts/criteria_metrology/case_studies.json``.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorphysics.metrology.criteria_semantics import corpus as C  # noqa: E402
from tensorphysics.metrology.criteria_semantics.replay import _recorded_flip as recorded_flip  # noqa: E402
from tensorphysics.metrology.criteria_semantics.model import (  # noqa: E402
    Criterion, Measurement, Metric, Verdict,
)
from tensorphysics.metrology.criteria_semantics.parser import CriterionRow  # noqa: E402
from tensorphysics.metrology.criteria_semantics.adaptive import evaluate, evaluate_baseline  # noqa: E402
from tensorphysics.metrology.criteria_semantics.degeneracy import classify  # noqa: E402

CORPUS = "artifacts/criteria_metrology/historical_criteria.jsonl"
OUT = "artifacts/criteria_metrology/case_studies.json"
MISSION_QUOTES = {
    7245: "artifacts/criteria_metrology/mission_7245_headline.json",
}


def corpus_rows_for(rows, mission_id: int):
    return [r for r in rows if (r.provenance or {}).get("mission_id") == mission_id]


def pick(rows, pattern: str):
    """First corpus row of that mission whose quantity/error cell matches ``pattern``."""
    for r in rows:
        p = r.provenance or {}
        blob = " ".join(str(p.get(k) or "") for k in
                        ("quantity_cell", "reference_cell", "error_cell", "verdict_cell"))
        if re.search(pattern, blob):
            return r
    return None


def row_from_mission_quote(mission_id: int, quantity: str, y: float, y_ref: float,
                           tau: float, metric: Metric, unit: str, quote_path: str,
                           pattern_doc: str) -> CriterionRow:
    """Build a criterion from a mission headline quote (provenance: the quoted file)."""
    with open(quote_path, encoding="utf-8") as fh:
        quote = json.load(fh)
    crit = Criterion(measurement=Measurement(y=y, y_ref=y_ref, unit=unit, quantity=quantity),
                     tau=tau, metric=metric, name=quantity,
                     source={"mission_id": mission_id, "quote": quote["headline"],
                             "quote_sha256": quote["sha256"], "extraction": pattern_doc})
    return CriterionRow(criterion=crit, recorded_verdict=quote.get("recorded_verdict", "unknown"),
                        provenance={"mission_id": mission_id, "quantity_cell": quantity,
                                    "reference_cell": quote.get("reference_hint", ""),
                                    "error_cell": quote["headline"][:160],
                                    "verdict_cell": quote.get("recorded_verdict", ""),
                                    "quote_sha256": quote["sha256"]})


def judge(row: CriterionRow, expect_direction: str) -> dict:
    prov = row.provenance or {}
    dg = classify(row.criterion, ref_text=prov.get("reference_cell"),
                  verdict_text=prov.get("verdict_cell"), error_text=prov.get("error_cell"))
    base = evaluate_baseline(row.criterion)
    ad = evaluate(row.criterion, degeneracy=dg)
    rel = ad.relative_error
    direction_ok = ad.verdict.value == expect_direction
    return {
        "name": row.criterion.name,
        "mission_id": prov.get("mission_id"),
        "y": row.criterion.measurement.y,
        "y_ref": row.criterion.measurement.y_ref,
        "ref_range": list(row.criterion.measurement.ref_range)
        if row.criterion.measurement.ref_range else None,
        "tau": row.criterion.tau,
        "competing_taus": list(row.criterion.competing_taus),
        "metric": ad.metric.value,
        "baseline": {"verdict": base.verdict.value, "metric": base.metric.value,
                     "error": base.error, "tau": base.tau, "reason": base.reason},
        "adaptive": {"verdict": ad.verdict.value, "metric": ad.metric.value,
                     "error": ad.error, "tau": ad.tau, "ratio": ad.ratio,
                     "remedy": ad.remedy, "reason": ad.reason},
        "alternative_criterion_value": ad.error,
        "alternative_criterion_ratio": ad.ratio,
        "relative_error_pct": None if rel is None else round(100.0 * rel, 4),
        "degeneracy": dg.kind.value,
        "degeneracy_witnesses": dg.witnesses,
        "recorded_verdict": row.recorded_verdict,
        "recorded_verdict_vs_adaptive": recorded_flip(row.recorded_verdict, ad.verdict),
        "expected_direction": expect_direction,
        "direction_consistent": bool(direction_ok),
        "provenance": {k: v for k, v in prov.items() if k != "heading"},
    }


def main() -> int:
    rows = C.load_corpus(CORPUS)
    out: dict = {"cases": {}, "quotes": {}}

    # ---- m7241: roll damping vs experiment band (the "部分" that must become FAIL) ----
    m7241 = corpus_rows_for(rows, 7241)
    r1 = pick(m7241, r"ζ_eq|阻尼比|总阻尼")
    out["cases"].setdefault("m7241", []).append(judge(r1, "FAIL"))

    # ---- m7243: added resistance ----
    m7243 = corpus_rows_for(rows, 7243)
    for pat, direction in ((r"STAWAVE-2 长波域", "FAIL"),
                           (r"STAWAVE-3", "WITHHELD"),
                           (r"谱叠加长波闭合", "FAIL"),
                           (r"失速 vs 线性化", "FAIL")):
        row = pick(m7243, pat)
        if row is not None:
            out["cases"].setdefault("m7243", []).append(judge(row, direction))

    # ---- m7248: pseudo-resonance ----
    m7248 = corpus_rows_for(rows, 7248)
    for pat, direction in ((r"3-D 球 sway 带外", "FAIL"),
                           (r"2-D 圆极点位置|极点位置", "PASS"),
                           (r"带外逐位", "PASS")):
        row = pick(m7248, pat)
        if row is not None:
            out["cases"].setdefault("m7248", []).append(judge(row, direction))

    # ---- m7245: emergency stop vs IMO 20L (one-sided engineering bound) ----
    qpath = MISSION_QUOTES[7245]
    with open(qpath, encoding="utf-8") as fh:
        q = json.load(fh)
    vals = q["extracted"]
    row_stop = row_from_mission_quote(
        7245, "停船距离 vs IMO 20L 衡准", vals["s_stop_m"], vals["imo_20L_m"], 0.0,
        Metric.BOUND, "m", qpath, "regex on headline: 停船距离<num>m ... ≤20L")
    row_kt = row_from_mission_quote(
        7245, "K_T 误差", 1.0 + vals["kT_err"] / 100.0, 1.0, 0.03, Metric.RELATIVE, "1",
        qpath, "regex on headline: K_T误差<num>%")
    out["cases"]["m7245"] = [judge(row_stop, "PASS"), judge(row_kt, "FAIL")]

    flat = [c for v in out["cases"].values() for c in v]
    out["n_cases"] = len(flat)
    out["n_direction_consistent"] = sum(1 for c in flat if c["direction_consistent"])
    out["m7241_explicit_fail"] = any(c["adaptive"]["verdict"] == "FAIL"
                                     for c in out["cases"].get("m7241", []))
    out["m7241_recorded_verdict"] = [c["recorded_verdict"] for c in out["cases"].get("m7241", [])]
    out["note"] = ("alternative_criterion_value is the error under the replacement metric and "
                   "alternative_criterion_ratio = error/tau (how many tolerances off).")
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    for mission, cases in out["cases"].items():
        for c in cases:
            print(f"{mission} {c['name'][:30]:32s} base={c['baseline']['verdict']:8s} "
                  f"adapt={c['adaptive']['verdict']:8s} ratio={c['adaptive']['ratio']} "
                  f"dg={c['degeneracy'][:12]:12s} dir_ok={c['direction_consistent']}")
    print("cases", out["n_cases"], "direction_consistent", out["n_direction_consistent"],
          "m7241_explicit_fail", out["m7241_explicit_fail"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())