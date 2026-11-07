#!/usr/bin/env python3
"""Contribution meter: map mission reports onto the six-class scientific-contribution ontology.

Design contract (see ``ontology.json`` for the normative text)
-------------------------------------------------------------
* **Evidence first.**  A label is only assigned when the machine-extracted evidence fields
  (error-table rows vs. an *external* reference, acceptance-gate PASS counts, evidence-level
  transitions, defect/regression markers, negative-result markers, verification-file names)
  satisfy the class gate.  Every decision carries the fields it used (``fields``/``trace``).
* **LLM only as fallback.**  If no rule gate fires, an optional LLM fallback may propose a
  label; it receives *only* the extracted field bundle, must echo the fields it used, and its
  answer is recorded as ``decided_by="llm_fallback"``.  The fallback is disabled by default and
  never runs during evaluation, so the gold-set metrics stay reproducible offline.
* **Abstain is a valid answer.**  ``NONE`` (no contribution met its minimum evidence gate) is a
  first-class outcome; it is what makes the "zero-contribution mission" ratio computable.

CLI
    classify   one report / mission / raw file
    report     window aggregate: unit counts, zero-contribution ratio, M_mission conversion table
    eval       gold-set metrics (macro F1, per-class F1, confusion matrix)

Examples
    python3 tools/contribution_meter/classifier.py classify --corpus data/corpus_reports.jsonl \
        --mission-id 5305
    python3 tools/contribution_meter/classifier.py report --window-hours 48 \
        --out data/contribution_report_48h.json
    python3 tools/contribution_meter/classifier.py eval --golden data/golden_set_contributions.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import extract, open_text, resolve_corpus, summarise_fields  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(HERE, "ontology.json")

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
#: single-label precedence for missions/reports that satisfy several gates.
#: Ordered by *durable knowledge* gain: a newly verified case outranks a correction, a
#: correction outranks a capability delta (capability can be re-derived), a negative result
#: outranks a method note, and pure infrastructure is last.  ``eval`` re-runs the metric under
#: the alternative order to show the result is not an artefact of this choice.
PRECEDENCE = ["C1", "C2", "C3", "C5", "C4", "C6"]
PRECEDENCE_ALT = ["C1", "C3", "C2", "C5", "C4", "C6"]
PREC_CORRECTION = ["C3", "C2", "C5", "C4", "C1", "C6"]
PREC_DOC = ["C4", "C2", "C5", "C1", "C3", "C6"]
PREC_LEVEL_UP = ["C2", "C1", "C3", "C5", "C4", "C6"]
PREC_NEGATIVE = ["C5", "C3", "C4", "C2", "C1", "C6"]
PREC_CAPABILITY = ["C2", "C1", "C3", "C5", "C4", "C6"]
PREC_DEFAULT = PRECEDENCE

#: contribution weights in "scientific contribution units" (SCU).  One SCU is defined as one
#: reproducible verification case against a reference that the code under test does not own.
WEIGHTS = {"C1": 1.0, "C2": 0.75, "C3": 0.6, "C5": 0.5, "C4": 0.3, "C6": 0.05, "NONE": 0.0}
#: classes that count as *scientific* contribution (used for the zero-contribution ratio)
SCIENTIFIC = {"C1", "C2", "C3", "C4", "C5"}

# gate thresholds (also recorded in ontology.json)
ERR_ACCEPT_PCT = 15.0      # platform's published accuracy bar
MIN_UNITS_PHYSICS = 3      # a physics-bearing report prints dimensional quantities
MIN_PHYSICS_TERMS = 4       # a verification/capability unit must name real physical quantities
MAX_ADV_RATIO = 0.40       # adversarial gold items: <=40% of an evidence class may be lookalikes


def load_ontology() -> dict:
    with open(ONTOLOGY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# rule cascade
# --------------------------------------------------------------------------------------
def _gates(ev: dict) -> dict:
    """Evaluate every class gate independently; return {class: (bool, [reasons])}.

    Thresholds mirror ontology.json v1.2 and are *not* fitted per item; the same numbers are
    used for every report in the window.
    """
    gate = ev["gate"]
    n_ext_rows = ev.get("n_external_error_rows", 0)
    min_ext = ev.get("min_external_err_pct")
    min_err = ev["min_err_pct"]
    err_ok = min_ext is not None and min_ext <= ERR_ACCEPT_PCT
    sub3 = min_ext is not None and min_ext <= 3.0
    physics = ev["n_units"] >= MIN_UNITS_PHYSICS and ev.get("n_physics_terms", 0) >= MIN_PHYSICS_TERMS
    reproducible = ev["n_verification_names"] >= 1 or any(
        n.endswith((".py", ".json", ".csv", ".npz")) for n in ev["files"]
    )
    correction = ev.get("n_correction_pattern", 0) >= 1
    title_shape = ev.get("n_title_shape", 0) >= 1
    level_up = ev.get("n_level_up", 0) >= 1
    pure_infra = ev["n_infra_markers"] >= 10 and ev["n_units"] < 25

    # ---- C1 verified-case -------------------------------------------------------------
    c1_reasons = []
    c1 = True
    if n_ext_rows < 2:
        c1 = False
        c1_reasons.append(f"only {n_ext_rows} error row(s) with an external reference (need >=2)")
    if not err_ok:
        c1 = False
        c1_reasons.append(f"min external error {min_ext} exceeds {ERR_ACCEPT_PCT}% or missing")
    if gate["n_pass"] < 1:
        c1 = False
        c1_reasons.append("no acceptance-gate PASS")
    if not reproducible:
        c1 = False
        c1_reasons.append("no reproducible comparison artefact")
    if not physics:
        c1 = False
        c1_reasons.append(f"n_units={ev['n_units']} < {MIN_UNITS_PHYSICS}")
    if correction:
        c1 = False
        c1_reasons.append("the mission states a correction of a known-wrong value -> C3 candidate")
    if title_shape:
        c1 = False
        c1_reasons.append("title shape is a document/method deliverable -> C4 candidate")

    # ---- C2 capability-delta ----------------------------------------------------------
    c2_reasons = []
    c2 = True
    new_capability = bool(re.search(r"新建|新增|首次|从零|补齐|REAL\s*化|转正|NOT_SUPPORTED|WITHHELD",
                                    ev["title"] + " " + (ev.get("summary") or "")))
    if not (level_up or new_capability):
        c2 = False
        c2_reasons.append("no stated capability/evidence-level change")
    if gate["n_pass"] < 1:
        c2 = False
        c2_reasons.append("no gate PASS proving the capability is exercised")
    if sub3 and n_ext_rows >= 2 and not level_up:
        c2 = False
        c2_reasons.append("external sub-3% error table present -> the unit is C1")

    # ---- C3 defect-fix ----------------------------------------------------------------
    c3_reasons = []
    c3 = True
    if ev["n_defect_markers"] < 1:
        c3 = False
        c3_reasons.append("no defect/fix marker")
    if not correction:
        c3 = False
        c3_reasons.append("no stated old-value -> new-value correction")
    if not (gate["n_pass"] >= 1 or ev["has_gate_lock"] or ev["n_pytest_mentions"] >= 2):
        c3 = False
        c3_reasons.append("no regression lock / gate PASS after the fix")
    if ev["n_error_rows"] < 1 and not err_ok:
        c3 = False
        c3_reasons.append("correction is not quantified")

    # ---- C4 method/doc ----------------------------------------------------------------
    c4_reasons = []
    c4 = True
    if not (ev["n_doc_shape_tokens"] >= 1 or ev["has_method_language"] or ev["n_formulas"] >= 3):
        c4 = False
        c4_reasons.append("deliverable is neither doc-shaped nor method-shaped")
    if ev["n_sections"] < 3:
        c4 = False
        c4_reasons.append("report too thin for a method/document contribution")

    # ---- C5 negative-result -----------------------------------------------------------
    c5_reasons = []
    c5 = True
    headline = (ev["title"] + " " + (ev.get("summary") or "")).lower()
    headline_hit = bool(re.search(
        r"fail|withheld|证伪|不可行|不成立|退役|ill-posed|根因|失效|超出适用域|不满足", headline))
    if not headline_hit:
        c5 = False
        c5_reasons.append("no failure verdict in the headline claim")
    if not (ev["n_failure_verdicts"] >= 1):
        c5 = False
        c5_reasons.append("no explicit verdict token in the body")
    max_err = ev.get("max_err_pct")
    summary_big = any(float(x) > ERR_ACCEPT_PCT for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", ev.get("summary") or ""))
    if not ((max_err is not None and max_err > ERR_ACCEPT_PCT) or gate["n_fail"] >= 3 or summary_big):
        c5 = False
        c5_reasons.append("verdict is not quantitatively load-bearing (>15% error or >=3 FAIL rows)")
    if sub3 and n_ext_rows >= 2 and gate["n_pass"] >= 2 and gate["n_fail"] < gate["n_pass"]:
        c5 = False
        c5_reasons.append("a sub-3% external verification dominates -> the unit is C1")

    # ---- C6 infra ---------------------------------------------------------------------
    c6 = True

    gates = {
        "C1": (c1, c1_reasons),
        "C2": (c2, c2_reasons),
        "C3": (c3, c3_reasons),
        "C4": (c4, c4_reasons),
        "C5": (c5, c5_reasons),
        "C6": (c6, []),
    }
    return gates


def pick_precedence(ev: dict) -> list[str]:
    """Ontology v1.2 precedence: the primary unit type follows the shape of the *claim*.

    1. a stated correction of a known-wrong value outranks everything (C3) - the mission's own
       headline is "the published number was wrong";
    2. a headline failure verdict outranks a sub-table verification (C5) - e.g. an honest-FAIL
       mission that also verifies a small sub-quantity against an analytic anchor;
    3. a document/method-shaped title (C4), 4. an explicit evidence-level change (C2),
    5. otherwise a verified case (C1), 6. infrastructure last.
    """
    if ev.get("n_correction_pattern", 0) >= 1:
        return PREC_CORRECTION
    headline = (ev["title"] + " " + (ev.get("summary") or "")).lower()
    if re.search(r"fail|withheld|证伪|不可行|不成立|退役|ill-posed|根因|超出适用域", headline) \
            and ev.get("n_failure_verdicts", 0) >= 1:
        return PREC_NEGATIVE
    if ev.get("n_title_shape", 0) >= 1:
        return PREC_DOC
    if ev.get("n_level_up", 0) >= 1:
        return PREC_LEVEL_UP
    return PREC_DEFAULT


def classify_evidence(ev: dict, precedence: list[str] | None = None,
                      llm_fallback: bool = False) -> dict:
    precedence = precedence or pick_precedence(ev)
    gates = _gates(ev)
    fired = [c for c in precedence if gates[c][0]]
    fields = summarise_fields(ev)
    if fired:
        label = fired[0]
        return {
            "mission_id": ev["mission_id"],
            "report_id": ev["report_id"],
            "label": label,
            "candidates": fired,
            "decided_by": "rule",
            "trace": fields,
            "reasons_against": {c: gates[c][1] for c in CLASSES if not gates[c][0]},
            "weight": WEIGHTS[label],
        }
    # no rule gate fired -> abstain, optionally ask the LLM with the field bundle only
    if llm_fallback:
        guess = _llm_fallback(ev, fields)
        if guess:
            return {
                "mission_id": ev["mission_id"],
                "report_id": ev["report_id"],
                "label": guess["label"],
                "candidates": [],
                "decided_by": "llm_fallback",
                "trace": fields,
                "llm_fields_used": guess["fields_used"],
                "reasons_against": {c: gates[c][1] for c in CLASSES if not gates[c][0]},
                "weight": WEIGHTS[guess["label"]],
            }
    return {
        "mission_id": ev["mission_id"],
        "report_id": ev["report_id"],
        "label": "NONE",
        "candidates": [],
        "decided_by": "rule",
        "trace": fields,
        "reasons_against": {c: gates[c][1] for c in CLASSES if not gates[c][0]},
        "weight": 0.0,
    }


def classify_report(report: dict, **kw) -> dict:
    return classify_evidence(extract(report), **kw)


def _llm_fallback(ev: dict, fields: list[str]) -> dict | None:
    """Optional last-resort LLM judgement driven *only* by extracted fields.

    Disabled unless ``CONTRIBUTION_METER_LLM=1`` and an OpenAI-compatible endpoint is
    configured.  Returns ``None`` when unavailable, in which case the meter abstains.
    """
    if os.environ.get("CONTRIBUTION_METER_LLM") != "1":
        return None
    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        return None
    import urllib.request

    prompt = (
        "You are a metrology assistant. Classify the mission into exactly one of "
        "C1 verified-case, C2 capability-delta, C3 defect-fix, C4 method-doc, "
        "C5 negative-result, C6 infra using ONLY these extracted fields:\n"
        + "\n".join(fields)
        + "\nAnswer JSON: {\"label\": \"Cx\", \"fields_used\": [\"...\"]}"
    )
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # pragma: no cover
            payload = json.loads(resp.read())
        text = payload["choices"][0]["message"]["content"]
        start, end = text.find("{"), text.rfind("}")
        guess = json.loads(text[start : end + 1])
        label = str(guess.get("label", "")).upper()
        if label not in CLASSES:
            return None
        return {"label": label, "fields_used": guess.get("fields_used") or fields}
    except Exception:  # pragma: no cover - network/model failures are not fatal
        return None


# --------------------------------------------------------------------------------------
# corpus / window aggregation
# --------------------------------------------------------------------------------------
def load_corpus(path: str) -> list[dict]:
    with open_text(resolve_corpus(path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_missions(path: str | None) -> list[dict]:
    if not path:
        return []
    path = resolve_corpus(path)
    if not os.path.exists(path):
        return []
    with open_text(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_ts(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover
        return None


def evaluate(preds: dict[str, str], gold: dict[str, str], classes: list[str] = None) -> dict:
    """Per-class precision/recall/F1 + macro average over strict single-label predictions."""
    classes = classes or CLASSES
    per_class = {}
    for c in classes:
        tp = sum(1 for k, g in gold.items() if g == c and preds.get(k) == c)
        fp = sum(1 for k, g in gold.items() if g != c and preds.get(k) == c)
        fn = sum(1 for k, g in gold.items() if g == c and preds.get(k) != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {
            "support": sum(1 for g in gold.values() if g == c),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        }
    macro = sum(v["f1"] for v in per_class.values()) / len(per_class)
    labels = classes + ["NONE"]
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for k, g in gold.items():
        confusion.setdefault(g, {}).setdefault(preds.get(k, "NONE"), 0)
        confusion[g][preds.get(k, "NONE")] += 1
    accuracy = sum(1 for k, g in gold.items() if preds.get(k) == g) / max(1, len(gold))
    return {
        "n": len(gold),
        "per_class": per_class,
        "macro_f1": round(macro, 4),
        "accuracy": round(accuracy, 4),
        "confusion": confusion,
        "classes_below_threshold": [c for c, v in per_class.items() if v["f1"] < 0.70],
    }


def build_window_report(
    corpus: list[dict],
    missions: list[dict],
    window_hours: float = 48.0,
    now: _dt.datetime | None = None,
    precedence: list[str] | None = None,
) -> dict:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(hours=window_hours)

    def in_window(ts_str):
        ts = _parse_ts(ts_str)
        return ts is not None and ts >= cutoff

    win_missions = [m for m in missions if in_window(m.get("created_at"))] if missions else []
    win_reports = [r for r in corpus if in_window(r.get("report_created_at") or r.get("mission_created_at"))]
    if not win_reports:  # fall back: corpus built inside the window
        win_reports = corpus

    per_report = []
    for r in win_reports:
        res = classify_report(r, precedence=precedence)
        per_report.append(
            {
                "mission_id": res["mission_id"],
                "report_id": res["report_id"],
                "label": res["label"],
                "weight": res["weight"],
                "decided_by": res["decided_by"],
                "trace": res["trace"],
                "title": (r.get("title") or "")[:90],
            }
        )

    # mission-level: best (highest weight) label among its reports
    best: dict[str, dict] = {}
    weights_rank = {c: WEIGHTS[c] for c in CLASSES}
    for row in per_report:
        row["mission_id"] = str(row["mission_id"])
        cur = best.get(row["mission_id"])
        if cur is None or weights_rank.get(row["label"], 0) > weights_rank.get(cur["label"], 0):
            best[row["mission_id"]] = row

    mission_ids_in_window = sorted({str(m.get("id")) for m in win_missions}) or sorted(best)
    for mid in sorted(best):  # reports from missions outside the mission table still count
        if mid not in mission_ids_in_window:
            mission_ids_in_window.append(mid)
    counts = {c: 0 for c in CLASSES}
    counts["NONE"] = 0
    weighted = 0.0
    zero_contribution: list[str] = []
    for mid in mission_ids_in_window:
        row = best.get(mid)
        label = row["label"] if row else "NONE"
        counts[label] = counts.get(label, 0) + 1
        weighted += WEIGHTS.get(label, 0.0)
        if label not in SCIENTIFIC:
            zero_contribution.append(mid)

    n_missions = len(mission_ids_in_window)
    n_reports = len(per_report)
    scientific_units = sum(counts[c] for c in SCIENTIFIC)
    total_units = sum(counts[c] for c in CLASSES)
    n_with_report = len({str(r["mission_id"]) for r in per_report})
    conversion = {
        "window_hours": window_hours,
        "window_start": cutoff.isoformat(),
        "window_end": now.isoformat(),
        "n_missions": n_missions,
        "n_reports": n_reports,
        "n_missions_with_report": n_with_report,
        "n_missions_without_report": n_missions - n_with_report,
        "units_by_class": counts,
        "n_scientific_units": scientific_units,
        "n_all_units_incl_infra": total_units,
        "sum_weighted_scu": round(weighted, 3),
        "scientific_units_per_mission": round(scientific_units / n_missions, 4) if n_missions else 0.0,
        "scu_per_mission": round(weighted / n_missions, 4) if n_missions else 0.0,
        "mission_units_per_scientific_contribution": round(n_missions / scientific_units, 4) if scientific_units else None,
        "zero_contribution_missions": sorted(zero_contribution),
        "zero_contribution_ratio": round(len(zero_contribution) / n_missions, 4) if n_missions else 0.0,
        "weights": WEIGHTS,
        "precedence": precedence or PRECEDENCE,
    }
    return {"conversion": conversion, "per_report": per_report,
            "mission_best": {k: v for k, v in best.items()}}


def _unit_examples(payload: dict, k: int = 3) -> dict:
    """A few labelled missions per class, so the audit trail survives even without the full dump."""
    out: dict[str, list] = {}
    for row in payload.get("per_report", []):
        bucket = out.setdefault(row["label"], [])
        if len(bucket) < k:
            bucket.append({"mission_id": row["mission_id"], "report_id": row["report_id"],
                           "title": row.get("title", ""), "trace": row.get("trace", [])[:4]})
    return out


def _classify_cli(args) -> int:
    if args.corpus:
        corpus = load_corpus(args.corpus)
    else:
        corpus = []
    rows = []
    for r in corpus:
        if args.mission_id and str(r["mission_id"]) != str(args.mission_id):
            continue
        if args.report_id and str(r["report_id"]) != str(args.report_id):
            continue
        rows.append(classify_report(r, llm_fallback=args.llm_fallback))
    if args.text:
        with open(args.text, encoding="utf-8") as fh:
            md = fh.read()
        rows.append(
            classify_report(
                {"mission_id": args.mission_id or "local", "report_id": "local",
                 "title": os.path.basename(args.text), "markdown": md},
                llm_fallback=args.llm_fallback,
            )
        )
    if not rows:
        print(json.dumps({"error": "no report matched", "mission_id": args.mission_id,
                          "report_id": args.report_id}, ensure_ascii=False))
        return 1
    rows.sort(key=lambda r: -r["weight"])
    out = dict(rows[0])
    out["all_labels"] = [{"report_id": r["report_id"], "label": r["label"]} for r in rows]
    out["n_reports"] = len(rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="classify one report/mission")
    c.add_argument("--corpus", default="data/corpus_reports.jsonl")
    c.add_argument("--mission-id", default=None)
    c.add_argument("--report-id", default=None)
    c.add_argument("--text", default=None, help="classify a raw markdown file instead")
    c.add_argument("--llm-fallback", action="store_true")
    c.set_defaults(func=_classify_cli)

    r = sub.add_parser("report", help="window aggregate + M_mission conversion table")
    r.add_argument("--corpus", default="data/corpus_reports.jsonl")
    r.add_argument("--missions", default="data/missions_window.jsonl")
    r.add_argument("--window-hours", type=float, default=48.0)
    r.add_argument("--out", default="data/contribution_report_48h.json")
    r.add_argument("--precedence", default="default", choices=["default", "alt"])
    r.set_defaults(func=_report_cli)

    e = sub.add_parser("eval", help="gold-set metrics")
    e.add_argument("--golden", default="data/golden_set_contributions.json")
    e.add_argument("--corpus", default="data/corpus_reports.jsonl")
    e.add_argument("--out", default="data/golden_eval.json")
    e.set_defaults(func=_eval_cli)

    args = ap.parse_args(argv)
    return args.func(args)


def _report_cli(args) -> int:
    corpus = load_corpus(args.corpus)
    missions = load_missions(args.missions)
    prec = PRECEDENCE_ALT if args.precedence == "alt" else PRECEDENCE
    payload = build_window_report(corpus, missions, args.window_hours, precedence=prec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    # compact companion artefact: the conversion table alone, small enough to commit
    summary_path = os.path.join(
        os.path.dirname(args.out),
        "contribution_summary_%dh.json" % int(round(args.window_hours)),
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"conversion": payload["conversion"],
                   "unit_examples": _unit_examples(payload)}, fh, ensure_ascii=False, indent=1)
    conv = payload["conversion"]
    print(json.dumps({k: conv[k] for k in (
        "window_hours", "n_missions", "n_reports", "n_missions_with_report",
        "units_by_class", "n_scientific_units", "scientific_units_per_mission",
        "zero_contribution_ratio", "mission_units_per_scientific_contribution", "sum_weighted_scu",
    )}, ensure_ascii=False, indent=2))
    print(f"-> {args.out}", file=sys.stderr)
    return 0


def _eval_cli(args) -> int:
    golden = json.load(open(args.golden, encoding="utf-8"))
    corpus = {str(r["report_id"]): r for r in load_corpus(args.corpus)}
    items = golden["items"] if isinstance(golden, dict) else golden
    gold, preds, rows = {}, {}, []
    for it in items:
        key = str(it["report_id"])
        rep = corpus.get(key)
        if rep is None:
            rows.append({"report_id": key, "error": "report not in corpus"})
            continue
        res = classify_report(rep)
        gold[key] = it["label"]
        preds[key] = res["label"]
        rows.append({"report_id": key, "mission_id": it["mission_id"], "gold": it["label"],
                     "pred": res["label"], "trace": res["trace"]})
    metrics = evaluate(preds, gold)
    metrics_alt = evaluate(
        {k: classify_report(corpus[k], precedence=PRECEDENCE_ALT)["label"] for k in gold}, gold
    )
    payload = {"metrics": metrics, "metrics_alt_precedence": metrics_alt, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"macro_f1": metrics["macro_f1"],
                      "macro_f1_alt_precedence": metrics_alt["macro_f1"],
                      "per_class_f1": {c: v["f1"] for c, v in metrics["per_class"].items()},
                      "below_threshold": metrics["classes_below_threshold"]},
                     ensure_ascii=False, indent=2))
    print(f"-> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())