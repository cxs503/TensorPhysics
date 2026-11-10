"""Negative controls and labelled agreement set for the degeneracy classifier.

Two corpora live in ``data/controls.json``:

``degenerate`` (12)  -- 4 specimens per degeneracy type, each with the *constructed*
  ground truth (``expect_classes``, ``expect_verdict``);
``benign`` (12)      -- healthy criteria, including the two traps named in the goal
  (1e-6-level consistency, and a legitimate zero reference judged with an absolute band);
``supplementary``    -- two extra specimens documenting where structural degeneracy and a
  bad result are *not* the same thing;
``labeled_extra``    -- 16 additional labelled specimens that complete the 40-record
  annotation set used by the pre-registered stop rule.

Scoring is deliberately two-sided: a degenerate specimen counts as a hit when the
predicted class set contains the expected type(s); a benign specimen is a false alarm when
the adaptive family reaches a *different* verdict than the constructed truth (failing a
healthy result is the operationally harmful error), and structural flags on benign
specimens are reported separately instead of being hidden.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping

from .adaptive import judge, judge_family
from .degeneracy import DEGENERACY_TYPES, analyze, multi_metric_families
from .model import CriterionRecord, Verdict
from .parser import extract_quadruple
from .replay import ReplayRow, replay

__all__ = [
    "DATA_PATH",
    "load_controls",
    "records_from_specs",
    "evaluate_controls",
    "evaluate_labeled",
    "labeled_set",
]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "controls.json")


def load_controls(path: str | None = None) -> dict[str, Any]:
    with open(path or DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _record(spec: Mapping[str, Any], kind: str) -> CriterionRecord:
    rid = f"{kind}:{spec['id']}"
    return extract_quadruple(
        spec["criterion"],
        spec.get("y"),
        spec.get("y_ref"),
        scale=spec.get("scale"),
        noise=spec.get("noise"),
        measurand=spec["id"],
        source=f"controls:{kind}",
        rid=rid,
        reported_error=spec.get("reported_error"),
        family=spec.get("family") or f"ctrl::{spec['id']}",
    )


def records_from_specs(specs: Iterable[Mapping[str, Any]], kind: str) -> list[CriterionRecord]:
    return [_record(s, kind) for s in specs]


def evaluate_controls(path: str | None = None) -> dict[str, Any]:
    """Run the classifier and the adaptive family over the 24 controls."""
    data = load_controls(path)
    degenerate = records_from_specs(data["degenerate"], "degenerate")
    benign = records_from_specs(data["benign"], "benign")
    supp = records_from_specs(data.get("supplementary", []), "supplementary")

    # shadow rows make the T3 families carry two competing metric kinds
    shadows: list[CriterionRecord] = []
    for spec in data["degenerate"]:
        sh = spec.get("shadow")
        if not sh:
            continue
        shadows.append(
            extract_quadruple(
                sh["criterion"], sh.get("y"), sh.get("y_ref"), scale=sh.get("scale"),
                measurand=spec["id"] + ":shadow", source="controls:shadow",
                rid=f"shadow:{spec['id']}", family=sh.get("family") or spec["id"],
            )
        )
    everything = degenerate + benign + supp + shadows
    classified = analyze(everything)
    deg_map = {rec.rid: deg for rec, deg in classified}
    fams = multi_metric_families(everything)
    fam_scale: dict[str, float] = {}
    for rec in everything:
        r = rec.y_ref
        if r:
            fam_scale.setdefault(rec.family_key, abs(r))

    # T3 families are judged as a family (that is the remedy); record that too
    fam_results = []
    by_family: dict[str, list[CriterionRecord]] = {}
    for rec in degenerate + shadows:
        if rec.rid.startswith("shadow:") or deg_map[rec.rid].has("T3_multi_metric_no_priority"):
            by_family.setdefault(rec.family_key, []).append(rec)
    for fam, recs in sorted(by_family.items()):
        fam_results.append(judge_family(recs, degeneracies=deg_map, family_scale=fam_scale.get(fam)))
    fam_by_id: dict[str, dict[str, Any]] = {}
    for spec in data["degenerate"]:
        if spec.get("shadow"):
            fam = spec.get("family") or spec["id"]
            merged = next((m for m in fam_results if m["family"] == fam), None)
            if merged is not None:
                fam_by_id[spec["id"]] = merged

    hits: list[dict[str, Any]] = []
    for spec, rec in zip(data["degenerate"], degenerate):
        deg = deg_map[rec.rid]
        verdict = judge(rec, deg)
        expect = set(spec["expect_classes"])
        hit = expect.issubset(set(deg.classes))
        merged = fam_by_id.get(spec["id"])
        # for T3 the decisive verdict is the *family* verdict: a single member's PASS is
        # exactly the "partial pass" the node forbids
        decisive = merged["merged_status"] if merged is not None else verdict.status
        hits.append({
            "id": spec["id"], "type": spec["type"], "expected": sorted(expect),
            "predicted": list(deg.classes), "undecidable": list(deg.undecidable),
            "hit": hit, "verdict": decisive, "member_verdict": verdict.status,
            "family_merged": merged["merged_status"] if merged is not None else None,
            "expect_verdict": spec["expect_verdict"],
            "verdict_ok": decisive == spec["expect_verdict"],
            "alternative_value": verdict.value, "alternative_metric": verdict.metric,
        })

    benign_rows: list[dict[str, Any]] = []
    for spec, rec in zip(data["benign"], benign):
        deg = deg_map[rec.rid]
        verdict = judge(rec, deg)
        expected_verdict = spec["expect_verdict"]
        benign_rows.append({
            "id": spec["id"], "criterion": rec.criterion, "expected_classes": spec["expect_classes"],
            "predicted": list(deg.classes), "structurally_flagged": deg.flagged,
            "verdict": verdict.status, "expect_verdict": expected_verdict,
            "false_alarm": verdict.status != expected_verdict,
            "value": verdict.value, "metric": verdict.metric,
        })

    supp_rows = []
    for spec, rec in zip(data.get("supplementary", []), supp):
        deg = deg_map[rec.rid]
        verdict = judge(rec, deg)
        supp_rows.append({"id": spec["id"], "lane": spec["lane"], "predicted": list(deg.classes),
                          "expect_classes": spec["expect_classes"], "verdict": verdict.status,
                          "expect_verdict": spec["expect_verdict"],
                          "hit": set(spec["expect_classes"]).issubset(set(deg.classes)),
                          "verdict_ok": verdict.status == spec["expect_verdict"]})

    n_deg = len(hits)
    n_hit = sum(1 for h in hits if h["hit"])
    n_benign = len(benign_rows)
    n_false_alarm = sum(1 for b in benign_rows if b["false_alarm"])
    n_struct_flag = sum(1 for b in benign_rows if b["structurally_flagged"])

    # ReplayRow objects for the case-study selector (T2/T3 representatives)
    extra_rows: list[tuple[str, ReplayRow]] = []
    ctrl_result = replay(degenerate + shadows + benign + supp)
    for row in ctrl_result.rows:
        if row.record.rid.startswith(("degenerate:", "supplementary:", "benign:")):
            extra_rows.append(("controls", row))

    return {
        "schema": "tensorphysics.metrology.criteria.controls/1",
        "n_degenerate": n_deg,
        "n_degenerate_hit": n_hit,
        "degenerate_hit_rate": n_hit / n_deg if n_deg else 0.0,
        "n_benign": n_benign,
        "n_benign_false_alarm": n_false_alarm,
        "n_benign_structurally_flagged": n_struct_flag,
        "benign_false_alarm_rate": n_false_alarm / n_benign if n_benign else 0.0,
        "degenerate_detail": hits,
        "benign_detail": benign_rows,
        "supplementary_detail": supp_rows,
        "family_merges": [
            {"family": m["family"], "n_members": m["n_members"], "members": m["members"],
             "merged_status": m["merged_status"], "merged_metric": m["merged_metric"],
             "merged_value": m["merged_value"], "partial_pass_risk": m["partial_pass_risk"]}
            for m in fam_results
        ],
        "by_type": {t: {"n": sum(1 for h in hits if h["type"] == t.split("_")[0]),
                        "hit": sum(1 for h in hits if h["type"] == t.split("_")[0] and h["hit"])}
                    for t in DEGENERACY_TYPES},
        "meets_degenerate_12_of_12": n_hit == n_deg == 12,
        "meets_benign_zero_false_alarm": n_false_alarm == 0,
        "_rows": extra_rows,
    }


# --------------------------------------------------------------------------------------
# 40-record labelled set (pre-registered stop rule)
# --------------------------------------------------------------------------------------
def labeled_set(path: str | None = None) -> list[dict[str, Any]]:
    """The 40 labelled specimens: 24 controls (construction truth) + 16 extras."""
    data = load_controls(path)
    out: list[dict[str, Any]] = []
    for spec in data["degenerate"]:
        out.append({"id": spec["id"], "labelA": "degenerate", "labelB_hint": "denominator",
                    "spec": spec, "kind": "degenerate"})
    for spec in data["benign"]:
        out.append({"id": spec["id"], "labelA": "benign", "labelB_hint": "absolute" if "|" in spec["criterion"] else "denominator",
                    "spec": spec, "kind": "benign"})
    for spec in data["labeled_extra"]:
        out.append({"id": spec["id"], "labelA": spec["labelA"], "labelB_hint": spec.get("labelB_hint", "denominator"),
                    "spec": spec, "kind": "extra"})
    return out


def label_from_text(record: CriterionRecord) -> str:
    """Independent, *text-only* annotator used to quantify annotation bias.

    It sees the criterion string only (never the numbers): a denominator-normalised metric
    without a declared magnitude guard is called ``degenerate`` (at risk of division by a
    vanishing reference), an absolute/epsilon criterion is called ``benign``.  Agreement
    with the data-verified label therefore measures how much the *string alone* can tell --
    which is exactly the gap this node is about.
    """
    metric = record.parsed.metric
    if metric in ("relative", "log_ratio"):
        text = record.criterion or ""
        if "scale" in text.lower() or "工程单位" in text:
            return "benign"
        return "degenerate"
    return "benign"


def evaluate_labeled(path: str | None = None) -> dict[str, Any]:
    """Classifier-vs-label agreement and inter-annotator agreement (Cohen's kappa)."""
    from .stats import agreement, cohen_kappa

    entries = labeled_set(path)
    records: list[CriterionRecord] = []
    for e in entries:
        rec = _record(e["spec"], e["kind"])
        records.append(rec)
    # shadow rows complete the T3 families; they are classified but never scored
    shadows: list[CriterionRecord] = []
    for e in entries:
        sh = e["spec"].get("shadow")
        if not sh:
            continue
        shadows.append(extract_quadruple(
            sh["criterion"], sh.get("y"), sh.get("y_ref"), scale=sh.get("scale"),
            measurand=e["id"] + ":shadow", source="controls:shadow",
            rid=f"shadow:{e['id']}", family=sh.get("family") or e["spec"].get("family"),
        ))
    classified = analyze(records + shadows)
    deg_map = {rec.rid: deg for rec, deg in classified}

    rows = []
    for e, rec in zip(entries, records):
        deg = deg_map[rec.rid]
        cls_label = "degenerate" if deg.flagged else "benign"
        text_label = label_from_text(rec)
        verdict = judge(rec, deg)
        rows.append({
            "id": e["id"], "labelA": e["labelA"], "labelB": text_label,
            "classifier": cls_label, "classes": list(deg.classes),
            "undecidable": list(deg.undecidable), "verdict": verdict.status,
            "criterion": rec.criterion, "y": rec.y, "y_ref": rec.y_ref,
            "agree_A_classifier": e["labelA"] == cls_label,
            "agree_A_text": e["labelA"] == text_label,
            "agree_text_classifier": text_label == cls_label,
        })
    a = [r["labelA"] for r in rows]
    b = [r["labelB"] for r in rows]
    c = [r["classifier"] for r in rows]
    labels = ("benign", "degenerate")
    return {
        "schema": "tensorphysics.metrology.criteria.labeled/1",
        "n": len(rows),
        "n_agree_A_classifier": sum(1 for r in rows if r["agree_A_classifier"]),
        "agreement_A_classifier": agreement(a, c),
        "kappa_A_classifier": cohen_kappa(a, c, labels),
        "agreement_A_text_annotator": agreement(a, b),
        "kappa_A_text_annotator": cohen_kappa(a, b, labels),
        "labels_A": {k: a.count(k) for k in labels},
        "labels_classifier": {k: c.count(k) for k in labels},
        "rows": rows,
        "stop_rule_threshold": 0.70,
        "meets_stop_rule": agreement(a, c) >= 0.70,
    }