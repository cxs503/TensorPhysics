"""Human labels, the second (text-only) annotator, and inter-annotator agreement.

The classification accuracy of :mod:`tensorphysics.metrology.criteria_semantics.degeneracy`
is measured against a frozen 40-row human-labelled set
(``artifacts/criteria_metrology/label_set.json``).  Two numbers are reported:

* **agreement A vs classifier** — the acceptance criterion (``>= 90 %``);
* **Cohen's kappa A vs B**, where B is an *independent text-only* labeller
  (:func:`label_text_only`).  B uses string markers only — no numeric witnesses, no ratio
  arithmetic — so it is a genuinely different information channel from the classifier, which
  works from the reconstructed numbers.  B is **not** a human annotator; it is a disclosed
  proxy for one, and the report says so.

Any single annotator carries the risk of assimilating the classifier's own definitions; the
honest way to bound that risk is to publish the labels, the disagreement list, and kappa, and
to state that no independent human was available.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .degeneracy import classify
from .model import Degeneracy
from .parser import CriterionRow, criterion_from_dict

DEFAULT_LABEL_SET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "artifacts", "criteria_metrology", "label_set.json")

LABELS = ("NONE", "T1", "T2", "T3")

_TEXT_T1 = (r"无基准", r"无实测", r"无实验", r"无解析", r"无离线", r"不适用", r"无法判定",
            r"未判定", r"WITHHELD", r"精确\s*0", r"兴波阻力=0")
_TEXT_T2 = (r"变号", r"跨零", r"符号", r"sign[- ]flip")
_TEXT_T3 = (r"区间", r"范围", r"带中心", r"带内", r"图谱带", r"放宽", r"判据带", r"–", r"—")


def _norm(label: str) -> str:
    label = (label or "").strip().upper()
    if label.startswith("T1"):
        return "T1"
    if label.startswith("T2"):
        return "T2"
    if label.startswith("T3"):
        return "T3"
    return "NONE"


def label_text_only(quantity: str = "", reference: str = "", error: str = "",
                    verdict: str = "") -> str:
    """Annotator B: label from textual markers only (no numbers, no arithmetic)."""
    cells = " ".join(str(c or "") for c in (quantity, reference, error, verdict))
    # order: a missing/zero reference dominates, then sign semantics, then plural standards
    if any(re.search(p, cells, re.I) for p in _TEXT_T1):
        return "T1"
    if any(re.search(p, cells) for p in _TEXT_T2):
        return "T2"
    if any(re.search(p, reference or "") for p in (r"区间", r"范围", r"带中心")):
        return "T3"
    if any(re.search(p, error or "") for p in (r"区间", r"带内")):
        return "T3"
    if any(re.search(p, verdict or "") for p in (r"放宽", r"判据带", r"工程精度内")):
        return "T3"
    return "NONE"


def confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Confusion matrix + agreement + Cohen's kappa over label pairs (a, b)."""
    pairs = list(pairs)
    n = len(pairs)
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    for a, b in pairs:
        matrix[a][b] += 1
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n if n else 0.0
    pa = {k: sum(matrix[k].values()) / n if n else 0.0 for k in LABELS}
    pb = {k: sum(matrix[a][k] for a in LABELS) / n if n else 0.0 for k in LABELS}
    p_e = sum(pa[k] * pb[k] for k in LABELS)
    kappa = ((p_o - p_e) / (1 - p_e)) if p_e < 1 else 0.0
    return {"n": n, "agree": agree, "agreement": p_o, "kappa": kappa,
            "matrix": matrix,
            "disagreements": [{"a": a, "b": b} for a, b in pairs if a != b]}


def load_label_set(path: str = DEFAULT_LABEL_SET) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def score_labels(label_set: dict[str, Any], *,
                 self_referential_as_degeneracy: bool = False) -> dict[str, Any]:
    """Compare the classifier and annotator B against the frozen human labels.

    The label set is self-contained (each record carries the parsed criterion and the raw
    cells), so this is immune to corpus regeneration.
    """
    records = label_set["records"]
    a_vs_clf: list[tuple[str, str]] = []
    a_vs_b: list[tuple[str, str]] = []
    clf_vs_b: list[tuple[str, str]] = []
    per_row = []
    for k, rec in enumerate(records):
        slot = f"L{k:02d}"
        row = criterion_from_dict(rec["criterion"])
        prov = row.provenance
        dg = classify(row.criterion, ref_text=prov.get("reference_cell"),
                      verdict_text=prov.get("verdict_cell"),
                      error_text=prov.get("error_cell"),
                      self_referential_as_degeneracy=self_referential_as_degeneracy)
        clf = dg.kind.value.split("_")[0]
        b = label_text_only(prov.get("quantity_cell", ""), prov.get("reference_cell", ""),
                            prov.get("error_cell", ""), prov.get("verdict_cell", ""))
        a = _norm(rec.get("label") or "NONE")
        a_vs_clf.append((a, clf))
        a_vs_b.append((a, b))
        clf_vs_b.append((clf, b))
        per_row.append({"slot": slot, "label_A": a, "label_B": b, "classifier": clf,
                        "name": row.name, "note": rec.get("note", ""),
                        "reference_cell": prov.get("reference_cell"),
                        "error_cell": prov.get("error_cell")})
    return {
        "n_labels": len(records),
        "annotator_A_vs_classifier": confusion(a_vs_clf),
        "annotator_A_vs_B": confusion(a_vs_b),
        "classifier_vs_B": confusion(clf_vs_b),
        "per_row": per_row,
    }