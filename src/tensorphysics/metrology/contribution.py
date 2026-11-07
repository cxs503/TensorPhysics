"""Scientific-contribution metrology for the Tensor agent ecosystem.

The canonical implementation lives in ``tools/contribution_meter/`` (ontology, evidence
extraction, rule classifier, window aggregation, CLI).  This module is the *library facade*:
it loads those files once and exposes a typed API so other TensorPhysics code (campaign gates,
chronicle reports, dashboards) can meter contributions without shelling out.

Design note: the facade deliberately does **not** re-implement the classifier - a second copy
would drift.  The tool files are the single source of truth; they are pure-stdlib and have no
imports from ``tensorphysics``, so loading them by path is safe.

Example
-------
>>> from tensorphysics.metrology.contribution import ContributionMeter
>>> meter = ContributionMeter()
>>> meter.ontology["classes"]["C1"]["name"]
'verified-case'
>>> meter.classify_report({"mission_id": 1, "report_id": "r", "title": "解析解验证",
...                        "summary": "误差 0.3%", "markdown": ""})["label"]
'C6'
"""
from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

_TOOL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "tools", "contribution_meter")
)
ONTOLOGY_PATH = os.path.join(_TOOL_DIR, "ontology.json")


def _load(name: str):
    path = os.path.join(_TOOL_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location("tp_contribution_" + name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVIDENCE = None
_CLASSIFIER = None


def _modules():
    global _EVIDENCE, _CLASSIFIER
    if _EVIDENCE is None:
        _EVIDENCE = _load("evidence")
        _CLASSIFIER = _load("classifier")
    return _EVIDENCE, _CLASSIFIER


class ContributionMeter:
    """Facade over the ontology, the evidence extractor and the rule classifier."""

    CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
    SCIENTIFIC = frozenset({"C1", "C2", "C3", "C4", "C5"})

    def __init__(self, ontology_path: str | None = None) -> None:
        self._ontology_path = ontology_path or ONTOLOGY_PATH
        with open(self._ontology_path, encoding="utf-8") as fh:
            self.ontology: dict[str, Any] = json.load(fh)
        self.WEIGHTS: dict[str, float] = dict(self.ontology["weights"])

    # -- single report -----------------------------------------------------------------
    def extract(self, report: dict) -> dict:
        """Extract the auditable evidence fields of one report."""
        evidence, _ = _modules()
        return evidence.extract(report)

    def classify_report(self, report: dict, llm_fallback: bool = False) -> dict:
        """Classify one report; returns label + the evidence fields used + reasons against."""
        _, classifier = _modules()
        return classifier.classify_report(report, llm_fallback=llm_fallback)

    # -- window ------------------------------------------------------------------------
    def window_report(self, corpus: list[dict], missions: list[dict] | None = None,
                      window_hours: float = 48.0) -> dict:
        """Aggregate a window: unit counts, zero-contribution enumeration, SCU conversion."""
        _, classifier = _modules()
        return classifier.build_window_report(corpus, missions or [], window_hours)

    # -- gold set ----------------------------------------------------------------------
    def evaluate(self, golden: dict, corpus_by_report: dict) -> dict:
        """Recompute gold-set metrics from scratch (no cached numbers)."""
        _, classifier = _modules()
        gold, preds = {}, {}
        for item in golden["items"]:
            key = str(item["report_id"])
            gold[key] = item["label"]
            preds[key] = self.classify_report(corpus_by_report[key])["label"]
        return classifier.evaluate(preds, gold)

    def weight(self, label: str) -> float:
        return self.WEIGHTS.get(label, 0.0)