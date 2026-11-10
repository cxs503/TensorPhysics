"""P266W1-N09 — criterion metrology: metric semantics, degeneracy taxonomy, adaptive family.

Acceptance-criterion tests, all self-contained (no network, no domain repo checkout):

* metric semantics and the "no unbounded number leaves the module" invariant,
* the T1/T2/T3 degeneracy classifier (including the containment / exact-zero exemptions),
* the adaptive criterion family (magnitude normalisation, symmetric log band, one-sided
  bound, deterministic multi-standard merge) and its three-valued verdicts,
* the 24-sample negative-control set (12 degenerate / 12 benign),
* the 40-row human-labelled set (agreement + Cohen's kappa),
* the replay re-adjudicator and its degeneracy ledger,
* the four historical case studies (m7241 explicit FAIL, 4/4 direction consistency).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorphysics.metrology.criteria_semantics import (  # noqa: E402
    Criterion, CriterionRow, Degeneracy, Measurement, Metric, Verdict,
    classify, controls as controls_mod, corpus as corpus_mod, evaluate, evaluate_baseline,
    labeling, replay as replay_mod, relative_error,
)
from tensorphysics.metrology.criteria_semantics import parser as parser_mod  # noqa: E402
from tensorphysics.metrology.criteria_semantics.controls import run_controls  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "criteria_metrology")


# --------------------------------------------------------------------------- metrics
def test_relative_error_refuses_zero_reference():
    assert relative_error(1.0, 0.0) is None
    assert relative_error(0.0, 0.0) is None
    assert relative_error(None, 1.0) is None
    assert relative_error(1.1, 1.0) == pytest.approx(0.1)


def test_measurement_effective_scale_is_never_zero_or_infinite():
    assert Measurement(y=0.0, y_ref=0.0).effective_scale() == 1.0
    assert Measurement(y=2.0, y_ref=0.0).effective_scale() == 2.0
    assert Measurement(y=0.0, y_ref=0.0, scale_y=5.0).effective_scale() == 5.0


def test_measurement_rejects_non_numeric_and_absorbs_nan():
    with pytest.raises(TypeError):
        Measurement(y="1.0", y_ref=1.0)
    m = Measurement(y=float("nan"), y_ref=1.0)
    assert m.y is None            # NaN is neither a value nor a claim


# --------------------------------------------------------------------------- parser
@pytest.mark.parametrize("cell,kind,value", [
    ("0.0172%", "relative", 0.000172),
    ("7754.92%", "relative", 77.5492),
    ("7.4e-10", "ratio", 7.4e-10),
    ("2.5e-06", "ratio", 2.5e-06),
    ("33 pass, 0 fail", "absolute", 33.0),
    ("—", "none", None),
])
def test_parse_error_cell(cell, kind, value):
    out = parser_mod.parse_error_cell(cell)
    assert out["kind"] == kind
    if value is None:
        assert out["value"] is None
    else:
        assert out["value"] == pytest.approx(value)


def test_parse_verdict_cell_vocabulary():
    assert parser_mod.parse_verdict_cell("**是**")["verdict"] == "pass"
    assert parser_mod.parse_verdict_cell("✅ 是")["verdict"] == "pass"
    assert parser_mod.parse_verdict_cell("否（线 5%）")["verdict"] == "fail"
    assert parser_mod.parse_verdict_cell("⚠️ 部分")["verdict"] == "partial"
    assert parser_mod.parse_verdict_cell("不适用")["verdict"] == "withheld"
    # the looser "判据带" tolerance stated inside the verdict cell is captured
    out = parser_mod.parse_verdict_cell("**否**（>3%，但 <10% 判据带）")
    assert out["taus"] == [pytest.approx(0.10)]
    # ... and at row level it becomes a *competing* tolerance against the 3 % header
    row = parser_mod.parse_row(["关键量", "基准", "误差", "是否 <3%"],
                              ["STAWAVE-2", "解析平台", "重叠域最差 5.99%",
                               "**否**（>3%，但 <10% 判据带）"])
    assert row.criterion.tau == pytest.approx(0.03)
    assert 0.10 in row.criterion.competing_taus


def test_parse_reference_cell_range_and_tolerance_only():
    rng = parser_mod.parse_reference_cell("试验 0.04-0.06")
    assert rng["range"] == (0.04, 0.06)
    tol = parser_mod.parse_reference_cell("≤10%")
    assert tol["absent"] is True and tol.get("tolerance_only") is True
    assert parser_mod.parse_reference_cell("")["absent"] is True


REPORT_MD = """
## 6. 误差判定

| 判据 | 要求 | 实际 | 是否满足 |
|---|---|---|---|
| (a) 解析锚 | <2% | B_F 解析锚误差 <1e-10 | 是 |
| (b) 阻尼比 vs 试验 | <10% | KVLCC2: 0.0175 vs 0.04-0.06 (偏差 ~60%) | ⚠️ 部分 |
| (c) 无基准项 | <3% | — | 不适用 |
"""


def test_parse_report_criteria_extracts_three_rows():
    rows = parser_mod.parse_report_criteria(REPORT_MD, source={"report_id": 1, "mission_id": 2})
    assert len(rows) == 3
    by_name = {r.name: r for r in rows}
    damp = next(r for n, r in by_name.items() if n.startswith("(b)"))
    assert damp.criterion.measurement.y == pytest.approx(0.0175)
    assert damp.criterion.measurement.y_ref == pytest.approx(0.04)
    assert damp.criterion.measurement.ref_range == (0.04, 0.06)
    assert damp.recorded_verdict == "partial"
    missing = next(r for n, r in by_name.items() if n.startswith("(c)"))
    assert missing.criterion.measurement.reference_absent is True


def test_parse_row_rejects_non_error_rows():
    header = ["判据", "要求", "实际", "是否满足"]
    assert parser_mod.parse_row(header, ["(d) pytest", "≥15 pass", "33 pass, 0 fail", "是"]) is None


# ------------------------------------------------------------------- degeneracy
def test_t1_reference_at_noise_floor():
    crit = Criterion(measurement=Measurement(y=0.0023, y_ref=0.0009, sigma_noise=0.001,
                                             scale_y=1.0), tau=0.05)
    dg = classify(crit)
    assert dg.kind is Degeneracy.T1_REFERENCE_NEAR_ZERO
    assert dg.witnesses["why"] == "below_noise_floor"


def test_t2_opposite_signs_and_declared_sign_sensitivity():
    crit = Criterion(measurement=Measurement(y=-0.4, y_ref=0.9), tau=0.03)
    assert classify(crit).kind is Degeneracy.T2_SIGN_CROSSING
    crit2 = Criterion(measurement=Measurement(y=0.02, y_ref=0.01), tau=0.5, sign_sensitive=True)
    assert classify(crit2).kind is Degeneracy.T2_SIGN_CROSSING


def test_t3_competing_tolerances_and_reference_interval():
    crit = Criterion(measurement=Measurement(y=1.06, y_ref=1.0), tau=0.03,
                     competing_taus=(0.10,))
    assert classify(crit).kind is Degeneracy.T3_MULTI_METRIC_NO_PRIORITY
    crit2 = Criterion(measurement=Measurement(y=0.0175, y_ref=0.04, ref_range=(0.04, 0.06)),
                      tau=0.10)
    assert classify(crit2).kind is Degeneracy.T3_MULTI_METRIC_NO_PRIORITY


def test_containment_criterion_is_not_degenerate():
    crit = Criterion(measurement=Measurement(y=None, y_ref=None, ref_range=(0.4, 0.8)), tau=0.03)
    dg = classify(crit, error_text="落在带内")
    assert dg.kind is Degeneracy.NONE
    assert dg.witnesses["why"] == "containment_criterion"


def test_exact_zero_error_is_metric_invariant():
    crit = Criterion(measurement=Measurement(y=0.0, y_ref=0.0), tau=0.03,
                     reported_error_is_zero=True)
    assert classify(crit).kind is Degeneracy.NONE


# ------------------------------------------------------------------- adaptive family
def test_no_partial_verdict_exists():
    assert {v.value for v in Verdict} == {"PASS", "FAIL", "WITHHELD"}


def test_t1_remedy_is_bounded_magnitude_normalised():
    crit = Criterion(measurement=Measurement(y=0.9, y_ref=1e-12, sigma_noise=1e-13, scale_y=1.0),
                     tau=0.03)
    ev = evaluate(crit)
    assert ev.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.WITHHELD)
    assert ev.error is not None and math.isfinite(ev.error)
    assert "magnitude_normalised" in ev.remedy


def test_t1_noise_indistinguishable_withholds():
    crit = Criterion(measurement=Measurement(y=0.0023, y_ref=0.0009, sigma_noise=0.001,
                                             scale_y=1.0), tau=0.05)
    ev = evaluate(crit)
    assert ev.verdict is Verdict.WITHHELD
    assert ev.reason == "difference_within_noise_floor"


def test_t2_log_ratio_band_and_sign_flip_forces_fail():
    crit = Criterion(measurement=Measurement(y=1.02, y_ref=1.0), tau=0.03, sign_sensitive=True)
    ev = evaluate(crit)
    assert ev.metric is Metric.LOG_RATIO
    assert ev.verdict is Verdict.PASS
    flip = Criterion(measurement=Measurement(y=-0.3, y_ref=0.3), tau=0.5)
    ev2 = evaluate(flip)
    assert ev2.verdict is Verdict.FAIL
    assert "sign_flip" in ev2.reason


def test_t3_strictest_tolerance_wins():
    crit = Criterion(measurement=Measurement(y=1.06, y_ref=1.0), tau=0.03,
                     competing_taus=(0.10,))
    ev = evaluate(crit)
    assert ev.tau == pytest.approx(0.03)
    assert ev.verdict is Verdict.FAIL           # 6 % passes at 10 %, fails at 3 %
    assert "strictest_tau_wins" in ev.remedy


def test_bound_metric_for_one_sided_thresholds():
    ok = Criterion(measurement=Measurement(y=1352.0, y_ref=3600.0, unit="m"), tau=0.0,
                   metric=Metric.BOUND)
    assert evaluate(ok).verdict is Verdict.PASS
    bad = Criterion(measurement=Measurement(y=3900.0, y_ref=3600.0, unit="m"), tau=0.0,
                    metric=Metric.BOUND)
    assert evaluate(bad).verdict is Verdict.FAIL


def test_missing_measurement_and_reference_withhold():
    assert evaluate(Criterion(measurement=Measurement(y=None, y_ref=1.0), tau=0.03)).verdict \
        is Verdict.WITHHELD
    ev = evaluate(Criterion(measurement=Measurement(y=1.0, y_ref=None), tau=0.03))
    assert ev.verdict is Verdict.WITHHELD and "reference_absent" in ev.reason


def test_metric_mismatch_withholds_not_passes():
    crit = Criterion(measurement=Measurement(y=7.4e-10, y_ref=None), tau=0.03,
                     metric=Metric.ABSOLUTE, tau_available=False)
    ev = evaluate(crit)
    assert ev.verdict is Verdict.WITHHELD
    assert ev.reason == "tolerance_missing_in_metric_space"


# --------------------------------------------------- invariance / NaN propagation
def test_invariance_no_unbounded_relative_error_when_reference_vanishes():
    """Property test: y_ref -> 0 must never produce an unbounded relative error."""
    rng = random.Random(20261110)
    bad = []
    for _ in range(1200):
        # log-uniform reference magnitudes from 1e-300 up to 1e3, plus exact zeros
        exp = rng.uniform(-300, 3)
        y_ref = 0.0 if rng.random() < 0.1 else math.copysign(10 ** exp, rng.choice((-1, 1)))
        y = math.copysign(10 ** rng.uniform(-300, 3), rng.choice((-1, 1)))
        crit = Criterion(measurement=Measurement(y=y, y_ref=y_ref), tau=rng.choice((0.0, 0.03, 0.1)))
        ev = evaluate(crit)
        if ev.verdict not in (Verdict.PASS, Verdict.FAIL, Verdict.WITHHELD):
            bad.append(("verdict", ev))
        if ev.error is not None and not math.isfinite(ev.error):
            bad.append(("error", ev))
        if ev.relative_error is not None and not math.isfinite(ev.relative_error):
            bad.append(("relative_error", ev))
        if y_ref == 0.0 and ev.relative_error is not None:
            bad.append(("relative_error_defined_at_zero_reference", ev))
    assert not bad, bad[:3]


@pytest.mark.parametrize("y,y_ref", [
    (float("nan"), 1.0), (1.0, float("nan")), (float("inf"), 1.0), (1.0, float("inf")),
    (None, None), (0.0, 0.0),
])
def test_nan_none_inf_propagation_is_safe(y, y_ref):
    crit = Criterion(measurement=Measurement(y=y, y_ref=y_ref), tau=0.03)
    ev = evaluate(crit)
    assert ev.verdict in (Verdict.PASS, Verdict.FAIL, Verdict.WITHHELD)
    assert ev.error is None or math.isfinite(ev.error)


# ------------------------------------------------------------------- controls
def test_negative_controls_degenerate_12_of_12():
    out = run_controls()
    assert out["degenerate_detected"] == 12, out["results"]
    types = {r.expected for r in out["results"] if r.group == "degenerate"}
    assert types == {"T1_REFERENCE_NEAR_ZERO", "T2_SIGN_CROSSING",
                     "T3_MULTI_METRIC_NO_PRIORITY"}


def test_negative_controls_benign_false_positives_zero():
    out = run_controls()
    assert out["benign_false_positive"] == 0
    assert out["benign_verdict_ok"] == 12


# ------------------------------------------------------------------- labels
def test_human_label_agreement_and_kappa():
    ls = labeling.load_label_set()
    assert len(ls["records"]) == 40
    res = labeling.score_labels(ls)
    a_clf = res["annotator_A_vs_classifier"]
    assert a_clf["agreement"] >= 0.90, a_clf
    assert a_clf["kappa"] >= 0.60, a_clf
    a_b = res["annotator_A_vs_B"]
    assert a_b["n"] == 40 and 0.0 <= a_b["kappa"] <= 1.0


# ------------------------------------------------------------------- replay
def _row(y, y_ref, tau, name, recorded="unknown", metric=Metric.RELATIVE, **kw):
    m = Measurement(y=y, y_ref=y_ref, **{k: v for k, v in kw.items()
                                         if k in ("scale_y", "sigma_noise", "ref_range", "unit")})
    crit = Criterion(measurement=m, tau=tau, metric=metric, name=name,
                     competing_taus=kw.get("competing_taus", ()))
    return CriterionRow(criterion=crit, recorded_verdict=recorded,
                        provenance={"report_id": 1, "mission_id": 9, "error_cell": "",
                                    "reference_cell": "", "verdict_cell": ""})


def test_replay_ledger_counts_and_flips():
    rows = [
        _row(0.0175, 0.04, 0.10, "damping", recorded="partial", ref_range=(0.04, 0.06)),
        _row(1.0004, 1.0, 0.000001, "identity"),
        _row(None, None, 0.03, "withheld", recorded="withheld"),
    ]
    ledger = replay_mod.replay_corpus(rows)
    assert ledger["n_criteria"] == 3
    assert ledger["n_degenerate_strict"] >= 1
    entries = ledger["entries"]
    damping = entries[0]
    assert damping.adaptive_verdict is Verdict.FAIL
    assert damping.recorded_flip == replay_mod.REC_SOFTENED_TO_FAIL
    assert damping.alternative_value is not None and damping.alternative_value > 1.0
    summary = replay_mod.summarize_ledger(ledger)
    assert isinstance(json.dumps(summary), str)      # JSON-serialisable


def test_replay_is_verdict_preserving_when_relative_error_is_well_defined():
    """The adaptive family must not silently re-grade non-degenerate rows."""
    rng = random.Random(7)
    for _ in range(200):
        y_ref = rng.uniform(0.5, 5.0)
        y = y_ref * (1 + rng.uniform(-0.5, 0.5))
        tau = rng.choice((0.01, 0.03, 0.1))
        crit = Criterion(measurement=Measurement(y=y, y_ref=y_ref), tau=tau)
        assert evaluate_baseline(crit).verdict is evaluate(crit).verdict


# ------------------------------------------------------------------- case studies
def test_case_studies_m7241_fails_and_directions_consistent():
    path = os.path.join(ART, "case_studies.json")
    if not os.path.exists(path):
        pytest.skip("case-study artifact not built")
    with open(path, encoding="utf-8") as fh:
        art = json.load(fh)
    m7241 = art["cases"]["m7241"]
    assert m7241, "m7241 case missing"
    assert m7241[0]["adaptive"]["verdict"] == "FAIL"
    assert m7241[0]["relative_error_pct"] > 50.0        # the 60 % deviation is explicit
    assert m7241[0]["recorded_verdict"] in ("partial", "pass")
    assert art["m7241_explicit_fail"] is True
    assert art["n_direction_consistent"] == art["n_cases"]


def test_frozen_corpus_is_loadable_and_large_enough():
    path = os.path.join(ART, "historical_criteria.jsonl")
    if not os.path.exists(path):
        pytest.skip("corpus artifact not built")
    rows = corpus_mod.load_corpus(path)
    assert len(rows) >= 200