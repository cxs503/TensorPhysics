"""Unit tests for the degeneracy classifier and the adaptive criterion family."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.metrology.criteria import (  # noqa: E402
    ABSOLUTE,
    FAIL,
    LOG_RATIO,
    PASS,
    RELATIVE,
    SIGNED,
    T1,
    T2,
    T3,
    WITHHELD,
    CriterionRecord,
    Degeneracy,
    MetricEvaluation,
    Verdict,
    analyze,
    bounded_relative_error,
    classify,
    deviation_fraction,
    direction,
    evaluate_controls,
    extract_quadruple,
    judge,
    judge_family,
    log_ratio,
    merge_verdicts,
    multi_metric_families,
    resolve_scale,
    scaled_delta,
    signed_abs,
)


# --------------------------------------------------------------------------------------
# degeneracy
# --------------------------------------------------------------------------------------
def test_t1_zero_reference_and_noise_floor():
    rec = extract_quadruple("相对误差 < 1%", 1e-3, 1e-9, scale=1.0)
    deg = classify(rec)
    assert deg.has(T1)
    assert deg.scale == pytest.approx(1.0)
    assert deg.noise == pytest.approx(1e-6)
    assert "T1_zero_reference" in deg.details


def test_t1_scale_fraction_rule():
    rec = extract_quadruple("相对误差 < 1%", 5e-3, 1e-2, scale=1.0)
    assert classify(rec).has(T1)  # 1 % of the engineering scale


def test_t1_undecidable_without_reference():
    rec = extract_quadruple("相对误差 < 1%", None, None)
    deg = classify(rec)
    assert deg.undecidable == (T1,)
    assert deg.flagged is True and deg.provable is False


def test_t1_not_applicable_for_absolute_metric():
    rec = extract_quadruple("max|Δ| <= 1e-12", 1e-13, 0.0, scale=1e-12)
    deg = classify(rec)
    assert not deg.has(T1) and not deg.has(T2)
    assert deg.flagged is False


def test_t2_sign_flip():
    rec = extract_quadruple("相对误差 < 5%", -1.0, 1.0, scale=1.0)
    deg = classify(rec)
    assert deg.has(T2)
    assert "sign" in deg.details[T2]


def test_t2_zero_crossing():
    rec = extract_quadruple("相对误差 < 5%", 0.5, 0.0, scale=10.0)
    deg = classify(rec)
    assert deg.has(T2) and deg.has(T1)


def test_t3_family_detection():
    a = extract_quadruple("相对误差 < 2%", 1.05, 1.0, family="fam", rid="a")
    b = extract_quadruple("max|Δ| <= 0.5", 1.05, 1.0, family="fam", rid="b")
    c = CriterionRecord(rid="c", source="s", measurand="m", criterion="x", family="fam2",
                        evidence={"priority": "relative"})
    d = extract_quadruple("相对误差 < 2%", 1.0, 1.0, family="fam2", rid="d")
    assert "fam" in multi_metric_families([a, b])
    assert "fam2" not in multi_metric_families([c, d])
    deg = analyze([a, b])[0][1]
    assert deg.has(T3)


def test_resolve_scale_priority():
    rec = extract_quadruple("相对误差 < 2%", 1.0, 1.0, scale=2.0)
    scale, noise, prov = resolve_scale(rec, family_scale=3.0)
    assert scale == pytest.approx(2.0) and prov == "declared"
    scale2, _noise2, prov2 = resolve_scale(extract_quadruple("相对误差 < 2%", 1.0, 1.0), family_scale=3.0)
    assert scale2 == pytest.approx(3.0) and prov2 == "family"
    scale3, _n3, prov3 = resolve_scale(extract_quadruple("相对误差 < 2%", 4.0, -2.0))
    assert scale3 == pytest.approx(4.0) and prov3 == "local"


def test_degeneracy_dataclass_helpers():
    deg = Degeneracy(classes=(T1,), undecidable=(T2,), details={"T1_zero_reference": "hit: x"})
    assert deg.flagged and deg.provable and deg.has(T1)
    payload = deg.as_dict()
    assert payload["classes"] == [T1] and payload["flagged"] is True


# --------------------------------------------------------------------------------------
# adaptive metrics
# --------------------------------------------------------------------------------------
def test_scaled_delta_is_bounded_and_scale_aware():
    assert scaled_delta(1.1, 1.0) == pytest.approx(0.1)
    assert scaled_delta(1.0, 0.0, scale=1.0) == pytest.approx(10.0)
    assert scaled_delta(1.0, 0.0) is None
    assert scaled_delta(None, 1.0) is None
    assert scaled_delta(1.0, 1.0, noise=1e-3) == pytest.approx(0.0)


def test_log_ratio_domain():
    assert log_ratio(2.0, 1.0) == pytest.approx(math.log(2.0))
    assert log_ratio(1.0, -1.0) is None
    assert log_ratio(0.0, 1.0) is None
    assert log_ratio(None, 1.0) is None
    assert log_ratio(1.0, 1.0, noise=2.0) is None


def test_signed_abs_and_direction():
    assert signed_abs(0.99, 1.0) == pytest.approx(-0.01)
    assert signed_abs(None, 1.0) is None
    assert direction(1.1, 1.0) == 1
    assert direction(0.9, 1.0) == -1
    assert direction(1.0, 1.0) == 0
    assert direction(None, 1.0) is None


def test_bounded_relative_error_semantics():
    assert bounded_relative_error(1.1, 1.0) == (pytest.approx(0.1), "defined")
    assert bounded_relative_error(1.0, 0.0)[1] == "withheld"
    assert bounded_relative_error(0.0, 0.0) == (0.0, "defined")
    value, status = bounded_relative_error(1.0, 1e-30)  # reference far below the floor
    assert status == "withheld" and value is None


def test_deviation_fraction():
    assert deviation_fraction(1.2, 1.0) == pytest.approx(0.2)
    assert deviation_fraction(0.5, 0.0, scale=1.0) == pytest.approx(0.5)
    assert deviation_fraction(None, 1.0) is None


# --------------------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------------------
def test_judge_regular_pass_and_fail():
    assert judge(extract_quadruple("相对误差 < 2%", 1.01, 1.0)).status == PASS
    assert judge(extract_quadruple("相对误差 < 2%", 1.05, 1.0)).status == FAIL


def test_judge_t1_uses_absolute_band():
    rec = extract_quadruple("相对误差 < 1%", 1e-3, 1e-9, scale=1.0)
    verdict = judge(rec)
    assert verdict.status == PASS
    assert verdict.metric == SIGNED
    assert verdict.threshold == pytest.approx(0.01)


def test_judge_t2_sign_flip_is_fail():
    verdict = judge(extract_quadruple("相对误差 < 5%", -1.0, 1.0, scale=1.0))
    assert verdict.status == FAIL
    assert verdict.metric == SIGNED
    assert verdict.direction == -1


def test_judge_t2_inside_noise_is_withheld():
    rec = extract_quadruple("相对误差 < 5%", -1e-9, 1e-9, scale=1.0, noise=1e-6)
    assert judge(rec).status == WITHHELD


def test_judge_recorded_relative_error_uses_log_bound():
    rec = extract_quadruple("相对误差 < 2%", None, None, reported_error=0.0167)
    verdict = judge(rec)
    assert verdict.status == PASS and verdict.metric == LOG_RATIO
    hard = extract_quadruple("相对误差 < 2%", None, None, reported_error=0.05)
    assert judge(hard).status == FAIL


def test_judge_absolute_deviation_without_reference():
    rec = extract_quadruple("max|Δ| <= 1e-12", 1e-13, None)
    verdict = judge(rec)
    assert verdict.status == PASS and verdict.metric == SIGNED
    bad = extract_quadruple("max|Δ| <= 1e-12", 1e-9, None)
    assert judge(bad).status == FAIL


def test_judge_no_evidence_is_withheld():
    rec = extract_quadruple("相对误差 < 2%", None, None)
    verdict = judge(rec)
    assert verdict.status == WITHHELD and verdict.reasons


def test_judge_is_domain_neutral():
    """Same numbers, different unit label: identical verdict (no domain vocabulary)."""
    a = extract_quadruple("max|Δ| <= 0.5 kN", 1.2, 1.0, scale=1.0)
    b = extract_quadruple("max|Δ| <= 0.5 V", 1.2, 1.0, scale=1.0)
    assert judge(a).status == judge(b).status == PASS


# --------------------------------------------------------------------------------------
# merge / family
# --------------------------------------------------------------------------------------
def test_merge_verdicts_strictest_and_absolute_first():
    assert merge_verdicts([Verdict(PASS, RELATIVE), Verdict(FAIL, ABSOLUTE)]).status == FAIL
    assert merge_verdicts([Verdict(WITHHELD, RELATIVE)]).status == WITHHELD
    assert merge_verdicts([]).status == WITHHELD
    ordered = merge_verdicts([Verdict(FAIL, RELATIVE), Verdict(PASS, ABSOLUTE)], policy="absolute_first")
    assert ordered.metric == ABSOLUTE


def test_judge_family_reports_members_and_merged():
    a = extract_quadruple("相对误差 < 2%", 1.001, 1.0, rid="a", family="f")
    b = extract_quadruple("max|Δ| <= 1e-4", 1.001, 1.0, rid="b", family="f")
    result = judge_family([a, b])
    assert result["merged"].status == FAIL
    assert result["partial_pass_risk"] is True
    assert result["n_members"] == 2
    assert [m["rid"] for m in result["members"]] == ["a", "b"]


def test_evaluate_controls_adds_no_partial_states():
    payload = evaluate_controls()
    for spec in payload["degenerate_detail"]:
        assert spec["verdict"] in (PASS, FAIL, WITHHELD)
    for spec in payload["benign_detail"]:
        assert spec["verdict"] in (PASS, FAIL, WITHHELD)