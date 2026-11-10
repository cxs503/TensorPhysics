"""Unit tests for the criterion parser and the metric-semantic model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.metrology.criteria import (  # noqa: E402
    ABSOLUTE,
    LOG_RATIO,
    RELATIVE,
    SIGNED,
    UNPARSED,
    CriterionRecord,
    extract_quadruple,
    finite,
    parse_criterion,
    parse_goal_criteria,
    parse_markdown_criteria_table,
    sign,
)


@pytest.mark.parametrize(
    "text,metric,tolerance",
    [
        ("相对误差 < 2%", RELATIVE, 0.02),
        ("rel-L2 temperature error < 3 %", RELATIVE, 0.03),
        ("relative error < 1e-5", RELATIVE, 1e-5),
        ("|Δm|/Σm ≤ 1e-12", RELATIVE, 1e-12),
        ("|w_FEM - w_Hetényi| / w_Hetényi < 0.02", RELATIVE, 0.02),
        ("max rel. error < 0.5 %", RELATIVE, 0.005),
        ("thermal sub-chain residual < 1e-12", RELATIVE, 1e-12),
    ],
)
def test_relative_forms(text, metric, tolerance):
    parsed = parse_criterion(text)
    assert parsed.metric == metric
    assert parsed.tolerance == pytest.approx(tolerance)
    assert parsed.tolerance_kind == "fraction"


@pytest.mark.parametrize(
    "text,tolerance",
    [
        ("max|A−B| ≤ 1e-12", 1e-12),
        ("eps_max < 1e-3", 1e-3),
        ("max|w(f1+f2) - w(f1) - w(f2)| ≤ 1e-10", 1e-10),
    ],
)
def test_absolute_forms(text, tolerance):
    parsed = parse_criterion(text)
    assert parsed.metric == ABSOLUTE
    assert parsed.tolerance == pytest.approx(tolerance)
    assert parsed.tolerance_kind == "absolute"


def test_log_and_signed_forms():
    log = parse_criterion("|ln(y/y_ref)| ≤ ln(1+0.03)")
    assert log.metric == LOG_RATIO and log.tolerance == pytest.approx(0.03)
    signed = parse_criterion("带符号偏差 |y−y_ref| ≤ 0.5 kN")
    assert signed.metric == SIGNED
    assert signed.tolerance is None


def test_unparsed_is_a_sentinel():
    assert parse_criterion("") is UNPARSED
    assert parse_criterion(None) is UNPARSED
    assert parse_criterion("no threshold here").metric == "unparsed"
    assert parse_criterion("no threshold here").has_tolerance is False


def test_quadruple_keeps_unit_and_condition():
    rec = extract_quadruple("max|Δ| <= 0.5 kN 在 Re=1e5 下", 1.0, 1.0)
    assert rec.parsed.tolerance == pytest.approx(0.5)
    assert rec.unit == "kN"
    assert rec.parsed.condition is not None
    assert rec.rid


def test_goal_criteria_parser():
    goal = """
    (a) 退化分类一致率≥90%（kappa可报）；
    (b) 回放≥200条历史判据；
    (c) 方向一致性4/4；
    (d) 负控：退化样本12/12命中，良性样本误报0/12；
    (e) 属性测试≥1000个随机样本
    """
    parsed = parse_goal_criteria(goal)
    kinds = [g.kind for g in parsed]
    assert "rate" in kinds
    assert "samples" in kinds
    assert any(g.num == 12 and g.den == 12 for g in parsed)
    rate = [g for g in parsed if g.kind == "rate"][0]
    assert rate.value == pytest.approx(0.90)
    assert parse_goal_criteria("") == []


def test_markdown_table_parse():
    md = """
| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| 位移 L2 误差 | <1% | 0.0034 | PASS |
"""
    rows = parse_markdown_criteria_table(md)
    assert rows and "判据" in rows[0]["header"]
    assert parse_markdown_criteria_table("no table") == []


def test_model_helpers():
    assert finite(None) is None
    assert finite(float("nan")) is None
    assert finite(float("inf")) is None
    assert finite(True) is None
    assert finite("1.5") == 1.5
    assert sign(-2.0) == -1 and sign(0.0) == 0 and sign(None) is None


def test_record_derived_relative_error():
    rec = CriterionRecord(rid="r", source="s", measurand="m", criterion="相对误差 < 2%", y=1.0, y_ref=0.0)
    assert rec.derived_relative_error() == math.inf
    rec0 = CriterionRecord(rid="r", source="s", measurand="m", criterion="c", y=0.0, y_ref=0.0)
    assert rec0.derived_relative_error() == 0.0
    recn = CriterionRecord(rid="r", source="s", measurand="m", criterion="c", y=None, y_ref=1.0)
    assert recn.derived_relative_error() is None
    assert recn.has_pair() is False


def test_family_key_ignores_row_ids():
    a = CriterionRecord(rid="r1", source="f.md", measurand="C1", criterion="相对误差 < 2%")
    b = CriterionRecord(rid="r2", source="f.md", measurand="C1", criterion="max|Δ| <= 1e-9")
    assert a.family_key != b.family_key
    c = CriterionRecord(rid="r3", source="f.md", measurand="位移 L2 相对误差", criterion="x")
    d = CriterionRecord(rid="r4", source="f.md", measurand="位移 L2 相对误差", criterion="y")
    assert c.family_key == d.family_key