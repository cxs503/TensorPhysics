"""Acceptance tests for the criterion metrology package (P266W2-N09).

Machine-checkable acceptance criteria of the node, mapped to test functions:

(a) ``test_degeneracy_agreement_*``      classification agreement >= 90 % (kappa reported)
(b) ``test_replay_corpus_*``             >= 200 replayed historical criteria, >= 5 % flagged,
                                         every flagged row carries an alternative criterion and
                                         a verdict-change flag
(c) ``test_case_studies_*``              4 case studies with alternative numbers, direction
                                         consistency 4/4, and > 50 % deviations must be FAIL
(d) ``test_controls_*``                  12/12 degenerate hits, 0/12 benign false alarms
(e) ``test_invariant_*``                 y_ref -> 0 never yields an unbounded relative error
                                         (>= 1000 randomised samples)
(f) every module of the package is exercised (coverage >= 80 % measured separately) and
    NaN/None propagate to WITHHELD, never to PASS
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.metrology.criteria import (  # noqa: E402
    ABSOLUTE,
    CORPUS_MIN_RECORDS,
    DEGRADED_MIN_FRACTION,
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
    Verdict,
    agreement,
    analyze,
    baseline_verdict,
    bounded_relative_error,
    build_corpus,
    case_studies,
    classify,
    cohen_kappa,
    evaluate_controls,
    evaluate_labeled,
    extract_quadruple,
    judge,
    judge_family,
    log_ratio,
    merge_verdicts,
    parse_criterion,
    parse_goal_criteria,
    replay,
    replay_repository,
    scaled_delta,
    signed_abs,
)


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def repository_replay():
    return replay_repository(str(ROOT))


@pytest.fixture(scope="module")
def controls():
    payload = evaluate_controls()
    payload.pop("_rows", None)
    return payload


@pytest.fixture(scope="module")
def labeled():
    return evaluate_labeled()


# --------------------------------------------------------------------------------------
# (a) classification agreement against the labelled set
# --------------------------------------------------------------------------------------
def test_degeneracy_agreement_gate(labeled):
    assert labeled["n"] == 40
    assert labeled["agreement_A_classifier"] >= 0.90, labeled["agreement_A_classifier"]
    assert labeled["n_agree_A_classifier"] >= 36
    assert labeled["kappa_A_classifier"] >= 0.80
    assert labeled["meets_stop_rule"] is True
    # the pre-registered confound: a text-only annotator cannot see the numbers, so its
    # agreement is reported (and is expected to be much lower)
    assert 0.0 <= labeled["agreement_A_text_annotator"] <= 1.0
    assert labeled["kappa_A_text_annotator"] <= labeled["kappa_A_classifier"]


def test_kappa_helper_edges():
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert cohen_kappa([], []) == 0.0
    assert 0.0 <= cohen_kappa(["x", "x"], ["x", "y"]) <= 1.0
    with pytest.raises(ValueError):
        cohen_kappa(["x"], ["x", "y"])
    assert agreement(["a"], ["a"]) == 1.0


# --------------------------------------------------------------------------------------
# (b) replay of the historical corpus
# --------------------------------------------------------------------------------------
def test_replay_corpus_size_and_degradation(repository_replay):
    kpi = repository_replay.kpi()
    assert kpi["n_records"] >= CORPUS_MIN_RECORDS, kpi["n_records"]
    assert kpi["meets_corpus_size"] is True
    assert kpi["degraded_fraction"] >= DEGRADED_MIN_FRACTION, kpi["degraded_fraction"]
    assert kpi["meets_degraded_fraction"] is True
    # the corpus is entirely real artifacts: every record carries its provenance
    for row in repository_replay.rows:
        assert row.record.source, row.record.rid
        assert Path(row.record.source).exists() or row.record.source.startswith("controls")


def test_every_degraded_row_has_alternative_and_change_flag(repository_replay):
    degraded = repository_replay.degraded_rows()
    assert degraded
    for row in degraded:
        payload = row.as_dict()
        assert "alternative" in payload
        alt = payload["alternative"]
        assert alt["status"] in (PASS, FAIL, WITHHELD)
        assert "flip" in payload and isinstance(payload["flip"], bool)
        assert "verdict_change" in payload and isinstance(payload["verdict_change"], bool)
        # a degraded row is either recomputable (numeric alternative) or explicitly
        # withheld with the reason recorded: never silently "partially passed"
        if alt["value"] is None:
            assert payload["adaptive"]["status"] == WITHHELD
            assert payload["adaptive"]["reasons"], payload["rid"]
    numeric = [r for r in degraded if r.as_dict()["alternative"]["value"] is not None]
    assert len(numeric) >= 20, len(numeric)


def test_replay_is_deterministic(repository_replay):
    records, _ = build_corpus(str(ROOT))
    again = replay(records)
    assert [r.record.rid for r in again.rows] == [r.record.rid for r in repository_replay.rows]
    assert again.kpi()["n_degraded"] == repository_replay.kpi()["n_degraded"]


def test_no_partial_verdict_in_corpus(repository_replay):
    vocabulary = {PASS, FAIL, WITHHELD}
    for row in repository_replay.rows:
        assert row.baseline.status in vocabulary
        assert row.adaptive.status in vocabulary


# --------------------------------------------------------------------------------------
# (c) case studies
# --------------------------------------------------------------------------------------
def test_case_studies_numbers_and_direction(repository_replay):
    controls = evaluate_controls()
    extra = controls.pop("_rows", [])
    cs = case_studies(repository_replay, extra_rows=extra, family_merges=controls["family_merges"])
    assert cs["n_cases"] == 4
    assert cs["n_direction_consistent"] == 4, cs["n_direction_consistent"]
    for case in cs["cases"]:
        alt = case["alternative"]
        assert alt["metric"] in (RELATIVE, ABSOLUTE, LOG_RATIO, SIGNED)
        assert alt["value"] is not None
        assert case["direction_consistent"] is True
    assert cs["n_deviation_gt_50pct"] >= 1
    # the >50 % rule is honoured by every case (vacuously true for small deviations)
    assert cs["n_explicit_fail_satisfied"] == cs["n_cases"]
    for case in cs["cases"]:
        if case["deviation_gt_50pct"]:
            assert case["adaptive"]["status"] == FAIL, case["rid"]


def test_multi_metric_case_merges_to_single_fail(repository_replay):
    controls = evaluate_controls()
    merged = [c for c in case_studies(repository_replay, controls.get("_rows", []),
                                      family_merges=controls["family_merges"])["cases"]
              if c["role"] == "multi_metric_merge"]
    assert merged, "T3 case study missing"
    case = merged[0]
    assert case["family_merge"]["merged_status"] == FAIL
    member_statuses = {m["status"] for m in case["family_merge"]["members"]}
    assert PASS in member_statuses and FAIL in member_statuses
    assert case["family_merge"]["partial_pass_risk"] is True


# --------------------------------------------------------------------------------------
# (d) negative controls
# --------------------------------------------------------------------------------------
def test_controls_degenerate_hits(controls):
    assert controls["n_degenerate"] == 12
    assert controls["n_degenerate_hit"] == 12
    assert controls["degenerate_hit_rate"] == 1.0
    assert controls["meets_degenerate_12_of_12"] is True
    for spec in controls["degenerate_detail"]:
        assert spec["hit"], spec
        assert spec["verdict"] == spec["expect_verdict"], spec


def test_controls_benign_false_alarms(controls):
    assert controls["n_benign"] == 12
    assert controls["n_benign_false_alarm"] == 0
    assert controls["meets_benign_zero_false_alarm"] is True
    for spec in controls["benign_detail"]:
        assert not spec["false_alarm"], spec


def test_controls_supplementary_zero_reference():
    payload = evaluate_controls()
    for spec in payload["supplementary_detail"]:
        assert spec["hit"], spec
        assert spec["verdict_ok"], spec


def test_controls_cli_writes_json(tmp_path):
    out = tmp_path / "controls.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tensorphysics.metrology.criteria", "controls", "--out", str(out)],
        cwd=str(ROOT), env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["n_degenerate_hit"] == 12


# --------------------------------------------------------------------------------------
# (e) invariant: no unbounded relative error at y_ref -> 0
# --------------------------------------------------------------------------------------
def test_invariant_no_unbounded_relative_error():
    rng = random.Random(20261110)
    unbounded = 0
    finite_count = 0
    for _ in range(1200):
        # log-uniform references spanning 1e-18 .. 1e3, including exact zeros
        y_ref = 0.0 if rng.random() < 0.25 else math.copysign(10.0 ** rng.uniform(-18, 3), rng.random() - 0.5)
        y = 10.0 ** rng.uniform(-18, 3) * math.copysign(1.0, rng.random() - 0.5)
        value, status = bounded_relative_error(y, y_ref)
        assert status in ("defined", "withheld")
        if status == "defined":
            assert value is not None and math.isfinite(value)
            finite_count += 1
        else:
            assert value is None
            unbounded += 1
        sd = scaled_delta(y, y_ref)
        if sd is not None:
            assert math.isfinite(sd)
    assert finite_count + unbounded == 1200
    assert unbounded > 0


def test_invariant_tiny_reference_never_passes_via_ratio():
    """A criterion whose denominator is 12 decades below the measurand must not decide."""
    rec = extract_quadruple("相对误差 < 1%", 1.0, 1e-12, scale=1.0)
    deg = classify(rec)
    assert deg.has(T1)
    verdict = judge(rec, deg)
    assert verdict.status in (PASS, FAIL)
    assert verdict.metric == SIGNED  # the degenerate ratio is not the deciding metric
    assert math.isfinite(verdict.value or 0.0)


# --------------------------------------------------------------------------------------
# (f) NaN / None propagation safety
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "not-a-number", True])
def test_nan_none_never_passes(bad):
    rec = extract_quadruple("相对误差 < 2%", bad, 1.0)
    verdict = judge(rec)
    assert verdict.status == WITHHELD
    rec2 = extract_quadruple("相对误差 < 2%", 1.0, bad)
    assert judge(rec2).status == WITHHELD
    rec3 = extract_quadruple("相对误差 < 2%", bad, bad)
    assert judge(rec3).status == WITHHELD
    assert math.isfinite(bounded_relative_error(bad, 1.0)[0] or 0.0)


def test_missing_tolerance_is_withheld():
    rec = extract_quadruple("no threshold here", 1.0, 1.0)
    assert judge(rec).status == WITHHELD
    assert baseline_verdict(rec).status == WITHHELD


def test_zero_over_zero_baseline_follows_nan_semantics():
    rec = extract_quadruple("相对误差 < 1e-6", 0.0, 0.0, scale=1e-12)
    base = baseline_verdict(rec)
    assert base.value == 0.0 and base.status == PASS
    assert judge(rec).status == PASS  # exact zero against zero is legitimate

# --------------------------------------------------------------------------------------
# CLI surface (the package must be callable, not only importable)
# --------------------------------------------------------------------------------------
def test_cli_commands_in_process(tmp_path):
    from tensorphysics.metrology.criteria.__main__ import main

    out_controls = tmp_path / "controls.json"
    assert main(["controls", "--out", str(out_controls)]) == 0
    assert json.loads(out_controls.read_text())["n_degenerate_hit"] == 12

    out_labeled = tmp_path / "labeled.json"
    assert main(["labeled", "--out", str(out_labeled)]) == 0
    assert json.loads(out_labeled.read_text())["n"] == 40

    assert main(["replay", "--root", str(ROOT), "--out", str(tmp_path / "replay.json"), "--quiet"]) == 0
    assert json.loads((tmp_path / "replay.json").read_text())["kpi"]["n_records"] >= CORPUS_MIN_RECORDS


def test_cli_all_command(tmp_path):
    from tensorphysics.metrology.criteria.__main__ import main

    assert main(["all", "--root", str(ROOT), "--out-dir", str(tmp_path), "--quiet"]) == 0
    for name in ("criteria_controls.json", "criteria_labeled40.json", "criteria_replay.json"):
        assert (tmp_path / name).exists(), name
