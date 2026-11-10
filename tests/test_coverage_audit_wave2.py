#!/usr/bin/env python3
"""Acceptance tests for node p270w2-n03 — statistical coverage audit (SBC), wave 2.

Every estimator / truth generator used here is synthetic (location–scale
problems only).  No subject-matter vocabulary enters the audit engine; AC8
enforces that lexically.

Acceptance criteria (node metadata):
  AC1 well-calibrated reference pipeline: coverage in [0.93, 0.97] at level 0
  AC2 model-error level: coverage < 0.93, verdict DISTORTED, detection >= 80 %
  AC3 sensitivity control: CI half-width x0.5 -> coverage < 0.85
  AC4 invariance control: CI width x3 -> coverage > 0.99 and CONSERVATIVE
  AC5 stopping rule: resolution table monotone, n_min reproducible
  AC6 case replay: uncalibrated CI (<0.90 coverage) is explicitly flagged
  AC7 artifact chain: numbers.json + sha256 + git commit, tamper-evident
  AC8 zero domain semantics in the audit engine
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.stats.audit_provenance import (MANIFEST_FILE, NUMBERS_FILE,
                                         SIDECAR_FILE, sha256_file,
                                         verify_provenance, write_provenance)
from tools.stats.coverage_audit import (CoverageAudit, coverage_resolution_table,
                                       detection_rate, judge_coverage_spec,
                                       scaled_estimator, solve_n_min,
                                       solve_n_min_from_resolution,
                                       wilson_half_width_continuous)

ART = ROOT / "artifacts" / "coverage_audit_wave2"
N_REP = 500


# ---------------------------------------------------------------------------
# synthetic pipelines (no domain semantics)
# ---------------------------------------------------------------------------

def gauss_truth(rng, n_samples, error_level):
    """Known-truth Gaussian data; level 1/2 inject model-structure error."""
    theta = 3.0
    if error_level == 0:
        data = rng.normal(theta, 1.0, size=n_samples)
    elif error_level == 1:
        k = max(1, int(round(0.15 * n_samples)))
        data = np.concatenate([rng.normal(theta, 1.0, size=n_samples - k),
                               rng.normal(theta + 2.4, 1.0, size=k)])
        rng.shuffle(data)
    elif error_level == 2:
        idx = np.arange(n_samples)
        data = rng.normal(theta, 1.0 + 0.02 * idx)
        data[:max(1, int(round(0.15 * n_samples)))] += 2.4
    else:
        raise ValueError("bad level")
    return data, theta


def t_interval(data):
    from scipy import stats as st
    x = np.asarray(data, float)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    q = float(st.t.ppf(0.975, df=n - 1))
    return m, m - q * se, m + q * se


def audit_level(estimator, level, n_rep=N_REP, n_samples=50, seed=42, nominal_width=None):
    res = CoverageAudit(gauss_truth, estimator, n_replications=n_rep,
                        n_samples=n_samples, seed=seed,
                        nominal_width=nominal_width).run(error_levels=(level,))
    return res.levels[0]


# ---------------------------------------------------------------------------
# AC1 / AC2 / AC3 / AC4
# ---------------------------------------------------------------------------

def test_ac1_reference_pipeline_is_well_calibrated():
    lv = audit_level(t_interval, 0)
    assert lv.n_replications == N_REP
    assert 0.93 <= lv.coverage <= 0.97, lv.coverage
    assert lv.spec_verdict == "WELL_CALIBRATED"


def test_ac2_model_error_is_detected():
    lv = audit_level(t_interval, 1)
    assert lv.coverage < 0.93, lv.coverage
    assert lv.spec_verdict == "DISTORTED"

    assert lv.misses_low + lv.misses_high == lv.n_replications - lv.n_covered
    assert max(lv.misses_low, lv.misses_high) > 0     # misses are attributed to a side

    det = detection_rate(gauss_truth, t_interval, error_level=1, n_trials=6,
                         n_replications=N_REP, n_samples=50, seed=42)
    assert det.detection_rate >= 0.80, det.detection_rate
    assert len(det.trial_coverages) == 6
    assert all(v == "DISTORTED" for v in det.trial_verdicts)


def test_ac3_sensitivity_control_narrowed_ci_undercovers():
    lv = audit_level(scaled_estimator(t_interval, 0.5), 0)
    assert lv.coverage < 0.85, lv.coverage


def test_ac4_invariance_control_widened_ci_overcoveres():
    base = audit_level(t_interval, 0)
    lv = audit_level(scaled_estimator(t_interval, 3.0), 0,
                     nominal_width=base.mean_ci_width)
    assert lv.coverage > 0.99, lv.coverage
    assert lv.spec_verdict == "CONSERVATIVE"
    assert lv.width_anomaly is True          # flagged separately, not mislabeled
    assert lv.ci_width_ratio > 2.5


def test_verdict_monotone_in_coverage():
    assert judge_coverage_spec(0.90) == "DISTORTED"
    assert judge_coverage_spec(0.95) == "WELL_CALIBRATED"
    assert judge_coverage_spec(0.98) == "CONSERVATIVE"
    assert judge_coverage_spec(1.00) == "CONSERVATIVE"


# ---------------------------------------------------------------------------
# AC5 stopping rule
# ---------------------------------------------------------------------------

def test_ac5_resolution_curve_is_monotone_and_n_min_reproducible():
    table = coverage_resolution_table(0.95)
    widths = [row["half_width"] for row in table]
    assert all(b < a for a, b in zip(widths, widths[1:])), widths

    n_min = solve_n_min_from_resolution(0.95, 0.02)
    assert wilson_half_width_continuous(0.95, n_min) <= 0.02
    assert wilson_half_width_continuous(0.95, n_min - 1) > 0.02
    # case-specific rule is never looser than the worst-case p=0.5 bound
    assert n_min <= solve_n_min(0.5, 0.02) == 2401
    # and it is a genuine reduction for a well-calibrated pipeline
    assert solve_n_min_from_resolution(0.5, 0.02) > n_min
    assert n_min == solve_n_min_from_resolution(0.95, 0.02)   # deterministic


def test_ac5_n500_meets_reported_resolution():
    hw = wilson_half_width_continuous(0.95, 500)
    assert 0.015 < hw < 0.025


# ---------------------------------------------------------------------------
# AC6 real-case replay (artifact-driven; skips if the runner has not been run)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (ART / NUMBERS_FILE).exists(),
                    reason="run tools/stats/n03_wave2_runner.py first")
def test_ac6_case_replay_flags_uncalibrated_ci():
    numbers = json.loads((ART / NUMBERS_FILE).read_text())
    cases = numbers["ac6_case_replay"]
    assert {"case_A", "case_B"} <= set(cases)

    case_a = cases["case_A"]
    assert case_a["declared"]["n_samples"] == 8
    assert case_a["declared"]["bootstrap"] == 1000
    for name, block in case_a["estimators"].items():
        cov = block["coverage_level0"]
        assert block["ci_calibrated"] is bool(cov >= 0.90)
        if cov < 0.90:
            assert "UNCALIBRATED" in block["label"], (name, block["label"])
        assert block["n_min_replications_for_hw_0p02"] > 0

    case_b = cases["case_B"]
    assert case_b["declared"]["n_samples"] == 60
    for block in case_b["estimators"].values():
        assert 0.0 <= block["coverage_level0"] <= 1.0

    # at least one declared 95 % interval in the replay must be reported broken
    assert any(not b["ci_calibrated"] for b in case_a["estimators"].values()) or \
        any(not b["ci_calibrated"] for b in case_b["estimators"].values())


# ---------------------------------------------------------------------------
# AC7 provenance chain
# ---------------------------------------------------------------------------

def test_ac7_provenance_chain_and_tamper_detection(tmp_path):
    src = ROOT / "tools" / "stats" / "coverage_audit.py"
    numbers = {"a": 1, "b": [0.95, 0.93], "nested": {"c": 3.0}}
    manifest = write_provenance(tmp_path, numbers, source_files=[src],
                               repo_root=ROOT, reproduce_cmd="pytest")

    assert (tmp_path / NUMBERS_FILE).exists()
    assert (tmp_path / SIDECAR_FILE).exists()
    assert (tmp_path / MANIFEST_FILE).exists()
    assert manifest["numbers_sha256"] == sha256_file(tmp_path / NUMBERS_FILE)
    assert manifest["source_sha256"][str(src)] == sha256_file(src)

    ok = verify_provenance(tmp_path)
    assert ok["ok"] is True, ok["problems"]
    assert ok["git_commit"] == manifest["git_commit"]
    assert ok["git_commit"]          # repo is under git

    # tamper: change one digit -> chain must break
    tampered = json.loads((tmp_path / NUMBERS_FILE).read_text())
    tampered["b"][0] = 0.96
    (tmp_path / NUMBERS_FILE).write_text(json.dumps(tampered, sort_keys=True, indent=2))
    bad = verify_provenance(tmp_path)
    assert bad["ok"] is False
    assert any("sha256 mismatch" in p for p in bad["problems"])


@pytest.mark.skipif(not (ART / NUMBERS_FILE).exists(),
                    reason="run tools/stats/n03_wave2_runner.py first")
def test_ac7_runner_artifacts_are_verified_on_disk():
    res = verify_provenance(ART)
    assert res["ok"] is True, res["problems"]
    numbers = json.loads((ART / NUMBERS_FILE).read_text())
    assert numbers["ac7_provenance"]["git_commit"]
    assert numbers["ac_rollup"]["AC1"] and numbers["ac_rollup"]["AC2"]
    findings = (ART / "figures")
    assert findings.is_dir() and list(findings.glob("*.png"))


# ---------------------------------------------------------------------------
# AC8 zero domain semantics in the audit engine
# ---------------------------------------------------------------------------

FORBIDDEN = [
    # vessel / offshore vocabulary
    "ship", "ships", "hull", "vessel", "propeller", "cavitation", "mooring",
    "offshore", "seakeeping", "draught", "draft", "keel", "bow", "stern",
    # ice / environment
    "ice", "icebreaker", "iceberg", "wave", "waves", "sea", "ocean",
    # materials / structural
    "fatigue", "stress", "strain", "bearing", "weld",
    # distribution names that only make sense as injected application content
    "weibull", "gumbel", "lindqvist", "iacs",
]


def _lexical_hits(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").lower()
    hits = []
    for word in FORBIDDEN:
        if re.search(rf"\b{re.escape(word)}\b", text):
            hits.append(word)
    return hits


def test_ac8_engine_has_no_domain_vocabulary():
    for name in ("coverage_audit.py", "audit_provenance.py"):
        hits = _lexical_hits(ROOT / "tools" / "stats" / name)
        assert hits == [], f"{name} leaks domain vocabulary: {hits}"


def test_ac8_engine_does_not_import_case_modules():
    text = (ROOT / "tools" / "stats" / "coverage_audit.py").read_text()
    for forbidden_module in ("case_studies", "n03_wave2_runner"):
        assert forbidden_module not in text


def test_ac8_engine_works_with_an_opaque_pipeline():
    """The engine must accept any callable pair — here a deliberately odd one."""
    def truth(rng, n, level):
        return {"values": rng.exponential(2.0, size=n)}, 2.0

    def est(data):
        v = data["values"]
        m = float(v.mean())
        return m, m - 2.0 / np.sqrt(len(v)), m + 2.0 / np.sqrt(len(v))

    res = CoverageAudit(truth, est, n_replications=50, n_samples=30).run(error_levels=(0,))
    assert 0.0 <= res.levels[0].coverage <= 1.0
    assert res.levels[0].spec_verdict in {"DISTORTED", "WELL_CALIBRATED", "CONSERVATIVE"}