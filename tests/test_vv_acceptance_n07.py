"""Regression tests for the mission IN57-N07 acceptance harness.

The harness (``tools/vv_acceptance_n07.py``) is the executable form of the six
machine criteria; these tests keep the criteria themselves honest (they are
cheap and do not need any solver checkout) and re-run the expensive
cross-repository parts only when their artifacts exist.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "vv_gate_n07"


def _load():
    path = ROOT / "tools" / "vv_acceptance_n07.py"
    spec = importlib.util.spec_from_file_location("vv_acceptance_n07", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accept = _load()


def test_f1_positive_control_recovers_second_order():
    f1 = accept.f1_positive_control()
    assert f1["passed"], f1["checks"]
    assert f1["order_error_vs_theory"] <= 1e-2
    assert f1["f_extrapolated_abs_error"] < 1e-6
    assert f1["checks"]["gci_decreases_with_r"]
    assert f1["five_grid"]["status"] == "PASS"
    # the conservative default caps the three-grid study at WARN
    assert f1["verdict_default_policy"] == "WARN"
    assert f1["verdict_with_declared_sigma"] == "PASS"


def test_f2_negative_controls_never_silently_pass():
    f2 = accept.f2_negative_controls(trials=8)
    assert f2["passed"], f2["checks"]
    for name, case in f2["noise"].items():
        assert case["silent_pass"] == 0, (name, case)
    assert f2["structural"]["oscillatory"]["status"] == "FAIL"
    assert f2["structural"]["diverging"]["status"] == "FAIL"
    assert f2["structural"]["stagnant"]["status"] == "FAIL"


def test_f2_noise_hardening_is_what_removes_the_silent_pass():
    """The three-grid undeclared-noise case is the one the hardening targets."""
    f2 = accept.f2_negative_controls(trials=8)
    case = f2["noise"]["three_grid_undeclared_sigma_1pct"]
    assert case["status_counts"]["PASS"] == 0
    # under the pre-hardening policy some seeds did pass silently
    assert case["legacy_policy_pass"] >= 1


def test_f4_vv20_table_matches_the_hand_computation():
    f4 = accept.f4_validation_table()
    assert f4["passed"]
    assert f4["n_agree"] == f4["n_rows"] == 10
    boundary = next(r for r in f4["rows"] if r["label"] == "boundary_exact")
    assert boundary["status_hand"] == "PASS" and boundary["status_gate"] == "PASS"


@pytest.mark.skipif(not (ART / "tensorlbm_dgmms" / "tensorlbm_dg_mms_native.json").exists()
                    or not (ART / "tensorfem_hconv" / "evidence.json").exists(),
                    reason="cross-repository artifacts not generated in this checkout")
def test_f3_real_crossrepo_runs():
    f3 = accept.f3_crossrepo()
    assert f3["passed"], f3["checks"]
    lbm = f3["runs"]["tensorlbm_dg_mms"]
    assert abs(lbm["order_p"] - 2.0) < 0.2
    assert lbm["status"] == "PASS"


@pytest.mark.skipif(not (ART / "tensorlbm_dgmms" / "tensorlbm_dg_mms_native.json").exists(),
                    reason="cross-repository artifacts not generated in this checkout")
def test_f5_adapters_conserve_fields():
    f5 = accept.f5_adapters()
    assert f5["passed"], f5["checks"]
    for repo, doc in f5["documents"].items():
        if "skipped" in doc:
            continue
        assert doc["field_conservation_max_abs_diff"] < 1e-12, repo


@pytest.mark.skipif(not (ROOT / "artifacts" / "vv_gate" /
                         "obs_tensorlbm_poiseuille_rel_l2_error.json").exists(),
                    reason="TensorLBM artifact not generated in this checkout")
def test_f6_existing_benchmark_readjudicated():
    f6 = accept.f6_zero_intrusion()
    assert f6["passed"], f6["checks"]
    assert f6["checks"]["solver_repositories_clean"]


def test_acceptance_record_is_json_serialisable(tmp_path):
    payload = {"criteria": {"F1": accept.f1_positive_control()}}
    text = json.dumps(payload)
    assert "order_p" in text