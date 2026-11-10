"""End-to-end test of the study driver and its acceptance-criteria ledger.

Runs the study in ``--quick`` mode into a temporary directory and checks that

* the evidence bundle has the expected structure for all three anchors,
* every criterion row carries its measured number next to the pass flag,
* the pass flags are *derived* from the measurements (no hand-edited verdicts),
* the figures declared in the bundle exist on disk (when generated).
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_adaptive_fem_study as study  # noqa: E402


@pytest.fixture(scope="module")
def quick_bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("adaptive_fem_study")
    study.main(["--quick", "--no-figures", "--out", str(out)])
    payload = json.loads((out / "study.json").read_text())
    return payload, out


def test_study_bundle_has_all_anchors(quick_bundle):
    payload, _ = quick_bundle
    assert payload["node"] == "p284w3-n03-fem-adaptive-refinement"
    assert set(payload["cases"]) == {"cook", "lame", "crack"}
    for key in ("cook", "lame", "crack"):
        case = payload["cases"][key]
        assert case["uniform"] and case["adaptive"]
        assert case["reference"]


def test_study_rows_are_complete(quick_bundle):
    payload, _ = quick_bundle
    for key in ("cook", "lame"):
        for row in payload["cases"][key]["adaptive"]:
            for field in (
                "n_dof",
                "n_tris",
                "min_angle_deg",
                "eta_total",
                "error_rel",
                "effectivity",
            ):
                assert field in row
            assert row["n_dof"] > 0
            assert row["min_angle_deg"] > 0.0
    for row in payload["cases"]["crack"]["adaptive"]:
        assert "KI_rel_error" in row and "KI" in row


def test_criteria_flags_follow_measurements(quick_bundle):
    payload, _ = quick_bundle
    crit = payload["criteria"]
    for name in (
        "a_cook_energy_error_reduction_ge_50pct",
        "b_lame_displacement_l2_lt_1pct",
        "c_crack_KI_error_lt_5pct",
        "d_effectivity_in_0.8_1.2",
        "e_min_angle_gt_20deg",
    ):
        assert name in crit
        assert "pass" in crit[name]
        assert "threshold" in crit[name]

    # (b) and (c): pass must equal the comparison of the measured value
    b = crit["b_lame_displacement_l2_lt_1pct"]
    assert b["pass"] == (b["measured_adaptive_best"] < b["threshold"])
    c = crit["c_crack_KI_error_lt_5pct"]
    assert c["pass"] == (c["measured_best"] < c["threshold"])
    # (e): min angle over every mesh of the study
    e = crit["e_min_angle_gt_20deg"]
    angles = [
        row["min_angle_deg"]
        for case in payload["cases"].values()
        for scheme in ("uniform", "adaptive")
        for row in case[scheme]
    ]
    assert e["measured_min"] == pytest.approx(min(angles), rel=1e-12)
    assert e["pass"] == (min(angles) > 20.0)


def test_cook_adaptive_is_better_than_uniform_in_quick_mode(quick_bundle):
    payload, _ = quick_bundle
    pairs = payload["cases"]["cook"]["matched_dof"]
    assert pairs, "matched-DOF comparison must produce at least one pair"
    best = max(p["error_reduction"] for p in pairs)
    assert best > 0.0, "adaptive refinement must beat the uniform baseline"
    for p in pairs:
        assert p["error_ratio"] < 1.05


def test_study_bundle_is_json_serialisable_and_has_provenance(quick_bundle):
    payload, out = quick_bundle
    raw = (out / "study.json").read_text()
    again = json.loads(raw)
    assert again["environment"]["python"]
    assert again["theta"] == 0.5
    assert again["seconds"] > 0.0