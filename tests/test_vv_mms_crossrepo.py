"""Artifact-level tests of the matured V&V gate (mission IN48-N03).

The unit tests of the gate itself live in ``test_vv_gci_core.py`` /
``test_vv_validation.py`` / ``test_vv_adapters_cli.py``.  This module tests the
*evidence*: the numbers in ``artifacts/vv_gate`` that the acceptance criteria are
read from, including the real cross-repository runs

* TensorFVM -- its own manufactured-solution study
  (``verification/native_order3d.manufactured_solution_study``) on three grids,
  executed on the 外协方 cluster and ingested by ``tools/vv_crossrepo_mms.py``;
* TensorLBM -- its shipped Poiseuille verification case, which the gate
  **fails**; the fitted profile shows why (the effective wall sits a fraction of
  a cell away from the analytic reference, a first-order reference mismatch).

Every test skips when the artifact is not present, so the suite also runs in a
checkout where the (regenerable) evidence directory has not been produced.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "vv_gate"
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.vv import GatePolicy, Observation, evaluate  # noqa: E402


def _load(rel: str):
    path = ART / rel
    if not path.exists():
        pytest.skip(f"{rel} not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# (a) manufactured positive control, (b) negative controls, (d) V&V 20 table
# ---------------------------------------------------------------------------


def test_positive_control_machine_numbers():
    positive = _load("controls/positive_control.json")
    assert abs(positive["order_p"] - 2.0) <= 0.01
    assert positive["f_extrapolated_abs_error"] < 1e-6
    assert positive["gci_decreases_with_r"] is True
    assert positive["status"] == "PASS"
    # GCI_fine = Fs * |df| / |f_fine| / (r^p - 1) recomputed independently
    f = [lvl["f"] for lvl in positive["levels"]]
    expected = 1.25 * abs(f[-1] - f[-2]) / abs(f[-1]) / (2.0 ** 2 - 1.0)
    assert math.isclose(positive["gci_fine"], expected, rel_tol=1e-12)


def test_negative_controls_never_pass():
    negative = _load("controls/negative_controls.json")
    for case in ("oscillatory", "diverging", "stagnant"):
        assert negative[case]["status"] == "FAIL", case
    cases = negative["one_percent_noise"]["cases"]
    assert cases["three_grid_declared_sigma"]["silent_pass"] == 0
    assert cases["five_grid_undeclared_sigma"]["silent_pass"] == 0
    # documented partial result: with an undeclared 1 % noise floor the three-grid
    # fit has no residual, so only the declared design order can catch it - and it
    # catches roughly half of the seeds, never all of them.
    partial = cases["three_grid_undeclared_sigma_design_order"]
    assert partial["declared_sigma"] is False
    assert partial["silent_pass"] < negative["one_percent_noise"]["seeds"]
    assert negative["three_grid_undeclared_noise"]["strict_status"] == "WARN"


def test_vv20_table_agrees_with_hand_computation():
    table = _load("controls/vv20_table.json")
    assert table["n_rows"] == 10
    assert table["n_agree"] == 10
    for row in table["rows"]:
        assert row["status_gate"] == row["status_hand"], row["label"]
        assert row["u_val_gate"] == pytest.approx(row["u_val_hand"], rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# (c) real cross-repository manufactured-solution run: TensorFVM
# ---------------------------------------------------------------------------


def test_tensorfvm_mms_raw_three_grids():
    raw = _load("tensorfvm_mms_raw.json")
    assert raw["schema"] == "tensorphysics.vv.crossrepo/1"
    assert raw["repo"] == "TensorFVM"
    levels = raw["levels"]
    assert len(levels) >= 3
    hs = [row["h"] for row in levels]
    assert hs == sorted(hs, reverse=True)
    for coarse, fine in zip(hs, hs[1:]):
        assert coarse / fine == pytest.approx(2.0)  # uniform r = 2
    for row in levels:
        assert row["residual"] < 1e-10  # the linear solve is converged
    assert raw["provenance"]["module_sha256"]["tensorfvm/verification/native_order3d.py"]


def test_tensorfvm_mms_order_recovers_design_order():
    gate = _load("gate_tensorfvm_mms_l2_interior.json")
    study = gate["study"]
    assert abs(study["order_p"] - 2.0) < 0.2  # acceptance criterion (c)
    assert study["convergence_type"] == "monotone"
    assert gate["status"] == "PASS"


def test_tensorfvm_mms_wall_contaminated_norms_stay_below_order_two():
    """Honest secondary result: the wall-adjacent layer is not second order."""
    interior = _load("gate_tensorfvm_mms_l2_interior.json")["study"]
    global_l2 = _load("gate_tensorfvm_mms_l2_global.json")["study"]
    max_norm = _load("gate_tensorfvm_mms_linf_global.json")["study"]
    assert global_l2["order_p"] < interior["order_p"]
    assert max_norm["order_p"] < global_l2["order_p"]
    assert max_norm["order_p"] < 1.6


def test_cli_parses_the_emitted_mms_observation():
    """Criterion (e): the same CLI parses the adapter output, no recomputation."""
    obs = ART / "obs_tensorfvm_mms_l2_interior.json"
    gate = ART / "gate_tensorfvm_mms_l2_interior.json"
    if not obs.exists() or not gate.exists():
        pytest.skip("MMS observations not generated in this checkout")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-m", "tensorphysics.vv", "--obs", str(obs), "--json"],
                          capture_output=True, text=True, env=env, cwd=ROOT, check=False)
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)
    reference = json.loads(gate.read_text(encoding="utf-8"))
    assert record["status"] == reference["status"]
    assert record["study"]["order_p"] == pytest.approx(reference["study"]["order_p"], rel=1e-12)
    assert record["study"]["gci_fine"] == pytest.approx(reference["study"]["gci_fine"], rel=1e-12)


def test_adapter_dialects_are_two_repositories():
    """Criterion (e): >= 2 real repositories parsed by the same gate/CLI."""
    repos = set()
    for name in ART.glob("gate_tensor*_*.json"):
        record = json.loads(name.read_text(encoding="utf-8"))
        adapter = record.get("adapter") or {}
        repo = adapter.get("repo") or ("TensorFVM" if "tensorfvm" in name.name else None)
        if repo:
            repos.add(repo)
    if not repos:
        pytest.skip("no cross-repository gate records in this checkout")
    assert len(repos) >= 2, repos


# ---------------------------------------------------------------------------
# The honest FAIL: TensorLBM's shipped Poiseuille verification case
# ---------------------------------------------------------------------------


def test_tensorlbm_poiseuille_is_failed_by_the_gate():
    initial = {"rel_l2_error": 2.394666e-02, "flow_rate_error": 2.902468e-02,
               "peak_velocity_error": 1.833719e-02}
    raw = _load("tensorlbm_poiseuille_raw.json")
    assert [row["ny"] for row in raw["levels"]] == [16, 32, 64]
    for observable in initial:
        gate = _load(f"gate_tensorlbm_poiseuille_{observable}.json")
        assert gate["status"] == "FAIL", observable
        assert gate["study"]["order_p"] < 1.2
        assert raw["levels"][0][observable] == pytest.approx(initial[observable], rel=1e-6)


def test_tensorlbm_poiseuille_failure_is_a_reference_mismatch():
    """Fitted profile: parabolic to 1e-5, but the wall sits ~0.57 cell off."""
    fit = _load("lbm_poiseuille_profile_fit.json")
    ny_keys = sorted((int(k) for k in fit["levels"]), key=int)
    assert len(ny_keys) >= 2
    for ny in ny_keys:
        row = fit["levels"][str(ny)]
        # the discrete profile is parabolic to ~1e-5 of u_max: the interior
        # stencil is not what limits the order
        assert row["parabola_residual_rel"] < 1e-3
        # ... but the effective wall does not sit where the boundary scheme puts
        # it: halfway bounce-back would give -0.5 cells, the fit gives ~+0.07
        assert 0.4 < row["offset_vs_halfway_cells"] < 1.0
        assert abs(row["offset_vs_node_cells"]) < 0.2
        assert 0.95 < row["amp_ratio"] <= 1.0
    # a wall displacement that is constant in cell units is an O(h) error, so the
    # amplitude error decays as h^0.8 and the measured order cannot reach 2
    assert 0.6 < fit["amplitude_order_h"] < 1.0


def test_zero_intrusion_digests_still_match():
    """No solver file was touched: the recorded digests equal the live files."""
    raw = _load("tensorfvm_mms_raw.json")
    src = Path(raw["provenance"]["source_root"])
    if not src.is_dir():
        pytest.skip(f"TensorFVM source snapshot not present at {src}")
    for rel, digest in raw["provenance"]["module_sha256"].items():
        if digest is None:
            continue
        path = src / rel
        if not path.is_file():
            pytest.skip(f"{rel} not present")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

# ---------------------------------------------------------------------------
# the document must quote the artifacts, not a hand-typed number
# ---------------------------------------------------------------------------


def test_doc_quotes_the_artifact_numbers():
    doc = ROOT / "docs" / "VV_GATE_MATURITY_IN48_N03.md"
    if not doc.exists():
        pytest.skip("maturity document not present")
    text = doc.read_text(encoding="utf-8")

    positive = _load("controls/positive_control.json")
    assert f"{positive['order_p']:.6f}" in text
    assert repr(positive["gci_fine"]) in text or f"{positive['gci_fine']:.18f}" in text

    table = _load("controls/vv20_table.json")
    assert f"{table['n_agree']}/{table['n_rows']}" in text

    mms = _load("gate_tensorfvm_mms_l2_interior.json")["study"]
    assert f"{mms['order_p']:.4f}" in text
    for level in mms["levels"]:
        assert f"{level['f']:.6e}" in text

    four = _load("gate_tensorfvm_mms_4level_l2_interior.json")["study"]
    assert f"{four['order_p']:.4f}" in text

    for observable in ("rel_l2_error", "flow_rate_error", "peak_velocity_error"):
        gate = _load(f"gate_tensorlbm_poiseuille_{observable}.json")["study"]
        assert f"{gate['order_p']:.4f}" in text

    fit = _load("lbm_poiseuille_profile_fit.json")
    assert f"{fit['amplitude_order_h']:.3f}" in text
