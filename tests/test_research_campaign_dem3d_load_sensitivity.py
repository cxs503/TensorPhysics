from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
from research_campaign_dem3d_load_sensitivity import _validate_solver_report, run_campaign


def test_rejects_invalid_campaign_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02,), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02, 0.02), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02, float("nan")), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.01, 0.02), 0)



def _valid_report(load: float = 0.02) -> dict:
    return {
        "protocol": "tensordem-dem3d-three-point-bending-v1",
        "verdict": "PASS",
        "peak_applied_load_N": load,
        "peak_support_reaction_abs_N": load,
        "peak_midspan_deflection_abs_m": 0.001,
        "broken_bonds": 0,
        "failed_checks": [],
        "signature_sha256": "a" * 64,
        "rows": [
            {"time_s": 0.0, "applied_load_N": 0.0, "support_reaction_z_N": 0.0,
             "vertical_force_balance_residual_N": 0.0, "midspan_deflection_m": 0.0,
             "broken_bonds": 0, "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.0,
             "mechanical_energy_J": 0.0},
            {"time_s": 0.1, "applied_load_N": load, "support_reaction_z_N": load,
             "vertical_force_balance_residual_N": 0.0, "midspan_deflection_m": -0.001,
             "broken_bonds": 0, "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.1,
             "mechanical_energy_J": 0.1},
        ],
    }


def test_valid_solver_report_is_independently_accepted(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_valid_report()), encoding="utf-8")
    report = _validate_solver_report(path, 0.02)
    assert report["verdict"] == "PASS"
    assert report["peak_applied_load_N"] == 0.02


@pytest.mark.parametrize("mutation", ["wrong_load", "bad_protocol", "missing_signature", "nonfinite_history", "time_reversal"])
def test_solver_report_identity_and_history_are_fail_closed(tmp_path: Path, mutation: str) -> None:
    report = _valid_report()
    if mutation == "wrong_load":
        report["peak_applied_load_N"] = 0.01
    elif mutation == "bad_protocol":
        report["protocol"] = "unknown-protocol"
    elif mutation == "missing_signature":
        report["signature_sha256"] = ""
    elif mutation == "nonfinite_history":
        report["rows"][1]["mechanical_energy_J"] = float("nan")
    else:
        report["rows"][1]["time_s"] = -1.0
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate_solver_report(path, 0.02)


def test_loads_must_be_ascending(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        run_campaign(tmp_path, tmp_path, (0.03, 0.02), 24)
