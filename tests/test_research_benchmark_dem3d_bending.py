"""Unit tests for independent DEM3D bending report validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
from research_benchmark_dem3d_bending import _check_report  # noqa: E402


def _report() -> dict:
    return {
        "protocol": "tensordem-dem3d-three-point-bending-v1",
        "verdict": "PASS",
        "peak_support_reaction_abs_N": 0.02,
        "peak_midspan_deflection_abs_m": 0.001,
        "rows": [
            {"time_s": 0.0, "applied_load_N": 0.0, "support_reaction_z_N": 0.0,
             "vertical_force_balance_residual_N": 0.0, "midspan_deflection_m": 0.0,
             "broken_bonds": 0, "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.0,
             "mechanical_energy_J": 0.0},
            {"time_s": 0.1, "applied_load_N": 0.02, "support_reaction_z_N": 0.02,
             "vertical_force_balance_residual_N": 0.0, "midspan_deflection_m": -0.001,
             "broken_bonds": 1, "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.1,
             "mechanical_energy_J": 0.1},
        ],
    }


def test_report_with_finite_ordered_rows_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    report, rows = _check_report(path)
    assert report["verdict"] == "PASS"
    assert len(rows) == 2


@pytest.mark.parametrize("mutation", ["nan_energy", "time_reversal", "fracture_reversal", "empty_rows"])
def test_invalid_history_is_rejected(tmp_path: Path, mutation: str) -> None:
    report = _report()
    if mutation == "nan_energy":
        report["rows"][1]["mechanical_energy_J"] = float("nan")
    elif mutation == "time_reversal":
        report["rows"][1]["time_s"] = -1
    elif mutation == "fracture_reversal":
        report["rows"][1]["broken_bonds"] = -1
    else:
        report["rows"] = []
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        _check_report(path)
