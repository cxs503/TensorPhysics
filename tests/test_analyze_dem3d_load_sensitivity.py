from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analyze_dem3d_load_sensitivity import analyze_campaign


def _solver_report(load: float, *, signature: str = "a" * 64) -> dict:
    return {
        "protocol": "tensordem-dem3d-three-point-bending-v1",
        "verdict": "PASS",
        "peak_applied_load_N": load,
        "peak_support_reaction_abs_N": load * 0.9,
        "peak_midspan_deflection_abs_m": load * 0.002,
        "signature_sha256": signature,
        "rows": [
            {
                "time_s": 0.0, "applied_load_N": 0.0,
                "support_reaction_z_N": 0.0,
                "vertical_force_balance_residual_N": 0.0,
                "midspan_deflection_m": 0.0, "broken_bonds": 0,
                "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.0,
                "mechanical_energy_J": 0.0,
            },
            {
                "time_s": 0.1, "applied_load_N": load,
                "support_reaction_z_N": -0.9 * load,
                "vertical_force_balance_residual_N": -0.1 * load,
                "midspan_deflection_m": -0.002 * load, "broken_bonds": 2,
                "kinetic_energy_J": 0.01, "bond_elastic_energy_J": 0.02,
                "mechanical_energy_J": 0.03,
            },
        ],
    }


def _campaign(tmp_path: Path, loads=(0.01, 0.02, 0.03)) -> Path:
    root = tmp_path
    runs = []
    for i, load in enumerate(loads, 1):
        run_dir = root / f"run-{i:02d}"
        run_dir.mkdir(parents=True)
        (run_dir / "three_point_bending_report.json").write_text(
            json.dumps(_solver_report(load, signature=f"{i:064x}")),
            encoding="utf-8",
        )
        runs.append({
            "load_N": load, "returncode": 0, "verdict": "PASS",
            "evidence_gate": "passed", "peak_applied_load_N": load,
        })
    (root / "load_sensitivity_report.json").write_text(json.dumps({
        "protocol": "dem3d-bending-load-sensitivity-v1",
        "decision": "all_runs_passed_numerical_gates",
        "runs": runs,
    }), encoding="utf-8")
    return root


def test_writes_quantitative_metrics_and_stable_campaign_signature(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    first = analyze_campaign(root)
    second = analyze_campaign(root)
    assert first["run_count"] == 3
    assert first["campaign_signature_sha256"] == second["campaign_signature_sha256"]
    assert first["scientific_status"] == "descriptive_numerical_response_not_physical_validation"
    assert first["metrics"][0]["load_transfer_ratio"] == pytest.approx(0.9)
    assert first["metrics"][0]["relative_force_balance_residual"] == pytest.approx(0.1)
    assert first["metrics"][0]["secant_compliance_m_per_N"] == pytest.approx(0.002)
    assert first["metrics"][0]["secant_stiffness_N_per_m"] == pytest.approx(500.0)
    assert first["metrics"][0]["relative_change_secant_compliance"] is None
    assert first["metrics"][1]["relative_change_secant_compliance"] == pytest.approx(0.0)
    with (root / "response_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[-1]["solver_signature_sha256"] == f"{3:064x}"
    assert (root / "quantitative_response_report.json").is_file()


@pytest.mark.parametrize("mutation", ["wrong_load", "bad_signature", "decreasing_time", "failed_gate"])
def test_analyzer_fails_closed_on_invalid_evidence(tmp_path: Path, mutation: str) -> None:
    root = _campaign(tmp_path)
    if mutation == "failed_gate":
        path = root / "load_sensitivity_report.json"
        campaign = json.loads(path.read_text(encoding="utf-8"))
        campaign["runs"][1]["evidence_gate"] = "failed"
        path.write_text(json.dumps(campaign), encoding="utf-8")
    else:
        path = root / "run-02" / "three_point_bending_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "wrong_load":
            report["peak_applied_load_N"] = 0.5
        elif mutation == "bad_signature":
            report["signature_sha256"] = "z" * 64
        else:
            report["rows"][1]["time_s"] = -1
        path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        analyze_campaign(root)
