from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from research_campaign_dem3d_timestep_refinement import _read_report


def _report(*, duration=0.002, dt=0.001, steps=2):
    rows = []
    for i in range(steps + 1):
        rows.append({
            "time_s": i * dt, "applied_load_N": 0.02 * i / steps,
            "support_reaction_z_N": -0.01 * i / steps,
            "vertical_force_balance_residual_N": -0.01 * i / steps,
            "midspan_deflection_m": -1e-4 * i / steps,
            "kinetic_energy_J": 0.0, "bond_elastic_energy_J": 0.0,
            "mechanical_energy_J": 0.0, "broken_bonds": i,
        })
    return {
        "protocol": "tensordem-dem3d-three-point-bending-v1",
        "verdict": "PASS", "peak_applied_load_N": 0.02,
        "signature_sha256": "a" * 64, "dt_s": dt,
        "simulated_duration_s": duration, "steps": steps, "rows": rows,
        "peak_support_reaction_abs_N": 0.01,
        "peak_midspan_deflection_abs_m": 1e-4,
    }


def test_report_gate_accepts_consistent_fixed_duration_history(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    result = _read_report(path, 0.02)
    assert result["steps"] == 2
    assert result["dt_s"] == pytest.approx(0.001)
    assert result["simulated_duration_s"] == pytest.approx(0.002)


@pytest.mark.parametrize("mutation", ["duration", "history_length", "signature", "load"])
def test_report_gate_rejects_inconsistent_evidence(tmp_path: Path, mutation: str):
    report = _report()
    if mutation == "duration":
        report["simulated_duration_s"] = 0.003
    elif mutation == "history_length":
        report["rows"].pop()
    elif mutation == "signature":
        report["signature_sha256"] = "g" * 64
    else:
        report["peak_applied_load_N"] = 0.03
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        _read_report(path, 0.02)
