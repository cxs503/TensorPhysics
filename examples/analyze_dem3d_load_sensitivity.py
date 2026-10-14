"""Derive reproducible response metrics from a completed DEM3D load campaign.

Metrics are descriptive numerical outputs, not claims of physical validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

PROTOCOL = "dem3d-bending-load-sensitivity-v1"
SOLVER_PROTOCOL = "tensordem-dem3d-three-point-bending-v1"
SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
CSV_FIELDS = [
    "run_index", "requested_load_N", "final_applied_load_N",
    "final_support_reaction_abs_N", "load_transfer_ratio",
    "relative_force_balance_residual", "final_midspan_deflection_abs_m",
    "secant_compliance_m_per_N", "secant_stiffness_N_per_m",
    "peak_support_reaction_abs_N", "peak_midspan_deflection_abs_m",
    "broken_bonds", "solver_signature_sha256",
    "relative_change_secant_compliance", "relative_change_load_transfer_ratio",
]


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _load_solver_report(path: Path, requested_load: float) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != SOLVER_PROTOCOL:
        raise ValueError(f"{path}: unexpected solver protocol")
    if report.get("verdict") != "PASS":
        raise ValueError(f"{path}: solver verdict is not PASS")
    reported_load = _finite_number(report.get("peak_applied_load_N"), "peak_applied_load_N")
    if not math.isclose(reported_load, requested_load, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{path}: requested load does not match solver report")
    signature = report.get("signature_sha256")
    if not isinstance(signature, str) or not SIGNATURE_RE.fullmatch(signature):
        raise ValueError(f"{path}: malformed solver signature")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{path}: at least two history rows are required")
    last_time = -math.inf
    last_broken = -1
    required = ("time_s", "applied_load_N", "support_reaction_z_N",
                "vertical_force_balance_residual_N", "midspan_deflection_m",
                "broken_bonds", "kinetic_energy_J", "bond_elastic_energy_J",
                "mechanical_energy_J")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: history row {index} is not an object")
        for key in required:
            if key not in row:
                raise ValueError(f"{path}: history row {index} missing {key}")
            if key == "broken_bonds":
                count = row[key]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"{path}: invalid broken-bond count at row {index}")
            else:
                _finite_number(row[key], f"row[{index}].{key}")
        t = float(row["time_s"])
        if t < last_time:
            raise ValueError(f"{path}: time history is not monotonic")
        if row["broken_bonds"] < last_broken:
            raise ValueError(f"{path}: broken-bond history is not monotonic")
        last_time, last_broken = t, row["broken_bonds"]
    final = rows[-1]
    final_load = _finite_number(final["applied_load_N"], "final applied load")
    if not math.isclose(final_load, requested_load, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{path}: final history load differs from requested load")
    return report


def analyze_campaign(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    source_path = work_dir / "load_sensitivity_report.json"
    campaign = json.loads(source_path.read_text(encoding="utf-8"))
    if campaign.get("protocol") != PROTOCOL:
        raise ValueError("unexpected campaign protocol")
    if campaign.get("decision") != "all_runs_passed_numerical_gates":
        raise ValueError("cannot analyze a campaign that did not pass all evidence gates")
    runs = campaign.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("campaign must contain at least two runs")
    metrics: list[dict[str, Any]] = []
    for index, run in enumerate(runs, 1):
        load = _finite_number(run.get("load_N"), f"run {index} load_N")
        if load <= 0 or run.get("evidence_gate") != "passed" or run.get("verdict") != "PASS":
            raise ValueError(f"run {index} did not pass the evidence gate")
        report_path = work_dir / f"run-{index:02d}" / "three_point_bending_report.json"
        report = _load_solver_report(report_path, load)
        final = report["rows"][-1]
        reaction = abs(float(final["support_reaction_z_N"]))
        deflection = abs(float(final["midspan_deflection_m"]))
        residual = abs(float(final["vertical_force_balance_residual_N"]))
        compliance = deflection / load
        metrics.append({
            "run_index": index,
            "requested_load_N": load,
            "final_applied_load_N": float(final["applied_load_N"]),
            "final_support_reaction_abs_N": reaction,
            "load_transfer_ratio": reaction / load,
            "relative_force_balance_residual": residual / load,
            "final_midspan_deflection_abs_m": deflection,
            "secant_compliance_m_per_N": compliance,
            "secant_stiffness_N_per_m": load / deflection if deflection > 0 else None,
            "peak_support_reaction_abs_N": _finite_number(
                report.get("peak_support_reaction_abs_N"), "peak support reaction"),
            "peak_midspan_deflection_abs_m": _finite_number(
                report.get("peak_midspan_deflection_abs_m"), "peak midspan deflection"),
            "broken_bonds": int(final["broken_bonds"]),
            "solver_signature_sha256": report["signature_sha256"],
            "relative_change_secant_compliance": None,
            "relative_change_load_transfer_ratio": None,
        })
    for previous, current in zip(metrics, metrics[1:]):
        for metric, output in (
            ("secant_compliance_m_per_N", "relative_change_secant_compliance"),
            ("load_transfer_ratio", "relative_change_load_transfer_ratio"),
        ):
            denominator = abs(previous[metric])
            current[output] = (
                (current[metric] - previous[metric]) / denominator
                if denominator > 0 else None
            )
    canonical = [
        {
            "run_index": row["run_index"],
            "requested_load_N": row["requested_load_N"],
            "solver_signature_sha256": row["solver_signature_sha256"],
            "load_transfer_ratio": row["load_transfer_ratio"],
            "relative_force_balance_residual": row["relative_force_balance_residual"],
            "final_midspan_deflection_abs_m": row["final_midspan_deflection_abs_m"],
            "secant_compliance_m_per_N": row["secant_compliance_m_per_N"],
            "broken_bonds": row["broken_bonds"],
        }
        for row in metrics
    ]
    campaign_signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = {
        "protocol": "dem3d-bending-quantitative-response-v1",
        "source_campaign_protocol": PROTOCOL,
        "run_count": len(metrics),
        "campaign_signature_sha256": campaign_signature,
        "metrics": metrics,
        "scientific_status": "descriptive_numerical_response_not_physical_validation",
        "metric_definitions": {
            "load_transfer_ratio": "abs(final support reaction z) / requested peak load",
            "relative_force_balance_residual": "abs(final force-balance residual) / requested peak load",
            "secant_compliance_m_per_N": "abs(final midspan deflection) / requested peak load",
            "secant_stiffness_N_per_m": "requested peak load / abs(final midspan deflection); null if deflection is zero",
            "relative_changes": "fractional change from the immediately preceding load case; null if its baseline metric is zero",
        },
        "limitations": [
            "Metrics describe the current discrete DEM fixture and its chosen loading/support implementation.",
            "No monotonic response trend is imposed as an acceptance criterion.",
            "No experimental agreement, material calibration, physical validity, or numerical convergence is established.",
        ],
    }
    (work_dir / "quantitative_response_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (work_dir / "response_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/dem3d-load-sensitivity"))
    args = parser.parse_args()
    result = analyze_campaign(args.work_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"} |
                     {"run_count": result["run_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
