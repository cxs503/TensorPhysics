"""Bounded load-amplitude sensitivity campaign; not physical validation."""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
from typing import Any


def _validate_solver_report(report_path: Path, requested_load_N: float) -> dict[str, Any]:
    """Validate report identity and numeric evidence independently of exit code."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != "tensordem-dem3d-three-point-bending-v1":
        raise ValueError("unexpected or missing benchmark protocol")
    if report.get("verdict") != "PASS":
        raise ValueError(f"solver benchmark verdict is {report.get('verdict')!r}")
    actual_load = report.get("peak_applied_load_N")
    if not isinstance(actual_load, (int, float)) or not math.isfinite(actual_load):
        raise ValueError("peak_applied_load_N must be a finite number")
    if not math.isclose(float(actual_load), requested_load_N, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(
            f"requested load {requested_load_N} N differs from report load {actual_load!r} N"
        )
    for key in ("peak_support_reaction_abs_N", "peak_midspan_deflection_abs_m"):
        value = report.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(report.get("signature_sha256"), str) or len(report["signature_sha256"]) != 64:
        raise ValueError("missing or malformed deterministic history signature")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("report rows must contain at least two samples")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"report row {i} is not an object")
        for key in ("time_s", "applied_load_N", "support_reaction_z_N",
                    "vertical_force_balance_residual_N", "midspan_deflection_m",
                    "kinetic_energy_J", "bond_elastic_energy_J", "mechanical_energy_J"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"report row {i} has invalid {key}")
        if not isinstance(row.get("broken_bonds"), int) or row["broken_bonds"] < 0:
            raise ValueError(f"report row {i} has invalid broken_bonds")
    times = [float(row["time_s"]) for row in rows]
    broken = [int(row["broken_bonds"]) for row in rows]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("report time history is not monotonic")
    if any(b < a for a, b in zip(broken, broken[1:])):
        raise ValueError("broken-bond history is not monotonic")
    return report

def run_campaign(root: Path, solver_root: Path, loads: tuple[float, ...], steps: int) -> dict:
    if steps < 1 or len(loads) < 2 or len(set(loads)) != len(loads):
        raise ValueError("steps must be positive and loads must be at least two unique values")
    if any(not math.isfinite(x) or x <= 0 for x in loads):
        raise ValueError("loads must be finite and positive")
    if any(b <= a for a, b in zip(loads, loads[1:])):
        raise ValueError("loads must be strictly increasing so sensitivity trends are interpretable")
    root.mkdir(parents=True, exist_ok=True)
    script = solver_root / "scripts" / "benchmark_dem3d_three_point_bending.py"
    runs = []
    for i, load in enumerate(loads, 1):
        out = root / f"run-{i:02d}"
        p = subprocess.run([sys.executable, str(script), "--output", str(out),
                            "--steps", str(steps), "--peak-load-N", str(load)],
                           cwd=solver_root, text=True, capture_output=True)
        report_path = out / "three_point_bending_report.json"
        row = {"load_N": load, "returncode": p.returncode}
        if report_path.is_file():
            try:
                r = _validate_solver_report(report_path, load)
                row.update({k: r.get(k) for k in (
                    "protocol", "case_id", "verdict", "peak_applied_load_N",
                    "peak_support_reaction_abs_N", "peak_midspan_deflection_abs_m",
                    "broken_bonds", "failed_checks", "signature_sha256",
                )})
                row["evidence_gate"] = "passed"
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                row["evidence_gate"] = "failed"
                row["evidence_error"] = str(exc)
                row["stderr_tail"] = p.stderr[-1000:]
        else:
            row["evidence_gate"] = "failed"
            row["evidence_error"] = "benchmark report file was not produced"
            row["stderr_tail"] = p.stderr[-1000:]
        runs.append(row)
        if p.returncode != 0 or row.get("evidence_gate") != "passed" or row.get("verdict") != "PASS":
            break
    for old, new in zip(runs, runs[1:]):
        for metric in ("peak_support_reaction_abs_N", "peak_midspan_deflection_abs_m"):
            a, b = old.get(metric), new.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                new.setdefault("relative_change_from_previous", {})[metric] = (b-a)/max(abs(a), 1e-30)
    result = {"protocol": "dem3d-bending-load-sensitivity-v1", "steps": steps, "runs": runs,
        "decision": "all_runs_passed_numerical_gates" if len(runs)==len(loads) and all(x.get("verdict")=="PASS" for x in runs) else "stop_after_failed_run",
        "scientific_status": "exploratory_numerical_sensitivity_not_physical_validation",
        "interpretation": "Load-amplitude response only; each report is independently checked for protocol, requested load identity, finite ordered history, and signature presence. This is not calibrated strength, experimental agreement, or numerical convergence."}
    (root / "load_sensitivity_report.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=Path("outputs/dem3d-load-sensitivity"))
    ap.add_argument("--tensordem-root", type=Path, default=Path("../TensorDEM"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--loads-N", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    a=ap.parse_args()
    r=run_campaign(a.work_dir.resolve(), a.tensordem_root.resolve(), tuple(a.loads_N), a.steps)
    print(json.dumps({k:v for k,v in r.items() if k!="runs"} | {"run_count":len(r["runs"])}, indent=2))
    if r["decision"]!="all_runs_passed_numerical_gates": raise SystemExit(1)
if __name__=="__main__": main()
