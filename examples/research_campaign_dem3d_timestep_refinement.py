"""Fixed-duration DEM3D time-step refinement campaign; not physical validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

PROTOCOL = "dem3d-bending-timestep-refinement-v1"
SOLVER_PROTOCOL = "tensordem-dem3d-three-point-bending-v1"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def _read_report(path: Path, load_N: float) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != SOLVER_PROTOCOL or report.get("verdict") != "PASS":
        raise ValueError(f"{path}: wrong solver protocol or non-PASS verdict")
    if not math.isclose(_finite(report.get("peak_applied_load_N"), "peak load"),
                        load_N, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{path}: peak load mismatch")
    signature = report.get("signature_sha256")
    if not isinstance(signature, str) or len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
        raise ValueError(f"{path}: malformed history signature")
    dt = _finite(report.get("dt_s"), "dt_s")
    duration = _finite(report.get("simulated_duration_s"), "simulated_duration_s")
    steps = report.get("steps")
    if dt <= 0 or duration <= 0 or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"{path}: invalid time discretization metadata")
    if not math.isclose(duration, dt * steps, rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError(f"{path}: reported duration is inconsistent with dt and steps")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != steps + 1 or len(rows) < 2:
        raise ValueError(f"{path}: history length does not match steps")
    last_t = -math.inf
    last_broken = -1
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {i} is not an object")
        for key in ("time_s", "applied_load_N", "support_reaction_z_N",
                    "vertical_force_balance_residual_N", "midspan_deflection_m",
                    "kinetic_energy_J", "bond_elastic_energy_J", "mechanical_energy_J"):
            _finite(row.get(key), f"row[{i}].{key}")
        broken = row.get("broken_bonds")
        if isinstance(broken, bool) or not isinstance(broken, int) or broken < last_broken:
            raise ValueError(f"{path}: invalid/nonmonotonic fracture count at row {i}")
        t = float(row["time_s"])
        if t < last_t:
            raise ValueError(f"{path}: nonmonotonic time at row {i}")
        last_t, last_broken = t, broken
    if not math.isclose(float(rows[-1]["applied_load_N"]), load_N, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{path}: final history load mismatch")
    return report


def run_campaign(solver_root: Path, work_dir: Path, *, load_N: float = 0.02,
                 base_steps: int = 24, relative_tolerance: float = 0.05) -> dict[str, Any]:
    if not math.isfinite(load_N) or load_N <= 0:
        raise ValueError("load_N must be finite and positive")
    if base_steps < 2:
        raise ValueError("base_steps must be >= 2")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be finite and positive")
    solver_root = solver_root.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    script = solver_root / "scripts" / "benchmark_dem3d_three_point_bending.py"
    if not script.is_file():
        raise FileNotFoundError(f"TensorDEM solver script not found: {script}")

    cases: list[dict[str, Any]] = []
    reference_dt: float | None = None
    reference_duration: float | None = None
    for level in range(3):
        out = work_dir / f"run-{level + 1:02d}"
        command = [sys.executable, str(script), "--output", str(out),
                   "--peak-load-N", str(load_N)]
        if level == 0:
            command += ["--steps", str(base_steps)]
        else:
            assert reference_dt is not None and reference_duration is not None
            command += ["--dt-s", repr(reference_dt / (2 ** level)),
                        "--duration-s", repr(reference_duration)]
        process = subprocess.run(command, cwd=solver_root, capture_output=True, text=True)
        item: dict[str, Any] = {
            "level": level, "run_index": level + 1, "returncode": process.returncode,
            "requested_load_N": load_N, "stdout_tail": process.stdout[-1000:],
            "stderr_tail": process.stderr[-1000:],
        }
        report_path = out / "three_point_bending_report.json"
        try:
            if process.returncode != 0:
                raise ValueError(f"solver exited with return code {process.returncode}")
            report = _read_report(report_path, load_N)
            item.update({
                "evidence_gate": "passed", "dt_s": report["dt_s"],
                "steps": report["steps"], "simulated_duration_s": report["simulated_duration_s"],
                "particle_count": report.get("particle_count"),
                "broken_bonds": int(report["rows"][-1]["broken_bonds"]),
                "final_support_reaction_abs_N": abs(float(report["rows"][-1]["support_reaction_z_N"])),
                "final_midspan_deflection_abs_m": abs(float(report["rows"][-1]["midspan_deflection_m"])),
                "peak_support_reaction_abs_N": _finite(report.get("peak_support_reaction_abs_N"), "peak reaction"),
                "peak_midspan_deflection_abs_m": _finite(report.get("peak_midspan_deflection_abs_m"), "peak deflection"),
                "solver_signature_sha256": report["signature_sha256"],
            })
            if level == 0:
                reference_dt = float(report["dt_s"])
                reference_duration = float(report["simulated_duration_s"])
            else:
                if not math.isclose(float(report["simulated_duration_s"]), reference_duration,
                                    rel_tol=1e-9, abs_tol=1e-14):
                    raise ValueError("refinement run changed physical simulation duration")
                if not math.isclose(float(report["dt_s"]), reference_dt / (2 ** level),
                                    rel_tol=1e-9, abs_tol=1e-14):
                    raise ValueError("refinement run did not use requested dt ratio")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            item["evidence_gate"] = "failed"
            item["evidence_error"] = str(exc)
        cases.append(item)
        if item["evidence_gate"] != "passed":
            break

    if len(cases) != 3 or any(c["evidence_gate"] != "passed" for c in cases):
        result = {
            "protocol": PROTOCOL, "decision": "failed_numerical_evidence_gate",
            "convergence_status": "not_assessed", "requested_load_N": load_N,
            "relative_tolerance": relative_tolerance, "runs": cases,
            "scientific_status": "numerical_verification_only_not_physical_validation",
        }
        (work_dir / "timestep_refinement_report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        raise RuntimeError(f"time-step campaign evidence gate failed; see {work_dir / 'timestep_refinement_report.json'}")

    comparisons = []
    for coarse, fine in zip(cases, cases[1:]):
        comparison = {"from_dt_s": coarse["dt_s"], "to_dt_s": fine["dt_s"]}
        for key, out_key in (
            ("final_midspan_deflection_abs_m", "relative_deflection_change"),
            ("final_support_reaction_abs_N", "relative_reaction_change"),
            ("peak_midspan_deflection_abs_m", "relative_peak_deflection_change"),
            ("peak_support_reaction_abs_N", "relative_peak_reaction_change"),
        ):
            a, b = float(coarse[key]), float(fine[key])
            comparison[out_key] = (abs(b - a) / abs(a)) if a != 0 else (0.0 if b == 0 else None)
        comparison["broken_bonds_coarse"] = coarse["broken_bonds"]
        comparison["broken_bonds_fine"] = fine["broken_bonds"]
        names = ("relative_deflection_change", "relative_reaction_change",
                 "relative_peak_deflection_change", "relative_peak_reaction_change")
        comparison["within_declared_tolerance"] = all(
            comparison[name] is not None and comparison[name] <= relative_tolerance for name in names
        )
        comparisons.append(comparison)

    observable_rows = [{
        "level": c["level"], "dt_s": c["dt_s"], "steps": c["steps"],
        "simulated_duration_s": c["simulated_duration_s"],
        "final_support_reaction_abs_N": c["final_support_reaction_abs_N"],
        "final_midspan_deflection_abs_m": c["final_midspan_deflection_abs_m"],
        "peak_support_reaction_abs_N": c["peak_support_reaction_abs_N"],
        "peak_midspan_deflection_abs_m": c["peak_midspan_deflection_abs_m"],
        "broken_bonds": c["broken_bonds"], "solver_signature_sha256": c["solver_signature_sha256"],
    } for c in cases]
    signature_payload = json.dumps(observable_rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result = {
        "protocol": PROTOCOL, "decision": "all_runs_passed_numerical_gates",
        "convergence_status": "within_declared_tolerance" if all(x["within_declared_tolerance"] for x in comparisons)
                             else "not_within_declared_tolerance",
        "requested_load_N": load_N, "base_steps": base_steps,
        "fixed_simulated_duration_s": reference_duration,
        "relative_tolerance": relative_tolerance,
        "levels": observable_rows, "comparisons": comparisons,
        "campaign_signature_sha256": hashlib.sha256(signature_payload.encode()).hexdigest(),
        "scientific_status": "numerical_time_step_sensitivity_not_physical_validation",
        "limitations": [
            "The study varies time step at fixed physical duration and peak load; it does not vary particle resolution.",
            "A tolerance pass is a numerical comparison for these observables, not proof of mathematical convergence or physical validity.",
            "Fracture is discrete; broken-bond counts are reported as diagnostics and are not forced to agree.",
            "The benchmark's fixed-particle supports and distributed nodal loading are idealizations requiring independent physical justification.",
        ],
    }
    (work_dir / "timestep_refinement_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (work_dir / "timestep_refinement_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(observable_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(observable_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensordem-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/dem3d-timestep-refinement"))
    parser.add_argument("--peak-load-N", type=float, default=0.02)
    parser.add_argument("--base-steps", type=int, default=24)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    result = run_campaign(args.tensordem_root, args.work_dir, load_N=args.peak_load_N,
                          base_steps=args.base_steps, relative_tolerance=args.relative_tolerance)
    print(json.dumps({k: v for k, v in result.items() if k not in ("levels", "comparisons")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
