"""Bounded load-amplitude sensitivity campaign; not physical validation."""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path

def run_campaign(root: Path, solver_root: Path, loads: tuple[float, ...], steps: int) -> dict:
    if steps < 1 or len(loads) < 2 or len(set(loads)) != len(loads):
        raise ValueError("steps must be positive and loads must be at least two unique values")
    if any(not math.isfinite(x) or x <= 0 for x in loads):
        raise ValueError("loads must be finite and positive")
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
            r = json.loads(report_path.read_text())
            row.update({k: r.get(k) for k in ("verdict", "peak_support_reaction_abs_N",
                "peak_midspan_deflection_abs_m", "broken_bonds", "failed_checks")})
        else:
            row["stderr_tail"] = p.stderr[-1000:]
        runs.append(row)
        if p.returncode != 0 or row.get("verdict") != "PASS":
            break
    for old, new in zip(runs, runs[1:]):
        for metric in ("peak_support_reaction_abs_N", "peak_midspan_deflection_abs_m"):
            a, b = old.get(metric), new.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                new.setdefault("relative_change_from_previous", {})[metric] = (b-a)/max(abs(a), 1e-30)
    result = {"protocol": "dem3d-bending-load-sensitivity-v1", "steps": steps, "runs": runs,
        "decision": "all_runs_passed_numerical_gates" if len(runs)==len(loads) and all(x.get("verdict")=="PASS" for x in runs) else "stop_after_failed_run",
        "scientific_status": "exploratory_numerical_sensitivity_not_physical_validation",
        "interpretation": "Load-amplitude response only; not calibrated strength, experimental agreement, or numerical convergence."}
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
