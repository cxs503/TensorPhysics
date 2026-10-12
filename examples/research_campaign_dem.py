"""Run a two-experiment bounded TensorDEM campaign with evidence-based gating.

This demonstrates a deterministic campaign controller. The parameter variation is
a software integration exercise, not a physically meaningful sensitivity study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_loop_dem import run_demo


def run_campaign(work_dir: Path, campaign_id: str, nx: int, ny: int, steps: int) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    # A fixed, explicit two-point design: change nx only; keep all other settings fixed.
    designs = ({"label": "baseline", "nx": nx}, {"label": "variant", "nx": nx + 1})
    runs: list[dict[str, Any]] = []
    for index, design in enumerate(designs, start=1):
        run_id = f"{campaign_id}-r{index:02d}"
        result = run_demo(
            work_dir / f"round-{index:02d}", run_id, design["nx"], ny, steps
        )
        runs.append({
            "round": index,
            "label": design["label"],
            "run_id": run_id,
            "parameters": {"nx": design["nx"], "ny": ny, "steps": steps, "device": "cpu"},
            "validation_status": result["validation_report"]["status"],
            "checks": result["validation_report"]["checks"],
            "manifest_artifact": result["manifest_artifact"],
            "next_action": result["research_memory"]["next_action"],
        })
        # Fail closed: do not spend the remaining budget after a failed gate.
        if result["validation_report"]["status"] != "passed":
            break

    all_passed = len(runs) == len(designs) and all(
        item["validation_status"] == "passed" for item in runs
    )
    campaign = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "design": "two-point bounded software integration sweep; nx varies by +1",
        "budget": {"max_runs": 2, "max_steps_per_run": steps},
        "runs": runs,
        "decision": (
            "bounded_campaign_complete_integrity_gates_passed"
            if all_passed else "stop_campaign_and_repair_failed_gate"
        ),
        "scientific_status": "not_a_physics_benchmark",
    }
    path = work_dir / "campaign_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/research-campaign-dem"))
    parser.add_argument("--campaign-id", default="research-campaign-dem-001")
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if min(args.nx, args.ny, args.steps) < 1:
        parser.error("--nx, --ny and --steps must be positive")
    result = run_campaign(args.work_dir, args.campaign_id, args.nx, args.ny, args.steps)
    print(json.dumps({
        "campaign_id": result["campaign_id"],
        "run_count": len(result["runs"]),
        "decision": result["decision"],
        "report": str(args.work_dir.resolve() / "campaign_report.json"),
    }, indent=2, sort_keys=True))
    if result["decision"] != "bounded_campaign_complete_integrity_gates_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
