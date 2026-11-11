"""P360W1-N04 driver: moving ice load -> hull-girder response -> FNV forecast.

Runs the acceptance study, writes the machine-readable payload, the figures and
the paper-grade report into the worktree:

    artifacts/p360w1_n04_moving_load_fnv/node_payload.json
    docs/reports/p360w1_n04_moving_load_fnv/figures/*.png
    docs/reports/p360w1_n04_moving_load_fnv/report.md

Usage:  python scripts/run_p360w1_n04_moving_load_fnv.py [--skip-figures]
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorphysics.moving_load import reference_hull_girder, run_acceptance_study  # noqa: E402

from p360_report_text import build_report  # noqa: E402
from p360_study_series import build_series, make_figures  # noqa: E402

ART = ROOT / "artifacts" / "p360w1_n04_moving_load_fnv"
REPORT_DIR = ROOT / "docs" / "reports" / "p360w1_n04_moving_load_fnv"
FIG_DIR = REPORT_DIR / "figures"


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def run_pytest() -> dict:
    """Run the module's regression suite and parse the pytest summary line."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_moving_load_fnv.py", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else ""
    passed = failed = 0
    m_pass = re.search(r"(\d+)\s+passed", summary)
    m_fail = re.search(r"(\d+)\s+(failed|error|errors)", summary)
    if m_pass:
        passed = int(m_pass.group(1))
    if m_fail:
        failed = int(m_fail.group(1))
    return {
        "passed": passed,
        "failed": failed,
        "returncode": proc.returncode,
        "summary": summary,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main(argv: list[str]) -> int:
    t0 = time.time()
    if "--report-only" in argv:
        from types import SimpleNamespace

        payload = json.loads((ART / "node_payload.json").read_text())
        beam_ns = SimpleNamespace(
            **{k: payload["beam"][k] for k in ("length_m", "ei_nm2", "rho_a_kg_m")}
        )
        rev = git_rev()
        payload["provenance"]["commit"] = rev
        (ART / "node_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (REPORT_DIR / "report.md").write_text(build_report(beam_ns, payload, payload["figures"], rev))
        print(f"report rebuilt from payload -> {REPORT_DIR / 'report.md'}")
        return 0

    beam = reference_hull_girder()
    print(f"[1/4] acceptance study on L={beam.length_m} m hull girder ...")
    payload = run_acceptance_study(beam)
    t_study = time.time() - t0

    print("[1.5/4] pytest regression suite ...")
    pytest_out = run_pytest()
    for item in payload["criteria"]:
        if item["key"] == "g":
            item["value"] = float(pytest_out["passed"])
            item["threshold"] = float(pytest_out["passed"] + pytest_out["failed"])
            item["passed"] = bool(pytest_out["failed"] == 0 and pytest_out["passed"] > 0)
            item["method"] = "pytest tests/test_moving_load_fnv.py (subprocess)"
    payload["all_passed"] = all(c["passed"] for c in payload["criteria"])

    print("[2/4] figure series ...")
    series = build_series(beam)
    t_series = time.time() - t0 - t_study

    figs = []
    if "--skip-figures" not in argv:
        print("[3/4] figures ...")
        figs = make_figures(beam, series, payload, FIG_DIR)
    else:
        figs = [
            {"name": p.name, "caption": p.stem} for p in sorted(FIG_DIR.glob("*.png"))
        ]

    print("[4/4] report + payload ...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    report_md = build_report(beam, payload, figs, git_rev())
    (REPORT_DIR / "report.md").write_text(report_md)

    payload_out = _jsonable(payload)
    payload_out["provenance"] = {
        "mission_id": 12502,
        "node": "p360w1-n04-moving-load-fnv",
        "repository": "TensorPhysics",
        "branch": "agent/m12502",
        "commit": git_rev(),
        "generated_by": "scripts/run_p360w1_n04_moving_load_fnv.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "random_seed": 20261111,
        "figures": [f["name"] for f in figs],
        "runtime_s": {"acceptance_study": round(t_study, 2), "figure_series": round(t_series, 2)},
        "artifacts": {
            "payload": "artifacts/p360w1_n04_moving_load_fnv/node_payload.json",
            "report": "docs/reports/p360w1_n04_moving_load_fnv/report.md",
            "figures_dir": "docs/reports/p360w1_n04_moving_load_fnv/figures",
            "tests": "tests/test_moving_load_fnv.py",
        },
    }
    payload_out["figures"] = figs
    payload_out["pytest"] = pytest_out
    with open(ART / "node_payload.json", "w") as fh:
        json.dump(payload_out, fh, indent=2, ensure_ascii=False)

    passed = sum(1 for c in payload["criteria"] if c["passed"])
    print(
        f"done in {time.time() - t0:.1f}s: criteria {passed}/{len(payload['criteria'])} pass, "
        f"all_passed={payload['all_passed']}"
    )
    print(f"  report : {REPORT_DIR / 'report.md'}")
    print(f"  payload: {ART / 'node_payload.json'}")
    print(f"  figures: {len(figs)}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))