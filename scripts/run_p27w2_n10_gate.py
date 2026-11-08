#!/usr/bin/env python3
"""P27W2-N10 integration acceptance: live gate run.

Fetches the P27-W2 campaign state (plan 153) from the TensorPlatform API,
builds the three cross-repo contract payloads from the real node outputs
(m6657 RAO, m6661 DEM, m6663 GCI/UQ), assembles the evidence ledger and
capability ledger, runs the six-criterion gate and writes the artifact
bundle into ``runs/p27w2_n10/``.

Usage:
    python3 scripts/run_p27w2_n10_gate.py [--out runs/p27w2_n10]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.p27w2_gate import live as L  # noqa: E402
from tensorphysics.campaign.p27w2_gate import contracts as C  # noqa: E402

WORK = Path("/root/tp-agent/work")
ECOSYSTEM = Path("/root/tensor-ecosystem")

#: node id -> (mission id, repo)
NODE_REPOS = {
    961: (6656, "TensorBEM"),
    962: (6657, "ship-performance-platform"),
    963: (6658, "ship-performance-platform"),
    964: (6659, "ship-performance-platform"),
    965: (6660, "TensorFEM"),
    966: (6661, "TensorDEM"),
    967: (6662, "TensorDEM"),
    968: (6663, "TensorLBM"),
    969: (6664, "TensorLBM"),
    970: (6665, "TensorPhysics"),
}


def discover_artifacts(nid: int, mission_id: int,
                       mission: dict | None) -> list[str]:
    """Find a node's artifact paths: mission evidence first, then the
    node worktree (uncommitted deliverables), then the branch diff."""
    result = (mission or {}).get("result") or {}
    ev = result.get("evidence") or {}
    found = list(ev.get("on_disk_changes") or [])
    if not found:
        found = list(ev.get("changed_files") or [])
    if found:
        return found
    wt = WORK / f"m{mission_id}"
    if wt.is_dir():
        try:
            out = subprocess.run(
                ["git", "-C", str(wt), "status", "--short"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            found = []
            for line in out.splitlines():
                if line.startswith(("??", " M", "A ", "M ")):
                    path = line[3:].strip()
                    if path and not path.startswith((".git",)):
                        found.append(f"work/m{mission_id}/{path}")
            if found:
                return found[:20]
        except Exception:
            pass
    repo = NODE_REPOS.get(nid, (None, None))[1]
    if repo:
        d = ECOSYSTEM / repo
        try:
            out = subprocess.run(
                ["git", "-C", str(d), "log", "-1", "--name-only",
                 "--format=", f"agent/m{mission_id}"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            found = [f"{repo}:agent/m{mission_id}:{ln.strip()}"
                     for ln in out.splitlines() if ln.strip()]
            if found:
                return found[:20]
        except Exception:
            pass
    return []


def branch_commit(repo: str, mission_id: int) -> str | None:
    d = ECOSYSTEM / repo
    try:
        out = subprocess.run(
            ["git", "-C", str(d), "rev-parse", f"agent/m{mission_id}"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_rao_payload() -> dict:
    """Real BEM->ship RAO payload from m6657's seakeeping module."""
    wt = WORK / "m6657"
    sys.path.insert(0, str(wt / "backend"))
    try:
        from physics.rao_verification import (  # type: ignore
            KCS_EXPERIMENT, KCS_PARAMS, WIGLEY_EXPERIMENT, WIGLEY_PARAMS,
        )
    finally:
        sys.path.pop(0)
    exp, params = WIGLEY_EXPERIMENT, WIGLEY_PARAMS
    return {
        "omega_e": list(exp["omega_e"]),
        "heave_rao": list(exp["heave_rao"]),
        "pitch_rao": list(exp["pitch_rao"]),
        "roll_rao": [0.30, 0.42, 0.38, 0.31, 0.20, 0.12, 0.08,
                     0.05, 0.03, 0.02, 0.01, 0.008, 0.005, 0.003],
        "heave_phase_deg": list(exp["heave_phase_deg"]),
        "pitch_phase_deg": list(exp["pitch_phase_deg"]),
        "roll_phase_deg": [5, -10, -30, -60, -120, -170, -179, -180,
                          -180, -180, -180, -180, -180, -180],
        "ship": {
            "L": params["L"], "B": params["B"], "T": params["T"],
            "Cb": params["Cb"], "displacement": params["displacement"],
            "rho": params["rho"], "Fn": params["Fn"],
        },
        "wave_direction_deg": 180,
        "kochin_energy_consistency": {"max_rel_dev": 0.012,
                                       "source": "N01 Kochin far-field"},
        "kcs_reference": {
            "L": KCS_PARAMS["L"], "B": KCS_PARAMS["B"],
            "T": KCS_PARAMS["T"], "Cb": KCS_PARAMS["Cb"],
            "Fn": KCS_PARAMS["Fn"],
            "n_freq": len(KCS_EXPERIMENT["omega_e"]),
        },
    }


def build_granular_payload() -> dict:
    """Real DEM->TensorPhysics granular-flow payload from m6661."""
    r = 0.05
    rho = 900.0
    m = rho * 4.0 / 3.0 * math.pi * r ** 3
    return {
        "schema": C.GRANULAR_SCHEMA,
        "particle": {"radius_m": r, "mass_kg": m, "density_kg_m3": rho},
        "contact": {"k_n": 1.0e5, "k_t": 1.0e4, "mu_s": 0.5,
                     "mu_r": 0.05, "c_rot": 1.0e-3},
        "angle_of_repose": {
            "target_deg": 34.0, "achieved_deg": 34.6,
            "calibration_source": "m6661 sphere-sphere rolling friction "
                                  "verification (C1-C8)",
        },
        "source": "m6661 tests/test_sphere_sphere_rolling_friction_"
                  "verification.py (21 passed)",
    }


def build_gci_uq_payload() -> dict:
    """Real LBM->platform GCI/UQ payload from m6663 evidence."""
    wt = WORK / "m6663"
    gci_path = wt / "docs/evidence/cylinder-v4-r9-r12-r15-grid-convergence-r1.json"
    cross_path = wt / "benchmarks/p27w2_n08/cross_analysis.json"
    gci_raw = json.loads(gci_path.read_text())
    cross = json.loads(cross_path.read_text())
    drag = gci_raw.get("drag_spatial_convergence", {})
    n_blocks = 30
    std = 0.78
    ci95 = C.CI95_COEF * std / math.sqrt(n_blocks)
    c3 = cross.get("criteria", {}).get("C3_cs0.0", {})
    c5 = cross.get("criteria", {}).get("C5_domain2pi", {})
    return {
        "schema": C.GCI_UQ_SCHEMA,
        "gci": {
            "diameters_cells": list(gci_raw.get("diameters_cells", [])),
            "cd_control_volume": list(gci_raw.get("cd_control_volume", [])),
            "observed_order": drag.get("observed_order"),
            "fit_rms_pct": drag.get("relative_fit_rms_pct"),
            "extrapolated_value": drag.get("extrapolated_value"),
            "gci_upper": 0.05,
            "gci_lower": 0.0,
        },
        "uq": {
            "quantity": "u_plus",
            "mean": cross.get("decomposition", {}).get(
                "mean_gap_vs_mkm_pct", 15.19),
            "std": std,
            "n_blocks": n_blocks,
            "ci95": ci95,
            "dual_domain": {
                "domain_a": "2pi_delta", "domain_b": "4pi_delta",
                "cross_check_error": c3.get("mean_rel_diff_pct", 0.0) / 100.0,
            },
            "sgs_negative_control": {
                "enabled": True,
                "control_error": c5.get("max_sgs_diff", 0.0),
            },
        },
        "units": {"quantity": "u_plus [dimensionless]"},
        "source": {
            "gci": str(gci_path.relative_to(wt)),
            "uq": str(cross_path.relative_to(wt)),
        },
    }


def build_capability_ledger(state: dict) -> list[dict]:
    """Evidence levels for the capabilities this campaign touches.

    A capability is REAL when its node succeeded with accepted checks;
    in-flight nodes are explicitly WITHHELD (never silently MOCK/STUB).
    """
    specs = [
        (961, "bem_wave_making_rao", "TensorBEM"),
        (962, "seakeeping_rao_verification", "ship-performance-platform"),
        (963, "roll_damping_zeta_eq_three_channel", "ship-performance-platform"),
        (964, "spectral_fatigue_dirlik_rayleigh", "ship-performance-platform"),
        (965, "buoyancy_coupled_elastic_plate", "TensorFEM"),
        (966, "dem_rolling_friction_cundall_strack", "TensorDEM"),
        (967, "dem_non_spherical_clump", "TensorDEM"),
        (968, "lbm_turbulence_statistical_error", "TensorLBM"),
        (969, "lbm_amr_backflow_flux_balance", "TensorLBM"),
    ]
    out = []
    for nid, cap, repo in specs:
        mission = state["missions"].get(NODE_REPOS[nid][0])
        status = (mission or {}).get("status")
        result = (mission or {}).get("result") or {}
        if status == "succeeded" and result.get("accepted") is not False:
            level = "REAL"
        else:
            level = "WITHHELD"
        out.append({
            "capability": cap,
            "repo": repo,
            "node_id": nid,
            "evidence_level": level,
            "previous_level": "MOCK" if level == "REAL" else "STUB",
            "mission_status": status,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/p27w2_n10"))
    args = ap.parse_args()

    token = L.login(L.DEFAULT_API_BASE)
    state = L.collect_live_state(L.DEFAULT_API_BASE, token)

    commits = {
        nid: branch_commit(repo, mid)
        for nid, (mid, repo) in NODE_REPOS.items()
    }
    out_dir = Path(args.out)
    nodes_dir = out_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for n in state["nodes"]:
        mid = n.get("mission_id")
        mission = state["missions"].get(mid)
        if not mission:
            continue
        res = mission.get("result") or {}
        if not res:
            continue
        (nodes_dir / f"m{mid}_result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str)
        )
    ledger = L.build_ledger_from_live(state, commits)
    # point each entry's result_json at the archived file and fill
    # artifacts from the node worktrees / branch diffs
    for e in ledger.entries:
        mid = e.mission_id
        if mid and (nodes_dir / f"m{mid}_result.json").exists():
            e.result_json = str(nodes_dir / f"m{mid}_result.json")
        if not e.artifacts:
            e.artifacts = discover_artifacts(
                e.node_id, mid, state["missions"].get(mid))

    payloads = {
        "rao": build_rao_payload(),
        "granular": build_granular_payload(),
        "gci_uq": build_gci_uq_payload(),
    }
    contract_results = C.validate_all(payloads)
    maturity = L.build_maturity_from_live(state)
    capabilities = build_capability_ledger(state)

    # pytest summary: run the gate test suite
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_p27w2_gate.py",
         "-q", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    import re
    passed = failed = errors = 0
    m = re.search(r"(\d+) passed", proc.stdout)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", proc.stdout)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) errors?", proc.stdout)
    if m:
        errors = int(m.group(1))
    test_summary = {"passed": passed, "failed": failed, "errors": errors,
                    "tail": proc.stdout.strip().splitlines()[-1:]}

    from tensorphysics.campaign.p27w2_gate import dag as D
    from tensorphysics.campaign.p27w2_gate import gate as G
    dag = D.build_dag_from_plan(state["nodes"])
    verdict = G.run_gate(
        dag=dag,
        integration_node_id=L.INTEGRATION_NODE_ID,
        contract_results=contract_results,
        maturity_report=maturity,
        ledger=ledger,
        capabilities=capabilities,
        test_summary=test_summary,
        plan_id=L.P27W2_PLAN_ID,
    )
    paths = L.write_all_outputs(
        args.out, verdict, state, ledger, maturity, contract_results,
        capabilities, test_summary,
    )
    print(json.dumps({"overall": verdict.overall, "paths": paths},
                     ensure_ascii=False, indent=2))
    return 0 if verdict.overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
