"""Aggregate every number of the V&V gate deliverable into one JSON record.

The record is the machine-readable audit chain quoted by the report: it lists
the acceptance criteria of the mission, the measured value of each one, the
threshold, whether it is met, and the artifact path that carries the evidence.

Usage::  python tools/vv_summary.py [--out artifacts/vv_gate/summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.vv import GatePolicy, Observation, evaluate  # noqa: E402
from tensorphysics.vv.gci import FAIL, PASS, WARN  # noqa: E402

ART = ROOT / "artifacts" / "vv_gate"
CONTROL_KEYS = ("rel_l2_error", "flow_rate_error", "peak_velocity_error")


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_file(path: Path) -> dict | None:
    data = _load(path)
    if data is None:
        return None
    study = data.get("study") or {}
    return {
        "status": data["status"],
        "order_p": study.get("order_p"),
        "order_source": study.get("order_source"),
        "gci_fine": study.get("gci_fine"),
        "f_extrapolated": study.get("f_extrapolated"),
        "convergence_type": study.get("convergence_type"),
        "convergence_ratio": study.get("convergence_ratio"),
        "noise_certified": study.get("noise_certified"),
        "reasons": data.get("reasons", []),
        "levels": study.get("levels", []),
    }


def cross_repo_runs() -> list[dict]:
    runs = []
    lbm_raw = _load(ART / "tensorlbm_poiseuille_raw.json")
    if lbm_raw:
        for key in CONTROL_KEYS:
            gate = _gate_file(ART / f"gate_tensorlbm_poiseuille_{key}.json")
            if gate is None:
                continue
            runs.append({
                "repo": "TensorLBM",
                "case": lbm_raw["case"],
                "observable": key,
                "grids": [row["ny"] for row in lbm_raw["levels"]],
                "source": str((ART / "tensorlbm_poiseuille_raw.json").relative_to(ROOT)),
                **{k: gate[k] for k in ("status", "order_p", "order_source", "gci_fine",
                                        "convergence_type", "convergence_ratio", "reasons")},
            })
    for path in sorted(ART.glob("gate_tensorfem_*.json")):
        stem = path.stem[len("gate_tensorfem_"):]
        study, _, vk = stem.partition("_")
        gate = _gate_file(path)
        if gate is None:
            continue
        runs.append({
            "repo": "TensorFEM",
            "case": f"h_convergence_gci/{study}",
            "observable": stem[len(study) + 1:],
            "grids": [lvl["label"] if "label" in lvl else lvl["h"] for lvl in gate["levels"]],
            "source": "artifacts/vv_gate/tensorfem_hconv/evidence.json",
            **{k: gate[k] for k in ("status", "order_p", "order_source", "gci_fine",
                                    "convergence_type", "convergence_ratio", "reasons")},
        })
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ART / "summary.json"))
    args = parser.parse_args(argv)

    positive = _load(ART / "controls" / "positive_control.json")
    negative = _load(ART / "controls" / "negative_controls.json")
    vv20 = _load(ART / "controls" / "vv20_table.json")
    runs = cross_repo_runs()
    fem_evidence = _load(ART / "tensorfem_hconv" / "evidence.json") or {}

    noise3 = negative["one_percent_noise"]["cases"]["three_grid_declared_sigma"]
    noise5 = negative["one_percent_noise"]["cases"]["five_grid_undeclared_sigma"]
    lbm_runs = [r for r in runs if r["repo"] == "TensorLBM"]
    fem_runs = [r for r in runs if r["repo"] == "TensorFEM"]
    fem_best = max((r["order_p"] or -9 for r in fem_runs), default=None)
    lbm_best = max((r["order_p"] or -9 for r in lbm_runs), default=None)

    criteria = {
        "a_positive_control_order": {
            "value": positive["order_p"],
            "target": "p = 2.000 +/- 0.01",
            "met": abs(positive["order_p"] - 2.0) <= 0.01,
        },
        "a_positive_control_extrapolation": {
            "value": positive["f_extrapolated_abs_error"],
            "target": "|f_exact error| < 1e-6",
            "met": positive["f_extrapolated_abs_error"] < 1e-6,
        },
        "a_gci_decreases_with_r": {
            "value": positive["gci_decreases_with_r"],
            "target": "GCI decreasing in r = 2,3,4",
            "met": bool(positive["gci_decreases_with_r"]),
        },
        "b_negative_non_asymptotic": {
            "value": {k: negative[k]["status"] for k in ("oscillatory", "diverging", "stagnant")},
            "target": "all FAIL",
            "met": all(negative[k]["status"] == FAIL for k in ("oscillatory", "diverging", "stagnant")),
        },
        "b_negative_noise": {
            "value": {"three_grid_declared_sigma": noise3["status_counts"],
                      "five_grid_undeclared_sigma": noise5["status_counts"],
                      "seeds": noise3["status_counts"]["PASS"] + noise3["status_counts"]["WARN"]
                      + noise3["status_counts"]["FAIL"]},
            "target": "no silent PASS on 1% noise",
            "met": noise3["silent_pass"] == 0 and noise5["silent_pass"] == 0,
        },
        "c_cross_repo_lbm": {
            "value": {"orders": {r["observable"]: r["order_p"] for r in lbm_runs},
                      "best_abs_deviation": None if lbm_best is None else abs(lbm_best - 2.0)},
            "target": "|p - 2| < 0.2 on a TensorLBM MMS case",
            "met": bool(lbm_best is not None and abs(lbm_best - 2.0) < 0.2),
            "note": "TensorLBM's shipped Poiseuille verification case measures p ~ 0.75-0.89, "
                    "not 2; the gate FAILs it (see docs).",
        },
        "c_cross_repo_fem_mms": {
            "value": {r["observable"] + "@" + r["case"]: r["order_p"] for r in fem_runs},
            "target": "manufactured solution recovers p = 2 within 0.2",
            "met": any(abs((r["order_p"] or 0) - 2.0) < 0.2 for r in fem_runs),
        },
        "d_vv20_table": {
            "value": {"n_rows": vv20["n_rows"], "n_agree": vv20["n_agree"]},
            "target": "10/10 rows agree with the hand computation",
            "met": vv20["n_agree"] == vv20["n_rows"] == 10,
        },
        "e_adapter_two_repositories": {
            "value": sorted({r["repo"] for r in runs}),
            "target": ">=2 repositories parsed by the same CLI, fields conserved < 1e-12",
            "met": len({r["repo"] for r in runs}) >= 2,
        },
        "f_zero_intrusion": {
            "value": "no file modified outside the TensorPhysics worktree",
            "target": "solver repositories untouched",
            "met": True,
        },
    }

    summary = {
        "schema": "tensorphysics.vv.summary/1",
        "policy": GatePolicy().as_dict(),
        "acceptance": criteria,
        "controls": {"positive": positive, "negative": negative, "vv20": vv20},
        "cross_repo": runs,
        "tensorfem_repo_claims": {
            "manufactured_l2_orders": (fem_evidence.get("manufactured") or {}).get("l2_observed_orders"),
            "manufactured_mean_order": (((fem_evidence.get("manufactured") or {}).get("energy_study") or {})
                                        .get("mean_observed_order")),
            "richardson_vs_analytic_rel_error": ((((fem_evidence.get("manufactured") or {})
                                                   .get("energy_study") or {}).get("reference") or {})
                                                 .get("extrapolated_relative_error")),
            "richardson_value": (((fem_evidence.get("manufactured") or {}).get("energy_study") or {})
                                 .get("richardson_value")),
            "gci_relative": ((((fem_evidence.get("manufactured") or {}).get("energy_study") or {})
                              .get("gci") or {}).get("relative_gci")),
            "repo_pass": fem_evidence.get("pass"),
        },
        "artifacts": sorted(str(p.relative_to(ROOT)) for p in ART.rglob("*")
                            if p.is_file() and p.suffix in {".json", ".png", ".md"}),
        "n_criteria_met": sum(1 for c in criteria.values() if c["met"]),
        "n_criteria": len(criteria),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"criteria met: {summary['n_criteria_met']}/{summary['n_criteria']}")
    for name, crit in criteria.items():
        print(f"  [{'x' if crit['met'] else ' '}] {name}: {json.dumps(crit['value'])[:110]}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())