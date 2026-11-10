"""Run the 3-D / 2-D controlled icebreaking comparison and persist artifacts.

Usage:  python scripts/run_fsi_ice_benchmark.py [--steps N] [--out DIR]

The two configurations share every parameter except ``dim`` (and therefore
``ny``); that is the predeclared controlled variable of the experiment.  All
numbers written to ``summary.json`` come from the run objects -- nothing is
hand-copied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.fsi_ice.runner import CouplingConfig, CouplingRunner  # noqa: E402


def run_one(dim: int, n_steps: int, fluid_steps: int) -> dict:
    cfg = CouplingConfig(dim=dim, ny=28 if dim == 3 else 1, n_steps=n_steps,
                         fluid_steps_per_couple=fluid_steps)
    t0 = time.time()
    res = CouplingRunner(cfg, label=f"{dim}d").run()
    comps = {k: v.tolist() for k, v in res.force_components.items()}
    summary = res.summary()
    summary["wall_time_s"] = round(time.time() - t0, 2)
    summary["components_mean"] = {k: float(np.mean(v)) for k, v in res.force_components.items()}
    summary["components_sum"] = {k: float(np.sum(v)) for k, v in res.force_components.items()}
    summary["ice_resistance_mean"] = float(
        np.mean(res.force_components["fracture"]) + np.mean(res.force_components["dem_contact"])
    )
    summary["fracture_mean"] = float(np.mean(res.force_components["fracture"]))
    summary["contact_mean"] = float(np.mean(res.force_components["dem_contact"]))
    summary["fluid_mean"] = float(np.mean(res.force_components["fluid"]))
    summary["diagnostics"] = res.diagnostics
    summary["ledger"] = res.ledger
    summary["manifest"] = res.manifest.to_dict()
    summary["n_steps"] = n_steps
    summary["times"] = res.times.tolist()
    summary["force_x"] = res.force_history[:, 0].tolist()
    summary["energy"] = {k: v.tolist() for k, v in res.energy_history.items()}
    summary["fragment_diameters"] = res.fragment_diameters.tolist()
    summary["components"] = comps
    summary["scaling"] = cfg.scaling.as_dict()
    summary["force_to_si"] = cfg.scaling.force_to_si(1.0)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--fluid-steps", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "m9428"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    r3 = run_one(3, args.steps, args.fluid_steps)
    r2 = run_one(2, args.steps, args.fluid_steps)

    f3 = r3["ice_resistance_mean"]
    f2 = r2["ice_resistance_mean"]
    rel = abs(f3 - f2) / max(abs(f2), 1e-30)
    d3 = r3["mean_fragment_diameter"]
    d2 = r2["mean_fragment_diameter"]
    payload = {
        "three_d": r3,
        "two_d": r2,
        "comparison": {
            "ice_resistance_3d": f3,
            "ice_resistance_2d": f2,
            "ice_resistance_relative_difference": rel,
            "ice_resistance_difference_percent": 100.0 * rel,
            "fragment_diameter_3d": d3,
            "fragment_diameter_2d": d2,
            "fragment_ratio_2d_over_3d": d2 / max(d3, 1e-30),
            "contact_force_3d": r3["contact_mean"],
            "contact_force_2d": r2["contact_mean"],
            "fracture_force_3d": r3["fracture_mean"],
            "fracture_force_2d": r2["fracture_mean"],
            "fluid_force_3d": r3["fluid_mean"],
            "fluid_force_2d": r2["fluid_mean"],
            "energy_error_3d": r3["energy_balance_error"],
            "energy_error_2d": r2["energy_balance_error"],
            "diverged_3d": r3["diverged"],
            "diverged_2d": r2["diverged"],
        },
        "criteria": {},
    }
    c = payload["comparison"]
    payload["criteria"] = {
        "C1_force_difference_15_30pct": {
            "value_percent": c["ice_resistance_difference_percent"],
            "pass": bool(15.0 <= c["ice_resistance_difference_percent"] <= 30.0),
            "note": "measured on the ice path (fracture + DEM hull contact)",
        },
        "C2_fragment_size_3d_smaller": {
            "value_ratio_2d_over_3d": c["fragment_ratio_2d_over_3d"],
            "pass": bool(c["fragment_diameter_3d"] < c["fragment_diameter_2d"]),
        },
        "C3_energy_error_below_5pct": {
            "value_3d": c["energy_error_3d"],
            "value_2d": c["energy_error_2d"],
            "pass": bool(c["energy_error_3d"] < 0.05 and c["energy_error_2d"] < 0.05),
        },
        "C4_100_steps_no_divergence": {
            "steps": args.steps,
            "pass": bool((not c["diverged_3d"]) and (not c["diverged_2d"]) and args.steps >= 100),
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({k: v for k, v in c.items()}, indent=2, default=float))
    print(json.dumps(payload["criteria"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())