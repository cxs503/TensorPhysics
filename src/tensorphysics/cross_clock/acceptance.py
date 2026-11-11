"""Machine-checkable acceptance record for the cross-clock coupling node.

Every criterion of the campaign node is evaluated here from *computed* numbers
(no hand-typed values), the record is written to ``artifacts/`` with a SHA256SUMS
file, and the narrative report is generated from that same record.

Criteria (node ``p474w1-n04``)
------------------------------
(a) conserved-quantity drift per coupling step below 0.1 %
(b) interpolation error bound computable and below the coupling tolerance
(c) adaptive clock effective at >= 3 clock ratios (1:1, 1:10, 1:100)
(d) >= 15 pytest cases covering interpolation / bound / adaptation / cases
(e) the interface contract document is complete
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .clock import AdaptiveClock, AdaptiveRun
from .contract import ContractError, drift_report
from .interpolate import interpolation_bounds, measured_error
from .scenarios import (
    DEM_CONTRACT,
    FEM_CONTRACT,
    attach_reference,
    default_dem_params,
    default_fem_params,
    dem_lbm_ode_reference,
    momentum_ledger,
    run_dem_lbm,
    run_fem_lbm,
)

__all__ = [
    "fem_study",
    "dem_study",
    "adaptive_study",
    "bound_soundness",
    "run_acceptance",
    "write_record",
    "criteria_table",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIFT_TARGET = 1.0e-3   # criterion (a): 0.1 % per coupling step
MIN_TESTS = 15          # criterion (d)


# --------------------------------------------------------------------------- #
# case studies (fixed clock)
# --------------------------------------------------------------------------- #
def fem_study(ratios: Sequence[int] = (1, 5, 10, 100), periods: float = 2.0,
              corrector: bool = True) -> dict:
    """Frequency error, trajectory error and drifts of the FEM-LBM case per ratio."""
    ref = run_fem_lbm(1, params={"periods": periods}, corrector=corrector)
    rows = []
    for r in ratios:
        case = run_fem_lbm(r, params={"periods": periods}, corrector=corrector)
        attach_reference(case, ref)
        d = drift_report(case.windows, scale=case.exchange_scale)
        xref = np.interp(case.times, ref.times, ref.x)
        traj = float(np.max(np.abs(case.x - xref)) / np.max(np.abs(ref.x)))
        rows.append(
            {
                "ratio": int(r),
                "coupling_step": case.derived["coupling_step"],
                "windows": case.derived["n_windows"],
                "frequency_hz": case.frequency,
                "frequency_error": case.frequency_error,
                "trajectory_error": traj,
                "ledger_drift_max": d["momentum"]["ledger_drift_max"],
                "reference_drift_max": d["momentum"]["reference_drift_max"],
                "stagger_residual_max": case.derived["stagger_max"],
                "bound_max": case.derived["bound_max"],
                "energy_final": float(0.5 * case.params["mass"] * case.v[-1] ** 2
                                      + 0.5 * case.params["stiffness"] * case.x[-1] ** 2),
            }
        )
    trend = [row["frequency_error"] for row in rows]
    return {
        "case": "fem_lbm_fsi",
        "params": default_fem_params() | {"periods": periods},
        "analytic": ref.analytic,
        "reference": {
            "ratio": 1,
            "frequency_hz": ref.frequency,
            "periods": periods,
            "windows": ref.derived["n_windows"],
        },
        "rows": rows,
        "frequency_error_monotone": bool(np.all(np.diff(trend) >= 0.0)),
        "contract": FEM_CONTRACT.as_dict(),
    }


def dem_study(ratios: Sequence[int] = (1, 5, 10, 100), periods: float = 3.0) -> dict:
    """Velocity error, momentum balance and drifts of the DEM-LBM case per ratio."""
    p = default_dem_params() | {"periods": periods}
    period = 2.0 * math.pi / p["omega"]
    ref = dem_lbm_ode_reference(1, p, periods * period)
    rows = []
    for r in ratios:
        case = run_dem_lbm(r, params=p, reference=ref)
        attach_reference(case, ref)
        # the drift normaliser of a momentum exchange is a momentum, not a velocity
        scale = p["mass"] * case["exchange_scale"]
        d = drift_report(case["windows"], scale=scale)
        led = momentum_ledger(case)
        rows.append(
            {
                "ratio": int(r),
                "coupling_step": case["coupling_step"],
                "windows": case["derived"]["n_windows"],
                "velocity_error": case["derived"]["velocity_error_max"],
                "fluid_velocity_error": case["derived"]["fluid_velocity_error_max"],
                "ledger_drift_max": d["momentum"]["ledger_drift_max"],
                "reference_drift_max": d["momentum"]["reference_drift_max"],
                "stagger_residual_max": case["derived"]["stagger_max"],
                "bound_max": case["derived"]["bound_max"],
                "momentum_balance_residual": led["relative_residual"],
            }
        )
    return {
        "case": "dem_lbm_particle",
        "params": p,
        "reference": {
            "kind": "closed-form two-degree-of-freedom linear model",
            "drag_rate": ref["drag_rate"],
            "forcing_period": period,
        },
        "rows": rows,
        "velocity_error_monotone": bool(np.all(np.diff([row["velocity_error"] for row in rows]) >= 0.0)),
        "contract": DEM_CONTRACT.as_dict(),
    }


# --------------------------------------------------------------------------- #
# adaptive clock study
# --------------------------------------------------------------------------- #
def adaptive_study(starts: Sequence[int] = (1, 10, 100), periods: float = 1.0,
                   dem_periods: float = 2.0) -> dict:
    """Run the controller from three different start ratios on both cases.

    ``bound_of`` is a *measured* bound (a fresh run at that ratio), never the
    controller's own model, so a controller refining in the wrong direction would
    be visible instead of being swallowed by its own assumption.
    """
    out = {}
    for name, contract, runner in (
        ("fem_lbm_fsi", FEM_CONTRACT, "fem"),
        ("dem_lbm_particle", DEM_CONTRACT, "dem"),
    ):
        fem_ref = run_fem_lbm(1, params={"periods": periods}) if runner == "fem" else None
        dem_p = (default_dem_params() | {"periods": dem_periods}) if runner == "dem" else None
        dem_ref = (
            dem_lbm_ode_reference(1, dem_p, dem_periods * 2.0 * math.pi / dem_p["omega"])
            if runner == "dem" else None
        )
        cache: dict[int, dict] = {}

        def measure(r: int) -> dict:
            r = int(max(1, r))
            if r in cache:
                return cache[r]
            if runner == "fem":
                case = run_fem_lbm(r, params={"periods": periods})
                attach_reference(case, fem_ref)
                d = drift_report(case.windows, scale=case.exchange_scale)
                rec = {
                    "ratio": r,
                    "bound": case.derived["bound_max"],
                    "ledger_drift": d["momentum"]["ledger_drift_max"],
                    "reference_drift": d["momentum"]["reference_drift_max"],
                    "stagger": case.derived["stagger_max"],
                    "frequency_error": case.frequency_error,
                }
            else:
                case = run_dem_lbm(r, params=dem_p, reference=dem_ref)
                attach_reference(case, dem_ref)
                scale = dem_p["mass"] * case["exchange_scale"]
                d = drift_report(case["windows"], scale=scale)
                rec = {
                    "ratio": r,
                    "bound": case["derived"]["bound_max"],
                    "ledger_drift": d["momentum"]["ledger_drift_max"],
                    "reference_drift": d["momentum"]["reference_drift_max"],
                    "stagger": case["derived"]["stagger_max"],
                    "velocity_error": case["derived"]["velocity_error_max"],
                }
            cache[r] = rec
            return rec

        starts_out = []
        for s in starts:
            ctrl = AdaptiveClock(
                ratio=int(s),
                tolerance=contract.tolerance,
                order=contract.order,
                min_ratio=contract.min_ratio,
                max_ratio=contract.max_ratio,
                growth=1.5,
                hysteresis=1.5,
            )
            run = AdaptiveRun(contract_tolerance=contract.tolerance)
            trail = []
            ratio = int(s)
            for _ in range(12):
                rec = measure(ratio)
                action = "hold"
                trail.append(rec)
                if rec["bound"] > contract.tolerance:
                    action = "refine"
                elif rec["bound"] * ctrl.hysteresis < contract.tolerance and ratio < contract.max_ratio:
                    action = "coarsen"
                run.record(ratio, rec["bound"], rec["reference_drift"], action)
                nxt = ctrl.update(rec["bound"])
                if nxt == ratio and (rec["bound"] <= contract.tolerance):
                    break
                ratio = nxt
            settled = measure(ratio)
            starts_out.append(
                {
                    "start_ratio": int(s),
                    "final_ratio": int(ratio),
                    "iterations": len(trail),
                    "final_bound": settled["bound"],
                    "final_ledger_drift": settled["ledger_drift"],
                    "final_reference_drift": settled["reference_drift"],
                    "final_accuracy": settled.get("frequency_error", settled.get("velocity_error")),
                    "tolerance": contract.tolerance,
                    "tolerance_met": bool(settled["bound"] <= contract.tolerance),
                    "drift_met": bool(settled["ledger_drift"] < DRIFT_TARGET
                                      and settled["reference_drift"] < DRIFT_TARGET),
                    "ratios_visited": run.ratios,
                    "bounds": run.bounds,
                    "actions": run.actions,
                    "trail": trail,
                }
            )
        out[name] = {"contract": contract.as_dict(), "starts": starts_out}
    return out


# --------------------------------------------------------------------------- #
# bound soundness on a manufactured solution
# --------------------------------------------------------------------------- #
def bound_soundness(hs: Sequence[float] = (0.02, 0.01, 0.005, 0.0025, 0.00125)) -> dict:
    """Check ``bound >= measured`` for a smooth analytic signal, and its tightness.

    The signal is ``r(t) = sin(2 pi t)`` sampled at ``dt = 2e-4`` (the producer's
    fine samples); the reconstruction is the window mean (hold).  A bound that is
    below the measured error would be unsound; a bound that is orders of magnitude
    above it would be useless.  Both are reported.
    """
    dt = 2.0e-4
    t = np.arange(0.0, 2.0 + dt / 2, dt)
    r = np.sin(2.0 * np.pi * t)
    rel = np.cos(2.0 * np.pi * t) * 2.0 * np.pi  # true derivative magnitude
    rows = []
    for h in hs:
        ratio = int(round(h / dt))
        if ratio < 1:
            continue
        h = ratio * dt
        bounds, trues = [], []
        for k in range(1, int(2.0 / h) - 1):
            t0 = k * h
            m = (t >= t0) & (t < t0 + h)
            if not np.any(m):
                continue
            b = interpolation_bounds(t[m], r[m], h, "hold", estimator="fine")
            e = measured_error(t, r, t0, h, "hold")
            bounds.append(b["integral_bound"])
            trues.append(e["integral_error"])
        bounds = np.asarray(bounds)
        trues = np.asarray(trues)
        if bounds.size == 0:
            continue
        rows.append(
            {
                "window": h,
                "windows": int(bounds.size),
                "bound_mean": float(np.mean(bounds)),
                "measured_mean": float(np.mean(trues)),
                "sound_windows": int(np.sum(bounds >= trues)),
                "windows_total": int(bounds.size),
                "tightness_median": float(np.median(bounds / np.maximum(trues, 1e-300))),
            }
        )
    all_sound = all(r["sound_windows"] == r["windows_total"] for r in rows)
    return {
        "signal": "sin(2*pi*t), dt=2e-4, hold reconstruction",
        "rows": rows,
        "sound": bool(all_sound),
        "tightness_max": float(max(r["tightness_median"] for r in rows)) if rows else None,
        "comment": "integral bound = M1 h^2 / 4 with M1 = max|dr/dt| from the fine samples",
    }


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def _test_count() -> dict:
    files = sorted((REPO_ROOT / "tests").glob("test_cross_clock_*.py"))
    n = 0
    per_file = {}
    for f in files:
        txt = f.read_text()
        k = len(re.findall(r"^def (test_\w+)", txt, flags=re.M))
        per_file[f.name] = k
        n += k
    return {"files": per_file, "total": n}


def _contract_doc_check() -> dict:
    path = REPO_ROOT / "docs" / "P474W1_N04_CROSS_CLOCK.md"
    required = [
        "## 1. Case introduction",
        "## 2. Method",
        "## 3. Results",
        "## 4. Figures",
        "## 5. Pitfalls and limits",
        "## 6. Error judgement",
        "## 7. Interface contract",
    ]
    if not path.exists():
        return {"path": str(path.relative_to(REPO_ROOT)), "exists": False, "missing": required}
    txt = path.read_text()
    missing = [h for h in required if h not in txt]
    # deliberately no character count / digest of the document here: the document
    # itself cites the record's digest, and storing a length would make the two
    # artefacts depend on each other (a fixpoint that never converges).
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "missing": missing,
        "complete": not missing,
    }


def criteria_table(record: dict) -> list[dict]:
    fem = record["fem_lbm_fsi"]
    dem = record["dem_lbm_particle"]
    ad = record["adaptive"]
    fem_drift = max(r["reference_drift_max"] for r in fem["rows"])
    dem_drift = max(r["reference_drift_max"] for r in dem["rows"])
    adapted_drift = max(
        s["final_ledger_drift"] for case in ad.values() for s in case["starts"]
    )
    adapted_ref = max(
        s["final_reference_drift"] for case in ad.values() for s in case["starts"]
    )
    tol_ok = all(s["tolerance_met"] for case in ad.values() for s in case["starts"])
    drift_ok = all(s["drift_met"] for case in ad.values() for s in case["starts"])
    starts_ok = all(len(case["starts"]) >= 3 for case in ad.values())
    tests = record["tests"]
    doc = record["contract_doc"]
    rows = [
        {
            "id": "a",
            "description": "conserved-quantity drift per coupling step < 0.1 %",
            "value": max(adapted_drift, adapted_ref),
            "detail": {
                "adapted_ledger_drift_max": adapted_drift,
                "adapted_reference_drift_max": adapted_ref,
                "fem_reference_drift_max_fixed_clock": fem_drift,
                "dem_reference_drift_max_fixed_clock": dem_drift,
            },
            "threshold": DRIFT_TARGET,
            "note": (
                "the ledger drift (produced vs applied) is an exchange invariant and is "
                "identically zero for the impulse-exact FEM pair; the reference drift "
                "(window integral vs the fine-clock reference) is the physical number"
            ),
            "pass": bool(max(adapted_drift, adapted_ref) < DRIFT_TARGET),
        },
        {
            "id": "b",
            "description": "interpolation error bound computable and < coupling tolerance",
            "value": max(
                s["final_bound"] / case["contract"]["tolerance"]
                for case in ad.values()
                for s in case["starts"]
            ),
            "detail": {
                "per_case": {
                    name: {
                        "tolerance": case["contract"]["tolerance"],
                        "bound_final": [s["final_bound"] for s in case["starts"]],
                        "within": [s["tolerance_met"] for s in case["starts"]],
                    }
                    for name, case in ad.items()
                },
                "soundness": record["bound_soundness"],
                "note": "value is bound/tolerance, so 1.0 means borderline",
            },
            "threshold": 1.0,
            "pass": bool(tol_ok and record["bound_soundness"]["sound"]),
        },
        {
            "id": "c",
            "description": "adaptive clock effective at >= 3 clock ratios (1:1, 1:10, 1:100)",
            "value": sum(len(case["starts"]) for case in ad.values()),
            "detail": {
                name: [
                    {
                        "start": s["start_ratio"],
                        "final": s["final_ratio"],
                        "tolerance_met": s["tolerance_met"],
                        "drift_met": s["drift_met"],
                        "iterations": s["iterations"],
                    }
                    for s in case["starts"]
                ]
                for name, case in ad.items()
            },
            "threshold": 3,
            "pass": bool(starts_ok and tol_ok and drift_ok and all(
                len(case["starts"]) >= 3 for case in ad.values())),
        },
        {
            "id": "d",
            "description": "pytest >= 15 cases over interpolation / bound / adaptation / cases",
            "value": tests["total"],
            "detail": tests,
            "threshold": MIN_TESTS,
            "pass": bool(tests["total"] >= MIN_TESTS),
        },
        {
            "id": "e",
            "description": "interface contract document complete",
            "value": 1 if doc.get("complete") else 0,
            "detail": doc,
            "threshold": 1,
            "pass": bool(doc.get("complete")),
        },
    ]
    return rows


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def run_acceptance(quick: bool = False) -> dict:
    ratios = (1, 10, 100) if quick else (1, 5, 10, 100)
    fem = fem_study(ratios=ratios, periods=1.0 if quick else 2.0)
    dem = dem_study(ratios=ratios, periods=2.0 if quick else 3.0)
    ad = adaptive_study(
        starts=(1, 10, 100),
        periods=0.5 if quick else 1.0,
        dem_periods=1.5 if quick else 2.0,
    )
    record = {
        "node": "p474w1-n04-cross-clock",
        "module": "tensorphysics.cross_clock",
        "fem_lbm_fsi": fem,
        "dem_lbm_particle": dem,
        "adaptive": ad,
        "bound_soundness": bound_soundness(),
        "tests": _test_count(),
        "contract_doc": _contract_doc_check(),
        "drift_target": DRIFT_TARGET,
    }
    record["criteria"] = criteria_table(record)
    record["accepted"] = bool(all(c["pass"] for c in record["criteria"]))
    return record


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_record(record: dict, out_dir: Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else REPO_ROOT / "artifacts" / "p474w1_n04_cross_clock"
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(record), indent=1, sort_keys=True)
    (out / "verification.json").write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    lines = []
    # SHA256SUMS cannot contain its own digest without being stale by construction,
    # so it is excluded from its own listing (the record digest is in the first line
    # of the report and in docs/P474W1_N04_CROSS_CLOCK.md).
    for name in sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256((out / name).read_bytes()).hexdigest()
        lines.append(f"{h}  {name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return {"dir": str(out.relative_to(REPO_ROOT)), "sha256": digest, "files": lines}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    rec = run_acceptance(quick=quick)
    info = write_record(rec)
    print(json.dumps({"accepted": rec["accepted"], "record": info["dir"], "sha256": info["sha256"]}, indent=1))
    for c in rec["criteria"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] ({c['id']}) {c['description']}: {c['value']:.6g} vs {c['threshold']}")
    return 0 if rec["accepted"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())