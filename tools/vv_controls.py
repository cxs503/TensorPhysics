"""Self-verification of the V&V gate: manufactured controls + ASME V&V 20 table.

Writes machine-readable evidence into ``artifacts/vv_gate/controls``:

* ``positive_control.json``  -- f(h) = f_exact + C h^2 on three grids: observed
  order, Richardson value, GCI and the GCI-vs-r sweep;
* ``negative_controls.json`` -- oscillatory / diverging / stagnant sequences and
  seeded 1 % noise (3 grids with declared sigma, 5 grids without);
* ``vv20_table.json``        -- ten (f_sim, f_exp, U_num, U_input) cases with the
  hand-computed verdict and the gate's verdict side by side.

Usage::  python tools/vv_controls.py [--out artifacts/vv_gate/controls]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.vv import GatePolicy, Observation, evaluate  # noqa: E402
from tensorphysics.vv.gci import sweep_refinement_ratio  # noqa: E402
from tensorphysics.vv.validation import evaluate_validation_table  # noqa: E402

HS3 = (1.0, 0.5, 0.25)
HS5 = (1.0, 0.5, 0.25, 0.125, 0.0625)


def positive_control() -> dict:
    f_exact, c = 3.0, 0.5
    model = lambda h: f_exact + c * h * h  # noqa: E731
    verdict = evaluate([Observation(h=h, f=model(h), label=f"h={h:g}") for h in HS3],
                       quantity="manufactured_f", policy=GatePolicy(expected_order=2.0))
    study = verdict.study
    sweep = sweep_refinement_ratio(model, h_coarse=1.0, p=2.0,
                                   ratios=(2.0, 3.0, 4.0), safety_factor=1.25)
    five = evaluate([Observation(h=h, f=model(h)) for h in HS5],
                    quantity="manufactured_f", policy=GatePolicy(expected_order=2.0))
    return {
        "case": "f(h) = 3 + 0.5 h^2, three uniformly refined grids r=2",
        "levels": [{"h": h, "f": model(h)} for h in HS3],
        "order_p": study.order_p,
        "order_error_vs_theory": abs(study.order_p - 2.0),
        "f_extrapolated": study.f_extrapolated,
        "f_exact": f_exact,
        "f_extrapolated_abs_error": abs(study.f_extrapolated - f_exact),
        "gci_fine": study.gci_fine,
        "gci_vs_r": [{"r": r, "gci": v} for r, v in sweep],
        "gci_decreases_with_r": all(a[1] > b[1] for a, b in zip(sweep, sweep[1:])),
        "status": verdict.status,
        "five_grid": {"order_p": five.study.order_p,
                      "fit_residual_rel": five.study.fit_residual_rel,
                      "status": five.status},
        "checks": [c["detail"] for c in verdict.checks],
    }


def negative_controls() -> dict:
    out = {}
    oscillatory = {1.0: 1.5, 0.5: 1.2, 0.25: 1.35, 0.125: 1.275}
    verdict = evaluate([Observation(h=h, f=f) for h, f in oscillatory.items()],
                       quantity="oscillatory")
    out["oscillatory"] = {"levels": [{"h": h, "f": f} for h, f in oscillatory.items()],
                          "convergence_type": verdict.study.convergence_type,
                          "convergence_ratio": verdict.study.convergence_ratio,
                          "status": verdict.status,
                          "reasons": list(verdict.reasons)}
    diverging = {1.0: 1.0, 0.5: 1.1, 0.25: 1.4, 0.125: 2.0}
    verdict = evaluate([Observation(h=h, f=f) for h, f in diverging.items()],
                       quantity="diverging")
    out["diverging"] = {"levels": [{"h": h, "f": f} for h, f in diverging.items()],
                        "convergence_type": verdict.study.convergence_type,
                        "convergence_ratio": verdict.study.convergence_ratio,
                        "status": verdict.status,
                        "reasons": list(verdict.reasons)}
    stagnant = evaluate([Observation(h=h, f=1.0) for h in HS3], quantity="stagnant")
    out["stagnant"] = {"status": stagnant.status, "reasons": list(stagnant.reasons)}

    model = lambda h: 3.0 + 0.5 * h * h  # noqa: E731
    noise = {"rel_noise": 0.01, "seeds": 25, "cases": {}}
    for name, hs, declare_sigma in (
        ("three_grid_declared_sigma", HS3, True),
        ("five_grid_undeclared_sigma", HS5, False),
    ):
        statuses, orders, ratios = [], [], []
        for seed in range(noise["seeds"]):
            rng = random.Random(seed)
            levels = []
            for h in hs:
                f = model(h)
                perturbed = f * (1.0 + rng.uniform(-0.01, 0.01))
                levels.append(Observation(h=h, f=perturbed,
                                          sigma=0.01 * f if declare_sigma else None))
            verdict = evaluate(levels, quantity=f"noise_{name}")
            statuses.append(verdict.status)
            orders.append(verdict.study.order_p)
            ratios.append(verdict.study.noise_ratio)
        noise["cases"][name] = {
            "grids": len(hs),
            "declared_sigma": declare_sigma,
            "status_counts": {s: statuses.count(s) for s in ("PASS", "WARN", "FAIL")},
            "silent_pass": statuses.count("PASS"),
            "order_min": min(orders), "order_max": max(orders),
            "noise_ratio_median": sorted(r for r in ratios if r is not None)[
                len([r for r in ratios if r is not None]) // 2] if any(ratios) else None,
        }
    out["one_percent_noise"] = noise

    noisy3 = []
    rng = random.Random(3)
    levels = []
    for h in HS3:
        f = model(h)
        levels.append(Observation(h=h, f=f * (1.0 + rng.uniform(-0.01, 0.01))))
    verdict = evaluate(levels, quantity="three_grid_undeclared")
    noisy3 = {"status": verdict.status,
              "noise_certified": verdict.study.noise_certified,
              "strict_status": evaluate(levels, quantity="three_grid_undeclared",
                                        policy=GatePolicy(strict=True)).status,
              "notes": list(verdict.study.notes)}
    out["three_grid_undeclared_noise"] = noisy3
    return out


HAND_TABLE = [
    ("case01", 1.00, 1.05, 0.03, 0.03, 0.04242640687119285, "FAIL"),
    ("case02", 1.00, 1.02, 0.02, 0.01, 0.022360679774997898, "PASS"),
    ("case03", 2.00, 2.00, 0.10, 0.10, 0.1414213562373095, "PASS"),
    ("case04", 10.00, 10.50, 0.00, 0.50, 0.5, "PASS"),
    ("case05", 10.00, 10.5000001, 0.00, 0.50, 0.5, "FAIL"),
    ("case06", 0.50, 0.45, 0.02, 0.03, 0.03605551275463989, "FAIL"),
    ("case07", -1.00, -1.02, 0.01, 0.01, 0.014142135623730951, "FAIL"),
    ("case08", 100.0, 100.0, 5.00, 0.00, 5.0, "PASS"),
    ("case09", 1.00, 1.00, 0.00, 0.00, 0.0, "PASS"),
    ("case10", 1.00, 1.001, 0.00, 0.00, 0.0, "FAIL"),
]


def vv20_table() -> dict:
    results = evaluate_validation_table([
        {"label": label, "f_sim": f_sim, "f_exp": f_exp, "u_num": u_num, "u_input": u_input}
        for label, f_sim, f_exp, u_num, u_input, _u, _s in HAND_TABLE
    ])
    rows, agree = [], 0
    for result, (label, f_sim, f_exp, u_num, u_input, u_hand, status_hand) in zip(results, HAND_TABLE):
        agrees = (result.status == status_hand
                  and abs(result.u_val - u_hand) <= 1e-12 * max(1.0, abs(u_hand)))
        agree += int(agrees)
        rows.append({
            "label": label, "f_sim": f_sim, "f_exp": f_exp, "u_num": u_num,
            "u_input": u_input, "u_val_gate": result.u_val, "u_val_hand": u_hand,
            "error": result.error, "abs_error_over_u_val": result.ratio,
            "status_gate": result.status, "status_hand": status_hand,
            "agrees": agrees,
        })
    return {"rows": rows, "n_rows": len(rows), "n_agree": agree,
            "agreement_fraction": agree / len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/vv_gate/controls")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    positive = positive_control()
    negative = negative_controls()
    table = vv20_table()
    (out / "positive_control.json").write_text(json.dumps(positive, indent=2) + "\n", encoding="utf-8")
    (out / "negative_controls.json").write_text(json.dumps(negative, indent=2) + "\n", encoding="utf-8")
    (out / "vv20_table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    # the CLI consumes the rows directly in table mode
    (out / "vv20_table_rows.json").write_text(
        json.dumps([{k: row[k] for k in ("label", "f_sim", "f_exp", "u_num", "u_input")}
                    for row in table["rows"]], indent=2) + "\n", encoding="utf-8")

    print(f"positive control : p = {positive['order_p']:.6f} "
          f"(|dp| = {positive['order_error_vs_theory']:.2e}), "
          f"|f_exact error| = {positive['f_extrapolated_abs_error']:.2e}, "
          f"GCI decreasing with r = {positive['gci_decreases_with_r']}, {positive['status']}")
    for name, case in negative.items():
        if name == "one_percent_noise":
            for tag, sub in case["cases"].items():
                print(f"1% noise {tag}: {sub['status_counts']} silent_pass={sub['silent_pass']}")
        elif isinstance(case, dict) and "status" in case:
            print(f"negative control {name}: {case['status']}")
    print(f"V&V 20 table: {table['n_agree']}/{table['n_rows']} rows agree with the hand computation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())