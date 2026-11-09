"""IN30-N01..N08 regression corpus for the V&V machine gate (plan 48).

The corpus is deliberately *analytic*: each case is a manufactured observable
``f(h)`` on a systematically refined grid family, so the true order ``p``, the
continuum value ``f_exact`` and the expected gate decision are known in closed
form and the golden file can be regenerated bit-for-bit on any machine.

Cases (the IN30 family of campaign plan 48, re-expressed solver-neutrally):

======  ====================================================  ============
case    physics analogue                                       expectation
======  ====================================================  ============
N01     2nd-order diffusion/heat-flux observable, 3 grids     WARN (pass but
        without a declared noise floor                          undeclared noise)
N02     same study with per-level round-off declared           PASS
N03     mixed 1.5-order advection-like observable, 4 grids    PASS, p ~ 1.5
N04     stagnant sequence (a solver that stopped converging)   FAIL
N05     oscillatory sequence (odd/even decoupling)             FAIL
N06     order outside the admissible band (p ~ 8)              FAIL
N07     ASME V&V 20 validation, |E| <= U_val                   PASS
N08     ASME V&V 20 validation, |E| >  U_val                   FAIL
======  ====================================================  ============

Usage::

    python tools/vv_in30_regression.py --out tests/data/vv_in30/golden_head.json

The emitted document carries a ``digest`` (sha256 of the canonical case list) so
a silent drift in the gate is detectable even if the human-readable file is
edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.vv.gci import GatePolicy, Observation, evaluate  # noqa: E402

SCHEMA = "tensorphysics.vv.in30-regression/1"


def _power_law(f_exact: float, coefficient: float, p: float):
    def model(h: float) -> float:
        return f_exact + coefficient * h**p

    return model


def _levels(model, hs, sigma=None):
    out = []
    for i, h in enumerate(hs):
        obs = Observation(h=h, f=model(h))
        if sigma is not None:
            obs = Observation(h=h, f=model(h), sigma=sigma[i])
        out.append(obs)
    return out


def build_cases() -> list[dict]:
    """The eight IN30 cases as gate invocations."""
    cases: list[dict] = []

    # N01 -- 2nd order, three grids, no declared round-off
    m = _power_law(0.724031, 0.185, 2.0)
    cases.append(
        dict(
            case="IN30-N01",
            quantity="heat_flux_nusselt",
            physics="second-order diffusion observable, three uniform grids rho=2",
            levels=_levels(m, (0.04, 0.02, 0.01)),
            kwargs=dict(policy=GatePolicy(expected_order=2.0)),
        )
    )

    # N02 -- same study, per-level round-off declared (>=4 grids + sigma)
    m = _power_law(0.724031, 0.185, 2.0)
    cases.append(
        dict(
            case="IN30-N02",
            quantity="heat_flux_nusselt",
            physics="same observable with a certified round-off floor (4 grids, sigma)",
            levels=_levels(m, (0.08, 0.04, 0.02, 0.01), sigma=(2e-6, 1e-6, 5e-7, 2.5e-7)),
            kwargs=dict(policy=GatePolicy(expected_order=2.0)),
        )
    )

    # N03 -- mixed order 1.5 (upwind-like advection term), four grids
    m = _power_law(1.0, 0.21, 1.5)
    cases.append(
        dict(
            case="IN30-N03",
            quantity="advected_scalar_l2_error",
            physics="mixed 1.5-order transport observable, four grids",
            levels=_levels(m, (0.08, 0.04, 0.02, 0.01)),
            kwargs=dict(policy=GatePolicy(expected_order=1.5)),
        )
    )

    # N04 -- stagnant sequence: identical values on every grid
    stagnant = [Observation(h=h, f=0.5) for h in (0.04, 0.02, 0.01)]
    cases.append(
        dict(
            case="IN30-N04",
            quantity="stagnant_observable",
            physics="solver stopped converging, three identical values",
            levels=stagnant,
            kwargs={},
        )
    )

    # N05 -- oscillatory: odd/even decoupling
    m = _power_law(0.5, 0.1, 2.0)
    osc = [
        Observation(h=0.04, f=m(0.04)),
        Observation(h=0.02, f=m(0.02) + 0.006),
        Observation(h=0.01, f=m(0.01) - 0.003),
    ]
    cases.append(
        dict(
            case="IN30-N05",
            quantity="oscillatory_observable",
            physics="odd/even decoupled sequence, differences change sign",
            levels=osc,
            kwargs={},
        )
    )

    # N06 -- order far above the admissible band
    m = _power_law(1.0, 0.02, 8.0)
    cases.append(
        dict(
            case="IN30-N06",
            quantity="over_order_observable",
            physics="spurious order p~8 (grid family too coarse to be asymptotic)",
            levels=_levels(m, (0.04, 0.02, 0.01)),
            kwargs={},
        )
    )

    # N07 -- ASME V&V 20 validation, comparison error inside U_val
    m = _power_law(1.0, 1.6, 2.0)
    cases.append(
        dict(
            case="IN30-N07",
            quantity="drag_coefficient",
            physics="validated against a reference inside U_val (|E| <= U_val)",
            levels=_levels(m, (0.05, 0.025, 0.0125)),
            kwargs=dict(f_exp=1.00030, u_input=0.00040),
        )
    )

    # N08 -- ASME V&V 20 validation, comparison error outside U_val (honest FAIL)
    m = _power_law(1.0, 1.6, 2.0)
    cases.append(
        dict(
            case="IN30-N08",
            quantity="drag_coefficient",
            physics="reference outside U_val (|E| > U_val): the model is not validated",
            levels=_levels(m, (0.05, 0.025, 0.0125)),
            kwargs=dict(f_exp=1.00120, u_input=0.00040),
        )
    )
    return cases


def _round(value, digits=12):
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return round(value, digits)
    return value


def run_case(case: dict) -> dict:
    verdict = evaluate(case["levels"], quantity=case["quantity"], **case["kwargs"])
    study = verdict.study
    record = {
        "case": case["case"],
        "quantity": case["quantity"],
        "physics": case["physics"],
        "status": verdict.status,
        "n_levels": None if study is None else study.n_levels,
        "h": [lvl.h for lvl in ([] if study is None else study.levels)],
        "f": [_round(lvl.f) for lvl in ([] if study is None else study.levels)],
        "order_p": _round(None if study is None else study.order_p),
        "order_source": None if study is None else study.order_source,
        "convergence_type": None if study is None else study.convergence_type,
        "convergence_ratio": _round(None if study is None else study.convergence_ratio),
        "f_fine": _round(None if study is None else study.f_fine),
        "f_extrapolated": _round(None if study is None else study.f_extrapolated),
        "gci_fine": _round(None if study is None else study.gci_fine),
        "gci_is_relative": None if study is None else study.gci_is_relative,
        "noise_certified": None if study is None else study.noise_certified,
        "validation": None
        if verdict.validation is None
        else {
            "error": _round(verdict.validation.error),
            "u_num": _round(verdict.validation.u_num),
            "u_val": _round(verdict.validation.u_val),
            "ratio": _round(verdict.validation.ratio),
            "passed": verdict.validation.passed,
        },
        "checks": [{"check": c["check"], "status": c["status"]} for c in verdict.checks],
        "reasons": list(verdict.reasons),
    }
    return record


def _canonical(records: list[dict]) -> str:
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]), text=True
        ).strip()
    except Exception:  # pragma: no cover - git optional
        return "unknown"


def build_document(git_head: str | None = None) -> dict:
    records = [run_case(case) for case in build_cases()]
    digest = hashlib.sha256(_canonical(records).encode()).hexdigest()
    return {
        "schema": SCHEMA,
        "campaign": "plan48 IN30 family",
        "gate_commit": _git_head() if git_head is None else git_head,
        "python": platform.python_version(),
        "policy": GatePolicy().as_dict(),
        "cases": records,
        "digest": digest,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the golden document here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    doc = build_document()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    if not args.quiet:
        for rec in doc["cases"]:
            val = rec["validation"]
            extra = "" if val is None else f" |E|/U_val={val['ratio']:.3f}"
            print(
                f"{rec['case']:>8}  {rec['status']:<6}  p={rec['order_p']}  "
                f"gci={rec['gci_fine']}{extra}  {rec['convergence_type']}"
            )
        print(f"digest={doc['digest']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())