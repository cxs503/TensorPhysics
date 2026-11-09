"""IN30-N01..N08 regression lock for the V&V machine gate (plan 48).

The golden document ``tests/data/vv_in30/golden_head_5de4f11.json`` was captured
**before** this node's changes (gate commit ``5de4f11``) by
``tools/vv_in30_regression.py``.  The test re-runs the same eight analytic cases
against the current gate and requires

1. the recorded case records to be *identical* -- verdict, observed order, GCI,
   convergence type, ASME comparison, per-check statuses and reasons -- so the
   mature features (``R = delta/dx`` resolvability, ``U_ref`` ledger) are proven
   to be additive and not to move a historical decision;
2. the sha256 ``digest`` of the canonical case list to match, which catches a
   silent edit of either the golden file or the gate;
3. the eight historical verdicts to still fall where plan 48 put them
   (WARN/PASS/PASS/FAIL/FAIL/FAIL/WARN/FAIL), i.e. the honest FAILs are not
   overwritten.

Regenerate the golden file only when a *deliberate* behaviour change is made::

    python tools/vv_in30_regression.py --out tests/data/vv_in30/golden_<commit>.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - pytest conftest does this
    sys.path.insert(0, str(_SRC))

from tools.vv_in30_regression import build_cases, build_document, run_case  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "data" / "vv_in30" / "golden_head_5de4f11.json"

_HISTORICAL_STATUS = {
    "IN30-N01": "WARN",
    "IN30-N02": "PASS",
    "IN30-N03": "PASS",
    "IN30-N04": "FAIL",
    "IN30-N05": "FAIL",
    "IN30-N06": "FAIL",
    "IN30-N07": "WARN",
    "IN30-N08": "FAIL",
}


@pytest.fixture(scope="module")
def golden() -> dict:
    assert _GOLDEN.is_file(), f"golden corpus missing: {_GOLDEN}"
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return build_document()


def test_golden_corpus_covers_the_eight_plan48_cases(golden):
    cases = [rec["case"] for rec in golden["cases"]]
    assert cases == [f"IN30-N{n:02d}" for n in range(1, 9)]
    assert golden["schema"] == "tensorphysics.vv.in30-regression/1"
    assert golden["gate_commit"] == "5de4f11"


def test_case_records_are_bit_identical_to_the_historical_golden(golden, current):
    assert len(current["cases"]) == len(golden["cases"])
    for old, new in zip(golden["cases"], current["cases"]):
        assert new == old, f"{old['case']} drifted: {new} != {old}"


def test_digest_matches(golden, current):
    assert current["digest"] == golden["digest"]


def test_historical_verdicts_are_not_overwritten(golden):
    got = {rec["case"]: rec["status"] for rec in golden["cases"]}
    assert got == _HISTORICAL_STATUS


def test_honest_failures_survive(golden):
    by_case = {rec["case"]: rec for rec in golden["cases"]}
    # N04 stagnant / N05 oscillatory: no order of accuracy exists
    assert by_case["IN30-N04"]["order_p"] is None
    assert by_case["IN30-N04"]["convergence_type"] == "stagnant"
    assert by_case["IN30-N05"]["convergence_type"] == "oscillatory"
    # N06: order 8.1997 outside the (0.05, 6] band
    assert by_case["IN30-N06"]["order_p"] > 6.0
    # N08: |E|/U_val = 1.872 -> FAIL, never softened
    assert by_case["IN30-N08"]["validation"]["ratio"] == pytest.approx(1.872, abs=5e-3)
    assert by_case["IN30-N08"]["validation"]["passed"] is False
    assert by_case["IN30-N08"]["status"] == "FAIL"


def test_second_order_gci_is_recovered(golden):
    """Independent closed-form check of the recorded GCI (Roache, Fs=1.25, rho=2)."""
    rec = next(r for r in golden["cases"] if r["case"] == "IN30-N02")
    f_coarse, f_fine = rec["f"][-2], rec["f"][-1]
    p = 2.0
    gci_expected = 1.25 * abs(f_fine - f_coarse) / abs(f_fine) / (2.0**p - 1.0)
    assert rec["gci_fine"] == pytest.approx(gci_expected, rel=1e-9)
    assert rec["order_p"] == pytest.approx(2.0, abs=1e-9)


def test_analytic_order_is_recovered_for_every_converging_case(golden):
    expected = {"IN30-N01": 2.0, "IN30-N02": 2.0, "IN30-N03": 1.5, "IN30-N07": 2.0, "IN30-N08": 2.0}
    for rec in golden["cases"]:
        if rec["case"] in expected:
            assert rec["order_p"] == pytest.approx(expected[rec["case"]], abs=1e-6)


def test_regeneration_is_deterministic():
    first = build_document()
    second = build_document()
    assert first["cases"] == second["cases"]
    assert first["digest"] == second["digest"]


def test_new_features_do_not_change_the_corpus_output():
    """The resolvability and ledger stages must be no-ops on the legacy corpus."""
    for case in build_cases():
        reference = run_case(case)
        from tensorphysics.vv import evaluate

        with_res = evaluate(
            case["levels"],
            quantity=case["quantity"],
            require_resolvability=False,
            **case["kwargs"],
        )
        assert with_res.status == reference["status"]
        assert not any(c["check"].startswith("resolvability") for c in with_res.checks)