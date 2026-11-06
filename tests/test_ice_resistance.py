"""Pytest: three-state ice-resistance model family vs the L2 anchor corpus.

Acceptance criteria implemented here (mission 4911 / node IN10-C1):

1. every one of the reference anchors declared in
   ``benchmarks/ice_resistance/corpus/l2_anchors.json`` passes inside its own
   tolerance (``>= 10`` required, 25 declared);
2. every negative control (out-of-domain request) is rejected or honestly
   truncated -- empirical laws are never silently extrapolated;
3. the speed-power artifact ``artifacts/vp_curves.json`` exists and every point
   satisfies the power identity ``P_D = R V / eta_D``.

The checks themselves live in ``benchmarks/ice_resistance/anchors_eval.py`` so
that the pytest suite and the benchmark runner share one implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (str(REPO / "src"), str(REPO), str(REPO / "benchmarks" / "ice_resistance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchors_eval import CHECKS, _load_corpus, evaluate_corpus  # noqa: E402
from tensorphysics.ice.resistance import (  # noqa: E402
    DomainLimits,
    IceDomainError,
    IceState,
    ShipGeometry,
    brash_ice_resistance,
    coverage_decay,
    ice_resistance,
    level_ice_resistance,
    ridge_resistance,
)

SHIP = ShipGeometry()


@pytest.fixture(scope="module")
def corpus() -> dict:
    return _load_corpus()


@pytest.fixture(scope="module")
def evaluation() -> dict:
    return evaluate_corpus()


# ---------------------------------------------------------------------------
# 1. reference anchors
# ---------------------------------------------------------------------------


def _anchor_ids() -> list:
    return [a["id"] for a in _load_corpus()["anchors"]]


@pytest.mark.parametrize("anchor_id", _anchor_ids())
def test_reference_anchor(anchor_id: str, corpus: dict) -> None:
    spec = next(a for a in corpus["anchors"] if a["id"] == anchor_id)
    res = CHECKS[spec["check"]]()
    assert res["error"] <= spec["tolerance"] + 1e-15, (
        f"{anchor_id}: {spec['statement']} failed, error={res['error']:.3e} "
        f"> tolerance={spec['tolerance']:.3e} ({res['detail']})"
    )
    assert res["passed"], f"{anchor_id} failed: {res}"


def _negative_ids() -> list:
    return [a["id"] for a in _load_corpus()["negative_controls"]]


@pytest.mark.parametrize("control_id", _negative_ids())
def test_negative_control(control_id: str, corpus: dict) -> None:
    spec = next(a for a in corpus["negative_controls"] if a["id"] == control_id)
    res = CHECKS[spec["check"]]()
    assert res["passed"], f"{control_id}: {spec['statement']} -> {res}"


def test_corpus_has_at_least_ten_reference_anchors(corpus: dict) -> None:
    assert len(corpus["anchors"]) >= 10


def test_whole_corpus_passes(evaluation: dict) -> None:
    assert evaluation["all_passed"], [
        a["id"] for a in evaluation["anchors"] if not a["passed"]
    ] + [n["id"] for n in evaluation["negative_controls"] if not n["passed"]]
    assert evaluation["n_reference_anchors_passed"] == evaluation["n_reference_anchors"]
    assert evaluation["n_negative_controls_passed"] == evaluation["n_negative_controls"]


# ---------------------------------------------------------------------------
# 2. domain / negative behaviour at the API level
# ---------------------------------------------------------------------------


def test_h_over_003L_is_refused_strict() -> None:
    """h > 0.03 L is outside the regression domain of the empirical law."""

    assert DomainLimits().h_limit(SHIP) <= 0.03 * SHIP.L_pp + 1e-12
    with pytest.raises(IceDomainError, match="validity limit"):
        level_ice_resistance(0.03 * SHIP.L_pp * 1.2, 3.0, SHIP, strict=True)


def test_h_over_003L_is_truncated_and_flagged_when_lenient() -> None:
    lim = DomainLimits().h_limit(SHIP)
    r = level_ice_resistance(5.0, 3.0, SHIP, strict=False)
    assert r["h_m"] == pytest.approx(lim)
    assert any(f.startswith("H_TRUNCATED") for f in r["flags"])
    assert not r["domain_ok"]


def test_speed_limit_refused() -> None:
    with pytest.raises(IceDomainError):
        level_ice_resistance(1.0, 12.0, SHIP, strict=True)


@pytest.mark.parametrize("bad_c", [-0.01, 1.01, float("nan")])
def test_coverage_domain(bad_c: float) -> None:
    with pytest.raises(IceDomainError):
        coverage_decay(bad_c)


def test_ridge_keel_limit_refused() -> None:
    with pytest.raises(IceDomainError):
        ridge_resistance(6.0, 2.0, SHIP, strict=True)


# ---------------------------------------------------------------------------
# 3. branch dispatch and cross-branch physics
# ---------------------------------------------------------------------------


def test_three_states_dispatch() -> None:
    states = [
        IceState(kind="level", thickness_m=0.8),
        IceState(kind="brash", thickness_m=0.8, coverage=0.5),
        IceState(kind="ridge", thickness_m=0.5, ridge_keel_m=2.5, ridge_spacing_m=200.0),
    ]
    out = [ice_resistance(s, 2.0, SHIP) for s in states]
    assert [o["state"] if "state" in o else None for o in out] == [
        "level", "brash", "ridge_field"
    ]
    assert all(o["R_total_N"] > 0.0 for o in out)


def test_unknown_state_kind_rejected() -> None:
    with pytest.raises(IceDomainError):
        ice_resistance(IceState(kind="slush"), 2.0, SHIP)


def test_brash_reduces_to_level_at_full_coverage() -> None:
    a = brash_ice_resistance(0.7, 2.5, 1.0, SHIP)["R_total_N"]
    b = level_ice_resistance(0.7, 2.5, SHIP)["R_total_N"]
    assert a == pytest.approx(b, rel=1e-12)


def test_component_breakdown_is_dominant_term_crushing_and_breaking() -> None:
    """Crushing + breaking dominate submergence for a PC6 ship in 1 m ice."""

    r = level_ice_resistance(1.0, 3.0, SHIP)
    assert r["R_submersion_N"] < 0.2 * r["R_total_N"]
    assert r["R_crush_N"] > 0.0 and r["R_break_N"] > 0.0


def test_level_ice_resistance_order_of_magnitude() -> None:
    """1 m level ice at 5 kn: resistance and power inside the design envelope."""

    V = 5.0 * 0.514444
    r = level_ice_resistance(1.0, V, SHIP)
    assert 1.0e6 < r["R_total_N"] < 5.0e6
    assert 5.0 < r["R_total_N"] * V / SHIP.eta_D / 1e6 < 20.0


# ---------------------------------------------------------------------------
# 4. artifact: speed-power matrix
# ---------------------------------------------------------------------------


def test_vp_curves_artifact() -> None:
    path = REPO / "artifacts" / "vp_curves.json"
    assert path.exists(), "run benchmarks/ice_resistance/run_ice_resistance.py first"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["artifact"] == "vp_curves"
    assert data["mission_id"] == 4911
    conds = data["conditions"]
    kinds = {c["kind"] for c in conds}
    assert kinds == {"level", "brash", "ridge"}, kinds
    assert len(conds) >= 10
    n_points = 0
    for c in conds:
        assert c["points"], c["id"]
        for p in c["points"]:
            n_points += 1
            r = ice_resistance(_state_from_condition(c), p["V_ms"], SHIP, strict=False)
            # artifact stores R to 0.01 kN and V to 1e-3 m/s
            assert p["R_total_kN"] == pytest.approx(r["R_total_N"] / 1e3, rel=1e-3)
            assert p["P_delivered_MW"] == pytest.approx(
                p["R_total_kN"] * 1e3 * p["V_ms"] / SHIP.eta_D / 1e6, rel=1e-3
            )
        # resistance must increase monotonically with speed within a condition
        rseq = [p["R_total_kN"] for p in c["points"]]
        assert all(b > a for a, b in zip(rseq, rseq[1:])), c["id"]
        assert c["speed_at_installed_power"]["V_ms"] > 0.0
    assert n_points >= 100
    assert data["acceptance"]["all_anchors_passed"] is True
    assert data["acceptance"]["n_reference_anchors"] >= 10


def _state_from_condition(cond: dict) -> IceState:
    """Rebuild the ice state from the machine-readable block in the artifact."""

    st = cond["state"]
    return IceState(
        kind=st["kind"],
        thickness_m=st["thickness_m"],
        coverage=st.get("coverage", 1.0),
        ridge_keel_m=st.get("ridge_keel_m", 0.0),
        ridge_spacing_m=st.get("ridge_spacing_m", 0.0),
        ridge_porosity=st.get("ridge_porosity", 0.30),
    )