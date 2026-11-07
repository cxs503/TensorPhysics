"""Anchor-corpus evaluator for the full-spectrum ice-resistance model.

The corpus itself lives in ``corpus/l2_anchors.json`` (declarations: id, check,
statement, source, tolerance).  This module owns the *checks* so that the pytest
suite and the benchmark runner share exactly one implementation of every
acceptance criterion.

Every check returns a dict::

    {"passed": bool, "observed": ..., "expected": ..., "detail": str}

``evaluate_corpus()`` additionally grades the negative controls (each must be
rejected by the model) and produces the audit record for the legacy in-repo
Lindqvist transcription (``ice_anchor.anchors.lindqvist_1989``), which is *not*
used as an anchor: it is the defect this benchmark documents.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (str(_REPO / "src"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.ice.resistance import (  # noqa: E402
    G,
    MU_HULL_ICE,
    SIGMA_C_REF,
    SIGMA_F_REF,
    DomainLimits,
    IceDomainError,
    IceState,
    ShipGeometry,
    brash_ice_resistance,
    coefficient_sensitivity,
    coverage_decay,
    delivered_power_MW,
    ice_resistance,
    level_ice_resistance,
    ram_capacity,
    ridge_field_mean_resistance,
    ridge_penetration_work,
    ridge_resistance,
    speed_for_power,
)

CORPUS_PATH = _HERE.parent / "corpus" / "l2_anchors.json"

SHIP = ShipGeometry()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d else 0.0


def _ok(observed: Any, expected: Any, tol: float, detail: str, cmp: str = "rel") -> Dict[str, Any]:
    if cmp == "rel":
        err = _rel(float(observed), float(expected))
    elif cmp == "abs":
        err = abs(float(observed) - float(expected))
    elif cmp == "sign":
        err = 0.0 if float(observed) > 0.0 else 1.0
    else:  # pragma: no cover - defensive
        raise ValueError(cmp)
    return {
        "passed": bool(err <= tol),
        "observed": observed,
        "expected": expected,
        "error": err,
        "detail": detail,
    }


def _load_corpus(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(Path(path or CORPUS_PATH).read_text(encoding="utf-8"))


def _reference_laws() -> Dict[str, Callable[..., Dict[str, float]]]:
    """Independent in-repo L2 transcriptions of the cited literature laws."""

    from ice_anchor.anchors import enkvist_1972, keinonen_1983, riska_1997

    return {"keinonen_1983": keinonen_1983, "enkvist_1972": enkvist_1972, "riska_1997": riska_1997}


# ---------------------------------------------------------------------------
# level ice (Lindqvist three components)
# ---------------------------------------------------------------------------


def level_superposition(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    r = level_ice_resistance(1.0, 3.0, ship)
    parts = r["R_crush_N"] + r["R_break_N"] + r["R_submersion_N"]
    return _ok(r["R_total_N"], parts, 1e-12, "R_total vs sum of components", "rel")


def level_crush_speed_independent(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    a = level_ice_resistance(1.0, 0.5, ship)["R_crush_N"]
    b = level_ice_resistance(1.0, 5.0, ship)["R_crush_N"]
    return _ok(a, b, 1e-12, "R_crush(V=0.5) vs R_crush(V=5.0)", "rel")


def level_submersion_loglog_slope(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    V, h, dV = 3.0, 1.0, 1e-5
    r0 = level_ice_resistance(h, V - dV, ship)["R_submersion_N"]
    r1 = level_ice_resistance(h, V + dV, ship)["R_submersion_N"]
    numeric = (math.log(r1) - math.log(r0)) / (math.log(V + dV) - math.log(V - dV))
    # R_s ∝ a V^2 + b V^3 with a = 0.15, b = 0.3/sqrt(g h)  =>  slope = (2 + 3x)/(1 + x),
    # x = b V / a = 2 Fr_h
    x = 2.0 * V / math.sqrt(G * h)
    analytic = (2.0 + 3.0 * x) / (1.0 + x)
    return _ok(numeric, analytic, 1e-6, "d ln R_s / d ln V: finite difference vs analytic", "rel")


def level_crush_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    base = level_ice_resistance(1.0, 3.0, ship)["R_crush_N"]
    e_sigma = level_ice_resistance(1.0, 3.0, ship, sigma_c=2 * SIGMA_C_REF)["R_crush_N"] / base
    e_h = level_ice_resistance(2.0, 3.0, ship)["R_crush_N"] / base
    res_sigma = _ok(e_sigma, 2.0, 1e-12, "exponent in sigma_c", "rel")
    res_h = _ok(e_h, 4.0, 1e-12, "exponent in h (x2 -> x4)", "rel")
    return {
        "passed": res_sigma["passed"] and res_h["passed"],
        "observed": {"sigma_c_x2_ratio": e_sigma, "h_x2_ratio": e_h},
        "expected": {"sigma_c_x2_ratio": 2.0, "h_x2_ratio": 4.0},
        "error": max(res_sigma["error"], res_h["error"]),
        "detail": "R_crush ∝ sigma_c^1 h^2",
    }


def level_break_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    base = level_ice_resistance(1.0, 3.0, ship)["R_break_N"]
    e_h = level_ice_resistance(2.0, 3.0, ship)["R_break_N"] / base
    e_B = level_ice_resistance(1.0, 3.0, ShipGeometry(**{**_ship_kwargs(ship), "B": 2 * ship.B}))[
        "R_break_N"
    ] / base
    e_sf = level_ice_resistance(1.0, 3.0, ship, sigma_f=2 * SIGMA_F_REF)["R_break_N"] / base
    errs = [abs(x - 2.0) / 2.0 for x in (e_h, e_B, e_sf)]
    return {
        "passed": max(errs) <= 1e-12,
        "observed": {"h_x2_ratio": e_h, "B_x2_ratio": e_B, "sigma_f_x2_ratio": e_sf},
        "expected": {"h_x2_ratio": 2.0, "B_x2_ratio": 2.0, "sigma_f_x2_ratio": 2.0},
        "error": max(errs),
        "detail": "R_break ∝ sigma_f^1 h^1 B^1",
    }


def _ship_kwargs(ship: ShipGeometry) -> Dict[str, Any]:
    return {
        "name": ship.name,
        "L_pp": ship.L_pp,
        "B": ship.B,
        "T": ship.T,
        "bow_angle_deg": ship.bow_angle_deg,
        "flare_deg": ship.flare_deg,
        "displacement_t": ship.displacement_t,
        "installed_power_MW": ship.installed_power_MW,
        "eta_D": ship.eta_D,
    }


def level_monotonicity(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    base = level_ice_resistance(1.0, 3.0, ship)["R_total_N"]
    probes = {
        "dR_dh": level_ice_resistance(1.05, 3.0, ship)["R_total_N"] - base,
        "dR_dV": level_ice_resistance(1.0, 3.05, ship)["R_total_N"] - base,
        "dR_dB": level_ice_resistance(
            1.0, 3.0, ShipGeometry(**{**_ship_kwargs(ship), "B": 1.05 * ship.B})
        )["R_total_N"]
        - base,
        "dR_dphi": level_ice_resistance(
            1.0, 3.0, ShipGeometry(**{**_ship_kwargs(ship), "bow_angle_deg": ship.bow_angle_deg + 2.0})
        )["R_total_N"]
        - base,
        "dR_dalpha": level_ice_resistance(
            1.0, 3.0, ShipGeometry(**{**_ship_kwargs(ship), "flare_deg": ship.flare_deg + 2.0})
        )["R_total_N"]
        - base,
        "dR_dmu": level_ice_resistance(1.0, 3.0, ship, mu=MU_HULL_ICE + 0.01)["R_total_N"] - base,
    }
    negative = [k for k, v in probes.items() if v <= 0.0]
    return {
        "passed": not negative,
        "observed": {k: round(v, 3) for k, v in probes.items()},
        "expected": "all derivatives > 0",
        "error": 0.0 if not negative else 1.0,
        "detail": f"violations: {negative}" if negative else "all six derivatives positive",
    }


def level_cross_law(ship: ShipGeometry = SHIP, tol: float = 0.25) -> Dict[str, Any]:
    laws = _reference_laws()
    rows = []
    for h in (0.5, 0.75, 1.0):
        mine = level_ice_resistance(h, 3.0, ship)["R_total_kN"]
        kei = laws["keinonen_1983"](h, ship.B, 3.0, ship.L_pp, ship.bow_angle_deg)["R_total_kN"]
        env = laws["enkvist_1972"](h, ship.B, 3.0, ship.L_pp, ship.bow_angle_deg)["R_total_kN"]
        ref = 0.5 * (kei + env)
        rows.append({"h_m": h, "model_kN": mine, "keinonen_kN": kei, "enkvist_kN": env,
                     "ref_kN": ref, "rel_dev": (mine - ref) / ref})
    worst = max(abs(r["rel_dev"]) for r in rows)
    return {
        "passed": worst <= tol,
        "observed": rows,
        "expected": f"|dev| <= {tol:.0%} of mean(Keinonen, Enkvist)",
        "error": worst,
        "detail": "cross-law consistency (transcription check, not accuracy validation)",
    }


def level_power_plausibility(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    V = 5.0 * 0.514444
    r = level_ice_resistance(1.0, V, ship)["R_total_N"]
    p = delivered_power_MW(r, V, ship.eta_D)
    lo, hi = 5.0, 20.0
    return {
        "passed": lo <= p <= hi,
        "observed": round(p, 3),
        "expected": [lo, hi],
        "error": 0.0 if lo <= p <= hi else min(abs(p - lo), abs(p - hi)) / hi,
        "detail": "delivered power for 1.0 m level ice at 5 kn (MW)",
    }


def level_domain_accept_inside(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    lim = DomainLimits()
    h = 0.999 * lim.h_limit(ship)
    r = level_ice_resistance(h, 3.0, ship, strict=True)
    return {
        "passed": r["domain_ok"] and not r["flags"],
        "observed": {"h_m": h, "flags": r["flags"], "domain_ok": r["domain_ok"]},
        "expected": "accepted, no truncation flags",
        "error": 0.0 if r["domain_ok"] else 1.0,
        "detail": f"h_limit={lim.h_limit(ship):.3f} m (min of 0.03L and 0.10B)",
    }


# ---------------------------------------------------------------------------
# brash ice (coverage decay law)
# ---------------------------------------------------------------------------


def brash_endpoint_full(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    b = brash_ice_resistance(0.6, 2.0, 1.0, ship)["R_total_N"]
    l = level_ice_resistance(0.6, 2.0, ship)["R_total_N"]
    return _ok(b, l, 1e-12, "R_brash(c=1) vs R_level", "rel")


def brash_endpoint_zero(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    b = brash_ice_resistance(0.6, 2.0, 0.0, ship)["R_total_N"]
    return _ok(b, 0.0, 1e-9, "R_brash(c=0) vs 0 N", "abs")


def brash_clearing_linearity() -> Dict[str, Any]:
    worst = 0.0
    rows = []
    for c in (0.1, 0.3, 0.6):
        d = coverage_decay(c)["D"]
        rows.append({"c": c, "D": d})
        worst = max(worst, abs(d - c))
    return {
        "passed": worst <= 1e-12,
        "observed": rows,
        "expected": "D(c) == c for c <= 0.6",
        "error": worst,
        "detail": "clearing-regime linearity",
    }


def brash_monotone_in_coverage() -> Dict[str, Any]:
    prev, bad = -1.0, []
    for i in range(101):
        c = i / 100.0
        d = coverage_decay(c)["D"]
        if d < prev - 1e-15:
            bad.append((c, prev, d))
        prev = d
    return {
        "passed": not bad,
        "observed": {"violations": bad},
        "expected": "non-decreasing D(c) on [0,1]",
        "error": 0.0 if not bad else 1.0,
        "detail": "101-point monotonicity sweep",
    }


def brash_bounded_by_level(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    lvl = level_ice_resistance(0.6, 2.0, ship)["R_total_N"]
    bad = []
    for i in range(0, 100):
        c = i / 100.0
        r = brash_ice_resistance(0.6, 2.0, c, ship)["R_total_N"]
        if r > lvl * (1.0 + 1e-12):
            bad.append((c, r, lvl))
    return {
        "passed": not bad,
        "observed": {"violations": bad[:5], "n_violations": len(bad)},
        "expected": "R_brash(c<1) < R_level",
        "error": 0.0 if not bad else 1.0,
        "detail": "c=0.00..0.99 sweep at h=0.6 m, V=2 m/s",
    }


def brash_packing_amplification() -> Dict[str, Any]:
    d6, d9 = coverage_decay(0.6)["D"], coverage_decay(0.9)["D"]
    slope = (d9 - d6) / 0.3
    return _ok(slope, 1.05, 0.0, "local slope above c_crit must exceed 1", "sign")


# ---------------------------------------------------------------------------
# ridge (Keinonen-anchored)
# ---------------------------------------------------------------------------


def ridge_keel_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    a = ridge_resistance(1.5, 2.0, ship)["R_keel_N"]
    b = ridge_resistance(3.0, 2.0, ship)["R_keel_N"]
    m_a = ridge_resistance(1.5, 2.0, ship)["R_momentum_N"]
    m_b = ridge_resistance(3.0, 2.0, ship)["R_momentum_N"]
    res_keel = _ok(b / a, 4.0, 1e-12, "R_keel(h_k x2)/R_keel(h_k)", "rel")
    res_mom = _ok(m_b / m_a, 2.0, 1e-12, "R_momentum(h_k x2)/R_momentum(h_k)", "rel")
    return {
        "passed": res_keel["passed"] and res_mom["passed"],
        "observed": {"keel_x2_ratio": b / a, "momentum_x2_ratio": m_b / m_a},
        "expected": {"keel_x2_ratio": 4.0, "momentum_x2_ratio": 2.0},
        "error": max(res_keel["error"], res_mom["error"]),
        "detail": "buoyancy/displacement term ∝ h_k^2, momentum term ∝ h_k (mass flow)",
    }


def ridge_porosity_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    a = ridge_resistance(2.5, 2.0, ship, porosity=0.2)["R_total_N"]
    b = ridge_resistance(2.5, 2.0, ship, porosity=0.4)["R_total_N"]
    return _ok(b / a, (1 - 0.4) / (1 - 0.2), 1e-12, "R(e=0.4)/R(e=0.2)", "rel")


def ridge_breadth_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    wide = ShipGeometry(**{**_ship_kwargs(ship), "B": 2 * ship.B})
    a = ridge_resistance(2.5, 2.0, ship)["R_total_N"]
    b = ridge_resistance(2.5, 2.0, wide)["R_total_N"]
    return _ok(b / a, 2.0, 1e-12, "R(2B)/R(B)", "rel")


def ridge_momentum_speed_scaling(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    a = ridge_resistance(2.5, 2.0, ship)["R_momentum_N"]
    b = ridge_resistance(2.5, 4.0, ship)["R_momentum_N"]
    return _ok(b / a, 4.0, 1e-12, "R_moment(V x2)/R_moment(V)", "rel")


def ridge_work_consistency(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    w = ridge_penetration_work(2.5, 2.0, ship, n_steps=2000)
    return _ok(w["W_quadrature_J"], w["W_analytic_J"], 1e-4, "quadrature vs closed form", "rel")


def ridge_field_mean_identity(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    s = 200.0
    r = ridge_field_mean_resistance(2.5, 2.0, ship, s, background_h=0.5)
    frac = r["ridge_width_m"] / s
    expected = r["R_level_N"] * (1.0 - frac) + r["R_ridge_instantaneous_N"] * frac
    return _ok(
        r["R_total_N"], expected, 1e-12,
        "R_mean vs R_level(1-w/s) + R_ridge(w/s)", "rel",
    )


def ridge_field_spacing_monotone(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    # thin background ice, deep keel: the ridge term dominates the field average
    vals = [
        ridge_field_mean_resistance(3.5, 2.0, ship, s, background_h=0.2)["R_total_N"]
        for s in (100.0, 200.0, 400.0)
    ]
    ok = vals[0] > vals[1] > vals[2]
    return {
        "passed": ok,
        "observed": vals,
        "expected": "R(100) > R(200) > R(400)",
        "error": 0.0 if ok else 1.0,
        "detail": "field-averaged resistance vs ridge spacing (h_k=3.5 m, h_bg=0.2 m)",
    }


# ---------------------------------------------------------------------------
# speed-power chain
# ---------------------------------------------------------------------------


def power_identity(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    r = level_ice_resistance(1.0, 3.0, ship)["R_total_N"]
    p = delivered_power_MW(r, 3.0, ship.eta_D)
    r_back = p * 1e6 * ship.eta_D / 3.0
    return _ok(r_back, r, 1e-12, "R -> P -> R round trip", "rel")


def power_inverse_consistency(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    state = IceState(kind="level", thickness_m=1.0)
    res = speed_for_power(state, ship, ship.installed_power_MW)
    err = abs(res["power_residual_MW"]) / ship.installed_power_MW
    return {
        "passed": (err <= 1e-8) or res["capped"],
        "observed": {"V_star_ms": res["V_ms"], "residual_MW": res["power_residual_MW"],
                     "capped": res["capped"]},
        "expected": "relative residual <= 1e-8 (or power-limited cap reported)",
        "error": err,
        "detail": "bisection inverse of the power chain at 22 MW",
    }


def power_monotone_in_speed(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    state = IceState(kind="level", thickness_m=1.0)
    prev, bad = -1.0, []
    for i in range(10, 51):
        v = i * 0.1
        r = ice_resistance(state, v, ship)["R_total_N"]
        p = delivered_power_MW(r, v, ship.eta_D)
        if p <= prev:
            bad.append((v, prev, p))
        prev = p
    return {
        "passed": not bad,
        "observed": {"violations": bad},
        "expected": "P_D strictly increasing",
        "error": 0.0 if not bad else 1.0,
        "detail": "0.5-5.0 m/s sweep at h=1.0 m",
    }


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------


def _expect_error(fn: Callable[[], Any]) -> Dict[str, Any]:
    try:
        fn()
    except IceDomainError as exc:
        return {"passed": True, "observed": str(exc), "expected": "IceDomainError", "error": 0.0,
                "detail": "correctly rejected"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"passed": False, "observed": f"{type(exc).__name__}: {exc}",
                "expected": "IceDomainError", "error": 1.0, "detail": "wrong exception type"}
    return {"passed": False, "observed": "no exception", "expected": "IceDomainError",
            "error": 1.0, "detail": "domain violation silently accepted"}


def neg_h_over_03L_strict(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    return _expect_error(lambda: level_ice_resistance(5.0, 3.0, ship, strict=True))


def neg_froude_strict(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    return _expect_error(lambda: level_ice_resistance(1.0, 12.0, ship, strict=True))


def neg_coverage_out_of_range() -> Dict[str, Any]:
    return _expect_error(lambda: coverage_decay(1.35))


def neg_ridge_keel_strict(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    return _expect_error(lambda: ridge_resistance(6.0, 2.0, ship, strict=True))


def neg_negative_thickness(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    return _expect_error(lambda: level_ice_resistance(-0.1, 3.0, ship))


def neg_porosity_range(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    return _expect_error(lambda: ridge_resistance(2.0, 2.0, ship, porosity=0.9))


def neg_lenient_truncation_flags(ship: ShipGeometry = SHIP) -> Dict[str, Any]:
    lim = DomainLimits()
    r = level_ice_resistance(5.0, 3.0, ship, strict=False)
    h_lim = lim.h_limit(ship)
    flagged = any(f.startswith("H_TRUNCATED") for f in r["flags"])
    consistent = abs(r["h_m"] - h_lim) <= 1e-12
    ref = level_ice_resistance(h_lim, 3.0, ship)["R_total_N"]
    return {
        "passed": flagged and consistent and abs(r["R_total_N"] - ref) <= 1e-9 * ref,
        "observed": {"h_used_m": r["h_m"], "flags": r["flags"], "R_kN": r["R_total_kN"]},
        "expected": f"h clipped to {h_lim:.3f} m, H_TRUNCATED flag, R evaluated at h_limit",
        "error": 0.0 if (flagged and consistent) else 1.0,
        "detail": "lenient mode truncates honestly instead of extrapolating",
    }


CHECKS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "level_superposition": level_superposition,
    "level_crush_speed_independent": level_crush_speed_independent,
    "level_submersion_loglog_slope": level_submersion_loglog_slope,
    "level_crush_scaling": level_crush_scaling,
    "level_break_scaling": level_break_scaling,
    "level_monotonicity": level_monotonicity,
    "level_cross_law": level_cross_law,
    "level_power_plausibility": level_power_plausibility,
    "level_domain_accept_inside": level_domain_accept_inside,
    "brash_endpoint_full": brash_endpoint_full,
    "brash_endpoint_zero": brash_endpoint_zero,
    "brash_clearing_linearity": brash_clearing_linearity,
    "brash_monotone_in_coverage": brash_monotone_in_coverage,
    "brash_bounded_by_level": brash_bounded_by_level,
    "brash_packing_amplification": brash_packing_amplification,
    "ridge_keel_scaling": ridge_keel_scaling,
    "ridge_porosity_scaling": ridge_porosity_scaling,
    "ridge_breadth_scaling": ridge_breadth_scaling,
    "ridge_momentum_speed_scaling": ridge_momentum_speed_scaling,
    "ridge_work_consistency": ridge_work_consistency,
    "ridge_field_mean_identity": ridge_field_mean_identity,
    "ridge_field_spacing_monotone": ridge_field_spacing_monotone,
    "power_identity": power_identity,
    "power_inverse_consistency": power_inverse_consistency,
    "power_monotone_in_speed": power_monotone_in_speed,
    "neg_h_over_03L_strict": neg_h_over_03L_strict,
    "neg_froude_strict": neg_froude_strict,
    "neg_coverage_out_of_range": neg_coverage_out_of_range,
    "neg_ridge_keel_strict": neg_ridge_keel_strict,
    "neg_negative_thickness": neg_negative_thickness,
    "neg_porosity_range": neg_porosity_range,
    "neg_lenient_truncation_flags": neg_lenient_truncation_flags,
}


# ---------------------------------------------------------------------------
# audit of the pre-existing in-repo transcriptions
# ---------------------------------------------------------------------------


def audit_legacy_transcriptions(ship: ShipGeometry = SHIP, h: float = 1.0, V: float = 3.0) -> Dict[str, Any]:
    """Cross-check ``ice_anchor.anchors`` (m598) against this module.

    Not an anchor: this is the defect record.  ``lindqvist_1989`` builds the
    crushing angle as ``alpha = 90 deg - phi/2`` (75 deg for a 30 deg stem
    angle), i.e. it uses the stem angle in the horizontal plane as if it were
    the plating inclination, inflating ``tan(alpha)`` from 0.47 to 3.73.  The
    same convention is used by ``riska_1997``.
    """

    from ice_anchor.anchors import enkvist_1972, keinonen_1983, lindqvist_1989, riska_1997

    mine = level_ice_resistance(h, V, ship)
    ref_kei = keinonen_1983(h, ship.B, V, ship.L_pp, ship.bow_angle_deg)["R_total_kN"]
    ref_env = enkvist_1972(h, ship.B, V, ship.L_pp, ship.bow_angle_deg)["R_total_kN"]
    legacy_lq = lindqvist_1989(h, ship.B, ship.bow_angle_deg, V, ship.L_pp)
    legacy_ri = riska_1997(h, ship.B, V, ship.L_pp, ship.bow_angle_deg)
    ref_mean = 0.5 * (ref_kei + ref_env)
    return {
        "h_m": h,
        "V_ms": V,
        "keinonen_kN": ref_kei,
        "enkvist_kN": ref_env,
        "reference_mean_kN": ref_mean,
        "benchmark_model_kN": mine["R_total_kN"],
        "benchmark_rel_dev": (mine["R_total_kN"] - ref_mean) / ref_mean,
        "legacy_lindqvist_kN": legacy_lq["R_total_kN"],
        "legacy_lindqvist_rel_dev": (legacy_lq["R_total_kN"] - ref_mean) / ref_mean,
        "legacy_lindqvist_angle_deg": math.degrees(math.pi / 2 - ship.bow_angle_rad / 2),
        "legacy_lindqvist_crush_kN": legacy_lq["R_c_N"] / 1e3,
        "legacy_riska_kN": legacy_ri["R_total_kN"],
        "legacy_riska_rel_dev": (legacy_ri["R_total_kN"] - ref_mean) / ref_mean,
        "finding": (
            "ice_anchor.anchors.lindqvist_1989 / riska_1997 use the horizontal-plane stem "
            "angle as the plating-inclination angle (alpha = 90 deg - phi/2), inflating "
            "tan(alpha) by 6-8x; the crushing term then dominates (10.8 MN, +276% vs the "
            "Keinonen/Enkvist mean). This benchmark uses the flare angle (25 deg) instead."
        ),
    }


# ---------------------------------------------------------------------------
# corpus evaluation
# ---------------------------------------------------------------------------


def evaluate_corpus(
    path: Optional[Path] = None, ship: ShipGeometry = SHIP
) -> Dict[str, Any]:
    corpus = _load_corpus(path)
    anchors: List[Dict[str, Any]] = []
    for spec in corpus["anchors"]:
        fn = CHECKS[spec["check"]]
        try:
            res = fn()
        except TypeError:  # checks without ship parameter
            res = fn()
        row = {
            "id": spec["id"],
            "group": spec["group"],
            "kind": spec["kind"],
            "statement": spec["statement"],
            "source": spec["source"],
            "tolerance": spec["tolerance"],
            "tolerance_type": spec["tolerance_type"],
        }
        row.update(res)
        anchors.append(row)

    negatives: List[Dict[str, Any]] = []
    for spec in corpus["negative_controls"]:
        fn = CHECKS[spec["check"]]
        res = fn()
        row = {"id": spec["id"], "statement": spec["statement"], "expected": spec["expected"]}
        row.update(res)
        negatives.append(row)

    n_pass = sum(1 for a in anchors if a["passed"])
    n_neg_pass = sum(1 for n in negatives if n["passed"])
    return {
        "corpus_version": corpus["corpus_version"],
        "evidence_level": corpus["evidence_level"],
        "n_reference_anchors": len(anchors),
        "n_reference_anchors_passed": n_pass,
        "n_negative_controls": len(negatives),
        "n_negative_controls_passed": n_neg_pass,
        "all_passed": n_pass == len(anchors) and n_neg_pass == len(negatives),
        "anchors": anchors,
        "negative_controls": negatives,
        "audit": audit_legacy_transcriptions(ship),
        "sensitivity": coefficient_sensitivity(1.0, 3.0, ship),
    }


if __name__ == "__main__":  # pragma: no cover - manual invocation
    rep = evaluate_corpus()
    print(json.dumps({k: v for k, v in rep.items() if k not in ("anchors", "negative_controls")},
                     ensure_ascii=False, indent=2))
    for a in rep["anchors"]:
        print(f"{a['id']:<8} {'PASS' if a['passed'] else 'FAIL':<4} {a['error']:.3e}  {a['detail'][:60]}")
    for n in rep["negative_controls"]:
        print(f"{n['id']:<8} {'PASS' if n['passed'] else 'FAIL':<4} {n['detail'][:60]}")
    print("ALL PASSED:", rep["all_passed"])