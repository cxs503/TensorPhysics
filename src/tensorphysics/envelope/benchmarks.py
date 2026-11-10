"""Synthetic cases with closed-form answers -- the engine graded against truth.

Nothing here is a physics statement: the cases are algebra on axes called
``alpha`` / ``beta`` whose window, boundary location, component count and area
are known in closed form.  They exist so that the engine is validated against an
answer it cannot have produced itself, and so that the contract checker's
*negative* controls (a boundary that bends back, a window that is mis-declared)
can be exercised deliberately.  Both the test suite and the acceptance script
import this module, so a synthetic case can never drift between the two.
"""

from __future__ import annotations

from typing import Any, Mapping

from .case import WindowCase, case_from_dict
from .engine import AnalyticReference

__all__ = [
    "SYNTHETIC_CASES",
    "analytic_band",
    "analytic_disc",
    "band_case",
    "disc_case",
    "empty_case",
    "equality_band_case",
    "folded_case",
    "lattice_case",
    "offset_bands_case",
    "parabolic_band_case",
    "strip_case",
    "table_case",
    "two_disc_case",
]

SCHEMA = "tensorphysics.envelope.case.v1"


def _case(payload: Mapping[str, Any]) -> WindowCase:
    payload = dict(payload)
    payload.setdefault("schema", SCHEMA)
    return case_from_dict(payload, source="<synthetic>")


def _affine_low(axis: str, offset: float) -> dict[str, Any]:
    return {
        "op": "sub",
        "a": {"op": "var", "name": axis},
        "b": {"op": "const", "value": offset},
    }


def _affine_high(axis: str, limit: float) -> dict[str, Any]:
    return {
        "op": "sub",
        "a": {"op": "const", "value": limit},
        "b": {"op": "var", "name": axis},
    }


def _square_of_offset(axis: str, centre: float) -> dict[str, Any]:
    return {
        "op": "square",
        "arg": {
            "op": "sub",
            "a": {"op": "var", "name": axis},
            "b": {"op": "const", "value": centre},
        },
    }


def strip_case() -> WindowCase:
    """One interval, both boundaries interior, affine on both sides."""

    return _case(
        {
            "case_id": "synth_strip",
            "title": "affine strip: 0.2 <= alpha <= 0.9",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_low",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _affine_low("alpha", 0.2),
                    "contract": {"slope": {"alpha": 1}, "convex": True},
                },
                {
                    "id": "c_high",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _affine_high("alpha", 0.9),
                    "contract": {"slope": {"alpha": -1}, "convex": True},
                },
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def folded_case() -> WindowCase:
    """Negative control: a band whose margin is declared monotone but is not."""

    return _case(
        {
            "case_id": "synth_folded",
            "title": "band margin declared increasing although it bends back",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_band",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "sub",
                        "a": {"op": "const", "value": 0.01},
                        "b": _square_of_offset("alpha", 0.5),
                    },
                    "contract": {"slope": {"alpha": 1}, "convex": True},
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def offset_bands_case() -> WindowCase:
    """Negative control: two separate feasible intervals declared as one."""

    return _case(
        {
            "case_id": "synth_two_bands",
            "title": "feasible set is the outside of a band: two intervals",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_outer",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "sub",
                        "a": _square_of_offset("alpha", 0.5),
                        "b": {"op": "const", "value": 0.04},
                    },
                    "contract": {"convex": True},
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def band_case() -> WindowCase:
    """A concave quadratic margin: one band, convex feasible set."""

    return _case(
        {
            "case_id": "synth_band",
            "title": "concave margin 0.06 - (alpha - 0.4)^2 >= 0",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_band",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "sub",
                        "a": {"op": "const", "value": 0.06},
                        "b": _square_of_offset("alpha", 0.4),
                    },
                    "contract": {"convex": True},
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def equality_band_case() -> WindowCase:
    """An equality constraint plus a tolerance: a band of half-width 0.05."""

    return _case(
        {
            "case_id": "synth_equality",
            "title": "|alpha - 0.4| <= 0.05",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_eq",
                    "kind": "equality",
                    "target": 0.4,
                    "tolerance": 0.05,
                    "value": {"op": "var", "name": "alpha"},
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def table_case() -> WindowCase:
    """Piecewise-linear margin injected as a lookup table."""

    return _case(
        {
            "case_id": "synth_table",
            "title": "lookup-table margin with two interior crossings",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_table",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "table",
                        "arg": {"op": "var", "name": "alpha"},
                        "x": [0.0, 0.25, 0.5, 0.75, 1.0],
                        "y": [-1.0, 0.5, 1.0, 0.5, -1.0],
                    },
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def lattice_case() -> WindowCase:
    """Continuous strip plus a discrete setting lattice."""

    return _case(
        {
            "case_id": "synth_lattice",
            "title": "0.2 <= alpha <= 0.9 with settings on a 0.1 lattice",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_low",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _affine_low("alpha", 0.2),
                    "contract": {"slope": {"alpha": 1}},
                },
                {
                    "id": "c_high",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _affine_high("alpha", 0.9),
                    "contract": {"slope": {"alpha": -1}},
                },
                {
                    "id": "c_setting",
                    "kind": "discrete",
                    "value": {"op": "var", "name": "alpha"},
                    "lattice": {"step": 0.1, "offset": 0.0, "tolerance": 1e-9},
                },
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )


def empty_case() -> WindowCase:
    """Negative control: no feasible point at all."""

    return _case(
        {
            "case_id": "synth_empty",
            "title": "margin always negative: the window is empty",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_never",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "neg",
                        "arg": {
                            "op": "sum",
                            "terms": [
                                {"op": "const", "value": 1.0},
                                _square_of_offset("alpha", 0.5),
                            ],
                        },
                    },
                }
            ],
            "contract": {"expect": {"intervals": 0, "infeasible": True}},
        }
    )


def disc_value(centre: tuple[float, float], radius: float) -> dict[str, Any]:
    cx, cy = centre
    return {
        "op": "sub",
        "a": {"op": "const", "value": radius * radius},
        "b": {
            "op": "sum",
            "terms": [
                _square_of_offset("alpha", cx),
                _square_of_offset("beta", cy),
            ],
        },
    }


def disc_case(
    radius: float = 0.75,
    centre: tuple[float, float] = (1.0, 1.0),
    span: tuple[float, float] = (0.0, 2.0),
    expect_components: int = 1,
) -> WindowCase:
    """A disc: analytic boundary, analytic area, one closed component."""

    return _case(
        {
            "case_id": "synth_disc",
            "title": "disc of known radius: closed-form boundary and area",
            "mode": "plane",
            "axes": [
                {"name": "alpha", "lo": span[0], "hi": span[1]},
                {"name": "beta", "lo": span[0], "hi": span[1]},
            ],
            "constants": {"radius": radius, "centre_a": centre[0], "centre_b": centre[1]},
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_disc",
                    "kind": "inequality",
                    "side": "ge",
                    "value": disc_value(centre, radius),
                    "contract": {"convex": True},
                }
            ],
            "anchor": {"alpha": centre[0], "beta": centre[1]},
            "contract": {
                "min_boundary_order": 1.9,
                "area_rel_tol": 1e-3,
                "expect": {"components": expect_components},
            },
        }
    )


def analytic_disc(
    radius: float = 0.75,
    centre: tuple[float, float] = (1.0, 1.0),
    name: str = "disc",
) -> AnalyticReference:
    cx, cy = centre

    def _half_width(alpha: float) -> float:
        return max(radius * radius - (alpha - cx) ** 2, 0.0) ** 0.5

    return AnalyticReference(
        name=name,
        lower=lambda alpha: cy - _half_width(alpha),
        upper=lambda alpha: cy + _half_width(alpha),
    )


def two_disc_case() -> WindowCase:
    """Two disjoint discs -> two components, area = two discs."""

    return _case(
        {
            "case_id": "synth_two_discs",
            "title": "union of two disjoint discs: two components",
            "mode": "plane",
            "axes": [
                {"name": "alpha", "lo": 0.0, "hi": 2.0},
                {"name": "beta", "lo": 0.0, "hi": 1.0},
            ],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_union",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "max",
                        "args": [
                            disc_value((0.5, 0.5), 0.3),
                            disc_value((1.5, 0.5), 0.3),
                        ],
                    },
                }
            ],
            "contract": {"expect": {"components": 2}},
        }
    )


def _quadratic_band_value(axis: str, centre: float, curvature: float, offset: float,
                          side: str) -> dict[str, Any]:
    """``offset + curvature * (axis - centre) ** 2`` entering as ``+/- beta``."""

    profile = {
        "op": "sum",
        "terms": [
            {"op": "const", "value": offset},
            {
                "op": "scale",
                "factor": curvature,
                "arg": _square_of_offset(axis, centre),
            },
        ],
    }
    if side == "lower":
        return {
            "op": "sub",
            "a": {"op": "var", "name": "beta"},
            "b": profile,
        }
    return {
        "op": "sub",
        "a": profile,
        "b": {"op": "var", "name": "beta"},
    }


def parabolic_band_case() -> WindowCase:
    """A band between two parabolas: smooth analytic boundaries, order-2 check."""

    return _case(
        {
            "case_id": "synth_parabolic_band",
            "title": "band between beta = 0.4 + 0.8 (alpha - 0.5)^2 and "
            "beta = 1.6 - 0.5 (alpha - 0.5)^2",
            "mode": "plane",
            "axes": [
                {"name": "alpha", "lo": 0.0, "hi": 1.0},
                {"name": "beta", "lo": 0.0, "hi": 2.0},
            ],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_low",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _quadratic_band_value("alpha", 0.5, 0.8, 0.4, "lower"),
                    "contract": {"slope": {"alpha": 0}, "convex": True},
                },
                {
                    "id": "c_high",
                    "kind": "inequality",
                    "side": "ge",
                    "value": _quadratic_band_value("alpha", 0.5, -0.5, 1.6, "upper"),
                    "contract": {"slope": {"alpha": 0}, "convex": True},
                },
            ],
            "anchor": {"alpha": 0.5, "beta": 1.0},
            "contract": {
                "min_boundary_order": 1.9,
                "expect": {"components": 1},
            },
        }
    )


def analytic_band(
    name: str = "parabolic_band",
) -> AnalyticReference:
    def lower(alpha: float) -> float:
        return 0.4 + 0.8 * (alpha - 0.5) ** 2

    def upper(alpha: float) -> float:
        return 1.6 - 0.5 * (alpha - 0.5) ** 2

    return AnalyticReference(name=name, lower=lower, upper=upper)



SYNTHETIC_CASES = {
    "strip": strip_case,
    "folded": folded_case,
    "two_bands": offset_bands_case,
    "band": band_case,
    "equality": equality_band_case,
    "table": table_case,
    "lattice": lattice_case,
    "empty": empty_case,
    "disc": disc_case,
    "two_discs": two_disc_case,
    "parabolic_band": parabolic_band_case,
}