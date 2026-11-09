"""Stage 3 of the envelope machinery: boundary refinement by bisection.

The probing operator (stage 2) only knows, for two adjacent samples, that one is
inside the capability's domain and the other is outside.  This module narrows
that bracket with a bisection on the *same* decision predicate until the
bracket is at most ``rel_tol`` of the axis span (default 1e-3 = 0.1 %), and
reports the boundary value with a confidence label.

A boundary is only refined when the decision predicate is a *point property*
(acceptance criterion / pathology).  Sequence properties (monotonicity and sign
laws) cannot be bisected; they are reported at the sampling resolution and the
caller is told so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .axis import AxisSpec

__all__ = ["Boundary", "bisect_boundary", "DEFAULT_REL_TOL"]

DEFAULT_REL_TOL = 1e-3
MAX_ITERATIONS = 80


@dataclass(frozen=True)
class Boundary:
    """A refined domain boundary on one axis."""

    axis: str
    side: str                 # "low" (domain starts here) or "high" (domain ends here)
    value: float              # best estimate of the boundary
    width: float              # remaining bracket width in value units
    span_fraction: float      # width / axis span
    inside_value: float       # value known to be inside the domain
    outside_value: float      # value known to be outside the domain
    source: str               # "criterion" | "pathology" | "law"
    iterations: int
    converged: bool
    confidence: float         # 1.0 = bisection converged; halved for sampling-limited

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.9:
            return "high"
        if self.confidence >= 0.5:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "side": self.side,
            "value": self.value,
            "width": self.width,
            "span_fraction": self.span_fraction,
            "inside_value": self.inside_value,
            "outside_value": self.outside_value,
            "source": self.source,
            "iterations": self.iterations,
            "converged": self.converged,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
        }


def bisect_boundary(
    axis: AxisSpec,
    inside: float,
    outside: float,
    predicate: Callable[[float], bool],
    *,
    source: str = "criterion",
    rel_tol: float = DEFAULT_REL_TOL,
    max_iterations: int = MAX_ITERATIONS,
) -> Boundary:
    """Locate the in/out transition between ``inside`` and ``outside``.

    ``predicate(v)`` returns ``True`` when ``v`` is inside the capability's
    domain (a point property).  Bisection works in *scale coordinates* so that
    geometric spans (``log`` axes) refine multiplicatively.
    """
    if not rel_tol > 0.0:
        raise ValueError("rel_tol must be positive")
    a_coord = axis.to_scale(inside)
    b_coord = axis.to_scale(outside)
    target = rel_tol * axis.scale_width
    lo, hi = (a_coord, b_coord) if a_coord < b_coord else (b_coord, a_coord)
    lo_inside = predicate(axis.from_scale(lo))
    iterations = 0
    while abs(hi - lo) > target and iterations < max_iterations:
        mid = 0.5 * (lo + hi)
        if predicate(axis.from_scale(mid)) == lo_inside:
            lo = mid
        else:
            hi = mid
        iterations += 1
    inside_coord, outside_coord = (lo, hi) if lo_inside else (hi, lo)
    width_coord = abs(hi - lo)
    # map the bracket back to value units (relative width is what matters)
    inside_value = axis.from_scale(inside_coord)
    outside_value = axis.from_scale(outside_coord)
    boundary_value = axis.from_scale(0.5 * (inside_coord + outside_coord))
    width = abs(outside_value - inside_value)
    span_fraction = width / axis.width if axis.width else 0.0
    converged = width_coord <= target
    return Boundary(
        axis=axis.name,
        side="low" if outside_value < inside_value else "high",
        value=boundary_value,
        width=width,
        span_fraction=span_fraction,
        inside_value=inside_value,
        outside_value=outside_value,
        source=source,
        iterations=iterations,
        converged=converged,
        confidence=1.0 if converged else 0.5,
    )