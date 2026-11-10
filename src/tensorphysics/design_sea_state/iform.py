"""IFORM — inverse first-order reliability method environmental contours.

Algorithm (Winterstein et al. 1993; DNV-RP-C205 App. 3)
-------------------------------------------------------
1. Rosenblatt-transform the joint metocean density onto independent standard
   normals (see :mod:`~tensorphysics.design_sea_state.rosenblatt`).
2. Draw a circle of radius ``beta`` in that space:
   ``u1 = beta cos(theta)``, ``u2 = beta sin(theta)``.
3. Map the circle back to physical space with the inverse transform.  The
   resulting closed curve is the ``beta``-level environmental contour.

Two properties follow from the construction and are used as exact anchors:

* **Probability content** — ``P(u1**2 + u2**2 > beta**2) = exp(-beta**2/2)``,
  independent of the marginals, so the contour's exceedance probability is
  analytic and equals the convention-C target
  (:func:`~tensorphysics.design_sea_state.reliability_index.beta_contour_from_pout`).
* **Extreme H_s** — because ``u1 = beta cos(theta) <= beta`` with equality at
  ``theta = 0``, the largest significant wave height on the contour is
  ``F_H^{-1}(Phi(beta))`` exactly, i.e. the marginal metocean return value at
  annual exceedance ``Phi(-beta)``.  Any numerical error beyond rounding here
  means the transform or the angle sweep is broken.

What the contour is **not**
---------------------------
The IFORM contour is *not* the chi-square (Mahalanobis) ellipse of the joint
normal approximation: the ellipse ``z^T R^-1 z = beta^2`` and the Rosenblatt
curve ``z_x^2 + (z_y - rho z_x)^2/(1 - rho^2) = beta^2`` share only a few points.
:func:`mahalanobis_residual` measures that gap, so a regression that silently
swaps one for the other is caught rather than admired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .rosenblatt import (
    BivariateNormalTransform,
    RosenblattTransform,
    standard_normal_cdf,
)

__all__ = [
    "ContourPoint",
    "iform_contour",
    "u_circle",
    "contour_array",
    "contour_max_hs",
    "analytic_max_hs",
    "quadratic_residual_normal",
    "mahalanobis_residual",
    "magnitude_on_contour",
    "max_response_on_contour",
    "contour_refinement_error",
]


@dataclass(frozen=True)
class ContourPoint:
    """One point of an environmental contour, kept in both spaces."""

    theta: float
    u1: float
    u2: float
    hs: float
    tz: float

    @property
    def u_radius2(self) -> float:
        return self.u1 * self.u1 + self.u2 * self.u2


def u_circle(beta: float, n_theta: int = 72) -> list[tuple[float, float, float]]:
    """``(theta, u1, u2)`` samples on the radius-``beta`` circle (closed, no repeat)."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if n_theta < 8:
        raise ValueError("need at least 8 angles to resolve a closed contour")
    out = []
    for i in range(n_theta):
        theta = 2.0 * math.pi * i / n_theta
        out.append((theta, beta * math.cos(theta), beta * math.sin(theta)))
    return out


def iform_contour(transform, beta: float, n_theta: int = 72) -> list[ContourPoint]:
    """Environmental contour at reliability radius ``beta`` (convention C).

    ``transform`` is any object exposing ``inverse(u1, u2) -> (x, y)``; both
    :class:`~tensorphysics.design_sea_state.rosenblatt.RosenblattTransform` and
    :class:`~tensorphysics.design_sea_state.rosenblatt.BivariateNormalTransform`
    qualify, which is what makes the analytic anchor reusable.
    """
    points = []
    for theta, u1, u2 in u_circle(beta, n_theta):
        hs, tz = transform.inverse(u1, u2)
        points.append(ContourPoint(theta=theta, u1=u1, u2=u2, hs=hs, tz=tz))
    return points


def contour_array(contour: Sequence[ContourPoint]) -> dict[str, list[float]]:
    """Column view of a contour (JSON/plot friendly)."""
    return {
        "theta": [p.theta for p in contour],
        "u1": [p.u1 for p in contour],
        "u2": [p.u2 for p in contour],
        "hs": [p.hs for p in contour],
        "tz": [p.tz for p in contour],
    }


def contour_max_hs(contour: Sequence[ContourPoint]) -> float:
    """Largest H_s attained on the contour (numerical, from the sampled points)."""
    return max(p.hs for p in contour)


def analytic_max_hs(transform: RosenblattTransform, beta: float) -> float:
    """``F_H^{-1}(Phi(beta))`` — the exact extreme H_s of the IFORM contour."""
    return transform.hs_marginal.ppf(standard_normal_cdf(beta))


def quadratic_residual_normal(
    transform: BivariateNormalTransform, contour: Sequence[ContourPoint]
) -> float:
    """Max relative error of ``u1**2 + u2**2`` recomputed from physical coordinates."""
    worst = 0.0
    for p in contour:
        q = transform.quadratic_form(p.hs, p.tz)
        worst = max(worst, abs(q / (p.u1**2 + p.u2**2) - 1.0))
    return worst


def mahalanobis_residual(
    transform: BivariateNormalTransform, contour: Sequence[ContourPoint]
) -> float:
    """Max relative deviation of the contour from the chi-square ellipse.

    A *large* value is the healthy outcome: it proves the code builds the
    Rosenblatt contour and not the (much cruder) correlated-normal ellipse.
    """
    worst = 0.0
    for p in contour:
        q = transform.mahalanobis_form(p.hs, p.tz)
        worst = max(worst, abs(q / p.u_radius2 - 1.0))
    return worst


def magnitude_on_contour(contour: Sequence[ContourPoint], fn) -> list[float]:
    """Evaluate a scalar response ``fn(hs, tz)`` on every contour point."""
    return [fn(p.hs, p.tz) for p in contour]


def max_response_on_contour(contour: Sequence[ContourPoint], fn) -> tuple[float, ContourPoint]:
    """Contour-route design response: ``max`` over the contour, with its argmax point.

    The argmax is *not* necessarily ``theta = 0``: for a response that grows with
    T_z as well as with H_s the trade-off moves the design point off the
    extreme-H_s point, and that shift is precisely what a marginal (Rayleigh
    only) approach cannot see.
    """
    best_val = -math.inf
    best_pt = contour[0]
    for p in contour:
        val = fn(p.hs, p.tz)
        if val > best_val:
            best_val, best_pt = val, p
    return best_val, best_pt


def contour_refinement_error(
    transform, beta: float, coarse: int = 36, fine: int = 288
) -> dict[str, float]:
    """Angle-refinement convergence: coarse vs fine contour, evaluated by response peak.

    Reported quantities are relative differences, so an under-resolved sweep
    shows up as a number instead of as a quietly biased design value.
    """
    q = lambda hs, tz: hs * hs * tz  # noqa: E731 - generic monotone probe response
    coarse_contour = iform_contour(transform, beta, coarse)
    fine_contour = iform_contour(transform, beta, fine)
    v_coarse, _ = max_response_on_contour(coarse_contour, q)
    v_fine, _ = max_response_on_contour(fine_contour, q)
    hs_coarse = contour_max_hs(coarse_contour)
    hs_fine = contour_max_hs(fine_contour)
    return {
        "response_rel_error": abs(v_coarse - v_fine) / abs(v_fine),
        "max_hs_rel_error": abs(hs_coarse - hs_fine) / abs(hs_fine),
        "n_coarse": coarse,
        "n_fine": fine,
    }