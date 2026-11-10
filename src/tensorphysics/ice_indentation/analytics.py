"""Closed-form references for contact and indentation verification.

Every formula used as a benchmark anchor is spelled out here (no numbers
buried in the drivers) so that a reviewer can check the algebra and the
conventions:

* Hertz elastic sphere contact and its yield-onset penetration;
* Sneddon's rigid flat punch on an elastic half-space;
* the exact uniaxial-strain (confined compression) solution used to verify the
  penalty contact implementation itself;
* the Johnson cavity-expansion mean pressure for a *deeply* plastic contact,
  kept only as a diagnostic with its validity ratio reported next to it
  (``E* a / (3 sigma_y) >> 1``); it is **not** used as an acceptance criterion,
  because for ice-like materials (``E*/sigma_y ~ 2000``) the ice indentation
  patches reachable in a small-strain finite element model sit below that
  window.

Conventions: ``E`` Young's modulus, ``nu`` Poisson ratio of the deformable
solid, the indenter is rigid, so the reduced modulus is
``E* = E / (1 - nu**2)``.  Stresses and pressures in Pa, lengths in m.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt

import numpy as np

from ..solid_fem.material import lame_parameters

__all__ = [
    "reduced_modulus",
    "hertz_contact_radius",
    "hertz_force",
    "hertz_mean_pressure",
    "hertz_yield_onset",
    "johnson_mean_pressure",
    "johnson_validity_ratio",
    "sneddon_flat_punch_force",
    "confined_uniaxial_strain",
    "IndentationReference",
]


def reduced_modulus(E: float, nu: float) -> float:
    """``E* = E / (1 - nu^2)`` for a rigid indenter."""
    if E <= 0.0 or not (-1.0 < nu < 0.5):
        raise ValueError("invalid elastic constants")
    return E / (1.0 - nu ** 2)


def hertz_contact_radius(radius: float, delta: float) -> float:
    """Hertz elastic contact radius ``a = sqrt(R delta)``."""
    if delta <= 0.0:
        return 0.0
    return sqrt(radius * delta)


def hertz_force(E_star: float, radius: float, delta: float) -> float:
    """Hertz elastic force ``P = 4/3 E* sqrt(R) delta**1.5``."""
    if delta <= 0.0:
        return 0.0
    return 4.0 / 3.0 * E_star * sqrt(radius) * delta ** 1.5


def hertz_mean_pressure(E_star: float, radius: float, delta: float) -> float:
    """Hertz mean contact pressure ``P / (pi a**2)``."""
    a = hertz_contact_radius(radius, delta)
    if a <= 0.0:
        return 0.0
    return hertz_force(E_star, radius, delta) / (pi * a * a)


def hertz_yield_onset(E: float, nu: float, sigma_y: float, radius: float) -> dict:
    """Penetration at which von Mises yield first appears under a sphere.

    First yield under a Hertzian contact occurs where the maximum pressure
    ``p_0 = 1.5 p_m`` reaches ``1.6 sigma_y`` (von Mises), i.e.
    ``p_m = 1.6 sigma_y / 1.5``.  With ``p_m = 4 E* a / (3 pi R)`` this gives
    ``a_y = 3 pi R p_m / (4 E*)`` and ``delta_y = a_y**2 / R`` (Hertz).
    """
    E_star = reduced_modulus(E, nu)
    p_m_yield = 1.6 * sigma_y / 1.5
    a = 3.0 * pi * radius * p_m_yield / (4.0 * E_star)
    delta = a * a / radius
    return {
        "penetration": delta,
        "contact_radius": a,
        "mean_pressure": p_m_yield,
        "force": hertz_force(E_star, radius, delta),
    }


def johnson_validity_ratio(E: float, nu: float, sigma_y: float, contact_radius: float) -> float:
    """``E* a / (3 sigma_y)`` -- the Johnson cavity-expansion validity ratio."""
    return reduced_modulus(E, nu) * contact_radius / (3.0 * sigma_y)


def johnson_mean_pressure(E: float, nu: float, sigma_y: float, contact_radius: float) -> float:
    """Johnson cavity-expansion mean pressure (elastic-perfectly-plastic).

    ``p_m = (2/3) sigma_y [1 + ln(E* a / (3 sigma_y))]`` -- Johnson, *Contact
    Mechanics* (1985), ch. 6, spherical indentation, frictionless, no
    hardening.  **Diagnostic only**: the expression is an asymptote of the deep
    plastic regime and is only meaningful for
    ``E* a / (3 sigma_y) >> 1``; callers must report
    :func:`johnson_validity_ratio` next to it (see the module docstring for why
    it is not an acceptance criterion here).
    """
    if contact_radius <= 0.0:
        return 0.0
    ratio = johnson_validity_ratio(E, nu, sigma_y, contact_radius)
    if ratio <= 1.0:
        raise ValueError(
            f"Johnson model needs E* a / (3 sigma_y) > 1, got {ratio:.3g}"
        )
    return 2.0 / 3.0 * sigma_y * (1.0 + log(ratio))


def sneddon_flat_punch_force(E: float, nu: float, punch_radius: float, delta: float) -> float:
    """Sneddon (1965) rigid flat punch on an elastic half-space: ``P = 2 E* a delta``."""
    if delta <= 0.0:
        return 0.0
    return 2.0 * reduced_modulus(E, nu) * punch_radius * delta


def confined_uniaxial_strain(E: float, nu: float, delta: float, height: float) -> dict:
    """Exact solution for a laterally confined block compressed by ``delta``.

    With rollers on the sides (``eps_xx = eps_yy = 0``) and a rigid platen on
    top, the state is homogeneous uniaxial strain: ``sigma_zz = (lam + 2 mu)
    delta / height``, ``sigma_xx = sigma_yy = lam delta / height``.  This is the
    analytic target for the *penalty contact* implementation itself.
    """
    lam, mu, _bulk = lame_parameters(E, nu)
    eps = delta / height
    return {
        "strain_zz": eps,
        "sigma_zz": (lam + 2.0 * mu) * eps,
        "sigma_xx": lam * eps,
        "force_per_area": (lam + 2.0 * mu) * eps,
    }


@dataclass(frozen=True)
class IndentationReference:
    """Reference force-displacement curve from one analytic model."""

    name: str
    delta: np.ndarray
    force: np.ndarray

    def relative_error(self, delta: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Pointwise ``|F_ref - F| / F_ref`` on the supplied grid."""
        ref = np.interp(delta, self.delta, self.force)
        return np.abs(ref - force) / np.maximum(ref, 1e-30)