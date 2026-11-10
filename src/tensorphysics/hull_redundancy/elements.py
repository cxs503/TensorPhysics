"""Element-level strength laws for the component-based hull-girder redundancy model.

The module re-uses the elastoplastic element curves of
:mod:`tensorphysics.elastoplastic_collapse` (plate load-end shortening curve and
the J2 uniaxial law) and adds the two component-level degradation laws required
by the RIF study:

``tripping``
    Stiffener torsional (lateral-torsional) buckling.  The stiffener stays
    welded to the plating (shear connection retained) but its lateral/torsional
    stiffness is exhausted, so the element stress is capped at the tripping
    stress :math:`\\sigma_{cr,t}` with *zero tangent stiffness* beyond it (an
    ideal-plastic plateau).  This is exactly the "stiffness -> 0, shear
    connection retained" semantics of the campaign specification.

``debond`` (loss of stiffener-plate composite action)
    The weld line no longer transfers the transverse/lateral action between
    stiffener and plating.  Two consequences are modelled: (i) the stiffener
    becomes a free-standing bare strut whose capacity is the Euler stress about
    its own weak axis (Johnson-Ostenfeld corrected), (ii) the plating it used to
    support now spans between the *remaining* connected longitudinals, i.e. its
    buckling width doubles.

Sign convention: ``eps > 0`` is compression, so ``sigma > 0`` is compressive
stress.  Tension follows the same J2 law with the opposite sign and is limited
by yield only (no fracture model - declared limitation).

References
----------
- Smith, C.S. (1977) Trans. SNAME 85: intercolumn collapse of stiffened plates.
- Faulkner, D. (1975) J. Ship Research 19(1): effective plating.
- Paik, J.K. & Thayamballi, A.K. (2003) Ultimate Limit State Design of
  Steel-Plated Structures, Wiley (element curves, Johnson-Ostenfeld correction).
- Timoshenko, S.P. & Gere, J.M. (1961) Theory of Elastic Stability (tripping /
  torsional buckling of flat-bar stiffeners).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..elastoplastic_collapse import (
    VonMisesMaterial,
    plate_load_shortening_stress,
    uniaxial_elastoplastic_stress,
)

__all__ = [
    "STEEL_SHEAR_MODULUS",
    "plate_buckling_coefficient",
    "plate_buckling_stress",
    "faulkner_plate_strength_ratio",
    "effective_plate_buckling_stress",
    "johnson_ostenfeld",
    "tripping_stress",
    "bare_strut_stress",
    "plate_element_stress",
    "strut_element_stress",
]

STEEL_SHEAR_MODULUS = 79.3e9  # Pa, G = E / (2(1+nu)) for nu = 0.3


def plate_buckling_coefficient(a_over_b: float) -> float:
    """Elastic plate buckling coefficient ``k`` for a simply supported panel."""
    if a_over_b <= 0.0:
        raise ValueError("a/b must be positive")
    if a_over_b >= 1.0:
        return 4.0
    return float((a_over_b + 1.0 / a_over_b) ** 2)


def plate_buckling_stress(
    material: VonMisesMaterial, b: float, t: float, a: float
) -> float:
    """Elastic plate buckling stress ``sigma_cr = k pi^2 E/(12(1-nu^2)) (t/b)^2``.

    Parameters
    ----------
    b : float
        Buckling width of the panel (longitudinal spacing) [m].
    t : float
        Plate thickness [m].
    a : float
        Panel length in the load direction (frame spacing) [m].
    """
    if b <= 0.0 or t <= 0.0 or a <= 0.0:
        raise ValueError("panel dimensions must be positive")
    k = plate_buckling_coefficient(a / b)
    return float(
        k * math.pi**2 * material.E / (12.0 * (1.0 - material.nu**2)) * (t / b) ** 2
    )


def faulkner_plate_strength_ratio(beta: float) -> float:
    """Faulkner (1975) plate strength ratio ``phi(beta)`` [-].

    ``phi = 1`` for ``beta <= 1``, ``phi = 2/beta - 1/beta^2`` otherwise.
    """
    if beta <= 0.0:
        raise ValueError("plate slenderness must be positive")
    if beta <= 1.0:
        return 1.0
    return float(2.0 / beta - 1.0 / beta**2)


def effective_plate_buckling_stress(
    material: VonMisesMaterial, b: float, t: float, a: float
) -> float:
    """Elastoplastic (Faulkner-calibrated) plate buckling stress [Pa].

    The *elastic* buckling stress of a slender hull panel (``k = 4``) is often
    above yield, in which case a naive element law never softens the plating and
    the section has no moment peak at all.  Real imperfect plating fails by
    elastoplastic buckling at the Faulkner strength ``phi(beta) sigma_y``.  This
    function supplies the effective buckling stress that makes the von Karman
    post-buckling branch of
    :func:`tensorphysics.elastoplastic_collapse.plate_load_shortening_stress`
    reproduce exactly that strength::

        sigma_cr,eff = min( sigma_cr,elastic , (phi sigma_y)^2 / sigma_y )
        beta = (b/t) sqrt(sigma_y/E)

    so ``sqrt(sigma_cr,eff sigma_y) = phi sigma_y`` and the plate softens after
    reaching ``phi sigma_y`` - the classical Smith-method plate element.
    """
    sigma_el = plate_buckling_stress(material, b=b, t=t, a=a)
    beta = (b / t) * math.sqrt(material.sigma_y0 / material.E)
    phi = faulkner_plate_strength_ratio(beta)
    sigma_target = phi * material.sigma_y0
    return float(min(sigma_el, sigma_target**2 / material.sigma_y0))


def johnson_ostenfeld(sigma_cr: float, sigma_y: float) -> float:
    """Elastoplastic column/plate strength (Johnson-Ostenfeld) [Pa]."""
    if sigma_cr <= 0.0 or sigma_y <= 0.0:
        raise ValueError("stresses must be positive")
    if sigma_cr >= 0.5 * sigma_y:
        return float(sigma_y * (1.0 - sigma_y / (4.0 * sigma_cr)))
    return float(sigma_cr)


def tripping_stress(
    material: VonMisesMaterial,
    h: float,
    t_w: float,
    span: float,
    plate_t: float,
    n_long: int = 1,
    lambda_trip: float = 1.0,
) -> float:
    """Elastic torsional (tripping) stress of a flat-bar longitudinal [Pa].

    Rotation is taken about the weld line at the plating (the plating supplies
    the rotational restraint).  With ``C_w = 0`` for a flat bar::

        sigma_cr,trip = (G J) / I_p ,  J = h t_w^3 / 3 ,
        I_p = t_w h^3 / 3 + b_e t^3 / 12 ,   b_e = min(b/2, 0.4 a)

    where the polar moment ``I_p`` includes the attached plate strip rotating
    with the stiffener (its own-thickness inertia only, since the plate lies on
    the rotation axis).  ``lambda_trip >= 1`` degrades the restraint.
    """
    if h <= 0.0 or t_w <= 0.0 or span <= 0.0 or plate_t <= 0.0:
        raise ValueError("stiffener dimensions must be positive")
    if lambda_trip <= 0.0:
        raise ValueError("lambda_trip must be positive")
    G = STEEL_SHEAR_MODULUS
    J = h * t_w**3 / 3.0
    b_e = min(0.4 * span, 0.5 * span)
    I_p = t_w * h**3 / 3.0 + b_e * plate_t**3 / 12.0
    sigma_el = G * J / I_p / lambda_trip
    return johnson_ostenfeld(sigma_el, material.sigma_y0)


def bare_strut_stress(
    material: VonMisesMaterial, h: float, t_w: float, span: float
) -> float:
    """Euler strut strength of a bare flat bar about its weak axis [Pa].

    ``r = h / sqrt(12)``, ``sigma_E = pi^2 E / (span/r)^2``, Johnson-Ostenfeld
    elastoplastic correction applied.
    """
    if h <= 0.0 or t_w <= 0.0 or span <= 0.0:
        raise ValueError("strut dimensions must be positive")
    r = h / math.sqrt(12.0)
    slenderness = span / r
    sigma_e = math.pi**2 * material.E / slenderness**2
    return johnson_ostenfeld(sigma_e, material.sigma_y0)


def plate_element_stress(
    eps: float,
    material: VonMisesMaterial,
    sigma_cr: float,
    softening_modulus_ratio: float = 0.02,
    residual_stress_ratio: float = 0.2,
) -> float:
    """Average element stress of a plate strip [Pa], compression positive."""
    if eps >= 0.0:
        return float(
            plate_load_shortening_stress(
                eps,
                material,
                sigma_cr,
                softening_modulus_ratio=softening_modulus_ratio,
                residual_stress_ratio=residual_stress_ratio,
            )
        )
    return float(uniaxial_elastoplastic_stress(eps, material))


def strut_element_stress(
    eps: float, material: VonMisesMaterial, cap: Optional[float] = None
) -> float:
    """Element stress of a stiffener fiber [Pa] with an optional compression cap.

    ``cap`` implements the zero-tangent plateau of tripping (``cap =
    sigma_cr,trip``) or of a debonded bare strut (``cap = sigma_E,bare``).
    """
    sigma = float(uniaxial_elastoplastic_stress(eps, material))
    if cap is not None and eps > 0.0 and sigma > cap:
        return float(cap)
    return sigma