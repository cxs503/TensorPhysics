"""Ultimate hull girder bending capacity and utilisation check.

Simplified IACS CSR-type capacity estimate
------------------------------------------
The ultimate bending capacity is estimated from the midship elastic section
modulus with a plate-buckling knockdown on the compressive flange::

    M_u = S_M * sigma_y * Phi(beta),   beta = (b/t) sqrt(sigma_y / E)

with the standard Falkner/John's effective-width strength curve

    Phi(beta) = 1                      for beta <= 1
    Phi(beta) = 2.25/beta - 1.25/beta^2  for beta >  1

This is a *first-yield with buckling reduction* estimate of the ultimate
moment: it captures the governing failure mode of a slender compression flange
and is the check quoted by classification practice, but it is **not** a
progressive-collapse (Smith-method) analysis, which would additionally require
the full sectional idealisation.  The evidence level of this module is
therefore "simplified rule-type estimate", and the utilisation numbers carry
that caveat.

The design check uses the standard partial factor on the ultimate capacity::

    utilisation = M_design / (M_u / gamma_m),   gamma_m = 1.1
"""

from __future__ import annotations

import math

from .model import ShipGirder

__all__ = [
    "YOUNG_STEEL",
    "plate_buckling_reduction",
    "ultimate_bending_moment",
    "elastic_bending_stress",
    "utilisation_check",
]

YOUNG_STEEL = 2.06e11  # Pa


def plate_buckling_reduction(beta: float) -> float:
    """Falkner/John's plate strength ratio ``Phi`` for slenderness ``beta``."""
    if beta <= 0.0:
        raise ValueError("plate slenderness must be positive")
    if beta <= 1.0:
        return 1.0
    return float(2.25 / beta - 1.25 / beta ** 2)


def ultimate_bending_moment(ship: ShipGirder, yield_stress: float | None = None) -> dict:
    """Simplified ultimate bending capacity of the midship section (N*m)."""
    sigma_y = ship.yield_stress if yield_stress is None else float(yield_stress)
    phi = plate_buckling_reduction(ship.plate_slenderness)
    m_u = ship.section_modulus * sigma_y * phi
    return {
        "M_ultimate_Nm": float(m_u),
        "section_modulus_m3": float(ship.section_modulus),
        "yield_stress_MPa": sigma_y / 1e6,
        "plate_slenderness_beta": float(ship.plate_slenderness),
        "buckling_reduction_phi": float(phi),
        "method": "simplified first-yield with Falkner/John's plate buckling reduction",
        "evidence_level": "simplified_rule_estimate",
    }


def elastic_bending_stress(moment: float, section_modulus: float) -> float:
    """Elastic bending stress ``sigma = M / S_M`` (Pa)."""
    if section_modulus <= 0.0:
        raise ValueError("section modulus must be positive")
    return float(abs(moment) / section_modulus)


def utilisation_check(
    moment_design: float,
    ultimate_moment: float,
    gamma_m: float = 1.1,
    allowable_ratio: float = 1.0,
) -> dict:
    """Compare the design bending moment with the factored ultimate capacity."""
    if ultimate_moment <= 0.0:
        raise ValueError("ultimate moment must be positive")
    if gamma_m <= 0.0:
        raise ValueError("partial factor must be positive")
    capacity = ultimate_moment / gamma_m
    utilisation = float(abs(moment_design) / capacity)
    return {
        "M_design_Nm": float(abs(moment_design)),
        "M_ultimate_Nm": float(ultimate_moment),
        "gamma_m": float(gamma_m),
        "M_capacity_factored_Nm": float(capacity),
        "utilisation": utilisation,
        "allowable_utilisation": float(allowable_ratio),
        "passes": bool(utilisation <= allowable_ratio),
        "margin_fraction": float(1.0 - utilisation / allowable_ratio),
    }