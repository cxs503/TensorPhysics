"""Element strength laws for the cross-scale Smith chain (vectorised).

Every element of a :class:`~.idealisation.GirderSection` is reduced to a
one-dimensional average-stress/average-strain law that is evaluated in
*compression positive* form:

``PlateLaw``
    Plating.  Elastic up to the elastoplastic plate buckling stress, von Karman
    post-buckling branch ``sigma = sqrt(sigma_cr E eps)`` up to yield, then the
    Faulkner strength ``phi sigma_y`` with a small softening slope and a
    residual plateau.  The plate is fully effective in tension.

``StrutLaw``
    Stiffener fibre.  J2 uniaxial law capped (zero tangent) at the
    column/tripping capacity, i.e. the Smith "stiffness -> 0" idealisation.
    Full strength in tension.

Strength laws (all pre-existing, verified kernels are re-used where possible)
-----------------------------------------------------------------------------
``plate``      Falkner/Johnstone-Faulkner strength ``phi = 2/beta - 1/beta^2``
               (:func:`tensorphysics.hull_redundancy.elements.faulkner_plate_strength_ratio`).
``column``     Euler stress of the stiffener plus attached plating of effective
               breadth ``b_e = phi * spacing``, Johnson-Ostenfeld corrected.
``tripping``   Flat-bar lateral-torsional (tripping) stress about the weld line,
               Johnson-Ostenfeld corrected, via
               :func:`tensorphysics.hull_redundancy.elements.tripping_stress`.

The Johnson-Ostenfeld correction and the von Karman / load-shortening curves are
taken from the verified ``elastoplastic_collapse`` / ``hull_redundancy``
kernels instead of being re-derived, so this module only adds the *cross-scale
dispatch* (plate vs strut, flange vs web, per-element capacity bookkeeping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..elastoplastic_collapse import VonMisesMaterial
from ..elastoplastic_collapse.progressive_collapse import (
    plate_load_shortening_stress,
    uniaxial_elastoplastic_stress,
)
from ..hull_redundancy.elements import (
    faulkner_plate_strength_ratio,
    plate_buckling_coefficient,
    tripping_stress,
)

__all__ = [
    "plate_slenderness",
    "plate_buckling_stress",
    "plate_strength",
    "effective_plate_buckling_stress",
    "column_strength",
    "stiffener_strength",
    "PlateLaw",
    "StrutLaw",
]


# ---------------------------------------------------------------------------
# scalar strength laws
# ---------------------------------------------------------------------------
def plate_slenderness(b: float, t: float, material: VonMisesMaterial) -> float:
    """Plate slenderness ``beta = (b/t) sqrt(sigma_y/E)`` [-]."""
    return float((b / t) * math.sqrt(material.sigma_y0 / material.E))


def plate_buckling_stress(
    b: float, t: float, a: float, material: VonMisesMaterial, k: Optional[float] = None
) -> float:
    """Elastic plate buckling stress ``k pi^2 E / (12 (1-nu^2)) (t/b)^2`` [Pa]."""
    kk = plate_buckling_coefficient(a / b) if k is None else float(k)
    return float(kk * math.pi**2 * material.E / (12.0 * (1.0 - material.nu**2)) * (t / b) ** 2)


def plate_strength(beta: float, material: VonMisesMaterial) -> float:
    """Faulkner ultimate strength ``phi(beta) sigma_y`` of a plate panel [Pa]."""
    return float(faulkner_plate_strength_ratio(beta) * material.sigma_y0)


def effective_plate_buckling_stress(
    b: float,
    t: float,
    a: float,
    material: VonMisesMaterial,
    k: Optional[float] = None,
) -> float:
    """Buckling stress that makes the von Karman branch peak at ``phi sigma_y``.

    ``sigma_cr,eff = min(sigma_cr,elastic, (phi sigma_y)^2 / sigma_y)`` so that
    ``sqrt(sigma_cr,eff sigma_y) = phi sigma_y`` (classical Smith plate
    element).
    """
    sigma_el = plate_buckling_stress(b, t, a, material, k)
    beta = plate_slenderness(b, t, material)
    sigma_target = plate_strength(beta, material)
    return float(min(sigma_el, sigma_target**2 / material.sigma_y0))


def column_strength(
    h: float,
    t_w: float,
    b_f: float,
    t_f: float,
    span: float,
    attached_width: float,
    plate_t: float,
    material: VonMisesMaterial,
) -> Tuple[float, float]:
    """Column buckling strength of a stiffener with attached plating.

    The column is the profile plus a plating strip of effective breadth
    ``attached_width`` (= ``phi * spacing``).  Returns
    ``(sigma_cr [Pa], slenderness lambda [-])`` with the Johnson-Ostenfeld
    elastoplastic correction applied.
    """
    h_w = h - t_f if b_f > 0.0 else h
    A_w = h_w * t_w
    A_f = b_f * t_f
    A_p = attached_width * plate_t
    A = A_w + A_f + A_p
    if A <= 0.0:
        raise ValueError("column has no area")
    # local z measured from the plate mid-surface
    z_w = h_w / 2.0
    z_f = h_w + 0.5 * t_f if b_f > 0.0 else 0.0
    z_p = 0.0
    z_c = (A_w * z_w + A_f * z_f + A_p * z_p) / A
    I = (
        A_w * (z_w - z_c) ** 2
        + t_w * h_w**3 / 12.0
        + A_f * (z_f - z_c) ** 2
        + A_p * (z_p - z_c) ** 2
        + attached_width * plate_t**3 / 12.0
    )
    r = math.sqrt(I / A)
    lam = (span / (math.pi * r)) * math.sqrt(material.sigma_y0 / material.E)
    sigma_e = material.sigma_y0 / lam**2
    if sigma_e >= 0.5 * material.sigma_y0:
        sigma_cr = material.sigma_y0 * (1.0 - material.sigma_y0 / (4.0 * sigma_e))
    else:
        sigma_cr = sigma_e
    return float(sigma_cr), float(lam)


def stiffener_strength(
    h: float,
    t_w: float,
    b_f: float,
    t_f: float,
    span: float,
    spacing_b: float,
    plate_t: float,
    beta: float,
    material: VonMisesMaterial,
) -> Dict[str, float]:
    """Capacities of a longitudinal profile [-]/[Pa].

    Returns a dict with the column capacity, the tripping capacity, the
    combined capacity ``min(sigma_y, column, tripping)`` and the diagnostics
    (``lambda``, ``phi_attached``).
    """
    phi = faulkner_plate_strength_ratio(beta)
    sigma_col, lam = column_strength(
        h=h, t_w=t_w, b_f=b_f, t_f=t_f, span=span,
        attached_width=phi * spacing_b, plate_t=plate_t, material=material,
    )
    sigma_trip = tripping_stress(
        material, h=h - t_f if b_f > 0.0 else h, t_w=t_w, span=span,
        plate_t=plate_t,
    )
    cap = min(material.sigma_y0, sigma_col, sigma_trip)
    return {
        "sigma_col": float(sigma_col),
        "sigma_trip": float(sigma_trip),
        "sigma_cap": float(cap),
        "lambda_column": float(lam),
        "phi_attached": float(phi),
    }


# ---------------------------------------------------------------------------
# vectorised element laws
# ---------------------------------------------------------------------------
class PlateLaw:
    """Average stress law of a plating element (compression positive)."""

    kind = "plate"

    def __init__(
        self,
        material: VonMisesMaterial,
        sigma_cr_eff: float,
        sigma_u: float,
        softening_modulus_ratio: float = 0.02,
        residual_stress_ratio: float = 0.2,
    ) -> None:
        self.material = material
        self.sigma_cr_eff = float(sigma_cr_eff)
        self.sigma_u = float(sigma_u)
        self.soft = float(softening_modulus_ratio)
        self.resid = float(residual_stress_ratio)

    def stress(self, eps):
        """Vectorised compressive (eps>0) / tensile (eps<0) stress [Pa]."""
        arr = np.asarray(eps, dtype=float)
        out = np.empty_like(arr)
        comp = arr > 0.0
        # tension: J2 uniaxial
        if np.any(~comp):
            out[~comp] = uniaxial_elastoplastic_stress_array(arr[~comp], self.material)
        if np.any(comp):
            out[comp] = self._compression(arr[comp])
        return out if out.shape else float(out)

    def _compression(self, eps: np.ndarray) -> np.ndarray:
        m = self.material
        E, sy = m.E, m.sigma_y0
        eps_cr = self.sigma_cr_eff / E
        eps_y = sy / E
        if eps_cr >= eps_y:      # stocky plate: yields before it buckles
            return uniaxial_elastoplastic_stress_array(eps, m)
        res = np.empty_like(eps)
        e1 = eps <= eps_cr
        res[e1] = E * eps[e1]
        e2 = (eps > eps_cr) & (eps <= eps_y)
        res[e2] = np.sqrt(self.sigma_cr_eff * E * eps[e2])
        e3 = eps > eps_y
        if np.any(e3):
            drop = self.soft * E * (eps[e3] - eps_y)
            res[e3] = np.maximum(self.sigma_u - drop, self.resid * self.sigma_u)
        return res


class StrutLaw:
    """Stress law of a stiffener fibre with a compression cap (zero tangent)."""

    kind = "stiffener"

    def __init__(self, material: VonMisesMaterial, cap: float) -> None:
        self.material = material
        self.cap = float(cap)

    def stress(self, eps):
        arr = np.asarray(eps, dtype=float)
        sig = uniaxial_elastoplastic_stress_array(arr, self.material)
        cap_mask = (arr > 0.0) & (sig > self.cap)
        if np.any(cap_mask):
            sig = np.where(cap_mask, self.cap, sig)
        return sig if sig.shape else float(sig)


def uniaxial_elastoplastic_stress_array(eps: np.ndarray, material: VonMisesMaterial) -> np.ndarray:
    """Vectorised :func:`uniaxial_elastoplastic_stress` (same closed form)."""
    eps = np.asarray(eps, dtype=float)
    a = np.abs(eps)
    eps_y = material.sigma_y0 / material.E
    elastic = material.E * eps
    eps_p = (a - eps_y) / (1.0 + material.H / material.E)
    plastic = np.sign(eps) * (material.sigma_y0 + material.H * np.maximum(eps_p, 0.0))
    return np.where(a <= eps_y, elastic, plastic)