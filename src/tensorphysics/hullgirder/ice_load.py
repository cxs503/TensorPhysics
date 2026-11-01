"""Ice pressure statistics acting on the hull girder.

Physical model
--------------
Level-ice / managed ice loading on a polar ship is treated as a stationary
random pressure field along the ship length::

    p(x, t) = mu_p + sigma_p * Z(x, t)

* ``mu_p``   global (whole-ship) ice pressure, anchored to the ISO 19906 /
  IACS Polar Class global-pressure level, which decreases with contact area:
  ``p_G = p_ref * (h / h_ref)^(-0.5)`` with ``p_ref = 1.2 MPa`` at
  ``h_ref = 1.0 m``;
* ``sigma_p = CoV * mu_p`` with ``CoV = 0.30`` (documented range 0.2-0.4 for
  ice-induced pressure peaks);
* ``Z``      unit-variance correlated field, exponential covariance
  ``C(dx) = exp(-|dx| / lambda_x)`` (see :mod:`tensorphysics.hullgirder.random_field`);
* ``lambda_x = f(h, c, v)`` is the longitudinal correlation length of the ice
  pressure field::

      lambda_x = lambda_ref * (h/h_ref)^a * (c/c_ref)^b * (v/v_ref)^g

  with ``lambda_ref = 2.0 m``, ``a = 0.75``, ``b = 0.40``, ``g = -0.20``.

Evidence level of the lambda_x coefficients
-------------------------------------------
The coefficients above are **literature-anchored**, not freshly calibrated
against a DEM run in this repository: the repository's DEM / ice-fracture
scenarios (``scenarios/suboff_icebreaking``) carry 24-particle demonstration
domains, far too small to resolve a hull-length pressure correlation scale.
They are anchored to published full-scale / DEM ice-load patch statistics that
place the longitudinal load correlation at ``lambda_x / h ~ 1.5 - 3`` for
level ice (mid-range value used for ``h = 1.0 m``: 2.0 m), with the expected
monotonic trends: thicker ice and higher concentration lengthen the
correlation, higher speed shortens it.

:func:`fit_correlation_length_model` exists so the same functional form can be
re-fitted from DEM-derived pressure records without touching the physics code
(see the unit test that fits synthetic records with known coefficients).  Any
run must therefore report which coefficient provenance it used; the values
consumed by the end-to-end example are marked ``literature_anchored``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .random_field import sample_correlated_field

__all__ = [
    "IceCondition",
    "REFERENCE_CORRELATION_COEFFICIENTS",
    "correlation_length",
    "fit_correlation_length_model",
    "global_ice_pressure",
    "pressure_coefficient_of_variation",
    "contact_patch_length",
    "ice_pressure_profiles",
    "ice_load_scenarios",
]


@dataclass(frozen=True)
class IceCondition:
    """Ice state and transit condition."""

    thickness: float  # m
    concentration: float  # 0..1
    speed: float  # m/s

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError("ice thickness must be positive")
        if not 0.0 < self.concentration <= 1.0:
            raise ValueError("ice concentration must be in (0, 1]")
        if self.speed < 0.0:
            raise ValueError("advance speed must be non-negative")

    def as_dict(self) -> dict:
        return {
            "thickness_m": self.thickness,
            "concentration": self.concentration,
            "speed_m_s": self.speed,
        }


REFERENCE_CORRELATION_COEFFICIENTS = {
    "lambda_ref_m": 2.0,
    "h_ref_m": 1.0,
    "c_ref": 0.8,
    "v_ref_m_s": 2.0,
    "thickness_exponent": 0.75,
    "concentration_exponent": 0.40,
    "speed_exponent": -0.20,
    "provenance": "literature_anchored",
    "provenance_note": (
        "anchored to published ice-load patch statistics (lambda_x/h ~ 1.5-3 for level ice); "
        "not re-calibrated against a DEM run in this repository"
    ),
}


def correlation_length(
    condition: IceCondition, coefficients: dict | None = None
) -> float:
    """Longitudinal pressure correlation length ``lambda_x`` (m)."""
    c = dict(REFERENCE_CORRELATION_COEFFICIENTS)
    if coefficients:
        c.update(coefficients)
    lam = (
        c["lambda_ref_m"]
        * (condition.thickness / c["h_ref_m"]) ** c["thickness_exponent"]
        * (condition.concentration / c["c_ref"]) ** c["concentration_exponent"]
        * (condition.speed / c["v_ref_m_s"]) ** c["speed_exponent"]
    )
    return float(max(lam, 1e-3))


def fit_correlation_length_model(records: list[dict]) -> dict:
    """Log-linear least-squares fit of ``lambda_x = f(h, c, v)`` from records.

    ``records`` items must provide ``thickness``, ``concentration``, ``speed``
    and ``correlation_length``.  Returns the coefficient dictionary in the same
    schema as :data:`REFERENCE_CORRELATION_COEFFICIENTS` (with
    ``provenance = "fitted"``), so it can be fed straight back into
    :func:`correlation_length`.
    """
    if len(records) < 4:
        raise ValueError("at least 4 records are required for the regression")
    h = np.array([r["thickness"] for r in records], dtype=float)
    c = np.array([r["concentration"] for r in records], dtype=float)
    v = np.array([r["speed"] for r in records], dtype=float)
    lam = np.array([r["correlation_length"] for r in records], dtype=float)
    if np.any(h <= 0) or np.any(c <= 0) or np.any(v <= 0) or np.any(lam <= 0):
        raise ValueError("all quantities must be strictly positive")
    # log lam = log lambda_ref + a log(h/h_ref) + b log(c/c_ref) + g log(v/v_ref)
    ref = REFERENCE_CORRELATION_COEFFICIENTS
    design = np.column_stack(
        [
            np.ones_like(h),
            np.log(h / ref["h_ref_m"]),
            np.log(c / ref["c_ref"]),
            np.log(v / ref["v_ref_m_s"]),
        ]
    )
    coef, *_ = np.linalg.lstsq(design, np.log(lam), rcond=None)
    return {
        "lambda_ref_m": float(np.exp(coef[0])),
        "h_ref_m": ref["h_ref_m"],
        "c_ref": ref["c_ref"],
        "v_ref_m_s": ref["v_ref_m_s"],
        "thickness_exponent": float(coef[1]),
        "concentration_exponent": float(coef[2]),
        "speed_exponent": float(coef[3]),
        "provenance": "fitted",
        "n_records": len(records),
    }


def global_ice_pressure(
    condition: IceCondition,
    p_reference: float = 1.2e6,
    thickness_reference: float = 1.0,
    exponent: float = -0.5,
) -> float:
    """Global (whole-ship) ice pressure in Pa, decreasing with contact area."""
    return float(p_reference * (condition.thickness / thickness_reference) ** exponent)


def pressure_coefficient_of_variation(condition: IceCondition | None = None) -> float:
    """Coefficient of variation of the ice pressure field (documented constant)."""
    return 0.30


def contact_patch_length(
    condition: IceCondition, lam: float, length_coefficient: float = 6.0
) -> float:
    """Longitudinal extent of the ice contact patch (m).

    ``L_c = max(c_L * h, 2 * lambda_x)``: the load patch scales with ice
    thickness, floored at two correlation lengths so that the correlated field
    is resolved by the patch (a patch shorter than the correlation length is
    indistinguishable from a uniform load).
    """
    return float(max(length_coefficient * condition.thickness, 2.0 * lam))


def ice_pressure_profiles(
    x: np.ndarray,
    condition: IceCondition,
    lam: float,
    centre: float,
    sigma_ratio: float | None = None,
    n_real: int = 1,
    seed: int | None = None,
    model: str = "exponential",
    coeffs: dict | None = None,
) -> np.ndarray:
    """Realise the correlated ice pressure field over the contact patch.

    Returns an array of shape ``(n_real, x.size)``, zero outside the patch.
    ``centre`` is the longitudinal position of the patch centre.
    """
    x = np.asarray(x, dtype=float)
    mu = global_ice_pressure(condition)
    cov = pressure_coefficient_of_variation() if sigma_ratio is None else float(sigma_ratio)
    patch = contact_patch_length(condition, lam)
    mask = np.abs(x - centre) <= 0.5 * patch
    if not np.any(mask):
        raise ValueError("contact patch does not overlap the beam grid")
    z = sample_correlated_field(
        x, lam, sigma=1.0, n_real=n_real, seed=seed, model=model
    )
    p = mu + cov * mu * z
    p = np.where(mask, p, 0.0)
    # ice pressure cannot be tensile: clip the (rare) negative excursions
    return np.clip(p, 0.0, None)


def ice_load_scenarios(
    x: np.ndarray, condition: IceCondition, lam: float, n_positions: int = 9
) -> list[float]:
    """Patch centre positions scanned along the ship to build the load envelope.

    The patch centre sweeps a symmetric range about midship covering half a
    patch length, which contains the worst-case longitudinal position for a
    symmetric ship (moving the patch from the ends towards midship).
    """
    x = np.asarray(x, dtype=float)
    patch = contact_patch_length(condition, lam)
    half = 0.5 * (x[-1] - x[0])
    span = min(half, 0.5 * (half - 0.5 * patch))
    if span <= 0.0 or n_positions < 1:
        return [0.0]
    return list(np.linspace(-span, span, int(n_positions)))