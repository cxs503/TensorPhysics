"""Flexural fracture of an ice plate: stress fields, criterion, crack initiation.

Stress recovery (Kirchhoff-Love, ``w > 0`` downwards, tension positive)
----------------------------------------------------------------------
    sigma_xx = -C (w_,xx + nu w_,yy),     sigma_yy = -C (w_,yy + nu w_,xx),
    tau_xy   = -C (1 - nu) w_,xy,          C = E h / (2 (1 - nu^2)),

for the *bottom* surface (``z = -h/2``); the top surface carries the opposite
sign.  A sagging plate (downward load) therefore has its bottom fibres in
tension beneath the load, which is where the first radial (star) cracks appear;
further out the plate lifts, the curvature reverses and the *top* surface goes
into tension, which is where ring (circumferential) cracks form.

Fracture criterion
------------------
Flexural fracture is assumed when the maximum principal bottom-surface stress
reaches the ice flexural strength:

    max( sigma_1 ) >= sigma_f        (sigma_f ~ 0.5 - 0.8 MPa, first-year ice).

Crack initiation location and time are the location/time of the first
exceedance; the fracture energy of the newly created crack is
``G_c * h * L_crack`` with ``G_c = K_Ic^2 / E ~ 1-3 J/m^2`` for fresh ice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StressField",
    "bottom_stress_field",
    "top_stress_field",
    "principal_stress",
    "max_tensile_stress",
    "crack_initiation",
    "ring_crack_radius",
    "fracture_energy",
]


@dataclass
class StressField:
    """Bottom-surface stress state on the plate grid [Pa]."""

    sigma_xx: np.ndarray
    sigma_yy: np.ndarray
    tau_xy: np.ndarray


def bottom_stress_field(derivs: dict, E: float, h: float, nu: float = 0.3) -> StressField:
    """Bottom-surface stress field from the second derivatives of ``w``."""
    C = -E * h / (2.0 * (1.0 - nu**2))   # sign for w > 0 downwards
    return StressField(
        sigma_xx=C * (derivs["xx"] + nu * derivs["yy"]),
        sigma_yy=C * (derivs["yy"] + nu * derivs["xx"]),
        tau_xy=C * (1.0 - nu) * derivs["xy"],
    )


def top_stress_field(derivs: dict, E: float, h: float, nu: float = 0.3) -> StressField:
    """Top-surface stress field (opposite sign of the bottom surface)."""
    b = bottom_stress_field(derivs, E, h, nu)
    return StressField(-b.sigma_xx, -b.sigma_yy, -b.tau_xy)


def principal_stress(field: StressField) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal stresses ``(sigma_1 >= sigma_2)`` and the major-axis angle."""
    sx, sy, txy = field.sigma_xx, field.sigma_yy, field.tau_xy
    mean = 0.5 * (sx + sy)
    radius = np.sqrt((0.5 * (sx - sy)) ** 2 + txy**2)
    s1 = mean + radius
    s2 = mean - radius
    theta = 0.5 * np.arctan2(2.0 * txy, sx - sy)
    return s1, s2, theta


def max_tensile_stress(field: StressField) -> np.ndarray:
    """Maximum principal (most tensile) stress field [Pa]."""
    s1, _, _ = principal_stress(field)
    return s1


def crack_initiation(
    sigma_max: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    strength: float,
    contact_xy: tuple[float, float] = (0.0, 0.0),
    characteristic_length: float | None = None,
    time: float | None = None,
) -> dict:
    """Locate the first flexural failure and report its position/time.

    Returns a dictionary with the peak stress, its grid position, the radial
    distance from the contact centre (normalised by the characteristic length
    when supplied), and a boolean ``triggered``.
    """
    s = np.asarray(sigma_max, dtype=float)
    iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
    peak = float(s[iy, ix])
    xc, yc = contact_xy
    r = float(np.hypot(x[iy, ix] - xc, y[iy, ix] - yc))
    out = {
        "triggered": bool(peak >= strength),
        "strength_Pa": float(strength),
        "peak_tensile_stress_Pa": peak,
        "stress_ratio": float(peak / strength),
        "x_m": float(x[iy, ix]),
        "y_m": float(y[iy, ix]),
        "radius_from_contact_m": r,
    }
    if characteristic_length:
        out["radius_over_lc"] = float(r / characteristic_length)
    if time is not None:
        out["time_s"] = float(time)
    return out


def ring_crack_radius(
    sigma_top_max: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    contact_xy: tuple[float, float] = (0.0, 0.0),
    inner_radius: float = 0.0,
    characteristic_length: float | None = None,
) -> dict:
    """Radius of the outermost top-surface tension peak (ring crack site).

    Ring (circumferential) cracks open where the *top* fibres reach the flexural
    strength far from the contact, i.e. at the outer local maximum of the
    top-surface tensile stress.
    """
    s = np.asarray(sigma_top_max, dtype=float)
    X, Y = np.asarray(x), np.asarray(y)
    R = np.hypot(X - contact_xy[0], Y - contact_xy[1])
    mask = R > inner_radius
    if not np.any(mask):
        raise ValueError("no grid points outside inner_radius")
    s_masked = np.where(mask, s, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(s_masked)), s.shape)
    r = float(R[iy, ix])
    out = {
        "ring_radius_m": r,
        "ring_peak_tensile_stress_Pa": float(s[iy, ix]),
        "x_m": float(X[iy, ix]),
        "y_m": float(Y[iy, ix]),
    }
    if characteristic_length:
        out["ring_radius_over_lc"] = float(r / characteristic_length)
    return out


def fracture_energy(
    crack_length: float,
    ice_thickness: float,
    critical_stress_intensity: float = 1.2e5,
    young_modulus: float = 9.0e9,
) -> dict:
    """Energy released by creating one through-thickness crack [J].

    ``G_c = K_Ic^2 / E`` (~1.6 J/m^2 for the default fresh-ice values).
    """
    if crack_length <= 0.0 or ice_thickness <= 0.0:
        raise ValueError("crack_length and ice_thickness must be positive")
    Gc = critical_stress_intensity**2 / young_modulus
    area = crack_length * ice_thickness
    return {
        "Gc_J_m2": float(Gc),
        "crack_area_m2": float(area),
        "energy_J": float(Gc * area),
    }