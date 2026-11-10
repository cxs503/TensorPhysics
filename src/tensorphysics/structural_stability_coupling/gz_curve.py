"""Righting-arm (GZ) curve computation with structural degradation support.

Physics background
------------------
The righting arm GZ(φ) is the transverse distance between the centre of
gravity G and the line of action of buoyancy at heel angle φ.  For a
wall-sided vessel at small angles::

    GZ(φ) ≈ GM · sin(φ)

For larger angles the full waterplane geometry must be integrated.  This
module provides both the simple wall-sided formula and a numerical
waterplane-integration method that accounts for hull-form effects and
structural degradation of the effective waterplane.

The structural degradation enters through the *effective* waterplane
second moment I_wp(φ, D), where D is a damage parameter (0 = intact,
1 = fully yielded).  Yielded plate panels lose their contribution to
the hull-girder section modulus, shifting the neutral axis and reducing
the effective waterplane area moment.

References
----------
- Lewis, E.V. (ed.), *Principles of Naval Architecture*, SNAME, 1988.
- DNV-Stable W21, Stability and intact stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "GZCurveResult",
    "compute_gz_curve",
    "wall_sided_gz",
    "HullFormData",
]


@dataclass(frozen=True)
class HullFormData:
    """Simplified hull-form description for GZ computation.

    Parameters
    ----------
    L, B, T :
        Length, breadth, draught (m).
    Cb :
        Block coefficient (-).
    Cwp :
        Waterplane area coefficient (-).
    KG :
        Vertical centre of gravity above keel (m).
    KB :
        Vertical centre of buoyancy above keel (m).
    BM_T :
        Transverse metacentric radius BM = I_T / ∇ (m).
    LCG, TCG :
        Longitudinal and transverse centre of gravity (m).
    """
    L: float
    B: float
    T: float
    Cb: float = 0.65
    Cwp: float = 0.72
    KG: float = 5.0
    KB: float = 2.5
    BM_T: float = 4.0
    LCG: float = 0.0
    TCG: float = 0.0

    @property
    def nabla(self) -> float:
        """Displaced volume ∇ (m³)."""
        return self.L * self.B * self.T * self.Cb

    @property
    def GM(self) -> float:
        """Initial transverse metacentric height (m)."""
        return self.KB + self.BM_T - self.KG

    @property
    def I_T(self) -> float:
        """Transverse waterplane second moment (m⁴)."""
        return self.BM_T * self.nabla

    @property
    def A_wp(self) -> float:
        """Waterplane area (m²)."""
        return self.L * self.B * self.Cwp


@dataclass
class GZCurveResult:
    """Result of a GZ curve computation."""
    angles_deg: np.ndarray
    gz_m: np.ndarray
    gz_corrected_m: Optional[np.ndarray]
    GM_m: float
    max_gz_m: float
    max_gz_angle_deg: float
    range_deg: float
    method: str
    degradation_applied: bool = False

    def as_dict(self) -> dict:
        d = {
            "angles_deg": self.angles_deg.tolist(),
            "gz_m": self.gz_m.tolist(),
            "GM_m": self.GM_m,
            "max_gz_m": self.max_gz_m,
            "max_gz_angle_deg": self.max_gz_angle_deg,
            "range_deg": self.range_deg,
            "method": self.method,
            "degradation_applied": self.degradation_applied,
        }
        if self.gz_corrected_m is not None:
            d["gz_corrected_m"] = self.gz_corrected_m.tolist()
        return d


def wall_sided_gz(phi_deg: float, GM: float) -> float:
    """Wall-sided formula: GZ = GM · sin(φ).

    Valid for small to moderate angles (φ < ~15°) with conventional hull forms.
    """
    return GM * math.sin(math.radians(phi_deg))


def _waterplane_half_breadths(
    hull: HullFormData,
    phi_rad: float,
    n_stations: int = 21,
    n_waterlines: int = 11,
    damage_factor: float = 0.0,
) -> tuple:
    """Compute offset table with optional structural damage.

    Returns (y_offsets, z_offsets) arrays for the waterplane at heel φ.
    The damage_factor reduces the effective waterplane second moment by
    simulating loss of deck/bottom plate contribution to hull girder.

    Simplified model: elliptical waterplane shape modified by heel.
    """
    # Station positions along length
    xi = np.linspace(-hull.L / 2, hull.L / 2, n_stations)
    # Half-breadth at each station (elliptical approximation)
    y_max = (hull.B / 2) * np.sqrt(np.maximum(0, 1 - (2 * xi / hull.L) ** 2))
    # Apply waterplane coefficient correction
    y_max *= math.sqrt(hull.Cwp)

    # Heel effect: one side emerges, other submerges
    # Transverse shift of waterline due to heel
    dz_heel = np.tan(phi_rad) * (hull.B / 2)

    # Effective waterplane breadth modification due to heel
    # Emerged side loses area, immersed side gains
    y_port = y_max + dz_heel  # immersed side
    y_star = y_max - dz_heel  # emerged side

    # Apply structural damage: reduce effective breadth on damaged side
    # Damage reduces the contribution of yielded plates to waterplane inertia
    if damage_factor > 0:
        # Damage primarily affects deck-side plates (tension side)
        y_star = y_star * (1 - 0.3 * damage_factor)
        y_port = y_port * (1 - 0.1 * damage_factor)

    # Ensure non-negative
    y_port = np.maximum(y_port, 0.0)
    y_star = np.maximum(y_star, 0.0)

    return xi, y_port, y_star


def _compute_gz_numerical(
    hull: HullFormData,
    phi_deg: float,
    damage_factor: float = 0.0,
    n_stations: int = 21,
) -> float:
    """Numerical GZ computation by waterplane integration at heel φ.

    Uses the method of integrating the shifted waterplane to find the
    new centre of buoyancy position, then computing GZ from geometry.

    GZ = (y_B - y_G) · cos(φ) + (z_B - z_G) · sin(φ)

    where y_B, z_B are the shifted buoyancy centre coordinates.
    """
    phi_rad = math.radians(phi_deg)

    if abs(phi_deg) < 1e-10:
        return 0.0

    xi, y_port, y_star = _waterplane_half_breadths(
        hull, phi_rad, n_stations, damage_factor=damage_factor
    )

    dx = xi[1] - xi[0] if len(xi) > 1 else hull.L / 20

    # Waterplane area properties at heel
    # Total waterplane breadth at each station
    y_total = y_port + y_star
    # Shift of waterplane centroid from centreline
    y_wp_shift = 0.5 * (y_port - y_star)  # asymmetric due to heel

    # Waterplane area (trapezoidal integration)
    A_wp = np.trapezoid(y_total, xi)

    # First moment of waterplane area about centreline (transverse shift)
    # This gives the shift in buoyancy due to the wedge immersion/emersion
    M_y = np.trapezoid(y_wp_shift * y_total * 0.5, xi)

    # Shift in centre of buoyancy
    # δy_B = M_y / ∇ (transverse shift of B due to heel)
    nabla = hull.nabla
    if nabla <= 0:
        return 0.0

    delta_y_B = M_y / nabla

    # Vertical shift of B due to heel (wedge transfer)
    # δz_B ≈ (1/∇) ∫ (1/2) · (y_shift²) dx · tan(φ)
    # Simplified: use the waterplane second moment
    I_T_eff = np.trapezoid(y_total ** 3 / 12.0, xi)
    delta_z_B = I_T_eff * math.tan(phi_rad) / (2.0 * nabla)

    # BM at this heel angle
    BM_eff = I_T_eff / nabla

    # GZ computation
    # GZ = (KB + δz_B + BM_eff · cos(φ) - KG) · sin(φ) + (δy_B - TCG) · cos(φ)
    # Simplified wall-sided with corrections:
    y_G = hull.TCG
    z_G = hull.KG

    gz = ((hull.KB + delta_z_B - z_G) * math.sin(phi_rad)
          + (delta_y_B - y_G) * math.cos(phi_rad))

    return gz


def compute_gz_curve(
    hull: HullFormData,
    angles_deg: Optional[Sequence[float]] = None,
    damage_factor: float = 0.0,
    method: str = "numerical",
) -> GZCurveResult:
    """Compute the GZ curve for a range of heel angles.

    Parameters
    ----------
    hull :
        Hull form data.
    angles_deg :
        Heel angles in degrees.  Default: 0 to 90 in 5° steps.
    damage_factor :
        Structural damage factor [0, 1].  0 = intact, 1 = fully yielded.
    method :
        'wall_sided' for simple GM·sin(φ), 'numerical' for waterplane
        integration.

    Returns
    -------
    GZCurveResult
    """
    if angles_deg is None:
        angles_deg = np.arange(0, 91, 2.5)
    angles = np.asarray(angles_deg, dtype=float)

    gz = np.zeros_like(angles)
    for i, phi in enumerate(angles):
        if method == "wall_sided":
            gz[i] = wall_sided_gz(phi, hull.GM)
        else:
            gz[i] = _compute_gz_numerical(hull, phi, damage_factor)

    # Find maximum GZ and range
    max_idx = np.argmax(gz)
    max_gz = gz[max_idx]
    max_angle = angles[max_idx]

    # Range: angle where GZ returns to zero (after the peak)
    range_angle = angles[-1]
    for i in range(max_idx + 1, len(angles)):
        if gz[i] <= 0:
            range_angle = angles[i]
            break

    return GZCurveResult(
        angles_deg=angles,
        gz_m=gz,
        gz_corrected_m=None,
        GM_m=hull.GM,
        max_gz_m=float(max_gz),
        max_gz_angle_deg=float(max_angle),
        range_deg=float(range_angle),
        method=method,
        degradation_applied=damage_factor > 0,
    )
