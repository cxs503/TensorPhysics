"""Tripping (lateral-torsional buckling) of stiffeners.

Implements the analysis of stiffener tripping, which is the lateral-torsional
buckling mode where the stiffener rotates about its attachment to the plate.

Tripping is a critical failure mode for stiffened panels under compression,
especially for:
- Bulb flats and flat bars (weak torsional rigidity)
- Slender stiffeners (high h_s/t_w ratio)
- Panels with large stiffener spacing

The tripping capacity depends on:
- Stiffener geometry (web, flange, attached plate)
- Material properties (E, σ_y)
- Lateral restraint from the plate (characterized by λ_trip)
- Length between lateral supports

Parameter λ_trip
----------------
The parameter λ_trip characterizes the lateral restraint provided by the plate:
- λ_trip = 1.0: Full lateral restraint (plate provides complete fixity)
- λ_trip = 2.5: Reduced lateral restraint (plate flexibility)

As λ_trip increases, the tripping capacity decreases and the failure mode
may shift from tripping to other modes (column buckling, yielding).

References
----------
- Hughes, O.F. (1988). "Ship Structure Design: A Rationale for Naval
  Architects, Marine Engineers, and Structural Engineers." SNAME.
- DNV-GL (2017). Recommended Practice DNV-RP-C201: Buckling of Ships.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design
  of Steel-Plated Structures. Wiley, Chapter 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

from .von_mises_plasticity import VonMisesMaterial

__all__ = [
    "TrippingAnalysis",
    "TrippingResult",
    "tripping_capacity",
]


@dataclass(frozen=True)
class TrippingResult:
    """Result of tripping analysis.

    Attributes
    ----------
    sigma_trip : float
        Tripping (critical) stress [Pa].
    lambda_trip : float
        Lateral restraint parameter used.
    failure_mode : str
        'tripping', 'column_buckling', 'yielding', or 'interactive'.
    capacity_reduction : float
        Ratio σ_trip / σ_y (capacity utilization).
    """

    sigma_trip: float
    lambda_trip: float
    failure_mode: str
    capacity_reduction: float


@dataclass
class TrippingAnalysis:
    """Analysis of stiffener tripping (lateral-torsional buckling).

    Parameters
    ----------
    a : float
        Stiffener span (length between lateral supports) [m].
    b : float
        Stiffener spacing (plate width) [m].
    t : float
        Plate thickness [m].
    h_s : float
        Stiffener web height [m].
    t_w : float
        Stiffener web thickness [m].
    b_f : float
        Stiffener flange width [m].
    t_f : float
        Stiffener flange thickness [m].
    material : VonMisesMaterial
        Material properties.
    """

    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    material: VonMisesMaterial

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("Panel dimensions must be positive")
        if self.h_s <= 0 or self.t_w <= 0:
            raise ValueError("Stiffener web dimensions must be positive")

    def section_properties(self) -> Dict[str, float]:
        """Compute section properties for tripping analysis.

        Returns
        -------
        dict with keys:
            I_z : float - Second moment of area about z-axis (lateral) [m⁴]
            I_w : float - Warping constant [m⁶]
            J : float - Saint-Venant torsion constant [m⁴]
            z_s : float - Shear center location from plate [m]
        """
        # Web contribution
        I_z_web = self.t_w * self.h_s**3 / 12.0
        J_web = self.h_s * self.t_w**3 / 3.0

        # Flange contribution (if present)
        if self.b_f > 0:
            I_z_flange = self.t_f * self.b_f**3 / 12.0
            # Flange about its own centroid + transfer to web tip
            I_z_flange += self.b_f * self.t_f * (self.h_s / 2.0)**2
            J_flange = self.b_f * self.t_f**3 / 3.0
        else:
            I_z_flange = 0.0
            J_flange = 0.0

        # Plate strip contribution (attached plate, width = spacing / 2 each side)
        b_plate = self.b / 2.0  # Half spacing each side
        I_z_plate = self.t * b_plate**3 / 12.0
        J_plate = b_plate * self.t**3 / 3.0

        # Total
        I_z = I_z_web + I_z_flange + I_z_plate
        J = J_web + J_flange + J_plate

        # Warping constant (approximate for T-section)
        # I_w ≈ I_z_flange * h_s² for T-section
        if self.b_f > 0:
            I_w = self.t_f * self.b_f**3 / 12.0 * (self.h_s + self.t_f / 2.0)**2
        else:
            I_w = 0.0

        # Shear center (approximate)
        z_s = self.h_s / 2.0  # Mid-web for symmetric sections

        return {
            "I_z": I_z,
            "I_w": I_w,
            "J": J,
            "z_s": z_s,
        }

    def elastic_tripping_stress(self, lambda_trip: float = 1.0) -> float:
        """Compute elastic tripping stress.

        Parameters
        ----------
        lambda_trip : float
            Lateral restraint parameter (1.0 = full fixity, 2.5 = reduced).

        Returns
        -------
        sigma_cr_trip : float
            Elastic tripping stress [Pa].
        """
        mat = self.material
        sec = self.section_properties()

        G = mat.shear_modulus
        E = mat.E

        # Torsional buckling stress
        # σ_T = (1/I_z) [G J + π² E I_w / a²]
        I_z = sec["I_z"]
        J = sec["J"]
        I_w = sec["I_w"]

        if I_z > 0:
            sigma_T = (G * J + np.pi**2 * E * I_w / self.a**2) / I_z
        else:
            sigma_T = 0.0

        # Lateral buckling stress (column buckling about weak axis)
        # σ_L = π² E I_z / (a² A)
        A = self.b * self.t + self.h_s * self.t_w
        if self.b_f > 0:
            A += self.b_f * self.t_f

        if A > 0 and I_z > 0:
            sigma_L = np.pi**2 * E * I_z / (self.a**2 * A)
        else:
            sigma_L = 0.0

        # Combined tripping stress (interaction)
        # 1/σ_trip = 1/σ_T + 1/σ_L (approximate)
        if sigma_T > 0 and sigma_L > 0:
            sigma_cr_trip = 1.0 / (1.0 / sigma_T + 1.0 / sigma_L)
        elif sigma_T > 0:
            sigma_cr_trip = sigma_T
        elif sigma_L > 0:
            sigma_cr_trip = sigma_L
        else:
            sigma_cr_trip = 0.0

        # Apply lateral restraint factor
        # λ_trip > 1 reduces the tripping capacity
        sigma_cr_trip /= lambda_trip

        return sigma_cr_trip

    def tripping_capacity(self, lambda_trip: float = 1.0) -> TrippingResult:
        """Compute tripping capacity with failure mode classification.

        Parameters
        ----------
        lambda_trip : float
            Lateral restraint parameter.

        Returns
        -------
        TrippingResult
            Tripping stress, failure mode, and capacity utilization.
        """
        mat = self.material
        sigma_y = mat.sigma_y0

        # Elastic tripping stress
        sigma_cr_trip = self.elastic_tripping_stress(lambda_trip)

        # Column buckling stress (strong axis)
        sec = self.section_properties()
        A = self.b * self.t + self.h_s * self.t_w
        if self.b_f > 0:
            A += self.b_f * self.t_f

        # Strong axis moment of inertia
        I_x = (self.b * self.t**3 / 12.0 +
               self.t_w * self.h_s**3 / 12.0 +
               self.h_s * self.t_w * (self.t / 2.0 + self.h_s / 2.0)**2)
        if self.b_f > 0:
            I_x += (self.b_f * self.t_f**3 / 12.0 +
                    self.b_f * self.t_f * (self.t / 2.0 + self.h_s + self.t_f / 2.0)**2)

        r_x = np.sqrt(I_x / A) if A > 0 else 0.0
        lam_x = self.a / r_x if r_x > 0 else 100.0
        sigma_cr_col = np.pi**2 * mat.E / lam_x**2 if lam_x > 0 else 0.0

        # Determine tripping stress (elastoplastic correction)
        lam_trip = np.sqrt(sigma_y / sigma_cr_trip) if sigma_cr_trip > 0 else 100.0

        if lam_trip < 0.5:
            # Stocky: yielding governs
            sigma_trip = sigma_y
            failure_mode = "yielding"
        elif lam_trip < 1.5:
            # Intermediate: elastoplastic interaction
            # Perry-Robertson type formula
            sigma_E = sigma_cr_trip
            eta = 0.001 * lam_trip**2
            a_coef = 1.0
            b_coef = -(sigma_y + sigma_E * (1.0 + eta))
            c_coef = sigma_y * sigma_E
            disc = b_coef**2 - 4*a_coef*c_coef
            if disc > 0:
                sigma_trip = (-b_coef - np.sqrt(disc)) / (2*a_coef)
            else:
                sigma_trip = min(sigma_y, sigma_E)
            failure_mode = "tripping"
        else:
            # Slender: elastic buckling
            sigma_trip = sigma_cr_trip
            failure_mode = "tripping"

        # Check column buckling interaction
        lam_col = np.sqrt(sigma_y / sigma_cr_col) if sigma_cr_col > 0 else 100.0
        if lam_col < lam_trip:
            # Column buckling may govern
            if lam_col < 0.5:
                sigma_col = sigma_y
            else:
                sigma_E = sigma_cr_col
                sigma_col = sigma_E
            if sigma_col < sigma_trip:
                sigma_trip = sigma_col
                failure_mode = "column_buckling"

        # Capacity reduction
        capacity_reduction = sigma_trip / sigma_y if sigma_y > 0 else 0.0

        return TrippingResult(
            sigma_trip=sigma_trip,
            lambda_trip=lambda_trip,
            failure_mode=failure_mode,
            capacity_reduction=capacity_reduction,
        )


def tripping_capacity(
    a: float,
    b: float,
    t: float,
    h_s: float,
    t_w: float,
    b_f: float,
    t_f: float,
    material: VonMisesMaterial,
    lambda_trip: float = 1.0,
) -> TrippingResult:
    """Convenience function for tripping capacity analysis.

    Parameters
    ----------
    a, b, t : float
        Panel dimensions [m].
    h_s, t_w, b_f, t_f : float
        Stiffener dimensions [m].
    material : VonMisesMaterial
        Material properties.
    lambda_trip : float
        Lateral restraint parameter.

    Returns
    -------
    TrippingResult
        Tripping analysis result.
    """
    analysis = TrippingAnalysis(
        a=a, b=b, t=t,
        h_s=h_s, t_w=t_w, b_f=b_f, t_f=t_f,
        material=material,
    )
    return analysis.tripping_capacity(lambda_trip)
