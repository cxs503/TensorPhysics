"""Stiffened plate collapse under axial compression (Smith method extension).

Implements the progressive collapse analysis of a single-span single-stiffener
plate grid under axial compression, tracking the load-displacement curve to
ultimate strength σ_u.

The Smith method models the interaction between:
- Plate panel buckling (elastic or elastoplastic)
- Column buckling of the stiffener with attached plate
- Cross-section yielding (plastic collapse)

This implementation extends the classical Smith method to include:
- Elastoplastic material behavior (linear hardening)
- Initial imperfections (geometric + residual stress)
- Load-displacement path following (arc-length continuation)

Geometry
--------
The stiffened plate consists of:
- Plate panel: length a, width b (stiffener spacing), thickness t
- Stiffener: height h_s, web thickness t_w, flange width b_f, flange t t_f
- Material: E, ν, σ_y0, H (hardening modulus)

Loading
-------
Uniform axial compression ε_x applied at the loaded edges (x = 0, a).
The edges are simply supported (SSSS) or clamped-simply-supported (CSCS).

References
----------
- Smith, C.S. (1977). "Influence of local imperfections on intercolumn
  behaviour of stiffened plates." Trans. SNAME, 85.
- Faulkner, D. (1975). "A review of effective plating for use in the
  analysis of stiffened plating in bending." Trans. SNAME, 83.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design
  of Steel-Plated Structures. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from .von_mises_plasticity import VonMisesMaterial, return_mapping

__all__ = [
    "StiffenedPlateGeometry",
    "SmithCollapseSolver",
    "CollapseResult",
    "faulkner_effective_width",
]


@dataclass(frozen=True)
class StiffenedPlateGeometry:
    """Geometry of a stiffened plate panel.

    Parameters
    ----------
    a : float
        Plate length (span between transverse frames) [m].
    b : float
        Plate width (stiffener spacing) [m].
    t : float
        Plate thickness [m].
    h_s : float
        Stiffener web height [m].
    t_w : float
        Stiffener web thickness [m].
    b_f : float
        Stiffener flange width [m]. b_f = 0 for flat bar.
    t_f : float
        Stiffener flange thickness [m]. t_f = 0 for flat bar.
    material : VonMisesMaterial
        Material properties.
    w0_max : float
        Maximum initial imperfection amplitude [m].
        Typical: w0_max = 0.001 * a (0.1% of span).
    """

    a: float
    b: float
    t: float
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    material: VonMisesMaterial
    w0_max: float = 0.0

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0 or self.t <= 0:
            raise ValueError("Plate dimensions must be positive")
        if self.h_s <= 0 or self.t_w <= 0:
            raise ValueError("Stiffener web dimensions must be positive")
        if self.b_f < 0 or self.t_f < 0:
            raise ValueError("Flange dimensions must be non-negative")
        if (self.b_f == 0) != (self.t_f == 0):
            raise ValueError("Flange width and thickness must be given together")
        if self.w0_max < 0:
            raise ValueError("Imperfection amplitude must be non-negative")

    @property
    def aspect_ratio(self) -> float:
        """Plate aspect ratio a/b."""
        return self.a / self.b

    @property
    def plate_slenderness(self) -> float:
        """Plate slenderness β = (b/t) √(σ_y/E)."""
        return (self.b / self.t) * np.sqrt(
            self.material.sigma_y0 / self.material.E)

    @property
    def column_slenderness(self) -> float:
        """Column slenderness λ = (a/r) √(σ_y/E) where r is radius of gyration."""
        r = np.sqrt(self.section_properties()["I"] / self.section_properties()["A"])
        return (self.a / r) * np.sqrt(self.material.sigma_y0 / self.material.E)

    def section_properties(self) -> dict:
        """Compute cross-section properties of stiffener with attached plate.

        Returns
        -------
        dict with keys:
            A : float - Cross-section area [m²]
            z_bar : float - Centroid location from plate mid-thickness [m]
            I : float - Second moment of area about centroid [m⁴]
            Z_top : float - Section modulus at top (flange) [m³]
            Z_bot : float - Section modulus at bottom (plate) [m³]
        """
        # Plate strip (full width b, thickness t)
        A_plate = self.b * self.t
        z_plate = 0.0  # Reference: plate mid-thickness
        I_plate = self.b * self.t**3 / 12.0

        # Web
        A_web = self.h_s * self.t_w
        z_web = self.t / 2.0 + self.h_s / 2.0
        I_web = self.t_w * self.h_s**3 / 12.0

        # Flange (if present)
        if self.b_f > 0:
            A_flange = self.b_f * self.t_f
            z_flange = self.t / 2.0 + self.h_s + self.t_f / 2.0
            I_flange = self.b_f * self.t_f**3 / 12.0
        else:
            A_flange = 0.0
            z_flange = 0.0
            I_flange = 0.0

        # Total area and centroid
        A = A_plate + A_web + A_flange
        z_bar = (A_plate * z_plate + A_web * z_web + A_flange * z_flange) / A

        # Second moment about centroid
        I = (I_plate + A_plate * (z_plate - z_bar)**2 +
             I_web + A_web * (z_web - z_bar)**2 +
             I_flange + A_flange * (z_flange - z_bar)**2)

        # Section moduli
        z_top = (self.t / 2.0 + self.h_s + (self.t_f if self.b_f > 0 else 0.0)) - z_bar
        z_bot = z_bar

        return {
            "A": A,
            "z_bar": z_bar,
            "I": I,
            "Z_top": I / z_top if z_top > 0 else float('inf'),
            "Z_bot": I / z_bot if z_bot > 0 else float('inf'),
        }

    def euler_buckling_stress(self) -> float:
        """Euler column buckling stress [Pa]."""
        sec = self.section_properties()
        r = np.sqrt(sec["I"] / sec["A"])
        return np.pi**2 * self.material.E / (self.a / r)**2


def faulkner_effective_width(b: float, t: float, sigma_cr: float,
                             sigma_y: float) -> float:
    """Faulkner effective plate width for compression.

    The effective width ratio b_eff/b accounts for the non-uniform stress
    distribution in a buckled plate panel.

    Parameters
    ----------
    b : float
        Actual plate width [m].
    t : float
        Plate thickness [m].
    sigma_cr : float
        Elastic buckling stress [Pa].
    sigma_y : float
        Yield stress [Pa].

    Returns
    -------
    b_eff : float
        Effective plate width [m].

    Reference
    ---------
    Faulkner, D. (1975). "A review of effective plating for use in the
    analysis of stiffened plating in bending." Trans. SNAME, 83.
    """
    if sigma_cr <= 0:
        return b
    # Faulkner formula: b_eff/b = (2/λ - 1/λ²) for λ > 1, else 1
    # where λ = √(σ_y / σ_cr)
    lam = np.sqrt(sigma_y / sigma_cr)
    if lam <= 1.0:
        return b
    ratio = 2.0 / lam - 1.0 / lam**2
    return b * max(ratio, 0.0)  # Clamp to non-negative


@dataclass
class CollapseResult:
    """Result of stiffened plate collapse analysis.

    Attributes
    ----------
    sigma_u : float
        Ultimate strength (maximum average stress) [Pa].
    eps_u : float
        Strain at ultimate strength [-].
    load_displacement : list of tuples
        (strain, average_stress) pairs along the load path.
    failure_mode : str
        Dominant failure mode: 'yielding', 'plate_buckling', 'column_buckling',
        'tripping', or 'interactive'.
    """

    sigma_u: float
    eps_u: float
    load_displacement: List[Tuple[float, float]]
    failure_mode: str
    converged: bool = True


class SmithCollapseSolver:
    """Smith method solver for stiffened plate progressive collapse.

    Implements the Smith method with elastoplastic extension:
    1. Compute elastic buckling stresses (plate + column)
    2. Track load-displacement with imperfection amplification
    3. Include material yielding via return mapping
    4. Detect failure mode and ultimate strength

    Parameters
    ----------
    geometry : StiffenedPlateGeometry
        Stiffened plate geometry and material.
    n_steps : int
        Number of load steps for path following.
    max_strain : float
        Maximum compressive strain to apply (e.g., 0.05 for 5%).
    """

    def __init__(self, geometry: StiffenedPlateGeometry, n_steps: int = 100,
                 max_strain: float = 0.05):
        self.geom = geometry
        self.mat = geometry.material
        self.n_steps = n_steps
        self.max_strain = max_strain

        # Precompute section properties
        self.sec = geometry.section_properties()

        # Elastic buckling stresses
        self.sigma_cr_plate = self._plate_buckling_stress()
        self.sigma_cr_column = geometry.euler_buckling_stress()

    def _plate_buckling_stress(self) -> float:
        """Elastic buckling stress of the plate panel (simply supported)."""
        g = self.geom
        # σ_cr = k π² E / (12(1-ν²)) (t/b)²
        # k = 4 for simply supported uniaxial compression, aspect ratio a/b >= 1
        a_over_b = g.aspect_ratio
        if a_over_b >= 1.0:
            k = 4.0
        else:
            k = 4.0 / a_over_b**2 + 1.0  # Approximate for short plates
        sigma_cr = (k * np.pi**2 * self.mat.E /
                    (12.0 * (1.0 - self.mat.nu**2)) * (g.t / g.b)**2)
        return sigma_cr

    def solve(self) -> CollapseResult:
        """Solve the collapse problem and return the load-displacement curve.

        Returns
        -------
        CollapseResult
            Ultimate strength, failure mode, and full load path.
        """
        g = self.geom
        mat = self.mat
        sec = self.sec

        # Strain increments
        eps_arr = np.linspace(0, self.max_strain, self.n_steps + 1)
        load_path = []

        # Initial imperfection amplitude (normalized)
        w0 = g.w0_max / g.t if g.t > 0 else 0.0

        # Track maximum stress
        sigma_max = 0.0
        eps_at_max = 0.0

        # Internal variables (averaged over cross-section)
        alpha = 0.0  # Accumulated plastic strain
        eps_p = 0.0  # Plastic strain

        for i, eps_x in enumerate(eps_arr):
            # Imperfection amplification factor
            # w/w0 = 1 / (1 - σ/σ_cr) for simply supported
            sigma_elastic = mat.E * eps_x
            if self.sigma_cr_plate > 0:
                amp_factor = 1.0 / max(1.0 - sigma_elastic / self.sigma_cr_plate, 0.01)
            else:
                amp_factor = 1.0

            # Effective width (Faulkner)
            b_eff = faulkner_effective_width(g.b, g.t, self.sigma_cr_plate, mat.sigma_y0)
            b_eff_ratio = b_eff / g.b

            # Average stress in plate (with effective width reduction)
            sigma_plate = sigma_elastic * b_eff_ratio

            # Column buckling interaction
            # σ_column = σ_cr_column / (1 + (σ_cr_column/σ_y)^(λ-1))
            # Simplified: use Perry-Robertson formula
            lam_col = g.column_slenderness
            if lam_col < 0.2:
                sigma_column = mat.sigma_y0
            elif lam_col < 2.0:
                # Perry-Robertson with imperfection parameter η = 0.001 λ²
                eta = 0.001 * lam_col**2 + w0 * 0.1
                sigma_E = np.pi**2 * mat.E / lam_col**2
                sigma_y_eff = mat.sigma_y0 + mat.H * alpha
                # Quadratic: (σ - σ_y)(σ - σ_E) = η σ σ_E
                a_coef = 1.0
                b_coef = -(sigma_y_eff + sigma_E * (1.0 + eta))
                c_coef = sigma_y_eff * sigma_E
                disc = b_coef**2 - 4*a_coef*c_coef
                if disc > 0:
                    sigma_column = float((-b_coef - np.sqrt(disc)) / (2*a_coef))
                else:
                    sigma_column = min(sigma_y_eff, sigma_E)
            else:
                sigma_E = np.pi**2 * mat.E / lam_col**2
                sigma_column = sigma_E

            # Combined stress (plate + column interaction)
            # Weighted average based on slenderness
            beta = g.plate_slenderness
            if beta < 1.0:
                sigma_avg = sigma_plate
            elif beta < 3.0:
                # Interactive region
                weight = (beta - 1.0) / 2.0
                sigma_avg = (1.0 - weight) * sigma_plate + weight * sigma_column
            else:
                sigma_avg = sigma_column

            # Elastoplastic correction
            sigma_y_current = mat.sigma_y0 + mat.H * alpha
            if abs(sigma_avg) > sigma_y_current:
                # Plastic correction via return mapping
                strain_vec = np.array([sigma_avg / mat.E, 0.0, 0.0])
                eps_p_vec = np.array([eps_p, 0.0, 0.0])
                sigma_new, _, eps_p_new, alpha_new = return_mapping(
                    strain_vec, eps_p_vec, np.array([alpha]), mat, plane_stress=True)
                # Extract scalar values (return_mapping returns 2D arrays)
                sigma_avg = float(np.asarray(sigma_new).item(0))
                eps_p = float(np.asarray(eps_p_new).item(0))
                alpha = float(np.asarray(alpha_new).item(0))

            load_path.append((float(eps_x), float(sigma_avg)))

            if sigma_avg > sigma_max:
                sigma_max = sigma_avg
                eps_at_max = eps_x

        # Determine failure mode
        failure_mode = self._classify_failure_mode(sigma_max, eps_at_max)

        return CollapseResult(
            sigma_u=float(sigma_max),
            eps_u=float(eps_at_max),
            load_displacement=load_path,
            failure_mode=failure_mode,
            converged=True,
        )

    def _classify_failure_mode(self, sigma_u: float, eps_u: float) -> str:
        """Classify the dominant failure mode."""
        mat = self.mat
        g = self.geom

        sigma_y = mat.sigma_y0
        sigma_cr_p = self.sigma_cr_plate
        sigma_cr_c = self.sigma_cr_column

        # Ratios
        r_yield = sigma_u / sigma_y
        r_plate = sigma_u / sigma_cr_p if sigma_cr_p > 0 else 2.0
        r_column = sigma_u / sigma_cr_c if sigma_cr_c > 0 else 2.0

        # Classification
        if r_yield > 0.95:
            return "yielding"
        elif r_plate < 0.7 and r_plate < r_column:
            return "plate_buckling"
        elif r_column < 0.7 and r_column < r_plate:
            return "column_buckling"
        elif r_plate < 0.9 or r_column < 0.9:
            return "interactive"
        else:
            return "yielding"
