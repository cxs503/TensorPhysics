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
from .panel_section import StiffenedPanelSection

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

    def panel_section(self) -> StiffenedPanelSection:
        """Full cross-section (elastic **and** plastic, plus torsion).

        Adds the plastic section modulus, the torsional constants and the
        tripping stress on top of :meth:`section_properties`; see
        :mod:`tensorphysics.elastoplastic_collapse.panel_section`.
        """
        return StiffenedPanelSection(
            a=self.a, b=self.b, t=self.t, h_s=self.h_s, t_w=self.t_w,
            material=self.material, b_f=self.b_f, t_f=self.t_f,
        )

    @property
    def sigma_cr_tripping(self) -> float:
        """Elastic tripping stress of the stiffener [Pa] (``inf`` for flat bars)."""
        return self.panel_section().sigma_tripping

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
        """Solve the collapse problem and return the load–displacement curve.

        Implementation note (defect fix, campaign node p167w2-n03)
        -----------------------------------------------------------
        The former implementation combined *constant* smeared formulae (a
        Faulkner effective width evaluated once at σ_y, a Perry-Robertson
        column blend weighted by plate slenderness) with a plastic correction
        that imposed ``ε_yy = 0`` (plane-strain lateral restraint) instead of
        free lateral contraction.  Measured consequences on the documented
        reference panel: ``σ_u = 1.13 σ_y`` for an *elastic-perfectly-plastic*
        steel (a yield-surface violation), no limit point (the load path was
        monotone to 5 % strain), and ``σ_u`` essentially independent of plate
        slenderness (β = 0.84 → 1.13 σ_y, β = 4.68 → 0.98 σ_y).

        The method now delegates to the incremental fiber-based Smith solver
        (:class:`tensorphysics.elastoplastic_collapse.progressive_collapse.
        ProgressiveCollapseSolver`): element load–shortening curves are
        integrated over the discretised cross-section and the curvature is
        obtained from the mid-span column equilibrium ``M = N·δ``, which yields
        a genuine limit point and a post-ultimate branch.

        Returns
        -------
        CollapseResult
            Ultimate strength, failure mode, and full load path.
        """
        from .progressive_collapse import ProgressiveCollapseSolver

        solver = ProgressiveCollapseSolver(
            self.geom,
            n_web_fibers=8,
            n_flange_fibers=2,
            n_steps=max(40, int(self.n_steps)),
            max_strain=float(self.max_strain),
        )
        res = solver.solve()
        # keep the elastic reference stresses consistent with the solver used
        self.sigma_cr_plate = res.sigma_cr_plate
        self.sigma_cr_column = res.sigma_cr_column
        self.progressive_result = res
        self.sigma_cr_tripping = self.geom.sigma_cr_tripping

        # Defect fix (p167w2-n04): the fiber solver only knows plate and column
        # buckling, so 'tripping' -- a failure mode advertised by this class but
        # never returned -- was unreachable.  Tripping governs when the elastic
        # torsional buckling stress is lower than both plate and column buckling.
        mode = res.failure_mode
        if self.sigma_cr_tripping < min(self.sigma_cr_plate, self.sigma_cr_column):
            mode = "tripping"

        return CollapseResult(
            sigma_u=float(res.sigma_u),
            eps_u=float(res.eps_u),
            load_displacement=[(float(e), float(sg)) for e, sg in res.load_displacement],
            failure_mode=mode,
            converged=bool(res.converged),
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
