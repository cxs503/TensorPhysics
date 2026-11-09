"""Elastoplastic analysis of plate-beam combination under ice load.

Analysis stages
---------------
1. **Elastic stage**: Both plate and beam respond elastically.
   - Plate deflection: w = alpha * p * b^4 / (D * beta^4) (plate theory)
   - Beam deflection: w = 5 * q * L^4 / (384 * EI) (beam theory)
   - Combined stiffness from compatibility

2. **Plate yield stage**: Plate reaches yield stress first (thinner element).
   - Plate forms plastic hinge lines along edges
   - Plate carries additional load through membrane action
   - Beam still elastic

3. **Plastic hinge stage**: Plate fully plastic, beam approaching yield.
   - Plate ultimate load via yield line theory
   - Beam moment increasing toward M_pl

4. **Beam yield stage**: Beam reaches plastic moment capacity.
   - Plastic hinge forms at beam midspan
   - Mechanism forms: plate plastic + beam plastic hinge

5. **Collapse stage**: Full mechanism, load-carrying capacity drops.
   - Progressive collapse of the combined system

Load-displacement curve
-----------------------
The analysis produces a load-displacement curve with:
- Linear elastic region
- Yield plateau (plate yielding)
- Hardening region (beam approaching plastic)
- Peak load (ultimate capacity)
- Post-peak softening (collapse)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

from .plate_beam import PlateBeamSection


@dataclass(frozen=True)
class ElasticPlateBeam:
    """Elastic response of plate-beam combination.

    The plate and beam share the load through compatibility of deflection.
    For a uniformly loaded panel:
    - Plate carries load directly to stiffeners
    - Stiffener (beam) carries load from plate to supports

    Simplified model: plate acts as distributed load on beam.
    """
    section: PlateBeamSection

    @property
    def L(self) -> float:
        """Beam span (frame spacing) [m]."""
        return self.section.plate.a

    @property
    def s(self) -> float:
        """Stiffener spacing [m]."""
        return self.section.plate.b

    @property
    def EI(self) -> float:
        return self.section.EI

    @property
    def D_plate(self) -> float:
        return self.section.plate_D

    def beam_deflection(self, p: float) -> float:
        """Maximum beam deflection under uniform pressure p [m].

        For simply supported beam with uniform load q = p * s:
        w_max = 5 * q * L^4 / (384 * EI)
        """
        q = p * self.s  # line load [N/m]
        return 5.0 * q * self.L**4 / (384.0 * self.EI)

    def plate_deflection(self, p: float) -> float:
        """Maximum plate deflection under uniform pressure p [m].

        For simply supported rectangular plate (Timoshenko):
        w_max = alpha * p * b^4 / D
        where alpha depends on aspect ratio a/b.
        """
        a_over_b = self.section.plate.aspect_ratio
        # Coefficient for simply supported plate (Timoshenko & Woinowsky-Krieger)
        if a_over_b >= 2.0:
            alpha = 0.0130  # long plate approximation
        elif a_over_b >= 1.5:
            alpha = 0.0122
        elif a_over_b >= 1.0:
            alpha = 0.00406 + 0.00894 * (a_over_b - 1.0)
        else:
            alpha = 0.00406  # square plate

        return alpha * p * self.s**4 / self.D_plate

    def combined_stiffness(self) -> float:
        """Combined elastic stiffness K = P / w [N/m].

        For unit pressure p = 1 Pa:
        - Beam deflection w_b
        - Plate deflection w_p
        - Combined: w = w_b (beam governs for stiffened panels)
        - K = (p * A) / w where A = L * s
        """
        p_test = 1e4  # 10 kPa test pressure
        w_beam = self.beam_deflection(p_test)
        w_plate = self.plate_deflection(p_test)

        # Combined deflection (series spring model)
        # The plate and beam deflect together, beam governs
        w_combined = w_beam  # beam is the flexible element

        # Total force for p_test
        F = p_test * self.L * self.s

        K = F / w_combined if w_combined > 0 else float('inf')
        return K

    def plate_stress(self, p: float) -> float:
        """Maximum plate bending stress [Pa].

        sigma_max = beta * p * (s/t_p)^2
        where beta depends on boundary condition and aspect ratio.
        """
        a_over_b = self.section.plate.aspect_ratio
        # Bending coefficient for simply supported plate
        if a_over_b >= 2.0:
            beta = 0.5  # long plate, clamped edges
        elif a_over_b >= 1.5:
            beta = 0.48
        elif a_over_b >= 1.0:
            beta = 0.30 + 0.18 * (a_over_b - 1.0)
        else:
            beta = 0.30  # square plate

        return beta * p * (self.s / self.section.plate.t_p)**2

    def beam_stress(self, p: float) -> float:
        """Maximum beam bending stress [Pa].

        sigma_max = M_max / Z_el
        M_max = q * L^2 / 8 = p * s * L^2 / 8
        """
        q = p * self.s
        M_max = q * self.L**2 / 8.0
        return M_max / self.section.Z_el

    def pressure_at_plate_yield(self) -> float:
        """Pressure at which plate first yields [Pa]."""
        sigma_plate = self.section.sigma_y
        a_over_b = self.section.plate.aspect_ratio
        if a_over_b >= 2.0:
            beta = 0.5
        elif a_over_b >= 1.5:
            beta = 0.48
        elif a_over_b >= 1.0:
            beta = 0.30 + 0.18 * (a_over_b - 1.0)
        else:
            beta = 0.30

        return sigma_plate / (beta * (self.s / self.section.plate.t_p)**2)

    def pressure_at_beam_yield(self) -> float:
        """Pressure at which beam first yields [Pa]."""
        M_el = self.section.M_el
        q_yield = 8.0 * M_el / self.L**2
        return q_yield / self.s

    def deflection_at_yield(self) -> float:
        """Deflection at first yield [m]."""
        p_plate = self.pressure_at_plate_yield()
        p_beam = self.pressure_at_beam_yield()
        p_yield = min(p_plate, p_beam)
        K = self.combined_stiffness()
        return p_yield * self.L * self.s / K


@dataclass
class ElastoplasticResponse:
    """Full elastoplastic load-displacement response.

    Attributes
    ----------
    displacements : array
        Deflection values [m]
    loads : array
        Total load values [N]
    pressures : array
        Pressure values [Pa]
    stages : list of str
        Response stage at each point
    plate_yielded : array of bool
        Whether plate has yielded
    beam_yielded : array of bool
        Whether beam has yielded
    """
    displacements: np.ndarray
    loads: np.ndarray
    pressures: np.ndarray
    stages: List[str]
    plate_yielded: np.ndarray
    beam_yielded: np.ndarray

    @property
    def P_yield(self) -> float:
        """Yield load (first yield, plate or beam) [N]."""
        idx = np.where(self.plate_yielded | self.beam_yielded)[0]
        if len(idx) == 0:
            return self.loads[-1]
        return float(self.loads[idx[0]])

    @property
    def P_ult(self) -> float:
        """Ultimate (peak) load [N]."""
        return float(np.max(self.loads))

    @property
    def w_at_P_ult(self) -> float:
        """Deflection at ultimate load [m]."""
        idx = np.argmax(self.loads)
        return float(self.displacements[idx])

    @property
    def K_elastic(self) -> float:
        """Initial elastic stiffness [N/m].

        Computed from points that are definitely in the elastic region
        (before any yielding occurs). Falls back to analytical stiffness
        if no elastic points are available.
        """
        # Find first yield index
        yield_idx = len(self.displacements)
        for i in range(len(self.displacements)):
            if self.plate_yielded[i] or self.beam_yielded[i]:
                yield_idx = i
                break

        # Use points before yield (need at least 2 non-zero points)
        if yield_idx >= 3:
            w = self.displacements[1:yield_idx]
            P = self.loads[1:yield_idx]
            if len(w) > 0 and w[-1] > 0:
                return float(np.sum(P * w) / np.sum(w * w))

        # Fallback: use first few non-zero points
        n_use = min(5, len(self.displacements) - 1)
        w = self.displacements[1:n_use+1]
        P = self.loads[1:n_use+1]
        if len(w) > 0 and w[-1] > 0:
            return float(np.sum(P * w) / np.sum(w * w))

        return 0.0


def compute_elastoplastic_response(section: PlateBeamSection,
                                    n_points: int = 200,
                                    max_deflection_ratio: float = 0.1) -> ElastoplasticResponse:
    """Compute full elastoplastic load-displacement response.

    Parameters
    ----------
    section : PlateBeamSection
        Plate-beam combination section
    n_points : int
        Number of points on the curve
    max_deflection_ratio : float
        Maximum deflection as fraction of span

    Returns
    -------
    response : ElastoplasticResponse
        Full response with stages and failure modes
    """
    elastic = ElasticPlateBeam(section)
    L = section.plate.a
    s = section.plate.b
    t_p = section.plate.t_p

    # Key pressures
    p_plate_yield = elastic.pressure_at_plate_yield()
    p_beam_yield = elastic.pressure_at_beam_yield()

    # Determine which yields first
    plate_yields_first = p_plate_yield < p_beam_yield

    # Maximum deflection
    w_max = max_deflection_ratio * L

    # Use non-uniform spacing: more points near origin for better elastic resolution
    # w = w_max * (i/n)^2 gives quadratic spacing
    i_arr = np.arange(n_points)
    w = w_max * (i_arr / (n_points - 1))**2

    # Compute response
    pressures = np.zeros(n_points)
    loads = np.zeros(n_points)
    stages = []
    plate_yielded = np.zeros(n_points, dtype=bool)
    beam_yielded = np.zeros(n_points, dtype=bool)

    # Elastic stiffness (analytical)
    K_el = elastic.combined_stiffness()

    # Yield deflection
    w_yield = elastic.deflection_at_yield()

    # Plate ultimate pressure (yield line theory)
    a_over_b = section.plate.aspect_ratio
    if a_over_b >= 2.0:
        p_ult_plate = 24.0 * section.plate_M_pl / s**2
    elif a_over_b >= 1.0:
        ratio = s / section.plate.a
        p_ult_plate = 24.0 * section.plate_M_pl / s**2 * (1 + ratio/2)**2 / (1 + ratio)
    else:
        p_ult_plate = 24.0 * section.plate_M_pl / s**2

    # Beam ultimate pressure (plastic mechanism)
    p_ult_beam = 8.0 * section.M_pl / (s * L**2)

    # Combined ultimate
    p_ult_combined = p_ult_plate + p_ult_beam * 0.7

    # Plastic hinge deflection
    w_plate_plastic = w_yield * 3.0
    w_beam_plastic = w_yield * 5.0

    for i in range(n_points):
        w_i = w[i]

        if w_i < 1e-15:
            pressures[i] = 0
            loads[i] = 0
            stages.append("initial")
            continue

        if w_i <= w_yield * 0.99:
            # Elastic stage - use analytical stiffness directly
            p_i = w_i * K_el / (L * s)
            stage = "elastic"
            py = False
            by = False
        elif w_i <= w_plate_plastic:
            # Plate yielding stage
            p_plate = p_ult_plate * 0.95
            w_beam_extra = w_i - w_yield
            p_beam_extra = w_beam_extra * K_el / (L * s) * 0.3
            p_i = p_plate + p_beam_extra
            stage = "plate_yielding"
            py = True
            by = False
        elif w_i <= w_beam_plastic:
            # Beam approaching yield
            p_plate = p_ult_plate
            t_beam = (w_i - w_plate_plastic) / (w_beam_plastic - w_plate_plastic)
            t_beam = min(max(t_beam, 0), 1)
            p_beam = p_beam_yield + t_beam * (p_ult_beam - p_beam_yield)
            p_i = p_plate + p_beam * 0.7
            stage = "beam_approaching_yield"
            py = True
            by = t_beam > 0.5
        else:
            # Post-peak / collapse
            t_collapse = (w_i - w_beam_plastic) / (w_max - w_beam_plastic)
            t_collapse = min(max(t_collapse, 0), 1)
            p_peak = p_ult_combined
            p_residual = p_ult_plate * 0.5 + p_ult_beam * 0.3
            p_i = p_peak - t_collapse * (p_peak - p_residual)
            stage = "collapse"
            py = True
            by = True

        pressures[i] = p_i
        loads[i] = p_i * L * s
        plate_yielded[i] = py
        beam_yielded[i] = by
        stages.append(stage)

    return ElastoplasticResponse(
        displacements=w,
        loads=loads,
        pressures=pressures,
        stages=stages,
        plate_yielded=plate_yielded,
        beam_yielded=beam_yielded
    )


def compute_analytical_stiffness(section: PlateBeamSection) -> dict:
    """Compute analytical elastic stiffness for verification.

    Returns dict with:
    - K_beam: beam stiffness
    - K_plate: plate stiffness
    - K_combined: combined stiffness
    - w_beam_at_1kPa: beam deflection at 1 kPa
    - w_plate_at_1kPa: plate deflection at 1 kPa
    """
    elastic = ElasticPlateBeam(section)
    L = section.plate.a
    s = section.plate.b

    # Beam stiffness: K = 384*EI / (5*L^3) for simply supported beam
    K_beam = 384.0 * section.EI / (5.0 * L**3)

    # Plate stiffness
    a_over_b = section.plate.aspect_ratio
    if a_over_b >= 2.0:
        alpha = 0.0130
    elif a_over_b >= 1.5:
        alpha = 0.0122
    elif a_over_b >= 1.0:
        alpha = 0.00406 + 0.00894 * (a_over_b - 1.0)
    else:
        alpha = 0.00406

    K_plate = section.plate_D * L / (alpha * s**4)

    # Combined (series): 1/K = 1/K_beam + 1/K_plate
    K_combined = 1.0 / (1.0 / K_beam + 1.0 / K_plate)

    # Deflections at 1 kPa
    p_test = 1e3
    w_beam = elastic.beam_deflection(p_test)
    w_plate = elastic.plate_deflection(p_test)

    return {
        "K_beam_Nm": K_beam,
        "K_plate_Nm": K_plate,
        "K_combined_Nm": K_combined,
        "w_beam_at_1kPa_m": w_beam,
        "w_plate_at_1kPa_m": w_plate,
        "alpha_plate": alpha,
    }
