"""Ultimate bearing capacity of plate-beam combination under ice load.

Ultimate capacity calculation
------------------------------
The ultimate capacity is computed as the combination of:
1. Plate ultimate strength (yield line theory)
2. Beam plastic moment capacity

For a stiffened panel under uniform pressure:

Plate ultimate (yield line theory):
- Simply supported long plate (a/b >= 2):
  p_ult_plate = 24 * M_pl_plate / s^2
- Rectangular plate (a/b >= 1):
  p_ult_plate = 24 * M_pl_plate / s^2 * (1 + s/(2a))^2 / (1 + s/a)

Beam ultimate (plastic mechanism):
- Simply supported beam with central plastic hinge:
  p_ult_beam = 8 * M_pl_beam / (s * L^2)

Combined ultimate:
- Plate and beam work together
- p_ult_combined = p_ult_plate + eta * p_ult_beam
- eta = interaction factor (typically 0.6-0.8)

Polar Code comparison
---------------------
IMO Polar Code Chapter 6 provides requirements for ice-strengthened structures.
The design ice pressure is compared with the structural capacity using
partial safety factors.

For local strength (shell plating and stiffeners):
- Design pressure from IACS UR I3
- Capacity from plastic analysis
- Safety factor gamma_M = 1.1 (material)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .plate_beam import PlateBeamSection


@dataclass(frozen=True)
class UltimateCapacity:
    """Ultimate bearing capacity of plate-beam combination.

    Parameters
    ----------
    section : PlateBeamSection
        Plate-beam combination section
    """
    section: PlateBeamSection

    @property
    def L(self) -> float:
        return self.section.plate.a

    @property
    def s(self) -> float:
        return self.section.plate.b

    @property
    def t_p(self) -> float:
        return self.section.plate.t_p

    @property
    def sigma_y(self) -> float:
        return self.section.sigma_y

    def plate_ultimate_pressure(self) -> float:
        """Plate ultimate pressure from yield line theory [Pa].

        For simply supported plate with uniform load.
        """
        M_pl = self.section.plate_M_pl
        a_over_b = self.section.plate.aspect_ratio

        if a_over_b >= 2.0:
            # Long plate: yield lines along supported edges
            # p_ult = 24 * M_pl / s^2
            return 24.0 * M_pl / self.s**2
        elif a_over_b >= 1.0:
            # Rectangular plate: Johansen yield line pattern
            # p_ult = 24 * M_pl / s^2 * (1 + s/(2a))^2 / (1 + s/a)
            ratio = self.s / self.L
            return 24.0 * M_pl / self.s**2 * (1 + ratio / 2)**2 / (1 + ratio)
        else:
            # Square plate (a/b < 1, plate longer than wide)
            # Use reciprocal aspect ratio
            ratio = self.L / self.s
            return 24.0 * M_pl / self.L**2 * (1 + ratio / 2)**2 / (1 + ratio)

    def beam_ultimate_pressure(self) -> float:
        """Beam ultimate pressure from plastic mechanism [Pa].

        Simply supported beam with central plastic hinge:
        M_max = p * s * L^2 / 8 = M_pl
        => p_ult = 8 * M_pl / (s * L^2)
        """
        return 8.0 * self.section.M_pl / (self.s * self.L**2)

    def combined_ultimate_pressure(self, eta: float = 0.7) -> float:
        """Combined ultimate pressure [Pa].

        p_ult = p_ult_plate + eta * p_ult_beam

        eta accounts for the interaction between plate and beam.
        When plate yields, it redistributes load to the beam.
        Typical eta = 0.6-0.8.
        """
        p_plate = self.plate_ultimate_pressure()
        p_beam = self.beam_ultimate_pressure()
        return p_plate + eta * p_beam

    def combined_ultimate_force(self, eta: float = 0.7) -> float:
        """Combined ultimate force [N]."""
        return self.combined_ultimate_pressure(eta) * self.L * self.s

    def polar_code_check(self, design_pressure_Pa: float,
                         gamma_M: float = 1.1) -> dict:
        """Check against IMO Polar Code requirements.

        Parameters
        ----------
        design_pressure_Pa : float
            Design ice pressure [Pa]
        gamma_M : float
            Material partial safety factor

        Returns
        -------
        result : dict
            Check results including utilisation and pass/fail
        """
        p_ult = self.combined_ultimate_pressure()
        p_capacity = p_ult / gamma_M

        utilisation = design_pressure_Pa / p_capacity

        return {
            "design_pressure_Pa": design_pressure_Pa,
            "design_pressure_MPa": design_pressure_Pa / 1e6,
            "p_ult_plate_Pa": self.plate_ultimate_pressure(),
            "p_ult_beam_Pa": self.beam_ultimate_pressure(),
            "p_ult_combined_Pa": p_ult,
            "p_capacity_Pa": p_capacity,
            "gamma_M": gamma_M,
            "utilisation": utilisation,
            "passes": utilisation <= 1.0,
            "margin_fraction": 1.0 - utilisation,
        }

    def iacs_ur_i3_check(self, pc: int, gamma_M: float = 1.1) -> dict:
        """Check against IACS UR I3 requirements.

        Parameters
        ----------
        pc : int
            Polar class (1-7)
        gamma_M : float
            Material partial safety factor

        Returns
        -------
        result : dict
            Check results
        """
        from .ice_load import IACSUR_I3_Pressure

        ice_model = IACSUR_I3_Pressure(pc=pc, A=self.L * self.s, AR=self.L / self.s)
        p_design = ice_model.peak_pressure_Pa()

        return self.polar_code_check(p_design, gamma_M)

    def summary(self) -> dict:
        """Summary of ultimate capacity analysis."""
        return {
            "plate_ultimate_pressure_Pa": self.plate_ultimate_pressure(),
            "plate_ultimate_pressure_MPa": self.plate_ultimate_pressure() / 1e6,
            "beam_ultimate_pressure_Pa": self.beam_ultimate_pressure(),
            "beam_ultimate_pressure_MPa": self.beam_ultimate_pressure() / 1e6,
            "combined_ultimate_pressure_Pa": self.combined_ultimate_pressure(),
            "combined_ultimate_pressure_MPa": self.combined_ultimate_pressure() / 1e6,
            "combined_ultimate_force_N": self.combined_ultimate_force(),
            "combined_ultimate_force_kN": self.combined_ultimate_force() / 1e3,
            "plate_M_pl_Nm_m": self.section.plate_M_pl,
            "beam_M_pl_Nm": self.section.M_pl,
            "shape_factor_plate": self.section.plate_M_pl / self.section.plate_M_el,
            "shape_factor_beam": self.section.shape_factor,
        }


def compute_ultimate_capacity(section: PlateBeamSection,
                               pc: Optional[int] = None) -> dict:
    """Compute ultimate capacity and code checks.

    Parameters
    ----------
    section : PlateBeamSection
        Plate-beam combination section
    pc : int, optional
        Polar class for IACS UR I3 check

    Returns
    -------
    result : dict
        Complete ultimate capacity analysis
    """
    uc = UltimateCapacity(section)
    result = uc.summary()

    if pc is not None:
        iacs_check = uc.iacs_ur_i3_check(pc)
        result["iacs_ur_i3_check"] = iacs_check

    return result
