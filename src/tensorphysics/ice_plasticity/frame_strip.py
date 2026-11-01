"""Frame strip model: beam/plate strip with plastic hinges under cyclic ice pressure.

Models:
- Simply supported beam with central plastic hinge
- Plate strip (wide beam) with distributed plasticity
- ISO 19906 cyclic pressure loading

For a simply supported beam of span L under uniform pressure p:
  M_max = p·b·L²/8  (b = stiffener spacing / tributary width)
  w_max = 5·p·b·L⁴/(384·EI)  (elastic deflection)

Plastic hinge forms when M_max = Mp (plastic moment capacity).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Union

from .plastic_section import (
    RectangularSection, EPPMaterial, KinematicHardeningMaterial,
    epp_mk_closed_form, epp_mk_numerical
)


@dataclass
class BeamStrip:
    """Simply supported beam strip model.

    Represents a stiffener with attached plating (effective width).
    """
    L: float          # span length [m]
    b: float          # tributary width / stiffener spacing [m]
    sec: RectangularSection  # cross-section
    mat: EPPMaterial

    @property
    def EI(self) -> float:
        return self.mat.E * self.sec.I

    @property
    def Me(self) -> float:
        """Elastic moment capacity (first yield)."""
        return self.mat.sigma_y * self.sec.Z_el

    @property
    def Mp(self) -> float:
        """Plastic moment capacity."""
        return self.mat.sigma_y * self.sec.Z_pl

    @property
    def kappa_y(self) -> float:
        """Yield curvature."""
        return 2.0 * self.mat.sigma_y / (self.mat.E * self.sec.h)

    def elastic_deflection(self, p: float) -> float:
        """Maximum elastic deflection under uniform pressure p [Pa]."""
        q = p * self.b  # line load [N/m]
        return 5.0 * q * self.L**4 / (384.0 * self.EI)

    def max_moment(self, p: float) -> float:
        """Maximum bending moment under uniform pressure p [Pa]."""
        q = p * self.b
        return q * self.L**2 / 8.0

    def pressure_at_first_yield(self) -> float:
        """Uniform pressure causing first yield [Pa]."""
        return 8.0 * self.Me / (self.b * self.L**2)

    def pressure_at_full_plastic(self) -> float:
        """Uniform pressure causing full plastic mechanism [Pa]."""
        return 8.0 * self.Mp / (self.b * self.L**2)

    def residual_deflection_epp(self, p_max: float, n_fibers: int = 200) -> float:
        """Residual deflection after loading to p_max and unloading (EPP).

        Uses M-κ relation to compute plastic curvature, then integrates
        to get deflection.

        Parameters
        ----------
        p_max : float
            Maximum pressure [Pa]
        n_fibers : int
            Fiber count for numerical integration

        Returns
        -------
        w_residual : float
            Residual deflection [m]
        """
        M_max = self.max_moment(p_max)

        if M_max <= self.Me:
            return 0.0  # purely elastic, no residual

        # Find curvature at M_max by inverting M-κ
        kappa = self._invert_mk(M_max, n_fibers)

        # Elastic unloading curvature
        kappa_elastic = M_max / self.EI

        # Residual curvature
        kappa_residual = kappa - kappa_elastic

        # For simply supported beam with constant residual curvature
        # (plastic hinge at midspan approximation):
        w_residual = kappa_residual * self.L**2 / 8.0

        return w_residual

    def _invert_mk(self, M_target: float, n_fibers: int = 200) -> float:
        """Find curvature κ such that M(κ) = M_target (EPP)."""
        from scipy.optimize import brentq

        def residual(kappa):
            M = epp_mk_numerical(np.array([kappa]), self.sec, self.mat, n_fibers)[0]
            return M - M_target

        # Bracket: from elastic to highly plastic
        kappa_lo = M_target / self.EI  # elastic estimate
        kappa_hi = 100.0 * self.kappa_y  # highly plastic

        # Ensure bracketing
        if residual(kappa_lo) * residual(kappa_hi) > 0:
            kappa_hi = 1000.0 * self.kappa_y

        try:
            return brentq(residual, kappa_lo, kappa_hi, xtol=1e-15)
        except ValueError:
            return kappa_lo  # fallback


@dataclass
class PlateStrip:
    """Plate strip model (wide beam approximation for plating between stiffeners).

    Unit width strip of plate, thickness t, span = stiffener spacing.
    """
    t: float          # plate thickness [m]
    s: float          # stiffener spacing / span [m]
    mat: EPPMaterial

    @property
    def sec(self) -> RectangularSection:
        return RectangularSection(b=1.0, h=self.t)  # unit width

    @property
    def EI(self) -> float:
        return self.mat.E * self.sec.I

    @property
    def Me(self) -> float:
        return self.mat.sigma_y * self.sec.Z_el

    @property
    def Mp(self) -> float:
        return self.mat.sigma_y * self.sec.Z_pl

    @property
    def kappa_y(self) -> float:
        return 2.0 * self.mat.sigma_y / (self.mat.E * self.t)

    def pressure_at_first_yield(self) -> float:
        """Pressure causing first yield in plate strip [Pa]."""
        return 8.0 * self.Me / (1.0 * self.s**2)

    def pressure_at_full_plastic(self) -> float:
        """Pressure causing full plastic mechanism [Pa]."""
        return 8.0 * self.Mp / (1.0 * self.s**2)

    def residual_deflection(self, p_max: float) -> float:
        """Residual deflection after loading to p_max and unloading."""
        M_max = p_max * 1.0 * self.s**2 / 8.0  # unit width

        if M_max <= self.Me:
            return 0.0

        # Invert M-κ for EPP rectangular section
        ratio = M_max / self.Mp
        if ratio >= 1.0:
            ratio = 0.999  # cap near plastic limit
        kappa_y = self.kappa_y
        kappa = kappa_y / np.sqrt(3.0 * (1.0 - ratio))

        kappa_elastic = M_max / self.EI
        kappa_residual = kappa - kappa_elastic

        return kappa_residual * self.s**2 / 8.0


def compute_frame_strip_response(params: dict) -> dict:
    """Compute frame strip response for given parameters.

    Parameters
    ----------
    params : dict
        Must contain: L, b, plate_thickness, E, sigma_y, nu, p0, ex

    Returns
    -------
    result : dict
        Contains pressures, deflections, and regime classification
    """
    sec = RectangularSection(b=params["b"], h=params["plate_thickness"])
    mat = EPPMaterial(E=params["E"], sigma_y=params["sigma_y"], nu=params["nu"])
    beam = BeamStrip(L=params["L"], b=params["b"], sec=sec, mat=mat)

    from .iso19906 import ISO19906Pressure
    ice = ISO19906Pressure(p0=params["p0"], ex=params["ex"])

    # Panel area
    A_panel = params["L"] * params["b"]
    p_ice = ice.pressure_Pa(A_panel)  # Pa (scalar)

    p_yield = beam.pressure_at_first_yield()
    p_plastic = beam.pressure_at_full_plastic()

    w_elastic = beam.elastic_deflection(p_ice)
    w_residual = beam.residual_deflection_epp(p_ice)

    # Regime classification
    if p_ice <= p_yield:
        regime = "elastic"
    elif p_ice <= p_plastic:
        regime = "plastic_shakedown"
    else:
        regime = "ratcheting"

    return {
        "p_ice_Pa": float(p_ice),
        "p_yield_Pa": p_yield,
        "p_plastic_Pa": p_plastic,
        "w_elastic_m": w_elastic,
        "w_residual_m": w_residual,
        "regime": regime,
        "Me_Nm": beam.Me,
        "Mp_Nm": beam.Mp,
        "shape_factor": beam.Mp / beam.Me,
    }
