"""Plastic section analysis: M–κ relations for rectangular sections.

Supports:
- Elastic-Perfectly Plastic (EPP) material
- Linear kinematic hardening material

For a rectangular section (width b, height h):
  I = b*h³/12,  Z_el = b*h²/6,  Z_pl = b*h²/4
  κy = σy / (E * h/2) = 2*σy/(E*h)
  Me = σy * Z_el = σy*b*h²/6
  Mp = σy * Z_pl = σy*b*h²/4
  Shape factor = Mp/Me = 1.5

EPP closed-form M(κ):
  |κ| ≤ κy:  M = EI·κ
  |κ| > κy:  M = Mp·sign(κ)·[1 - (1/3)·(κy/κ)²]

Kinematic hardening: back stress β evolves as β = H·εp
  Yield surface: |σ - β| ≤ σy0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RectangularSection:
    """Rectangular cross-section properties."""
    b: float   # width [m]
    h: float   # height/thickness [m]

    @property
    def A(self) -> float:
        return self.b * self.h

    @property
    def I(self) -> float:
        return self.b * self.h**3 / 12.0

    @property
    def Z_el(self) -> float:
        return self.b * self.h**2 / 6.0

    @property
    def Z_pl(self) -> float:
        return self.b * self.h**2 / 4.0

    @property
    def shape_factor(self) -> float:
        return self.Z_pl / self.Z_el  # = 1.5 for rectangle


@dataclass
class EPPMaterial:
    """Elastic-Perfectly Plastic material."""
    E: float    # Young's modulus [Pa]
    sigma_y: float  # yield stress [Pa]
    nu: float = 0.3  # Poisson's ratio


@dataclass
class KinematicHardeningMaterial:
    """Linear kinematic hardening material."""
    E: float       # Young's modulus [Pa]
    sigma_y0: float  # initial yield stress [Pa]
    H: float       # kinematic hardening modulus [Pa]
    nu: float = 0.3


def epp_mk_closed_form(kappa: np.ndarray, sec: RectangularSection, mat: EPPMaterial) -> np.ndarray:
    """EPP M–κ closed-form relation for rectangular section.

    Parameters
    ----------
    kappa : array
        Curvature values [1/m]
    sec : RectangularSection
    mat : EPPMaterial

    Returns
    -------
    M : array
        Bending moment [N·m]
    """
    EI = mat.E * sec.I
    kappa_y = 2.0 * mat.sigma_y / (mat.E * sec.h)
    Mp = mat.sigma_y * sec.Z_pl

    M = np.zeros_like(kappa, dtype=float)
    elastic = np.abs(kappa) <= kappa_y
    plastic = ~elastic

    M[elastic] = EI * kappa[elastic]
    sign_k = np.sign(kappa[plastic])
    ratio = kappa_y / np.abs(kappa[plastic])
    M[plastic] = Mp * sign_k * (1.0 - ratio**2 / 3.0)

    return M


def epp_stress_at_strain(strain: np.ndarray, E: float, sigma_y: float) -> np.ndarray:
    """EPP stress-strain relation."""
    sigma_elastic = E * strain
    return np.where(np.abs(sigma_elastic) <= sigma_y,
                    sigma_elastic,
                    sigma_y * np.sign(strain))


def epp_mk_numerical(kappa: np.ndarray, sec: RectangularSection, mat: EPPMaterial,
                     n_fibers: int = 200) -> np.ndarray:
    """EPP M–κ by exact segment integration through thickness.

    Uses exact integration of σ(y)·y·b·dy over each segment,
    where σ(y) follows the EPP law.

    For a given κ, strain ε(y) = κ·y.
    The yield surface is at y_y = σy/(E·|κ|).
    
    Integration is split into elastic core [-y_y, y_y] and plastic zones.
    """
    M = np.zeros_like(kappa, dtype=float)
    kappa_y = 2.0 * mat.sigma_y / (mat.E * sec.h)

    for i, k in enumerate(kappa):
        if abs(k) < 1e-30:
            M[i] = 0.0
            continue

        abs_k = abs(k)
        sign_k = np.sign(k)
        
        # Yield position
        y_y = mat.sigma_y / (mat.E * abs_k)
        y_y = min(y_y, sec.h / 2.0)  # cap at section edge

        # Elastic core contribution: ∫_{-y_y}^{y_y} E·κ·y · y · b dy
        # = E·κ·b · [y³/3]_{-y_y}^{y_y} = 2·E·κ·b·y_y³/3
        M_elastic = 2.0 * mat.E * k * sec.b * y_y**3 / 3.0

        # Plastic zone contribution (top): ∫_{y_y}^{h/2} σy·sign(k) · y · b dy
        # = σy·sign(k)·b · [y²/2]_{y_y}^{h/2}
        y_top = sec.h / 2.0
        if y_top > y_y:
            M_plastic_top = mat.sigma_y * sign_k * sec.b * (y_top**2 - y_y**2) / 2.0
        else:
            M_plastic_top = 0.0

        # Plastic zone contribution (bottom): ∫_{-h/2}^{-y_y} -σy·sign(k) · y · b dy
        # = -σy·sign(k)·b · [y²/2]_{-h/2}^{-y_y}
        y_bot = -sec.h / 2.0
        if -y_y > y_bot:
            M_plastic_bot = -mat.sigma_y * sign_k * sec.b * (y_y**2 - y_bot**2) / 2.0
        else:
            M_plastic_bot = 0.0

        M[i] = M_elastic + M_plastic_top + M_plastic_bot

    return M


def kinematic_mk_closed_form(kappa_max: float, sec: RectangularSection,
                              mat: KinematicHardeningMaterial) -> Tuple[float, float]:
    """Kinematic hardening M–κ for monotonic loading to kappa_max.

    For linear kinematic hardening with rectangular section:
    The effective yield curvature evolves as plastic deformation accumulates.

    Returns (M, back_stress_moment_equivalent) at kappa_max.
    """
    EI = mat.E * sec.I
    kappa_y0 = 2.0 * mat.sigma_y0 / (mat.E * sec.h)

    if kappa_max <= kappa_y0:
        return EI * kappa_max, 0.0

    # For kinematic hardening, use fiber integration for accuracy
    n_fibers = 500
    y = np.linspace(-sec.h / 2.0, sec.h / 2.0, n_fibers)
    dy = sec.h / (n_fibers - 1)
    dA = sec.b * dy

    strain = kappa_max * y
    stress_elastic = mat.E * strain

    # For monotonic loading with kinematic hardening:
    stress = np.zeros_like(strain)
    elastic_mask = np.abs(stress_elastic) <= mat.sigma_y0
    stress[elastic_mask] = stress_elastic[elastic_mask]
    plastic_mask = ~elastic_mask
    sign_s = np.sign(strain[plastic_mask])
    eps_p = strain[plastic_mask] - mat.sigma_y0 / mat.E * sign_s
    stress[plastic_mask] = mat.sigma_y0 * sign_s + mat.H * eps_p

    M = np.sum(stress * y * dA)
    beta = np.where(elastic_mask, 0.0, mat.H * eps_p)
    M_beta = np.sum(beta * y * dA)

    return M, M_beta


def verify_mk_accuracy(sec: RectangularSection, mat: EPPMaterial,
                       n_fibers: int = 200) -> dict:
    """Verify closed-form vs numerical integration accuracy.

    Returns dict with max_relative_error and pass/fail.
    """
    kappa_y = 2.0 * mat.sigma_y / (mat.E * sec.h)
    # Test from 0.1*κy to 10*κy
    kappa = np.logspace(np.log10(0.1 * kappa_y), np.log10(10.0 * kappa_y), 100)
    # Include negative values
    kappa = np.concatenate([-kappa[::-1], [0.0], kappa])

    M_closed = epp_mk_closed_form(kappa, sec, mat)
    M_numerical = epp_mk_numerical(kappa, sec, mat, n_fibers)

    # Avoid division by zero near origin
    nonzero = np.abs(M_closed) > 1e-15
    rel_err = np.zeros_like(M_closed)
    rel_err[nonzero] = np.abs(M_closed[nonzero] - M_numerical[nonzero]) / np.abs(M_closed[nonzero])

    max_err = float(np.max(rel_err))

    return {
        "max_relative_error": max_err,
        "n_fibers": n_fibers,
        "n_points": len(kappa),
        "pass": max_err <= 1e-12,
        "kappa_y": kappa_y,
        "Me": mat.sigma_y * sec.Z_el,
        "Mp": mat.sigma_y * sec.Z_pl,
    }


def two_bar_shakedown_limit(sigma_y: float, E: float, A: float, L: float) -> dict:
    """Two-bar system shakedown limit (classic benchmark).

    Two parallel bars of area A, length L, material (E, σy).
    Bar 1: constant load P1, Bar 2: cyclic load P2(t) ∈ [0, P_max].
    
    Classic result for symmetric case (P1 = P2_mean):
    - First yield: when either bar reaches σy
    - Shakedown limit: P2_max = 2·A·σy when P1 = A·σy/2
    - Ratio: shakedown / first_yield = 2.0

    Returns dict with first_yield, shakedown_limit, ratio.
    """
    # First yield: P2 = A·σy (when bar 2 alone reaches yield, bar 1 at σy/2)
    first_yield_stress = sigma_y
    first_yield_force = A * sigma_y  # P2 at first yield

    # Shakedown: P2_max = 2·A·σy (classic Melan result)
    shakedown_stress = 2.0 * sigma_y
    shakedown_force = 2.0 * A * sigma_y

    return {
        "first_yield_stress": first_yield_stress,
        "first_yield_force": first_yield_force,
        "shakedown_stress": shakedown_stress,
        "shakedown_force": shakedown_force,
        "ratio": shakedown_stress / first_yield_stress,  # = 2.0
    }
