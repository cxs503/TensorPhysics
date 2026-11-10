"""C1: Ice plate bending stress vs Timoshenko & Woinowsky-Krieger (1959).

Physics
-------
An infinite ice sheet modelled as a thin (Kirchhoff) plate on a Winkler
foundation (k = ρ_w g for a floating ice sheet) under a concentrated vertical
load P.  The characteristic length is l = (D/k)^{1/4} where D = E h^3 /
[12(1-ν²)] is the flexural rigidity.

The Timoshenko (1959, Art. 61) / Hetényi (1946) closed-form solution for the
deflection of an infinite plate under a point load uses Kelvin functions:

    w(r) = -P kei(r/l) / (2π k l²)

For a UNIFORM PATCH LOAD of radius a and total load P, the centre deflection is:

    w(0) = P / (8 k l²) × [1 - (a²/(2l²)) × (ln(a/(2l)) + γ - 1/2)]

where γ ≈ 0.5772 is the Euler-Mascheroni constant.  For a/l → 0 this tends to
the point-load value P/(8kl²).

The bending moments at the centre for a patch load are approximately:

    M_r(0) ≈ M_θ(0) ≈ P(1+ν)/(8π) × [1 - O((a/l)²)]

The maximum bending stress (at the bottom fibre, r = 0) is:

    σ_max = 6 M(0) / h²

Verification strategy
---------------------
We use the existing :mod:`tensorphysics.fem_ice_plate.foundation_plate` solver
(axisymmetric Hermite elements on a Winkler foundation) with a patch load and
compare against the analytical patch-load solution from
:mod:`tensorphysics.elastic_foundation.plates`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tensorphysics.elastic_foundation.plates import (
    winkler_plate_flexural_rigidity,
    winkler_plate_char_length,
    winkler_plate_patch_load,
    winkler_plate_patch_moment_r,
)
from tensorphysics.fem_ice_plate.foundation_plate import (
    plate_flexural_rigidity,
    solve_axisym_plate_foundation,
)


@dataclass
class TimoshenkoPlateSolution:
    """Analytical solution for an infinite plate on Winkler foundation.

    Parameters
    ----------
    P : float
        Total load [N].
    E : float
        Young's modulus of ice [Pa].
    h : float
        Ice thickness [m].
    nu : float
        Poisson's ratio.
    k : float
        Foundation modulus [N/m³] (for floating ice: ρ_w g).
    a : float, optional
        Patch radius [m].  If None, uses point-load formulas.
    """
    P: float
    E: float
    h: float
    nu: float
    k: float
    a: Optional[float] = None

    @property
    def D(self) -> float:
        """Flexural rigidity [N·m]."""
        return winkler_plate_flexural_rigidity(self.E, self.h, self.nu)

    @property
    def l(self) -> float:
        """Characteristic length [m]."""
        return winkler_plate_char_length(self.D, self.k)

    def centre_deflection(self) -> float:
        """w(0) [m].

        For patch load: uses Hankel integral solution.
        For point load: w(0) = P / (8 k l²).
        """
        if self.a is not None:
            return float(winkler_plate_patch_load(np.array([0.0]), self.P, self.a, self.D, self.k)[0])
        else:
            return self.P / (8.0 * self.k * self.l**2)

    def centre_moment(self) -> float:
        """M_r(0) = M_θ(0) [N·m/m].

        For patch load: uses analytical patch-load moment.
        For point load: M(0) = P(1+ν)/(8π).
        """
        if self.a is not None:
            return float(winkler_plate_patch_moment_r(np.array([0.0]), self.P, self.a, self.D, self.k, self.nu)[0])
        else:
            return self.P * (1.0 + self.nu) / (8.0 * math.pi)

    def max_bending_stress(self) -> float:
        """σ_max = 6 M(0) / h² [Pa]."""
        return 6.0 * abs(self.centre_moment()) / self.h**2

    def deflection_profile(self, r: np.ndarray) -> np.ndarray:
        """Deflection w(r)."""
        if self.a is not None:
            return winkler_plate_patch_load(r, self.P, self.a, self.D, self.k)
        else:
            from tensorphysics.elastic_foundation.plates import winkler_plate_point_load
            return winkler_plate_point_load(r, self.P, self.D, self.k)

    def moment_profile(self, r: np.ndarray) -> np.ndarray:
        """Radial bending moment M_r(r)."""
        if self.a is not None:
            return winkler_plate_patch_moment_r(r, self.P, self.a, self.D, self.k, self.nu)
        else:
            from tensorphysics.elastic_foundation.plates import winkler_plate_moment_r
            return winkler_plate_moment_r(r, self.P, self.D, self.k, self.nu)


@dataclass
class PlateBendingResult:
    """Result of the plate bending verification (C1)."""
    # Analytical
    sigma_analytical: float  # [Pa]
    w0_analytical: float  # [m]
    M0_analytical: float  # [N·m/m]
    # FEM
    sigma_fem: float  # [Pa]
    w0_fem: float  # [m]
    M0_fem: float  # [N·m/m]
    # Error
    stress_error_rel: float  # |σ_FEM - σ_analytical| / σ_analytical
    deflection_error_rel: float  # |w_FEM - w_analytical| / w_analytical
    # Parameters
    P: float
    E: float
    h: float
    nu: float
    k: float
    l_char: float
    a_patch: float
    n_el: int
    # Pass/fail
    c1_pass: bool  # stress error < 5%

    def as_dict(self) -> dict:
        return {
            "sigma_analytical_Pa": self.sigma_analytical,
            "sigma_fem_Pa": self.sigma_fem,
            "stress_error_rel": self.stress_error_rel,
            "w0_analytical_m": self.w0_analytical,
            "w0_fem_m": self.w0_fem,
            "deflection_error_rel": self.deflection_error_rel,
            "M0_analytical_Nm_m": self.M0_analytical,
            "M0_fem_Nm_m": self.M0_fem,
            "P_N": self.P,
            "E_Pa": self.E,
            "h_m": self.h,
            "nu": self.nu,
            "k_N_m3": self.k,
            "l_char_m": self.l_char,
            "a_patch_m": self.a_patch,
            "n_el": self.n_el,
            "c1_pass": self.c1_pass,
        }


def timoshenko_plate_point_load_stress(
    P: float, E: float, h: float, nu: float, k: float
) -> float:
    """Maximum bending stress under a point load on an infinite plate.

    σ_max = 3P(1+ν) / (4π h²)

    This is the limit as patch radius a → 0.

    Parameters
    ----------
    P : float
        Point load [N].
    E : float
        Young's modulus [Pa].
    h : float
        Plate thickness [m].
    nu : float
        Poisson's ratio.
    k : float
        Foundation modulus [N/m³].

    Returns
    -------
    float
        Maximum bending stress [Pa].
    """
    return 3.0 * P * (1.0 + nu) / (4.0 * math.pi * h**2)


def solve_plate_bending_verification(
    P: float = 50e3,
    E: float = 5.0e9,
    h: float = 1.0,
    nu: float = 0.3,
    k: float = 9810.0,
    n_el: int = 200,
    patch_radius_frac: float = 0.05,
) -> PlateBendingResult:
    """Run the C1 plate bending verification.

    The FEM solver uses a patch load of radius a = patch_radius_frac × l.
    We compare against the analytical patch-load solution.

    Parameters
    ----------
    P : float
        Total load [N].
    E, h, nu, k : float
        Plate and foundation parameters.
    n_el : int
        Number of radial FEM elements.
    patch_radius_frac : float
        Patch radius as fraction of characteristic length l.

    Returns
    -------
    PlateBendingResult
    """
    D = plate_flexural_rigidity(E, h, nu)
    l_char = (D / k) ** 0.25
    a = patch_radius_frac * l_char

    # --- Analytical (patch load) ---
    analytical = TimoshenkoPlateSolution(P, E, h, nu, k, a=a)
    sigma_ana = analytical.max_bending_stress()
    w0_ana = analytical.centre_deflection()
    M0_ana = analytical.centre_moment()

    # --- FEM solution ---
    w0_guess = float(winkler_plate_patch_load(np.array([0.0]), P, a, D, k)[0])
    res = solve_axisym_plate_foundation(
        D, nu, k,
        patch_radius=a,
        n_el=n_el,
        grading=1.8,
        w_center_steps=np.array([w0_guess]),
        tol=1e-12,
    )

    # The FEM gives moments at element centroids.  The peak moment is at the
    # first element centroid (nearest the centre).
    M0_fem_r = res.M_r_elem[0]
    M0_fem_theta = res.M_theta_elem[0]
    # Average for the biaxial moment at centre
    M0_fem = 0.5 * (M0_fem_r + M0_fem_theta)

    # FEM stress from the average moment
    sigma_fem = 6.0 * abs(M0_fem) / h**2

    # FEM deflection at centre
    w0_fem = res.w[0]

    # Errors
    stress_err = abs(sigma_fem - sigma_ana) / sigma_ana
    defl_err = abs(w0_fem - w0_ana) / abs(w0_ana)

    return PlateBendingResult(
        sigma_analytical=sigma_ana,
        w0_analytical=w0_ana,
        M0_analytical=M0_ana,
        sigma_fem=sigma_fem,
        w0_fem=w0_fem,
        M0_fem=M0_fem,
        stress_error_rel=stress_err,
        deflection_error_rel=defl_err,
        P=P,
        E=E,
        h=h,
        nu=nu,
        k=k,
        l_char=l_char,
        a_patch=a,
        n_el=n_el,
        c1_pass=stress_err < 0.05,
    )
