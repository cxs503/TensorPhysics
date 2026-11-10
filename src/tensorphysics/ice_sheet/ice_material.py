"""Temperature- and loading-rate-dependent material state of a floating ice sheet.

Why this module
---------------
The level-ice bending chain (:mod:`tensorphysics.ice_sheet.kirchhoff_plate_fem`,
:mod:`tensorphysics.ice_sheet.pasternak_foundation`) takes the ice thickness
``h``, the elastic modulus ``E`` and the flexural strength ``sigma_f`` as *given
numbers*.  In reality all three are state variables of the ice cover:

* ``E`` **increases** as the ice gets colder — the brine/pore liquid fraction
  (and hence the compliant phase) drops.  Measured ranges for granular
  freshwater/sea ice are ``E ~ 4-5 GPa`` at ``-5 degC`` up to
  ``E ~ 8-10 GPa`` at ``-30 degC`` (Michel 1978; Timco & Weeks 2010 review,
  where sea-ice moduli span 1-7 GPa with temperature/porosity);
* ``E`` **decreases** with decreasing *loading rate* once the creep (ductile)
  regime is entered: the secant modulus of a Norton-Bailey material
  ``eps_dot = B sigma^n`` is ``E_sec = B^(-1/n) eps_dot^((1-n)/n)``, i.e.
  ``E_sec ~ eps_dot^(-(n-1)/n)`` with ``n ~ 3``;
* the flexural strength ``sigma_f`` rises with ``|T|`` (colder ice is stronger:
  ``~0.5-0.7 MPa`` near the melting point, ``~1.2-1.6 MPa`` at ``-20 degC`` for
  freshwater ice) and rises with the strain rate up to the ductile-brittle
  transition (``eps_dot ~ 1e-3 1/s``), above which it saturates
  (Michel 1978; Sanderson 1988).

This module supplies that missing leg as an explicit, *inspectable*
parameterisation.  It is deliberately **not** presented as a constitutive law:
it is a monotone engineering interpolation through published measured ranges,
with every constant named, documented and overridable.  Its purpose is to make
the *dependence laws* of the design quantities (peak bending stress, first-crack
load) auditable instead of anecdotal.

Loading rate -> strain rate
---------------------------
The chain is driven by an indentation/loading rate ``v`` [m/s].  The extreme
fibre of a Kirchhoff plate at the load point sees

.. math:: \\dot\\varepsilon = \\frac{h}{2}\\,\\dot\\kappa,
          \\qquad \\dot\\kappa \\sim \\frac{v}{l^2},

with ``l = (D/k)^(1/4)`` the characteristic (foundation) length: a deflection
rate ``v`` at the load point produces a curvature rate of order ``v/l^2`` when
the deflection bowl of width ``~l`` deepens at that rate.  Hence
``eps_dot = h v / (2 l^2)``.  Because ``l`` itself depends on ``E`` (and ``E``
on the rate), the state is closed by a short fixed-point iteration.

Conventions
-----------
* ``temp_C`` is the *mean ice temperature* in degrees Celsius, ``<= 0``;
  temperature enters through ``|T|`` only (the model is not meant below the
  melting point of the brine, nor for ``|T| > 40 K``).
* SI units throughout; stresses in Pa, moduli in Pa, ``G_p`` in N/m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "GRAVITY",
    "SEA_WATER_DENSITY",
    "temperature_stiffening",
    "creep_rate_factor",
    "ice_elastic_modulus",
    "ice_flexural_strength",
    "strain_rate_from_indentation_rate",
    "ice_shear_modulus",
    "IceSheetMaterial",
    "material_sweep",
]

GRAVITY = 9.81
SEA_WATER_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# scalar building blocks
# ---------------------------------------------------------------------------
def _abs_temperature(temp_C: float) -> float:
    """``|T|`` for a (non-positive) ice temperature, with validation."""
    t = float(temp_C)
    if not np.isfinite(t):
        raise ValueError(f"temperature must be finite, got {temp_C!r}")
    if t > 1e-9:
        raise ValueError(
            f"ice temperature must be <= 0 degC (mean ice temperature), got {temp_C!r}"
        )
    return abs(t)


def temperature_stiffening(
    temp_C: float,
    ref_C: float = 10.0,
    slope_per_K: float = 0.03,
    floor: float = 0.2,
) -> float:
    """Linear temperature factor ``1 + slope (|T| - |T_ref|)`` (floored).

    ``ref_C`` is the *reference* ``|T|`` at which the factor is 1.  The default
    ``slope_per_K = 0.03`` reproduces the measured trend ``E ~ 4.2 GPa`` at
    ``-5 degC``, ``5.0 GPa`` at ``-10 degC``, ``6.5 GPa`` at ``-20 degC`` and
    ``8.0 GPa`` at ``-30 degC`` when ``E_ref = 5 GPa``.
    """
    a = _abs_temperature(temp_C)
    if not np.isfinite(slope_per_K) or slope_per_K < 0.0:
        raise ValueError(f"slope_per_K must be finite and >= 0, got {slope_per_K!r}")
    if not np.isfinite(ref_C) or ref_C < 0.0:
        raise ValueError(f"ref_C must be finite and >= 0, got {ref_C!r}")
    if not (0.0 < floor <= 1.0):
        raise ValueError(f"floor must lie in (0, 1], got {floor!r}")
    return max(floor, 1.0 + slope_per_K * (a - abs(ref_C)))


def creep_rate_factor(
    eps_dot: float,
    eps_ref: float = 1.0e-6,
    creep_exponent: float = 3.0,
) -> float:
    """Secant-modulus reduction of a Norton-Bailey material, in ``(0, 1)``.

    ``f = 1 / (1 + (eps_ref/eps_dot)^((n-1)/n))``.  For ``eps_dot >> eps_ref``
    the factor tends to 1 (elastic/brittle branch); for
    ``eps_dot << eps_ref`` it tends to ``(eps_dot/eps_ref)^((n-1)/n)``, the
    ``eps_dot^(-2/3)`` creep scaling for ``n = 3``.
    """
    e = float(eps_dot)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError(f"strain rate must be positive and finite, got {eps_dot!r}")
    if not np.isfinite(eps_ref) or eps_ref <= 0.0:
        raise ValueError(f"eps_ref must be positive and finite, got {eps_ref!r}")
    if not np.isfinite(creep_exponent) or creep_exponent <= 1.0:
        raise ValueError(
            f"creep_exponent must be finite and > 1, got {creep_exponent!r}"
        )
    m = (creep_exponent - 1.0) / creep_exponent
    return 1.0 / (1.0 + (eps_ref / e) ** m)


def ice_elastic_modulus(
    temp_C: float,
    eps_dot: float,
    E_ref: float = 5.0e9,
    ref_C: float = 10.0,
    temp_slope_per_K: float = 0.03,
    eps_ref: float = 1.0e-6,
    creep_exponent: float = 3.0,
) -> float:
    """Rate- and temperature-dependent ice modulus ``E(T, eps_dot)`` [Pa].

    ``E = E_ref * temperature_stiffening(T) * creep_rate_factor(eps_dot)``.
    """
    if not np.isfinite(E_ref) or E_ref <= 0.0:
        raise ValueError(f"E_ref must be positive and finite, got {E_ref!r}")
    return (
        E_ref
        * temperature_stiffening(temp_C, ref_C, temp_slope_per_K)
        * creep_rate_factor(eps_dot, eps_ref, creep_exponent)
    )


def ice_flexural_strength(
    temp_C: float,
    eps_dot: float,
    sigma_ref: float = 0.70e6,
    ref_C: float = 10.0,
    temp_slope_per_K: float = 0.04,
    eps_ref: float = 1.0e-5,
    creep_exponent: float = 3.0,
    rate_cap: float = 2.0,
) -> float:
    """Flexural strength ``sigma_f(T, eps_dot)`` [Pa] (rise, then saturation).

    ``sigma_f = sigma_ref * temperature_stiffening(T, slope=0.04)
    * min((eps_dot/eps_ref)^(1/n), rate_cap)``.

    The ``1/n`` rate exponent is the Norton-Bailey creep exponent expressed as a
    strength: the *ductile* strength scales as ``eps_dot^(1/n)``; the cap
    ``rate_cap`` is the ductile-brittle transition, above which measured ice
    strength saturates (and eventually softens through micro-cracking, a branch
    that is **not** modelled here).
    """
    if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
        raise ValueError(f"sigma_ref must be positive and finite, got {sigma_ref!r}")
    if not np.isfinite(rate_cap) or rate_cap < 1.0:
        raise ValueError(f"rate_cap must be finite and >= 1, got {rate_cap!r}")
    e = float(eps_dot)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError(f"strain rate must be positive and finite, got {eps_dot!r}")
    if not np.isfinite(eps_ref) or eps_ref <= 0.0:
        raise ValueError(f"eps_ref must be positive and finite, got {eps_ref!r}")
    if not np.isfinite(creep_exponent) or creep_exponent <= 1.0:
        raise ValueError(
            f"creep_exponent must be finite and > 1, got {creep_exponent!r}"
        )
    rate = min((e / eps_ref) ** (1.0 / creep_exponent), rate_cap)
    return (
        sigma_ref
        * temperature_stiffening(temp_C, ref_C, temp_slope_per_K)
        * rate
    )


def strain_rate_from_indentation_rate(
    v_indent: float, h: float, char_length: float
) -> float:
    """Extreme-fibre strain rate of a plate loaded at rate ``v`` [1/s].

    ``eps_dot = h v / (2 l^2)`` (see the module docstring).  ``v_indent`` is the
    indentation / loading rate [m/s], ``h`` the ice thickness [m] and
    ``char_length`` the characteristic length ``l = (D/k)^(1/4)`` [m].
    """
    v = float(v_indent)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"indentation rate must be positive and finite, got {v_indent!r}")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"ice thickness must be positive and finite, got {h!r}")
    if not np.isfinite(char_length) or char_length <= 0.0:
        raise ValueError(
            f"char_length must be positive and finite, got {char_length!r}"
        )
    return 0.5 * h * v / (char_length * char_length)


def ice_shear_modulus(E: float, nu: float = 0.3) -> float:
    """Isotropic shear modulus ``G = E / [2 (1 + nu)]`` [Pa]."""
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must lie in (-1, 0.5), got {nu!r}")
    return E / (2.0 * (1.0 + nu))


# ---------------------------------------------------------------------------
# the coupled ice-sheet state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IceSheetMaterial:
    """Self-consistent state of a floating ice sheet under a given load rate.

    The class closes the temperature -> modulus -> characteristic length ->
    strain rate loop with a fixed point iteration (:meth:`derived`), so that
    ``E``, ``l``, ``eps_dot`` and ``sigma_f`` are mutually consistent instead of
    being picked independently (the usual way these numbers are chosen in ice
    load chains, and a documented source of spread).

    Parameters
    ----------
    h : float
        Ice thickness [m].
    temp_C : float
        Mean ice temperature [degC], ``<= 0``.
    indentation_rate_m_s : float
        Loading / indentation rate [m/s] driving the strain rate.
    nu : float
        Poisson ratio (default 0.3).
    rho_w, g : float
        Sea-water density [kg/m3] and gravity [m/s2]; the Winkler modulus of a
        floating sheet is ``k = rho_w g`` (Archimedes).
    shear_layer_beta : float
        Calibration of the Pasternak shear layer, ``G_p = beta G_ice h``.
        ``beta = 0`` is exactly a Winkler foundation; ``beta = 1`` is the
        continuous-shear-layer reading used by the P54-W3 campaign.
    E_ref, sigma_ref, ref_C, temp_slope_E, temp_slope_sigma, eps_ref,
    eps_strength_ref, creep_exponent, strength_rate_cap
        Material-model constants; see :func:`ice_elastic_modulus` and
        :func:`ice_flexural_strength`.
    """

    h: float
    temp_C: float = -10.0
    indentation_rate_m_s: float = 0.05
    nu: float = 0.3
    rho_w: float = SEA_WATER_DENSITY
    g: float = GRAVITY
    shear_layer_beta: float = 1.0
    E_ref: float = 5.0e9
    sigma_ref: float = 0.70e6
    ref_C: float = 10.0
    temp_slope_E: float = 0.03
    temp_slope_sigma: float = 0.04
    eps_ref: float = 1.0e-6
    eps_strength_ref: float = 1.0e-5
    creep_exponent: float = 3.0
    strength_rate_cap: float = 2.0
    _fixed_point_tol: float = field(default=1.0e-12, repr=False)

    # -- validation ------------------------------------------------------
    def __post_init__(self) -> None:
        if not np.isfinite(self.h) or self.h <= 0.0:
            raise ValueError(f"h must be positive and finite, got {self.h!r}")
        _abs_temperature(self.temp_C)
        if not np.isfinite(self.indentation_rate_m_s) or self.indentation_rate_m_s <= 0.0:
            raise ValueError(
                "indentation_rate_m_s must be positive and finite, "
                f"got {self.indentation_rate_m_s!r}"
            )
        if not np.isfinite(self.shear_layer_beta) or self.shear_layer_beta < 0.0:
            raise ValueError(
                f"shear_layer_beta must be finite and >= 0, got {self.shear_layer_beta!r}"
            )
        for name, val in (("rho_w", self.rho_w), ("g", self.g), ("E_ref", self.E_ref),
                          ("sigma_ref", self.sigma_ref), ("eps_ref", self.eps_ref),
                          ("eps_strength_ref", self.eps_strength_ref)):
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {val!r}")

    # -- single-pass properties -----------------------------------------
    @property
    def abs_temp_C(self) -> float:
        """``|T|`` [K]."""
        return _abs_temperature(self.temp_C)

    @property
    def winkler_modulus(self) -> float:
        """Buoyancy subgrade modulus ``k = rho_w g`` [Pa/m]."""
        return self.rho_w * self.g

    def elastic_modulus(self, eps_dot: float) -> float:
        """``E(T, eps_dot)`` [Pa]."""
        return ice_elastic_modulus(
            self.temp_C,
            eps_dot,
            E_ref=self.E_ref,
            ref_C=self.ref_C,
            temp_slope_per_K=self.temp_slope_E,
            eps_ref=self.eps_ref,
            creep_exponent=self.creep_exponent,
        )

    def flexural_strength(self, eps_dot: float) -> float:
        """``sigma_f(T, eps_dot)`` [Pa]."""
        return ice_flexural_strength(
            self.temp_C,
            eps_dot,
            sigma_ref=self.sigma_ref,
            ref_C=self.ref_C,
            temp_slope_per_K=self.temp_slope_sigma,
            eps_ref=self.eps_strength_ref,
            creep_exponent=self.creep_exponent,
            rate_cap=self.strength_rate_cap,
        )

    @staticmethod
    def flexural_rigidity(E: float, h: float, nu: float = 0.3) -> float:
        """``D = E h^3 / [12 (1 - nu^2)]`` [N m]."""
        return E * h**3 / (12.0 * (1.0 - nu**2))

    # -- closed state ----------------------------------------------------
    def derived(self, max_iter: int = 60) -> Dict[str, float]:
        """Self-consistent state (fixed point in ``eps_dot <-> l <-> E``).

        Returns a dict with ``eps_dot``, ``E``, ``D``, ``l``, ``sigma_f``,
        ``k_w``, ``G_p``, ``gamma`` (Pasternak coupling), ``moment_capacity``
        (``M_f = sigma_f h^2 / 6``) and the iteration diagnostics.
        """
        k_w = self.winkler_modulus
        # start from the elastic (rate-independent) modulus
        E = self.E_ref * temperature_stiffening(
            self.temp_C, self.ref_C, self.temp_slope_E
        )
        eps = 1.0e-6
        for it in range(1, max_iter + 1):
            D = self.flexural_rigidity(E, self.h, self.nu)
            l = (D / k_w) ** 0.25
            eps_new = strain_rate_from_indentation_rate(
                self.indentation_rate_m_s, self.h, l
            )
            E_new = self.elastic_modulus(eps_new)
            rel = abs(E_new - E) / E
            E, eps = E_new, eps_new
            if rel < self._fixed_point_tol:
                break
        else:  # pragma: no cover - the map is a contraction for all sane inputs
            raise RuntimeError(
                "IceSheetMaterial.derived did not converge: "
                f"|dE|/E = {rel:.3e} after {max_iter} iterations"
            )
        D = self.flexural_rigidity(E, self.h, self.nu)
        l = (D / k_w) ** 0.25
        G_ice = ice_shear_modulus(E, self.nu)
        G_p = self.shear_layer_beta * G_ice * self.h
        sigma_f = self.flexural_strength(eps)
        gamma = G_p / math.sqrt(D * k_w)
        return {
            "h_m": float(self.h),
            "temp_C": float(self.temp_C),
            "abs_temp_C": float(self.abs_temp_C),
            "indentation_rate_m_s": float(self.indentation_rate_m_s),
            "eps_dot_1_s": float(eps),
            "E_Pa": float(E),
            "D_Nm": float(D),
            "k_w_Pa_per_m": float(k_w),
            "char_length_m": float(l),
            "G_ice_Pa": float(G_ice),
            "G_p_N_per_m": float(G_p),
            "pasternak_gamma": float(gamma),
            "sigma_f_Pa": float(sigma_f),
            "moment_capacity_Nm_per_m": float(sigma_f * self.h**2 / 6.0),
            "fixed_point_iterations": int(it),
        }

    # -- convenience -----------------------------------------------------
    def with_state(self, **kwargs) -> "IceSheetMaterial":
        """Copy with overridden fields (``h=..., temp_C=..., ...``)."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, object]:
        """All model constants plus the closed state (JSON-friendly)."""
        out: Dict[str, object] = {
            "constants": {
                "E_ref_Pa": self.E_ref,
                "sigma_ref_Pa": self.sigma_ref,
                "ref_C": self.ref_C,
                "temp_slope_E_per_K": self.temp_slope_E,
                "temp_slope_sigma_per_K": self.temp_slope_sigma,
                "eps_ref_1_s": self.eps_ref,
                "eps_strength_ref_1_s": self.eps_strength_ref,
                "creep_exponent": self.creep_exponent,
                "strength_rate_cap": self.strength_rate_cap,
                "nu": self.nu,
                "rho_w": self.rho_w,
                "g": self.g,
                "shear_layer_beta": self.shear_layer_beta,
            }
        }
        out.update(self.derived())
        return out


def material_sweep(
    h_values: List[float] | Tuple[float, ...],
    temp_values: List[float] | Tuple[float, ...],
    rate_values: List[float] | Tuple[float, ...],
    **kwargs,
) -> List[Dict[str, float]]:
    """Evaluate :meth:`IceSheetMaterial.derived` over a Cartesian grid.

    Returns one dict per ``(h, T, v)`` combination (order: h, then T, then v),
    each with the closed-state quantities plus the inputs.
    """
    rows: List[Dict[str, float]] = []
    for h in h_values:
        for t in temp_values:
            for v in rate_values:
                m = IceSheetMaterial(
                    h=float(h), temp_C=float(t), indentation_rate_m_s=float(v), **kwargs
                )
                row = m.derived()
                row["h_m"] = float(h)
                row["temp_C"] = float(t)
                row["indentation_rate_m_s"] = float(v)
                rows.append(row)
    return rows