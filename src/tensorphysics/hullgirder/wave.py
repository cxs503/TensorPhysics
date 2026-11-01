"""Wave-induced vertical bending moment of the hull girder.

Two complementary models are provided.

1. Regulatory / recommended formula (ITTC-1978 form, used by IACS UR S11)::

       M_wh = +0.19 * C * L^2 * B * Cb          [kN m]   hogging
       M_ws = -0.11 * C * L^2 * B * (Cb + 0.7)  [kN m]   sagging
       C    = 10.75 - ((300 - L) / 100)^1.5     for 90 <= L <= 300 m

   These are the design wave bending moments for the North Atlantic
   (probability level 1e-8) and are the reference against which the spectral
   machinery below is verified.

2. Linear long-wave strip model (Froude-Krylov + hydrostatic restoring + ship
   inertia, i.e. the classical "ship hull as a beam on waves" formulation)::

       q(x) = rho g b(x) [ d(x) - zeta_fk(x) ] + omega^2 (m(x) + a33(x)) d(x)
       zeta_fk(x) = zeta_a exp(-k T) cos(k x)      (Smith correction)

   where ``d(x) = eta + theta x`` is the rigid-body heave+pitch response, solved
   from the two equilibrium conditions ``int q dx = 0`` and ``int x q dx = 0``.
   The ``exp(-k T)`` factor is the standard Smith correction (the wave pressure
   decays with depth, so the excitation of a section of draught ``T`` is
   attenuated).  Sectional heave added mass uses the high-frequency flat-plate
   value ``rho pi (B/2)^2`` scaled by ``added_mass_factor`` (0.5 for a hull with
   tapered waterplane).  Radiation damping and diffraction are *not* modelled;
   the model is therefore a long-wave / design-wave approximation, valid for
   ``lambda_w >= L`` and used as such (see the module docstring of the
   verification harness for the quantified deviation).

For a rectangular barge in the long-wave limit the model has an exact closed
form solution (see :func:`long_wave_closed_form_moment`), which is used as a
rigorous numerical check of the solver.
"""

from __future__ import annotations

import numpy as np

from .girder import shear_and_moment
from .model import GRAVITY, RHO_WATER, ShipGirder

__all__ = [
    "rule_coefficient_c",
    "ittc_design_wave_moment",
    "wave_moment_rao",
    "design_wave_moment",
    "long_wave_closed_form_moment",
    "ittc_spectrum",
    "moment_rms_from_spectrum",
    "north_atlantic_scatter",
    "long_term_design_moment",
]


# --------------------------------------------------------------------- rules
def rule_coefficient_c(length: float) -> float:
    """IACS UR S11 / ITTC-1978 wave coefficient ``C(L)``."""
    if length < 90.0 or length > 500.0:
        raise ValueError("rule formula is defined for 90 m <= L <= 500 m")
    if length <= 300.0:
        return 10.75 - ((300.0 - length) / 100.0) ** 1.5
    if length <= 350.0:
        return 10.75
    return 10.75 - ((length - 350.0) / 150.0) ** 1.5


def ittc_design_wave_moment(length: float, breadth: float, block: float) -> dict:
    """ITTC-1978 / IACS UR S11 vertical wave bending moments (N*m)."""
    c = rule_coefficient_c(length)
    hogging_knm = 0.19 * c * length ** 2 * breadth * block
    sagging_knm = -0.11 * c * length ** 2 * breadth * (block + 0.7)
    return {
        "coefficient_c": c,
        "hogging_Nm": float(hogging_knm * 1e3),
        "sagging_Nm": float(sagging_knm * 1e3),
        "formula": "ITTC-1978 / IACS UR S11: Mwh=0.19 C L^2 B Cb, Mws=-0.11 C L^2 B (Cb+0.7) [kN m]",
        "probability_level": 1e-8,
    }


# ------------------------------------------------------------- strip solver
def _sectional_properties(
    ship: ShipGirder, x: np.ndarray, added_mass_factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = ship.waterplane_width(x)
    m = ship.mass_per_length(x)
    a33 = added_mass_factor * RHO_WATER * np.pi * (0.5 * ship.breadth) ** 2 * np.ones_like(x)
    return b, m, a33


def wave_moment_rao(
    ship: ShipGirder,
    omega: float,
    amplitude: float = 1.0,
    smith: bool = True,
    added_mass_factor: float = 0.5,
    x: np.ndarray | None = None,
) -> np.ndarray:
    """Vertical bending moment ``M(x)`` for a unit ocean-wave amplitude (RAO).

    The returned array is the amplitude of the linear response to the regular
    wave ``zeta = amplitude * cos(k x - omega t)`` (waves travelling from the
    bow, head seas), i.e. it can be scaled linearly by the wave amplitude.
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    x = ship.grid() if x is None else np.asarray(x, dtype=float)
    b, m, a33 = _sectional_properties(ship, x, added_mass_factor)
    k = omega ** 2 / GRAVITY
    zeta = amplitude * np.cos(k * x)
    zeta_fk = zeta * np.exp(-k * ship.draught) if smith else zeta
    rho_g_b = RHO_WATER * GRAVITY * b
    inertia = omega ** 2 * (m + a33)

    w_load = rho_g_b + inertia
    s0 = float(np.trapezoid(w_load, x))
    s1 = float(np.trapezoid(w_load * x, x))
    s2 = float(np.trapezoid(w_load * x * x, x))
    r0 = float(np.trapezoid(rho_g_b * zeta_fk, x))
    r1 = float(np.trapezoid(rho_g_b * zeta_fk * x, x))
    matrix = np.array([[s0, s1], [s1, s2]])
    rhs = np.array([r0, r1])
    try:
        eta, theta = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate hull
        raise ValueError(f"heave/pitch system is singular for omega={omega}") from exc

    d = eta + theta * x
    q = rho_g_b * (d - zeta_fk) + inertia * d
    return shear_and_moment(x, q, support="free")[1]


def design_wave_moment(
    ship: ShipGirder,
    wavelength_factor: float = 1.0,
    wave_height_ratio: float = 20.0,
    smith: bool = True,
    added_mass_factor: float = 0.5,
) -> dict:
    """Wave bending moment for the classical design wave.

    Design wave convention: ``lambda_w = wavelength_factor * L`` and
    ``h_w = L / wave_height_ratio`` (``zeta_a = h_w / 2``).  Returns the extreme
    |M| and the midship value together with the wave parameters used.
    """
    wavelength = wavelength_factor * ship.length
    amplitude = 0.5 * ship.length / wave_height_ratio
    omega = float(np.sqrt(2.0 * np.pi * GRAVITY / wavelength))
    m = wave_moment_rao(
        ship, omega, amplitude=amplitude, smith=smith, added_mass_factor=added_mass_factor
    )
    idx = int(np.argmax(np.abs(m)))
    return {
        "wavelength_m": float(wavelength),
        "wave_height_m": float(2.0 * amplitude),
        "omega_rad_s": omega,
        "M_max_Nm": float(np.max(np.abs(m))),
        "M_at_max_x_m": float(ship.grid()[idx]),
        "M_midship_Nm": float(m[m.size // 2]),
        "smith_correction": bool(smith),
        "added_mass_factor": float(added_mass_factor),
    }


def long_wave_closed_form_moment(
    length: float, breadth: float, wavelength: float, amplitude: float
) -> float:
    """Exact long-wave midship moment of a uniform barge (rectangular section).

    For ``k L -> 0`` the static balance of a uniform free-free beam in a cosine
    wave reduces to ``q(x) = rho g B zeta_a k^2 (L^2/24 - x^2/2)`` and therefore

        |M(L/2)| = rho g B zeta_a k^2 L^4 / 384,   k = 2 pi / wavelength.
    """
    k = 2.0 * np.pi / wavelength
    return float(RHO_WATER * GRAVITY * breadth * amplitude * k ** 2 * length ** 4 / 384.0)


# ---------------------------------------------------------------- statistics
def ittc_spectrum(omega: np.ndarray, hs: float, t1: float) -> np.ndarray:
    """ITTC-1978 two-parameter wave spectrum (``t1`` = mean period)."""
    omega = np.asarray(omega, dtype=float)
    a = 173.0 * hs ** 2 / t1 ** 4
    b = 691.0 / t1 ** 4
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        s = (a / omega ** 5) * np.exp(-b / omega ** 4)
    return np.where(omega > 0.0, s, 0.0)


def moment_rms_from_spectrum(
    ship: ShipGirder,
    hs: float,
    t1: float,
    n_omega: int = 120,
    smith: bool = True,
    added_mass_factor: float = 0.5,
) -> float:
    """Short-term RMS of the midship wave bending moment (spectral integration)."""
    omega = np.linspace(0.15, 2.5, n_omega)
    spectrum = ittc_spectrum(omega, hs, t1)
    rao = np.array(
        [
            wave_moment_rao(
                ship, float(w), amplitude=1.0, smith=smith, added_mass_factor=added_mass_factor
            )[ship.grid().size // 2]
            for w in omega
        ]
    )
    var = float(np.trapezoid(rao ** 2 * spectrum, omega))
    return float(np.sqrt(max(var, 0.0)))


def north_atlantic_scatter(
    shape: float = 1.5, scale: float = 3.0, n_cells: int = 32, hs_max: float = 18.0
) -> list[dict]:
    """Coarse North Atlantic scatter approximation.

    ``Hs`` follows a two-parameter Weibull (default shape 1.5, scale 3.0 m),
    ``T1 = 3.6 sqrt(Hs)``, and the cycle weight of a cell is proportional to
    ``P(Hs cell) / T1``.  This is a *coarse* model of the climate (documented
    limitation): it is used for the long-term cross-check of the wave bending
    moment, not as a substitute for a full scatter diagram.
    """
    edges = np.linspace(0.5, hs_max, n_cells + 1)
    hs = 0.5 * (edges[1:] + edges[:-1])
    p = np.exp(-(edges[:-1] / scale) ** shape) - np.exp(-(edges[1:] / scale) ** shape)
    t1 = 3.6 * np.sqrt(hs)
    w = p / t1
    w = w / w.sum()
    return [
        {"hs": float(a), "t1": float(b), "weight": float(c)} for a, b, c in zip(hs, t1, w)
    ]


def long_term_design_moment(
    sigma_by_cell: list[float],
    weights: list[float],
    target_probability: float,
) -> dict:
    """Long-term extreme bending moment (Rice/Rayleigh cell model).

    ``P(M > m) = sum_i w_i exp(-m^2 / (2 sigma_i^2))`` is inverted for the
    requested per-cycle exceedance probability.
    """
    sig = np.asarray(sigma_by_cell, dtype=float)
    w = np.asarray(weights, dtype=float)
    if sig.shape != w.shape:
        raise ValueError("sigma and weights must have the same length")
    if np.any(sig <= 0.0):
        raise ValueError("all short-term RMS values must be positive")
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target probability must be in (0, 1)")

    def exceedance(m: float) -> float:
        return float(np.sum(w * np.exp(-m ** 2 / (2.0 * sig ** 2))))

    lo, hi = 0.0, 20.0 * float(sig.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if exceedance(mid) > target_probability:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9 * max(1.0, hi):
            break
    value = 0.5 * (lo + hi)
    return {
        "M_design_Nm": float(value),
        "target_probability_per_cycle": float(target_probability),
        "sigma_max_cell_Nm": float(sig.max()),
        "extreme_to_rms_ratio": float(value / sig.max()),
        "n_cells": int(sig.size),
    }