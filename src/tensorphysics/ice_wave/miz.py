"""Marginal ice zone (MIZ) wave attenuation: exponential decay and calibration.

Physical model
--------------
Brash ice / fragmented ice in a MIZ is homogenised as an effective Kelvin-Voigt
viscoelastic plate occupying a fraction ``n_c`` (ice concentration) of the surface.
Parallel (Voigt) mixing of the floe response gives the effective properties

    D_eff      = n_c * D_floe * (1 + i tan delta)
    (rho h)_eff = n_c * rho_ice * h_ice

with ``tan delta`` the floe-scale loss factor (energy dissipated by floe-floe
interaction, overwash and floe flexure). Wave amplitudes then decay spatially as

    A(x) = A0 exp(-alpha x),     alpha = Im(k(omega)),

where the complex wavenumber solves the same dispersion relation as
:mod:`tensorphysics.ice_wave.flexural` with the complex rigidity:

    omega^2 = (g k + D_eff k^5 / rho_w) tanh(k h).

Three independent ways of obtaining ``alpha`` are provided so it can be
cross-validated rather than asserted:

1. :func:`kelvin_voigt_wavenumber` -- complex k at real frequency (Newton), the
   primary value;
2. :func:`temporal_attenuation` -- ``alpha = |Im omega| / c_g`` from the complex
   eigenfrequency at real k (spectral collocation eigensolver in
   :mod:`tensorphysics.ice_wave.flexural`), a different discretisation and a
   different formulation.

The two routes must agree in the weak-damping limit (``tan delta << 1``); their
discrepancy is reported as the numerical-validation residual of ``alpha``.

The literature comparison (:data:`LITERATURE_ALPHA_BAND`) is an order-of-magnitude
band taken from MIZ field/model studies (Kohout et al. 2011; Meylan et al. 2018
review values). The specific tabulated values of Meylan et al. (2018) could not be
re-read in the environment used for this task, so that leg is reported as
``WITHHELD`` provenance rather than claimed as a ``<15%`` validation.
"""

from __future__ import annotations

from math import isfinite, pi, sqrt, tanh
from typing import Any, Iterable

import numpy as np

from .flexural import (
    FlexuralWaveError,
    flexural_gravity_dispersion,
    flexural_gravity_wavenumber,
    flexural_rigidity,
    group_speed,
    spectral_dispersion,
)

GRAVITY = 9.81
RHO_WATER = 1025.0
RHO_ICE = 917.0


class MIZError(ValueError):
    """Invalid input for the MIZ attenuation model."""


#: Order-of-magnitude literature band for MIZ attenuation of ~10 s waves
#: (alpha in 1/m). Provenance: MIZ field and model studies reporting
#: 1e-5 ... 1e-4 1/m for ice concentrations of 0.5-0.9 and 0.5-2 m thick ice.
LITERATURE_ALPHA_BAND = {
    "alpha_min": 1.0e-5,
    "alpha_max": 1.0e-4,
    "reference_wave_period_s": 10.0,
    "sources": "Kohout et al. (2011) MIZ attenuation scaling; Meylan et al. (2018) MIZ review",
    "provenance": "WITHHELD",
    "note": (
        "Band width is about one decade; the specific published values of "
        "Meylan et al. (2018) were not available in this environment, so the "
        "comparison is order-of-magnitude only and MUST NOT be read as a <15% "
        "validation against that paper."
    ),
}


def effective_medium_properties(
    concentration: float,
    ice_thickness: float,
    *,
    loss_factor: float = 0.02,
    young_modulus: float = 5.0e9,
    poisson_ratio: float = 0.3,
    rho_ice: float = RHO_ICE,
) -> dict[str, float]:
    """Voigt-mixing effective plate properties for a brash-ice MIZ.

    Returns ``D_real``, ``D_imag``, ``loss_factor`` and the effective mass per unit
    area ``rho_h_eff``.
    """
    n_c = float(concentration)
    if not isfinite(n_c) or not (0.0 < n_c <= 1.0):
        raise MIZError(f"concentration must be in (0, 1], got {concentration!r}")
    h_i = float(ice_thickness)
    if not isfinite(h_i) or h_i <= 0.0:
        raise MIZError(f"ice_thickness must be finite and positive, got {ice_thickness!r}")
    loss = float(loss_factor)
    if not isfinite(loss) or loss < 0.0:
        raise MIZError(f"loss_factor must be finite and non-negative, got {loss_factor!r}")
    if not isfinite(float(rho_ice)) or float(rho_ice) <= 0.0:
        raise MIZError(f"rho_ice must be finite and positive, got {rho_ice!r}")
    d_floe = flexural_rigidity(h_i, young_modulus, poisson_ratio)
    return {
        "D_real": n_c * d_floe,
        "D_imag": n_c * d_floe * loss,
        "loss_factor": loss,
        "rho_h_eff": n_c * float(rho_ice) * h_i,
        "D_floe": d_floe,
        "concentration": n_c,
        "ice_thickness": h_i,
    }


def kelvin_voigt_wavenumber(
    period: float,
    depth: float,
    D_real: float,
    D_imag: float,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    tol: float = 1e-14,
    max_iter: int = 300,
) -> complex:
    """Complex wavenumber ``k`` at real frequency ``omega = 2 pi / period``.

    Solved with a damped complex Newton iteration warm-started from the undamped
    root. ``Im(k) > 0`` corresponds to spatial decay ``exp(-Im(k) x)``.
    """
    period = float(period)
    if not isfinite(period) or period <= 0.0:
        raise MIZError(f"period must be finite and positive, got {period!r}")
    depth = float(depth)
    if not isfinite(depth) or depth <= 0.0:
        raise MIZError(f"depth must be finite and positive, got {depth!r}")
    D = complex(D_real, D_imag)
    if not isfinite(D.real) or not isfinite(D.imag) or D.real < 0.0:
        raise MIZError(f"rigidity must be finite with non-negative real part, got {D!r}")
    omega = 2.0 * pi / period
    target = omega**2
    k0 = flexural_gravity_wavenumber(omega, depth, D.real, rho_w=rho_w, g=g)
    k = complex(k0, 0.0)

    def f(kk: complex) -> complex:
        return (g * kk + D * kk**5 / rho_w) * np.tanh(kk * depth) - target

    for _ in range(max_iter):
        val = f(k)
        h = 1e-7 * max(abs(k), 1e-8)
        deriv = (f(k + h) - f(k - h)) / (2.0 * h)
        step = val / deriv
        k_new = k - step
        if abs(step) <= tol * max(1.0, abs(k)):
            k = k_new
            break
        k = k_new
        if not np.isfinite(k):
            raise MIZError("complex Newton iteration diverged")
    if k.imag < 0.0:  # ensure the decaying branch
        k = complex(k.real, abs(k.imag))
    return complex(k)


def attenuation_from_complex_wavenumber(k: complex) -> float:
    """Spatial attenuation coefficient ``alpha = Im(k)`` (root of ``exp(-alpha x)``)."""
    return float(abs(complex(k).imag))


def temporal_attenuation(
    period: float,
    depth: float,
    D_real: float,
    D_imag: float,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    n_modes: int = 24,
) -> dict[str, float]:
    """Cross-check ``alpha = |Im omega| / c_g`` from the spectral eigensolver.

    The complex eigenfrequency at *real* wavenumber is obtained from the
    collocation eigensolver of :func:`tensorphysics.ice_wave.flexural.spectral_dispersion`
    with a complex rigidity, i.e. a different discretisation and formulation than
    the complex-k Newton solve.
    """
    omega = 2.0 * pi / float(period)
    k_real = flexural_gravity_wavenumber(omega, depth, D_real, rho_w=rho_w, g=g)
    omega_num = spectral_dispersion(
        k_real,
        depth,
        complex(D_real, D_imag),
        n_modes=n_modes,
        rho_w=rho_w,
        g=g,
    )
    cg = group_speed(k_real, depth, D_real, rho_w=rho_w, g=g)
    alpha = abs(omega_num.imag) / abs(cg)
    return {
        "k_real": float(k_real),
        "omega_real": float(omega_num.real),
        "omega_imag": float(omega_num.imag),
        "group_speed": float(cg),
        "alpha": float(alpha),
        "phase": "temporal",
    }


def exponential_decay_profile(
    x: Any,
    amplitude0: float,
    alpha: float,
) -> np.ndarray:
    """``A(x) = A0 exp(-alpha x)``."""
    alpha = float(alpha)
    if not isfinite(alpha) or alpha < 0.0:
        raise MIZError(f"alpha must be finite and non-negative, got {alpha!r}")
    a0 = float(amplitude0)
    if not isfinite(a0):
        raise MIZError(f"amplitude0 must be finite, got {amplitude0!r}")
    x_arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x_arr)):
        raise MIZError("x contains non-finite values")
    return a0 * np.exp(-alpha * x_arr)


def fit_attenuation_coefficient(
    x: Any,
    amplitude: Any,
) -> dict[str, float]:
    """Least-squares fit of ``ln A = ln A0 - alpha x``.

    Returns ``alpha``, ``amplitude0``, ``r2`` and ``rmse_log``.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    a_arr = np.asarray(amplitude, dtype=float).ravel()
    if x_arr.shape != a_arr.shape:
        raise MIZError(f"x and amplitude lengths differ: {x_arr.shape} vs {a_arr.shape}")
    if x_arr.size < 3:
        raise MIZError(f"at least 3 samples are required, got {x_arr.size}")
    if np.any(a_arr <= 0.0):
        raise MIZError("amplitude must be strictly positive for the log fit")
    y = np.log(a_arr)
    design = np.column_stack([np.ones_like(x_arr), -x_arr])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "amplitude0": float(np.exp(coef[0])),
        "alpha": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
        "rmse_log": float(sqrt(ss_res / x_arr.size)),
    }


def sweep_attenuation(
    concentrations: Iterable[float],
    ice_thicknesses: Iterable[float],
    *,
    period: float = 10.0,
    depth: float = 1000.0,
    loss_factor: float = 0.02,
) -> dict[str, Any]:
    """Tabulate ``alpha`` versus ice concentration and ice thickness.

    Also returns the fitted power-law exponents ``alpha ~ n_c^p`` and
    ``alpha ~ h_i^q`` (log-log least squares) for the calibration requested by the
    campaign node.
    """
    n_list = [float(v) for v in concentrations]
    h_list = [float(v) for v in ice_thicknesses]
    if not n_list or not h_list:
        raise MIZError("concentration and thickness lists must be non-empty")
    grid = np.zeros((len(h_list), len(n_list)))
    rows: list[dict[str, float]] = []
    for i, h_i in enumerate(h_list):
        for j, n_c in enumerate(n_list):
            props = effective_medium_properties(n_c, h_i, loss_factor=loss_factor)
            k = kelvin_voigt_wavenumber(
                period, depth, props["D_real"], props["D_imag"]
            )
            alpha = attenuation_from_complex_wavenumber(k)
            grid[i, j] = alpha
            rows.append(
                {
                    "concentration": n_c,
                    "ice_thickness": h_i,
                    "alpha": alpha,
                    "k_real": float(k.real),
                    "D_eff": props["D_real"],
                }
            )

    def _exponent(xs: np.ndarray, ys: np.ndarray) -> float:
        mask = (xs > 0) & (ys > 0)
        if mask.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)[0])

    # Concentration exponent at fixed (middle) thickness.
    i_mid = len(h_list) // 2
    p_n = _exponent(np.array(n_list), grid[i_mid, :])
    j_mid = len(n_list) // 2
    q_h = _exponent(np.array(h_list), grid[:, j_mid])
    return {
        "concentrations": n_list,
        "ice_thicknesses": h_list,
        "period": float(period),
        "depth": float(depth),
        "loss_factor": float(loss_factor),
        "alpha_grid": grid,
        "rows": rows,
        "alpha_vs_concentration_exponent": p_n,
        "alpha_vs_thickness_exponent": q_h,
    }