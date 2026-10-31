"""mu(I) inertial-number rheology for granular / brash-ice shear flows.

Definitions (da Cruz et al. 2005; Jop, Forterre & Pouliquen 2006):

    I = gammadot * d / sqrt(p / rho_s)          dimensionless inertial number
    mu(I) = mu_s + (mu_2 - mu_s) / (I_0 / I + 1)   effective friction law
    mu = |tau| / p                               stress ratio

The fit is linear in the two stiffness parameters, so it is solved by variable
projection rather than a general non-linear optimiser: for fixed ``I_0`` the model
is ``mu = a + b / (I_0 + I)`` with ``a = mu_2`` and ``b = I_0 (mu_s - mu_2)``,
which is an ordinary least-squares problem in ``(a, b)``. ``I_0`` is scanned on a
logarithmic grid and refined by golden-section search; the reported ``R^2`` is the
coefficient of determination of the best combination.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any

import numpy as np


class MuIError(ValueError):
    """Invalid input for the mu(I) rheology model."""


def _positive(name: str, value: Any) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise MuIError(f"{name} must be finite and positive, got {value!r}")
    return value


def inertial_number(shear_rate: float, grain_diameter: float, pressure: float,
                    density: float) -> float:
    """``I = gammadot d / sqrt(p / rho_s)``.

    ``pressure`` is the 2-D (per unit thickness) pressure in N/m when working with
    the 2-D shear cell; ``density`` must then be the 2-D density in kg/m^2.
    """
    gamma = _positive("shear_rate", shear_rate)
    d = _positive("grain_diameter", grain_diameter)
    p = _positive("pressure", pressure)
    rho = _positive("density", density)
    return gamma * d / sqrt(p / rho)


def mu_jop(I: Any, mu_s: float = 0.2, mu_2: float = 0.4, I_0: float = 0.3) -> np.ndarray:
    """Effective friction coefficient from the Jop-Forterre-Pouliquen law."""
    mu_s = float(mu_s)
    mu_2 = float(mu_2)
    I_0 = _positive("I_0", I_0)
    if not isfinite(mu_s) or not isfinite(mu_2):
        raise MuIError("mu_s and mu_2 must be finite")
    if mu_2 < mu_s:
        raise MuIError(f"mu_2 ({mu_2}) must be >= mu_s ({mu_s})")
    I_arr = np.asarray(I, dtype=float)
    if np.any(I_arr < 0.0) or not np.all(np.isfinite(I_arr)):
        raise MuIError("I must be finite and non-negative")
    return mu_s + (mu_2 - mu_s) / (I_0 / np.maximum(I_arr, 1e-300) + 1.0)


def _fit_for_i0(I: np.ndarray, mu: np.ndarray, I_0: float) -> tuple[float, float, float]:
    """Least-squares ``mu = a + b/(I_0 + I)`` for a fixed ``I_0``."""
    basis = np.column_stack([np.ones_like(I), 1.0 / (I_0 + I)])
    coef, *_ = np.linalg.lstsq(basis, mu, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    mu_s = a + b / I_0
    mu_2 = a
    resid = mu - basis @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((mu - mu.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    if not (0.0 <= mu_s <= mu_2):
        # Parameter set is not admissible for a shear-thickening mu(I) law; report
        # an impossible score so the search never selects it.
        r2 = -np.inf
    return mu_s, mu_2, r2


def fit_mu_i(I: Any, mu: Any, *, i0_bounds: tuple[float, float] = (1e-4, 10.0),
             n_scan: int = 400) -> dict[str, float]:
    """Calibrate ``mu_s``, ``mu_2`` and ``I_0`` from (I, mu) pairs.

    Returns a dict with ``mu_s``, ``mu_2``, ``I_0``, ``r2``, ``rmse``,
    ``n_points`` and the residual array under ``residuals``.
    """
    I_arr = np.asarray(I, dtype=float).ravel()
    mu_arr = np.asarray(mu, dtype=float).ravel()
    if I_arr.shape != mu_arr.shape:
        raise MuIError(f"I and mu must have the same length, got {I_arr.shape} vs {mu_arr.shape}")
    if I_arr.size < 4:
        raise MuIError(f"at least 4 data pairs are required, got {I_arr.size}")
    if np.any(I_arr <= 0.0) or not np.all(np.isfinite(I_arr)):
        raise MuIError("I must be finite and positive for the log-scan")
    if not np.all(np.isfinite(mu_arr)):
        raise MuIError("mu must be finite")

    lo, hi = (float(i0_bounds[0]), float(i0_bounds[1]))
    if not (0.0 < lo < hi):
        raise MuIError(f"invalid i0_bounds {i0_bounds!r}")

    grid = np.logspace(np.log10(lo), np.log10(hi), int(n_scan))
    scores = np.array([_fit_for_i0(I_arr, mu_arr, float(i0))[2] for i0 in grid])
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        raise MuIError(
            "no admissible (mu_s <= mu_2, mu_s >= 0) parameter set fits the data; "
            "the measured mu(I) data are not consistent with the shear-thickening "
            "Jop law (refusing to fabricate a calibration)"
        )
    # Golden-section refinement around the best grid point.
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, grid.size - 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = left, right
    for _ in range(80):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if _fit_for_i0(I_arr, mu_arr, c)[2] < _fit_for_i0(I_arr, mu_arr, d)[2]:
            a = c
        else:
            b = d
    I_0 = float(np.clip(0.5 * (a + b), lo, hi))
    mu_s, mu_2, r2 = _fit_for_i0(I_arr, mu_arr, I_0)
    model = mu_jop(I_arr, mu_s=mu_s, mu_2=mu_2, I_0=I_0)
    residuals = mu_arr - model
    return {
        "mu_s": mu_s,
        "mu_2": mu_2,
        "I_0": I_0,
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "n_points": int(I_arr.size),
        "residuals": residuals,
    }