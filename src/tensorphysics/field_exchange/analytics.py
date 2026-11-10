"""Closed-form / series reference solutions used as verification anchors.

* Navier double-series solution of the simply-supported Kirchhoff plate for an
  *arbitrary piecewise-constant pressure field* (the load field handed over by
  the exchange operator).  Coefficients are evaluated analytically per panel
  (separable sine integrals), so no quadrature error pollutes the reference --
  only the series truncation, which is reported.
* Cylindrical-bending (plate strip) formulas: uniform and sinusoidal pressure.
* Hertzian contact force for a sphere on a flat.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sine_panel_integral",
    "navier_coefficients",
    "navier_center_deflection",
    "navier_field",
    "strip_uniform_pressure_center",
    "strip_sine_pressure_center",
    "hertz_force",
]


def sine_panel_integral(x0: float, x1: float, m: int, length: float) -> float:
    """``int_{x0}^{x1} sin(m pi x / L) dx`` (analytic)."""
    k = m * np.pi / length
    return float((np.cos(k * x0) - np.cos(k * x1)) / k)


def navier_coefficients(
    a: float, b: float, x_edges, y_edges, pressure: np.ndarray, m_max: int, n_max: int
) -> np.ndarray:
    """Fourier-sine coefficients ``p_mn`` of a piecewise-constant pressure field.

    ``p(x, y) = sum_m sum_n p_mn sin(m pi x/a) sin(n pi y/b)``.
    """
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    p = np.asarray(pressure, dtype=float)
    Ix = np.zeros((m_max, len(xe) - 1))
    Iy = np.zeros((n_max, len(ye) - 1))
    for m in range(1, m_max + 1):
        for pi_ in range(len(xe) - 1):
            Ix[m - 1, pi_] = sine_panel_integral(xe[pi_], xe[pi_ + 1], m, a)
    for n in range(1, n_max + 1):
        for pj in range(len(ye) - 1):
            Iy[n - 1, pj] = sine_panel_integral(ye[pj], ye[pj + 1], n, b)
    coef = np.zeros((m_max, n_max))
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            coef[m - 1, n - 1] = 4.0 / (a * b) * (Ix[m - 1] @ p @ Iy[n - 1])
    return coef


def _denominator(a: float, b: float, D: float, m: int, n: int) -> float:
    return D * np.pi ** 4 * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2


def navier_center_deflection(a: float, b: float, D: float, coef: np.ndarray) -> float:
    """Centre deflection ``w(a/2, b/2)`` from the sine coefficients."""
    w = 0.0
    for m in range(1, coef.shape[0] + 1):
        sm = np.sin(m * np.pi / 2)
        if abs(sm) < 1e-14 and m % 2 == 0:
            continue
        for n in range(1, coef.shape[1] + 1):
            sn = np.sin(n * np.pi / 2)
            if abs(sn) < 1e-14:
                continue
            w += coef[m - 1, n - 1] / _denominator(a, b, D, m, n) * sm * sn
    return float(w)


def navier_field(a: float, b: float, D: float, coef: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Deflection field on a tensor-product grid ``(len(ys), len(xs))``."""
    W = np.zeros((len(ys), len(xs)))
    for m in range(1, coef.shape[0] + 1):
        sx = np.sin(m * np.pi * np.asarray(xs) / a)
        for n in range(1, coef.shape[1] + 1):
            c = coef[m - 1, n - 1]
            if c == 0.0:
                continue
            W += (c / _denominator(a, b, D, m, n)) * np.outer(np.sin(n * np.pi * np.asarray(ys) / b), sx)
    return W


def strip_uniform_pressure_center(q: float, L: float, D: float, bc: str = "simple-simple") -> float:
    """Cylindrical bending of a plate strip under uniform pressure (exact)."""
    if bc == "simple-simple":
        return float(5.0 * q * L ** 4 / (384.0 * D))
    if bc == "clamped-clamped":
        return float(q * L ** 4 / (384.0 * D))
    raise ValueError("unknown bc")


def strip_sine_pressure_center(p0: float, L: float, D: float) -> float:
    """Simple-simple strip under ``p0 sin(pi x / L)``: ``w_c = p0 L^4/(pi^4 D)``."""
    return float(p0 * L ** 4 / (np.pi ** 4 * D))


def hertz_force(E1: float, nu1: float, E2: float, nu2: float, R: float, delta: float) -> float:
    """Hertzian contact force for a sphere of radius ``R`` pressed by ``delta``."""
    estar = 1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2)
    return float(4.0 / 3.0 * estar * np.sqrt(R) * max(delta, 0.0) ** 1.5)