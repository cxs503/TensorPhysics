"""Hot-spot (structural) stress extrapolation to a weld toe.

Implements the extrapolation step of the hot-spot stress method: the surface
stress is sampled at fixed fractions of the plate thickness ``t`` away from the
weld toe and extrapolated back to the toe (``d = 0``).

The extrapolation here is **exact Lagrange interpolation** of the supplied
sampling points, evaluated at ``d = 0``:

    sigma_hs = sum_i c_i * sigma(d_i),   c_i = prod_{j != i} (0 - d_j)/(d_i - d_j)

so a constant stress field is reproduced exactly (coefficients sum to 1) and a
linear field is reproduced exactly by the linear variant.  This is verified in
the test-suite.  Note: the coefficient sets printed in some code books (e.g. the
"1.875 / -1.25 / 0.375" set quoted for the 0.4t-0.9t-1.4t points) are *fitted*
rather than interpolating forms; they do not reproduce a constant field
exactly.  The interpolating coefficients used here for the same points are
2.52 / -2.24 / 0.72.  Which set is used must be stated, because the difference is
several percent of the extrapolated value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Sampling distances (in plate thicknesses) of the standard extrapolation sets.
POINTS_IIW_QUADRATIC: Tuple[float, ...] = (0.4, 0.9, 1.4)
POINTS_TASK_QUADRATIC: Tuple[float, ...] = (0.5, 1.0, 1.5)
POINTS_IIW_LINEAR: Tuple[float, ...] = (0.4, 1.0)


@dataclass
class HotSpotResult:
    """Result of a toe extrapolation."""

    hot_spot_stress: float
    method: str
    points_in_thickness: Tuple[float, ...]
    distances: np.ndarray          # sampling distances from the toe (mm)
    stresses: np.ndarray           # sampled surface stresses (MPa)
    coefficients: np.ndarray       # Lagrange coefficients
    sample_x: Optional[np.ndarray] = None   # absolute coordinates of the samples

    def as_dict(self) -> Dict[str, object]:
        return {
            "hot_spot_stress": self.hot_spot_stress,
            "method": self.method,
            "points_in_thickness": list(self.points_in_thickness),
            "distances": self.distances.tolist(),
            "sample_x": None if self.sample_x is None else self.sample_x.tolist(),
            "stresses": self.stresses.tolist(),
            "coefficients": self.coefficients.tolist(),
        }


def lagrange_extrapolate(distances: Sequence[float],
                         stresses: Sequence[float],
                         x_target: float = 0.0) -> Tuple[float, np.ndarray]:
    """Extrapolate a polynomial through ``(distances, stresses)`` to ``x_target``.

    Returns ``(value, coefficients)`` with ``value = sum(c_i * stresses_i)``.
    """
    d = np.asarray(distances, dtype=float).ravel()
    s = np.asarray(stresses, dtype=float).ravel()
    if d.size != s.size:
        raise ValueError("distances and stresses must have the same length")
    if d.size < 1:
        raise ValueError("at least one sampling point is required")
    if np.unique(np.round(d, 12)).size != d.size:
        raise ValueError("sampling distances must be distinct")
    coeffs = np.ones_like(d)
    for i in range(d.size):
        for j in range(d.size):
            if i == j:
                continue
            coeffs[i] *= (x_target - d[j]) / (d[i] - d[j])
    return float(np.dot(coeffs, s)), coeffs


def _sample_profile(distances: np.ndarray, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Linear interpolation of the surface stress profile at the given distances."""
    if x.size < 2:
        raise ValueError("stress profile needs at least two points")
    d_min, d_max = float(np.min(distances)), float(np.max(distances))
    if x[0] > d_min or x[-1] < d_max:
        raise ValueError(
            f"stress profile does not cover the sampling window "
            f"[{d_min:.3f}, {d_max:.3f}] mm (profile spans "
            f"[{x[0]:.3f}, {x[-1]:.3f}])")
    return np.interp(distances, x, sigma)


def extrapolate_from_profile(x: Sequence[float], sigma: Sequence[float],
                             x_toe: float, thickness: float,
                             points: Sequence[float] = POINTS_IIW_QUADRATIC,
                             method: Optional[str] = None) -> HotSpotResult:
    """Extrapolate a surface-stress profile measured from a weld toe.

    Parameters
    ----------
    x, sigma : sequences
        Surface coordinates (mm) and the corresponding stress component (MPa).
        ``x`` must be sorted ascending.
    x_toe : float
        Coordinate of the weld toe.
    thickness : float
        Plate thickness ``t`` (mm); the sampling distances are
        ``points[i] * thickness`` from the toe.
    points : sequence of float
        Sampling distances in thicknesses (default 0.4 / 0.9 / 1.4).
    method : str, optional
        Label recorded in the result (default derived from the point count).
    """
    x_arr = np.asarray(x, dtype=float)
    s_arr = np.asarray(sigma, dtype=float)
    if np.any(np.diff(x_arr) < 0):
        order = np.argsort(x_arr)
        x_arr, s_arr = x_arr[order], s_arr[order]
    distances = np.asarray(points, dtype=float) * float(thickness)

    # which side of the toe does the profile live on?
    if x_arr[-1] <= x_toe + 1e-12:
        side = -1.0
    elif x_arr[0] >= x_toe - 1e-12:
        side = 1.0
    else:
        raise ValueError("stress profile must lie entirely on one side of the toe")
    sample_x = x_toe + side * distances
    sampled = np.interp(sample_x, x_arr, s_arr)
    if (sample_x.min() < x_arr[0] - 1e-9) or (sample_x.max() > x_arr[-1] + 1e-9):
        raise ValueError(
            "stress profile does not cover the sampling window: samples at "
            f"x = {sample_x.round(3).tolist()} mm, profile spans "
            f"[{x_arr[0]:.3f}, {x_arr[-1]:.3f}] mm (toe at {x_toe:.3f} mm)")
    value, coeffs = lagrange_extrapolate(distances, sampled)
    label = method or {2: "linear_2point", 3: "quadratic_3point"}.get(
        len(points), f"polynomial_{len(points)}point")
    return HotSpotResult(
        hot_spot_stress=value,
        method=label,
        points_in_thickness=tuple(float(p) for p in points),
        distances=distances,
        stresses=sampled,
        coefficients=coeffs,
        sample_x=sample_x,
    )


def standard_extrapolations(x: Sequence[float], sigma: Sequence[float],
                            x_toe: float, thickness: float) -> Dict[str, HotSpotResult]:
    """Compute the standard linear and quadratic hot-spot extrapolations."""
    return {
        "iiw_quadratic_0.4_0.9_1.4": extrapolate_from_profile(
            x, sigma, x_toe, thickness, POINTS_IIW_QUADRATIC),
        "iiw_linear_0.4_1.0": extrapolate_from_profile(
            x, sigma, x_toe, thickness, POINTS_IIW_LINEAR,
            method="linear_2point"),
        "task_quadratic_0.5_1.0_1.5": extrapolate_from_profile(
            x, sigma, x_toe, thickness, POINTS_TASK_QUADRATIC),
    }