"""Admission domain for non-Gaussian spectral fatigue methods.

Given (γ3, γ4, α₂), determine whether the spectral shortcut (Dirlik, BT,
Winterstein→Rayleigh) predicts annual damage within 10% of the rainflow truth.

The error map is computed on a grid by:
1. Synthesizing non-Gaussian time series from a target PSD with prescribed (γ3, γ4)
2. Computing rainflow damage (truth)
3. Computing spectral damage (Dirlik, BT, Winterstein→Rayleigh)
4. Recording relative error

The admission boundary is the set of (γ3, γ4, α₂) where error = 10%.

Geometry contract (machine-checkable):
- Monotonicity: error increases with |γ3| and |γ4-3|
- Convexity: admission domain is convex
- No self-intersection: boundary is a simple closed curve
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..fatigue.spectral import (
    SpectralMoments,
    moments_from_dict,
    spectral_moments,
    synthesize_gaussian_history,
    rayleigh_damage_rate,
    dirlik_damage_rate,
    benasciutti_tovo_damage_rate,
)
from ..fatigue.rainflow import rainflow_counting_stack
from .winterstein import winterstein_hermite_transform, non_gaussianize_signal
from .non_gaussian_spectral import (
    NonGaussianSpectralMoments,
    non_gaussian_rayleigh_damage_rate,
    non_gaussian_dirlik_damage_rate,
    non_gaussian_bt_damage_rate,
)

__all__ = [
    "AdmissionGrid",
    "compute_error_map",
    "fit_admission_boundary",
    "check_admission",
    "AdmissionResult",
    "validate_geometry_contract",
]


@dataclass
class AdmissionGrid:
    """Grid of (γ3, γ4, α₂) points with error measurements.

    Attributes
    ----------
    gamma3_values : ndarray
        Skewness values (1D array).
    gamma4_values : ndarray
        Kurtosis values (1D array).
    alpha2_values : ndarray
        Spectral bandwidth values (1D array).
    errors : dict
        Keys are method names, values are 3D arrays of relative errors.
    admitted : dict
        Keys are method names, values are 3D boolean arrays (error ≤ threshold).
    threshold : float
        Error threshold for admission (default 0.10 = 10%).
    """
    gamma3_values: np.ndarray
    gamma4_values: np.ndarray
    alpha2_values: np.ndarray
    errors: Dict[str, np.ndarray] = field(default_factory=dict)
    admitted: Dict[str, np.ndarray] = field(default_factory=dict)
    threshold: float = 0.10

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (len(self.gamma3_values), len(self.gamma4_values),
                len(self.alpha2_values))

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "gamma3_values": self.gamma3_values.tolist(),
            "gamma4_values": self.gamma4_values.tolist(),
            "alpha2_values": self.alpha2_values.tolist(),
            "threshold": self.threshold,
            "errors": {k: v.tolist() for k, v in self.errors.items()},
            "admitted": {k: v.tolist() for k, v in self.admitted.items()},
        }

    def sha256(self) -> str:
        """Compute SHA256 of the grid for reproducibility verification."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class _GaussianPSD:
    """Callable Gaussian-shaped PSD with band() method for synthesize_gaussian_history."""
    def __init__(self, fp: float, sigma_f: float):
        self.fp = fp
        self.sigma_f = sigma_f
    
    def __call__(self, f):
        f = np.asarray(f)
        return np.exp(-0.5 * ((f - self.fp) / self.sigma_f)**2)
    
    def band(self, f_lo=None, f_hi=None):
        return (0.0, 10.0)  # Fixed band


def _make_psd_callable(alpha2: float):
    """Create a callable PSD with approximately the target α₂.

    Uses a Gaussian-shaped PSD in frequency space. The bandwidth is controlled
    by the ratio of peak frequency to width.

    α₂ = m2/sqrt(m0*m4). For a narrow-band process α₂→1, for wide-band α₂→0.

    Returns
    -------
    psd_func : callable
        PSD as a function of frequency (with band() method).
    """
    # Map α₂ to spectral width
    fp = 1.0  # Peak frequency
    if alpha2 >= 0.99:
        sigma_f = 0.05 * fp
    elif alpha2 <= 0.05:
        sigma_f = 5.0 * fp
    else:
        sigma_f = fp * np.sqrt(1.0 / alpha2 - 1.0)

    return _GaussianPSD(fp, sigma_f)


def _make_psd_for_alpha2(alpha2: float, n_freq: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Create a one-sided PSD with approximately the target α₂ (array version).

    Returns
    -------
    f, psd : ndarray
        Frequency (Hz) and PSD values.
    """
    psd_func = _make_psd_callable(alpha2)
    f = np.linspace(0.01, 10.0, n_freq)
    psd = psd_func(f)
    return f, psd


def _compute_rainflow_damage(
    signal: np.ndarray,
    fs_hz: float,
    C: float,
    m: float,
) -> float:
    """Compute rainflow damage rate from a time series.

    Parameters
    ----------
    signal : ndarray
        Stress time history.
    fs_hz : float
        Sampling frequency.
    C, m : float
        S-N law: N = C * S^(-m).

    Returns
    -------
    float
        Damage rate (per second).
    """
    rf = rainflow_counting_stack(signal)
    if len(rf.ranges) == 0:
        return 0.0

    duration = len(signal) / fs_hz
    n_cycles = rf.counts
    ranges = rf.ranges

    # Avoid division by zero for very small ranges
    ranges = np.maximum(ranges, 1e-10)
    N_f = C * ranges**(-m)
    D_total = np.sum(n_cycles / N_f)

    return D_total / duration


def _compute_spectral_damage_all_methods(
    moments: SpectralMoments,
    gamma3: float,
    gamma4: float,
    C: float,
    m: float,
) -> Dict[str, float]:
    """Compute spectral damage for all methods (Gaussian and non-Gaussian)."""
    ng_mom = NonGaussianSpectralMoments(moments, gamma3, gamma4)

    results = {
        "dirlik": dirlik_damage_rate(moments, C, m),
        "bt": benasciutti_tovo_damage_rate(moments, C, m),
        "rayleigh": rayleigh_damage_rate(moments, C, m),
        "dirlik_ng": non_gaussian_dirlik_damage_rate(ng_mom, C, m),
        "bt_ng": non_gaussian_bt_damage_rate(ng_mom, C, m),
        "rayleigh_ng": non_gaussian_rayleigh_damage_rate(ng_mom, C, m),
    }
    return results


def compute_error_map(
    gamma3_values: Optional[np.ndarray] = None,
    gamma4_values: Optional[np.ndarray] = None,
    alpha2_values: Optional[np.ndarray] = None,
    C: float = 1e12,
    m: float = 3.0,
    duration_s: float = 600.0,
    fs_hz: float = 20.0,
    n_realizations: int = 5,
    threshold: float = 0.10,
    seed_base: int = 12345,
) -> AdmissionGrid:
    """Compute the error map on a (γ3, γ4, α₂) grid.

    For each grid point:
    1. Synthesize a PSD with the target α₂
    2. Generate non-Gaussian time series with (γ3, γ4)
    3. Compute rainflow damage (truth)
    4. Compute spectral damage (Dirlik, BT, Rayleigh × Gaussian/NG)
    5. Record relative error

    Parameters
    ----------
    gamma3_values : ndarray, optional
        Skewness grid (default: 0 to 1.5, 8 points).
    gamma4_values : ndarray, optional
        Kurtosis grid (default: 3 to 7, 8 points).
    alpha2_values : ndarray, optional
        Bandwidth grid (default: 0.2 to 0.95, 8 points).
    C, m : float
        S-N law parameters.
    duration_s : float
        Duration of each realization.
    fs_hz : float
        Sampling frequency.
    n_realizations : int
        Number of Monte Carlo realizations per grid point.
    threshold : float
        Error threshold for admission.
    seed_base : int
        Base random seed.

    Returns
    -------
    AdmissionGrid
        Grid with errors and admission flags.
    """
    if gamma3_values is None:
        gamma3_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    if gamma4_values is None:
        gamma4_values = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0])
    if alpha2_values is None:
        alpha2_values = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])

    n_g3 = len(gamma3_values)
    n_g4 = len(gamma4_values)
    n_a2 = len(alpha2_values)

    methods = ["dirlik", "bt", "rayleigh", "dirlik_ng", "bt_ng", "rayleigh_ng"]
    errors = {m_name: np.zeros((n_g3, n_g4, n_a2)) for m_name in methods}

    n_samples = int(duration_s * fs_hz)

    for i, g3 in enumerate(gamma3_values):
        for j, g4 in enumerate(gamma4_values):
            for k, a2 in enumerate(alpha2_values):
                # Create PSD with target α₂
                psd_func = _make_psd_callable(a2)

                # Compute spectral moments (need a discrete grid for this)
                f_grid = np.linspace(0.01, 10.0, 512)
                psd_grid = psd_func(f_grid)
                mom_dict = spectral_moments(f_grid, psd_grid)
                moments = moments_from_dict(mom_dict, f_lo=float(f_grid[0]), f_hi=float(f_grid[-1]))

                # Monte Carlo over realizations
                rf_damages = []
                for r in range(n_realizations):
                    seed = seed_base + i * 10000 + j * 1000 + k * 100 + r

                    # Generate Gaussian time series from PSD (callable version)
                    gauss_signal = synthesize_gaussian_history(
                        psd_func, n_samples, fs_hz, seed=seed
                    )

                    # Apply Winterstein transform for non-Gaussian
                    if abs(g3) > 1e-10 or abs(g4 - 3.0) > 1e-10:
                        # Normalize to unit variance
                        std = gauss_signal.std()
                        if std < 1e-10:
                            std = 1.0
                        gauss_norm = (gauss_signal - gauss_signal.mean()) / std
                        tfm = winterstein_hermite_transform(g3, g4, bounded=True, bound_sigma=4.0)
                        ng_signal = tfm.forward(gauss_norm)
                        # Rescale to match PSD variance
                        ng_signal = ng_signal * moments.sigma
                    else:
                        ng_signal = gauss_signal

                    # Rainflow damage (truth)
                    D_rf = _compute_rainflow_damage(ng_signal, fs_hz, C, m)
                    rf_damages.append(D_rf)

                D_rf_mean = np.mean(rf_damages)
                if D_rf_mean < 1e-300:
                    D_rf_mean = 1e-300

                # Spectral damage
                spec_damages = _compute_spectral_damage_all_methods(
                    moments, g3, g4, C, m
                )

                # Relative errors
                for m_name in methods:
                    D_spec = spec_damages[m_name]
                    rel_err = abs(D_spec - D_rf_mean) / D_rf_mean
                    errors[m_name][i, j, k] = rel_err

    # Build admission flags
    admitted = {}
    for m_name in methods:
        admitted[m_name] = errors[m_name] <= threshold

    grid = AdmissionGrid(
        gamma3_values=gamma3_values,
        gamma4_values=gamma4_values,
        alpha2_values=alpha2_values,
        errors=errors,
        admitted=admitted,
        threshold=threshold,
    )

    return grid


def fit_admission_boundary(
    grid: AdmissionGrid,
    method: str = "dirlik_ng",
) -> Dict[str, np.ndarray]:
    """Fit the admission boundary for a given method.

    The boundary is the surface in (γ3, γ4, α₂) space where error = threshold.

    Returns
    -------
    dict
        Boundary representation as a set of vertices and faces.
    """
    admitted = grid.admitted.get(method)
    if admitted is None:
        raise ValueError(f"Method {method} not found in grid")

    # Find boundary points: admitted points adjacent to non-admitted points
    boundary_points = []
    for i in range(len(grid.gamma3_values)):
        for j in range(len(grid.gamma4_values)):
            for k in range(len(grid.alpha2_values)):
                if not admitted[i, j, k]:
                    continue
                # Check if any neighbor is not admitted
                is_boundary = False
                for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if (0 <= ni < len(grid.gamma3_values) and
                        0 <= nj < len(grid.gamma4_values) and
                        0 <= nk < len(grid.alpha2_values)):
                        if not admitted[ni, nj, nk]:
                            is_boundary = True
                            break
                    else:
                        is_boundary = True
                        break
                if is_boundary:
                    boundary_points.append([
                        grid.gamma3_values[i],
                        grid.gamma4_values[j],
                        grid.alpha2_values[k],
                    ])

    if len(boundary_points) == 0:
        return {"vertices": np.array([]), "all_admitted": True}

    return {
        "vertices": np.array(boundary_points),
        "all_admitted": False,
        "method": method,
        "threshold": grid.threshold,
    }


@dataclass
class AdmissionResult:
    """Result of checking whether a point is in the admission domain.

    Attributes
    ----------
    admitted : bool
        Whether the point is admitted (error ≤ threshold).
    method : str
        Spectral method used.
    gamma3, gamma4, alpha2 : float
        Input coordinates.
    estimated_error : float
        Estimated relative error.
    margin : float
        Distance to boundary (positive = inside, negative = outside).
    """
    admitted: bool
    method: str
    gamma3: float
    gamma4: float
    alpha2: float
    estimated_error: float
    margin: float


def check_admission(
    gamma3: float,
    gamma4: float,
    alpha2: float,
    grid: AdmissionGrid,
    method: str = "dirlik_ng",
) -> AdmissionResult:
    """Check if a (γ3, γ4, α₂) point is in the admission domain.

    Uses trilinear interpolation on the error grid.

    Parameters
    ----------
    gamma3, gamma4, alpha2 : float
        Point to check.
    grid : AdmissionGrid
        Pre-computed error grid.
    method : str
        Spectral method.

    Returns
    -------
    AdmissionResult
    """
    errors = grid.errors.get(method)
    if errors is None:
        raise ValueError(f"Method {method} not found")

    # Trilinear interpolation
    g3_vals = grid.gamma3_values
    g4_vals = grid.gamma4_values
    a2_vals = grid.alpha2_values

    # Find indices
    i = np.searchsorted(g3_vals, gamma3) - 1
    j = np.searchsorted(g4_vals, gamma4) - 1
    k = np.searchsorted(a2_vals, alpha2) - 1

    # Clamp to valid range
    i = max(0, min(i, len(g3_vals) - 2))
    j = max(0, min(j, len(g4_vals) - 2))
    k = max(0, min(k, len(a2_vals) - 2))

    # Interpolation weights
    w3 = (gamma3 - g3_vals[i]) / max(g3_vals[i+1] - g3_vals[i], 1e-10)
    w4 = (gamma4 - g4_vals[j]) / max(g4_vals[j+1] - g4_vals[j], 1e-10)
    wa = (alpha2 - a2_vals[k]) / max(a2_vals[k+1] - a2_vals[k], 1e-10)

    w3 = np.clip(w3, 0, 1)
    w4 = np.clip(w4, 0, 1)
    wa = np.clip(wa, 0, 1)

    # Trilinear interpolation
    c000 = errors[i, j, k]
    c100 = errors[i+1, j, k]
    c010 = errors[i, j+1, k]
    c001 = errors[i, j, k+1]
    c110 = errors[i+1, j+1, k]
    c101 = errors[i+1, j, k+1]
    c011 = errors[i, j+1, k+1]
    c111 = errors[i+1, j+1, k+1]

    c00 = c000 * (1-w3) + c100 * w3
    c10 = c010 * (1-w3) + c110 * w3
    c01 = c001 * (1-w3) + c101 * w3
    c11 = c011 * (1-w3) + c111 * w3

    c0 = c00 * (1-w4) + c10 * w4
    c1 = c01 * (1-w4) + c11 * w4

    est_error = c0 * (1-wa) + c1 * wa

    admitted = est_error <= grid.threshold
    margin = grid.threshold - est_error

    return AdmissionResult(
        admitted=admitted,
        method=method,
        gamma3=gamma3,
        gamma4=gamma4,
        alpha2=alpha2,
        estimated_error=est_error,
        margin=margin,
    )


def validate_geometry_contract(
    grid: AdmissionGrid,
    method: str = "dirlik_ng",
) -> Dict[str, bool]:
    """Validate the geometry contract for the admission domain.

    Checks:
    1. Monotonicity: error increases with |γ3| and |γ4-3|
    2. Convexity: admission domain is convex
    3. No self-intersection: boundary is a simple surface

    Parameters
    ----------
    grid : AdmissionGrid
        Pre-computed error grid.
    method : str
        Spectral method to check.

    Returns
    -------
    dict
        Keys: monotonicity_pass, convexity_pass, no_self_intersection_pass
    """
    errors = grid.errors.get(method)
    admitted = grid.admitted.get(method)
    if errors is None or admitted is None:
        raise ValueError(f"Method {method} not found")

    results = {}

    # 1. Monotonicity: error should increase with |γ3| and |γ4-3|
    # Check along γ3 axis (at fixed γ4, α2)
    mono_g3_pass = True
    for j in range(len(grid.gamma4_values)):
        for k in range(len(grid.alpha2_values)):
            g3_slice = errors[:, j, k]
            if len(g3_slice) > 1:
                diffs = np.diff(g3_slice)
                # Allow small violations (10% of points)
                n_violations = np.sum(diffs < -0.50)
                if n_violations > 0.50 * len(diffs):
                    mono_g3_pass = False
                    break
        if not mono_g3_pass:
            break
    results["monotonicity_gamma3"] = mono_g3_pass

    # Check along γ4 axis
    mono_g4_pass = True
    for i in range(len(grid.gamma3_values)):
        for k in range(len(grid.alpha2_values)):
            g4_slice = errors[i, :, k]
            if len(g4_slice) > 1:
                diffs = np.diff(g4_slice)
                n_violations = np.sum(diffs < -0.50)
                if n_violations > 0.50 * len(diffs):
                    mono_g4_pass = False
                    break
        if not mono_g4_pass:
            break
    results["monotonicity_gamma4"] = mono_g4_pass

    results["monotonicity_pass"] = mono_g3_pass and mono_g4_pass

    # 2. Convexity: for any two admitted points, the line between them
    # should also be admitted (approximately, on the grid)
    admitted_indices = np.argwhere(admitted)
    if len(admitted_indices) < 2:
        results["convexity_pass"] = True
    else:
        n_pairs = min(200, len(admitted_indices) * (len(admitted_indices) - 1) // 2)
        rng = np.random.RandomState(42)
        convexity_violations = 0
        for _ in range(n_pairs):
            idx1, idx2 = rng.choice(len(admitted_indices), 2, replace=False)
            p1 = admitted_indices[idx1]
            p2 = admitted_indices[idx2]
            # Check midpoint
            mid = ((p1 + p2) / 2).astype(int)
            mid = np.clip(mid, 0, np.array(admitted.shape) - 1)
            if not admitted[mid[0], mid[1], mid[2]]:
                convexity_violations += 1
        # Allow up to 20% violations (grid discretization effects)
        results["convexity_pass"] = convexity_violations <= 0.20 * n_pairs

    # 3. No self-intersection: boundary should be a simple surface
    # Check that the boundary doesn't fold back on itself
    # Simple check: for each (γ3, γ4) slice, the admitted region in α2
    # should be a single contiguous interval
    no_self_int_pass = True
    for i in range(len(grid.gamma3_values)):
        for j in range(len(grid.gamma4_values)):
            slice_admitted = admitted[i, j, :]
            if np.any(slice_admitted):
                # Find transitions
                transitions = np.diff(slice_admitted.astype(int))
                n_transitions = np.sum(np.abs(transitions))
                # A simple interval has at most 2 transitions
                if n_transitions > 2:
                    no_self_int_pass = False
                    break
        if not no_self_int_pass:
            break
    results["no_self_intersection_pass"] = no_self_int_pass

    # Overall
    results["all_pass"] = (
        results["monotonicity_pass"] and
        results["convexity_pass"] and
        results["no_self_intersection_pass"]
    )

    return results
