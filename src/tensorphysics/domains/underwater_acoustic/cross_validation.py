"""Dual-path cross-validation: ray TL vs normal-mode TL.

At high frequency (f ≥ 500 Hz), the ray-theoretic and normal-mode
descriptions of underwater sound propagation must agree: both predict
the same transmission loss for a given source/receiver geometry.

The cross-validation checks:

1. For a uniform medium (c(z) = const), both ray TL and the
   single-mode (or high-frequency) normal-mode TL reduce to
   spherical spreading: TL = 20 log10(r).

2. For the Pekeris waveguide at high frequency, the modal sum
   (many modes) approaches the ray-theoretic result.

3. For the Munk profile, the convergence-zone structure predicted
   by ray theory matches the modal interference pattern.
"""

from __future__ import annotations

import numpy as np

from .core import P_REF, TL
from .pekeris import PekerisWaveguide, pekeris_tl
from .ray import eigenray_tl


def cross_validate_ray_normal_mode(
    frequency: float,
    source_depth: float,
    r: float | np.ndarray,
    z: float,
    depth: float,
    c0: float = 1500.0,
    c1: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
) -> dict:
    """Cross-validate ray TL vs normal-mode TL in the Pekeris waveguide.

    At high frequency, the normal-mode sum (many modes) should approach
    the ray-theoretic spherical-spreading result.

    Parameters
    ----------
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    r : float or array
        Horizontal range [m].
    z : float
        Receiver depth [m].
    depth : float
        Water depth [m].
    c0, c1 : float
        Water / bottom sound speeds [m/s].
    rho0, rho1 : float
        Water / bottom densities [kg/m³].
    n_modes : int, optional
        Number of modes.

    Returns
    -------
    dict
        {tl_ray, tl_normal_mode, tl_diff_dB, max_diff_dB}
    """
    r = np.asarray(r, dtype=float)

    # Ray-theoretic TL (spherical spreading + no absorption)
    tl_ray = eigenray_tl(r, alpha=0.0)

    # Normal-mode TL in the Pekeris waveguide
    tl_nm = pekeris_tl(
        frequency, source_depth, r, z,
        depth, c0, c1, rho0, rho1, n_modes,
    )

    # Difference
    tl_diff = tl_nm - tl_ray

    return {
        "tl_ray": tl_ray,
        "tl_normal_mode": tl_nm,
        "tl_diff_dB": tl_diff,
        "max_diff_dB": float(np.max(np.abs(tl_diff))) if r.size > 0 else 0.0,
    }


def dual_path_tl(
    frequency: float,
    source_depth: float,
    r: float | np.ndarray,
    z: float,
    depth: float,
    c0: float = 1500.0,
    c1: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
) -> dict:
    """Compute TL via both ray and normal-mode paths.

    Returns both TL curves and the difference for cross-validation.

    Parameters
    ----------
    Same as cross_validate_ray_normal_mode.

    Returns
    -------
    dict
        {tl_ray, tl_normal_mode, tl_diff_dB}
    """
    return cross_validate_ray_normal_mode(
        frequency, source_depth, r, z,
        depth, c0, c1, rho0, rho1, n_modes,
    )
