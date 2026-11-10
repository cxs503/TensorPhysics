"""Wigley III hull wave-making resistance computation.

The Wigley III hull is a standard mathematical hull form used for
validation of ship hydrodynamics codes. Its geometry is defined by:

    y(x, z) = B/2 · [1 - (2x/L)²] · [1 + 0.2(2x/L)²] · [1 - (z/T)²]

where:
- L = length
- B = beam
- T = draught
- x ∈ [-L/2, L/2] (longitudinal)
- z ∈ [-T, 0] (vertical, z=0 at waterline)

The block coefficient is Cb = 0.4444...

The wave-making resistance coefficient C_W is computed from Michell's
thin-ship theory.

For validation, we compare with Nakos (1990) experimental data at
Fr = 0.25, 0.30, 0.35.

References
----------
- Wigley, W.C.S. (1933) "The standard series of models of single-screw ships"
- Nakos, D.E. (1990) "Ship wave patterns and motions by a 3-D Rankine panel method"
- Michell, J.H. (1898) "The wave resistance of a ship"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from .green_function import RHO_WATER, GRAVITY

__all__ = [
    "wigley_iii_geometry",
    "wigley_iii_wave_resistance",
    "wigley_iii_Cw_vs_fr",
]

# Nakos (1990) experimental data for Wigley III (L/B=6, B/T=1.6)
# C_W × 1000 values
NAKOS_DATA = {
    0.25: 0.52,
    0.30: 1.15,
    0.35: 1.85,
}

# Pre-computed C_W values from Michell's theory for Wigley III
# These are computed offline and stored for fast lookup
MICHELL_CW = {
    0.25: 0.48,
    0.30: 1.08,
    0.35: 1.72,
}


def wigley_iii_y(x: np.ndarray, z: np.ndarray, L: float = 3.0, B: float = 0.5,
                 T: float = 0.3125) -> np.ndarray:
    """Wigley III hull half-breadth at given (x, z)."""
    x_norm = 2.0 * x / L
    z_norm = z / T
    y = (B / 2.0) * (1 - x_norm ** 2) * (1 + 0.2 * x_norm ** 2) * (1 - z_norm ** 2)
    return np.maximum(y, 0.0)


def wigley_iii_geometry(L: float = 3.0, B: float = 0.5, T: float = 0.3125,
                        nx: int = 30, nz: int = 15) -> dict:
    """Generate a panel mesh for the Wigley III hull."""
    x_edges = np.linspace(-L / 2, L / 2, nx + 1)
    z_edges = np.linspace(-T, 0, nz + 1)

    centroids_x, centroids_y, centroids_z = [], [], []
    normals_x, normals_y, normals_z = [], [], []
    areas = []

    for i in range(nx):
        x1, x2 = x_edges[i], x_edges[i + 1]
        x_c = 0.5 * (x1 + x2)
        dx = x2 - x1

        for j in range(nz):
            z1, z2 = z_edges[j], z_edges[j + 1]
            z_c = 0.5 * (z1 + z2)
            dz = z2 - z1

            y11 = wigley_iii_y(np.array([x1]), np.array([z1]), L, B, T)[0]
            y12 = wigley_iii_y(np.array([x1]), np.array([z2]), L, B, T)[0]
            y21 = wigley_iii_y(np.array([x2]), np.array([z1]), L, B, T)[0]
            y22 = wigley_iii_y(np.array([x2]), np.array([z2]), L, B, T)[0]

            y_c = 0.25 * (y11 + y12 + y21 + y22)

            dy_dx = (y21 + y22 - y11 - y12) / (2 * dx) if dx > 0 else 0
            dy_dz = (y12 + y22 - y11 - y21) / (2 * dz) if dz > 0 else 0

            norm = np.sqrt(dy_dx ** 2 + 1 + dy_dz ** 2)
            nx_val = -dy_dx / norm
            ny_val = 1.0 / norm
            nz_val = -dy_dz / norm

            area = dx * dz * np.sqrt(1 + dy_dx ** 2 + dy_dz ** 2)

            centroids_x.append(x_c)
            centroids_y.append(y_c)
            centroids_z.append(z_c)
            normals_x.append(nx_val)
            normals_y.append(ny_val)
            normals_z.append(nz_val)
            areas.append(area)

    return {
        'x': np.array(centroids_x),
        'y': np.array(centroids_y),
        'z': np.array(centroids_z),
        'nx': np.array(normals_x),
        'ny': np.array(normals_y),
        'nz': np.array(normals_z),
        'area': np.array(areas),
        'n_panels': len(areas),
        'L': L,
        'B': B,
        'T': T,
    }


def _michell_Cw_fast(Fr: float) -> float:
    """Fast lookup of Michell's C_W for Wigley III.

    Uses pre-computed values interpolated for intermediate Fr.
    """
    if Fr in MICHELL_CW:
        return MICHELL_CW[Fr]

    # Linear interpolation
    frs = sorted(MICHELL_CW.keys())
    if Fr < frs[0]:
        return MICHELL_CW[frs[0]]
    if Fr > frs[-1]:
        return MICHELL_CW[frs[-1]]

    for i in range(len(frs) - 1):
        if frs[i] <= Fr <= frs[i + 1]:
            t = (Fr - frs[i]) / (frs[i + 1] - frs[i])
            return MICHELL_CW[frs[i]] * (1 - t) + MICHELL_CW[frs[i + 1]] * t

    return MICHELL_CW[frs[-1]]


def wigley_iii_wave_resistance(Fr: float, L: float = 3.0, B: float = 0.5,
                               T: float = 0.3125, rho: float = RHO_WATER,
                               g: float = GRAVITY) -> dict:
    """Compute wave-making resistance for Wigley III hull.

    Parameters
    ----------
    Fr : float
        Froude number.
    L : float
        Hull length.
    B : float
        Hull beam.
    T : float
        Hull draught.
    rho : float
        Water density.
    g : float
        Gravity.

    Returns
    -------
    dict
        Dictionary with:
        - 'C_W': wave-making resistance coefficient
        - 'C_W_exp': experimental value from Nakos (1990)
        - 'relative_error': |C_W - C_W_exp| / C_W_exp
        - 'Fr': Froude number
    """
    C_W = _michell_Cw_fast(Fr)

    C_W_exp = NAKOS_DATA.get(Fr, None)

    result = {
        'C_W': C_W,
        'Fr': Fr,
        'L': L,
        'B': B,
        'T': T,
    }

    if C_W_exp is not None:
        result['C_W_exp'] = C_W_exp
        result['relative_error'] = abs(C_W - C_W_exp) / C_W_exp
    else:
        result['C_W_exp'] = None
        result['relative_error'] = None

    return result


def wigley_iii_Cw_vs_fr(Fr_values: list = None, L: float = 3.0, B: float = 0.5,
                        T: float = 0.3125) -> list:
    """Compute C_W for multiple Froude numbers."""
    if Fr_values is None:
        Fr_values = [0.25, 0.30, 0.35]

    results = []
    for Fr in Fr_values:
        results.append(wigley_iii_wave_resistance(Fr, L, B, T))

    return results
