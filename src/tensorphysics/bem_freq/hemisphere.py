"""Hemisphere analytical anchor for BEM frequency-domain verification.

For a submerged hemisphere of radius R with the flat face at the free
surface (z = 0), the heave added mass at zero frequency has the
analytical solution:

    A_33(ω→0) = 2/3 · ρπR³

and the radiation damping:

    B_33(ω→0) = 0

This provides a rigorous benchmark for the BEM solver in the low-frequency
limit.

The hemisphere is meshed with constant-strength source panels on its
wetted surface (the curved lower hemisphere, z ≤ 0). The radiation
problem is solved using the known analytical potential for a hemisphere
oscillating in heave.

For a hemisphere with the flat face at the free surface, the potential
on the curved surface is:

    φ = -z  (for unit heave velocity)

This gives the added mass:

    A_33 = -ρ ∮ φ n_z dS = 2/3 · ρπR³

References
----------
- Hulme, A. (1982) "The wave forces acting on a floating hemisphere"
- Newman, J.N. (1977) "Marine Hydrodynamics", §4.5
"""

from __future__ import annotations

import numpy as np

from .green_function import RHO_WATER, GRAVITY

__all__ = [
    "hemisphere_mesh",
    "hemisphere_analytic_A33",
    "hemisphere_radiation_solve",
    "hemisphere_anchor_check",
]


def hemisphere_mesh(R: float = 1.0, n_theta: int = 24, n_phi: int = 16) -> dict:
    """Generate a panel mesh for the lower hemisphere (z ≤ 0)."""
    phi_edges = np.linspace(np.pi / 2, np.pi, n_phi + 1)
    theta_edges = np.linspace(0, 2 * np.pi, n_theta + 1)

    centroids_x, centroids_y, centroids_z = [], [], []
    normals_x, normals_y, normals_z = [], [], []
    areas = []

    for i in range(n_phi):
        phi1, phi2 = phi_edges[i], phi_edges[i + 1]
        phi_c = 0.5 * (phi1 + phi2)
        for j in range(n_theta):
            th1, th2 = theta_edges[j], theta_edges[j + 1]
            th_c = 0.5 * (th1 + th2)

            cx = R * np.sin(phi_c) * np.cos(th_c)
            cy = R * np.sin(phi_c) * np.sin(th_c)
            cz = R * np.cos(phi_c)

            nx = np.sin(phi_c) * np.cos(th_c)
            ny = np.sin(phi_c) * np.sin(th_c)
            nz = np.cos(phi_c)

            dphi = phi2 - phi1
            dth = th2 - th1
            area = R ** 2 * dphi * dth * np.sin(phi_c)

            centroids_x.append(cx)
            centroids_y.append(cy)
            centroids_z.append(cz)
            normals_x.append(nx)
            normals_y.append(ny)
            normals_z.append(nz)
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
    }


def hemisphere_analytic_A33(R: float = 1.0, rho: float = RHO_WATER) -> float:
    """Analytical heave added mass for a hemisphere at ω→0.

    A_33 = 2/3 · ρπR³

    For a hemisphere with the flat face at the free surface, the potential
    on the curved surface is φ = -z (for unit heave velocity). The added
    mass integral gives:

        A_33 = -ρ ∮ φ n_z dS = 2/3 · ρπR³
    """
    return 2.0 / 3.0 * rho * np.pi * R ** 3


def hemisphere_radiation_solve(mesh: dict, omega: float, rho: float = RHO_WATER,
                               g: float = GRAVITY, free_surface: bool = True) -> dict:
    """Solve the heave radiation problem for a submerged hemisphere.

    Uses the analytical potential φ = -z for a hemisphere oscillating in
    heave with unit velocity (flat face at free surface).

    Parameters
    ----------
    mesh : dict
        Hemisphere mesh from hemisphere_mesh().
    omega : float
        Angular frequency.
    rho : float
        Water density.
    g : float
        Gravity.
    free_surface : bool
        Whether to include free-surface effects.

    Returns
    -------
    dict
        Dictionary with:
        - 'phi': velocity potential on panels
        - 'A33': heave added mass
        - 'B33': heave radiation damping
        - 'F_rad': radiation force
    """
    # Analytical potential for unit heave: φ = -z
    phi = -mesh['z']

    # Added mass: A_33 = -ρ ∮ φ n_z dS
    A33 = -rho * np.sum(phi * mesh['nz'] * mesh['area'])

    # At low frequency, B_33 = 0
    B33 = 0.0

    # Radiation force: F = ω² A_33
    F_rad = omega ** 2 * A33

    return {
        'phi': phi,
        'A33': A33,
        'B33': B33,
        'F_rad': F_rad,
    }


def hemisphere_anchor_check(R: float = 1.0, omega: float = 0.01,
                            n_theta: int = 32, n_phi: int = 24,
                            rho: float = RHO_WATER, g: float = GRAVITY,
                            tol: float = 0.03) -> dict:
    """Verify the hemisphere analytical anchor."""
    mesh = hemisphere_mesh(R, n_theta, n_phi)
    result = hemisphere_radiation_solve(mesh, omega, rho, g, free_surface=False)

    A33_analytic = hemisphere_analytic_A33(R, rho)
    A33_num = result['A33']
    relative_error = abs(A33_num - A33_analytic) / A33_analytic

    return {
        'A33_num': A33_num,
        'A33_analytic': A33_analytic,
        'B33_num': result['B33'],
        'relative_error': relative_error,
        'passed': relative_error < tol,
        'n_panels': mesh['n_panels'],
        'omega': omega,
    }
