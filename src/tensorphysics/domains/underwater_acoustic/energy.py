"""Energy conservation for underwater acoustic propagation.

Checks that the total acoustic energy in the field matches the source
power.

For a point source in a lossless waveguide, the total power crossing
any cylindrical surface at range r is:

    P(r) = ∫_0^H |p(r,z)|² / (ρ0 c0) · 2πr · dz

For a correctly normalized modal sum, the average power over a range
interval should equal the source power (sum of modal powers).

The closure check computes:

    closure = |P_avg - P_source| / P_source

where P_avg is the range-averaged field power and P_source is the
sum of modal powers.  Closure ≤ 1% is required for acceptance.
"""

from __future__ import annotations

import numpy as np

from .core import RHO_WATER, P_REF
from .pekeris import PekerisWaveguide


def _source_power(
    wg: PekerisWaveguide,
    frequency: float,
    source_depth: float,
) -> float:
    """Compute the source power from the modal amplitudes.

    P_source = Σ_n |A_n|² |ψ_n(z_s)|² / (2 k_n) · (1/ρ0) · (1/2π)

    Parameters
    ----------
    wg : PekerisWaveguide
        Waveguide.
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].

    Returns
    -------
    float
        Source power [W].
    """
    omega = 2.0 * np.pi * frequency
    k0 = omega / wg.c0

    eigenvalues = wg.mode_eigenvalues(frequency)
    amplitudes = wg.mode_amplitudes(frequency)

    P_source = 0.0
    for k_n, A_n in zip(eigenvalues, amplitudes):
        gamma0 = np.sqrt(k_n**2 - k0**2)
        psi_s = np.sin(gamma0 * source_depth)
        # Mode power
        P_n = (A_n**2 * psi_s**2) / (2.0 * k_n) * (1.0 / RHO_WATER) * (1.0 / (2.0 * np.pi))
        P_source += P_n

    return float(P_source)


def _field_power(
    wg: PekerisWaveguide,
    frequency: float,
    source_depth: float,
    r: float,
    n_z: int = 200,
) -> float:
    """Compute the field power crossing a cylindrical surface at range r.

    P(r) = ∫_0^H |p(r,z)|² / (ρ0 c0) · 2πr · dz

    Parameters
    ----------
    wg : PekerisWaveguide
        Waveguide.
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    r : float
        Range [m].
    n_z : int
        Number of depth points for the integral.

    Returns
    -------
    float
        Field power [W].
    """
    omega = 2.0 * np.pi * frequency
    k0 = omega / wg.c0

    eigenvalues = wg.mode_eigenvalues(frequency)
    amplitudes = wg.mode_amplitudes(frequency)

    # Depth grid
    z = np.linspace(0.0, wg.depth, n_z)

    # Field at range r as a function of depth
    p_field = np.zeros(n_z, dtype=complex)
    for k_n, A_n in zip(eigenvalues, amplitudes):
        gamma0 = np.sqrt(k_n**2 - k0**2)
        psi_s = np.sin(gamma0 * source_depth)
        psi_z = np.sin(gamma0 * z)
        p_n = A_n * psi_s * psi_z * np.exp(1j * k_n * r) / np.sqrt(k_n * r)
        p_field += p_n

    # Power: P = ∫|p|² / (ρ0 c0) · 2πr · dz
    p_abs_sq = np.abs(p_field)**2
    P = np.trapezoid(p_abs_sq, z) * 2.0 * np.pi * r / (RHO_WATER * wg.c0)
    return float(P)


def modal_energy_closure(
    frequency: float,
    source_depth: float,
    depth: float,
    c0: float = 1500.0,
    c1: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
    r_min: float = 1000.0,
    r_max: float = 50000.0,
    n_r: int = 20,
    n_z: int = 200,
) -> float:
    """Energy closure: range-averaged field power vs source power.

    Computes the average field power over a range interval and compares
    it to the source power (sum of modal powers).

    Parameters
    ----------
    frequency : float
        Source frequency [Hz].
    source_depth : float
        Source depth [m].
    depth : float
        Water depth [m].
    c0, c1 : float
        Water / bottom sound speeds [m/s].
    rho0, rho1 : float
        Water / bottom densities [kg/m³].
    n_modes : int, optional
        Number of modes.
    r_min, r_max : float
        Range interval for the power average [m].
    n_r : int
        Number of range points.
    n_z : int
        Number of depth points.

    Returns
    -------
    float
        Energy closure (relative difference, ≤ 1% for acceptance).
    """
    wg = PekerisWaveguide(depth=depth, c0=c0, c1=c1, rho0=rho0, rho1=rho1)

    # Source power
    P_source = _source_power(wg, frequency, source_depth)

    # Average field power over the range interval
    r = np.linspace(r_min, r_max, n_r)
    P_field = np.array([_field_power(wg, frequency, source_depth, rr, n_z) for rr in r])
    P_avg = np.mean(P_field)

    if P_source < 1e-30:
        return 1.0  # no modes → no energy (degenerate)
    closure = abs(P_avg - P_source) / P_source
    return float(closure)


def energy_conservation_check(
    frequency: float,
    source_depth: float,
    depth: float,
    c0: float = 1500.0,
    c1: float = 1450.0,
    rho0: float = 1025.0,
    rho1: float = 2000.0,
    n_modes: int | None = None,
    r_min: float = 1000.0,
    r_max: float = 50000.0,
    n_r: int = 20,
    n_z: int = 200,
    tol: float = 0.01,
) -> bool:
    """Check energy conservation (closure ≤ 1%).

    Parameters
    ----------
    Same as modal_energy_closure, plus:
    tol : float
        Acceptance tolerance (default 1%).

    Returns
    -------
    bool
        True if energy closure ≤ tol.
    """
    closure = modal_energy_closure(
        frequency, source_depth, depth,
        c0, c1, rho0, rho1, n_modes, r_min, r_max, n_r, n_z,
    )
    return closure <= tol
