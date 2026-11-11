"""NLS conservation invariants: mass, momentum, energy.

The focusing NLS i A_t + (1/2) A_xx + |A|^2 A = 0 has three exact invariants:

    Mass     M = Int |A|^2 dx
    Momentum P = Int Im( A^* A_x ) dx        (up to a constant factor)
    Energy   E = Int [ (1/2)|A_x|^2 - (1/2)|A|^4 ] dx

C3 criterion: all three invariants must drift by <= 1e-6 over 1e6 steps of the
numerical integration.  The *drift* is the absolute change in the invariant
(relative drift is reported for the mass and energy, which have non-zero
initial values; the momentum of a symmetric field is near zero so its relative
drift is not a meaningful measure -- the absolute drift is used).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "nls_mass",
    "nls_momentum",
    "nls_energy",
    "conservation_invariants",
    "conservation_drift_study",
]


def nls_mass(A: np.ndarray, dx: float) -> float:
    """Mass invariant M = Int |A|^2 dx."""
    A = np.asarray(A, dtype=complex)
    return float(np.sum(np.abs(A) ** 2) * dx)


def nls_momentum(A: np.ndarray, dx: float) -> float:
    """Momentum invariant P = Int Im( A^* A_x ) dx.

    Computed via the spectral derivative (exact on the periodic grid).
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    A_hat = np.fft.fft(A)
    A_x = np.fft.ifft(1j * k * A_hat)
    return float(np.sum(np.imag(np.conj(A) * A_x)) * dx)


def nls_energy(A: np.ndarray, dx: float) -> float:
    """Energy invariant E = Int [ (1/2)|A_x|^2 - (1/2)|A|^4 ] dx."""
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    A_hat = np.fft.fft(A)
    A_x = np.fft.ifft(1j * k * A_hat)
    return float(np.sum(0.5 * np.abs(A_x) ** 2 - 0.5 * np.abs(A) ** 4) * dx)


def conservation_invariants(A: np.ndarray, dx: float) -> dict:
    """Return the three NLS invariants of a single envelope."""
    return {
        "mass": nls_mass(A, dx),
        "momentum": nls_momentum(A, dx),
        "energy": nls_energy(A, dx),
    }


def conservation_drift_study(
    A_traj: np.ndarray,
    dx: float,
    t: np.ndarray,
) -> dict:
    """Drift of the three invariants over a trajectory.

    Parameters
    ----------
    A_traj : complex ndarray, shape (n_t, n_x)
        Envelope at each recorded time.
    dx : float
        Spatial step.
    t : ndarray
        Time grid.

    Returns
    -------
    dict
        With keys: mass, momentum, energy (each a dict of initial, final,
        absolute drift, relative drift), and max_abs_drift (the maximum
        absolute drift -- the C3 criterion metric).
    """
    A_traj = np.asarray(A_traj, dtype=complex)
    n_t = A_traj.shape[0]
    mass = np.array([nls_mass(A_traj[i], dx) for i in range(n_t)])
    mom = np.array([nls_momentum(A_traj[i], dx) for i in range(n_t)])
    ene = np.array([nls_energy(A_traj[i], dx) for i in range(n_t)])

    def _drift(vals):
        init = float(vals[0])
        fin = float(vals[-1])
        abs_drift = abs(fin - init)
        rel_drift = abs_drift / abs(init) if abs(init) > 1e-30 else abs_drift
        return {"initial": init, "final": fin, "abs_drift": abs_drift,
                "rel_drift": rel_drift}

    d_mass = _drift(mass)
    d_mom = _drift(mom)
    d_ene = _drift(ene)
    # the C3 criterion uses the absolute drift (the relative drift of the
    # momentum is not meaningful for a near-zero initial value)
    max_abs = max(d_mass["abs_drift"], d_mom["abs_drift"], d_ene["abs_drift"])
    return {
        "mass": d_mass,
        "momentum": d_mom,
        "energy": d_ene,
        "max_abs_drift": max_abs,
        "max_relative_drift": max(
            d_mass["rel_drift"], d_mom["rel_drift"], d_ene["rel_drift"]),
        "n_t": int(n_t),
        "t_final": float(t[-1]),
    }
