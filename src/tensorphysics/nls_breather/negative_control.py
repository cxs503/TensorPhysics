"""Negative control: as the nonlinearity parameter epsilon -> 0 the NLS
solution degenerates to the linear spectral synthesis (C7).

The focusing NLS with a small nonlinearity parameter epsilon is

    i A_t + (1/2) A_xx + epsilon |A|^2 A = 0.

As epsilon -> 0 the equation becomes the linear Schrodinger equation, whose
solution is the linear superposition of Fourier modes -- i.e. the *linear*
spectral synthesis.  The C7 criterion: the NLS solution at small epsilon must
match the linear spectral synthesis to <= 1%.
"""

from __future__ import annotations

import numpy as np

from .nls_solver import split_step_fourier_step

__all__ = [
    "linear_degeneration_check",
    "nls_linear_limit",
    "epsilon_linear_spectral_error",
]


def _linear_schrodinger_solution(
    A0: np.ndarray,
    k: np.ndarray,
    t: float,
) -> np.ndarray:
    """Exact solution of the linear Schrodinger equation i A_t + (1/2) A_xx = 0.

    A(x, t) = F^{-1}[ e^{-i (1/2) k^2 t} F[A0] ].
    """
    A_hat = np.fft.fft(A0)
    A_hat = A_hat * np.exp(-0.5j * k ** 2 * t)
    return np.fft.ifft(A_hat)


def nls_linear_limit(
    A0: np.ndarray,
    k: np.ndarray,
    dt: float,
    n_steps: int,
    epsilon: float,
) -> np.ndarray:
    """NLS solution at small nonlinearity epsilon (via the split-step with the
    nonlinear term scaled by epsilon)."""
    A = A0.copy()
    for _ in range(n_steps):
        # nonlinear phase rotation scaled by epsilon
        A_nl = A * np.exp(1j * epsilon * np.abs(A) ** 2 * dt)
        A_hat = np.fft.fft(A_nl)
        A_hat = A_hat * np.exp(-0.5j * k ** 2 * dt)
        A = np.fft.ifft(A_hat)
    return A


def epsilon_linear_spectral_error(
    A0: np.ndarray,
    k: np.ndarray,
    dt: float,
    n_steps: int,
    epsilon: float,
) -> float:
    """Relative L2 error of the small-epsilon NLS solution vs the linear
    Schrodinger solution at the same time.

    Parameters
    ----------
    A0 : complex ndarray
        Initial envelope.
    k : ndarray
        Wavenumber grid.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    epsilon : float
        Nonlinearity parameter.

    Returns
    -------
    float
        Relative L2 error (0 = exact linear).
    """
    t_final = n_steps * dt
    A_nls = nls_linear_limit(A0, k, dt, n_steps, epsilon)
    A_lin = _linear_schrodinger_solution(A0, k, t_final)
    num = np.sqrt(np.sum(np.abs(A_nls - A_lin) ** 2))
    den = np.sqrt(np.sum(np.abs(A_lin) ** 2))
    if den < 1e-30:
        return 0.0
    return float(num / den)


def linear_degeneration_check(
    A0: np.ndarray,
    k: np.ndarray,
    dt: float,
    n_steps: int,
    epsilons: np.ndarray,
    tol: float = 0.01,
) -> dict:
    """Negative-control check: the NLS solution at small epsilon matches the
    linear spectral synthesis to <= tol (default 1%).

    Parameters
    ----------
    A0 : complex ndarray
        Initial envelope.
    k : ndarray
        Wavenumber grid.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    epsilons : ndarray
        Nonlinearity parameters to scan (small values).
    tol : float
        Tolerance (default 0.01 = 1%).

    Returns
    -------
    dict
        With keys: epsilons, errors, max_error, verdict.
    """
    eps = np.asarray(epsilons, dtype=float)
    errors = np.array([epsilon_linear_spectral_error(A0, k, dt, n_steps, e)
                       for e in eps])
    max_err = float(np.max(errors))
    verdict = "PASS" if max_err <= tol else "FAIL"
    return {
        "epsilons": eps.tolist(),
        "errors": errors.tolist(),
        "max_error": max_err,
        "verdict": verdict,
        "tolerance": float(tol),
    }
