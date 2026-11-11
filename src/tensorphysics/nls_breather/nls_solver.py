"""Pseudo-spectral split-step Fourier solver for the one-dimensional focusing
NLS (Zakharov-reduction spectral form).

The deep-water wave envelope A(x, t) obeys

    i A_t + (1/2) A_xx + |A|^2 A = 0

(carrier omega_0 = 1, k_0 = 1, unit background).  The split-step Fourier (SSF)
method advances the solution by alternating the exact linear propagator in
Fourier space with the pointwise nonlinear phase rotation.

The *Strang* (symmetric, second-order) splitting is used:

    A(x, t+dt) = F^{-1}[ e^{-i (1/2) k^2 (dt/2)}
                         F[ e^{i |A|^2 dt}
                            F^{-1}[ e^{-i (1/2) k^2 (dt/2)} F[A(x, t)] ] ] ],

i.e. linear half-step, nonlinear full-step, linear half-step.  This is
second-order accurate in time and conserves the three NLS invariants (mass,
momentum, energy) to O(dt^2) per step -- the basis of the C3 criterion
(drift <= 1e-6 over 1e6 steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "NLSSolver",
    "NLSResult",
    "split_step_fourier_step",
    "nls_spectral_solve",
]


def split_step_fourier_step(
    A: np.ndarray,
    k: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One symmetric (Strang) split-step Fourier step of the focusing NLS.

    Parameters
    ----------
    A : complex ndarray
        Current envelope on the spatial grid.
    k : ndarray
        Wavenumber grid (Fourier-space, including the Nyquist ordering).
    dt : float
        Time step.

    Returns
    -------
    complex ndarray
        Envelope advanced by dt.
    """
    # linear half-step
    A_hat = np.fft.fft(A)
    A_hat = A_hat * np.exp(-0.5j * k ** 2 * (0.5 * dt))
    A_half = np.fft.ifft(A_hat)
    # nonlinear full-step (phase rotation)
    A_nl = A_half * np.exp(1j * np.abs(A_half) ** 2 * dt)
    # linear half-step
    A_hat = np.fft.fft(A_nl)
    A_hat = A_hat * np.exp(-0.5j * k ** 2 * (0.5 * dt))
    return np.fft.ifft(A_hat)


@dataclass
class NLSResult:
    """Result of an NLS spectral solve.

    Attributes
    ----------
    x : ndarray
        Spatial grid.
    t : ndarray
        Time grid (absolute, starting at t0).
    A : complex ndarray, shape (n_t, n_x)
        Envelope at each time.
    k : ndarray
        Wavenumber grid.
    """
    x: np.ndarray
    t: np.ndarray
    A: np.ndarray
    k: np.ndarray

    def envelope_at(self, t_idx: int) -> np.ndarray:
        """Envelope at a given time index."""
        return self.A[t_idx]

    def amplitude(self, t_idx: int) -> np.ndarray:
        """|A| at a given time index."""
        return np.abs(self.A[t_idx])

    def peak_amplitude(self) -> float:
        """Global peak |A| over the whole run."""
        return float(np.max(np.abs(self.A)))


@dataclass
class NLSSolver:
    """Pseudo-spectral split-step Fourier NLS solver.

    Parameters
    ----------
    n_x : int
        Number of spatial points (power of two preferred).
    L : float
        Spatial domain half-width (domain is [-L, L]).
    dt : float
        Time step.
    n_t : int
        Number of time steps.
    """

    n_x: int = 512
    L: float = 20.0
    dt: float = 1e-3
    n_t: int = 2000

    def __post_init__(self) -> None:
        if self.n_x < 16:
            raise ValueError("n_x must be >= 16")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.n_t < 1:
            raise ValueError("n_t must be >= 1")
        # spatial grid
        self._x = np.linspace(-self.L, self.L, self.n_x, endpoint=False)
        self._dx = self._x[1] - self._x[0]
        # wavenumber grid (fftfreq ordering)
        self._k = 2.0 * np.pi * np.fft.fftfreq(self.n_x, d=self._dx)

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def k(self) -> np.ndarray:
        return self._k

    @property
    def dx(self) -> float:
        return self._dx

    def solve(
        self,
        A0: np.ndarray,
        n_steps: Optional[int] = None,
        record_every: int = 1,
        t0: float = 0.0,
    ) -> NLSResult:
        """Advance the NLS from initial condition A0.

        Parameters
        ----------
        A0 : complex ndarray
            Initial envelope on the spatial grid (length n_x).
        n_steps : int, optional
            Number of time steps (default: self.n_t).
        record_every : int
            Record the envelope every this many steps.
        t0 : float
            Initial (absolute) time of the initial condition.

        Returns
        -------
        NLSResult
        """
        A0 = np.asarray(A0, dtype=complex)
        if A0.shape[0] != self.n_x:
            raise ValueError(f"A0 must have length {self.n_x}, got {A0.shape[0]}")
        n_steps = int(n_steps if n_steps is not None else self.n_t)
        n_rec = n_steps // record_every + 1
        A = np.empty((n_rec, self.n_x), dtype=complex)
        t = np.empty(n_rec)
        A[0] = A0
        t[0] = t0
        cur = A0.copy()
        for i in range(1, n_steps + 1):
            cur = split_step_fourier_step(cur, self._k, self.dt)
            if i % record_every == 0:
                A[i // record_every] = cur
                t[i // record_every] = t0 + i * self.dt
        return NLSResult(x=self._x, t=t, A=A, k=self._k)


def nls_spectral_solve(
    A0: np.ndarray,
    x: np.ndarray,
    dt: float,
    n_steps: int,
    record_every: int = 1,
    t0: float = 0.0,
) -> NLSResult:
    """Convenience one-call NLS spectral solve.

    Parameters
    ----------
    A0 : complex ndarray
        Initial envelope.
    x : ndarray
        Spatial grid.
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    record_every : int
        Record every this many steps.
    t0 : float
        Initial (absolute) time.

    Returns
    -------
    NLSResult
    """
    n_x = len(x)
    solver = NLSSolver(n_x=n_x, L=float(np.max(np.abs(x))), dt=dt, n_t=n_steps)
    return solver.solve(A0, n_steps=n_steps, record_every=record_every, t0=t0)
