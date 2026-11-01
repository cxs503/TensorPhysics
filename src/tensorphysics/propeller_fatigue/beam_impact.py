"""Structural response of a blade span to a short ice-impact pulse.

Three routes to the same physics are provided so the fatigue chain can be
validated against a closed-form anchor:

``Beam`` + ``AnalyticImpulse``
    Euler-Bernoulli beam, modal superposition, **closed-form** response to a
    half-sine pressure pulse (``simply_supported`` or ``cantilever`` bounds).
``numerical_impulse``
    Same modal beam integrated numerically in state space with classical RK4
    (independent of the closed-form algebra).
``finite_difference_beam``
    Explicit central-difference integration of the 4th-order beam PDE on a
    finite-difference grid (fully independent discretisation; simply-supported
    anchor configuration required by acceptance criterion (a)).

Stress from curvature: ``sigma = E * c * |w''|``, ``c`` = outer-fibre distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

_CANTI_BETA_EXTRA = {3: 7.854757438, 4: 10.99554073, 5: 14.13716839, 6: 17.27875966}


def _cantilever_beta_L(n: int) -> float:
    if n == 1:
        return 1.875104069
    if n == 2:
        return 4.694091133
    if n in _CANTI_BETA_EXTRA:
        return _CANTI_BETA_EXTRA[n]
    return (2 * n - 1) * np.pi / 2.0


@dataclass
class Beam:
    """Modal database of a uniform Euler-Bernoulli beam with known E and I."""

    length: float                 # L [m]
    E: float                      # Young's modulus [Pa]
    I: float                      # second moment of area [m^4]
    mass_per_length: float        # m [kg/m]
    bc: str = "simply_supported"
    n_modes: int = 30
    outer_fibre_distance: float = 0.12   # c [m]
    beta: np.ndarray = field(init=False)
    omega: np.ndarray = field(init=False)
    modal_mass: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.bc not in ("simply_supported", "cantilever"):
            raise ValueError(f"unsupported bc {self.bc!r}")
        n = np.arange(1, self.n_modes + 1)
        if self.bc == "simply_supported":
            self.beta = n * np.pi / self.length
        else:
            self.beta = np.array([_cantilever_beta_L(int(k)) / self.length for k in n])
        self.omega = self.beta ** 2 * np.sqrt(self.E * self.I / self.mass_per_length)
        # modal mass: int phi^2 dx  ->  mL/2 for sin(n pi x/L), mL for the
        # cantilever normalisation used in shape()
        mass_factor = 0.5 if self.bc == "simply_supported" else 1.0
        self.modal_mass = np.full(self.n_modes, mass_factor * self.mass_per_length * self.length)

    # ------------------------------------------------------------- geometry
    def shape(self, x, n: int):
        """Mode shape ``phi_n(x)`` (1-based ``n``)."""
        x = np.asarray(x, dtype=float)
        b = self.beta[n - 1]
        if self.bc == "simply_supported":
            return np.sin(b * x)
        bl = b * self.length
        s = (np.cosh(bl) + np.cos(bl)) / (np.sinh(bl) + np.sin(bl))
        return np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x))

    def shape_dd(self, x, n: int):
        """Second derivative ``phi_n''(x)``."""
        x = np.asarray(x, dtype=float)
        b = self.beta[n - 1]
        if self.bc == "simply_supported":
            return -(b ** 2) * np.sin(b * x)
        bl = b * self.length
        s = (np.cosh(bl) + np.cos(bl)) / (np.sinh(bl) + np.sin(bl))
        return b ** 2 * (np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x)))

    def modal_force(self, x_load: float) -> np.ndarray:
        return np.array([float(self.shape(x_load, n)) for n in range(1, self.n_modes + 1)])

    def modal_curvature(self, x_obs: float) -> np.ndarray:
        return np.array([float(self.shape_dd(x_obs, n)) for n in range(1, self.n_modes + 1)])

    def modal_force_derivative(self, x_load: float) -> np.ndarray:
        """First derivative of the shape (needed for a moment load)."""
        return np.array([float(self._shape_d(x_load, n)) for n in range(1, self.n_modes + 1)])

    def _shape_d(self, x: float, n: int) -> float:
        b = self.beta[n - 1]
        if self.bc == "simply_supported":
            return b * np.cos(b * x)
        bl = b * self.length
        s = (np.cosh(bl) + np.cos(bl)) / (np.sinh(bl) + np.sin(bl))
        return b * (np.sinh(b * x) + np.sin(b * x) - s * (np.cosh(b * x) - np.cos(b * x)))

    # -------------------------------------------------------------- statics
    def static_curvature(self, M0: float = 0.0, P: float = 0.0,
                         x_load: Optional[float] = None,
                         x_obs: Optional[float] = None) -> float:
        """Curvature at ``x_obs`` for an end moment ``M0`` and/or point load ``P``.

        Generalised modal load: for a point force ``F`` the modal force is
        ``F phi(x_load)``; for an applied end moment ``M0`` at ``x = 0`` it is
        ``M0 phi'(0)``.
        """
        x_load = self.length / 2 if x_load is None else x_load
        x_obs = self.length / 2 if x_obs is None else x_obs
        k_n = self.modal_mass * self.omega ** 2
        gen = P * self.modal_force(x_load) + M0 * self.modal_force_derivative(0.0)
        q = gen / k_n
        return float(np.dot(q, self.modal_curvature(x_obs)))

    def static_stress(self, M0: float = 0.0, P: float = 0.0,
                      x_load: Optional[float] = None,
                      x_obs: Optional[float] = None) -> float:
        return self.E * abs(self.static_curvature(M0, P, x_load, x_obs)) * self.outer_fibre_distance

    @property
    def fundamental_period(self) -> float:
        return float(2 * np.pi / self.omega[0])

    @property
    def fundamental_frequency(self) -> float:
        return float(self.omega[0] / (2 * np.pi))


class AnalyticImpulse:
    """Closed-form response of :class:`Beam` to a half-sine pulse.

    ``P(t) = P0 sin(pi t / tau)`` for ``0 <= t <= tau`` applied at ``x_load``.

    On ``0 <= t <= tau``::

        q_n(t) = A_n [w_n sin(w t) - w sin(w_n t)] / [w_n (w_n^2 - w^2)]
        A_n    = P0 phi_n(x_load) / M_n ,   w = pi / tau
        q_n'(t) = A_n w [cos(w t) - cos(w_n t)] / (w_n^2 - w^2)

    and the free vibration after the pulse is continued analytically from the
    exact end-of-pulse state (no numerical damping, no time stepping).
    """

    def __init__(self, beam: Beam, P0: float, tau: float,
                 x_load: Optional[float] = None, x_obs: Optional[float] = None,
                 M0: float = 0.0):
        self.beam = beam
        self.P0 = float(P0)
        self.M0 = float(M0)
        self.tau = float(tau)
        self.x_load = beam.length / 2.0 if x_load is None else float(x_load)
        self.x_obs = beam.length / 2.0 if x_obs is None else float(x_obs)
        self.A = (self.P0 * beam.modal_force(self.x_load)
                  + self.M0 * beam.modal_force_derivative(0.0)) / beam.modal_mass
        self.w_pulse = np.pi / self.tau
        self.curv_w = beam.modal_curvature(self.x_obs)
        wn = beam.omega
        w = self.w_pulse
        denom = wn ** 2 - w ** 2
        bad = np.abs(denom) < 1e-9 * wn ** 2
        self._denom = np.where(bad, 1e-9 * wn ** 2, denom)
        self.near_resonance = bool(np.any(bad))

    # ---------------------------------------------------------------- state
    def q(self, t: float) -> np.ndarray:
        wn = self.beam.omega
        w = self.w_pulse
        if t <= self.tau:
            return self.A * (wn * np.sin(w * t) - w * np.sin(wn * t)) / (wn * self._denom)
        q_tau = self.A * (wn * np.sin(w * self.tau) - w * np.sin(wn * self.tau)) / (wn * self._denom)
        qd_tau = self.A * w * (np.cos(w * self.tau) - np.cos(wn * self.tau)) / self._denom
        dt = t - self.tau
        return q_tau * np.cos(wn * dt) + qd_tau / wn * np.sin(wn * dt)

    def curvature(self, t: float) -> float:
        return float(np.dot(self.q(t), self.curv_w))

    def stress(self, t: float) -> float:
        return self.beam.E * abs(self.curvature(t)) * self.beam.outer_fibre_distance

    # -------------------------------------------------------------- outputs
    def history(self, duration: Optional[float] = None,
                n_samples: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        T1 = self.beam.fundamental_period
        duration = duration if duration is not None else max(6.0 * T1, 3.0 * self.tau)
        t = np.linspace(0.0, duration, n_samples)
        sig = np.array([self.stress(tt) for tt in t])
        return t, sig

    def peak_stress(self, duration: Optional[float] = None,
                    n_samples: int = 20000) -> float:
        _, sig = self.history(duration, n_samples)
        return float(sig.max())

    def peak_time(self, duration: Optional[float] = None,
                  n_samples: int = 20000) -> float:
        t, sig = self.history(duration, n_samples)
        return float(t[int(np.argmax(sig))])

    def dynamic_load_factor(self, duration: Optional[float] = None,
                            n_samples: int = 20000) -> float:
        static = self.beam.static_stress(self.M0, self.P0, self.x_load, self.x_obs)
        return self.peak_stress(duration, n_samples) / static


# --------------------------------------------------------------------------- #
#  Numerical modal integration (state space, RK4)
# --------------------------------------------------------------------------- #
def numerical_impulse(beam: Beam, P0: float, tau: float,
                      x_load: Optional[float] = None, x_obs: Optional[float] = None,
                      duration: Optional[float] = None, dt: Optional[float] = None,
                      M0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 integration of the modal ODEs (independent of the closed form).

    ``dt`` defaults to the RK4 stability bound ``0.4 / max(omega_n)``.
    """
    x_load = beam.length / 2 if x_load is None else x_load
    x_obs = beam.length / 2 if x_obs is None else x_obs
    f_n = (P0 * beam.modal_force(x_load) + M0 * beam.modal_force_derivative(0.0))
    curv = beam.modal_curvature(x_obs)
    wn = beam.omega
    duration = duration if duration is not None else max(6.0 * beam.fundamental_period, 3.0 * tau)
    if dt is None:
        dt = 0.4 / float(wn.max())
    n = int(np.ceil(duration / dt))
    t = np.arange(n) * dt
    q = np.zeros_like(wn)
    qd = np.zeros_like(wn)
    m = beam.modal_mass
    k = m * wn ** 2
    out = np.empty(n)

    def rhs(qq, qdd, ttt):
        # f_n already carries P0 (modal load); F is the dimensionless pulse shape
        F = np.sin(np.pi * ttt / tau) if ttt <= tau else 0.0
        return qdd, (F * f_n - k * qq) / m

    for i in range(n):
        tt = t[i]
        out[i] = abs(float(np.dot(q, curv))) * beam.E * beam.outer_fibre_distance
        k1q, k1v = rhs(q, qd, tt)
        k2q, k2v = rhs(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, tt + 0.5 * dt)
        k3q, k3v = rhs(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, tt + 0.5 * dt)
        k4q, k4v = rhs(q + dt * k3q, qd + dt * k3v, tt + dt)
        q = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd = qd + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return t, out


# --------------------------------------------------------------------------- #
#  Finite-difference PDE solution (fully independent discretisation)
# --------------------------------------------------------------------------- #
def finite_difference_beam(length: float, E: float, I: float, mass_per_length: float,
                           P0: float, tau: float, outer_fibre_distance: float,
                           n_nodes: int = 61, duration: Optional[float] = None,
                           cfl: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit central-difference solution of ``m w_tt + EI w_xxxx = P(t) d(x-xp)``.

    Simply-supported ends (``w = w'' = 0``), 5-point 4th-derivative stencil,
    half-sine pulse at midspan.  Returns ``(t, sigma_mid)``.

    The time step is stability limited: ``dt = cfl * h^2 / (2 sqrt(EI/m))``.
    """
    if n_nodes % 2 == 0:
        n_nodes += 1
    EI = E * I
    h = length / (n_nodes - 1)
    alpha = EI / mass_per_length
    dt = cfl * h ** 2 / (2.0 * np.sqrt(alpha))
    T1 = 2 * np.pi * length ** 2 / (np.pi ** 2 * np.sqrt(alpha))
    duration = duration if duration is not None else max(6.0 * T1, 3.0 * tau)
    nsteps = int(np.ceil(duration / dt))
    t = np.arange(nsteps) * dt

    n = n_nodes
    mid = n // 2
    wext = np.zeros(n + 2)
    acc = np.zeros(n)
    w = np.zeros(n)
    w_prev = np.zeros(n)
    sigma = np.empty(nsteps)
    h4 = h ** 4
    nodal_force_scale = 1.0 / (mass_per_length * h)

    for step in range(nsteps):
        tt = t[step]
        sigma[step] = E * outer_fibre_distance * abs(
            (w[mid + 1] - 2.0 * w[mid] + w[mid - 1]) / h ** 2)
        # ghost nodes from the simply-supported condition w'' = 0
        wext[0] = -w[1]
        wext[1:n + 1] = w
        wext[n + 1] = -w[n - 2]
        d4 = (wext[0:n - 2] - 4.0 * wext[1:n - 1] + 6.0 * wext[2:n]
              - 4.0 * wext[3:n + 1] + wext[4:n + 2]) / h4
        acc[0] = 0.0
        acc[n - 1] = 0.0
        acc[1:n - 1] = -alpha * d4
        p = P0 * np.sin(np.pi * tt / tau) if tt <= tau else 0.0
        acc[mid] += p * nodal_force_scale
        w_new = 2.0 * w - w_prev + dt * dt * acc
        w_new[0] = 0.0
        w_new[-1] = 0.0
        w_prev, w = w, w_new
    return t, sigma
