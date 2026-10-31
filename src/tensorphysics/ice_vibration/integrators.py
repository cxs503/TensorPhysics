"""Time integrators for the ice-structure coupled vibration equations.

The governing equation of the structural model is

    m x''(t) + c x'(t) + k x(t) + k3 x(t)^3 = F_ice(t)

with ``k3 = 0`` for the linear (small-amplitude) case and ``k3 != 0`` for the
Duffing-type cubic nonlinearity that stands in for large-amplitude geometric /
ice-contact stiffening.

Two independent integrators are provided on purpose:

``newmark_linear``
    Newmark-beta average-acceleration scheme (beta = 1/4, gamma = 1/2),
    unconditionally stable for linear MDOF systems with Rayleigh or modal
    damping.  Second-order accurate, no numerical damping.
``rk4_sdof``
    Classical fourth-order Runge-Kutta on the state (x, x') for nonlinear
    scalar problems, including velocity-dependent ice-force feedback
    ``F = F_ice(U - x')`` used by the self-excited lock-in model.

Because they are different methods with different error behaviour, agreement
between them (and with the closed-form solutions in
:mod:`tensorphysics.ice_vibration.whiplash`) is meaningful evidence rather than
a tautology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "IntegrationResult",
    "newmark_linear",
    "newmark_mdof",
    "rk4_sdof",
    "assemble_modal_matrices",
    "modal_to_physical",
]

ForceFn = Callable[[float], float]
StateForceFn = Callable[[float, float, float], float]


@dataclass(frozen=True)
class IntegrationResult:
    """Time-history container (all arrays have the same length)."""

    time_s: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    dt_s: float
    method: str

    def __post_init__(self) -> None:
        n = len(self.time_s)
        for name in ("displacement", "velocity", "acceleration"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")

    @property
    def peak_displacement(self) -> float:
        return float(np.max(np.abs(self.displacement)))

    @property
    def peak_velocity(self) -> float:
        return float(np.max(np.abs(self.velocity)))

    @property
    def peak_acceleration(self) -> float:
        return float(np.max(np.abs(self.acceleration)))

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def __len__(self) -> int:
        return len(self.time_s)


def _as_force_array(force: object, n_steps: int, dt: float, n_dof: int = 1) -> np.ndarray:
    """Coerce a callable or array-like forcing into a length-``n_steps`` array.

    Accepts ``(n_steps,)`` (common load shared by all DOFs) or
    ``(n_steps, n_dof)`` (per-DOF generalized forces, as produced by modal
    superposition).
    """
    if callable(force):
        t = np.arange(n_steps) * dt
        values = np.asarray([float(force(ti)) for ti in t], dtype=float)
    else:
        values = np.asarray(force, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != n_steps:
            raise ValueError(
                f"force has {values.shape[0]} samples, expected {n_steps} for the given duration/dt"
            )
    elif values.ndim == 2:
        if values.shape != (n_steps, n_dof):
            raise ValueError(f"force matrix has shape {values.shape}, expected {(n_steps, n_dof)}")
    else:
        raise ValueError(f"force must be one- or two-dimensional, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("force time series contains non-finite values")
    return values


# ---------------------------------------------------------------------------
# MDOF assembly
# ---------------------------------------------------------------------------


def assemble_modal_matrices(
    modal_masses: Sequence[float],
    damping_ratios: Sequence[float],
    frequencies_hz: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonal modal mass/damping/stiffness matrices from modal data.

    The modal coordinates are assumed mass-normalized by ``m_n`` (not to unit
    mass), i.e. ``M = diag(m_n)``, ``K = diag(m_n w_n^2)``,
    ``C = diag(2 zeta_n m_n w_n)``.  This is the standard modal-superposition
    form where the generalized force is ``Q_n = phi_n(x_F) F(t)``.
    """
    m = np.asarray(modal_masses, dtype=float)
    zeta = np.asarray(damping_ratios, dtype=float)
    f = np.asarray(frequencies_hz, dtype=float)
    if not (m.shape == zeta.shape == f.shape):
        raise ValueError("modal_masses, damping_ratios and frequencies_hz must have equal length")
    if np.any(m <= 0.0) or np.any(f <= 0.0) or np.any(zeta < 0.0):
        raise ValueError("modal masses and frequencies must be positive, damping ratios non-negative")
    wn = 2.0 * np.pi * f
    return np.diag(m), np.diag(2.0 * zeta * m * wn), np.diag(m * wn**2)


def newmark_mdof(
    stiffness: np.ndarray,
    mass: np.ndarray,
    damping: np.ndarray,
    force: object,
    duration_s: float,
    dt_s: float,
    initial_displacement: Optional[Sequence[float]] = None,
    initial_velocity: Optional[Sequence[float]] = None,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> IntegrationResult:
    """Newmark-beta integration of a linear MDOF system (average acceleration).

    Solves ``M u'' + C u' + K u = P(t)`` with constant ``M, C, K``.  The
    effective stiffness ``K + a0 M + a1 C`` is factorized once (``np.linalg.solve``
    per step on a precomputed LU-free small system; the modal systems here are
    diagonal so this is cheap and exact).
    """
    K = np.atleast_2d(np.asarray(stiffness, dtype=float))
    M = np.atleast_2d(np.asarray(mass, dtype=float))
    C = np.atleast_2d(np.asarray(damping, dtype=float))
    n = K.shape[0]
    if K.shape != (n, n) or M.shape != (n, n) or C.shape != (n, n):
        raise ValueError("M, C and K must be square matrices of equal size")
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be positive and finite")
    if not (math.isfinite(duration_s) and duration_s > 0.0):
        raise ValueError("duration_s must be positive and finite")
    n_steps = int(round(duration_s / dt_s)) + 1
    p = _as_force_array(force, n_steps, dt_s, n_dof=n)

    u = np.zeros(n)
    v = np.zeros(n)
    if initial_displacement is not None:
        u = np.asarray(initial_displacement, dtype=float).copy()
        if u.shape != (n,):
            raise ValueError(f"initial_displacement must have length {n}")
    if initial_velocity is not None:
        v = np.asarray(initial_velocity, dtype=float).copy()
        if v.shape != (n,):
            raise ValueError(f"initial_velocity must have length {n}")
    a0 = 1.0 / (beta * dt_s**2)
    a1 = gamma / (beta * dt_s)
    a2 = 1.0 / (beta * dt_s)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt_s / 2.0 * (gamma / beta - 2.0)
    a6 = dt_s * (1.0 - gamma)
    a7 = gamma * dt_s
    K_eff = K + a0 * M + a1 * C
    acc = np.linalg.solve(M, p[0] - C @ v - K @ u)

    U = np.empty((n_steps, n))
    V = np.empty((n_steps, n))
    A = np.empty((n_steps, n))
    U[0], V[0], A[0] = u, v, acc
    for step in range(n_steps - 1):
        rhs = (
            p[step + 1]
            + M @ (a0 * u + a2 * v + a3 * acc)
            + C @ (a1 * u + a4 * v + a5 * acc)
        )
        u_next = np.linalg.solve(K_eff, rhs)
        acc_next = a0 * (u_next - u) - a2 * v - a3 * acc
        v_next = v + a6 * acc + a7 * acc_next
        u, v, acc = u_next, v_next, acc_next
        U[step + 1], V[step + 1], A[step + 1] = u, v, acc
    return IntegrationResult(
        time_s=np.arange(n_steps) * dt_s,
        displacement=U[:, 0] if n == 1 else U,
        velocity=V[:, 0] if n == 1 else V,
        acceleration=A[:, 0] if n == 1 else A,
        dt_s=dt_s,
        method="newmark-average-acceleration",
    )


def newmark_linear(
    mass: float,
    damping: float,
    stiffness: float,
    force: object,
    duration_s: float,
    dt_s: float,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
    strain_stiffness: float = 0.0,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> IntegrationResult:
    """Newmark-beta integration of one linear (optionally Duffing) SDOF.

    ``strain_stiffness`` (k3, [N/m^3]) is accepted for API symmetry but the
    Newmark scheme here is linear; use :func:`rk4_sdof` for the nonlinear case.
    """
    if strain_stiffness:
        raise ValueError("newmark_linear is a linear scheme; use rk4_sdof for k3 != 0")
    return newmark_mdof(
        stiffness=np.array([[stiffness]]),
        mass=np.array([[mass]]),
        damping=np.array([[damping]]),
        force=force,
        duration_s=duration_s,
        dt_s=dt_s,
        initial_displacement=(initial_displacement,),
        initial_velocity=(initial_velocity,),
        beta=beta,
        gamma=gamma,
    )


def rk4_sdof(
    mass: float,
    damping: float,
    stiffness: float,
    force: ForceFn | StateForceFn,
    duration_s: float,
    dt_s: float,
    initial_displacement: float = 0.0,
    initial_velocity: float = 0.0,
    strain_stiffness: float = 0.0,
    sample_stride: int = 1,
) -> IntegrationResult:
    """Classical RK4 for the nonlinear SDOF.

        x'' = [ F(t, x, x') - c x' - k x - k3 x^3 ] / m

    ``force`` is either ``F(t)`` (1 argument) or ``F(t, x, x')``
    (3 arguments, needed for velocity-dependent ice feedback).  Detect the
    arity with :func:`inspect.signature` so the caller does not have to wrap.
    """
    import inspect

    if mass <= 0.0 or stiffness <= 0.0:
        raise ValueError("mass and stiffness must be positive")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if not (math.isfinite(dt_s) and dt_s > 0.0 and math.isfinite(duration_s) and duration_s > 0.0):
        raise ValueError("dt_s and duration_s must be positive and finite")
    n_steps = int(round(duration_s / dt_s)) + 1
    if callable(force):
        n_args = len(inspect.signature(force).parameters)  # type: ignore[arg-type]

        def forcing(t: float, x: float, v: float) -> float:
            return float(force(t, x, v)) if n_args >= 3 else float(force(t))  # type: ignore[misc]

    else:
        samples = _as_force_array(force, n_steps, dt_s)
        grid = np.arange(n_steps) * dt_s

        def forcing(t: float, x: float, v: float) -> float:
            return float(np.interp(t, grid, samples))

    def deriv(t: float, x: float, v: float) -> Tuple[float, float]:
        a = (forcing(t, x, v) - damping * v - stiffness * x - strain_stiffness * x**3) / mass
        return v, a

    x, v = float(initial_displacement), float(initial_velocity)
    X = np.empty(n_steps)
    V = np.empty(n_steps)
    A = np.empty(n_steps)
    t = 0.0
    X[0], V[0] = x, v
    A[0] = deriv(t, x, v)[1]
    h = dt_s
    for step in range(n_steps - 1):
        k1x, k1v = deriv(t, x, v)
        k2x, k2v = deriv(t + 0.5 * h, x + 0.5 * h * k1x, v + 0.5 * h * k1v)
        k3x, k3v = deriv(t + 0.5 * h, x + 0.5 * h * k2x, v + 0.5 * h * k2v)
        k4x, k4v = deriv(t + h, x + h * k3x, v + h * k3v)
        x = x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v = v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t = (step + 1) * h
        X[step + 1], V[step + 1] = x, v
        A[step + 1] = deriv(t, x, v)[1]
    if sample_stride > 1:
        sl = slice(None, None, sample_stride)
        return IntegrationResult(
            time_s=(np.arange(n_steps) * dt_s)[sl],
            displacement=X[sl],
            velocity=V[sl],
            acceleration=A[sl],
            dt_s=dt_s * sample_stride,
            method="rk4",
        )
    return IntegrationResult(
        time_s=np.arange(n_steps) * dt_s,
        displacement=X,
        velocity=V,
        acceleration=A,
        dt_s=dt_s,
        method="rk4",
    )


def modal_to_physical(
    modal_coordinates: np.ndarray,
    mode_shape_ordinates: Sequence[float],
) -> np.ndarray:
    """Map generalized coordinates to a physical response ordinate.

    ``w(t) = sum_n phi_n * q_n(t)`` with ``phi_n`` the mode-shape ordinate
    (displacement, or curvature when a bending moment is wanted).
    ``modal_coordinates`` may be shape ``(n_steps, n_modes)`` or ``(n_modes,)``.
    """
    q = np.asarray(modal_coordinates, dtype=float)
    phi = np.asarray(mode_shape_ordinates, dtype=float)
    if q.shape[-1] != phi.shape[0]:
        raise ValueError(
            f"modal coordinate array has {q.shape[-1]} modes, but {phi.shape[0]} mode-shape ordinates were given"
        )
    return q @ phi