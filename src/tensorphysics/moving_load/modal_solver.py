"""Modal-superposition time integration of the moving-load equation.

Solves the beam equation

    M u'' + C u' + K u = F(x, t)

by projecting on the first ``n_modes`` SS modes, which yields a set of
decoupled scalar equations.  The time integration is the Newmark average
acceleration scheme (``beta = 1/4``, ``gamma = 1/2``) applied mode by mode --
vectorised over the modal axis -- which is unconditionally stable for the
linear systems considered here and second-order accurate with no numerical
damping.

Two bookkeeping quantities are produced alongside the response because they are
acceptance criteria in their own right:

* the modal mechanical energy ``E(t) = 0.5 sum(M_n qd_n^2 + K_n q_n^2)``,
* the work injected by the moving load ``W(t) = int_0^t sum(Q_n qd_n) dtau``.

For an undamped (or Rayleigh-damped, zero initial energy) system the power
balance ``E(t) = W(t)`` must hold to time-discretisation accuracy; the residual
is reported as the energy-conservation criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .beam import HullGirderBeam
from .paths import LoadPath, modal_force_matrix, total_load_series

__all__ = ["ModalResponse", "solve_modal_response", "modal_matrices"]


@dataclass
class ModalResponse:
    """Container for a modal-superposition moving-load solution."""

    time_s: np.ndarray
    q: np.ndarray          # (n_time, n_modes) modal displacement
    qd: np.ndarray         # modal velocity
    qdd: np.ndarray        # modal acceleration
    modal_force: np.ndarray
    energy_j: np.ndarray       # modal mechanical energy
    work_j: np.ndarray         # cumulative work injected by the load
    dissipated_j: np.ndarray   # cumulative energy removed by modal damping
    beam: HullGirderBeam
    n_modes: int
    dt_s: float
    damping_ratio: float
    load_total_n: np.ndarray

    # -- derived physical fields -------------------------------------------
    def _phi(self, x_m: float) -> np.ndarray:
        n = np.arange(1, self.n_modes + 1, dtype=float)
        return np.sin(n * np.pi * float(x_m) / self.beam.length_m)

    def displacement_at(self, x_m: float) -> np.ndarray:
        return self.q @ self._phi(x_m)

    def velocity_at(self, x_m: float) -> np.ndarray:
        return self.qd @ self._phi(x_m)

    def acceleration_at(self, x_m: float) -> np.ndarray:
        return self.qdd @ self._phi(x_m)

    def deflection_profile(self, t_index: int, n_points: int = 201) -> tuple:
        x = np.linspace(0.0, self.beam.length_m, n_points)
        n = np.arange(1, self.n_modes + 1, dtype=float)
        phi = np.sin(n[:, None] * np.pi * x[None, :] / self.beam.length_m)
        return x, self.q[t_index, :] @ phi

    @property
    def peak_modal_energy_j(self) -> float:
        return float(np.max(np.abs(self.energy_j)))

    def energy_residual(self) -> dict:
        """Relative power balance ``E_mech + E_dissipated = W_injected``.

        For an undamped system this reduces to plain mechanical-energy
        conservation; with damping the cumulative dissipated energy
        ``int c_n qd_n^2 dt`` is carried on the same balance, so the residual
        stays a pure time-discretisation measure in both cases.
        """
        scale = max(
            np.max(np.abs(self.work_j)),
            np.max(np.abs(self.energy_j)),
            np.max(np.abs(self.dissipated_j)),
            1e-30,
        )
        diff = self.energy_j + self.dissipated_j - self.work_j
        return {
            "max_abs_residual_j": float(np.max(np.abs(diff))),
            "peak_energy_j": float(np.max(np.abs(self.energy_j))),
            "peak_work_j": float(np.max(np.abs(self.work_j))),
            "peak_dissipated_j": float(np.max(np.abs(self.dissipated_j))),
            "relative_residual": float(np.max(np.abs(diff)) / scale),
        }


def modal_matrices(beam: HullGirderBeam, n_modes: int, damping_ratio: float = 0.0) -> tuple:
    """Diagonal modal mass / stiffness / damping matrices (arrays, in order)."""
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if damping_ratio < 0.0:
        raise ValueError("damping_ratio must be non-negative")
    m = np.array([beam.modal_mass(n) for n in range(1, n_modes + 1)], dtype=float)
    w = np.array([beam.omega(n) for n in range(1, n_modes + 1)], dtype=float)
    k = m * w ** 2
    c = 2.0 * float(damping_ratio) * m * w
    return m, k, c


def solve_modal_response(
    beam: HullGirderBeam,
    paths: Sequence[LoadPath],
    duration_s: float,
    dt_s: float,
    n_modes: int = 10,
    damping_ratio: float = 0.0,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> ModalResponse:
    """Integrate the modal system under a set of moving load paths.

    Initial conditions are at rest (``q = qd = 0``), matching the closed-form
    reference in :mod:`tensorphysics.moving_load.analytic`.
    """
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be positive and finite")

    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.arange(n_steps) * dt_s
    Q = modal_force_matrix(beam, paths, time_s, n_modes)
    m, k, c = modal_matrices(beam, n_modes, damping_ratio)

    a0 = 1.0 / (beta * dt_s ** 2)
    a1 = gamma / (beta * dt_s)
    a2 = 1.0 / (beta * dt_s)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt_s * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 * m + a1 * c

    q = np.zeros(n_modes)
    qd = np.zeros(n_modes)
    qdd = np.zeros(n_modes)

    q_hist = np.zeros((n_steps, n_modes))
    qd_hist = np.zeros((n_steps, n_modes))
    qdd_hist = np.zeros((n_steps, n_modes))
    energy = np.zeros(n_steps)
    work = np.zeros(n_steps)
    dissipated = np.zeros(n_steps)

    energy[0] = 0.5 * float(np.sum(m * qd ** 2 + k * q ** 2))
    for i in range(n_steps - 1):
        p_eff = Q[i + 1] + m * (a0 * q + a2 * qd + a3 * qdd) + c * (a1 * q + a4 * qd + a5 * qdd)
        q_next = p_eff / k_eff
        qdd_next = a0 * (q_next - q) - a2 * qd - a3 * qdd
        qd_next = qd + dt_s * ((1.0 - gamma) * qdd + gamma * qdd_next)

        q, qd, qdd = q_next, qd_next, qdd_next
        q_hist[i + 1] = q
        qd_hist[i + 1] = qd
        qdd_hist[i + 1] = qdd
        energy[i + 1] = 0.5 * float(np.sum(m * qd ** 2 + k * q ** 2))
        work[i + 1] = work[i] + 0.5 * dt_s * float(
            np.dot(Q[i], qd_hist[i]) + np.dot(Q[i + 1], qd)
        )
        dissipated[i + 1] = dissipated[i] + 0.5 * dt_s * float(
            np.dot(c * qd_hist[i] ** 2, np.ones(n_modes)) + np.dot(c * qd ** 2, np.ones(n_modes))
        )

    return ModalResponse(
        time_s=time_s,
        q=q_hist,
        qd=qd_hist,
        qdd=qdd_hist,
        modal_force=Q,
        energy_j=energy,
        work_j=work,
        dissipated_j=dissipated,
        beam=beam,
        n_modes=int(n_modes),
        dt_s=float(dt_s),
        damping_ratio=float(damping_ratio),
        load_total_n=total_load_series(paths, time_s),
    )