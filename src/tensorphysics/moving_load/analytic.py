"""Closed-form undamped response of an SS beam to a moving point force.

The classical modal solution (Fryba, *Vibration of Solids and Structures under
Moving Loads*, 1972) for a point force ``P`` that enters a simply supported
beam at ``x = x_e`` at ``t = 0`` and travels with constant speed ``V`` is, per
mode ``n`` (``phi_n = sin(n pi x / L)``, ``M_n = rho A L / 2``):

    q_n(t) = P / [ M_n (w_n^2 - W_n^2) ] *
             [ sin(W_n t + psi_n) - sin(psi_n) cos(w_n t)
               - (W_n / w_n) cos(psi_n) sin(w_n t) ]

with ``W_n = n pi V / L`` and ``psi_n = n pi x_e / L``.  This module adds the
``W_n -> w_n`` limit (which is exactly what happens at ``V = V_cr``, where
``W_1 = w_1``), obtained by L'Hopital:

    q_n(t) = P / (2 M_n w_n) * [ cos(psi_n) sin(w_n t) / w_n
                                 - t cos(w_n t + psi_n) ]

and the free-vibration continuation after the load has left the span
(``t > t_c``), which is exact because the modal systems are undamped.  The
result is an *independent* reference for the time-stepping solvers, not a
re-derivation of them.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .beam import HullGirderBeam

__all__ = [
    "analytic_undamped_modal_state",
    "analytic_undamped_response",
    "RESONANCE_RELATIVE_TOLERANCE",
]

#: Relative tolerance on ``|w_n^2 - W_n^2| / w_n^2`` below which the closed-form
#: resonance limit is used instead of the regular expression (avoids the
#: catastrophic cancellation of ``sin(W t) - (W/w) sin(w t)`` near resonance).
RESONANCE_RELATIVE_TOLERANCE = 1.0e-8


def _modal_state_during(
    beam: HullGirderBeam,
    modes: np.ndarray,
    load_n: float,
    speed_m_s: float,
    x_entry_m: float,
    t_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal state ``(q, qd, qdd)`` while the load is on the span."""
    w = np.array([beam.omega(int(n)) for n in modes], dtype=float)
    W = np.array([beam.generalized_excitation_frequency(int(n), speed_m_s) for n in modes], dtype=float)
    M = np.array([beam.modal_mass(int(n)) for n in modes], dtype=float)
    psi = modes * math.pi * x_entry_m / beam.length_m

    d = w ** 2 - W ** 2
    resonant = np.abs(d) <= RESONANCE_RELATIVE_TOLERANCE * w ** 2

    q = np.empty_like(w)
    qd = np.empty_like(w)

    regular = ~resonant
    if np.any(regular):
        w_r, W_r, M_r, psi_r, d_r = w[regular], W[regular], M[regular], psi[regular], d[regular]
        arg = W_r * t_s + psi_r
        q[regular] = (load_n / (M_r * d_r)) * (
            np.sin(arg) - np.sin(psi_r) * np.cos(w_r * t_s) - (W_r / w_r) * np.cos(psi_r) * np.sin(w_r * t_s)
        )
        qd[regular] = (load_n / (M_r * d_r)) * (
            W_r * np.cos(arg) + w_r * np.sin(psi_r) * np.sin(w_r * t_s) - W_r * np.cos(psi_r) * np.cos(w_r * t_s)
        )

    if np.any(resonant):
        w_s, M_s, psi_s = w[resonant], M[resonant], psi[resonant]
        q[resonant] = (load_n / (2.0 * M_s * w_s)) * (
            np.cos(psi_s) * np.sin(w_s * t_s) / w_s - t_s * np.cos(w_s * t_s + psi_s)
        )
        qd[resonant] = (load_n / (2.0 * M_s)) * (
            np.cos(psi_s) * np.cos(w_s * t_s) - np.cos(w_s * t_s + psi_s) + w_s * t_s * np.sin(w_s * t_s + psi_s)
        )

    Q = load_n * np.sin(W * t_s + psi)
    qdd = Q / M - w ** 2 * q
    return q, qd, qdd


def analytic_undamped_modal_state(
    beam: HullGirderBeam,
    load_n: float,
    speed_m_s: float,
    x_entry_m: float,
    t_s: float,
    n_modes: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal state of the undamped SS beam under a moving point force at ``t_s``."""
    n_modes = int(n_modes)
    modes = np.arange(1, n_modes + 1)
    duration = (beam.length_m - float(x_entry_m)) / float(speed_m_s)
    if t_s <= duration:
        return _modal_state_during(beam, modes, load_n, speed_m_s, x_entry_m, float(t_s))

    q_c, qd_c, _ = _modal_state_during(beam, modes, load_n, speed_m_s, x_entry_m, duration)
    w = np.array([beam.omega(int(n)) for n in modes], dtype=float)
    tau = float(t_s) - duration
    q = q_c * np.cos(w * tau) + (qd_c / w) * np.sin(w * tau)
    qd = -q_c * w * np.sin(w * tau) + qd_c * np.cos(w * tau)
    qdd = -w ** 2 * q
    return q, qd, qdd


def analytic_undamped_response(
    beam: HullGirderBeam,
    load_n: float,
    speed_m_s: float,
    x_entry_m: float,
    time_s: Sequence[float],
    x_eval_m: float,
    n_modes: int = 60,
) -> dict:
    """Physical displacement/velocity/acceleration at one station.

    Returns a dict with ``time_s``, ``displacement_m``, ``velocity_m_s``,
    ``acceleration_m_s2`` and the mode-shape weights used.
    """
    t = np.asarray(time_s, dtype=float)
    n_modes = int(n_modes)
    modes = np.arange(1, n_modes + 1)
    phi = np.sin(modes * math.pi * float(x_eval_m) / beam.length_m)
    u = np.empty_like(t)
    v = np.empty_like(t)
    a = np.empty_like(t)
    for i, ti in enumerate(t):
        q, qd, qdd = analytic_undamped_modal_state(beam, load_n, speed_m_s, x_entry_m, float(ti), n_modes)
        u[i] = float(np.dot(q, phi))
        v[i] = float(np.dot(qd, phi))
        a[i] = float(np.dot(qdd, phi))
    return {
        "time_s": t,
        "displacement_m": u,
        "velocity_m_s": v,
        "acceleration_m_s2": a,
        "n_modes": n_modes,
        "x_eval_m": float(x_eval_m),
    }