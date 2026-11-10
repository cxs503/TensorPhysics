"""Exact modal superposition for a piecewise-linear ice load (and energy audit).

Each modal coordinate obeys

    m_i qddot_i + (c_struct_i + c_rad_i) qdot_i + k_i q_i = Q_i(t)      (1)

with ``k_i = m_i omega_i^2``, ``c_struct_i = 2 zeta_i omega_i m_i`` and the
radiation damping ``c_rad_i`` supplied by the acoustic chain (see
:mod:`tensorphysics.ice_vib_noise.radiator`).  The force is sampled at the
acoustic time step and treated as *linear between samples*; for that forcing
the state transition is closed form (Nigam-Jennings / exact Duhamel):

    q(t+dt) = e^{-s dt}(C1 cos(wd dt) + C2 sin(wd dt)) + a + b dt
    a = (Q_n - c b)/k,   b = (Q_{n+1} - Q_n)/(k dt),   s = zeta w

so the *only* numerical error in the response is the piecewise-linear
interpolation of the load.  That matters for the acceptance criterion
"input work = dissipated + radiated energy": with an exact transition the
energy balance closes to the level of the quadrature used to evaluate the
integrals, which is a genuine test of the bookkeeping rather than of the
integrator's damping.

The energy identity used is, per mode,

    W_in = int Q qdot dt                       (work done by the generalised force)
    E_strain = 1/2 k q^2,  E_kin = 1/2 m qdot^2
    E_struct = int c_struct qdot^2 dt          (structural dissipation)
    E_rad    = int c_rad qdot^2 dt             (acoustic radiation)

    W_in - (E_strain + E_kin + E_struct + E_rad) = 0                 (2)

Cross-checks implemented here: a Newmark-beta (average acceleration) integrator
and a closed-form steady-state amplifier, both used in the test suite to prove
that (1) is solved correctly and not merely self-consistently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ModalResponse",
    "solve_modal_exact",
    "solve_modal_newmark",
    "steady_state_amplitude",
    "energy_balance",
]


@dataclass(frozen=True)
class ModalResponse:
    """Time history of the modal coordinates and the energy ledger."""

    time_s: np.ndarray
    q: np.ndarray            # (n_modes, n_time)
    qdot: np.ndarray
    qddot: np.ndarray
    force_generalized_n: np.ndarray   # (n_modes, n_time)
    modal_masses_kg: Tuple[float, ...]
    omegas_rad_s: Tuple[float, ...]
    damping_total: Tuple[float, ...]   # c_struct + c_rad  [N s/m]
    damping_structural: Tuple[float, ...]
    damping_radiation: Tuple[float, ...]
    stiffness_n_m: Tuple[float, ...]

    def energy_ledger(self) -> Dict[str, Any]:
        return energy_balance(
            time_s=self.time_s,
            q=self.q,
            qdot=self.qdot,
            force_generalized_n=self.force_generalized_n,
            modal_masses_kg=self.modal_masses_kg,
            stiffness_n_m=self.stiffness_n_m,
            damping_structural=self.damping_structural,
            damping_radiation=self.damping_radiation,
        )

    def surface_velocity_spectrum(
        self,
        x_query: Sequence[float],
        phi_at_query: np.ndarray,
        n_fft: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Surface velocity spectrum ``v(x, f)`` from the modal velocities.

        ``phi_at_query`` has shape ``(n_modes, n_query)``.  Returns
        ``(frequencies_hz, v_complex)`` with ``v_complex`` of shape
        ``(n_freq, n_query)`` and one-sided scaling such that
        ``sum |v|^2`` over the positive-frequency bins reproduces the mean
        square of the velocity history (Parseval, see ``bands``).
        """
        n = self.qdot.shape[1]
        n_fft = int(n_fft or n)
        if n_fft < n:
            raise ValueError("n_fft must be >= number of samples")
        dt = float(self.time_s[1] - self.time_s[0])
        phi = np.asarray(phi_at_query, dtype=float)
        out = np.empty((n_fft // 2 + 1, phi.shape[1]), dtype=complex)
        for j in range(phi.shape[1]):
            v = phi[:, j] @ self.qdot  # (n_time,)
            spec = np.fft.rfft(v, n=n_fft)
            amp = 2.0 * spec / n_fft          # peak-amplitude spectrum
            amp[0] = spec[0] / n_fft          # DC bin is not doubled
            if n_fft % 2 == 0:
                amp[-1] = spec[-1] / n_fft    # Nyquist bin is not doubled
            out[:, j] = amp
        freqs = np.fft.rfftfreq(n_fft, dt)
        return freqs, out


def _modal_transition(
    omega: float,
    zeta: float,
    dt: float,
) -> Tuple[complex, complex, float, float]:
    """Closed-form (A, B) of the piecewise-linear forced SDOF transition."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if not 0.0 <= zeta < 1.0:
        raise ValueError("only underdamped modes are supported (0 <= zeta < 1)")
    wd = omega * math.sqrt(1.0 - zeta * zeta)
    s = zeta * omega
    e = math.exp(-s * dt)
    c = math.cos(wd * dt)
    si = math.sin(wd * dt)
    return e, complex(c, si), wd, s


def solve_modal_exact(
    force_generalized_n: np.ndarray,
    dt_s: float,
    omegas_rad_s: Sequence[float],
    damping_ratios: Sequence[float],
    masses_kg: Sequence[float],
    c_rad: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact response to a piecewise-linear generalised force.

    Returns ``(q, qdot, qddot)`` each of shape ``(n_modes, n_time)``.  The load
    is taken as linear between samples; ``qddot`` is recovered from (1), so it
    is consistent with the integrated state rather than differentiated.
    """
    Q = np.asarray(force_generalized_n, dtype=float)
    if Q.ndim != 2:
        raise ValueError("force must be (n_modes, n_time)")
    n_modes, n_t = Q.shape
    if n_t < 2:
        raise ValueError("need at least two time samples")
    if len(omegas_rad_s) != n_modes or len(damping_ratios) != n_modes or len(masses_kg) != n_modes:
        raise ValueError("modal property lists must match the force rows")
    c_rad = np.zeros(n_modes) if c_rad is None else np.asarray(c_rad, dtype=float)
    if c_rad.size != n_modes:
        raise ValueError("c_rad must have one entry per mode")

    q = np.zeros((n_modes, n_t))
    qd = np.zeros((n_modes, n_t))
    qdd = np.zeros((n_modes, n_t))
    for i in range(n_modes):
        m = float(masses_kg[i])
        w = float(omegas_rad_s[i])
        zeta_struct = float(damping_ratios[i])
        c_struct = 2.0 * zeta_struct * w * m
        c_total = c_struct + float(c_rad[i])
        k = m * w * w
        zeta_eff = c_total / (2.0 * m * w)
        e_s, unit_s, wd_s, s_s = _modal_transition(w, zeta_eff, dt_s)
        cos_ = unit_s.real
        sin_ = unit_s.imag
        for n in range(n_t - 1):
            q_n = q[i, n]
            v_n = qd[i, n]
            Q0 = Q[i, n]
            Q1 = Q[i, n + 1]
            b = (Q1 - Q0) / (k * dt_s)
            a = (Q0 - c_total * b) / k
            C1 = q_n - a
            C2 = (v_n - b + s_s * C1) / wd_s
            q[i, n + 1] = e_s * (C1 * cos_ + C2 * sin_) + a + b * dt_s
            qd[i, n + 1] = e_s * (
                (-s_s * C1 + wd_s * C2) * cos_ + (-s_s * C2 - wd_s * C1) * sin_
            ) + b
        qdd[i] = (Q[i] - c_total * qd[i] - k * q[i]) / m
    return q, qd, qdd


def solve_modal_newmark(
    force_generalized_n: np.ndarray,
    dt_s: float,
    omegas_rad_s: Sequence[float],
    damping_ratios: Sequence[float],
    masses_kg: Sequence[float],
    c_rad: Optional[Sequence[float]] = None,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newmark-beta (default: average acceleration) reference integrator."""
    Q = np.asarray(force_generalized_n, dtype=float)
    n_modes, n_t = Q.shape
    c_rad = np.zeros(n_modes) if c_rad is None else np.asarray(c_rad, dtype=float)
    q = np.zeros((n_modes, n_t))
    qd = np.zeros((n_modes, n_t))
    qdd = np.zeros((n_modes, n_t))
    for i in range(n_modes):
        m = float(masses_kg[i])
        w = float(omegas_rad_s[i])
        c = 2.0 * float(damping_ratios[i]) * w * m + float(c_rad[i])
        k = m * w * w
        qdd[i, 0] = (Q[i, 0] - c * qd[i, 0] - k * q[i, 0]) / m
        keff = k + gamma / (beta * dt_s) * c + m / (beta * dt_s ** 2)
        for n in range(n_t - 1):
            rhs = (
                Q[i, n + 1]
                + c * (gamma / (beta * dt_s) * q[i, n] + (gamma / beta - 1.0) * qd[i, n]
                       + dt_s * (gamma / (2.0 * beta) - 1.0) * qdd[i, n])
                + m * (q[i, n] / (beta * dt_s ** 2) + qd[i, n] / (beta * dt_s)
                       + (1.0 / (2.0 * beta) - 1.0) * qdd[i, n])
            )
            q_new = rhs / keff
            qd_new = gamma / (beta * dt_s) * (q_new - q[i, n]) + (1.0 - gamma / beta) * qd[i, n] \
                + dt_s * (1.0 - gamma / (2.0 * beta)) * qdd[i, n]
            qdd_new = (q_new - q[i, n]) / (beta * dt_s ** 2) - qd[i, n] / (beta * dt_s) \
                - (1.0 / (2.0 * beta) - 1.0) * qdd[i, n]
            q[i, n + 1], qd[i, n + 1], qdd[i, n + 1] = q_new, qd_new, qdd_new
    return q, qd, qdd


def steady_state_amplitude(
    omega_excitation: float,
    omega_natural: float,
    damping_ratio: float,
    force_amplitude: float,
    mass: float,
) -> float:
    """Closed-form SDOF harmonic displacement amplitude (verification anchor)."""
    r = omega_excitation / omega_natural
    k = mass * omega_natural ** 2
    return force_amplitude / k / math.sqrt((1.0 - r * r) ** 2 + (2.0 * damping_ratio * r) ** 2)


def energy_balance(
    time_s: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    force_generalized_n: np.ndarray,
    modal_masses_kg: Sequence[float],
    stiffness_n_m: Sequence[float],
    damping_structural: Sequence[float],
    damping_radiation: Sequence[float],
) -> Dict[str, Any]:
    """Per-mode and total energy ledger, Eq. (2)."""
    t = np.asarray(time_s, dtype=float)
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qdot, dtype=float)
    Q = np.asarray(force_generalized_n, dtype=float)
    n_modes = q.shape[0]
    rows = []
    tot = {k: 0.0 for k in ("work_in", "strain", "kinetic", "diss_struct", "diss_rad", "residual", "scale")}
    for i in range(n_modes):
        m = float(modal_masses_kg[i])
        k = float(stiffness_n_m[i])
        work = float(np.trapezoid(Q[i] * qd[i], t))
        strain = 0.5 * k * float(q[i, -1] ** 2 - q[i, 0] ** 2)
        kinetic = 0.5 * m * float(qd[i, -1] ** 2 - qd[i, 0] ** 2)
        diss_s = float(np.trapezoid(float(damping_structural[i]) * qd[i] ** 2, t))
        diss_r = float(np.trapezoid(float(damping_radiation[i]) * qd[i] ** 2, t))
        residual = work - (strain + kinetic + diss_s + diss_r)
        scale = max(abs(work), abs(diss_s) + abs(diss_r) + abs(strain) + abs(kinetic), 1e-30)
        rows.append(
            {
                "mode": i + 1,
                "work_in_j": work,
                "strain_change_j": strain,
                "kinetic_change_j": kinetic,
                "dissipated_structural_j": diss_s,
                "radiated_acoustic_j": diss_r,
                "residual_j": residual,
                "residual_over_scale": abs(residual) / scale,
            }
        )
        for key, val in (
            ("work_in", work),
            ("strain", strain),
            ("kinetic", kinetic),
            ("diss_struct", diss_s),
            ("diss_rad", diss_r),
            ("residual", residual),
            ("scale", scale),
        ):
            tot[key] += val
    tot["residual_over_scale"] = abs(tot["residual"]) / max(tot["scale"], 1e-30)
    tot["per_mode"] = rows
    return tot