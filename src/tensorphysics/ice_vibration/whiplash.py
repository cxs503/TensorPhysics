"""Transient whiplash (springing/whipping) response of the hull girder.

A hard ice impact at the bow is a short, high-amplitude pulse.  Its duration is
comparable to a fraction of the first hull-girder period, so it excites several
vertical bending modes at once and the structure responds *transiently*: the
response peaks within one or two mode periods and then decays with the modal
damping.  This is the whipping (whiplash) response.

Model
-----
Modal superposition with the modes taken from :class:`ModalData`
(extraction source: TensorFEM, or the Euler-Bernoulli reference):

    q_n'' + 2 zeta_n w_n q_n' + w_n^2 q_n = Q_n(t) / m_n
    Q_n(t) = phi_n(x_bow) * F_ice(t)

Physical responses recovered from the generalized coordinates:

    w_bow(t)  = sum_n phi_n(x_bow) q_n(t)                  [m]
    M_mid(t)  = EI * sum_n phi_n''(L/2) q_n(t)             [N*m]

Reference solution
------------------
For a half-sine pulse ``F(t) = F0 sin(pi t / T_p)`` on ``0 <= t <= T_p`` the
damped SDOF response has a closed form (see
:func:`sdof_half_sine_response`), so the modal-superposition peak can be checked
against an independent analytic expression rather than against another
numerical scheme.  The time-marched peak and the analytic peak are compared in
:func:`whiplash_response`; their relative difference is the acceptance metric
"whiplash peak vs analytic modal superposition".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .integrators import newmark_mdof
from .modal import ModalData

__all__ = [
    "WhiplashResult",
    "sdof_half_sine_response",
    "sdof_half_sine_state",
    "whiplash_response",
    "quasi_static_midship_moment",
]


def _damped_particular_coefficients(
    omega_n: float,
    zeta: float,
    pulse_duration_s: float,
) -> Tuple[float, float, float, float, float, float]:
    """(A, B, denom, omega, omega_d, excitation_omega) for the half-sine pulse."""
    if omega_n <= 0.0 or pulse_duration_s <= 0.0:
        raise ValueError("omega_n and pulse_duration_s must be positive")
    if not 0.0 <= zeta < 1.0:
        raise ValueError("zeta must satisfy 0 <= zeta < 1")
    big_omega = math.pi / pulse_duration_s
    omega_d = omega_n * math.sqrt(max(1.0 - zeta**2, 0.0))
    a_coef = omega_n**2 - big_omega**2
    b_coef = 2.0 * zeta * omega_n * big_omega
    denom = a_coef**2 + b_coef**2
    return a_coef, b_coef, denom, omega_n, omega_d, big_omega


def sdof_half_sine_state(
    omega_n: float,
    zeta: float,
    force_over_mass: float,
    pulse_duration_s: float,
) -> Dict[str, float]:
    """State of a damped SDOF at the end of the half-sine pulse (exact).

    Also returns the coefficients used, so a caller can continue the free
    decay analytically without re-deriving them.
    """
    a_coef, b_coef, denom, omega, omega_d, big_omega = _damped_particular_coefficients(
        omega_n, zeta, pulse_duration_s
    )
    a = force_over_mass
    c_coef = a * b_coef / denom
    d_coef = (-a * a_coef * big_omega / denom + zeta * omega * c_coef) / omega_d
    tp = pulse_duration_s
    wdt = omega_d * tp
    e = math.exp(-zeta * omega * tp)
    q1 = a / denom * (a_coef * math.sin(big_omega * tp) - b_coef * math.cos(big_omega * tp)) + e * (
        c_coef * math.cos(wdt) + d_coef * math.sin(wdt)
    )
    v1 = (
        a / denom * (a_coef * big_omega * math.cos(big_omega * tp) + b_coef * big_omega * math.sin(big_omega * tp))
        + e
        * (
            (-zeta * omega * c_coef + omega_d * d_coef) * math.cos(wdt)
            + (-zeta * omega * d_coef - omega_d * c_coef) * math.sin(wdt)
        )
    )
    return {
        "displacement_m": q1,
        "velocity_m_s": v1,
        "A": a_coef,
        "B": b_coef,
        "denom": denom,
        "C": c_coef,
        "D": d_coef,
        "omega_n": omega,
        "omega_d": omega_d,
        "omega_excitation": big_omega,
        "denom_relative": denom / max(omega**4, 1e-300),
    }


def sdof_half_sine_response(
    time_s: np.ndarray,
    omega_n: float,
    zeta: float,
    force_over_mass: float,
    pulse_duration_s: float,
) -> np.ndarray:
    """Closed-form displacement response of a damped SDOF to a half-sine pulse.

    ``x'' + 2 zeta w x' + w^2 x = a sin(W t)`` for ``0 <= t <= Tp``, zero
    initial conditions, then free decay.  ``a = F0/m``.

    For ``|w^2 - W^2| / w^2 < 1e-8`` (excitation exactly at the natural
    frequency) the particular-solution coefficients become singular; that
    single degenerate case falls back to a 16-point Gauss quadrature of the
    Duhamel integral, which is still independent of the Newmark time march.
    """
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("time_s must be a non-empty 1-D array")
    if np.any(t < 0.0):
        raise ValueError("time_s must be non-negative")
    a_coef, b_coef, denom, omega, omega_d, big_omega = _damped_particular_coefficients(
        omega_n, zeta, pulse_duration_s
    )
    a = float(force_over_mass)
    if denom / max(omega**4, 1e-300) < 1e-8:
        return _duhamel_half_sine(t, omega, zeta, a, pulse_duration_s)
    c_coef = a * b_coef / denom
    d_coef = (-a * a_coef * big_omega / denom + zeta * omega * c_coef) / omega_d
    tp = pulse_duration_s
    x = np.empty_like(t)
    inside = t <= tp
    ti = t[inside]
    x[inside] = a / denom * (a_coef * np.sin(big_omega * ti) - b_coef * np.cos(big_omega * ti)) + np.exp(
        -zeta * omega * ti
    ) * (c_coef * np.cos(omega_d * ti) + d_coef * np.sin(omega_d * ti))
    outside = ~inside
    if np.any(outside):
        state = sdof_half_sine_state(omega_n, zeta, a, tp)
        tau = t[outside] - tp
        q1, v1 = state["displacement_m"], state["velocity_m_s"]
        x[outside] = np.exp(-zeta * omega * tau) * (
            q1 * np.cos(omega_d * tau) + (v1 + zeta * omega * q1) / omega_d * np.sin(omega_d * tau)
        )
    return x


def _duhamel_half_sine(t: np.ndarray, omega: float, zeta: float, a: float, tp: float) -> np.ndarray:
    """Duhamel-integral fallback (Gauss-Legendre) for the resonant degenerate case."""
    omega_d = omega * math.sqrt(max(1.0 - zeta**2, 0.0))
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    for index, ti in enumerate(t):
        upper = min(ti, tp)
        if upper <= 0.0:
            out[index] = 0.0
            continue
        nodes, weights = np.polynomial.legendre.leggauss(32)
        tau = 0.5 * upper * (nodes + 1.0)
        w = 0.5 * upper * weights
        f = a * np.sin(math.pi * tau / tp)
        h = np.exp(-zeta * omega * (ti - tau)) * np.sin(omega_d * (ti - tau)) / omega_d
        out[index] = float(np.sum(w * f * h))
    return out


@dataclass(frozen=True)
class WhiplashResult:
    """Transient whiplash response of the hull girder under a bow pulse."""

    time_s: np.ndarray
    modal_coordinates: np.ndarray          # (n_steps, n_modes) time-marched
    modal_coordinates_analytic: np.ndarray  # (n_steps, n_modes) closed form
    bow_displacement_m: np.ndarray
    bow_displacement_analytic_m: np.ndarray
    midship_moment_nm: np.ndarray
    midship_moment_analytic_nm: np.ndarray
    quarter_moment_nm: np.ndarray
    quarter_moment_analytic_nm: np.ndarray
    force_n: np.ndarray
    pulse_duration_s: float
    force_amplitude_n: float
    bending_stiffness_nm2: float
    modal: ModalData
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # -- peak metrics -------------------------------------------------------
    @property
    def peak_midship_moment_nm(self) -> float:
        return float(np.max(np.abs(self.midship_moment_nm)))

    @property
    def peak_midship_moment_analytic_nm(self) -> float:
        return float(np.max(np.abs(self.midship_moment_analytic_nm)))

    @property
    def peak_bow_displacement_m(self) -> float:
        return float(np.max(np.abs(self.bow_displacement_m)))

    @property
    def peak_bow_displacement_analytic_m(self) -> float:
        return float(np.max(np.abs(self.bow_displacement_analytic_m)))

    def moment_error_relative(self) -> float:
        num = abs(self.peak_midship_moment_nm - self.peak_midship_moment_analytic_nm)
        return float(num / self.peak_midship_moment_analytic_nm)

    def displacement_error_relative(self) -> float:
        num = abs(self.peak_bow_displacement_m - self.peak_bow_displacement_analytic_m)
        return float(num / self.peak_bow_displacement_analytic_m)

    @property
    def peak_quarter_moment_nm(self) -> float:
        return float(np.max(np.abs(self.quarter_moment_nm)))

    @property
    def peak_quarter_moment_analytic_nm(self) -> float:
        return float(np.max(np.abs(self.quarter_moment_analytic_nm)))

    def quarter_moment_error_relative(self) -> float:
        if self.peak_quarter_moment_analytic_nm <= 0.0:
            return float("nan")
        num = abs(self.peak_quarter_moment_nm - self.peak_quarter_moment_analytic_nm)
        return float(num / self.peak_quarter_moment_analytic_nm)

    def quarter_mode_shares(self) -> List[float]:
        """Share of the peak quarter-point moment carried by each mode."""
        curv = np.asarray(self.modal.mode_shape_quarter_curvature, dtype=float)
        if curv.size != self.modal.n_modes:
            return []
        peaks = np.abs(self.modal_coordinates) * np.abs(curv)[None, :]
        per_mode = [float(np.max(peaks[:, n])) for n in range(peaks.shape[1])]
        total = float(sum(per_mode))
        return [p / total if total > 0.0 else 0.0 for p in per_mode]

    @property
    def time_to_peak_s(self) -> float:
        idx = int(np.argmax(np.abs(self.midship_moment_nm)))
        return float(self.time_s[idx])

    @property
    def pulse_to_period_ratio(self) -> float:
        return self.pulse_duration_s * self.modal.frequencies_hz[0]

    def modal_peak_shares(self) -> Dict[str, float]:
        """Share of the peak midspan moment carried by each mode."""
        curv = np.asarray(self.modal.mode_shape_midship_curvature, dtype=float)
        peaks = np.abs(self.modal_coordinates) * curv[None, :]
        total = float(np.max(np.sum(peaks, axis=1)))
        per_mode = [float(np.max(peaks[:, n])) for n in range(peaks.shape[1])]
        return {
            "per_mode_curvature_peak_per_m": per_mode,
            "per_mode_share": [p / total if total > 0 else 0.0 for p in per_mode],
            "total_curvature_peak_per_m": total,
        }

    def to_json_dict(self) -> Dict[str, Any]:
        shares = self.modal_peak_shares()
        return {
            "class": "WhiplashResult",
            "force_amplitude_n": self.force_amplitude_n,
            "pulse_duration_s": self.pulse_duration_s,
            "pulse_duration_times_first_frequency": self.pulse_to_period_ratio,
            "bending_stiffness_nm2": self.bending_stiffness_nm2,
            "peak_midship_moment_nm": self.peak_midship_moment_nm,
            "peak_midship_moment_analytic_nm": self.peak_midship_moment_analytic_nm,
            "midship_moment_relative_error": self.moment_error_relative(),
            "peak_quarter_moment_nm": self.peak_quarter_moment_nm,
            "peak_quarter_moment_analytic_nm": self.peak_quarter_moment_analytic_nm,
            "quarter_moment_relative_error": self.quarter_moment_error_relative(),
            "quarter_moment_mode_shares": self.quarter_mode_shares(),
            "peak_bow_displacement_m": self.peak_bow_displacement_m,
            "peak_bow_displacement_analytic_m": self.peak_bow_displacement_analytic_m,
            "bow_displacement_relative_error": self.displacement_error_relative(),
            "time_to_peak_s": self.time_to_peak_s,
            "modal_shares": shares,
            "modal": self.modal.to_json_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def quasi_static_midship_moment(
    modal: ModalData,
    bending_stiffness_nm2: float,
    peak_bow_force_n: float,
) -> float:
    """Midspan moment if the peak bow force were applied statically.

    ``M_static = EI * sum_n phi_n''(L/2) * Q_n_peak / k_n`` with
    ``Q_n_peak = phi_n(bow) * F_peak`` and ``k_n = m_n w_n^2``.  Used as the
    denominator of the whiplash dynamic amplification of the bending moment.
    """
    phi_bow = np.asarray(modal.mode_shape_bow, dtype=float)
    phi_pp = np.asarray(modal.mode_shape_midship_curvature, dtype=float)
    k_n = modal.generalized_stiffness()
    q_static = phi_bow * peak_bow_force_n / k_n
    return float(bending_stiffness_nm2 * abs(float(np.dot(phi_pp, q_static))))


def whiplash_response(
    modal: ModalData,
    bending_stiffness_nm2: float,
    force_amplitude_n: float,
    pulse_duration_s: float,
    duration_s: Optional[float] = None,
    dt_s: Optional[float] = None,
) -> WhiplashResult:
    """Modal-superposition whiplash response to a bow half-sine ice impact.

    The generalized systems are integrated with Newmark-beta (average
    acceleration, unconditionally stable) and simultaneously evaluated from the
    closed-form half-sine solution; the two peaks differ only by discretization
    error, so their relative difference quantifies the numerical fidelity of the
    transient response (acceptance target < 8 %).
    """
    if modal.n_modes < 1:
        raise ValueError("modal data must contain at least one mode")
    if len(modal.mode_shape_midship_curvature) != modal.n_modes:
        raise ValueError(
            "modal data must carry midship curvature ordinates for every mode "
            f"({len(modal.mode_shape_midship_curvature)} given, {modal.n_modes} modes)"
        )
    if force_amplitude_n <= 0.0 or pulse_duration_s <= 0.0:
        raise ValueError("force amplitude and pulse duration must be positive")
    f1 = modal.frequencies_hz[0]
    if duration_s is None:
        duration_s = max(12.0 / f1, 8.0 * pulse_duration_s)
    if dt_s is None:
        dt_s = 1.0 / (200.0 * modal.frequencies_hz[-1])
    n_steps = int(round(duration_s / dt_s)) + 1
    time_s = np.arange(n_steps) * dt_s
    force = np.zeros(n_steps)
    mask = time_s <= pulse_duration_s
    force[mask] = force_amplitude_n * np.sin(math.pi * time_s[mask] / pulse_duration_s)

    phi_bow = np.asarray(modal.mode_shape_bow, dtype=float)
    phi_pp = np.asarray(modal.mode_shape_midship_curvature, dtype=float)
    mass = np.diag(np.asarray(modal.modal_masses_kg, dtype=float))
    damping = np.diag(modal.generalized_damping())
    stiffness = np.diag(modal.generalized_stiffness())
    force_matrix = np.outer(force, phi_bow)  # (n_steps, n_modes): Q_n(t) = phi_n(bow) F(t)

    marched = newmark_mdof(
        stiffness=stiffness,
        mass=mass,
        damping=damping,
        force=force_matrix,
        duration_s=duration_s,
        dt_s=dt_s,
    )
    q_marched = np.asarray(marched.displacement, dtype=float)
    if q_marched.ndim == 1:
        q_marched = q_marched[:, None]

    q_analytic = np.empty_like(q_marched)
    for n in range(modal.n_modes):
        q_analytic[:, n] = sdof_half_sine_response(
            time_s,
            omega_n=2.0 * math.pi * modal.frequencies_hz[n],
            zeta=modal.damping_ratios[n],
            force_over_mass=phi_bow[n] * force_amplitude_n / modal.modal_masses_kg[n],
            pulse_duration_s=pulse_duration_s,
        )

    phi_q = np.asarray(modal.mode_shape_quarter_curvature, dtype=float)
    if phi_q.size == modal.n_modes:
        quarter_marched = bending_stiffness_nm2 * (q_marched @ phi_q)
        quarter_analytic = bending_stiffness_nm2 * (q_analytic @ phi_q)
    else:
        quarter_marched = np.zeros_like(q_marched[:, 0])
        quarter_analytic = np.zeros_like(q_analytic[:, 0])
    bow_marched = q_marched @ phi_bow
    bow_analytic = q_analytic @ phi_bow
    moment_marched = bending_stiffness_nm2 * (q_marched @ phi_pp)
    moment_analytic = bending_stiffness_nm2 * (q_analytic @ phi_pp)

    static_moment = quasi_static_midship_moment(modal, bending_stiffness_nm2, force_amplitude_n)
    diagnostics = {
        "method_marched": marched.method,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "duration_s": duration_s,
        "n_modes": modal.n_modes,
        "quasi_static_midship_moment_nm": static_moment,
        "whiplash_dynamic_amplification": float(
            np.max(np.abs(moment_marched)) / static_moment if static_moment > 0.0 else float("nan")
        ),
        "modal_source": modal.source,
    }
    return WhiplashResult(
        time_s=time_s,
        modal_coordinates=q_marched,
        modal_coordinates_analytic=q_analytic,
        bow_displacement_m=bow_marched,
        bow_displacement_analytic_m=bow_analytic,
        midship_moment_nm=moment_marched,
        midship_moment_analytic_nm=moment_analytic,
        quarter_moment_nm=quarter_marched,
        quarter_moment_analytic_nm=quarter_analytic,
        force_n=force,
        pulse_duration_s=pulse_duration_s,
        force_amplitude_n=force_amplitude_n,
        bending_stiffness_nm2=bending_stiffness_nm2,
        modal=modal,
        diagnostics=diagnostics,
    )