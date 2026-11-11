"""Moving-load response of the hull-girder beam: modal superposition + closed-form checks.

Two independent solution paths are implemented and cross-verified:

``analytic_point_load_response``
    Closed-form modal series of a *constant* force ``P`` traversing a simply
    supported beam at speed ``V``, zero initial conditions, no damping::

        q_n(t) = (P/m_n) / (omega_n^2 - Omega_n^2) [sin(Omega_n t)
                                                   - (Omega_n/omega_n) sin(omega_n t)]

    with ``Omega_n = n pi V / L``.  At the critical speed ``Omega_n = omega_n``
    the resonant limit ``q_n = (P/m_n) [(sin w t)/w - t cos w t] / (2w)`` is used
    (L'Hopital).  This is the reference of acceptance criterion A2.

``modal_newmark``
    Newmark average-acceleration (``beta = 1/4``, ``gamma = 1/2``) integration of
    the decoupled modal equations for an *arbitrary* moving load
    ``F(t)`` at position ``x_p(t)``::

        m_n q''_n + c_n q'_n + k_n q_n = F(t) phi_n(x_p(t))

    This is the production path: it carries damping, an arbitrary force history
    (broken-ice pulse train) and a load that travels along the hull.

``steady_state_fundamental_amplitude``
    Independent *frequency-domain* route: the complex Fourier coefficient of the
    excitation multiplied by the modal frequency-response function.  It shares no
    code with the time integrator and is used to verify the steady-state
    amplitude of a periodic pulse train.

Because all matrices are diagonal in modal space, every quantity of the energy
balance required by acceptance criterion A4 is available at every time step:
the work done by the moving load, the strain energy, the kinetic energy and the
damping dissipation.

Units: SI (m, kg, s, N, N m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .beam import SIMPLY_SUPPORTED, GirderBeam, ModalBasis

__all__ = [
    "MovingLoadResponse",
    "analytic_point_load_response",
    "modal_newmark",
    "run_moving_point_load",
    "run_contact_history",
    "steady_state_fundamental_amplitude",
    "fourier_amplitude",
]

_RESONANCE_RELATIVE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovingLoadResponse:
    """Time history of the moving-load response plus the energy ledger."""

    time_s: np.ndarray
    monitor_x_m: float
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    modal_q: np.ndarray
    modal_qdot: np.ndarray
    work_input_j: np.ndarray
    strain_energy_j: np.ndarray
    kinetic_energy_j: np.ndarray
    damping_dissipated_j: np.ndarray
    speed_m_s: float
    zeta: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    analytic_displacement_m: np.ndarray | None = None

    # -- scalars -----------------------------------------------------------
    @property
    def peak_displacement_m(self) -> float:
        return float(np.max(np.abs(self.displacement_m)))

    @property
    def peak_acceleration_m_s2(self) -> float:
        return float(np.max(np.abs(self.acceleration_m_s2)))

    @property
    def energy_residual_j(self) -> np.ndarray:
        return self.work_input_j - self.damping_dissipated_j - self.strain_energy_j - self.kinetic_energy_j

    @property
    def energy_residual_relative(self) -> float:
        """Worst-case energy-balance residual, normalised by the peak mechanical energy.

        ``|W - D - (SE + KE)| / max(SE + KE)``.  Normalising by the peak
        mechanical energy (rather than by ``W``) keeps the measure meaningful
        when the load has left the beam and ``W`` stops growing.
        """
        mech = self.strain_energy_j + self.kinetic_energy_j
        scale = float(np.max(mech)) if mech.size else 0.0
        if scale <= 0.0:
            return float("inf")
        return float(np.max(np.abs(self.energy_residual_j)) / scale)

    def dynamic_amplification(self, static_displacement_m: float) -> float:
        """``max|u| / u_static`` for the same load magnitude at the monitor station."""
        if static_displacement_m <= 0.0:
            raise ValueError("static displacement must be positive")
        return self.peak_displacement_m / static_displacement_m

    def analytic_error(self, reference: np.ndarray | None = None) -> float | None:
        """Max relative error against the closed-form displacement history (if any)."""
        ref = self.analytic_displacement_m if reference is None else reference
        if ref is None:
            return None
        scale = float(np.max(np.abs(ref)))
        if scale <= 0.0:
            return None
        return float(np.max(np.abs(self.displacement_m - ref)) / scale)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = {
            "class": "MovingLoadResponse",
            "speed_m_s": self.speed_m_s,
            "zeta": self.zeta,
            "monitor_x_m": self.monitor_x_m,
            "n_samples": int(self.time_s.size),
            "dt_s": float(np.mean(np.diff(self.time_s))) if self.time_s.size > 1 else float("nan"),
            "peak_displacement_m": self.peak_displacement_m,
            "peak_acceleration_m_s2": self.peak_acceleration_m_s2,
            "work_input_j": float(self.work_input_j[-1]),
            "damping_dissipated_j": float(self.damping_dissipated_j[-1]),
            "strain_energy_peak_j": float(np.max(self.strain_energy_j)),
            "kinetic_energy_peak_j": float(np.max(self.kinetic_energy_j)),
            "energy_residual_relative": self.energy_residual_relative,
            "analytic_error": self.analytic_error(),
        }
        payload.update(dict(self.metadata))
        return payload


# ---------------------------------------------------------------------------
# Closed-form reference (constant force crossing a simply supported beam)
# ---------------------------------------------------------------------------


def analytic_point_load_response(
    beam: GirderBeam,
    force_n: float,
    speed_m_s: float,
    time_s: np.ndarray,
    n_modes: int = 200,
    monitor_x_m: float | None = None,
) -> Dict[str, np.ndarray]:
    """Closed-form undamped displacement history of a traversing constant force.

    Valid for a simply supported beam and for ``0 <= t <= L / V`` (the load is
    on the span).  ``monitor_x_m`` defaults to midspan.
    """
    if force_n <= 0.0 or speed_m_s <= 0.0:
        raise ValueError("force and speed must be positive")
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time must be a 1-D array with at least 2 samples")
    x_mon = 0.5 * beam.length_m if monitor_x_m is None else float(monitor_x_m)
    basis = beam.modal_basis(support=SIMPLY_SUPPORTED, n_modes=n_modes)
    omega = basis.omega_rad_s
    m_n = basis.modal_mass_kg
    phi_mon = basis.shapes([x_mon])[:, 0]
    n = basis.mode_numbers
    Omega = n * math.pi * speed_m_s / beam.length_m

    resonant = np.abs(omega - Omega) / omega < _RESONANCE_RELATIVE_TOL

    q = np.empty((n_modes, t.size))
    for i in range(n_modes):
        w, W = omega[i], Omega[i]
        if resonant[i]:
            q[i] = (force_n / m_n[i]) * (np.sin(w * t) / w - t * np.cos(w * t)) / (2.0 * w)
        else:
            q[i] = (
                (force_n / m_n[i])
                / (w**2 - W**2)
                * (np.sin(W * t) - (W / w) * np.sin(w * t))
            )
    u = phi_mon @ q
    return {
        "time_s": t,
        "displacement_m": u,
        "modal_q": q,
        "omega_n_rad_s": omega,
        "Omega_n_rad_s": Omega,
        "resonant_mode": resonant,
    }


# ---------------------------------------------------------------------------
# Newmark integration of the modal equations
# ---------------------------------------------------------------------------


def modal_newmark(
    basis: ModalBasis,
    time_s: np.ndarray,
    force_n: np.ndarray,
    position_m: np.ndarray,
    monitor_x_m: float | None = None,
    zeta: float | Sequence[float] = 0.0,
    analytic_displacement_m: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MovingLoadResponse:
    """Newmark average-acceleration solution of the modal moving-load problem.

    Parameters
    ----------
    basis:
        Modal basis (support condition, number of modes).
    time_s / force_n / position_m:
        Sampled load: total force magnitude [N] and along-hull application
        coordinate of every sample [m].  Zero force samples carry no load.
    monitor_x_m:
        Station at which the physical response is reported; defaults to midspan.
    zeta:
        Modal damping ratios (scalar or per-mode sequence).
    """
    t = np.asarray(time_s, dtype=float)
    f = np.asarray(force_n, dtype=float)
    xp = np.asarray(position_m, dtype=float)
    if not (t.size == f.size == xp.size) or t.size < 3:
        raise ValueError("time, force and position must be equally sized with >= 3 samples")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
        raise ValueError("time grid must be uniform")

    L = basis.beam.length_m
    x_mon = 0.5 * L if monitor_x_m is None else float(monitor_x_m)
    nm = basis.n_modes
    omega = basis.omega_rad_s
    m_n = basis.modal_mass_kg
    k_n = basis.modal_stiffness_nm
    c_n = basis.damping(zeta)
    zetas = c_n / (2.0 * omega * m_n)
    phi_mon = basis.shapes([x_mon])[:, 0]

    # generalized force: phi_n evaluated at the instantaneous load position
    x_clipped = np.clip(xp, 0.0, L)
    Q = basis.shapes(x_clipped) * f[None, :]

    beta, gamma = 0.25, 0.5
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k_n + a0 * m_n + a1 * c_n

    q = np.zeros(nm)
    qd = np.zeros(nm)
    qdd = (Q[:, 0] - c_n * qd - k_n * q) / m_n  # at rest at t = 0

    nt = t.size
    Qstore = np.empty((nm, nt))
    Qdstore = np.empty((nm, nt))
    Qddstore = np.empty((nm, nt))
    disp = np.empty(nt)
    vel = np.empty(nt)
    acc = np.empty(nt)
    work = np.zeros(nt)
    dissip = np.zeros(nt)
    se = np.empty(nt)
    ke = np.empty(nt)

    def store(i: int) -> None:
        Qstore[:, i] = q
        Qdstore[:, i] = qd
        Qddstore[:, i] = qdd
        disp[i] = float(phi_mon @ q)
        vel[i] = float(phi_mon @ qd)
        acc[i] = float(phi_mon @ qdd)
        se[i] = 0.5 * float(np.sum(k_n * q * q))
        ke[i] = 0.5 * float(np.sum(m_n * qd * qd))

    store(0)
    for i in range(1, nt):
        q_prev = q.copy()
        qd_prev = qd.copy()
        qdd_prev = qdd.copy()
        rhs = (
            Q[:, i]
            + m_n * (a0 * q_prev + a2 * qd_prev + a3 * qdd_prev)
            + c_n * (a1 * q_prev + a4 * qd_prev + a5 * qdd_prev)
        )
        q = rhs / k_eff
        qdd = a0 * (q - q_prev) - a2 * qd_prev - a3 * qdd_prev
        qd = qd_prev + dt * ((1.0 - gamma) * qdd_prev + gamma * qdd)
        store(i)
        # trapezoidal energy bookkeeping (input power Q q', damping power c q'^2)
        work[i] = work[i - 1] + 0.5 * dt * float(
            np.sum(Q[:, i] * qd) + np.sum(Q[:, i - 1] * qd_prev)
        )
        dissip[i] = dissip[i - 1] + 0.5 * dt * float(
            np.sum(c_n * qd * qd) + np.sum(c_n * qd_prev * qd_prev)
        )

    return MovingLoadResponse(
        time_s=t,
        monitor_x_m=x_mon,
        displacement_m=disp,
        velocity_m_s=vel,
        acceleration_m_s2=acc,
        modal_q=Qstore,
        modal_qdot=Qdstore,
        work_input_j=work,
        strain_energy_j=se,
        kinetic_energy_j=ke,
        damping_dissipated_j=dissip,
        speed_m_s=float((metadata or {}).get("speed_m_s", float("nan"))),
        zeta=float(np.mean(zetas)),
        metadata=dict(metadata or {}),
        analytic_displacement_m=analytic_displacement_m,
    )


# ---------------------------------------------------------------------------
# Convenience drivers
# ---------------------------------------------------------------------------


def run_moving_point_load(
    beam: GirderBeam,
    force_n: float,
    speed_m_s: float,
    n_modes: int = 10,
    zeta: float | Sequence[float] = 0.0,
    support: str = SIMPLY_SUPPORTED,
    monitor_x_m: float | None = None,
    samples_per_transit: int = 4000,
    n_extra_transits: int = 0,
) -> MovingLoadResponse:
    """A single constant force traversing the beam (with optional repeat transits).

    The force enters at ``x = 0`` at ``t = 0`` and leaves at ``x = L`` after
    ``L / V``; ``n_extra_transits`` re-injects it so that a periodic
    (steady-state) response can be reached - useful for the strongly amplified
    resonance of a *large* critical speed, where the transit is short compared
    with the build-up time.
    """
    if force_n <= 0.0 or speed_m_s <= 0.0:
        raise ValueError("force and speed must be positive")
    basis = beam.modal_basis(support=support, n_modes=n_modes)
    t_transit = beam.length_m / speed_m_s
    total = t_transit * (1 + n_extra_transits)
    n = max(int(samples_per_transit * (1 + n_extra_transits)), 1001)
    t = np.linspace(0.0, total, n)
    if support == SIMPLY_SUPPORTED and n_extra_transits == 0 and speed_m_s > 0:
        ana = analytic_point_load_response(
            beam, force_n, speed_m_s, t, n_modes=max(200, 4 * n_modes), monitor_x_m=monitor_x_m
        )["displacement_m"]
    else:
        ana = None
    cycle = np.mod(t, t_transit)
    x = speed_m_s * cycle
    on_span = x <= beam.length_m
    f = np.where(on_span, force_n, 0.0)
    resp = modal_newmark(
        basis,
        t,
        f,
        np.clip(x, 0.0, beam.length_m),
        monitor_x_m=monitor_x_m,
        zeta=zeta,
        analytic_displacement_m=ana,
        metadata={
            "case": "constant_point_load",
            "speed_m_s": speed_m_s,
            "force_n": force_n,
            "support": support,
            "transit_time_s": t_transit,
            "n_transits": 1 + n_extra_transits,
        },
    )
    return resp


def run_contact_history(
    beam: GirderBeam,
    history,
    n_modes: int = 10,
    zeta: float | Sequence[float] = 0.005,
    support: str = "free_free",
    monitor_x_m: float | None = None,
) -> MovingLoadResponse:
    """Integrate the response to a :class:`~.ice_field.ContactForceHistory`."""
    basis = beam.modal_basis(support=support, n_modes=n_modes)
    return modal_newmark(
        basis,
        history.time_s,
        history.force_n,
        history.position_m,
        monitor_x_m=monitor_x_m,
        zeta=zeta,
        metadata={
            "case": "broken_ice_contact_train",
            "speed_m_s": history.speed_m_s,
            "sweep": history.sweep,
            "support": support,
            "n_contact_events": int(history.patches_t0.size),
        },
    )


# ---------------------------------------------------------------------------
# Frequency-domain cross-check of a periodic pulse train
# ---------------------------------------------------------------------------


def fourier_amplitude(time_s: np.ndarray, signal: np.ndarray, frequency_hz: float) -> complex:
    """Complex single-sided Fourier coefficient of ``signal`` at ``frequency_hz``.

    Uses the exact trapezoidal projection ``(2/T) int y exp(-i 2 pi f t) dt``, so
    ``|c|`` *is* the single-sided amplitude of that harmonic (``|c| = A`` for
    ``y = A cos(2 pi f t)``).
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.size != y.size or t.size < 4:
        raise ValueError("time and signal must be equally sized with >= 4 samples")
    dt = float(np.mean(np.diff(t)))
    duration = float(t[-1] - t[0] + dt)
    kernel = np.exp(-2j * math.pi * frequency_hz * t)
    return 2.0 / duration * np.sum(y * kernel) * dt


def steady_state_fundamental_amplitude(
    basis: ModalBasis,
    time_s: np.ndarray,
    force_n: np.ndarray,
    load_x_m: float,
    monitor_x_m: float,
    frequency_hz: float,
    zeta: float | Sequence[float] = 0.005,
) -> Dict[str, float]:
    """Steady-state harmonic amplitude of a *stationary* periodic excitation.

    Independent of the time integrator: the force harmonic at
    ``frequency_hz`` is projected out with :func:`fourier_amplitude`, mapped to
    generalized coordinates through ``phi_n(load_x)`` and multiplied by the
    modal FRF ``1/(m_n (omega_n^2 - Omega^2 + 2 i zeta_n omega_n Omega))``.
    Returns the fundamental amplitudes of displacement, velocity and
    acceleration at the monitor station.
    """
    c = fourier_amplitude(time_s, force_n, frequency_hz)
    omega = basis.omega_rad_s
    m_n = basis.modal_mass_kg
    c_n = basis.damping(zeta)
    phi_load = basis.shapes([load_x_m])[:, 0]
    phi_mon = basis.shapes([monitor_x_m])[:, 0]
    Om = 2.0 * math.pi * frequency_hz
    denom = m_n * (omega**2 - Om**2 + 2j * (c_n / (2.0 * m_n * omega)) * omega * Om)
    q_amp = phi_load * c / denom
    u = complex(np.sum(phi_mon * q_amp))
    return {
        "frequency_hz": float(frequency_hz),
        "displacement_amplitude_m": float(abs(u)),
        "velocity_amplitude_m_s": float(abs(u) * Om),
        "acceleration_amplitude_m_s2": float(abs(u) * Om**2),
        "force_harmonic_amplitude_n": float(abs(c)),
    }